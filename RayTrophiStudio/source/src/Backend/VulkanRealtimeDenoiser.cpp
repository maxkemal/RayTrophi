#include "Backend/VulkanRealtimeDenoiser.h"
#include "Backend/VulkanBackend.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace VulkanRT {
namespace {

struct Push {
    uint32_t width, height, stage, stepWidth, historyFlags, historyMax;
    float fireflyClip, rawBlend;
    float camera[4];   // origin xyz, pixel size per unit distance
    uint32_t sampleCount, pad[3];
};
static_assert(sizeof(Push) == 64, "realtime_svgf.comp ABI");
// historyFlags bits, mirrored in realtime_svgf.comp.
constexpr uint32_t kSvgfHistory = 1u;    // previous SVGF dispatch = previous RT dispatch
constexpr uint32_t kSvgfRtHistory = 2u;  // RT record slots hold this and the previous dispatch
constexpr uint32_t kSvgfRtSlot1 = 4u;    // this dispatch's RT record is motionHistory1
constexpr uint32_t kSvgfStatic = 8u;     // still camera: spatial-only over the accumulation
constexpr uint32_t kSvgfDebugHistory = 16u;  // compose shows history length
constexpr uint32_t kBindings = 15;       // 0-11 images, 12 statistics, 13-14 RT records
constexpr uint32_t kSlots = 2;
constexpr uint32_t kQueries = 5;

void barrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkPipelineStageFlags dst,
             VkAccessFlags read, VkAccessFlags write) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = read;
    b.dstAccessMask = write;
    vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void computeBarrier(VkCommandBuffer cmd) {
    barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void copyImage(VkCommandBuffer cmd, const ImageHandle& src, const ImageHandle& dst) {
    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = region.srcSubresource;
    region.extent = {src.width, src.height, 1};
    vkCmdCopyImage(cmd, src.image, VK_IMAGE_LAYOUT_GENERAL,
                   dst.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

std::vector<uint32_t> loadShader(bool raw32) {
    const std::string name = raw32 ? "realtime_svgf32.spv" : "realtime_svgf.spv";
    for (const auto& dir : {"shaders/", "source/shaders/", "../shaders/"}) {
        std::ifstream f(std::string(dir) + name, std::ios::binary | std::ios::ate);
        if (!f) {
            continue;
        }
        const auto bytes = f.tellg();
        if (bytes <= 0 || static_cast<size_t>(bytes) % 4 != 0) {
            continue;
        }
        std::vector<uint32_t> code(static_cast<size_t>(bytes) / 4);
        f.seekg(0);
        if (f.read(reinterpret_cast<char*>(code.data()), bytes)) {
            return code;
        }
    }
    return {};
}
} // namespace

struct VulkanRealtimeDenoiser::Impl {
    RealtimeDenoiserSettings settings;
    RealtimeDenoiserStats status;
    ImageHandle raw, albedo, normal, position, motion, backup;
    ImageHandle records[2];   // raygen motionHistory0/1, not owned
    ImageHandle color[2], moments[2], work[2];
    BufferHandle counters[2];
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet sets[2][2]{};
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkQueryPool queries = VK_NULL_HANDLE;
    float timestampPeriod = 0.0f;
    uint32_t timestampBits = 0;
    bool queried[2]{};
    struct HistoryTag {
        uint64_t epoch = 0;
        int sample = 0;
        uint64_t dispatch = 0;
    };
    HistoryTag tags[2];
    bool historyValid = false;
    bool warm = false;
    uint64_t warmEpoch = 0;
    bool staticFrame = false;   // decided in wantsFrame, used by begin
    int staticLimit = 0;
    bool failed = false;
    bool composed = false;
    uint32_t parity = 0;
    uint32_t activeSlot = 0;

    void collect(VulkanDevice& device, uint32_t slot) {
        if (!queried[slot] || !queries) {
            return;
        }
        uint64_t values[kQueries]{};
        const auto result = vkGetQueryPoolResults(device.getDevice(), queries,
            slot * kQueries, kQueries, sizeof(values), values, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT);
        if (result != VK_SUCCESS) {
            return;
        }
        const uint64_t mask = timestampBits == 64 ? ~uint64_t(0) :
            ((uint64_t(1) << timestampBits) - 1);
        auto ms = [&](uint32_t a, uint32_t b) {
            return double((values[b] - values[a]) & mask) * timestampPeriod * 1e-6;
        };
        status.temporal_ms = ms(0, 1);
        status.variance_ms = ms(1, 2);
        status.atrous_ms = ms(2, 3);
        status.total_ms = ms(0, 4);
        status.gpu_timing_available = true;
        queried[slot] = false;
    }
};

VulkanRealtimeDenoiser::VulkanRealtimeDenoiser() : m_impl(std::make_unique<Impl>()) {}
VulkanRealtimeDenoiser::~VulkanRealtimeDenoiser() = default;

void VulkanRealtimeDenoiser::configure(const RealtimeDenoiserSettings& settings, bool moving) {
    auto& s = *m_impl;
    if (s.settings.enabled != settings.enabled ||
        s.settings.history_max != settings.history_max ||
        s.settings.atrous_iterations != settings.atrous_iterations ||
        s.settings.firefly_clip != settings.firefly_clip ||
        s.settings.static_samples != settings.static_samples) {
        invalidate();
        s.failed = false;
    }
    if (s.status.moving && !moving) {
        s.warm = true;
        s.warmEpoch = 0;
    }
    s.settings = settings;
    s.status.moving = moving;
    s.status.presenting = false;
    if (!settings.enabled) {
        s.warm = false;
        s.status.presenting = false;
    }
}

void VulkanRealtimeDenoiser::invalidate() {
    m_impl->historyValid = false;
    m_impl->warm = false;
    ++m_impl->status.history_resets;
}

bool VulkanRealtimeDenoiser::wantsFrame(int sample, int targetSamples, bool finalRender,
    bool debug, bool staticAllowed) const {
    auto& s = *m_impl;
    if (!s.status.moving && sample >= 8) {
        s.warm = false;
    }
    // Capped one BELOW the target: the async ping-pong shows the previous
    // submit's image, and once accumulation completes the last submit is never
    // consumed - the image left on screen is sample target-1. With the cap at
    // the target that frame was still slightly filtered (B3c live test: limit
    // 128 of 128 left meanabs 0.84 / max 27 on the converged view; limit 100
    // was bit-identical).
    s.staticLimit = targetSamples > 0 ? std::min(s.settings.static_samples, targetSamples - 1)
                                      : s.settings.static_samples;
    s.staticFrame = !s.status.moving && staticAllowed && s.staticLimit > 1 &&
                    sample < s.staticLimit;
    const bool wanted = s.settings.enabled && !finalRender && !debug &&
        (s.status.moving || s.staticFrame || (s.warm && sample < 8));
    if (!wanted) {
        s.status.static_active = false;
    }
    return wanted;
}

bool VulkanRealtimeDenoiser::ownsViewport() const {
    const auto& s = *m_impl;
    return s.settings.enabled && s.status.available && s.status.moving && s.status.presenting;
}

bool VulkanRealtimeDenoiser::ensure(VulkanDevice& device, const ImageHandle& raw,
    const ImageHandle& albedo, const ImageHandle& normal, const ImageHandle& position,
    const ImageHandle& motion, const ImageHandle& record0, const ImageHandle& record1) {
    auto& s = *m_impl;
    if (!raw.image || !albedo.image || !normal.image || !position.image || !motion.image ||
        !record0.image || !record1.image) {
        s.status.available = false;
        s.status.reason = "motion or first-hit AOVs unavailable";
        return false;
    }
    if (s.failed && s.raw.view == raw.view && s.motion.view == motion.view) {
        return false;
    }
    if (s.pipeline && s.raw.view == raw.view && s.albedo.view == albedo.view &&
        s.normal.view == normal.view && s.position.view == position.view &&
        s.motion.view == motion.view && s.records[0].view == record0.view &&
        s.records[1].view == record1.view) {
        return true;
    }
    // Only allocation / image replacement waits. Normal frames remain asynchronous.
    device.waitIdle();
    destroy(device);
    s.raw = raw;
    s.albedo = albedo;
    s.normal = normal;
    s.position = position;
    s.motion = motion;
    s.records[0] = record0;
    s.records[1] = record1;
    const auto vk = device.getDevice();
    auto fail = [&](const char* reason) {
        destroy(device);
        s.status.reason = reason;
        s.failed = true;
        return false;
    };
    const auto code = loadShader(raw.format == VK_FORMAT_R32G32B32A32_SFLOAT);
    if (code.empty()) {
        return fail("realtime_svgf.spv missing (compile both SVGF variants)");
    }
    const auto usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    s.backup = device.createImage2D(raw.width, raw.height, raw.format, usage);
    for (auto* array : {s.color, s.moments, s.work}) {
        for (uint32_t i = 0; i < 2; ++i) {
            array[i] = device.createImage2D(raw.width, raw.height,
                VK_FORMAT_R32G32B32A32_SFLOAT, usage);
            if (!array[i].image) {
                return fail("SVGF history allocation failed");
            }
        }
    }
    if (!s.backup.image) {
        return fail("SVGF backup allocation failed");
    }
    for (auto& buffer : s.counters) {
        BufferCreateInfo ci;
        ci.size = 2 * sizeof(uint32_t);
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::CPU_TO_GPU;
        buffer = device.createBuffer(ci);
        if (!buffer.buffer) {
            return fail("SVGF statistics allocation failed");
        }
    }
    VkDescriptorSetLayoutBinding bindings[kBindings]{};
    for (uint32_t i = 0; i < kBindings; ++i) {
        bindings[i] = {i, i == 12 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER :
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    di.bindingCount = kBindings;
    di.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(vk, &di, nullptr, &s.descriptors) != VK_SUCCESS) {
        return fail("SVGF descriptor layout failed");
    }
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 * (kBindings - 1)},
                                    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 4;
    pi.poolSizeCount = 2;
    pi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(vk, &pi, nullptr, &s.pool) != VK_SUCCESS) {
        return fail("SVGF descriptor pool failed");
    }
    for (uint32_t p = 0; p < 2; ++p) {
        for (uint32_t w = 0; w < 2; ++w) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = s.pool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &s.descriptors;
            if (vkAllocateDescriptorSets(vk, &ai, &s.sets[p][w]) != VK_SUCCESS) {
                return fail("SVGF descriptor allocation failed");
            }
            // Index = binding; 12 is the statistics buffer.
            const ImageHandle* images[] = {&s.raw, &s.albedo, &s.normal, &s.position,
                &s.motion, &s.color[1 - p], &s.color[p], &s.moments[1 - p],
                &s.moments[p], &s.work[w], &s.work[1 - w], &s.backup, nullptr,
                &s.records[0], &s.records[1]};
            VkDescriptorImageInfo info[kBindings]{};
            VkWriteDescriptorSet writes[kBindings]{};
            VkDescriptorBufferInfo buffer{s.counters[p].buffer, 0, 2 * sizeof(uint32_t)};
            for (uint32_t i = 0; i < kBindings; ++i) {
                writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                writes[i].dstSet = s.sets[p][w];
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                if (i == 12) {
                    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[i].pBufferInfo = &buffer;
                    continue;
                }
                info[i].imageView = images[i]->view;
                info[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[i].pImageInfo = &info[i];
            }
            vkUpdateDescriptorSets(vk, kBindings, writes, 0, nullptr);
        }
    }
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts = &s.descriptors;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(vk, &li, nullptr, &s.layout) != VK_SUCCESS) {
        return fail("SVGF pipeline layout failed");
    }
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = code.size() * 4;
    mi.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(vk, &mi, nullptr, &module) != VK_SUCCESS) {
        return fail("SVGF shader module failed");
    }
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.layout = s.layout;
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    auto result = vkCreateComputePipelines(vk, VK_NULL_HANDLE, 1, &ci, nullptr, &s.pipeline);
    vkDestroyShaderModule(vk, module, nullptr);
    if (result != VK_SUCCESS) {
        return fail("SVGF compute pipeline failed");
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device.getPhysicalDevice(), &properties);
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device.getPhysicalDevice(), &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device.getPhysicalDevice(), &count, families.data());
    s.timestampBits = families[device.getComputeQueueFamily()].timestampValidBits;
    s.timestampPeriod = properties.limits.timestampPeriod;
    if (s.timestampBits > 0) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kSlots * kQueries;
        vkCreateQueryPool(vk, &qi, nullptr, &s.queries);
    }
    std::vector<VulkanDevice::ImageClearRequest> clears;
    for (auto* array : {s.color, s.moments, s.work}) {
        for (uint32_t i = 0; i < 2; ++i) {
            clears.push_back({&array[i], 0, 0, 0, 0});
        }
    }
    clears.push_back({&s.backup, 0, 0, 0, 0});
    device.clearImages(clears);
    s.status.available = true;
    s.status.reason.clear();
    return true;
}

void VulkanRealtimeDenoiser::begin(VulkanDevice& device, VkCommandBuffer cmd,
    uint32_t slot, uint64_t epoch, int sample, uint64_t rtDispatch,
    bool rtHistory, uint32_t rtWriteSlot, const float camera[4]) {
    auto& s = *m_impl;
    s.composed = false;
    if (!s.status.available || slot >= kSlots) {
        return;
    }
    // Called after this frame slot's fence. Query reads cannot race slot reuse.
    s.collect(device, slot);
    if (!s.status.moving) {
        if (s.status.epoch != epoch && (!s.warm || s.warmEpoch != 0)) {
            invalidate();
        }
        if (s.warm && s.warmEpoch == 0) {
            s.warmEpoch = epoch;
        }
    }
    s.status.epoch = epoch;
    s.status.sample = sample;
    s.activeSlot = slot;
    auto stamp = [&](uint32_t q) {
        if (s.queries) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                s.queries, slot * kQueries + q);
        }
    };
    if (s.queries) {
        vkCmdResetQueryPool(cmd, s.queries, slot * kQueries, kQueries);
    }
    stamp(0);
    barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    copyImage(cmd, s.raw, s.backup);
    vkCmdFillBuffer(cmd, s.counters[s.parity].buffer, 0, 2 * sizeof(uint32_t), 0);
    barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeline);
    const bool consecutive = s.historyValid &&
        s.tags[1 - s.parity].dispatch + 1 == rtDispatch;
    const bool still = s.staticFrame && !s.status.moving;
    const uint32_t flags = (consecutive ? kSvgfHistory : 0u) |
        (rtHistory ? kSvgfRtHistory : 0u) | ((rtWriteSlot & 1u) ? kSvgfRtSlot1 : 0u) |
        (still ? kSvgfStatic : 0u) | (s.settings.debug_history ? kSvgfDebugHistory : 0u);
    // Moving: filtered only. Still frame: fade to raw by the static limit.
    // Warm bridge after a stop (OIDN on): fade over 8 samples.
    const float rawBlend = (s.status.moving || s.settings.debug_history) ? 0.0f :
        still ? std::min(float(sample) / float(s.staticLimit), 1.0f) :
                std::min(float(sample) / 8.0f, 1.0f);
    s.status.static_active = still;
    Push push{s.raw.width, s.raw.height, 0, 1, flags,
        static_cast<uint32_t>(s.settings.history_max), s.settings.firefly_clip,
        rawBlend, {camera[0], camera[1], camera[2], camera[3]},
        static_cast<uint32_t>(std::max(sample, 1)), {0u, 0u, 0u}};
    auto dispatch = [&](uint32_t stage, uint32_t input, uint32_t step) {
        push.stage = stage;
        push.stepWidth = step;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.layout,
            0, 1, &s.sets[s.parity][input], 0, nullptr);
        vkCmdPushConstants(cmd, s.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, (push.width + 7) / 8, (push.height + 7) / 8, 1);
        computeBarrier(cmd);
    };
    dispatch(0, 1, 1); // temporal -> work[0]
    stamp(1);
    dispatch(1, 0, 1); // variance -> work[1]
    stamp(2);
    uint32_t input = 1;
    for (int i = 0; i < s.settings.atrous_iterations; ++i) {
        dispatch(2, input, 1u << i);
        input = 1 - input;
    }
    stamp(3);
    dispatch(3, input, 1);
    s.composed = true;
    s.status.presenting = true;
    s.historyValid = true;
    ++s.status.dispatches;
    s.tags[s.parity] = {epoch, sample, rtDispatch};
    s.parity = 1 - s.parity;
}

void VulkanRealtimeDenoiser::end(VkCommandBuffer cmd) {
    auto& s = *m_impl;
    if (!s.composed) {
        return;
    }
    barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    copyImage(cmd, s.backup, s.raw);
    barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    if (s.queries) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            s.queries, s.activeSlot * kQueries + 4);
        s.queried[s.activeSlot] = true;
    }
    s.composed = false;
}

RealtimeDenoiserStats VulkanRealtimeDenoiser::stats(VulkanDevice& device, bool synchronize) {
    auto& s = *m_impl;
    if (synchronize && s.status.available) {
        device.waitIdle();
        for (uint32_t i = 0; i < kSlots; ++i) {
            s.collect(device, i);
        }
        uint32_t counts[2]{};
        if (s.status.dispatches > 0) {
            device.downloadBuffer(s.counters[1 - s.parity], counts, sizeof(counts));
            s.status.history_length_mean = counts[1] > 0 ? double(counts[0]) / counts[1] : 0.0;
        }
    }
    return s.status;
}

void VulkanRealtimeDenoiser::destroy(VulkanDevice& device) {
    auto& s = *m_impl;
    const auto vk = device.getDevice();
    if (s.pipeline) {
        vkDestroyPipeline(vk, s.pipeline, nullptr);
    }
    if (s.layout) {
        vkDestroyPipelineLayout(vk, s.layout, nullptr);
    }
    if (s.pool) {
        vkDestroyDescriptorPool(vk, s.pool, nullptr);
    }
    if (s.descriptors) {
        vkDestroyDescriptorSetLayout(vk, s.descriptors, nullptr);
    }
    if (s.queries) {
        vkDestroyQueryPool(vk, s.queries, nullptr);
    }
    s.pipeline = VK_NULL_HANDLE;
    s.layout = VK_NULL_HANDLE;
    s.pool = VK_NULL_HANDLE;
    s.descriptors = VK_NULL_HANDLE;
    s.queries = VK_NULL_HANDLE;
    s.queried[0] = s.queried[1] = false;
    for (auto* array : {s.color, s.moments, s.work}) {
        for (uint32_t i = 0; i < 2; ++i) {
            if (array[i].image) {
                device.destroyImage(array[i]);
            }
            array[i] = {};
        }
    }
    if (s.backup.image) {
        device.destroyImage(s.backup);
    }
    s.backup = {};
    s.records[0] = s.records[1] = {};
    for (auto& buffer : s.counters) {
        if (buffer.buffer) {
            device.destroyBuffer(buffer);
        }
        buffer = {};
    }
    s.status.available = false;
    s.status.presenting = false;
    s.failed = false;
    invalidate();
}
} // namespace VulkanRT
