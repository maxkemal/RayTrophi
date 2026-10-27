#include "Backend/VulkanFsrUpscaler.h"
#include "Backend/VulkanFsrLoader.h"
#include "Backend/VulkanUpscaleInputs.h"
#include "Backend/VulkanBackend.h"
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace VulkanRT {
namespace {
// SDK 1.1.4 requests the extension spelling even on Vulkan 1.2+ devices.
// A promoted core command need not expose its KHR alias unless that extension
// was enabled on the logical device. Supply the ABI-identical core spelling.
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fsrDeviceProcAddr(VkDevice device,
    const char* name) {
    auto function = vkGetDeviceProcAddr(device, name);
    if (!function && std::strcmp(name, "vkGetBufferMemoryRequirements2KHR") == 0) {
        function = vkGetDeviceProcAddr(device, "vkGetBufferMemoryRequirements2");
    }
    return function;
}

bool fsrFunctionsAvailable(VkDevice device, std::string& reason) {
    // Required device commands loaded by the pinned SDK's ffx_vk.cpp.
    // Debug labels and AMD breadcrumb marker commands are optional.
    static constexpr const char* required[] = {
        "vkFlushMappedMemoryRanges",
        "vkCreateDescriptorPool",
        "vkCreateSampler",
        "vkCreateDescriptorSetLayout",
        "vkCreateBuffer",
        "vkCreateBufferView",
        "vkCreateImage",
        "vkCreateImageView",
        "vkCreateShaderModule",
        "vkCreatePipelineLayout",
        "vkCreateComputePipelines",
        "vkDestroyPipelineLayout",
        "vkDestroyPipeline",
        "vkDestroyImage",
        "vkDestroyImageView",
        "vkDestroyBuffer",
        "vkDestroyBufferView",
        "vkDestroyDescriptorSetLayout",
        "vkDestroyDescriptorPool",
        "vkDestroySampler",
        "vkDestroyShaderModule",
        "vkGetBufferMemoryRequirements",
        "vkGetBufferMemoryRequirements2KHR",
        "vkGetImageMemoryRequirements",
        "vkAllocateDescriptorSets",
        "vkFreeDescriptorSets",
        "vkAllocateMemory",
        "vkFreeMemory",
        "vkMapMemory",
        "vkUnmapMemory",
        "vkBindBufferMemory",
        "vkBindImageMemory",
        "vkUpdateDescriptorSets",
        "vkCmdPipelineBarrier",
        "vkCmdBindPipeline",
        "vkCmdBindDescriptorSets",
        "vkCmdDispatch",
        "vkCmdDispatchIndirect",
        "vkCmdCopyBuffer",
        "vkCmdCopyImage",
        "vkCmdCopyBufferToImage",
        "vkCmdClearColorImage",
        "vkCmdFillBuffer",
    };
    for (const char* name : required) {
        if (!fsrDeviceProcAddr(device, name)) {
            reason = std::string("FSR Vulkan function unavailable: ") + name;
            return false;
        }
    }
    return true;
}

FfxApiResource resource(const ImageHandle& image) {
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = image.format;
    info.extent = {image.width, image.height, 1};
    info.arrayLayers = 1;
    info.mipLevels = 1;
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    // AMD restores registered resources to their incoming state at unregister.
    // UNORDERED_ACCESS maps to GENERAL, matching our storage-image contract.
    return ffxApiGetResourceVK(reinterpret_cast<void*>(image.image),
        ffxApiGetImageResourceDescriptionVK(image.image, info, 0),
        FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
}
bool formatsSupported(VulkanDevice& device) {
    for (const auto format : {VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R8_UNORM, VK_FORMAT_R32_SFLOAT}) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(device.getPhysicalDevice(), format, &properties);
        const auto required = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & required) != required) {
            return false;
        }
    }
    return true;
}
} // namespace

struct VulkanFsrUpscaler::Impl {
    UpscaleSettings settings;
    UpscaleStats status;
    VulkanFsrLoader loader;
    VulkanUpscaleInputs inputs;
    ffxContext context = nullptr;
    // The API requires create-description pointers to outlive the context.
    ffxCreateBackendVKDesc backend{};
    ffxCreateContextDescUpscale create{};
    ImageHandle raw;
    bool eligible = false, resetPending = false, resetFsr = true, failed = false;
    bool reducedData = false, bridgeValid = false, retry = false;
    bool slotValid[2] = {true, true};
    uint64_t bridgeEpoch = 0, serial = 0, collectedSerial = 0;
    float lastCamera[4][4]{};
    VkQueryPool queries = VK_NULL_HANDLE;
    bool queryPending[2]{};
    uint64_t querySerial[2]{};
    uint32_t timestampBits = 0;
    float timestampPeriod = 0.0f;
    std::chrono::steady_clock::time_point previousFrame;

    void deactivate(const char* reason) {
        if (status.active) {
            resetPending = true;
        }
        status.active = false;
        status.inactive_reason = reason;
        status.render_w = status.display_w;
        status.render_h = status.display_h;
        status.effective_scale = 1.0f;
        reducedData = false;
    }
    void collect(VulkanDevice& device, uint32_t slot) {
        if (!queries || !queryPending[slot]) {
            return;
        }
        uint64_t values[2]{};
        if (vkGetQueryPoolResults(device.getDevice(), queries, slot * 2, 2,
                sizeof(values), values, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) {
            return;
        }
        if (querySerial[slot] >= collectedSerial) {
            const uint64_t mask = timestampBits == 64 ? ~uint64_t(0) :
                (uint64_t(1) << timestampBits) - 1;
            status.gpu_ms = double((values[1] - values[0]) & mask) * timestampPeriod * 1e-6;
            status.gpu_timing_available = true;
            collectedSerial = querySerial[slot];
        }
        queryPending[slot] = false;
    }
};

VulkanFsrUpscaler::VulkanFsrUpscaler() : m_impl(std::make_unique<Impl>()) {}
VulkanFsrUpscaler::~VulkanFsrUpscaler() = default;

void VulkanFsrUpscaler::configure(const UpscaleSettings& settings, bool moving) {
    auto& s = *m_impl;
    if (settings.mode != s.settings.mode || settings.quality != s.settings.quality ||
        settings.sharpness != s.settings.sharpness) {
        s.resetPending = s.settings.mode != UpscaleMode::Off ||
            settings.mode != UpscaleMode::Off;
        s.retry = s.retry || s.failed;
        s.failed = false;
        s.bridgeValid = false;
    }
    if (moving != s.status.moving && settings.mode != UpscaleMode::Off) {
        s.resetPending = true;
        if (moving) {
            s.bridgeValid = false;
        } else {
            s.bridgeEpoch = 0;
        }
    }
    s.settings = settings;
    s.status.bridge_active = false;
    s.status.moving = moving;
    if (settings.mode == UpscaleMode::Off) {
        s.bridgeValid = false;
        s.deactivate("mode is off");
    } else if (!moving) {
        s.deactivate("stationary: full-resolution accumulation");
    }
}

void VulkanFsrUpscaler::setEligible(bool eligible, const char* reason) {
    auto& s = *m_impl;
    s.eligible = eligible;
    if (!eligible) {
        s.bridgeValid = false;
        s.deactivate(reason);
    }
}

bool VulkanFsrUpscaler::prepare(VulkanDevice& device, const ImageHandle& raw,
    const ImageHandle& position, const ImageHandle& motion) {
    auto& s = *m_impl;
    s.status.display_w = raw.width;
    s.status.display_h = raw.height;
    if (s.settings.mode == UpscaleMode::Off || !s.status.moving || !s.eligible) {
        const char* reason = s.settings.mode == UpscaleMode::Off ? "mode is off" :
            (!s.status.moving ? "stationary: full-resolution accumulation" :
                s.status.inactive_reason.c_str());
        if (!s.context && !s.failed && s.settings.mode != UpscaleMode::Off) {
            s.status.reason = "FSR context is checked on the next eligible moving frame";
        }
        s.deactivate(reason);
        return false;
    }
    if (s.raw.view != raw.view || s.retry) {
        device.waitIdle();
        destroy(device);
        s.raw = raw;
        s.failed = false;
        s.retry = false;
    }
    if (s.failed) {
        // A failed dispatch may have recorded GPU work. Tear it down here,
        // outside command recording, after the queue has completed it.
        if (s.context) {
            device.waitIdle();
            destroy(device);
            s.raw = raw;
            s.failed = true;
        }
        s.deactivate(s.status.reason.c_str());
        return false;
    }
    auto fail = [&](const std::string& reason) {
        s.deactivate("FSR unavailable; full-resolution fallback");
        device.waitIdle();
        destroy(device);
        s.raw = raw;
        s.status.available = false;
        s.status.reason = reason;
        s.failed = true;
        ++s.status.failures;
        return false;
    };
    if (!s.context) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device.getPhysicalDevice(), &properties);
        if (!device.fsrImageFeaturesEnabled() || properties.apiVersion < VK_API_VERSION_1_2 ||
            !formatsSupported(device)) {
            return fail("required logical-device storage-image features/formats unavailable");
        }
        std::string reason;
        if (!fsrFunctionsAvailable(device.getDevice(), reason)) {
            return fail(reason);
        }
        if (!s.loader.open(reason)) {
            return fail(reason);
        }
        s.status.dll_loaded = true;
        s.backend.header = {FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK, nullptr};
        s.backend.vkDevice = device.getDevice();
        s.backend.vkPhysicalDevice = device.getPhysicalDevice();
        s.backend.vkDeviceProcAddr = fsrDeviceProcAddr;
        s.create.header = {FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, &s.backend.header};
        s.create.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE |
            FFX_UPSCALE_ENABLE_DEPTH_INVERTED | FFX_UPSCALE_ENABLE_DEPTH_INFINITE |
            FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION;
        s.create.maxRenderSize = {raw.width, raw.height};
        s.create.maxUpscaleSize = {raw.width, raw.height};
        s.create.fpMessage = nullptr;
        const auto result = s.loader.create(&s.context, &s.create.header, nullptr);
        if (result != FFX_API_RETURN_OK) {
            return fail("ffxCreateContext failed: " + std::to_string(result));
        }
        ffxQueryGetProviderVersion version{};
        version.header = {FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION, nullptr};
        if (s.loader.query(&s.context, &version.header) != FFX_API_RETURN_OK ||
            !version.versionName) {
            return fail("FSR provider-version query failed");
        }
        s.status.ffx_version = version.versionName;
        if (!s.inputs.ensure(device, raw, position, motion, reason)) {
            return fail(reason);
        }
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device.getPhysicalDevice(), &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device.getPhysicalDevice(), &count, families.data());
        s.timestampBits = families[device.getComputeQueueFamily()].timestampValidBits;
        s.timestampPeriod = properties.limits.timestampPeriod;
        if (s.timestampBits != 0) {
            VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qi.queryCount = 4;
            vkCreateQueryPool(device.getDevice(), &qi, nullptr, &s.queries);
        }
        s.resetFsr = true;
    }
    uint32_t width = 0, height = 0;
    ffxQueryDescUpscaleGetRenderResolutionFromQualityMode resolution{};
    resolution.header = {FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE, nullptr};
    resolution.displayWidth = raw.width;
    resolution.displayHeight = raw.height;
    resolution.qualityMode = static_cast<uint32_t>(s.settings.quality);
    resolution.pOutRenderWidth = &width;
    resolution.pOutRenderHeight = &height;
    if (s.loader.query(&s.context, &resolution.header) != FFX_API_RETURN_OK ||
        width == 0 || height == 0 || width > raw.width || height > raw.height) {
        return fail("FSR render-size query returned an invalid extent");
    }
    if (!s.status.active || s.status.render_w != width || s.status.render_h != height) {
        s.resetPending = true;
    }
    s.status.render_w = width;
    s.status.render_h = height;
    s.status.effective_scale = float(width) / float(raw.width);
    s.status.active = true;
    s.reducedData = true;
    s.status.available = true;
    s.status.reason.clear();
    s.status.inactive_reason.clear();
    return true;
}

bool VulkanFsrUpscaler::needsReset() const {
    return m_impl->resetPending;
}
void VulkanFsrUpscaler::acceptReset(uint64_t epoch) {
    m_impl->resetPending = false;
    m_impl->resetFsr = true;
    if (m_impl->bridgeValid && !m_impl->status.moving && m_impl->bridgeEpoch == 0) {
        m_impl->bridgeEpoch = epoch;
    }
    ++m_impl->status.history_resets;
}
void VulkanFsrUpscaler::invalidate() {
    m_impl->resetFsr = true;
    m_impl->bridgeValid = false;
}
bool VulkanFsrUpscaler::active() const {
    return m_impl->status.active;
}
bool VulkanFsrUpscaler::protectsDenoiser() const {
    return m_impl->reducedData;
}
uint32_t VulkanFsrUpscaler::renderWidth(uint32_t display) const {
    return active() ? m_impl->status.render_w : display;
}
uint32_t VulkanFsrUpscaler::renderHeight(uint32_t display) const {
    return active() ? m_impl->status.render_h : display;
}
bool VulkanFsrUpscaler::jitter(uint64_t frame, float out[2]) {
    auto& s = *m_impl;
    if (!active()) {
        return false;
    }
    int32_t count = 0;
    ffxQueryDescUpscaleGetJitterPhaseCount phase{};
    phase.header = {FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT, nullptr};
    phase.renderWidth = s.status.render_w;
    phase.displayWidth = s.status.display_w;
    phase.pOutPhaseCount = &count;
    if (s.loader.query(&s.context, &phase.header) != FFX_API_RETURN_OK || count <= 0) {
        s.failed = true;
        s.status.available = false;
        s.status.reason = "FSR jitter phase query failed";
        s.deactivate("invalid jitter; full-resolution fallback");
        return false;
    }
    ffxQueryDescUpscaleGetJitterOffset offset{};
    offset.header = {FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET, nullptr};
    offset.index = static_cast<int32_t>(frame % static_cast<uint64_t>(count));
    offset.phaseCount = count;
    offset.pOutX = &out[0];
    offset.pOutY = &out[1];
    if (s.loader.query(&s.context, &offset.header) != FFX_API_RETURN_OK ||
        !std::isfinite(out[0]) || !std::isfinite(out[1])) {
        s.failed = true;
        s.status.available = false;
        s.status.reason = "FSR jitter offset query failed";
        s.deactivate("invalid jitter; full-resolution fallback");
        return false;
    }
    s.status.jitter_phase_count = count;
    s.status.jitter_x = out[0];
    s.status.jitter_y = out[1];
    return true;
}

bool VulkanFsrUpscaler::record(VulkanDevice& device, VkCommandBuffer cmd, uint32_t slot,
    uint64_t epoch, int sample, const float camera[4][4], float fov,
    float preExposure, bool motionHistory) {
    auto& s = *m_impl;
    if (!active() || !s.context || slot >= 2) {
        return false;
    }
    s.status.epoch = epoch;
    s.status.sample = sample;
    s.collect(device, slot);
    if (s.queries) {
        vkCmdResetQueryPool(cmd, s.queries, slot * 2, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, s.queries, slot * 2);
    }
    s.inputs.recordInputs(cmd, s.status.render_w, s.status.render_h, camera,
        s.settings.rejection_reactive ? 1.0f : 0.0f);
    ffxDispatchDescUpscale dispatch{};
    dispatch.header = {FFX_API_DISPATCH_DESC_TYPE_UPSCALE, nullptr};
    dispatch.commandList = cmd;
    dispatch.color = resource(s.raw);
    dispatch.depth = resource(s.inputs.depth());
    dispatch.motionVectors = resource(s.inputs.velocity());
    dispatch.reactive = resource(s.inputs.reactive());
    dispatch.output = resource(s.inputs.output());
    dispatch.jitterOffset = {s.status.jitter_x, s.status.jitter_y};
    // B2 already stores unjittered PREVIOUS minus CURRENT in render pixels.
    // The SDK divides by renderSize internally: normalized-vector examples'
    // renderW/H scale would multiply these pixel vectors twice.
    dispatch.motionVectorScale = {1.0f, 1.0f};
    dispatch.renderSize = {s.status.render_w, s.status.render_h};
    dispatch.upscaleSize = {s.raw.width, s.raw.height};
    dispatch.enableSharpening = s.settings.sharpness > 0.0f;
    dispatch.sharpness = s.settings.sharpness;
    const auto now = std::chrono::steady_clock::now();
    dispatch.frameTimeDelta = s.previousFrame.time_since_epoch().count() == 0 ? 16.667f :
        std::clamp(std::chrono::duration<float, std::milli>(now - s.previousFrame).count(),
            1.0f, 1000.0f);
    s.previousFrame = now;
    dispatch.preExposure = std::max(preExposure, 1e-6f);
    dispatch.reset = s.resetFsr || !motionHistory;
    // The SDK validation names near/far in reversed device-depth order.
    dispatch.cameraNear = std::numeric_limits<float>::max();
    dispatch.cameraFar = 0.01f;
    dispatch.cameraFovAngleVertical = fov * 0.0174532925199433f;
    dispatch.viewSpaceToMetersFactor = 1.0f;
    const auto result = s.loader.dispatch(&s.context, &dispatch.header);
    if (result != FFX_API_RETURN_OK) {
        s.failed = true;
        s.resetPending = true;
        s.status.available = false;
        s.status.active = false;
        s.bridgeValid = false;
        s.status.reason = "ffxDispatch upscale failed: " + std::to_string(result);
        ++s.status.failures;
        return false;
    }
    s.inputs.compose(cmd, s.status.render_w, s.status.render_h, 1.0f, false, camera);
    s.inputs.savePosition(cmd, s.status.render_w, s.status.render_h);
    std::memcpy(s.lastCamera, camera, sizeof(s.lastCamera));
    s.bridgeValid = true;
    s.bridgeEpoch = 0;
    s.resetFsr = false;
    ++s.status.dispatches;
    if (s.queries) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, s.queries, slot * 2 + 1);
        s.queryPending[slot] = true;
        s.querySerial[slot] = ++s.serial;
    }
    return true;
}

bool VulkanFsrUpscaler::bridgeWanted(int sample, bool finalRender, bool debug) const {
    const auto& s = *m_impl;
    return s.bridgeValid && s.eligible && s.settings.mode != UpscaleMode::Off &&
        !s.status.moving && !finalRender && !debug && sample < 64;
}
void VulkanFsrUpscaler::recordBridge(VkCommandBuffer cmd, uint64_t epoch, int sample,
    const float camera[4][4]) {
    auto& s = *m_impl;
    s.status.bridge_active = false;
    if (!s.bridgeValid || (s.bridgeEpoch != 0 && s.bridgeEpoch != epoch) ||
        std::memcmp(s.lastCamera, camera, sizeof(s.lastCamera)) != 0) {
        s.bridgeValid = false;
        return;
    }
    s.bridgeEpoch = epoch;
    const float n = 24.0f;
    const float weight = std::max(0.0f, 1.0f - float(sample) / 64.0f) * n / (n + sample);
    s.inputs.compose(cmd, s.raw.width, s.raw.height, weight, true, camera);
    s.status.bridge_active = true;
}
void VulkanFsrUpscaler::setSlotValid(uint32_t slot, bool valid) {
    if (slot < 2) {
        m_impl->slotValid[slot] = valid;
        if (!valid) {
            ++m_impl->status.rejected_slots;
        }
    }
}
bool VulkanFsrUpscaler::slotValid(uint32_t slot) const {
    return slot < 2 && m_impl->slotValid[slot];
}
UpscaleStats VulkanFsrUpscaler::stats(VulkanDevice& device, bool synchronize) {
    auto& s = *m_impl;
    if (synchronize && s.queries) {
        device.waitIdle();
        s.collect(device, 0);
        s.collect(device, 1);
    }
    s.status.dll_loaded = s.loader.loaded();
    return s.status;
}
void VulkanFsrUpscaler::destroy(VulkanDevice& device) {
    auto& s = *m_impl;
    if (s.context) {
        s.loader.destroy(&s.context, nullptr);
        s.context = nullptr;
    }
    if (s.queries) {
        vkDestroyQueryPool(device.getDevice(), s.queries, nullptr);
        s.queries = VK_NULL_HANDLE;
    }
    s.inputs.destroy(device);
    s.queryPending[0] = s.queryPending[1] = false;
    s.status.available = false;
    s.status.gpu_timing_available = false;
    s.status.active = false;
    s.raw = {};
    s.bridgeValid = false;
    s.reducedData = false;
    s.resetFsr = true;
}
} // namespace VulkanRT
