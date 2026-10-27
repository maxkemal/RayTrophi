#include "Backend/VulkanUpscaleInputs.h"
#include "Backend/VulkanBackend.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

namespace VulkanRT {
namespace {
struct Push {
    uint32_t renderW, renderH, displayW, displayH;
    float origin[4], forward[4];
    uint32_t stage;
    float bridgeWeight;
    uint32_t oldWidth, oldHeight;
};
static_assert(sizeof(Push) == 64, "upscale_inputs.comp ABI");

void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags src,
    VkPipelineStageFlags dst, VkAccessFlags source, VkAccessFlags destination) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = source;
    barrier.dstAccessMask = destination;
    vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}
std::vector<uint32_t> shader(bool raw32) {
    for (const char* dir : {"shaders/", "source/shaders/", "../shaders/"}) {
        std::ifstream file(std::string(dir) +
            (raw32 ? "upscale_inputs32.spv" : "upscale_inputs.spv"),
            std::ios::binary | std::ios::ate);
        if (!file) {
            continue;
        }
        const std::streamsize size = file.tellg();
        if (size <= 0 || size % 4 != 0) {
            continue;
        }
        std::vector<uint32_t> code(static_cast<size_t>(size) / 4);
        file.seekg(0);
        if (file.read(reinterpret_cast<char*>(code.data()), size)) {
            return code;
        }
    }
    return {};
}
} // namespace

struct VulkanUpscaleInputs::Impl {
    ImageHandle raw, position, motion, depth, velocity, reactive, output, historyPosition;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t oldW = 0, oldH = 0;

    void record(VkCommandBuffer cmd, uint32_t width, uint32_t height, uint32_t stage,
        float weight, const float camera[4][4]) {
        Push push{width, height, raw.width, raw.height, {}, {}, stage, weight, oldW, oldH};
        float length2 = 0.0f, vertical2 = 0.0f;
        for (int i = 0; i < 3; ++i) {
            push.origin[i] = camera[0][i];
            push.forward[i] = camera[1][i] + 0.5f * camera[2][i] +
                0.5f * camera[3][i] - camera[0][i];
            length2 += push.forward[i] * push.forward[i];
            vertical2 += camera[3][i] * camera[3][i];
        }
        const float distance = std::sqrt(std::max(length2, 1e-12f));
        for (int i = 0; i < 3; ++i) {
            push.forward[i] /= distance;
        }
        push.forward[3] = 0.01f;
        push.origin[3] = std::sqrt(vertical2) /
            (distance * static_cast<float>(std::max(oldH, 1u)));
        memoryBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
            0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        const uint32_t w = stage == 0 ? width : raw.width;
        const uint32_t h = stage == 0 ? height : raw.height;
        vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }
};

VulkanUpscaleInputs::VulkanUpscaleInputs() : m_impl(std::make_unique<Impl>()) {}
VulkanUpscaleInputs::~VulkanUpscaleInputs() = default;

bool VulkanUpscaleInputs::ensure(VulkanDevice& device, const ImageHandle& raw,
    const ImageHandle& position, const ImageHandle& motion, std::string& reason) {
    auto& s = *m_impl;
    if (s.pipeline && s.raw.view == raw.view && s.position.view == position.view &&
        s.motion.view == motion.view) {
        return true;
    }
    device.waitIdle();
    destroy(device);
    s.raw = raw;
    s.position = position;
    s.motion = motion;
    const auto code = shader(raw.format == VK_FORMAT_R32G32B32A32_SFLOAT);
    auto fail = [&](const char* error) {
        reason = error;
        destroy(device);
        return false;
    };
    if (code.empty()) {
        return fail("upscale_inputs SPIR-V missing; compile both shader variants");
    }
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    s.depth = device.createImage2D(raw.width, raw.height, VK_FORMAT_R32_SFLOAT, usage);
    s.velocity = device.createImage2D(raw.width, raw.height, VK_FORMAT_R16G16_SFLOAT, usage);
    s.reactive = device.createImage2D(raw.width, raw.height, VK_FORMAT_R8_UNORM, usage);
    s.output = device.createImage2D(raw.width, raw.height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
    s.historyPosition = device.createImage2D(raw.width, raw.height,
        VK_FORMAT_R32G32B32A32_SFLOAT, usage);
    if (!s.depth.image || !s.velocity.image || !s.reactive.image ||
        !s.output.image || !s.historyPosition.image) {
        return fail("FSR input/display image allocation failed");
    }
    const auto vk = device.getDevice();
    VkDescriptorSetLayoutBinding bindings[8]{};
    for (uint32_t i = 0; i < 8; ++i) {
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr};
    }
    VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    di.bindingCount = 8;
    di.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(vk, &di, nullptr, &s.descriptors) != VK_SUCCESS) {
        return fail("FSR input descriptor layout failed");
    }
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 1;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &size;
    if (vkCreateDescriptorPool(vk, &pi, nullptr, &s.pool) != VK_SUCCESS) {
        return fail("FSR input descriptor pool failed");
    }
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = s.pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &s.descriptors;
    if (vkAllocateDescriptorSets(vk, &ai, &s.set) != VK_SUCCESS) {
        return fail("FSR input descriptor allocation failed");
    }
    const ImageHandle* images[] = {&s.raw, &s.position, &s.motion, &s.depth,
        &s.velocity, &s.reactive, &s.output, &s.historyPosition};
    VkDescriptorImageInfo info[8]{};
    VkWriteDescriptorSet writes[8]{};
    for (uint32_t i = 0; i < 8; ++i) {
        info[i].imageView = images[i]->view;
        info[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = s.set;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].descriptorCount = 1;
        writes[i].pImageInfo = &info[i];
    }
    vkUpdateDescriptorSets(vk, 8, writes, 0, nullptr);
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts = &s.descriptors;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(vk, &li, nullptr, &s.layout) != VK_SUCCESS) {
        return fail("FSR input pipeline layout failed");
    }
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = code.size() * 4;
    mi.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(vk, &mi, nullptr, &module) != VK_SUCCESS) {
        return fail("FSR input shader module failed");
    }
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.layout = s.layout;
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    const auto result = vkCreateComputePipelines(vk, VK_NULL_HANDLE, 1, &ci, nullptr, &s.pipeline);
    vkDestroyShaderModule(vk, module, nullptr);
    if (result != VK_SUCCESS) {
        return fail("FSR input compute pipeline failed");
    }
    device.clearImages({{&s.depth, 0, 0, 0, 0}, {&s.velocity, 0, 0, 0, 0},
        {&s.reactive, 0, 0, 0, 0}, {&s.output, 0, 0, 0, 0},
        {&s.historyPosition, 0, 0, 0, 0}});
    return true;
}

void VulkanUpscaleInputs::recordInputs(VkCommandBuffer cmd, uint32_t width, uint32_t height,
    const float camera[4][4], float rejectionReactive) {
    // Stage 0 does not use the bridge weight; the slot carries the strength.
    m_impl->record(cmd, width, height, 0, rejectionReactive, camera);
}
void VulkanUpscaleInputs::compose(VkCommandBuffer cmd, uint32_t width, uint32_t height,
    float bridgeWeight, bool bridge, const float camera[4][4]) {
    m_impl->record(cmd, width, height, bridge ? 2u : 1u, bridgeWeight, camera);
}
void VulkanUpscaleInputs::savePosition(VkCommandBuffer cmd, uint32_t width, uint32_t height) {
    auto& s = *m_impl;
    memoryBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy copy{};
    copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.dstSubresource = copy.srcSubresource;
    copy.extent = {width, height, 1};
    vkCmdCopyImage(cmd, s.position.image, VK_IMAGE_LAYOUT_GENERAL,
        s.historyPosition.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    memoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    s.oldW = width;
    s.oldH = height;
}
const ImageHandle& VulkanUpscaleInputs::depth() const {
    return m_impl->depth;
}
const ImageHandle& VulkanUpscaleInputs::velocity() const {
    return m_impl->velocity;
}
const ImageHandle& VulkanUpscaleInputs::reactive() const {
    return m_impl->reactive;
}
const ImageHandle& VulkanUpscaleInputs::output() const {
    return m_impl->output;
}

void VulkanUpscaleInputs::destroy(VulkanDevice& device) {
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
    s.pipeline = VK_NULL_HANDLE;
    s.layout = VK_NULL_HANDLE;
    s.pool = VK_NULL_HANDLE;
    s.descriptors = VK_NULL_HANDLE;
    for (auto* image : {&s.depth, &s.velocity, &s.reactive, &s.output, &s.historyPosition}) {
        if (image->image) {
            device.destroyImage(*image);
        }
        *image = {};
    }
    s.oldW = s.oldH = 0;
}
} // namespace VulkanRT
