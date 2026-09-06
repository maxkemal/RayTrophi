#include "Backend/VulkanBackend.h"
#include "globals.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <vector>

namespace Backend {
namespace {

constexpr uint32_t kIrradianceWidth = 64u;
constexpr uint32_t kIrradianceHeight = 32u;
constexpr uint32_t kPrefilterWidth = 256u;
constexpr uint32_t kPrefilterHeight = 128u;
constexpr uint32_t kPrefilterMipCount = 9u;
constexpr uint32_t kBrdfSize = 256u;
constexpr uint32_t kDescriptorSetCount = 1u + kPrefilterMipCount + 1u;

struct IblPush {
    uint32_t phase = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t sampleCount = 0;
    float roughness = 0.0f;
    float sourceMaxLod = 0.0f;
    float pad0 = 0.0f;
    float pad1 = 0.0f;
};
static_assert(sizeof(IblPush) == 32u, "Material preview IBL push ABI changed");

std::vector<uint32_t> loadIblSpv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || (bytes % 4) != 0) return {};
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4u);
    if (!file.read(reinterpret_cast<char*>(words.data()), bytes)) return {};
    return words;
}

VkSampler makeIblSampler(VkDevice device, float maxLod, bool repeatU) {
    VkSamplerCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter = VK_FILTER_LINEAR;
    ci.minFilter = VK_FILTER_LINEAR;
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.addressModeU = repeatU
        ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.minLod = 0.0f;
    ci.maxLod = maxLod;
    VkSampler sampler = VK_NULL_HANDLE;
    return vkCreateSampler(device, &ci, nullptr, &sampler) == VK_SUCCESS
        ? sampler : VK_NULL_HANDLE;
}

void recordIblImageBarrier(VkCommandBuffer cmd, VkImage image, uint32_t mipLevels,
                           VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1};
    if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    } else if (newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
            ? VK_ACCESS_TRANSFER_WRITE_BIT : 0u;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    } else {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
}

} // namespace

class MaterialPreviewIblResources {
public:
    VulkanRT::ImageHandle irradiance;
    VulkanRT::ImageHandle prefiltered;
    VulkanRT::ImageHandle brdf;
    std::array<VkImageView, kPrefilterMipCount> prefilterMipViews{};
    VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kDescriptorSetCount> descriptorSets{};
    VkDescriptorSet boundPreviewSet = VK_NULL_HANDLE;
    int64_t sourceTextureId = 0;
    bool brdfReady = false;
    bool generated = false;
    bool ready = false;
};

bool VulkanBackendAdapter::ensureMaterialPreviewIblResources(const std::string& shaderDir) {
    if (!m_device || !m_device->isInitialized()) return false;
    if (m_materialPreviewIbl && m_materialPreviewIbl->ready) return true;

    const auto words = loadIblSpv(shaderDir + "/material_preview_ibl.spv");
    if (words.empty()) {
        if (!m_materialPreviewIbl)
            m_materialPreviewIbl = std::make_shared<MaterialPreviewIblResources>();
        static bool warned = false;
        if (!warned) {
            SCENE_LOG_WARN("[MaterialPreview] IBL compute shader is missing; HDRI uses "
                           "the bounded raw-environment fallback until shaders are rebuilt.");
            warned = true;
        }
        return false;
    }

    destroyMaterialPreviewIblResources();
    auto state = std::make_shared<MaterialPreviewIblResources>();
    m_materialPreviewIbl = state;
    VkDevice device = m_device->getDevice();
    VkFormatProperties formatProperties{};
    vkGetPhysicalDeviceFormatProperties(
        m_device->getPhysicalDevice(), VK_FORMAT_R32G32B32A32_SFLOAT,
        &formatProperties);
    if ((formatProperties.optimalTilingFeatures &
         VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) == 0u) {
        SCENE_LOG_WARN("[MaterialPreview] RGBA32F storage images are unsupported; "
                       "HDRI IBL remains on the reported raw-environment fallback.");
        return false;
    }

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 2;
    dlci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &dlci, nullptr,
                                    &state->descriptorLayout) != VK_SUCCESS)
        return false;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = sizeof(IblPush);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &state->descriptorLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &state->pipelineLayout) != VK_SUCCESS) {
        destroyMaterialPreviewIblResources();
        return false;
    }

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = words.size() * sizeof(uint32_t);
    smci.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &smci, nullptr, &module) != VK_SUCCESS) {
        destroyMaterialPreviewIblResources();
        return false;
    }
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.layout = state->pipelineLayout;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    const VkResult pipelineResult = vkCreateComputePipelines(
        device, VK_NULL_HANDLE, 1, &cpci, nullptr, &state->pipeline);
    vkDestroyShaderModule(device, module, nullptr);
    if (pipelineResult != VK_SUCCESS) {
        destroyMaterialPreviewIblResources();
        return false;
    }

    VkDescriptorPoolSize poolSizes[2] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kDescriptorSetCount},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kDescriptorSetCount}
    };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = poolSizes;
    dpci.maxSets = kDescriptorSetCount;
    if (vkCreateDescriptorPool(device, &dpci, nullptr, &state->descriptorPool) != VK_SUCCESS) {
        destroyMaterialPreviewIblResources();
        return false;
    }
    std::array<VkDescriptorSetLayout, kDescriptorSetCount> layouts{};
    layouts.fill(state->descriptorLayout);
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = state->descriptorPool;
    dsai.descriptorSetCount = kDescriptorSetCount;
    dsai.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device, &dsai,
                                 state->descriptorSets.data()) != VK_SUCCESS) {
        destroyMaterialPreviewIblResources();
        return false;
    }

    constexpr VkImageUsageFlags outputUsage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    state->irradiance = m_device->createImage2D(
        kIrradianceWidth, kIrradianceHeight,
        VK_FORMAT_R32G32B32A32_SFLOAT, outputUsage);
    state->prefiltered = m_device->createImage2DWithMips(
        kPrefilterWidth, kPrefilterHeight, kPrefilterMipCount,
        VK_FORMAT_R32G32B32A32_SFLOAT, outputUsage);
    state->brdf = m_device->createImage2D(
        kBrdfSize, kBrdfSize, VK_FORMAT_R32G32B32A32_SFLOAT, outputUsage);
    if (!state->irradiance.image || !state->prefiltered.image || !state->brdf.image) {
        destroyMaterialPreviewIblResources();
        return false;
    }

    state->irradiance.sampler = makeIblSampler(device, 0.0f, true);
    state->prefiltered.sampler = makeIblSampler(
        device, static_cast<float>(kPrefilterMipCount - 1u), true);
    state->brdf.sampler = makeIblSampler(device, 0.0f, false);
    if (!state->irradiance.sampler || !state->prefiltered.sampler || !state->brdf.sampler) {
        destroyMaterialPreviewIblResources();
        return false;
    }

    for (uint32_t mip = 0; mip < kPrefilterMipCount; ++mip) {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = state->prefiltered.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = state->prefiltered.format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
        if (vkCreateImageView(device, &vci, nullptr,
                              &state->prefilterMipViews[mip]) != VK_SUCCESS) {
            destroyMaterialPreviewIblResources();
            return false;
        }
    }

    VkCommandBuffer transition = m_device->beginSingleTimeCommands();
    if (transition == VK_NULL_HANDLE) {
        destroyMaterialPreviewIblResources();
        return false;
    }
    recordIblImageBarrier(transition, state->prefiltered.image,
                          kPrefilterMipCount,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_GENERAL);
    m_device->endSingleTimeCommands(transition);

    state->ready = true;
    refreshMaterialPreviewIbl();
    return true;
}

void VulkanBackendAdapter::refreshMaterialPreviewIbl() {
    auto state = m_materialPreviewIbl;
    if (!state || !state->ready || !m_device || m_envTexID <= 0) return;
    auto sourceIt = m_uploadedImages.find(m_envTexID);
    if (sourceIt == m_uploadedImages.end() || !sourceIt->second.view ||
        !sourceIt->second.sampler)
        return;
    if (state->generated && state->sourceTextureId == m_envTexID) return;

    const VulkanRT::ImageHandle& source = sourceIt->second;
    VkDescriptorImageInfo sourceInfo{};
    sourceInfo.sampler = source.sampler;
    sourceInfo.imageView = source.view;
    sourceInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    for (uint32_t setIndex = 0; setIndex < kDescriptorSetCount; ++setIndex) {
        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView = setIndex == 0u ? state->irradiance.view :
            setIndex <= kPrefilterMipCount
                ? state->prefilterMipViews[setIndex - 1u] : state->brdf.view;
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = state->descriptorSets[setIndex];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &sourceInfo;
        writes[1] = writes[0];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outputInfo;
        vkUpdateDescriptorSets(m_device->getDevice(), 2, writes, 0, nullptr);
    }

    VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return;
    if (state->generated) {
        recordIblImageBarrier(cmd, state->irradiance.image, 1,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              VK_IMAGE_LAYOUT_GENERAL);
        recordIblImageBarrier(cmd, state->prefiltered.image, kPrefilterMipCount,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              VK_IMAGE_LAYOUT_GENERAL);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, state->pipeline);

    auto dispatch = [&](uint32_t setIndex, uint32_t phase, uint32_t width,
                        uint32_t height, uint32_t sampleCount, float roughness) {
        IblPush push{};
        push.phase = phase;
        push.width = width;
        push.height = height;
        push.sampleCount = sampleCount;
        push.roughness = roughness;
        push.sourceMaxLod = 0.0f;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                state->pipelineLayout, 0, 1,
                                &state->descriptorSets[setIndex], 0, nullptr);
        vkCmdPushConstants(cmd, state->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        vkCmdDispatch(cmd, (width + 7u) / 8u, (height + 7u) / 8u, 1);
    };

    dispatch(0u, 0u, kIrradianceWidth, kIrradianceHeight, 128u, 0.0f);
    for (uint32_t mip = 0; mip < kPrefilterMipCount; ++mip) {
        const uint32_t width = std::max(1u, kPrefilterWidth >> mip);
        const uint32_t height = std::max(1u, kPrefilterHeight >> mip);
        const float roughness = static_cast<float>(mip) /
            static_cast<float>(kPrefilterMipCount - 1u);
        dispatch(1u + mip, 1u, width, height, 128u, roughness);
    }
    if (!state->brdfReady)
        dispatch(kDescriptorSetCount - 1u, 2u, kBrdfSize, kBrdfSize, 256u, 0.0f);

    recordIblImageBarrier(cmd, state->irradiance.image, 1,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    recordIblImageBarrier(cmd, state->prefiltered.image, kPrefilterMipCount,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (!state->brdfReady) {
        recordIblImageBarrier(cmd, state->brdf.image, 1,
                              VK_IMAGE_LAYOUT_GENERAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    m_device->endSingleTimeCommands(cmd);
    state->sourceTextureId = m_envTexID;
    state->brdfReady = true;
    state->generated = true;
    state->boundPreviewSet = VK_NULL_HANDLE;
}

bool VulkanBackendAdapter::bindMaterialPreviewIblDescriptors(
    VkDescriptorSet set, const VulkanRT::ImageHandle& fallback) {
    auto state = m_materialPreviewIbl;
    if (!m_device || set == VK_NULL_HANDLE || !fallback.view || !fallback.sampler)
        return false;
    if (!state) {
        state = std::make_shared<MaterialPreviewIblResources>();
        m_materialPreviewIbl = state;
    }
    const bool current = state && state->ready && state->generated &&
        state->sourceTextureId == m_envTexID && m_envTexID > 0;
    if (state && state->boundPreviewSet != set) {
        drainInteractiveViewportInFlight();
        const VulkanRT::ImageHandle* images[3] = {
            current ? &state->irradiance : &fallback,
            current ? &state->prefiltered : &fallback,
            current ? &state->brdf : &fallback
        };
        VkDescriptorImageInfo infos[3]{};
        VkWriteDescriptorSet writes[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            infos[i].sampler = images[i]->sampler;
            infos[i].imageView = images[i]->view;
            infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = 13u + i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo = &infos[i];
        }
        vkUpdateDescriptorSets(m_device->getDevice(), 3, writes, 0, nullptr);
        state->boundPreviewSet = set;
    }
    return current;
}

bool VulkanBackendAdapter::getMaterialPreviewIblStatus(
    MaterialPreviewIblStatus& out) const {
    out = {};
    const auto state = m_materialPreviewIbl;
    if (!state) return false;
    out.supported = state->ready && state->pipeline != VK_NULL_HANDLE;
    out.ready = out.supported && state->generated &&
        state->sourceTextureId == m_envTexID && m_envTexID > 0;
    return true;
}

void VulkanBackendAdapter::destroyMaterialPreviewIblResources() {
    if (!m_materialPreviewIbl || !m_device) {
        m_materialPreviewIbl.reset();
        return;
    }
    auto& state = *m_materialPreviewIbl;
    VkDevice device = m_device->getDevice();
    for (VkImageView view : state.prefilterMipViews)
        if (view) vkDestroyImageView(device, view, nullptr);
    if (state.irradiance.image) m_device->destroyImage(state.irradiance);
    if (state.prefiltered.image) m_device->destroyImage(state.prefiltered);
    if (state.brdf.image) m_device->destroyImage(state.brdf);
    if (state.pipeline) vkDestroyPipeline(device, state.pipeline, nullptr);
    if (state.pipelineLayout) vkDestroyPipelineLayout(device, state.pipelineLayout, nullptr);
    if (state.descriptorPool) vkDestroyDescriptorPool(device, state.descriptorPool, nullptr);
    if (state.descriptorLayout) vkDestroyDescriptorSetLayout(device, state.descriptorLayout, nullptr);
    m_materialPreviewIbl.reset();
}

} // namespace Backend
