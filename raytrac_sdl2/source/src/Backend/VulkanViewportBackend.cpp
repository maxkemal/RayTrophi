#include "Backend/VulkanViewportBackend.h"
#include "HittableInstance.h"
#include "HittableList.h"
#include "InstanceManager.h"
#include "ParallelBVHNode.h"
#include "Triangle.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <functional>
#include <unordered_map>
#include <unordered_set>

extern RenderSettings render_settings;

namespace Backend {
namespace {

inline bool viewportMatchesNodeNameForInstance(const std::string& instanceNodeName,
                                               const std::string& queryNodeName) {
    if (queryNodeName.empty() || instanceNodeName.empty()) return false;
    if (instanceNodeName == queryNodeName) return true;
    const std::string matPrefix = queryNodeName + "_mat_";
    return instanceNodeName.rfind(matPrefix, 0) == 0;
}

inline std::vector<uint32_t> loadViewportSPV(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};

    const std::streamsize size = file.tellg();
    if (size <= 0 || (size % static_cast<std::streamsize>(sizeof(uint32_t))) != 0) return {};

    std::vector<uint32_t> buffer(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return {};
    }
    return buffer;
}

} // namespace

void VulkanViewportBackend::renderProgressive(
    void* outSurface,
    void* outWindow,
    void* outRenderer,
    int width,
    int height,
    void* outFramebuffer,
    void* outTexture) {
    renderProgressiveImpl(outSurface, outWindow, outRenderer, width, height, outFramebuffer, outTexture);
}

void VulkanViewportBackend::setInteractiveViewportMatcap(int64_t textureID) {
    this->setInteractiveViewportMatcapImpl(textureID);
}

void VulkanViewportBackend::setInteractiveViewportMatcapPreset(int preset) {
    this->setInteractiveViewportMatcapPresetImpl(preset);
}

void VulkanViewportBackend::setInteractiveViewportMatcapImpl(int64_t textureID) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;

    if (textureID <= 0) {
        m_interactiveViewport.matcapUserLoaded = false;
        m_interactiveViewport.dirty = true;
        SCENE_LOG_INFO("[Matcap] Cleared — reverting to procedural matcap");
        return;
    }

    auto it = m_uploadedImages.find(textureID);
    if (it == m_uploadedImages.end()) {
        SCENE_LOG_INFO("[Matcap] textureID not found in uploaded images");
        return;
    }

    VkDevice vkDevice = m_device->getDevice();
    m_interactiveViewport.matcapImage = it->second;
    m_interactiveViewport.matcapUserLoaded = true;

    if (m_interactiveViewport.matcapDescSet == VK_NULL_HANDLE &&
        m_interactiveViewport.matcapDescPool != VK_NULL_HANDLE &&
        m_interactiveViewport.matcapDescLayout != VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = m_interactiveViewport.matcapDescPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &m_interactiveViewport.matcapDescLayout;
        vkAllocateDescriptorSets(vkDevice, &dsai, &m_interactiveViewport.matcapDescSet);
    }

    if (m_interactiveViewport.matcapDescSet != VK_NULL_HANDLE && m_interactiveViewport.matcapImage.image) {
        VkDescriptorImageInfo di{};
        di.sampler = m_interactiveViewport.matcapImage.sampler;
        di.imageView = m_interactiveViewport.matcapImage.view;
        di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet wds{};
        wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds.dstSet = m_interactiveViewport.matcapDescSet;
        wds.dstBinding = 0;
        wds.dstArrayElement = 0;
        wds.descriptorCount = 1;
        wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wds.pImageInfo = &di;

        vkUpdateDescriptorSets(vkDevice, 1, &wds, 0, nullptr);
        SCENE_LOG_INFO("[Matcap] Texture bound: " + std::to_string(it->second.width) + "x" + std::to_string(it->second.height));
    } else {
        SCENE_LOG_INFO("[Matcap] WARNING: Could not bind texture — descSet="
                       + std::to_string((uintptr_t)m_interactiveViewport.matcapDescSet)
                       + " pool=" + std::to_string((uintptr_t)m_interactiveViewport.matcapDescPool)
                       + " layout=" + std::to_string((uintptr_t)m_interactiveViewport.matcapDescLayout));
    }

    m_interactiveViewport.dirty = true;
}

void VulkanViewportBackend::setInteractiveViewportMatcapPresetImpl(int preset) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;

    if (preset < 0) preset = 0;
    if (preset > 9) preset = 9;
    m_interactiveViewport.matcapUserLoaded = false;
    m_interactiveViewport.matcapPreset = preset;
    m_interactiveViewport.dirty = true;
    SCENE_LOG_INFO("[Matcap] Preset selected: " + std::to_string(preset));
}

void VulkanViewportBackend::setExternalMaterialBuffer(VkBuffer buffer, VkDeviceSize size) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_externalMaterialBuffer = buffer;
    m_externalMaterialBufferSize = size;
    if (!m_device) return;

    VkDevice vkDevice = m_device->getDevice();
    if (vkDevice == VK_NULL_HANDLE) return;

    if (m_interactiveViewport.materialPreviewDescSet != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo matBufInfo{};
        matBufInfo.buffer = m_externalMaterialBuffer;
        matBufInfo.offset = 0;
        matBufInfo.range = (m_externalMaterialBufferSize > 0) ? m_externalMaterialBufferSize : VK_WHOLE_SIZE;

        VkWriteDescriptorSet mpWds{};
        mpWds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        mpWds.dstSet = m_interactiveViewport.materialPreviewDescSet;
        mpWds.dstBinding = 0;
        mpWds.dstArrayElement = 0;
        mpWds.descriptorCount = 1;
        mpWds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        mpWds.pBufferInfo = &matBufInfo;

        vkUpdateDescriptorSets(vkDevice, 1, &mpWds, 0, nullptr);
        SCENE_LOG_INFO("[VulkanViewportBackend] External material buffer bound to viewport descriptor set.");
    }
}

void VulkanViewportBackend::destroyInteractiveViewportResourcesImpl(bool keepPipeline) {
    if (!m_device) return;
    VkDevice vkDevice = m_device->getDevice();

    if (m_interactiveViewport.framebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(vkDevice, m_interactiveViewport.framebuffer, nullptr);
        m_interactiveViewport.framebuffer = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.renderPass != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyRenderPass(vkDevice, m_interactiveViewport.renderPass, nullptr);
        m_interactiveViewport.renderPass = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.solidPipeline != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipeline(vkDevice, m_interactiveViewport.solidPipeline, nullptr);
        m_interactiveViewport.solidPipeline = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.hairLinePipeline != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipeline(vkDevice, m_interactiveViewport.hairLinePipeline, nullptr);
        m_interactiveViewport.hairLinePipeline = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.hairLineVertexBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.hairLineVertexBuffer);
        m_interactiveViewport.hairLineVertexCount = 0;
    }
    if (m_interactiveViewport.pipelineLayout != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipelineLayout(vkDevice, m_interactiveViewport.pipelineLayout, nullptr);
        m_interactiveViewport.pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.colorImage.image) {
        m_device->destroyImage(m_interactiveViewport.colorImage);
    }
    if (m_interactiveViewport.depthImage.image) {
        m_device->destroyImage(m_interactiveViewport.depthImage);
    }
    if (m_interactiveViewport.stagingBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.stagingBuffer);
    }
    if (m_interactiveViewport.matcapDescSet != VK_NULL_HANDLE) {
        m_interactiveViewport.matcapDescSet = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.matcapDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vkDevice, m_interactiveViewport.matcapDescPool, nullptr);
        m_interactiveViewport.matcapDescPool = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.matcapDescLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vkDevice, m_interactiveViewport.matcapDescLayout, nullptr);
        m_interactiveViewport.matcapDescLayout = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.gridVertexBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.gridVertexBuffer);
        m_interactiveViewport.gridVertexCount = 0;
    }
    if (m_interactiveViewport.gridNormalBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.gridNormalBuffer);
    }
    if (m_interactiveViewport.identityInstanceBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.identityInstanceBuffer);
    }
    m_interactiveViewport.width = 0;
    m_interactiveViewport.height = 0;
    if (!keepPipeline) {
        m_interactiveViewport.initialized = false;
    }
}

bool VulkanViewportBackend::ensureInteractiveViewportResourcesImpl(const std::string& shaderDir, int width, int height) {
    if (!m_device || !m_device->isInitialized() || width <= 0 || height <= 0) return false;
    if (!m_device->supportsGraphicsQueue()) return false;

    if (!m_device->hasSkinningPipeline() && std::filesystem::exists(shaderDir + "/skinning.spv")) {
        std::vector<uint32_t> skinningSPV = loadViewportSPV(shaderDir + "/skinning.spv");
        if (!skinningSPV.empty()) {
            if (m_device->createSkinningPipeline(skinningSPV)) {
                for (auto& [meshKey, mesh] : m_rasterMeshes) {
                    mesh.skinningDescSet = VK_NULL_HANDLE;
                }
            }
        }
    }

    VkDevice vkDevice = m_device->getDevice();

    if (m_interactiveViewport.solidPipeline == VK_NULL_HANDLE) {
        const std::string vertPath = shaderDir + "/solid.spv";
        const std::string fragPath = shaderDir + "/solid_frag.spv";
        if (!std::filesystem::exists(vertPath) || !std::filesystem::exists(fragPath)) {
            return false;
        }

        std::vector<uint32_t> vertSPV = loadViewportSPV(vertPath);
        std::vector<uint32_t> fragSPV = loadViewportSPV(fragPath);
        if (vertSPV.empty() || fragSPV.empty()) {
            return false;
        }

        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = vertSPV.size() * sizeof(uint32_t);
        shaderInfo.pCode = vertSPV.data();
        VkShaderModule vertModule = VK_NULL_HANDLE;
        if (vkCreateShaderModule(vkDevice, &shaderInfo, nullptr, &vertModule) != VK_SUCCESS) {
            return false;
        }

        shaderInfo.codeSize = fragSPV.size() * sizeof(uint32_t);
        shaderInfo.pCode = fragSPV.data();
        VkShaderModule fragModule = VK_NULL_HANDLE;
        if (vkCreateShaderModule(vkDevice, &shaderInfo, nullptr, &fragModule) != VK_SUCCESS) {
            vkDestroyShaderModule(vkDevice, vertModule, nullptr);
            return false;
        }

        VkAttachmentDescription attachments[2]{};
        attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_GENERAL;

        attachments[1].format = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthRef{};
        depthRef.attachment = 1;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 2;
        renderPassInfo.pAttachments = attachments;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;
        if (vkCreateRenderPass(vkDevice, &renderPassInfo, nullptr, &m_interactiveViewport.renderPass) != VK_SUCCESS) {
            vkDestroyShaderModule(vkDevice, fragModule, nullptr);
            vkDestroyShaderModule(vkDevice, vertModule, nullptr);
            return false;
        }

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(float) * 32 + sizeof(int) + sizeof(float) * (3 + 3 + 3 + 2);

        VkDescriptorSetLayoutBinding matcapBinding{};
        matcapBinding.binding = 0;
        matcapBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        matcapBinding.descriptorCount = 1;
        matcapBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        matcapBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 1;
        dslci.pBindings = &matcapBinding;
        if (vkCreateDescriptorSetLayout(vkDevice, &dslci, nullptr, &m_interactiveViewport.matcapDescLayout) != VK_SUCCESS) {
            vkDestroyShaderModule(vkDevice, fragModule, nullptr);
            vkDestroyShaderModule(vkDevice, vertModule, nullptr);
            return false;
        }

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSize.descriptorCount = 1;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &poolSize;
        dpci.maxSets = 1;
        if (vkCreateDescriptorPool(vkDevice, &dpci, nullptr, &m_interactiveViewport.matcapDescPool) != VK_SUCCESS) {
            vkDestroyDescriptorSetLayout(vkDevice, m_interactiveViewport.matcapDescLayout, nullptr);
            m_interactiveViewport.matcapDescLayout = VK_NULL_HANDLE;
            vkDestroyShaderModule(vkDevice, fragModule, nullptr);
            vkDestroyShaderModule(vkDevice, vertModule, nullptr);
            return false;
        }

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_interactiveViewport.matcapDescLayout;
        if (vkCreatePipelineLayout(vkDevice, &pipelineLayoutInfo, nullptr, &m_interactiveViewport.pipelineLayout) != VK_SUCCESS) {
            vkDestroyRenderPass(vkDevice, m_interactiveViewport.renderPass, nullptr);
            m_interactiveViewport.renderPass = VK_NULL_HANDLE;
            vkDestroyShaderModule(vkDevice, fragModule, nullptr);
            vkDestroyShaderModule(vkDevice, vertModule, nullptr);
            return false;
        }

        VkPipelineShaderStageCreateInfo shaderStages[2]{};
        shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[0].module = vertModule;
        shaderStages[0].pName = "main";
        shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[1].module = fragModule;
        shaderStages[1].pName = "main";

        VkVertexInputBindingDescription bindings[3]{};
        bindings[0].binding = 0;
        bindings[0].stride = sizeof(float) * 3;
        bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindings[1].binding = 1;
        bindings[1].stride = sizeof(float) * 3;
        bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindings[2].binding = 2;
        bindings[2].stride = sizeof(float) * 16;
        bindings[2].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

        VkVertexInputAttributeDescription attributes[6]{};
        attributes[0].location = 0;
        attributes[0].binding = 0;
        attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[0].offset = 0;
        attributes[1].location = 1;
        attributes[1].binding = 1;
        attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[1].offset = 0;
        attributes[2].location = 2;
        attributes[2].binding = 2;
        attributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributes[2].offset = sizeof(float) * 0;
        attributes[3].location = 3;
        attributes[3].binding = 2;
        attributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributes[3].offset = sizeof(float) * 4;
        attributes[4].location = 4;
        attributes[4].binding = 2;
        attributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributes[4].offset = sizeof(float) * 8;
        attributes[5].location = 5;
        attributes[5].binding = 2;
        attributes[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributes[5].offset = sizeof(float) * 12;

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = 3;
        vertexInput.pVertexBindingDescriptions = bindings;
        vertexInput.vertexAttributeDescriptionCount = 6;
        vertexInput.pVertexAttributeDescriptions = attributes;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        // Edit workflows can open meshes (delete face/extrude), so Solid viewport
        // should remain readable even when the camera sees backfaces through holes.
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = m_interactiveViewport.pipelineLayout;
        pipelineInfo.renderPass = m_interactiveViewport.renderPass;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_interactiveViewport.solidPipeline) != VK_SUCCESS) {
            vkDestroyPipelineLayout(vkDevice, m_interactiveViewport.pipelineLayout, nullptr);
            m_interactiveViewport.pipelineLayout = VK_NULL_HANDLE;
            vkDestroyRenderPass(vkDevice, m_interactiveViewport.renderPass, nullptr);
            m_interactiveViewport.renderPass = VK_NULL_HANDLE;
            vkDestroyShaderModule(vkDevice, fragModule, nullptr);
            vkDestroyShaderModule(vkDevice, vertModule, nullptr);
            return false;
        }

        vkDestroyShaderModule(vkDevice, fragModule, nullptr);
        vkDestroyShaderModule(vkDevice, vertModule, nullptr);

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = m_interactiveViewport.matcapDescPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &m_interactiveViewport.matcapDescLayout;
        VkDescriptorSet descSet = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(vkDevice, &dsai, &descSet) == VK_SUCCESS) {
            m_interactiveViewport.matcapDescSet = descSet;

            if (!m_interactiveViewport.matcapImage.image) {
                std::vector<uint8_t> white(4 * 2 * 2, 255);
                int64_t id = this->uploadTexture2D(white.data(), 2, 2, 4, false, false);
                if (id && m_uploadedImages.find(id) != m_uploadedImages.end()) {
                    m_interactiveViewport.matcapImage = m_uploadedImages[id];
                }
            }

            if (m_interactiveViewport.matcapImage.image) {
                VkDescriptorImageInfo di{};
                di.sampler = m_interactiveViewport.matcapImage.sampler;
                di.imageView = m_interactiveViewport.matcapImage.view;
                di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet wds{};
                wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wds.dstSet = descSet;
                wds.dstBinding = 0;
                wds.dstArrayElement = 0;
                wds.descriptorCount = 1;
                wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                wds.pImageInfo = &di;

                vkUpdateDescriptorSets(vkDevice, 1, &wds, 0, nullptr);
            }
        }
        m_interactiveViewport.initialized = true;
    }

    // --- Hair Line Pipeline ---
    if (m_interactiveViewport.hairLinePipeline == VK_NULL_HANDLE) {
        const std::string hairVertPath = shaderDir + "/hair_viewport.spv";
        const std::string hairFragPath = shaderDir + "/hair_viewport_frag.spv";
        if (std::filesystem::exists(hairVertPath) && std::filesystem::exists(hairFragPath)) {
            std::vector<uint32_t> hvSPV = loadViewportSPV(hairVertPath);
            std::vector<uint32_t> hfSPV = loadViewportSPV(hairFragPath);
            if (!hvSPV.empty() && !hfSPV.empty()) {
                VkShaderModuleCreateInfo smci{};
                smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

                smci.codeSize = hvSPV.size() * sizeof(uint32_t);
                smci.pCode = hvSPV.data();
                VkShaderModule hVert = VK_NULL_HANDLE;
                vkCreateShaderModule(vkDevice, &smci, nullptr, &hVert);

                smci.codeSize = hfSPV.size() * sizeof(uint32_t);
                smci.pCode = hfSPV.data();
                VkShaderModule hFrag = VK_NULL_HANDLE;
                vkCreateShaderModule(vkDevice, &smci, nullptr, &hFrag);

                if (hVert && hFrag) {
                    VkPipelineShaderStageCreateInfo hStages[2]{};
                    hStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    hStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
                    hStages[0].module = hVert;
                    hStages[0].pName  = "main";
                    hStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    hStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
                    hStages[1].module = hFrag;
                    hStages[1].pName  = "main";

                    // Vertex: {vec3 position, float v_coord} = 16 bytes
                    VkVertexInputBindingDescription hBinding{};
                    hBinding.binding   = 0;
                    hBinding.stride    = sizeof(float) * 4;
                    hBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

                    VkVertexInputAttributeDescription hAttribs[2]{};
                    hAttribs[0].location = 0; hAttribs[0].binding = 0;
                    hAttribs[0].format   = VK_FORMAT_R32G32B32_SFLOAT; hAttribs[0].offset = 0;
                    hAttribs[1].location = 1; hAttribs[1].binding = 0;
                    hAttribs[1].format   = VK_FORMAT_R32_SFLOAT;       hAttribs[1].offset = sizeof(float) * 3;

                    VkPipelineVertexInputStateCreateInfo hVI{};
                    hVI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
                    hVI.vertexBindingDescriptionCount   = 1;
                    hVI.pVertexBindingDescriptions      = &hBinding;
                    hVI.vertexAttributeDescriptionCount = 2;
                    hVI.pVertexAttributeDescriptions    = hAttribs;

                    VkPipelineInputAssemblyStateCreateInfo hIA{};
                    hIA.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
                    hIA.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

                    VkPipelineViewportStateCreateInfo hVP{};
                    hVP.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
                    hVP.viewportCount = 1;
                    hVP.scissorCount  = 1;

                    VkPipelineRasterizationStateCreateInfo hRas{};
                    hRas.sType     = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                    hRas.polygonMode = VK_POLYGON_MODE_FILL;
                    hRas.lineWidth   = 1.0f;
                    hRas.cullMode    = VK_CULL_MODE_NONE;
                    hRas.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

                    VkPipelineMultisampleStateCreateInfo hMS{};
                    hMS.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
                    hMS.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

                    VkPipelineDepthStencilStateCreateInfo hDS{};
                    hDS.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
                    hDS.depthTestEnable  = VK_TRUE;
                    hDS.depthWriteEnable = VK_FALSE; // lines don't write depth (avoid z-fighting)
                    hDS.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

                    VkPipelineColorBlendAttachmentState hCBA{};
                    hCBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

                    VkPipelineColorBlendStateCreateInfo hCB{};
                    hCB.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
                    hCB.attachmentCount = 1;
                    hCB.pAttachments    = &hCBA;

                    VkDynamicState hDyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
                    VkPipelineDynamicStateCreateInfo hDynState{};
                    hDynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
                    hDynState.dynamicStateCount = 2;
                    hDynState.pDynamicStates    = hDyn;

                    VkGraphicsPipelineCreateInfo hPI{};
                    hPI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                    hPI.stageCount          = 2;
                    hPI.pStages             = hStages;
                    hPI.pVertexInputState   = &hVI;
                    hPI.pInputAssemblyState = &hIA;
                    hPI.pViewportState      = &hVP;
                    hPI.pRasterizationState = &hRas;
                    hPI.pMultisampleState   = &hMS;
                    hPI.pDepthStencilState  = &hDS;
                    hPI.pColorBlendState    = &hCB;
                    hPI.pDynamicState       = &hDynState;
                    hPI.layout              = m_interactiveViewport.pipelineLayout;
                    hPI.renderPass          = m_interactiveViewport.renderPass;
                    hPI.subpass             = 0;

                    vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &hPI, nullptr,
                                              &m_interactiveViewport.hairLinePipeline);
                }
                vkDestroyShaderModule(vkDevice, hVert, nullptr);
                vkDestroyShaderModule(vkDevice, hFrag, nullptr);
            }
        }
    }

    if (m_interactiveViewport.width == width &&
        m_interactiveViewport.height == height &&
        m_interactiveViewport.framebuffer != VK_NULL_HANDLE &&
        m_interactiveViewport.colorImage.image != VK_NULL_HANDLE &&
        m_interactiveViewport.depthImage.image != VK_NULL_HANDLE &&
        m_interactiveViewport.stagingBuffer.buffer != VK_NULL_HANDLE) {
        return true;
    }

    destroyInteractiveViewportResourcesImpl(true);

    m_interactiveViewport.colorImage = m_device->createImage2D(
        (uint32_t)width, (uint32_t)height, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
    m_interactiveViewport.depthImage = m_device->createImage2D(
        (uint32_t)width, (uint32_t)height, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    VulkanRT::BufferCreateInfo stagingInfo{};
    stagingInfo.size = (uint64_t)width * (uint64_t)height * 4ull;
    stagingInfo.usage = VulkanRT::BufferUsage::TRANSFER_DST;
    stagingInfo.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
    m_interactiveViewport.stagingBuffer = m_device->createBuffer(stagingInfo);

    if (!m_interactiveViewport.colorImage.image ||
        !m_interactiveViewport.depthImage.image ||
        !m_interactiveViewport.stagingBuffer.buffer) {
        destroyInteractiveViewportResourcesImpl(true);
        return false;
    }

    VkImageView attachments[2] = {
        m_interactiveViewport.colorImage.view,
        m_interactiveViewport.depthImage.view
    };
    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = m_interactiveViewport.renderPass;
    framebufferInfo.attachmentCount = 2;
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = (uint32_t)width;
    framebufferInfo.height = (uint32_t)height;
    framebufferInfo.layers = 1;
    if (vkCreateFramebuffer(vkDevice, &framebufferInfo, nullptr, &m_interactiveViewport.framebuffer) != VK_SUCCESS) {
        destroyInteractiveViewportResourcesImpl(true);
        return false;
    }

    m_interactiveViewport.width = width;
    m_interactiveViewport.height = height;
    return true;
}

void VulkanViewportBackend::renderInteractiveViewportImpl(void* s, int width, int height, void* fb, void* tex) {
    const auto frameStart = std::chrono::steady_clock::now();
    m_interactiveViewport.initialized = true;
    const ViewportMode requestedMode = m_viewportMode;

    if (!m_loggedInteractiveViewportFallback) {
        const char* mode_name = "Interactive";
        switch (requestedMode) {
            case ViewportMode::Solid: mode_name = "Solid"; break;
            case ViewportMode::MaterialPreview: mode_name = "MaterialPreview"; break;
            case ViewportMode::Matcap: mode_name = "Matcap"; break;
            case ViewportMode::Rendered: mode_name = "Rendered"; break;
        }
        SCENE_LOG_INFO(std::string("[Vulkan] Interactive viewport mode selected: '")
                       + mode_name + "'.");
        m_loggedInteractiveViewportFallback = true;
    }

    auto resolveShaderDir = [&]() -> std::string {
        std::string shaderDir = "shaders";
        if (!std::filesystem::exists(shaderDir + "/solid.spv")) shaderDir = "source/shaders";
        if (!std::filesystem::exists(shaderDir + "/solid.spv")) shaderDir = "../shaders";
        if (!std::filesystem::exists(shaderDir + "/solid.spv")) {
            shaderDir = std::filesystem::current_path().string() + "/shaders";
        }
        return shaderDir;
    };

    const bool rasterModeRequested = (requestedMode == ViewportMode::Solid ||
                                      requestedMode == ViewportMode::Matcap ||
                                      requestedMode == ViewportMode::MaterialPreview);
    const std::string shaderDir = resolveShaderDir();
    if (!rasterModeRequested ||
        !m_device ||
        !m_device->supportsGraphicsQueue() ||
        !ensureInteractiveViewportResourcesImpl(shaderDir, width, height)) {
        m_viewportMode = ViewportMode::Rendered;
        renderProgressive(s, nullptr, nullptr, width, height, fb, tex);
        m_viewportMode = requestedMode;
        return;
    }

    auto hashCamera = [](const CameraParams& c) -> uint64_t {
        uint64_t h = 14695981039346656037ull;
        auto mix = [&](float v) { uint32_t bits; std::memcpy(&bits, &v, 4); h ^= bits; h *= 1099511628211ull; };
        mix(c.origin.x); mix(c.origin.y); mix(c.origin.z);
        mix(c.lookAt.x); mix(c.lookAt.y); mix(c.lookAt.z);
        mix(c.up.x); mix(c.up.y); mix(c.up.z);
        mix(c.fov);
        return h;
    };
    uint64_t camHash = hashCamera(m_camera);
    if (!m_interactiveViewport.dirty && camHash == m_lastCameraHash &&
        m_interactiveViewport.width == width && m_interactiveViewport.height == height) {
        std::vector<uint32_t>* framebuffer = static_cast<std::vector<uint32_t>*>(fb);
        if (framebuffer && !framebuffer->empty()) {
            if (s) {
                SDL_Surface* outSurf = static_cast<SDL_Surface*>(s);
                if (outSurf->pixels && outSurf->w == width && outSurf->h == height) {
                    std::memcpy(outSurf->pixels, framebuffer->data(), framebuffer->size() * sizeof(uint32_t));
                }
            }
            if (tex) {
                SDL_UpdateTexture(static_cast<SDL_Texture*>(tex), nullptr, framebuffer->data(), width * 4);
            }
        }
        return;
    }
    m_lastCameraHash = camHash;
    m_interactiveViewport.dirty = false;

    struct SolidPushConstants {
        float viewProj[16];
        float view[16];
        int useMatcap;
        float overrideR, overrideG, overrideB;
    };

    auto matrixToGL = [](const Matrix4x4& mat, float out[16]) {
        Matrix4x4 t = mat.transpose();
        int k = 0;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out[k++] = t.m[r][c];
            }
        }
    };

    auto makeViewMatrix = [](const Vec3& eye, const Vec3& center, const Vec3& up) {
        Vec3 f = (center - eye).normalize();
        Vec3 sAxis = Vec3::cross(f, up).normalize();
        if (sAxis.length() < 0.0001f) sAxis = Vec3(1.0f, 0.0f, 0.0f);
        Vec3 uAxis = Vec3::cross(sAxis, f);
        return Matrix4x4(
            sAxis.x, sAxis.y, sAxis.z, -Vec3::dot(sAxis, eye),
            uAxis.x, uAxis.y, uAxis.z, -Vec3::dot(uAxis, eye),
            -f.x,   -f.y,   -f.z,    Vec3::dot(f, eye),
            0.0f,   0.0f,   0.0f,    1.0f
        );
    };

    auto makePerspectiveMatrix = [&](float fovDeg, float aspect, float zNear, float zFar) {
        const float f = 1.0f / std::tan(fovDeg * 0.5f * 3.14159265358979f / 180.0f);
        Matrix4x4 proj = Matrix4x4::zero();
        proj.m[0][0] = f / aspect;
        proj.m[1][1] = -f;
        proj.m[2][2] = zFar / (zNear - zFar);
        proj.m[2][3] = (zFar * zNear) / (zNear - zFar);
        proj.m[3][2] = -1.0f;
        return proj;
    };

    Matrix4x4 view = makeViewMatrix(m_camera.origin, m_camera.lookAt, m_camera.up);
    const float aspect = (height > 0) ? ((float)width / (float)height) : 1.0f;
    Matrix4x4 proj = makePerspectiveMatrix(
        m_camera.fov > 1.0f ? m_camera.fov : 60.0f,
        aspect,
        0.01f,
        1000000.0f);
    Matrix4x4 viewProj = proj * view;

    const float fovDeg = m_camera.fov > 1.0f ? m_camera.fov : 60.0f;
    const float tanHalfFov = std::tan(fovDeg * 0.5f * 3.14159265358979f / 180.0f);
    m_rasterCullCameraPosition = m_camera.origin;
    m_rasterCullFocalLengthPixels = (height > 0 && tanHalfFov > 1e-4f)
        ? ((0.5f * static_cast<float>(height)) / tanHalfFov)
        : 0.0f;
    m_rasterMinChunkScreenRadiusPixels = 3.0f;
    {
        double rasterQualityScale = 1.0;
        bool allowAdaptiveRasterBudget = false;
        if (m_viewportMode == ViewportMode::Solid || m_viewportMode == ViewportMode::Matcap) {
            switch (::render_settings.raster_viewport_quality_preset) {
                case ::RasterViewportQualityPreset::Performance:
                    rasterQualityScale = 0.58;
                    break;
                case ::RasterViewportQualityPreset::Balanced:
                    rasterQualityScale = 0.78;
                    break;
                case ::RasterViewportQualityPreset::Quality:
                    rasterQualityScale = 1.0;
                    break;
                case ::RasterViewportQualityPreset::Auto:
                default:
                    allowAdaptiveRasterBudget = true;
                    break;
            }
        }
        const double pixelCount = std::max(1.0, static_cast<double>(width) * static_cast<double>(height));
        const double referencePixels = 1920.0 * 1080.0;
        const double resolutionScale = std::sqrt(referencePixels / pixelCount);
        const double feedbackScale = allowAdaptiveRasterBudget
            ? std::clamp(static_cast<double>(m_rasterScatterBudgetScale), 0.35, 1.0)
            : 1.0;
        const uint64_t adaptiveBudget = static_cast<uint64_t>(72.0 * 1000.0 * 1000.0 * resolutionScale * rasterQualityScale * feedbackScale);
        m_rasterScatterTriangleBudget = std::clamp<uint64_t>(adaptiveBudget,
                                                             24ull * 1000ull * 1000ull,
                                                             96ull * 1000ull * 1000ull);
        if (!allowAdaptiveRasterBudget) {
            m_rasterScatterBudgetScale = 1.0f;
        }
    }

    // ── Frustum Culling: extract planes & re-upload only visible instances ──
    extractFrustumPlanes(viewProj);
    if (!m_rasterInstances.empty()) {
        for (auto& [key, mesh] : m_rasterMeshes) {
            uploadVisibleRasterInstances(mesh);
        }
    }

    if (!m_interactiveViewport.identityInstanceBuffer.buffer) {
        struct RasterInstanceGPU {
            float model[16];
        } identityGpu{};
        matrixToGL(Matrix4x4::identity(), identityGpu.model);
        VulkanRT::BufferCreateInfo ici{};
        ici.size = sizeof(identityGpu);
        ici.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        ici.location = VulkanRT::MemoryLocation::GPU_ONLY;
        ici.initialData = nullptr;
        m_interactiveViewport.identityInstanceBuffer = m_device->createBuffer(ici);
        if (m_interactiveViewport.identityInstanceBuffer.buffer) {
            m_device->uploadBuffer(m_interactiveViewport.identityInstanceBuffer,
                                   &identityGpu,
                                   sizeof(identityGpu),
                                   0);
        }
    }

    VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        return;
    }

    VkClearValue clearValues[2]{};
    const bool hasWorldBg = (m_cachedWorld.color.x != 0.0f || m_cachedWorld.color.y != 0.0f || m_cachedWorld.color.z != 0.0f);
    clearValues[0].color = { {
        hasWorldBg ? m_cachedWorld.color.x : 0.13f,
        hasWorldBg ? m_cachedWorld.color.y : 0.14f,
        hasWorldBg ? m_cachedWorld.color.z : 0.16f,
        1.0f } };
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_interactiveViewport.renderPass;
    renderPassInfo.framebuffer = m_interactiveViewport.framebuffer;
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = { (uint32_t)width, (uint32_t)height };
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.solidPipeline);
    if (m_interactiveViewport.matcapDescSet != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.pipelineLayout, 0, 1, &m_interactiveViewport.matcapDescSet, 0, nullptr);
    }

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)width;
    viewport.height = (float)height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = { (uint32_t)width, (uint32_t)height };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    uint64_t visibleTriangleTotal = 0;
    uint64_t visibleFullTriangleTotal = 0;
    uint64_t visibleProxyTriangleTotal = 0;
    uint64_t maxDrawTriangles = 0;
    uint32_t maxDrawInstances = 0;
    std::string maxDrawMeshKey;

    if (!m_rasterInstances.empty()) {
        for (const auto& [meshKey, rmb] : m_rasterMeshes) {
            if (!rmb.vertexBuffer.buffer || !rmb.instanceBuffer.buffer || rmb.vertexCount == 0 || rmb.instanceCount == 0) continue;
            const uint64_t trianglesPerInstance = (rmb.indexBuffer.buffer && rmb.indexCount > 0)
                ? (static_cast<uint64_t>(rmb.indexCount) / 3ull)
                : (static_cast<uint64_t>(rmb.vertexCount) / 3ull);
            const uint64_t drawTriangles = trianglesPerInstance * static_cast<uint64_t>(rmb.instanceCount);
            visibleTriangleTotal += drawTriangles;
            if (rmb.isScatterProxy) {
                visibleProxyTriangleTotal += drawTriangles;
            } else {
                visibleFullTriangleTotal += drawTriangles;
            }
            if (drawTriangles > maxDrawTriangles) {
                maxDrawTriangles = drawTriangles;
                maxDrawInstances = rmb.instanceCount;
                maxDrawMeshKey = meshKey;
            }

            SolidPushConstants push{};
            matrixToGL(viewProj, push.viewProj);
            matrixToGL(view, push.view);
            const bool isMatcap = (m_viewportMode == ViewportMode::Matcap);
            push.useMatcap = isMatcap ? (m_interactiveViewport.matcapUserLoaded ? 1 : m_interactiveViewport.matcapPreset) : 0;
            vkCmdPushConstants(cmd, m_interactiveViewport.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SolidPushConstants), &push);

            VkBuffer vertexBuffers[3] = {
                rmb.vertexBuffer.buffer,
                rmb.normalBuffer.buffer ? rmb.normalBuffer.buffer : rmb.vertexBuffer.buffer,
                rmb.instanceBuffer.buffer
            };
            VkDeviceSize offsets[3] = { 0, 0, 0 };
            vkCmdBindVertexBuffers(cmd, 0, 3, vertexBuffers, offsets);

            if (rmb.indexBuffer.buffer && rmb.indexCount > 0) {
                vkCmdBindIndexBuffer(cmd, rmb.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, rmb.indexCount, rmb.instanceCount, 0, 0, 0);
            } else {
                vkCmdDraw(cmd, rmb.vertexCount, rmb.instanceCount, 0, 0);
            }
        }
    } else if (m_rasterGeometryDirty) {
        for (const auto& instance : m_vkInstances) {
            if (instance.mask == 0) continue;
            if (instance.blasIndex >= m_device->m_blasList.size()) continue;

            const auto& blas = m_device->m_blasList[instance.blasIndex];
            if (!blas.vertexBuffer.buffer || blas.vertexCount == 0) continue;

            SolidPushConstants push{};
            matrixToGL(viewProj, push.viewProj);
            matrixToGL(view, push.view);
            const bool isMatcap = (m_viewportMode == ViewportMode::Matcap);
            push.useMatcap = isMatcap ? (m_interactiveViewport.matcapUserLoaded ? 1 : m_interactiveViewport.matcapPreset) : 0;
            vkCmdPushConstants(cmd, m_interactiveViewport.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SolidPushConstants), &push);

            VkBuffer vertexBuffers[3] = {
                blas.vertexBuffer.buffer,
                blas.normalBuffer.buffer ? blas.normalBuffer.buffer : blas.vertexBuffer.buffer,
                m_interactiveViewport.identityInstanceBuffer.buffer
            };
            VkDeviceSize offsets[3] = { 0, 0, 0 };
            vkCmdBindVertexBuffers(cmd, 0, 3, vertexBuffers, offsets);

            if (blas.indexBuffer.buffer && blas.indexCount > 0) {
                vkCmdBindIndexBuffer(cmd, blas.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, blas.indexCount, 1, 0, 0, 0);
            } else {
                vkCmdDraw(cmd, blas.vertexCount, 1, 0, 0);
            }
        }
    }

    if (!m_interactiveViewport.gridVertexBuffer.buffer) {
        const float gridHalf = 10.0f;
        const float step = 1.0f;
        const float thin = 0.005f;
        const float axisThin = 0.020f;

        std::vector<float> positions, normals;
        auto addLineQuad = [&](Vec3 a, Vec3 b, Vec3 widthDir) {
            Vec3 w = widthDir;
            Vec3 p0 = a - w, p1 = a + w, p2 = b + w, p3 = b - w;
            Vec3 n(0.0f, 1.0f, 0.0f);
            // Emit triangles (front-facing) and their reversed winding (back-facing)
            for (const Vec3& p : { p0, p1, p2, p0, p2, p3, p2, p1, p0, p3, p2, p0 }) {
                positions.push_back(p.x); positions.push_back(p.y); positions.push_back(p.z);
                normals.push_back(n.x); normals.push_back(n.y); normals.push_back(n.z);
            }
        };

        uint32_t seg0Start = 0;
        for (float x = -gridHalf; x <= gridHalf + 0.001f; x += step) {
            if (std::abs(x) < 0.01f) continue;
            addLineQuad(Vec3(x, 0.0f, -gridHalf), Vec3(x, 0.0f, gridHalf), Vec3(thin, 0.0f, 0.0f));
        }
        for (float z = -gridHalf; z <= gridHalf + 0.001f; z += step) {
            if (std::abs(z) < 0.01f) continue;
            addLineQuad(Vec3(-gridHalf, 0.0f, z), Vec3(gridHalf, 0.0f, z), Vec3(0.0f, 0.0f, thin));
        }
        uint32_t seg0Count = (uint32_t)(positions.size() / 3);

        uint32_t seg1Start = (uint32_t)(positions.size() / 3);
        addLineQuad(Vec3(0.0f, 0.001f, 0.0f), Vec3(gridHalf, 0.001f, 0.0f), Vec3(0.0f, 0.0f, axisThin));
        uint32_t seg1Count = (uint32_t)(positions.size() / 3) - seg1Start;

        uint32_t seg2Start = (uint32_t)(positions.size() / 3);
        // Use same small Y offset as +X axis and slightly larger width so it's visible
        addLineQuad(Vec3(0.0f, 0.001f, 0.0f), Vec3(0.0f, 0.001f, gridHalf), Vec3(axisThin, 0.0f, 0.0f));
        uint32_t seg2Count = (uint32_t)(positions.size() / 3) - seg2Start;

        uint32_t seg3Start = (uint32_t)(positions.size() / 3);
        addLineQuad(Vec3(-gridHalf, 0.001f, 0.0f), Vec3(0.0f, 0.001f, 0.0f), Vec3(0.0f, 0.0f, thin));
        addLineQuad(Vec3(0.0f, 0.001f, -gridHalf), Vec3(0.0f, 0.001f, 0.0f), Vec3(thin, 0.0f, 0.0f));
        uint32_t seg3Count = (uint32_t)(positions.size() / 3) - seg3Start;

        m_interactiveViewport.gridVertexCount = (uint32_t)(positions.size() / 3);
        m_interactiveViewport.gridSegments[0] = seg0Start;
        m_interactiveViewport.gridSegments[1] = seg0Count;
        m_interactiveViewport.gridSegments[2] = seg1Start;
        m_interactiveViewport.gridSegments[3] = seg1Count;
        m_interactiveViewport.gridSegments[4] = seg2Start;
        m_interactiveViewport.gridSegments[5] = seg2Count;
        m_interactiveViewport.gridSegments[6] = seg3Start;
        m_interactiveViewport.gridSegments[7] = seg3Count;
       
        if (seg2Count > 0) {
            std::string sample = "[Grid] +Z sample:";
            uint32_t vstart = seg2Start * 3;
            uint32_t vend = std::min((uint32_t)positions.size(), vstart + std::min(seg2Count, (uint32_t)6) * 3u);
            for (uint32_t vi = vstart; vi + 2 < vend; vi += 3) {
                sample += " (" + std::to_string(positions[vi]) + "," + std::to_string(positions[vi+1]) + "," + std::to_string(positions[vi+2]) + ")";
            }
            SCENE_LOG_INFO(sample);
        }

        VulkanRT::BufferCreateInfo vci{};
        vci.size = positions.size() * sizeof(float);
        vci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST | VulkanRT::BufferUsage::STORAGE;
        vci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        m_interactiveViewport.gridVertexBuffer = m_device->createBuffer(vci);
        if (m_interactiveViewport.gridVertexBuffer.buffer) {
            m_device->uploadBuffer(m_interactiveViewport.gridVertexBuffer, positions.data(), vci.size, 0);
        }

        VulkanRT::BufferCreateInfo nci{};
        nci.size = normals.size() * sizeof(float);
        nci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST | VulkanRT::BufferUsage::STORAGE;
        nci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        m_interactiveViewport.gridNormalBuffer = m_device->createBuffer(nci);
        if (m_interactiveViewport.gridNormalBuffer.buffer) {
            m_device->uploadBuffer(m_interactiveViewport.gridNormalBuffer, normals.data(), nci.size, 0);
        }
    }

    if (m_interactiveViewport.gridVertexBuffer.buffer && m_interactiveViewport.gridVertexCount > 0) {
        Matrix4x4 identity = Matrix4x4::identity();
        Matrix4x4 gridMvp = proj * view;

        VkBuffer gridBufs[3] = {
            m_interactiveViewport.gridVertexBuffer.buffer,
            m_interactiveViewport.gridNormalBuffer.buffer,
            m_interactiveViewport.identityInstanceBuffer.buffer
        };
        VkDeviceSize gridOff[3] = { 0, 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 3, gridBufs, gridOff);

        auto drawSeg = [&](uint32_t first, uint32_t count, float r, float g, float b) {
            if (!count) return;
            SolidPushConstants gp{};
            matrixToGL(gridMvp, gp.viewProj);
            matrixToGL(identity, gp.view);
            gp.useMatcap = -1;
            gp.overrideR = r; gp.overrideG = g; gp.overrideB = b;
            vkCmdPushConstants(cmd, m_interactiveViewport.pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SolidPushConstants), &gp);
            vkCmdDraw(cmd, count, 1, first, 0);
        };

        const auto* segments = m_interactiveViewport.gridSegments;
        drawSeg(segments[0], segments[1], 0.38f, 0.38f, 0.38f);
        drawSeg(segments[2], segments[3], 0.75f, 0.15f, 0.15f);
        // TEMP: brighten +Z axis for debugging
        drawSeg(segments[4], segments[5], 0.20f, 0.45f, 0.95f);
        drawSeg(segments[6], segments[7], 0.30f, 0.30f, 0.30f);
    }

    // --- Hair polyline overlay ---
    if (m_interactiveViewport.hairLinePipeline != VK_NULL_HANDLE &&
        m_interactiveViewport.hairLineVertexBuffer.buffer &&
        m_interactiveViewport.hairLineVertexCount > 0) {

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          m_interactiveViewport.hairLinePipeline);

        SolidPushConstants hp{};
        matrixToGL(viewProj, hp.viewProj);
        matrixToGL(view, hp.view);
        hp.useMatcap = 0;
        vkCmdPushConstants(cmd, m_interactiveViewport.pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(SolidPushConstants), &hp);

        VkBuffer hb = m_interactiveViewport.hairLineVertexBuffer.buffer;
        VkDeviceSize hOff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &hb, &hOff);
        vkCmdDraw(cmd, m_interactiveViewport.hairLineVertexCount, 1, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    m_device->endSingleTimeCommands(cmd);

    m_device->copyImageToBuffer(m_interactiveViewport.colorImage, m_interactiveViewport.stagingBuffer);

    std::vector<uint32_t>* framebuffer = static_cast<std::vector<uint32_t>*>(fb);
    const size_t pixelCount = (size_t)width * (size_t)height;
    if (framebuffer->size() != pixelCount) {
        framebuffer->resize(pixelCount);
    }
    m_device->downloadBuffer(m_interactiveViewport.stagingBuffer, framebuffer->data(), pixelCount * sizeof(uint32_t));

    if (s) {
        SDL_Surface* outSurf = static_cast<SDL_Surface*>(s);
        if (outSurf->pixels && outSurf->w == width && outSurf->h == height) {
            std::memcpy(outSurf->pixels, framebuffer->data(), pixelCount * sizeof(uint32_t));
        }
    }
    if (tex) {
        SDL_UpdateTexture(static_cast<SDL_Texture*>(tex), nullptr, framebuffer->data(), width * 4);
    }

    static uint64_t s_prevVisibleTriangleTotal = 0;
    static auto s_lastDiagLog = std::chrono::steady_clock::time_point{};
    const auto frameEnd = std::chrono::steady_clock::now();
    const double frameMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
    if (::render_settings.raster_viewport_quality_preset == ::RasterViewportQualityPreset::Auto) {
        if (frameMs > 120.0) {
            m_rasterScatterBudgetScale = std::max(0.35f, m_rasterScatterBudgetScale * 0.55f);
        } else if (frameMs > 60.0) {
            m_rasterScatterBudgetScale = std::max(0.35f, m_rasterScatterBudgetScale * 0.75f);
        } else if (frameMs > 40.0) {
            m_rasterScatterBudgetScale = std::max(0.35f, m_rasterScatterBudgetScale * 0.88f);
        } else if (frameMs < 20.0) {
            m_rasterScatterBudgetScale = std::min(1.0f, m_rasterScatterBudgetScale * 1.04f);
        }
    } else {
        m_rasterScatterBudgetScale = 1.0f;
    }
    const bool visibleTriangleCliff =
        s_prevVisibleTriangleTotal > 0 &&
        visibleTriangleTotal > (s_prevVisibleTriangleTotal * 3ull) / 2ull;
    const bool slowFrame = frameMs >= 100.0;
    const bool canLog = s_lastDiagLog.time_since_epoch().count() == 0 ||
        std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - s_lastDiagLog).count() >= 750;
    if (canLog && (slowFrame || visibleTriangleCliff)) {
        SCENE_LOG_INFO(
            "[Perf] [ViewportSolid] frame_ms=" + std::to_string(frameMs) +
            " visible_tris=" + std::to_string(visibleTriangleTotal) +
            " full_tris=" + std::to_string(visibleFullTriangleTotal) +
            " proxy_tris=" + std::to_string(visibleProxyTriangleTotal) +
            " scatter_budget=" + std::to_string(m_rasterScatterTriangleBudget) +
            " max_draw_tris=" + std::to_string(maxDrawTriangles) +
            " max_draw_instances=" + std::to_string(maxDrawInstances) +
            " max_draw_mesh='" + maxDrawMeshKey + "'");
        s_lastDiagLog = frameEnd;
    }
    s_prevVisibleTriangleTotal = visibleTriangleTotal;
}

bool VulkanViewportBackend::updateRasterMeshFromTriangles(
    const std::string& nodeName,
    const std::vector<std::shared_ptr<Triangle>>& triangles) {
    if (!m_device || !m_device->isInitialized() || triangles.empty()) return false;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    std::string targetKey;
    for (const auto& ri : m_rasterInstances) {
        if (viewportMatchesNodeNameForInstance(ri.nodeName, nodeName) ||
            viewportMatchesNodeNameForInstance(nodeName, ri.nodeName) ||
            ri.meshKey.find(nodeName) != std::string::npos) {
            targetKey = ri.meshKey;
            break;
        }
    }
    if (targetKey.empty()) return false;

    auto meshIt = m_rasterMeshes.find(targetKey);
    if (meshIt == m_rasterMeshes.end()) return false;

    auto& rmb = meshIt->second;
    const size_t vertCount = triangles.size() * 3;
    const size_t floatCount = vertCount * 3;

    std::vector<float> newPositions(floatCount);
    std::vector<float> newNormals(floatCount);

    size_t idx = 0;
    for (const auto& tri : triangles) {
        if (!tri) continue;
        const bool hasSharedTransform = (tri->getTransformHandle() != nullptr);
        for (int v = 0; v < 3; ++v) {
            Vec3 p = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
            Vec3 n = hasSharedTransform ? tri->getOriginalVertexNormal(v) : tri->getOriginalVertexNormal(v);
            newPositions[idx]   = p.x; newPositions[idx + 1] = p.y; newPositions[idx + 2] = p.z;
            newNormals[idx]     = n.x; newNormals[idx + 1]   = n.y; newNormals[idx + 2]   = n.z;
            idx += 3;
        }
    }

    const uint32_t newVertCount = static_cast<uint32_t>(vertCount);
    if (newVertCount != rmb.vertexCount) {
        const std::vector<uint32_t> preservedInstanceIndices = rmb.instanceIndices;
        m_device->waitIdle();
        destroyRasterMesh(rmb);
        rmb.instanceIndices = preservedInstanceIndices;

        rmb.vertexCount = newVertCount;
        VulkanRT::BufferCreateInfo vci{};
        vci.size = floatCount * sizeof(float);
        vci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST | VulkanRT::BufferUsage::STORAGE;
        vci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        vci.initialData = nullptr;
        rmb.vertexBuffer = m_device->createBuffer(vci);

        VulkanRT::BufferCreateInfo nci{};
        nci.size = floatCount * sizeof(float);
        nci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST | VulkanRT::BufferUsage::STORAGE;
        nci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        nci.initialData = nullptr;
        rmb.normalBuffer = m_device->createBuffer(nci);

        if (rmb.vertexBuffer.buffer) {
            m_device->uploadBuffer(rmb.vertexBuffer, newPositions.data(), floatCount * sizeof(float), 0);
        }
        if (rmb.normalBuffer.buffer) {
            m_device->uploadBuffer(rmb.normalBuffer, newNormals.data(), floatCount * sizeof(float), 0);
        }

        rmb.cpuPositions = std::move(newPositions);
        rmb.cpuNormals = std::move(newNormals);
        if (!rmb.instanceIndices.empty()) {
            uploadRasterInstanceBuffer(rmb);
        }
    } else if (rmb.cpuPositions.size() == floatCount) {
        size_t dirtyMin = floatCount;
        size_t dirtyMax = 0;
        for (size_t i = 0; i < floatCount; ++i) {
            if (newPositions[i] != rmb.cpuPositions[i] || newNormals[i] != rmb.cpuNormals[i]) {
                if (i < dirtyMin) dirtyMin = i;
                if (i > dirtyMax) dirtyMax = i;
            }
        }

        if (dirtyMin <= dirtyMax) {
            dirtyMin = (dirtyMin / 3) * 3;
            dirtyMax = ((dirtyMax / 3) + 1) * 3;
            if (dirtyMax > floatCount) dirtyMax = floatCount;

            const uint64_t byteOffset = dirtyMin * sizeof(float);
            const uint64_t byteSize = (dirtyMax - dirtyMin) * sizeof(float);

            m_device->uploadBuffer(rmb.vertexBuffer, &newPositions[dirtyMin], byteSize, byteOffset);
            m_device->uploadBuffer(rmb.normalBuffer, &newNormals[dirtyMin], byteSize, byteOffset);

            std::memcpy(&rmb.cpuPositions[dirtyMin], &newPositions[dirtyMin], byteSize);
            std::memcpy(&rmb.cpuNormals[dirtyMin], &newNormals[dirtyMin], byteSize);
        }
    } else {
        m_device->uploadBuffer(rmb.vertexBuffer, newPositions.data(), floatCount * sizeof(float));
        m_device->uploadBuffer(rmb.normalBuffer, newNormals.data(), floatCount * sizeof(float));
        rmb.cpuPositions = std::move(newPositions);
        rmb.cpuNormals = std::move(newNormals);
    }

    m_interactiveViewport.dirty = true;
    return true;
}

bool VulkanViewportBackend::patchRasterMeshTriangles(
    const std::string& nodeName,
    const std::vector<size_t>& dirtyIndices,
    const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& meshEntries) {
    if (!m_device || !m_device->isInitialized() || dirtyIndices.empty() || meshEntries.empty()) {
        return false;
    }
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    std::string targetKey;
    for (const auto& ri : m_rasterInstances) {
        if (viewportMatchesNodeNameForInstance(ri.nodeName, nodeName) ||
            viewportMatchesNodeNameForInstance(nodeName, ri.nodeName) ||
            ri.meshKey.find(nodeName) != std::string::npos) {
            targetKey = ri.meshKey;
            break;
        }
    }
    if (targetKey.empty()) return false;

    auto meshIt = m_rasterMeshes.find(targetKey);
    if (meshIt == m_rasterMeshes.end()) return false;

    auto& rmb = meshIt->second;
    const size_t expectedVertCount = meshEntries.size() * 3;
    const size_t expectedFloatCount = expectedVertCount * 3;

    if (rmb.vertexCount != static_cast<uint32_t>(expectedVertCount) ||
        rmb.cpuPositions.size() != expectedFloatCount) {
        return false;
    }

    size_t dirtyMinFloat = expectedFloatCount;
    size_t dirtyMaxFloat = 0;

    for (const size_t triIdx : dirtyIndices) {
        if (triIdx >= meshEntries.size()) continue;
        const auto& tri = meshEntries[triIdx].second;
        if (!tri) continue;

        const size_t baseFloat = triIdx * 9;
        if (baseFloat + 8 >= expectedFloatCount) continue;

        const bool hasSharedTransform = (tri->getTransformHandle() != nullptr);
        for (int v = 0; v < 3; ++v) {
            Vec3 p = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
            Vec3 n = hasSharedTransform ? tri->getOriginalVertexNormal(v) : tri->getOriginalVertexNormal(v);
            const size_t idx = baseFloat + static_cast<size_t>(v) * 3;
            rmb.cpuPositions[idx]     = p.x;
            rmb.cpuPositions[idx + 1] = p.y;
            rmb.cpuPositions[idx + 2] = p.z;
            rmb.cpuNormals[idx]       = n.x;
            rmb.cpuNormals[idx + 1]   = n.y;
            rmb.cpuNormals[idx + 2]   = n.z;
        }

        if (baseFloat < dirtyMinFloat) dirtyMinFloat = baseFloat;
        if (baseFloat + 8 > dirtyMaxFloat) dirtyMaxFloat = baseFloat + 8;
    }

    if (dirtyMinFloat <= dirtyMaxFloat) {
        dirtyMinFloat = (dirtyMinFloat / 3) * 3;
        dirtyMaxFloat = ((dirtyMaxFloat / 3) + 1) * 3;
        if (dirtyMaxFloat > expectedFloatCount) dirtyMaxFloat = expectedFloatCount;

        const uint64_t byteOffset = dirtyMinFloat * sizeof(float);
        const uint64_t byteSize = (dirtyMaxFloat - dirtyMinFloat) * sizeof(float);

        m_device->uploadBuffer(rmb.vertexBuffer, &rmb.cpuPositions[dirtyMinFloat], byteSize, byteOffset);
        m_device->uploadBuffer(rmb.normalBuffer, &rmb.cpuNormals[dirtyMinFloat], byteSize, byteOffset);
    }

    m_interactiveViewport.dirty = true;
    return true;
}

void VulkanViewportBackend::buildRasterGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (!m_device || !m_device->isInitialized()) return;

    // Skip rebuild if raster cache is still valid for the current scene generation.
    {
        extern std::atomic<uint64_t> g_scene_geometry_generation;
        const uint64_t curGen = g_scene_geometry_generation.load(std::memory_order_acquire);
        if (!m_rasterMeshes.empty() && m_rasterBuiltGeometryGeneration == curGen) {
            // Geometry hasn't changed — reuse existing raster buffers.
            m_rasterGeometryDirty = false;
            return;
        }
    }

    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_device->waitIdle();

    destroyAllRasterMeshes();

    auto hasInstancePrefix = [](const std::string& nodeName) -> bool {
        return nodeName.rfind("_inst_gid", 0) == 0;
    };

    size_t baseObjectCount = objects.size();
    while (baseObjectCount > 0) {
        const auto& obj = objects[baseObjectCount - 1];
        auto inst = std::dynamic_pointer_cast<HittableInstance>(obj);
        if (!inst || !hasInstancePrefix(inst->node_name)) {
            break;
        }
        --baseObjectCount;
    }

    auto ensureRasterMeshForTriangles = [&](const std::string& meshKey,
                                            const std::vector<std::shared_ptr<Triangle>>& triangles) {
        if (triangles.empty()) return;
        if (m_rasterMeshes.find(meshKey) != m_rasterMeshes.end()) return;

        std::vector<float> positions;
        std::vector<float> normals;
        positions.reserve(triangles.size() * 9);
        normals.reserve(triangles.size() * 9);

        // Compute local-space AABB while iterating vertices
        Vec3 bMin(1e18f, 1e18f, 1e18f), bMax(-1e18f, -1e18f, -1e18f);

        for (const auto& t : triangles) {
            if (!t) continue;
            for (int v = 0; v < 3; ++v) {
                Vec3 p = t->getOriginalVertexPosition(v);
                Vec3 n = t->getOriginalVertexNormal(v);
                positions.push_back(p.x); positions.push_back(p.y); positions.push_back(p.z);
                normals.push_back(n.x); normals.push_back(n.y); normals.push_back(n.z);
                bMin.x = std::min(bMin.x, p.x); bMin.y = std::min(bMin.y, p.y); bMin.z = std::min(bMin.z, p.z);
                bMax.x = std::max(bMax.x, p.x); bMax.y = std::max(bMax.y, p.y); bMax.z = std::max(bMax.z, p.z);
            }
        }
        // Cache the local bounding box for this mesh key
        m_rasterMeshBBoxes[meshKey] = AABB(bMin, bMax);

        RasterMeshBuffer rmb;
        rmb.vertexCount = static_cast<uint32_t>(positions.size() / 3);
        VulkanRT::BufferCreateInfo vci{};
        vci.size = positions.size() * sizeof(float);
        vci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST | VulkanRT::BufferUsage::STORAGE;
        vci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        vci.initialData = nullptr;
        rmb.vertexBuffer = m_device->createBuffer(vci);

        VulkanRT::BufferCreateInfo nci{};
        nci.size = normals.size() * sizeof(float);
        nci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST | VulkanRT::BufferUsage::STORAGE;
        nci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        nci.initialData = nullptr;
        rmb.normalBuffer = m_device->createBuffer(nci);

        if (rmb.vertexBuffer.buffer) {
            m_device->uploadBuffer(rmb.vertexBuffer, positions.data(), positions.size() * sizeof(float), 0);
        }
        if (rmb.normalBuffer.buffer) {
            m_device->uploadBuffer(rmb.normalBuffer, normals.data(), normals.size() * sizeof(float), 0);
        }

        bool hasSkinning = false;
        for (const auto& tri : triangles) {
            if (tri && tri->hasSkinData()) {
                hasSkinning = true;
                break;
            }
        }
        if (hasSkinning) {
            std::vector<int32_t> boneIndices;
            std::vector<float> boneWeights;
            boneIndices.reserve(rmb.vertexCount * 4);
            boneWeights.reserve(rmb.vertexCount * 4);

            for (const auto& tri : triangles) {
                if (!tri) continue;
                for (int v = 0; v < 3; ++v) {
                    const auto& weights = tri->getSkinBoneWeights(v);
                    for (int i = 0; i < 4; ++i) {
                        if (i < static_cast<int>(weights.size())) {
                            boneIndices.push_back(weights[i].first);
                            boneWeights.push_back(weights[i].second);
                        } else {
                            boneIndices.push_back(-1);
                            boneWeights.push_back(0.0f);
                        }
                    }
                }
            }

            VulkanRT::BufferCreateInfo sci{};
            sci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_DST;
            sci.location = VulkanRT::MemoryLocation::GPU_ONLY;

            sci.size = positions.size() * sizeof(float);
            rmb.baseVertexBuffer = m_device->createBuffer(sci);
            if (rmb.baseVertexBuffer.buffer) {
                m_device->uploadBuffer(rmb.baseVertexBuffer, positions.data(), sci.size, 0);
            }

            sci.size = normals.size() * sizeof(float);
            rmb.baseNormalBuffer = m_device->createBuffer(sci);
            if (rmb.baseNormalBuffer.buffer) {
                m_device->uploadBuffer(rmb.baseNormalBuffer, normals.data(), sci.size, 0);
            }

            sci.size = boneIndices.size() * sizeof(int32_t);
            rmb.boneIndexBuffer = m_device->createBuffer(sci);
            if (rmb.boneIndexBuffer.buffer) {
                m_device->uploadBuffer(rmb.boneIndexBuffer, boneIndices.data(), sci.size, 0);
            }

            sci.size = boneWeights.size() * sizeof(float);
            rmb.boneWeightBuffer = m_device->createBuffer(sci);
            if (rmb.boneWeightBuffer.buffer) {
                m_device->uploadBuffer(rmb.boneWeightBuffer, boneWeights.data(), sci.size, 0);
            }

            rmb.hasSkinning = rmb.baseVertexBuffer.buffer && rmb.baseNormalBuffer.buffer &&
                              rmb.boneIndexBuffer.buffer && rmb.boneWeightBuffer.buffer;
        }

        m_rasterMeshes[meshKey] = rmb;
    };

    auto ensureScatterProxyMesh = [&](const std::string& sourceMeshKey,
                                     const std::vector<std::shared_ptr<Triangle>>* sourceTriangles) {
        const std::string proxyMeshKey = sourceMeshKey + "::proxy";
        if (m_rasterMeshes.find(proxyMeshKey) != m_rasterMeshes.end()) {
            auto sourceIt = m_rasterMeshes.find(sourceMeshKey);
            if (sourceIt != m_rasterMeshes.end()) {
                sourceIt->second.proxyMeshKey = proxyMeshKey;
            }
            return;
        }

        auto bboxIt = m_rasterMeshBBoxes.find(sourceMeshKey);
        if (bboxIt == m_rasterMeshBBoxes.end() || !bboxIt->second.is_valid()) return;
        const AABB& bbox = bboxIt->second;

        const float minX = bbox.min.x;
        const float maxX = bbox.max.x;
        const float minY = bbox.min.y;
        const float maxY = bbox.max.y;
        const float minZ = bbox.min.z;
        const float maxZ = bbox.max.z;
        const float centerX = (minX + maxX) * 0.5f;
        const float centerZ = (minZ + maxZ) * 0.5f;
        const float height = std::max(maxY - minY, 1e-4f);
        const float fallbackRadius = std::max({ maxX - minX, maxZ - minZ, 1e-3f }) * 0.5f;

        std::vector<float> positions;
        std::vector<float> normals;
        positions.reserve(96 * 3);
        normals.reserve(96 * 3);

        auto addTri = [&](const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& n) {
            positions.insert(positions.end(), { a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z });
            normals.insert(normals.end(), { n.x, n.y, n.z, n.x, n.y, n.z, n.x, n.y, n.z });
        };

        auto buildFallbackCross = [&]() {
            const Vec3 x0(centerX, minY, minZ);
            const Vec3 x1(centerX, maxY, minZ);
            const Vec3 x2(centerX, maxY, maxZ);
            const Vec3 x3(centerX, minY, maxZ);
            addTri(x0, x1, x2, Vec3(1.0f, 0.0f, 0.0f));
            addTri(x0, x2, x3, Vec3(1.0f, 0.0f, 0.0f));
            addTri(x3, x2, x1, Vec3(-1.0f, 0.0f, 0.0f));
            addTri(x3, x1, x0, Vec3(-1.0f, 0.0f, 0.0f));

            const Vec3 z0(minX, minY, centerZ);
            const Vec3 z1(minX, maxY, centerZ);
            const Vec3 z2(maxX, maxY, centerZ);
            const Vec3 z3(maxX, minY, centerZ);
            addTri(z0, z1, z2, Vec3(0.0f, 0.0f, 1.0f));
            addTri(z0, z2, z3, Vec3(0.0f, 0.0f, 1.0f));
            addTri(z3, z2, z1, Vec3(0.0f, 0.0f, -1.0f));
            addTri(z3, z1, z0, Vec3(0.0f, 0.0f, -1.0f));
        };

        bool builtProfileProxy = false;
        if (sourceTriangles && !sourceTriangles->empty()) {
            static constexpr int kSliceCount = 7;
            struct ProxyPlaneProfile {
                Vec3 axis;
                Vec3 normal;
                float extents[kSliceCount]{};
                bool touched[kSliceCount]{};
            };

            const float invSqrt2 = 0.70710678f;
            ProxyPlaneProfile planes[] = {
                { Vec3(1.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f) },
                { Vec3(0.0f, 0.0f, 1.0f), Vec3(-1.0f, 0.0f, 0.0f) },
                { Vec3(invSqrt2, 0.0f, invSqrt2), Vec3(-invSqrt2, 0.0f, invSqrt2) },
                { Vec3(invSqrt2, 0.0f, -invSqrt2), Vec3(invSqrt2, 0.0f, invSqrt2) },
            };

            for (const auto& tri : *sourceTriangles) {
                if (!tri) continue;
                for (int v = 0; v < 3; ++v) {
                    const Vec3 p = tri->getOriginalVertexPosition(v);
                    const float t = std::clamp((p.y - minY) / height, 0.0f, 1.0f);
                    const int slice = static_cast<int>(std::round(t * static_cast<float>(kSliceCount - 1)));
                    const float dx = p.x - centerX;
                    const float dz = p.z - centerZ;
                    for (auto& plane : planes) {
                        const float extent = std::abs(dx * plane.axis.x + dz * plane.axis.z);
                        plane.extents[slice] = std::max(plane.extents[slice], extent);
                        plane.touched[slice] = true;
                    }
                }
            }

            for (auto& plane : planes) {
                int firstTouched = -1;
                for (int i = 0; i < kSliceCount; ++i) {
                    if (plane.touched[i]) {
                        firstTouched = i;
                        break;
                    }
                }
                if (firstTouched < 0) {
                    continue;
                }

                for (int i = firstTouched - 1; i >= 0; --i) {
                    plane.extents[i] = plane.extents[firstTouched];
                }

                int lastTouched = firstTouched;
                for (int i = firstTouched + 1; i < kSliceCount; ++i) {
                    if (plane.touched[i]) {
                        const int gap = i - lastTouched;
                        if (gap > 1) {
                            const float start = plane.extents[lastTouched];
                            const float end = plane.extents[i];
                            for (int fill = 1; fill < gap; ++fill) {
                                const float alpha = static_cast<float>(fill) / static_cast<float>(gap);
                                plane.extents[lastTouched + fill] = start + (end - start) * alpha;
                            }
                        }
                        lastTouched = i;
                    }
                }
                for (int i = lastTouched + 1; i < kSliceCount; ++i) {
                    plane.extents[i] = plane.extents[lastTouched];
                }

                for (int i = 0; i < kSliceCount; ++i) {
                    const float minExtent = fallbackRadius * 0.06f;
                    plane.extents[i] = std::max(plane.extents[i] * 0.96f, minExtent);
                }
            }

            auto addDoubleSidedRibbonQuad = [&](const Vec3& a0, const Vec3& a1,
                                                const Vec3& b0, const Vec3& b1,
                                                const Vec3& n) {
                addTri(a0, a1, b1, n);
                addTri(a0, b1, b0, n);
                addTri(b0, b1, a1, n * -1.0f);
                addTri(b0, a1, a0, n * -1.0f);
            };

            for (const auto& plane : planes) {
                for (int slice = 0; slice < kSliceCount - 1; ++slice) {
                    const float y0 = minY + height * (static_cast<float>(slice) / static_cast<float>(kSliceCount - 1));
                    const float y1 = minY + height * (static_cast<float>(slice + 1) / static_cast<float>(kSliceCount - 1));
                    const float e0 = plane.extents[slice];
                    const float e1 = plane.extents[slice + 1];

                    const Vec3 p0(centerX - plane.axis.x * e0, y0, centerZ - plane.axis.z * e0);
                    const Vec3 p1(centerX + plane.axis.x * e0, y0, centerZ + plane.axis.z * e0);
                    const Vec3 p2(centerX + plane.axis.x * e1, y1, centerZ + plane.axis.z * e1);
                    const Vec3 p3(centerX - plane.axis.x * e1, y1, centerZ - plane.axis.z * e1);
                    addDoubleSidedRibbonQuad(p0, p1, p3, p2, plane.normal);
                }
            }

            builtProfileProxy = !positions.empty();
        }

        if (!builtProfileProxy) {
            buildFallbackCross();
        }

        Vec3 proxyMin(1e18f, 1e18f, 1e18f);
        Vec3 proxyMax(-1e18f, -1e18f, -1e18f);
        for (size_t i = 0; i + 2 < positions.size(); i += 3) {
            proxyMin.x = std::min(proxyMin.x, positions[i + 0]);
            proxyMin.y = std::min(proxyMin.y, positions[i + 1]);
            proxyMin.z = std::min(proxyMin.z, positions[i + 2]);
            proxyMax.x = std::max(proxyMax.x, positions[i + 0]);
            proxyMax.y = std::max(proxyMax.y, positions[i + 1]);
            proxyMax.z = std::max(proxyMax.z, positions[i + 2]);
        }

        RasterMeshBuffer proxyMesh;
        proxyMesh.vertexCount = static_cast<uint32_t>(positions.size() / 3);
        proxyMesh.isScatterProxy = true;

        VulkanRT::BufferCreateInfo vci{};
        vci.size = positions.size() * sizeof(float);
        vci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST | VulkanRT::BufferUsage::STORAGE;
        vci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        proxyMesh.vertexBuffer = m_device->createBuffer(vci);

        VulkanRT::BufferCreateInfo nci{};
        nci.size = normals.size() * sizeof(float);
        nci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST | VulkanRT::BufferUsage::STORAGE;
        nci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        proxyMesh.normalBuffer = m_device->createBuffer(nci);

        if (proxyMesh.vertexBuffer.buffer) {
            m_device->uploadBuffer(proxyMesh.vertexBuffer, positions.data(), vci.size, 0);
        }
        if (proxyMesh.normalBuffer.buffer) {
            m_device->uploadBuffer(proxyMesh.normalBuffer, normals.data(), nci.size, 0);
        }

        m_rasterMeshBBoxes[proxyMeshKey] = AABB(proxyMin, proxyMax);
        m_rasterMeshes[proxyMeshKey] = std::move(proxyMesh);

        auto sourceIt = m_rasterMeshes.find(sourceMeshKey);
        if (sourceIt != m_rasterMeshes.end()) {
            sourceIt->second.proxyMeshKey = proxyMeshKey;
        }
    };

    struct RasterTriGroup {
        std::string meshKey;
        std::string nodeName;
        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<std::shared_ptr<Triangle>> triangles;
        Matrix4x4 transform;
        uint8_t mask = 0xFF;
    };

    std::vector<RasterTriGroup> groups;
    std::unordered_map<void*, size_t> groupByKey;

    std::function<void(const std::shared_ptr<Hittable>&)> processObj;
    processObj = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj) return;

        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (!inst->visible || !inst->source_triangles || inst->source_triangles->empty()) return;
            if (hasInstancePrefix(inst->node_name)) return;

            const auto srcPtr = reinterpret_cast<uintptr_t>(inst->source_triangles.get());
            std::string meshKey = "[Raster]-" + std::to_string(srcPtr) +
                                  "-" + std::to_string(inst->source_triangles->size());

            ensureRasterMeshForTriangles(meshKey, *inst->source_triangles);

            RasterInstance ri;
            ri.meshKey = meshKey;
            ri.nodeName = inst->node_name;
            ri.transform = inst->transform;
            ri.mask = 0xFF;
            m_rasterInstances.push_back(ri);

        } else if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
            for (auto& child : list->objects) processObj(child);
        } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            processObj(bvh->left);
            processObj(bvh->right);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (!tri->visible) return;

            void* groupKey = nullptr;
            auto triTransformHandle = tri->getTransformHandle();
            if (triTransformHandle) groupKey = triTransformHandle.get();
            else groupKey = tri.get();

            const bool hasSharedTransform = (triTransformHandle != nullptr);

            auto found = groupByKey.find(groupKey);
            if (found == groupByKey.end()) {
                RasterTriGroup g;
                std::string nodeName = tri->getNodeName();
                if (nodeName.empty()) nodeName = "[Solo-" + std::to_string(groups.size()) + "]";
                g.meshKey = "[Raster-Solo]-" + nodeName;
                g.nodeName = nodeName;
                g.transform = hasSharedTransform ? tri->getTransformMatrix() : Matrix4x4::identity();
                groups.push_back(std::move(g));
                found = groupByKey.emplace(groupKey, groups.size() - 1).first;
            }

            auto& grp = groups[found->second];
            grp.triangles.push_back(tri);
            for (int v = 0; v < 3; ++v) {
                Vec3 p = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
                Vec3 n = hasSharedTransform ? tri->getOriginalVertexNormal(v) : tri->getOriginalVertexNormal(v);
                grp.positions.push_back(p.x); grp.positions.push_back(p.y); grp.positions.push_back(p.z);
                grp.normals.push_back(n.x); grp.normals.push_back(n.y); grp.normals.push_back(n.z);
            }
        }
    };

    for (size_t i = 0; i < baseObjectCount; ++i) {
        processObj(objects[i]);
    }

    for (auto& grp : groups) {
        if (grp.positions.empty()) continue;

        // Compute local-space AABB from collected positions
        Vec3 bMin(1e18f, 1e18f, 1e18f), bMax(-1e18f, -1e18f, -1e18f);
        for (size_t pi = 0; pi + 2 < grp.positions.size(); pi += 3) {
            float px = grp.positions[pi], py = grp.positions[pi+1], pz = grp.positions[pi+2];
            bMin.x = std::min(bMin.x, px); bMin.y = std::min(bMin.y, py); bMin.z = std::min(bMin.z, pz);
            bMax.x = std::max(bMax.x, px); bMax.y = std::max(bMax.y, py); bMax.z = std::max(bMax.z, pz);
        }
        m_rasterMeshBBoxes[grp.meshKey] = AABB(bMin, bMax);

        RasterMeshBuffer rmb;
        rmb.vertexCount = static_cast<uint32_t>(grp.positions.size() / 3);

        VulkanRT::BufferCreateInfo vci{};
        vci.size = grp.positions.size() * sizeof(float);
        vci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST | VulkanRT::BufferUsage::STORAGE;
        vci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        vci.initialData = nullptr;
        rmb.vertexBuffer = m_device->createBuffer(vci);

        VulkanRT::BufferCreateInfo nci{};
        nci.size = grp.normals.size() * sizeof(float);
        nci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST | VulkanRT::BufferUsage::STORAGE;
        nci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        nci.initialData = nullptr;
        rmb.normalBuffer = m_device->createBuffer(nci);

        if (rmb.vertexBuffer.buffer) {
            m_device->uploadBuffer(rmb.vertexBuffer, grp.positions.data(), grp.positions.size() * sizeof(float), 0);
        }
        if (rmb.normalBuffer.buffer) {
            m_device->uploadBuffer(rmb.normalBuffer, grp.normals.data(), grp.normals.size() * sizeof(float), 0);
        }

        bool hasSkinning = false;
        for (const auto& tri : grp.triangles) {
            if (tri && tri->hasSkinData()) {
                hasSkinning = true;
                break;
            }
        }
        if (hasSkinning) {
            std::vector<int32_t> boneIndices;
            std::vector<float> boneWeights;
            boneIndices.reserve(rmb.vertexCount * 4);
            boneWeights.reserve(rmb.vertexCount * 4);

            for (const auto& tri : grp.triangles) {
                if (!tri) continue;
                for (int v = 0; v < 3; ++v) {
                    const auto& weights = tri->getSkinBoneWeights(v);
                    for (int i = 0; i < 4; ++i) {
                        if (i < static_cast<int>(weights.size())) {
                            boneIndices.push_back(weights[i].first);
                            boneWeights.push_back(weights[i].second);
                        } else {
                            boneIndices.push_back(-1);
                            boneWeights.push_back(0.0f);
                        }
                    }
                }
            }

            VulkanRT::BufferCreateInfo sci{};
            sci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_DST;
            sci.location = VulkanRT::MemoryLocation::GPU_ONLY;

            sci.size = grp.positions.size() * sizeof(float);
            rmb.baseVertexBuffer = m_device->createBuffer(sci);
            if (rmb.baseVertexBuffer.buffer) {
                m_device->uploadBuffer(rmb.baseVertexBuffer, grp.positions.data(), sci.size, 0);
            }

            sci.size = grp.normals.size() * sizeof(float);
            rmb.baseNormalBuffer = m_device->createBuffer(sci);
            if (rmb.baseNormalBuffer.buffer) {
                m_device->uploadBuffer(rmb.baseNormalBuffer, grp.normals.data(), sci.size, 0);
            }

            sci.size = boneIndices.size() * sizeof(int32_t);
            rmb.boneIndexBuffer = m_device->createBuffer(sci);
            if (rmb.boneIndexBuffer.buffer) {
                m_device->uploadBuffer(rmb.boneIndexBuffer, boneIndices.data(), sci.size, 0);
            }

            sci.size = boneWeights.size() * sizeof(float);
            rmb.boneWeightBuffer = m_device->createBuffer(sci);
            if (rmb.boneWeightBuffer.buffer) {
                m_device->uploadBuffer(rmb.boneWeightBuffer, boneWeights.data(), sci.size, 0);
            }

            rmb.hasSkinning = rmb.baseVertexBuffer.buffer && rmb.baseNormalBuffer.buffer &&
                              rmb.boneIndexBuffer.buffer && rmb.boneWeightBuffer.buffer;
        }

        m_rasterMeshes[grp.meshKey] = rmb;

        RasterInstance ri;
        ri.meshKey = grp.meshKey;
        ri.nodeName = grp.nodeName;
        ri.transform = grp.transform;
        ri.mask = 0xFF;
        m_rasterInstances.push_back(ri);
    }

    const auto& instanceGroups = InstanceManager::getInstance().getGroups();
    for (const auto& group : instanceGroups) {
        if (group.instances.empty() || group.sources.empty()) continue;

        for (size_t instanceIdx = 0; instanceIdx < group.instances.size(); ++instanceIdx) {
            const auto& inst = group.instances[instanceIdx];
            int srcIdx = inst.source_index;
            if (srcIdx < 0 || srcIdx >= static_cast<int>(group.sources.size())) srcIdx = 0;
            const auto& source = group.sources[srcIdx];

            const auto* triSource = source.centered_triangles_ptr ? source.centered_triangles_ptr.get() : nullptr;
            if ((!triSource || triSource->empty()) && source.triangles.empty()) continue;

            std::string meshKey;
            if (triSource) {
                const auto srcPtr = reinterpret_cast<uintptr_t>(triSource);
                meshKey = "[Raster-Group]-" + std::to_string(group.id) + "-" + std::to_string(srcIdx) +
                          "-" + std::to_string(srcPtr) + "-" + std::to_string(triSource->size());
                ensureRasterMeshForTriangles(meshKey, *triSource);
            } else {
                const auto srcPtr = reinterpret_cast<uintptr_t>(&source.triangles);
                meshKey = "[Raster-Group]-" + std::to_string(group.id) + "-" + std::to_string(srcIdx) +
                          "-" + std::to_string(srcPtr) + "-" + std::to_string(source.triangles.size());
                ensureRasterMeshForTriangles(meshKey, source.triangles);
            }

            auto rasterMeshIt = m_rasterMeshes.find(meshKey);
            if (rasterMeshIt != m_rasterMeshes.end()) {
                rasterMeshIt->second.isScatterGroup = true;
                ensureScatterProxyMesh(meshKey, triSource ? triSource : &source.triangles);
            }

            RasterInstance ri;
            ri.meshKey = meshKey;
            ri.nodeName = "_inst_gid" + std::to_string(group.id) + "_" + std::to_string(instanceIdx);
            ri.transform = inst.toMatrix();
            ri.mask = 0xFF;
            m_rasterInstances.push_back(std::move(ri));
        }
    }

    // Assign localBBox from cached mesh AABB and compute worldBBox per instance
    for (auto& ri : m_rasterInstances) {
        auto bboxIt = m_rasterMeshBBoxes.find(ri.meshKey);
        if (bboxIt != m_rasterMeshBBoxes.end()) {
            ri.localBBox = bboxIt->second;
            updateRasterInstanceWorldBBox(ri);
        }
    }

    for (uint32_t i = 0; i < static_cast<uint32_t>(m_rasterInstances.size()); ++i) {
        auto meshIt = m_rasterMeshes.find(m_rasterInstances[i].meshKey);
        if (meshIt == m_rasterMeshes.end()) continue;
        meshIt->second.instanceIndices.push_back(i);
    }
    for (auto& [key, mesh] : m_rasterMeshes) {
        uploadRasterInstanceBuffer(mesh);
    }

    m_rasterGeometryDirty = false;
    m_interactiveViewport.dirty = true;

    // Stamp current scene generation so we can skip redundant rebuilds later.
    {
        extern std::atomic<uint64_t> g_scene_geometry_generation;
        m_rasterBuiltGeometryGeneration = g_scene_geometry_generation.load(std::memory_order_acquire);
    }
}

void VulkanViewportBackend::syncRasterInstanceTransforms(
    const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (m_rasterInstances.empty()) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    std::unordered_map<std::string, Matrix4x4> transformMap;

    std::function<void(const std::shared_ptr<Hittable>&)> collectTransforms;
    collectTransforms = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj) return;
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (!inst->node_name.empty()) {
                transformMap[inst->node_name] = inst->transform;
            }
        } else if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
            for (auto& child : list->objects) collectTransforms(child);
        } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            collectTransforms(bvh->left);
            collectTransforms(bvh->right);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            auto th = tri->getTransformHandle();
            std::string name = tri->getNodeName();
            if (!name.empty() && th) {
                transformMap[name] = tri->getTransformMatrix();
            }
        }
    };

    for (const auto& obj : objects) {
        collectTransforms(obj);
    }

    bool changed = false;
    std::unordered_set<std::string> dirtyMeshKeys;
    for (auto& ri : m_rasterInstances) {
        auto it = transformMap.find(ri.nodeName);
        if (it != transformMap.end()) {
            if (!(ri.transform == it->second)) {
                ri.transform = it->second;
                changed = true;
                dirtyMeshKeys.insert(ri.meshKey);
            }
        }
    }
    if (changed) {
        // Recompute worldBBox for instances whose transforms changed
        for (auto& ri : m_rasterInstances) {
            if (dirtyMeshKeys.count(ri.meshKey)) {
                updateRasterInstanceWorldBBox(ri);
            }
        }
        for (const auto& meshKey : dirtyMeshKeys) {
            auto meshIt = m_rasterMeshes.find(meshKey);
            if (meshIt != m_rasterMeshes.end()) {
                uploadRasterInstanceBuffer(meshIt->second);
            }
        }
        m_interactiveViewport.dirty = true;
    }
}

void VulkanViewportBackend::syncRasterSkinnedVertices(
    const std::vector<std::shared_ptr<Hittable>>& objects,
    const std::vector<Matrix4x4>& boneMatrices) {
    if (m_rasterInstances.empty() || m_rasterMeshes.empty() || boneMatrices.empty()) return;
    if (m_viewportMode != ViewportMode::Solid && m_viewportMode != ViewportMode::Matcap) return;

    uint32_t skinnedMeshCount = 0;
    uint32_t gpuDispatchCount = 0;
    for (auto& [meshKey, mesh] : m_rasterMeshes) {
        if (!mesh.hasSkinning) continue;
        ++skinnedMeshCount;
        const bool dispatched = m_device->dispatchSkinningToBuffers(
            mesh.baseVertexBuffer,
            mesh.baseNormalBuffer,
            mesh.boneIndexBuffer,
            mesh.boneWeightBuffer,
            mesh.persistentBoneMatsBuffer,
            mesh.persistentBoneMatsBufSize,
            mesh.skinningDescSet,
            mesh.vertexBuffer,
            mesh.normalBuffer,
            mesh.vertexCount,
            boneMatrices);
        if (dispatched) {
            ++gpuDispatchCount;
        }
    }

    static bool loggedGpuViewportSkinning = false;
    static bool loggedCpuViewportSkinningFallback = false;
    static bool loggedMixedViewportSkinningPath = false;

    if (skinnedMeshCount > 0 && gpuDispatchCount == skinnedMeshCount) {
        if (!loggedGpuViewportSkinning) {
            VK_INFO() << "[ViewportSkinning] Using GPU compute skinning for "
                      << gpuDispatchCount << " raster mesh(es)" << std::endl;
            loggedGpuViewportSkinning = true;
        }
        m_interactiveViewport.dirty = true;
        return;
    }

    if (skinnedMeshCount > 0 && gpuDispatchCount == 0 && !loggedCpuViewportSkinningFallback) {
        VK_INFO() << "[ViewportSkinning] GPU skinning unavailable, falling back to CPU skinning for "
                  << skinnedMeshCount << " raster mesh(es)" << std::endl;
        loggedCpuViewportSkinningFallback = true;
    } else if (skinnedMeshCount > 0 && gpuDispatchCount != skinnedMeshCount && !loggedMixedViewportSkinningPath) {
        VK_INFO() << "[ViewportSkinning] Partial GPU skinning failure ("
                  << gpuDispatchCount << "/" << skinnedMeshCount
                  << "), using CPU fallback for viewport update" << std::endl;
        loggedMixedViewportSkinningPath = true;
    }

    struct SkinnedGroup {
        std::string meshKey;
        std::vector<std::shared_ptr<Triangle>> triangles;
    };
    std::unordered_map<void*, SkinnedGroup> skinnedGroups;

    std::unordered_map<std::string, std::string> nodeToMeshKey;
    for (const auto& ri : m_rasterInstances) {
        nodeToMeshKey[ri.nodeName] = ri.meshKey;
    }

    std::function<void(const std::shared_ptr<Hittable>&)> collectSkinned;
    collectSkinned = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj) return;
        if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
            for (auto& child : list->objects) collectSkinned(child);
        } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            collectSkinned(bvh->left);
            collectSkinned(bvh->right);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (!tri->visible || !tri->hasSkinData()) return;
            auto th = tri->getTransformHandle();
            void* groupKey = th ? (void*)th.get() : (void*)tri.get();
            auto& grp = skinnedGroups[groupKey];
            if (grp.meshKey.empty()) {
                std::string nodeName = tri->getNodeName();
                auto it = nodeToMeshKey.find(nodeName);
                if (it != nodeToMeshKey.end()) {
                    grp.meshKey = it->second;
                }
            }
            grp.triangles.push_back(tri);
        }
    };
    for (const auto& obj : objects) collectSkinned(obj);

    if (skinnedGroups.empty()) return;

    for (auto& [key, grp] : skinnedGroups) {
        if (grp.meshKey.empty() || grp.triangles.empty()) continue;
        auto meshIt = m_rasterMeshes.find(grp.meshKey);
        if (meshIt == m_rasterMeshes.end()) continue;

        auto& rmb = meshIt->second;
        const size_t vertCount = grp.triangles.size() * 3;
        const size_t floatCount = vertCount * 3;
        if (rmb.vertexCount != static_cast<uint32_t>(vertCount)) continue;

        std::vector<float> newPositions(floatCount);
        std::vector<float> newNormals(floatCount);
        size_t idx = 0;

        for (const auto& tri : grp.triangles) {
            for (int v = 0; v < 3; ++v) {
                Vec3 p = tri->apply_bone_to_vertex(v, boneMatrices);
                Vec3 n = tri->apply_bone_to_normal(
                    tri->getOriginalVertexNormal(v),
                    tri->getSkinBoneWeights(v),
                    boneMatrices);
                newPositions[idx]   = p.x; newPositions[idx + 1] = p.y; newPositions[idx + 2] = p.z;
                newNormals[idx]     = n.x; newNormals[idx + 1]   = n.y; newNormals[idx + 2]   = n.z;
                idx += 3;
            }
        }

        if (rmb.cpuPositions.size() == floatCount) {
            size_t dirtyMin = floatCount, dirtyMax = 0;
            for (size_t i = 0; i < floatCount; ++i) {
                if (newPositions[i] != rmb.cpuPositions[i] || newNormals[i] != rmb.cpuNormals[i]) {
                    if (i < dirtyMin) dirtyMin = i;
                    if (i > dirtyMax) dirtyMax = i;
                }
            }
            if (dirtyMin <= dirtyMax) {
                dirtyMin = (dirtyMin / 3) * 3;
                dirtyMax = ((dirtyMax / 3) + 1) * 3;
                if (dirtyMax > floatCount) dirtyMax = floatCount;
                const uint64_t byteOff = dirtyMin * sizeof(float);
                const uint64_t byteLen = (dirtyMax - dirtyMin) * sizeof(float);
                m_device->uploadBuffer(rmb.vertexBuffer, &newPositions[dirtyMin], byteLen, byteOff);
                m_device->uploadBuffer(rmb.normalBuffer, &newNormals[dirtyMin], byteLen, byteOff);
                std::memcpy(&rmb.cpuPositions[dirtyMin], &newPositions[dirtyMin], byteLen);
                std::memcpy(&rmb.cpuNormals[dirtyMin], &newNormals[dirtyMin], byteLen);
            }
        } else {
            m_device->uploadBuffer(rmb.vertexBuffer, newPositions.data(), floatCount * sizeof(float));
            m_device->uploadBuffer(rmb.normalBuffer, newNormals.data(), floatCount * sizeof(float));
            rmb.cpuPositions = std::move(newPositions);
            rmb.cpuNormals = std::move(newNormals);
        }
    }

    m_interactiveViewport.dirty = true;
}

void VulkanBackendAdapter::uploadHairViewportLines(const std::vector<float>& vertexData, uint32_t vertexCount) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;

    if (vertexData.empty() || vertexCount == 0) {
        if (m_interactiveViewport.hairLineVertexBuffer.buffer) {
            m_device->destroyBuffer(m_interactiveViewport.hairLineVertexBuffer);
        }
        m_interactiveViewport.hairLineVertexCount = 0;
        m_interactiveViewport.dirty = true;
        return;
    }

    const uint64_t byteSize = vertexData.size() * sizeof(float);

    // Reallocate only when current buffer is too small
    if (!m_interactiveViewport.hairLineVertexBuffer.buffer ||
        m_interactiveViewport.hairLineVertexBuffer.size < byteSize) {
        if (m_interactiveViewport.hairLineVertexBuffer.buffer) {
            m_device->destroyBuffer(m_interactiveViewport.hairLineVertexBuffer);
        }
        VulkanRT::BufferCreateInfo bci{};
        bci.size     = byteSize;
        bci.usage    = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        bci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        m_interactiveViewport.hairLineVertexBuffer = m_device->createBuffer(bci);
    }

    if (m_interactiveViewport.hairLineVertexBuffer.buffer) {
        m_device->uploadBuffer(m_interactiveViewport.hairLineVertexBuffer,
                               vertexData.data(), byteSize, 0);
        m_interactiveViewport.hairLineVertexCount = vertexCount;
        m_interactiveViewport.dirty = true;
        m_currentSamples = 0; // Allow render loop to re-render the viewport
    }
}

} // namespace Backend
