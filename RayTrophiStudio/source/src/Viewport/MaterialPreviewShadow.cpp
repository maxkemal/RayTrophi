#include "Backend/VulkanBackend.h"
#include "Viewport/RasterGpuCull.h"
#include "Viewport/RasterInstanceUpload.h"
#include "AreaLight.h"
#include "SpotLight.h"
#include "globals.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace Backend {
namespace {

constexpr uint32_t kAtlasSize = 4096u;
constexpr uint32_t kMaxLights = static_cast<uint32_t>(::kMaterialPreviewMaxSceneLights);
constexpr uint32_t kShadowRecordCount = kMaxLights + 1u;
constexpr uint32_t kMaxFaces = 6u;

struct alignas(16) ShadowRecordGPU {
    float viewProj[kMaxFaces][16]{};
    float atlasRect[kMaxFaces][4]{};
    uint32_t meta[4]{};
    float params[4]{};
};
static_assert(sizeof(ShadowRecordGPU) == 512u, "Preview shadow GLSL ABI changed");

struct ShadowView {
    Matrix4x4 viewProj;
    uint32_t tile = 0;
};

struct PreviewPush {
    float viewProj[16]{};
    float view[16]{};
    float cameraPos[4]{};
    float lightDir0[4]{};
    float lightDir1[4]{};
    float lightDir2[4]{};
    uint32_t materialMeta[4]{};
};
static_assert(sizeof(PreviewPush) == 208u, "Material preview push ABI changed");

void matrixToGL(const Matrix4x4& matrix, float out[16]) {
    const Matrix4x4 transposed = matrix.transpose();
    std::memcpy(out, transposed.m, sizeof(float) * 16u);
}

Vec3 safeDirection(const Vec3& value, const Vec3& fallback) {
    return value.length() > 1e-5f ? value.normalize() : fallback;
}

Matrix4x4 makeView(const Vec3& eye, const Vec3& center, Vec3 up) {
    Vec3 forward = safeDirection(center - eye, Vec3(0.0f, 0.0f, -1.0f));
    up = safeDirection(up, Vec3(0.0f, 1.0f, 0.0f));
    if (std::abs(Vec3::dot(forward, up)) > 0.98f)
        up = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 side = Vec3::cross(forward, up).normalize();
    if (side.length() < 1e-5f) side = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 cameraUp = Vec3::cross(side, forward);
    return Matrix4x4(
        side.x, side.y, side.z, -Vec3::dot(side, eye),
        cameraUp.x, cameraUp.y, cameraUp.z, -Vec3::dot(cameraUp, eye),
        -forward.x, -forward.y, -forward.z, Vec3::dot(forward, eye),
        0.0f, 0.0f, 0.0f, 1.0f);
}

Matrix4x4 makePerspective(float fovDegrees, float aspect, float nearZ, float farZ) {
    const float f = 1.0f / std::tan(fovDegrees * 0.5f * 3.14159265358979f / 180.0f);
    Matrix4x4 p = Matrix4x4::zero();
    p.m[0][0] = f / aspect;
    p.m[1][1] = -f;
    p.m[2][2] = farZ / (nearZ - farZ);
    p.m[2][3] = (farZ * nearZ) / (nearZ - farZ);
    p.m[3][2] = -1.0f;
    return p;
}

Matrix4x4 makeOrtho(float halfX, float halfY, float nearZ, float farZ) {
    Matrix4x4 p = Matrix4x4::zero();
    p.m[0][0] = 1.0f / halfX;
    p.m[1][1] = -1.0f / halfY;
    p.m[2][2] = 1.0f / (nearZ - farZ);
    p.m[2][3] = nearZ / (nearZ - farZ);
    p.m[3][3] = 1.0f;
    return p;
}

std::vector<uint32_t> loadSpv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || (bytes % 4) != 0) return {};
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4u);
    if (!file.read(reinterpret_cast<char*>(words.data()), bytes)) return {};
    return words;
}

VkShaderModule makeModule(VkDevice device, const std::vector<uint32_t>& words) {
    if (words.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = words.size() * sizeof(uint32_t);
    ci.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    return vkCreateShaderModule(device, &ci, nullptr, &module) == VK_SUCCESS
        ? module : VK_NULL_HANDLE;
}

} // namespace

void VulkanBackendAdapter::updateMaterialPreviewProgramBinding() {
    if (!m_device) return;
    const VkDescriptorSet set = m_interactiveViewport.materialPreviewDescSet;
    if (set == VK_NULL_HANDLE) return;

    if (!m_device->m_matProgramBuffer.buffer) {
        const uint32_t emptyProgram = 0u;
        m_device->updateMatProgramBuffer(&emptyProgram, sizeof(emptyProgram));
    }
    if (!m_device->m_matProgramBuffer.buffer) return;

    VkDescriptorBufferInfo info{};
    info.buffer = m_device->m_matProgramBuffer.buffer;
    info.offset = 0;
    info.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 16;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(m_device->getDevice(), 1, &write, 0, nullptr);
}

class MaterialPreviewShadowResources {
public:
    VulkanRT::ImageHandle atlas;
    VkSampler sampler = VK_NULL_HANDLE;
    VulkanRT::BufferHandle records;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkPipeline opaquePipeline = VK_NULL_HANDLE;
    VkPipeline alphaPipeline = VK_NULL_HANDLE;
    VkPipeline skyPipeline = VK_NULL_HANDLE;
    VkDescriptorSet boundSet = VK_NULL_HANDLE;
    VkImageView boundWorldView = VK_NULL_HANDLE;
    std::array<VkImageView, 3> boundLutViews{};
    VkImageLayout atlasLayout = VK_IMAGE_LAYOUT_GENERAL;
    std::array<ShadowRecordGPU, kShadowRecordCount> cpuRecords{};
    // ★ Boyut SceneGlobals ile ayni olmali (bkz. static_assert). Kucuk
    //   kalirsa memcmp degisimi goremez ve post ayarlari GPU'ya HIC gitmez.
    std::array<unsigned char, 144> lastGlobals{};
    bool globalsValid = false;
    std::vector<ShadowView> views;
    uint32_t frameTileSize = 512u;
    uint32_t frameTilesPerRow = 8u;
    uint32_t frameTileCapacity = 64u;
    uint32_t frameLightBudget = 8u;
    uint32_t shadowedLights = 0;
    bool worldSunShadow = false;
    bool ready = false;
};

bool VulkanBackendAdapter::ensureMaterialPreviewShadowResources(const std::string& shaderDir) {
    if (!m_device || !m_device->isInitialized() ||
        m_interactiveViewport.materialPreviewPipelineLayout == VK_NULL_HANDLE)
        return false;
    if (m_materialPreviewShadows && m_materialPreviewShadows->ready) return true;

    // Fail before allocating the 4096^2 atlas when a packaged build has stale
    // shader outputs. The viewport can remain usable without repeatedly
    // allocating and destroying a large depth image on every resource check.
    const auto opaqueWords = loadSpv(shaderDir + "/material_preview_shadow_opaque.spv");
    const auto alphaVertWords = loadSpv(shaderDir + "/material_preview_shadow.spv");
    const auto alphaFragWords = loadSpv(shaderDir + "/material_preview_shadow_frag.spv");
    const auto skyVertWords = loadSpv(shaderDir + "/material_preview_sky_vert.spv");
    const auto skyFragWords = loadSpv(shaderDir + "/material_preview_sky.spv");
    if (opaqueWords.empty() || alphaVertWords.empty() || alphaFragWords.empty() ||
        skyVertWords.empty() || skyFragWords.empty()) {
        if (!m_materialPreviewShadows)
            m_materialPreviewShadows = std::make_shared<MaterialPreviewShadowResources>();
        static bool warnedMissingShaders = false;
        if (!warnedMissingShaders) {
            SCENE_LOG_WARN(
                "[MaterialPreview] Shadow/sky shaders are missing; Scene lighting "
                "continues without the shadow atlas or world background until "
                "shader outputs are rebuilt.");
            warnedMissingShaders = true;
        }
        return false;
    }

    destroyMaterialPreviewShadowResources();
    auto state = std::make_shared<MaterialPreviewShadowResources>();
    VkDevice device = m_device->getDevice();
    state->atlas = m_device->createImage2D(
        kAtlasSize, kAtlasSize, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT);
    if (!state->atlas.image) return false;

    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sci.maxLod = 0.0f;
    if (vkCreateSampler(device, &sci, nullptr, &state->sampler) != VK_SUCCESS) {
        m_device->destroyImage(state->atlas);
        return false;
    }

    VulkanRT::BufferCreateInfo bci{};
    bci.size = sizeof(state->cpuRecords);
    bci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_DST;
    bci.location = VulkanRT::MemoryLocation::GPU_ONLY;
    state->records = m_device->createBuffer(bci);
    if (!state->records.buffer) {
        vkDestroySampler(device, state->sampler, nullptr);
        m_device->destroyImage(state->atlas);
        return false;
    }

    VkAttachmentDescription attachment{};
    attachment.format = VK_FORMAT_D32_SFLOAT;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depthRef{0u, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &depthRef;
    VkRenderPassCreateInfo rpci{};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1;
    rpci.pAttachments = &attachment;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    if (vkCreateRenderPass(device, &rpci, nullptr, &state->renderPass) != VK_SUCCESS) {
        m_materialPreviewShadows = state;
        destroyMaterialPreviewShadowResources();
        return false;
    }
    VkFramebufferCreateInfo fbci{};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = state->renderPass;
    fbci.attachmentCount = 1;
    fbci.pAttachments = &state->atlas.view;
    fbci.width = kAtlasSize;
    fbci.height = kAtlasSize;
    fbci.layers = 1;
    if (vkCreateFramebuffer(device, &fbci, nullptr, &state->framebuffer) != VK_SUCCESS) {
        m_materialPreviewShadows = state;
        destroyMaterialPreviewShadowResources();
        return false;
    }

    VkShaderModule opaqueVert = makeModule(device, opaqueWords);
    VkShaderModule alphaVert = makeModule(device, alphaVertWords);
    VkShaderModule alphaFrag = makeModule(device, alphaFragWords);
    VkShaderModule skyVert = makeModule(device, skyVertWords);
    VkShaderModule skyFrag = makeModule(device, skyFragWords);

    auto createPipeline = [&](bool alpha, VkPipeline& output) -> bool {
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = alpha ? alphaVert : opaqueVert;
        stages[0].pName = "main";
        uint32_t stageCount = 1;
        if (alpha) {
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = alphaFrag;
            stages[1].pName = "main";
            stageCount = 2;
        }
        if (!stages[0].module || (alpha && !stages[1].module)) return false;

        VkVertexInputBindingDescription bindings[4]{};
        bindings[0] = {0u, sizeof(float) * 3u, VK_VERTEX_INPUT_RATE_VERTEX};
        bindings[1] = {2u, sizeof(uint32_t), VK_VERTEX_INPUT_RATE_VERTEX};
        bindings[2] = {3u, sizeof(float) * 16u, VK_VERTEX_INPUT_RATE_INSTANCE};
        bindings[3] = {4u, sizeof(float) * 2u, VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs[7]{};
        attrs[0] = {0u, 0u, VK_FORMAT_R32G32B32_SFLOAT, 0u};
        attrs[1] = {2u, 2u, VK_FORMAT_R32_UINT, 0u};
        attrs[2] = {3u, 3u, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 0u};
        attrs[3] = {4u, 3u, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 4u};
        attrs[4] = {5u, 3u, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 8u};
        attrs[5] = {6u, 3u, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 12u};
        attrs[6] = {7u, 4u, VK_FORMAT_R32G32_SFLOAT, 0u};
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = alpha ? 4u : 2u;
        vi.pVertexBindingDescriptions = bindings;
        if (!alpha) {
            // Opaque uses sparse bindings 0 and 3; build a compact description.
            bindings[1] = bindings[2];
            attrs[1] = attrs[2]; attrs[2] = attrs[3]; attrs[3] = attrs[4]; attrs[4] = attrs[5];
            vi.vertexAttributeDescriptionCount = 5u;
        } else {
            vi.vertexAttributeDescriptionCount = 7u;
        }
        vi.pVertexAttributeDescriptions = attrs;
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.depthBiasEnable = VK_TRUE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                      VK_DYNAMIC_STATE_DEPTH_BIAS};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 3; dyn.pDynamicStates = dynStates;
        VkGraphicsPipelineCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pci.stageCount = stageCount; pci.pStages = stages;
        pci.pVertexInputState = &vi; pci.pInputAssemblyState = &ia;
        pci.pViewportState = &vp; pci.pRasterizationState = &rs;
        pci.pMultisampleState = &ms; pci.pDepthStencilState = &ds;
        pci.pColorBlendState = &cb; pci.pDynamicState = &dyn;
        pci.layout = m_interactiveViewport.materialPreviewPipelineLayout;
        pci.renderPass = state->renderPass;
        return vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &output) == VK_SUCCESS;
    };

    const bool pipelinesOk = createPipeline(false, state->opaquePipeline) &&
                             createPipeline(true, state->alphaPipeline);

    auto createSkyPipeline = [&]() -> bool {
        if (!skyVert || !skyFrag || m_interactiveViewport.renderPass == VK_NULL_HANDLE)
            return false;
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = skyVert;
        stages[0].pName = "main";
        stages[1] = stages[0];
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = skyFrag;
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_FALSE;
        ds.depthWriteEnable = VK_FALSE;
        VkPipelineColorBlendAttachmentState attachmentBlend{};
        attachmentBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &attachmentBlend;
        VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dynStates;
        VkGraphicsPipelineCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pci.stageCount = 2;
        pci.pStages = stages;
        pci.pVertexInputState = &vi;
        pci.pInputAssemblyState = &ia;
        pci.pViewportState = &vp;
        pci.pRasterizationState = &rs;
        pci.pMultisampleState = &ms;
        pci.pDepthStencilState = &ds;
        pci.pColorBlendState = &cb;
        pci.pDynamicState = &dyn;
        pci.layout = m_interactiveViewport.materialPreviewPipelineLayout;
        pci.renderPass = m_interactiveViewport.renderPass;
        return vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr,
                                         &state->skyPipeline) == VK_SUCCESS;
    };
    const bool skyPipelineOk = pipelinesOk && createSkyPipeline();
    if (opaqueVert) vkDestroyShaderModule(device, opaqueVert, nullptr);
    if (alphaVert) vkDestroyShaderModule(device, alphaVert, nullptr);
    if (alphaFrag) vkDestroyShaderModule(device, alphaFrag, nullptr);
    if (skyVert) vkDestroyShaderModule(device, skyVert, nullptr);
    if (skyFrag) vkDestroyShaderModule(device, skyFrag, nullptr);
    if (!skyPipelineOk) {
        m_materialPreviewShadows = state;
        destroyMaterialPreviewShadowResources();
        return false;
    }

    state->ready = true;
    m_materialPreviewShadows = std::move(state);
    return true;
}

void VulkanBackendAdapter::destroyMaterialPreviewShadowResources() {
    if (!m_materialPreviewShadows || !m_device) {
        m_materialPreviewShadows.reset();
        return;
    }
    VkDevice device = m_device->getDevice();
    auto& s = *m_materialPreviewShadows;
    if (s.opaquePipeline) vkDestroyPipeline(device, s.opaquePipeline, nullptr);
    if (s.alphaPipeline) vkDestroyPipeline(device, s.alphaPipeline, nullptr);
    if (s.skyPipeline) vkDestroyPipeline(device, s.skyPipeline, nullptr);
    if (s.framebuffer) vkDestroyFramebuffer(device, s.framebuffer, nullptr);
    if (s.renderPass) vkDestroyRenderPass(device, s.renderPass, nullptr);
    if (s.records.buffer) m_device->destroyBuffer(s.records);
    if (s.sampler) vkDestroySampler(device, s.sampler, nullptr);
    if (s.atlas.image) m_device->destroyImage(s.atlas);
    m_materialPreviewShadows.reset();
}

void VulkanBackendAdapter::prepareMaterialPreviewShadowFrame() {
    auto state = m_materialPreviewShadows;
    if (!state || !state->ready) return;
    state->views.clear();
    state->shadowedLights = 0;
    state->worldSunShadow = false;
    state->cpuRecords = {};
    if (m_viewportMode != ViewportMode::MaterialPreview ||
        ::render_settings.material_preview_lighting_preset != MaterialPreviewLightingPreset::Scene) {
        return;
    }

    // The CPU snapshot is copied with vkCmdUpdateBuffer in the same command
    // buffer that renders the atlas. This preserves frame-ring concurrency:
    // camera motion never overwrites storage read by an older in-flight frame.
    const float cameraDistance = (m_camera.origin - m_camera.lookAt).length();
    const float cascadeRadius = std::clamp(cameraDistance * 2.0f, 10.0f, 250.0f);
    const Vec3 cascadeCenter = m_camera.lookAt;
    uint32_t nextTile = 0;
    uint32_t gpuLightIndex = 0;
    state->frameTileSize = static_cast<uint32_t>(rasterShadowTileSize(
        ::render_settings.raster_viewport_quality_preset));
    state->frameTilesPerRow = kAtlasSize / state->frameTileSize;
    state->frameTileCapacity = state->frameTilesPerRow * state->frameTilesPerRow;
    state->frameLightBudget = static_cast<uint32_t>(rasterShadowLightBudget(
        ::render_settings.raster_viewport_quality_preset));
    const uint32_t requestedDirectionalCascades = static_cast<uint32_t>(
        rasterDirectionalShadowCascades(
            ::render_settings.raster_viewport_quality_preset));

    auto addView = [&](ShadowRecordGPU& record, uint32_t face, const Matrix4x4& matrix) {
        const uint32_t tile = nextTile++;
        matrixToGL(matrix, record.viewProj[face]);
        const float scale = static_cast<float>(state->frameTileSize) /
                            static_cast<float>(kAtlasSize);
        record.atlasRect[face][0] =
            static_cast<float>(tile % state->frameTilesPerRow) * scale;
        record.atlasRect[face][1] =
            static_cast<float>(tile / state->frameTilesPerRow) * scale;
        record.atlasRect[face][2] = scale;
        record.atlasRect[face][3] = scale;
        state->views.push_back({matrix, tile});
    };

    auto addDirectionalViews = [&](ShadowRecordGPU& record, Vec3 toLight,
                                   uint32_t cascadeCount) {
        toLight = safeDirection(toLight, Vec3(0.0f, 1.0f, 0.0f));
        Vec3 up(0.0f, 1.0f, 0.0f);
        const Vec3 forward = toLight * -1.0f;
        if (std::abs(Vec3::dot(forward, up)) > 0.98f)
            up = Vec3(0.0f, 0.0f, 1.0f);
        const Vec3 side = Vec3::cross(forward, up).normalize();
        const Vec3 cameraUp = Vec3::cross(side, forward).normalize();
        static constexpr float radiusFactors[3] = {0.18f, 0.45f, 1.0f};
        static constexpr float minimumRadii[3] = {4.0f, 8.0f, 10.0f};
        for (uint32_t cascade = 0; cascade < cascadeCount; ++cascade) {
            // If atlas pressure leaves fewer cascades, preserve the far
            // coverage first: 1 -> far, 2 -> middle+far, 3 -> near+middle+far.
            const uint32_t profile = cascade + (3u - cascadeCount);
            const float radius = std::max(
                cascadeRadius * radiusFactors[profile], minimumRadii[profile]);
            const float worldTexel = (radius * 2.0f) /
                                     static_cast<float>(state->frameTileSize);
            const float centerX = Vec3::dot(side, cascadeCenter);
            const float centerY = Vec3::dot(cameraUp, cascadeCenter);
            const float snappedX = std::floor(centerX / worldTexel) * worldTexel;
            const float snappedY = std::floor(centerY / worldTexel) * worldTexel;
            const Vec3 snappedCenter = cascadeCenter +
                side * (snappedX - centerX) + cameraUp * (snappedY - centerY);
            const Vec3 eye = snappedCenter + toLight * radius * 2.0f;
            const Matrix4x4 view = makeView(eye, snappedCenter, up);
            addView(record, cascade,
                    makeOrtho(radius, radius, 0.05f, radius * 4.0f) * view);
        }
    };

    if (m_cachedWorld.mode == WORLD_MODE_NISHITA &&
        m_cachedWorld.nishita.sun_intensity > 0.0f &&
        nextTile < state->frameTileCapacity) {
        const uint32_t sunCascadeCount = std::min(
            requestedDirectionalCascades,
            state->frameTileCapacity - nextTile);
        ShadowRecordGPU& sunRecord = state->cpuRecords[kMaxLights];
        sunRecord.meta[0] = 1u;
        sunRecord.meta[1] = 1u;
        sunRecord.meta[2] = sunCascadeCount;
        sunRecord.meta[3] = kMaxLights;
        sunRecord.params[0] = 0.0015f;
        sunRecord.params[1] = 0.01f;
        sunRecord.params[2] = 1.0f / static_cast<float>(kAtlasSize);
        sunRecord.params[3] = 1.0f;
        addDirectionalViews(
            sunRecord,
            Vec3(m_cachedWorld.nishita.sun_direction.x,
                 m_cachedWorld.nishita.sun_direction.y,
                 m_cachedWorld.nishita.sun_direction.z),
            sunCascadeCount);
        state->worldSunShadow = true;
    }

    for (const auto& light : m_cachedLights) {
        if (!light || !light->visible) continue;
        if (gpuLightIndex >= kMaxLights) break;
        ShadowRecordGPU& record = state->cpuRecords[gpuLightIndex];
        const LightType type = light->type();
        const uint32_t faces = type == LightType::Point ? 6u :
            (type == LightType::Directional
                ? requestedDirectionalCascades : 1u);
        if (state->shadowedLights >= state->frameLightBudget ||
            faces == 0u ||
            nextTile + faces > state->frameTileCapacity) {
            ++gpuLightIndex;
            continue;
        }

        record.meta[0] = 1u;
        record.meta[1] = type == LightType::Point ? 0u :
                         type == LightType::Directional ? 1u :
                         type == LightType::Area ? 2u : 3u;
        record.meta[2] = faces;
        record.meta[3] = gpuLightIndex;
        record.params[0] = 0.0015f;
        record.params[1] = 0.01f;
        record.params[2] = 1.0f / static_cast<float>(kAtlasSize);
        record.params[3] = type == LightType::Area ? 1.75f : 1.0f;

        if (type == LightType::Directional) {
            addDirectionalViews(record, light->direction * -1.0f, faces);
        } else if (type == LightType::Point) {
            static const Vec3 dirs[6] = {
                Vec3(1,0,0), Vec3(-1,0,0), Vec3(0,1,0),
                Vec3(0,-1,0), Vec3(0,0,1), Vec3(0,0,-1)};
            static const Vec3 ups[6] = {
                Vec3(0,-1,0), Vec3(0,-1,0), Vec3(0,0,1),
                Vec3(0,0,-1), Vec3(0,-1,0), Vec3(0,-1,0)};
            const float farZ = std::clamp((light->position - cascadeCenter).length() + cascadeRadius,
                                          10.0f, 2000.0f);
            Matrix4x4 proj = makePerspective(90.0f, 1.0f, 0.05f, farZ);
            for (uint32_t face = 0; face < 6u; ++face)
                addView(record, face, proj * makeView(light->position,
                                                      light->position + dirs[face], ups[face]));
        } else {
            Vec3 direction = safeDirection(light->direction, Vec3(0.0f, -1.0f, 0.0f));
            float fov = 120.0f;
            if (type == LightType::Spot) {
                if (auto spot = std::dynamic_pointer_cast<SpotLight>(light))
                    fov = std::clamp(spot->getAngleDegrees() * 2.0f, 2.0f, 175.0f);
            } else if (type == LightType::Area) {
                if (auto area = std::dynamic_pointer_cast<AreaLight>(light))
                    direction = safeDirection(Vec3::cross(area->getU(), area->getV()),
                                              Vec3(0.0f, -1.0f, 0.0f));
            }
            const float farZ = std::clamp((light->position - cascadeCenter).length() + cascadeRadius,
                                          10.0f, 2000.0f);
            addView(record, 0u,
                    makePerspective(fov, 1.0f, 0.05f, farZ) *
                    makeView(light->position, light->position + direction, Vec3(0,1,0)));
        }
        ++state->shadowedLights;
        ++gpuLightIndex;
    }
}

void VulkanBackendAdapter::updateMaterialPreviewShadowWorldBindings() {
    auto state = m_materialPreviewShadows;
    const VkDescriptorSet set = m_interactiveViewport.materialPreviewDescSet;
    if (!m_device || set == VK_NULL_HANDLE) return;

    VulkanRT::ImageHandle fallback{};
    auto fallbackIt = m_uploadedImages.find(m_interactiveViewport.envMapStudioID);
    if (fallbackIt != m_uploadedImages.end()) fallback = fallbackIt->second;
    VulkanRT::ImageHandle world = fallback;
    auto worldIt = m_uploadedImages.find(m_envTexID);
    const bool hasWorldEnv = m_envTexID > 0 && worldIt != m_uploadedImages.end();
    if (hasWorldEnv) world = worldIt->second;
    if (!world.view || !world.sampler) return;

    std::array<VulkanRT::ImageHandle, 3> atmosphereLuts{};
    bool hasAtmosphereLuts = true;
    for (uint32_t i = 0; i < 3; ++i) {
        atmosphereLuts[i] = m_device->m_lutImages[i];
        if (!atmosphereLuts[i].view || !atmosphereLuts[i].sampler)
            hasAtmosphereLuts = false;
    }

    const bool descriptorChanged = !state || state->boundSet != set ||
        state->boundWorldView != world.view ||
        (state && (state->boundLutViews[0] != atmosphereLuts[0].view ||
                   state->boundLutViews[1] != atmosphereLuts[1].view ||
                   state->boundLutViews[2] != atmosphereLuts[2].view));
    if (descriptorChanged) {
        drainInteractiveViewportInFlight();
        VkDescriptorBufferInfo shadowBuffer{};
        shadowBuffer.buffer = state && state->records.buffer
            ? state->records.buffer : m_interactiveViewport.materialPreviewSceneGlobals.buffer;
        shadowBuffer.range = VK_WHOLE_SIZE;
        VkDescriptorImageInfo shadowImage{};
        shadowImage.sampler = state && state->sampler ? state->sampler : world.sampler;
        shadowImage.imageView = state && state->atlas.view ? state->atlas.view : world.view;
        shadowImage.imageLayout = state && state->atlas.view
            ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo worldImage{};
        worldImage.sampler = world.sampler;
        worldImage.imageView = world.view;
        worldImage.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo lutImages[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            const auto& image = hasAtmosphereLuts ? atmosphereLuts[i] : world;
            lutImages[i].sampler = image.sampler;
            lutImages[i].imageView = image.view;
            lutImages[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        VkWriteDescriptorSet writes[6]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = set; writes[0].dstBinding = 7;
        writes[0].descriptorCount = 1; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &shadowBuffer;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = set; writes[1].dstBinding = 8;
        writes[1].descriptorCount = 1; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &shadowImage;
        writes[2] = writes[1]; writes[2].dstBinding = 9; writes[2].pImageInfo = &worldImage;
        for (uint32_t i = 0; i < 3; ++i) {
            writes[3 + i] = writes[1];
            writes[3 + i].dstBinding = 10 + i;
            writes[3 + i].pImageInfo = &lutImages[i];
        }
        vkUpdateDescriptorSets(m_device->getDevice(), 6, writes, 0, nullptr);
        if (state) {
            state->boundSet = set;
            state->boundWorldView = world.view;
            for (uint32_t i = 0; i < 3; ++i)
                state->boundLutViews[i] = atmosphereLuts[i].view;
        }
    }
    const bool hasPrefilteredIbl =
        bindMaterialPreviewIblDescriptors(set, world) &&
        m_cachedWorld.mode == WORLD_MODE_HDRI;

    if (m_interactiveViewport.materialPreviewSceneGlobalsMapped) {
        struct alignas(16) SceneGlobals {
            uint32_t lightCount, flags, shadowedCount, worldMode;
            float worldColor[4];
            float worldParams[4];
            float worldSun[4];
            float atmosphereA[4]; // multi enabled, factor, mie g, mie density
            float atmosphereB[4]; // planet radius, atmosphere height, 0, 0
            // ★★★ post.* aynasi. AYNI buffer'i material_preview_frag.frag ve
            //   material_preview_sky.frag okuyor; duzen o iki dosyada da yazili.
            //   ABI 112 -> 144: buffer boyutu, memset ve lastGlobals dizisi ayni
            //   partide buyutuldu. Birini atlarsan belirti "renk biraz farkli"
            //   olur, ve o belirti kimse tarafindan bug diye raporlanmaz.
            float postA[4];   // exposure, gamma, saturation, colorTemperature
            float postB[4];   // vignetteStrength, toneMapping, vignetteEnabled, cameraExposure
            float postC[4];   // viewportWidth, viewportHeight, 0, 0
        } globals{};
        static_assert(sizeof(SceneGlobals) == 144u, "Preview scene globals GLSL ABI changed");
        globals.lightCount = m_device->m_lightCount > kMaxLights ? kMaxLights : m_device->m_lightCount;
        const bool nishitaOverlay =
            m_cachedWorld.mode == WORLD_MODE_NISHITA &&
            m_cachedWorld.advanced.env_overlay_enabled != 0 && hasWorldEnv;
        globals.flags = (state && state->ready ? 1u : 0u) |
                        (hasWorldEnv ? 2u : 0u) |
                        (nishitaOverlay ? 4u : 0u) |
                        (hasAtmosphereLuts ? 8u : 0u) |
                        (state && state->worldSunShadow ? 16u : 0u) |
                        (hasPrefilteredIbl ? 32u : 0u);
        globals.shadowedCount = state ? state->shadowedLights : 0u;
        globals.worldMode = static_cast<uint32_t>(m_cachedWorld.mode);
        globals.worldColor[0] = m_cachedWorld.color.x;
        globals.worldColor[1] = m_cachedWorld.color.y;
        globals.worldColor[2] = m_cachedWorld.color.z;
        globals.worldColor[3] = m_cachedWorld.mode == WORLD_MODE_NISHITA
            ? m_cachedWorld.nishita.sun_size : m_cachedWorld.color_intensity;
        globals.worldParams[0] = nishitaOverlay
            ? m_cachedWorld.advanced.env_overlay_rotation *
                (3.14159265358979323846f / 180.0f)
            : m_cachedWorld.env_rotation;
        globals.worldParams[1] = nishitaOverlay
            ? m_cachedWorld.advanced.env_overlay_intensity
            : m_cachedWorld.env_intensity;
        globals.worldParams[2] = m_cachedWorld.nishita.atmosphere_intensity;
        globals.worldParams[3] = nishitaOverlay
            ? static_cast<float>(m_cachedWorld.advanced.env_overlay_blend_mode)
            : 0.0f;
        globals.worldSun[0] = m_cachedWorld.nishita.sun_direction.x;
        globals.worldSun[1] = m_cachedWorld.nishita.sun_direction.y;
        globals.worldSun[2] = m_cachedWorld.nishita.sun_direction.z;
        globals.worldSun[3] = m_cachedWorld.nishita.sun_intensity;
        globals.atmosphereA[0] = static_cast<float>(m_cachedWorld.advanced.multi_scatter_enabled);
        globals.atmosphereA[1] = m_cachedWorld.advanced.multi_scatter_factor;
        globals.atmosphereA[2] = m_cachedWorld.nishita.mie_anisotropy;
        globals.atmosphereA[3] = m_cachedWorld.nishita.mie_density;
        globals.atmosphereB[0] = m_cachedWorld.nishita.planet_radius;
        globals.atmosphereB[1] = m_cachedWorld.nishita.atmosphere_height;

        // ★★ Onizleme artik projenin post zincirinden geciyor (post_chain.glsl).
        //   Kaynak g_display_post, onun kaynagi da ColorProcessor -- yani
        //   Rendered ile Realtime AYNI sayilari okuyor. Ayri bir onizleme
        //   pozlamasi tutmak, tam olarak kapatmaya calistigimiz ayrisma olurdu.
        globals.postA[0] = g_display_post.exposure;
        globals.postA[1] = g_display_post.gamma;
        globals.postA[2] = g_display_post.saturation;
        globals.postA[3] = g_display_post.color_temperature;
        globals.postB[0] = g_display_post.vignette_strength;
        globals.postB[1] = static_cast<float>(g_display_post.tone_mapping);
        globals.postB[2] = static_cast<float>(g_display_post.vignette_enabled);
        // ★ Eskiden bos olan slot: kamera pozlama ucgeni. ABI (112) degismedi.
        globals.postB[3] = g_display_post.camera_exposure;
        // Vignette CPU formulunu tutturmak icin piksel boyutu gerekiyor.
        globals.postC[0] = static_cast<float>(m_interactiveViewport.width);
        globals.postC[1] = static_cast<float>(m_interactiveViewport.height);
        globals.postC[2] = 0.0f;
        globals.postC[3] = 0.0f;

        const bool changed = !state || !state->globalsValid ||
            std::memcmp(state->lastGlobals.data(), &globals, sizeof(globals)) != 0;
        if (changed) {
            drainInteractiveViewportInFlight();
            std::memcpy(m_interactiveViewport.materialPreviewSceneGlobalsMapped,
                        &globals, sizeof(globals));
            if (state) {
                std::memcpy(state->lastGlobals.data(), &globals, sizeof(globals));
                state->globalsValid = true;
            }
        }
    }
}

void VulkanBackendAdapter::recordMaterialPreviewSkyPass(
    VkCommandBuffer cmd, const Matrix4x4& view, uint32_t width, uint32_t height) {
    auto state = m_materialPreviewShadows;
    if (!state || !state->ready || !state->skyPipeline || cmd == VK_NULL_HANDLE ||
        width == 0u || height == 0u ||
        m_viewportMode != ViewportMode::MaterialPreview ||
        ::render_settings.material_preview_lighting_preset !=
            MaterialPreviewLightingPreset::Scene)
        return;

    VkViewport viewport{};
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, {width, height}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, state->skyPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_interactiveViewport.materialPreviewPipelineLayout,
                            0, 1, &m_interactiveViewport.materialPreviewDescSet,
                            0, nullptr);

    PreviewPush push{};
    matrixToGL(view, push.view);
    push.cameraPos[0] = m_camera.origin.x;
    push.cameraPos[1] = m_camera.origin.y;
    push.cameraPos[2] = m_camera.origin.z;
    push.cameraPos[3] = static_cast<float>(width) / static_cast<float>(height);
    const float fov = std::clamp(m_camera.fov, 1.0f, 179.0f);
    push.lightDir0[3] = std::tan(fov * 0.5f * 3.14159265358979f / 180.0f);
    vkCmdPushConstants(cmd, m_interactiveViewport.materialPreviewPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(push), &push);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void VulkanBackendAdapter::recordMaterialPreviewShadowPass(VkCommandBuffer cmd) {
    auto state = m_materialPreviewShadows;
    if (!state || !state->ready || state->views.empty() || cmd == VK_NULL_HANDLE) return;

    VkBufferMemoryBarrier recordsBarrier{};
    recordsBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    recordsBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    recordsBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    recordsBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    recordsBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    recordsBarrier.buffer = state->records.buffer;
    recordsBarrier.offset = 0;
    recordsBarrier.size = sizeof(state->cpuRecords);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 1, &recordsBarrier, 0, nullptr);
    vkCmdUpdateBuffer(cmd, state->records.buffer, 0,
                      sizeof(state->cpuRecords), state->cpuRecords.data());
    recordsBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    recordsBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 1, &recordsBarrier, 0, nullptr);

    VkImageMemoryBarrier toDepth{};
    toDepth.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toDepth.oldLayout = state->atlasLayout;
    toDepth.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    toDepth.srcAccessMask = state->atlasLayout == VK_IMAGE_LAYOUT_GENERAL ? 0u : VK_ACCESS_SHADER_READ_BIT;
    toDepth.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    toDepth.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDepth.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDepth.image = state->atlas.image;
    toDepth.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    toDepth.subresourceRange.levelCount = 1;
    toDepth.subresourceRange.layerCount = 1;
    const VkPipelineStageFlags atlasSourceStage =
        state->atlasLayout == VK_IMAGE_LAYOUT_GENERAL
            ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    vkCmdPipelineBarrier(cmd, atlasSourceStage,
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &toDepth);

    VkClearValue clear{};
    clear.depthStencil = {1.0f, 0u};
    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = state->renderPass;
    begin.framebuffer = state->framebuffer;
    begin.renderArea.extent = {kAtlasSize, kAtlasSize};
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetDepthBias(cmd, 1.25f, 0.0f, 1.75f);

    PreviewPush push{};
    push.materialMeta[0] = m_interactiveViewport.materialPreviewBoundMaterialCount;
    push.materialMeta[3] = m_interactiveViewport.materialPreviewTextureArrayLen;
    for (const ShadowView& shadowView : state->views) {
        const uint32_t tileX = shadowView.tile % state->frameTilesPerRow;
        const uint32_t tileY = shadowView.tile / state->frameTilesPerRow;
        VkViewport viewport{};
        viewport.x = static_cast<float>(tileX * state->frameTileSize);
        viewport.y = static_cast<float>(tileY * state->frameTileSize);
        viewport.width = static_cast<float>(state->frameTileSize);
        viewport.height = static_cast<float>(state->frameTileSize);
        viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
        VkRect2D scissor{{static_cast<int32_t>(tileX * state->frameTileSize),
                          static_cast<int32_t>(tileY * state->frameTileSize)},
                         {state->frameTileSize, state->frameTileSize}};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        matrixToGL(shadowView.viewProj, push.viewProj);
        vkCmdPushConstants(cmd, m_interactiveViewport.materialPreviewPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(push), &push);

        for (const auto& [meshKey, mesh] : m_rasterMeshes) {
            const bool cullDriven = m_rasterGpuCullActive && m_rasterGpuCull &&
                                    mesh.cullDrawSlot != UINT32_MAX;
            VkBuffer instanceBuffer = cullDriven
                ? static_cast<VkBuffer>(m_rasterGpuCull->compactedInstanceBuffer())
                : ((m_rasterUseGlobalInstBuffer && m_rasterGlobalInstBuf)
                    ? static_cast<VkBuffer>(m_rasterGlobalInstBuf->vkBuffer())
                    : mesh.instanceBuffer.buffer);
            if (!mesh.vertexBuffer.buffer || !instanceBuffer || mesh.vertexCount == 0u) continue;
            if (!cullDriven && mesh.instanceCount == 0u) continue;
            const VkDeviceSize instanceOffset = cullDriven
                ? static_cast<VkDeviceSize>(mesh.cullOutBase) * 64ull : 0ull;
            // Any mesh with material IDs takes the alpha-aware path so scalar
            // opacity also controls casting. UV-less geometry falls back to a
            // dummy stream; its coordinates are only observed if a texture is
            // actually assigned.
            const bool alpha = mesh.matIdBuffer.buffer != VK_NULL_HANDLE;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              alpha ? state->alphaPipeline : state->opaquePipeline);
            if (alpha) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        m_interactiveViewport.materialPreviewPipelineLayout,
                                        0, 1, &m_interactiveViewport.materialPreviewDescSet, 0, nullptr);
            }
            VkBuffer buffers[5] = {
                mesh.vertexBuffer.buffer,
                mesh.normalBuffer.buffer ? mesh.normalBuffer.buffer : mesh.vertexBuffer.buffer,
                mesh.matIdBuffer.buffer ? mesh.matIdBuffer.buffer : mesh.vertexBuffer.buffer,
                instanceBuffer,
                mesh.uvBuffer.buffer ? mesh.uvBuffer.buffer : mesh.vertexBuffer.buffer};
            VkDeviceSize offsets[5] = {0, 0, 0, instanceOffset, 0};
            vkCmdBindVertexBuffers(cmd, 0, 5, buffers, offsets);
            if (cullDriven) {
                VkBuffer indirect = static_cast<VkBuffer>(m_rasterGpuCull->commandBuffer());
                VkDeviceSize commandOffset = RasterGpuCull::commandOffset(mesh.cullDrawSlot);
                if (mesh.indexBuffer.buffer && mesh.indexCount > 0u) {
                    vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexedIndirect(cmd, indirect, commandOffset, 1, RasterGpuCull::kCommandStride);
                } else {
                    vkCmdDrawIndirect(cmd, indirect, commandOffset, 1, RasterGpuCull::kCommandStride);
                }
            } else {
                const uint32_t first = m_rasterUseGlobalInstBuffer ? mesh.firstInstance : 0u;
                if (mesh.indexBuffer.buffer && mesh.indexCount > 0u) {
                    vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(cmd, mesh.indexCount, mesh.instanceCount, 0, 0, first);
                } else {
                    vkCmdDraw(cmd, mesh.vertexCount, mesh.instanceCount, 0, first);
                }
            }
        }
    }
    vkCmdEndRenderPass(cmd);

    VkImageMemoryBarrier toSample = toDepth;
    toSample.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    toSample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toSample.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    toSample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &toSample);
    state->atlasLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

} // namespace Backend
