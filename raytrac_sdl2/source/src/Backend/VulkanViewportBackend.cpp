#include "Backend/VulkanViewportBackend.h"
#include "HittableInstance.h"
#include "HittableList.h"
#include "InstanceManager.h"
#include "ParallelBVHNode.h"
#include "Texture.h"
#include "Triangle.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#if defined(_WIN32)
#include <windows.h>
#include <excpt.h>
#endif
#include <cstring>
#include <fstream>
#include <filesystem>
#include <functional>
#include <future>
#include <thread>
#include <unordered_map>
#include <unordered_set>

extern RenderSettings render_settings;

namespace Backend {
namespace {

#if defined(_WIN32)
static VkResult safeCreateMaterialPreviewPipeline(VkDevice device,
                                                  const VkGraphicsPipelineCreateInfo* info,
                                                  VkPipeline* outPipeline) {
    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    __try {
        result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, info, nullptr, outPipeline);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        if (outPipeline) *outPipeline = VK_NULL_HANDLE;
        result = VK_ERROR_INITIALIZATION_FAILED;
    }
    return result;
}
#else
static VkResult safeCreateMaterialPreviewPipeline(VkDevice device,
                                                  const VkGraphicsPipelineCreateInfo* info,
                                                  VkPipeline* outPipeline) {
    return vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, info, nullptr, outPipeline);
}
#endif

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

// ── Preview environment map bake (CPU replication of material_preview_frag.frag GLSL) ──
// Generates a 128×64 RGBA32F equirectangular map from the procedural studio/outdoor
// functions so the fragment shader can do a single texture lookup for specular reflections
// instead of the former analytical approximation.

struct BakeVec3 { float x, y, z; };

static inline float bkDot(BakeVec3 a, BakeVec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline BakeVec3 bkNorm(BakeVec3 v) {
    float l = std::sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return (l > 1e-7f) ? BakeVec3{v.x/l, v.y/l, v.z/l} : BakeVec3{0,1,0};
}
static inline float bkSmooth(float e0, float e1, float x) {
    float t = std::fmaxf(0.f, std::fminf(1.f, (x-e0)/(e1-e0)));
    return t*t*(3.f-2.f*t);
}
static inline float bkMix(float a, float b, float t) { return a*(1.f-t)+b*t; }
static inline float bkClamp(float v, float lo, float hi) { return std::fmaxf(lo, std::fminf(hi, v)); }

static BakeVec3 bakeStudioEnv(BakeVec3 dir) {
    float up = bkClamp(dir.y*0.5f+0.5f, 0.f, 1.f);
    float t = bkSmooth(0.05f, 1.f, up);
    BakeVec3 base{bkMix(0.07f,0.42f,t), bkMix(0.07f,0.44f,t), bkMix(0.08f,0.46f,t)};

    auto lobe = [&](BakeVec3 d, float exp_) {
        return std::powf(std::fmaxf(bkDot(dir, bkNorm(d)), 0.f), exp_);
    };
    float key  = lobe({0.62f, 0.54f, 0.56f}, 28.f);
    float fill = lobe({-0.52f, 0.38f, 0.76f}, 20.f);
    float rim  = lobe({-0.18f, 0.26f,-0.95f}, 56.f);

    return {
        base.x + 1.00f*1.8f*key + 0.72f*0.9f*fill + 0.95f*0.7f*rim,
        base.y + 0.98f*1.8f*key + 0.80f*0.9f*fill + 0.96f*0.7f*rim,
        base.z + 0.95f*1.8f*key + 0.96f*0.9f*fill + 1.00f*0.7f*rim
    };
}

static BakeVec3 bakeOutdoorEnv(BakeVec3 dir) {
    float up = bkClamp(dir.y*0.5f+0.5f, 0.f, 1.f);
    BakeVec3 sky{bkMix(0.22f,0.55f,bkSmooth(0.1f,1.f,up)),
                  bkMix(0.26f,0.70f,bkSmooth(0.1f,1.f,up)),
                  bkMix(0.32f,0.96f,bkSmooth(0.1f,1.f,up))};
    BakeVec3 gnd{0.12f,0.10f,0.08f};
    float sun = std::powf(std::fmaxf(bkDot(dir, bkNorm({0.32f,0.82f,0.46f})), 0.f), 220.f);
    float sunScale = 2.2f;
    return {
        bkMix(gnd.x,sky.x,up) + 1.0f*sunScale*sun,
        bkMix(gnd.y,sky.y,up) + 0.96f*sunScale*sun,
        bkMix(gnd.z,sky.z,up) + 0.82f*sunScale*sun
    };
}

// Returns a 128×64 RGBA32F buffer (4 floats/pixel, alpha = 1).
static std::vector<float> bakeEnvMap(bool outdoor) {
    constexpr int W = 128, H = 64;
    std::vector<float> pixels(W * H * 4);
    constexpr float PI = 3.14159265359f;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float u = (x + 0.5f) / float(W);
            float v = (y + 0.5f) / float(H);
            float phi   = (2.f*u - 1.f) * PI;   // -π … π
            float theta = v * PI;                  // 0 … π (top→bottom)
            BakeVec3 dir{
                std::sinf(theta) * std::cosf(phi),
                std::cosf(theta),
                std::sinf(theta) * std::sinf(phi)
            };
            BakeVec3 c = outdoor ? bakeOutdoorEnv(dir) : bakeStudioEnv(dir);
            float* p = &pixels[(y*W + x)*4];
            p[0]=c.x; p[1]=c.y; p[2]=c.z; p[3]=1.f;
        }
    }
    return pixels;
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

void VulkanViewportBackend::uploadTerrainLayerMaterials(const std::vector<TerrainLayerData>& layers) {
    // Viewport-only implementation: does NOT call the RT base class path.
    // No RT descriptor sets are touched — this backend has no RT pipeline.
    if (!m_device || !m_device->isInitialized() || layers.empty()) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_device->waitIdle();

    // Convert IBackend::TerrainLayerData → VkTerrainLayerData.
    // Splat map textures are resolved from the viewport backend's own uploaded-image cache
    // (they were already uploaded by the material texture upload pass).
    std::vector<VulkanRT::VkTerrainLayerData> gpuLayers;
    gpuLayers.reserve(layers.size());

    for (const auto& ld : layers) {
        VulkanRT::VkTerrainLayerData gld{};
        for (int k = 0; k < 4; ++k) {
            gld.layer_mat_id[k]   = ld.layer_mat_id[k];
            gld.layer_uv_scale[k] = ld.layer_uv_scale[k];
        }
        gld.layer_count = ld.layer_count;

        // Resolve splat map: check viewport texture cache first, upload if missing.
        if (ld.splatMapTexture) {
            Texture* splatTex = reinterpret_cast<Texture*>(ld.splatMapTexture);
            if (splatTex && splatTex->is_loaded()) {
                uint64_t cacheKey = static_cast<uint64_t>(ld.splatMapTexture) << 1;
                if (splatTex->vulkan_dirty) {
                    auto oldIt = m_uploadedImageIDs.find(cacheKey);
                    if (oldIt != m_uploadedImageIDs.end()) {
                        int64_t oldId = oldIt->second;
                        if (oldId) this->destroyTexture(oldId);
                    }
                    splatTex->vulkan_dirty = false;
                }
                auto it = m_uploadedImageIDs.find(cacheKey);
                if (it != m_uploadedImageIDs.end()) {
                    gld.splat_map_tex = static_cast<uint32_t>(it->second);
                } else {
                    const auto& px = splatTex->pixels;
                    if (!px.empty()) {
                        std::vector<uint8_t> tmp(splatTex->width * splatTex->height * 4);
                        for (size_t i = 0; i < px.size(); ++i) {
                            tmp[i*4+0]=px[i].r; tmp[i*4+1]=px[i].g;
                            tmp[i*4+2]=px[i].b; tmp[i*4+3]=px[i].a;
                        }
                        int64_t id = this->uploadTexture2D(tmp.data(), splatTex->width, splatTex->height, 4, false, false);
                        if (id > 0) { m_uploadedImageIDs[cacheKey] = id; gld.splat_map_tex = static_cast<uint32_t>(id); }
                    }
                }
            }
        }
        gpuLayers.push_back(gld);
    }

    // Upload terrain layer SSBO to this backend's device (no RT descriptor touched).
    m_device->updateTerrainLayerBuffer(
        gpuLayers.data(),
        gpuLayers.size() * sizeof(VulkanRT::VkTerrainLayerData),
        static_cast<uint32_t>(gpuLayers.size()));

    // Update material preview descriptor binding 3.
    const VkDescriptorSet ds = m_interactiveViewport.materialPreviewDescSet;
    if (ds == VK_NULL_HANDLE || !m_device->m_terrainLayerBuffer.buffer) return;
    VkDevice vkDev = m_device->getDevice();
    if (!vkDev) return;

    VkDescriptorBufferInfo tbi{};
    tbi.buffer = m_device->m_terrainLayerBuffer.buffer;
    tbi.offset = 0;
    tbi.range  = m_device->m_terrainLayerBuffer.size > 0 ? m_device->m_terrainLayerBuffer.size : VK_WHOLE_SIZE;
    VkWriteDescriptorSet wds{};
    wds.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet          = ds;
    wds.dstBinding      = 3;
    wds.descriptorCount = 1;
    wds.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo     = &tbi;
    vkUpdateDescriptorSets(vkDev, 1, &wds, 0, nullptr);
    m_interactiveViewport.dirty = true;
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
    m_externalMaterialCount = static_cast<uint32_t>(
        (size > 0) ? (size / sizeof(VulkanRT::VkGpuMaterial)) : 0);
    m_interactiveViewport.dirty = true;
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
    if (m_interactiveViewport.materialPreviewPipeline != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipeline(vkDevice, m_interactiveViewport.materialPreviewPipeline, nullptr);
        m_interactiveViewport.materialPreviewPipeline = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.materialPreviewPipelineLayout != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipelineLayout(vkDevice, m_interactiveViewport.materialPreviewPipelineLayout, nullptr);
        m_interactiveViewport.materialPreviewPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.materialPreviewDescPool != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyDescriptorPool(vkDevice, m_interactiveViewport.materialPreviewDescPool, nullptr);
        m_interactiveViewport.materialPreviewDescPool = VK_NULL_HANDLE;
        m_interactiveViewport.materialPreviewDescSet = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.materialPreviewDescLayout != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyDescriptorSetLayout(vkDevice, m_interactiveViewport.materialPreviewDescLayout, nullptr);
        m_interactiveViewport.materialPreviewDescLayout = VK_NULL_HANDLE;
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

    if (m_interactiveViewport.materialPreviewPipeline == VK_NULL_HANDLE && !m_materialPreviewPipelineGaveUp) {
        const std::string mpVertPath = shaderDir + "/material_preview.spv";
        const std::string mpFragPath = shaderDir + "/material_preview_frag.spv";
        // ── Capability gate ───────────────────────────────────────────────
        // Material preview fragment shader uses `sampler2D textures[]` with
        // nonuniformEXT indexing and push constants of 208 bytes. Without
        // VK_EXT_descriptor_indexing (Vulkan 1.2 core) the runtime-sized
        // texture array is undefined; older/low-end drivers also cap push
        // constants at 128 bytes. When either requirement is missing we
        // silently fall back to Matcap — solid pipeline + matcap descriptor
        // set — and notify the user via the HUD message channel.
        const bool mpHasDescIdx =
            m_device && m_device->getCapabilities().supportsDescriptorIndexing;
        constexpr uint32_t kMpPushBytes =
            sizeof(float) * 48 + sizeof(uint32_t) * 4; // 208 bytes
        uint32_t mpMaxPushBytes = 0;
        if (m_device && m_device->getPhysicalDevice() != VK_NULL_HANDLE) {
            VkPhysicalDeviceProperties mpDevProps{};
            vkGetPhysicalDeviceProperties(m_device->getPhysicalDevice(), &mpDevProps);
            mpMaxPushBytes = mpDevProps.limits.maxPushConstantsSize;
        }
        const bool mpPushOk = mpMaxPushBytes >= kMpPushBytes;
        const bool mpCapsOk = mpHasDescIdx && mpPushOk;

        if (!mpCapsOk) {
            m_materialPreviewPipelineGaveUp = true;
            if (!m_materialPreviewUnsupportedNotified) {
                m_materialPreviewUnsupportedNotified = true;
                std::string reason = !mpHasDescIdx
                    ? "VK_EXT_descriptor_indexing missing"
                    : ("maxPushConstantsSize=" + std::to_string(mpMaxPushBytes) +
                       " < " + std::to_string(kMpPushBytes));
                SCENE_LOG_WARN("[VulkanViewportBackend] Material preview disabled: " + reason +
                               ". Falling back to Matcap.");
                if (m_statusCallback) {
                    m_statusCallback(
                        "Material Preview not supported on this GPU (" + reason +
                        "). Falling back to Matcap mode.",
                        1 /* warning */);
                }
            }
            // Leave materialPreviewPipeline == VK_NULL_HANDLE so the render
            // path at useMaterialPreview check naturally binds the solid
            // (matcap) pipeline instead.
        } else
        if (std::filesystem::exists(mpVertPath) && std::filesystem::exists(mpFragPath)) {
            std::vector<uint32_t> mpVertSPV = loadViewportSPV(mpVertPath);
            std::vector<uint32_t> mpFragSPV = loadViewportSPV(mpFragPath);
            if (!mpVertSPV.empty() && !mpFragSPV.empty()) {
                VkShaderModuleCreateInfo mpSmci{};
                mpSmci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

                mpSmci.codeSize = mpVertSPV.size() * sizeof(uint32_t);
                mpSmci.pCode = mpVertSPV.data();
                VkShaderModule mpVertModule = VK_NULL_HANDLE;
                vkCreateShaderModule(vkDevice, &mpSmci, nullptr, &mpVertModule);

                mpSmci.codeSize = mpFragSPV.size() * sizeof(uint32_t);
                mpSmci.pCode = mpFragSPV.data();
                VkShaderModule mpFragModule = VK_NULL_HANDLE;
                vkCreateShaderModule(vkDevice, &mpSmci, nullptr, &mpFragModule);

                if (mpVertModule && mpFragModule) {
                    // Binding 0: material SSBO
                    // Binding 1: texture sampler2D array — requires VK_EXT_descriptor_indexing
                    //   (Vulkan 1.2 core, no RT hardware needed).
                    //   No PARTIALLY_BOUND/UPDATE_AFTER_BIND flags: the shader only accesses
                    //   slots that were actually written (albedo_tex > 0 guard), so unwritten
                    //   slots are never reached. Updating the set between frames (not during
                    //   command-buffer recording) is always valid without those flags.
                    // Binding 2: baked env maps [0]=studio, [1]=outdoor (specular reflection lookup)
                    // Binding 3: terrain layer SSBO (same data as RT binding 12)
                    const bool hasDescIdx = m_device && m_device->getCapabilities().supportsDescriptorIndexing;

                    // Clamp the bindless texture array length to what this driver can
                    // actually handle in a single fragment-stage descriptor set. Some
                    // ICDs advertise descriptor indexing but cap non-UPDATE_AFTER_BIND
                    // sampled-image arrays at a value well below VULKAN_TEXTURE_CAPACITY
                    // (2048). Passing too-large a count slips past layout creation and
                    // then crashes inside vkCreateGraphicsPipelines with a null deref
                    // in nvoglv64.dll (fault_addr=0x8). Use the runtime limit instead.
                    uint32_t mpTextureArrayLen = 1u;
                    if (hasDescIdx && m_device && m_device->getPhysicalDevice() != VK_NULL_HANDLE) {
                        VkPhysicalDeviceProperties mpProps{};
                        vkGetPhysicalDeviceProperties(m_device->getPhysicalDevice(), &mpProps);
                        const uint32_t limit = mpProps.limits.maxPerStageDescriptorSampledImages;
                        const uint32_t want = static_cast<uint32_t>(VULKAN_TEXTURE_CAPACITY);
                        // Reserve 2 slots for binding 2 (env maps) which shares the same stage limit.
                        mpTextureArrayLen = (limit > 2u) ? (std::min)(want, limit - 2u) : 1u;
                       /* SCENE_LOG_INFO(std::string("[MP-init] maxPerStageDescriptorSampledImages=") +
                            std::to_string(limit) +
                            " → binding1 descriptorCount=" + std::to_string(mpTextureArrayLen));*/
                    }

                    VkDescriptorSetLayoutBinding mpDslBindings[4]{};
                    mpDslBindings[0].binding = 0;
                    mpDslBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    mpDslBindings[0].descriptorCount = 1;
                    mpDslBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
                    mpDslBindings[1].binding = 1;
                    mpDslBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    mpDslBindings[1].descriptorCount = hasDescIdx ? mpTextureArrayLen : 1u;
                    mpDslBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
                    mpDslBindings[2].binding = 2;
                    mpDslBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    mpDslBindings[2].descriptorCount = 2; // [0]=studio, [1]=outdoor
                    mpDslBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
                    mpDslBindings[3].binding = 3;
                    mpDslBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    mpDslBindings[3].descriptorCount = 1;
                    mpDslBindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

                    VkDescriptorSetLayoutCreateInfo mpDslci{};
                    mpDslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                    mpDslci.bindingCount = 4;
                    mpDslci.pBindings = mpDslBindings;

                    // Binding 1 is a sparse sampler2D array: only the slots whose texture IDs
                    // were actually uploaded get written. Without PARTIALLY_BOUND, unwritten
                    // slots are "undefined" per spec and stricter drivers (non-RT Intel/AMD
                    // ICDs observed crashing in vkCmdBindDescriptorSets/first draw) dereference
                    // them unconditionally. PARTIALLY_BOUND is core in Vulkan 1.2 and only
                    // requires VK_EXT_descriptor_indexing — already gated by `hasDescIdx`.
                    VkDescriptorBindingFlags mpBindingFlags[4] = {
                        0,
                        hasDescIdx ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : VkDescriptorBindingFlags{0},
                        0,
                        0
                    };
                    VkDescriptorSetLayoutBindingFlagsCreateInfo mpBindingFlagsCI{};
                    mpBindingFlagsCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
                    mpBindingFlagsCI.bindingCount = 4;
                    mpBindingFlagsCI.pBindingFlags = mpBindingFlags;
                    if (hasDescIdx) {
                        mpDslci.pNext = &mpBindingFlagsCI;
                    }
                    vkCreateDescriptorSetLayout(vkDevice, &mpDslci, nullptr,
                                                &m_interactiveViewport.materialPreviewDescLayout);

                    if (m_interactiveViewport.materialPreviewDescLayout != VK_NULL_HANDLE) {
                    VkDescriptorPoolSize mpPoolSizes[2]{};
                    mpPoolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    mpPoolSizes[0].descriptorCount = 2; // binding 0 (materials) + binding 3 (terrain)
                    mpPoolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    // +2 for the 2 env map slots at binding 2
                    mpPoolSizes[1].descriptorCount = (hasDescIdx ? mpTextureArrayLen : 1u) + 2u;
                    VkDescriptorPoolCreateInfo mpDpci{};
                    mpDpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                    mpDpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
                    mpDpci.poolSizeCount = 2;
                    mpDpci.pPoolSizes = mpPoolSizes;
                    mpDpci.maxSets = 1;
                    vkCreateDescriptorPool(vkDevice, &mpDpci, nullptr,
                                           &m_interactiveViewport.materialPreviewDescPool);

                    VkPushConstantRange mpPushRange{};
                    mpPushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
                    mpPushRange.offset = 0;
                    mpPushRange.size = sizeof(float) * 48 + sizeof(uint32_t) * 4;

                    VkPipelineLayoutCreateInfo mpPlci{};
                    mpPlci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                    mpPlci.pushConstantRangeCount = 1;
                    mpPlci.pPushConstantRanges = &mpPushRange;
                    mpPlci.setLayoutCount = 1;
                    mpPlci.pSetLayouts = &m_interactiveViewport.materialPreviewDescLayout;
                    // Only create layout if desc layout and pool succeeded —
                    // passing null desc layout to vkCreatePipelineLayout crashes old drivers.
                    if (m_interactiveViewport.materialPreviewDescPool != VK_NULL_HANDLE) {
                        vkCreatePipelineLayout(vkDevice, &mpPlci, nullptr,
                                               &m_interactiveViewport.materialPreviewPipelineLayout);
                    }
                    } // end if materialPreviewDescLayout valid

                    VkPipelineShaderStageCreateInfo mpStages[2]{};
                    mpStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    mpStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
                    mpStages[0].module = mpVertModule;
                    mpStages[0].pName = "main";
                    mpStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    mpStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
                    mpStages[1].module = mpFragModule;
                    mpStages[1].pName = "main";

                    VkVertexInputBindingDescription mpBindings[5]{};
                    mpBindings[0].binding = 0; mpBindings[0].stride = sizeof(float) * 3;  mpBindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
                    mpBindings[1].binding = 1; mpBindings[1].stride = sizeof(float) * 3;  mpBindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
                    mpBindings[2].binding = 2; mpBindings[2].stride = sizeof(uint32_t);   mpBindings[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
                    mpBindings[3].binding = 3; mpBindings[3].stride = sizeof(float) * 16; mpBindings[3].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
                    mpBindings[4].binding = 4; mpBindings[4].stride = sizeof(float) * 2;  mpBindings[4].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

                    VkVertexInputAttributeDescription mpAttribs[8]{};
                    mpAttribs[0].location = 0; mpAttribs[0].binding = 0; mpAttribs[0].format = VK_FORMAT_R32G32B32_SFLOAT;    mpAttribs[0].offset = 0;
                    mpAttribs[1].location = 1; mpAttribs[1].binding = 1; mpAttribs[1].format = VK_FORMAT_R32G32B32_SFLOAT;    mpAttribs[1].offset = 0;
                    mpAttribs[2].location = 2; mpAttribs[2].binding = 2; mpAttribs[2].format = VK_FORMAT_R32_UINT;            mpAttribs[2].offset = 0;
                    mpAttribs[3].location = 3; mpAttribs[3].binding = 3; mpAttribs[3].format = VK_FORMAT_R32G32B32A32_SFLOAT; mpAttribs[3].offset = sizeof(float) * 0;
                    mpAttribs[4].location = 4; mpAttribs[4].binding = 3; mpAttribs[4].format = VK_FORMAT_R32G32B32A32_SFLOAT; mpAttribs[4].offset = sizeof(float) * 4;
                    mpAttribs[5].location = 5; mpAttribs[5].binding = 3; mpAttribs[5].format = VK_FORMAT_R32G32B32A32_SFLOAT; mpAttribs[5].offset = sizeof(float) * 8;
                    mpAttribs[6].location = 6; mpAttribs[6].binding = 3; mpAttribs[6].format = VK_FORMAT_R32G32B32A32_SFLOAT; mpAttribs[6].offset = sizeof(float) * 12;
                    mpAttribs[7].location = 7; mpAttribs[7].binding = 4; mpAttribs[7].format = VK_FORMAT_R32G32_SFLOAT;       mpAttribs[7].offset = 0;

                    VkPipelineVertexInputStateCreateInfo mpVertInput{};
                    mpVertInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
                    mpVertInput.vertexBindingDescriptionCount = 5;
                    mpVertInput.pVertexBindingDescriptions = mpBindings;
                    mpVertInput.vertexAttributeDescriptionCount = 8;
                    mpVertInput.pVertexAttributeDescriptions = mpAttribs;

                    VkPipelineInputAssemblyStateCreateInfo mpIA{};
                    mpIA.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
                    mpIA.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

                    VkPipelineViewportStateCreateInfo mpVpState{};
                    mpVpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
                    mpVpState.viewportCount = 1;
                    mpVpState.scissorCount = 1;

                    VkPipelineRasterizationStateCreateInfo mpRast{};
                    mpRast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                    mpRast.polygonMode = VK_POLYGON_MODE_FILL;
                    mpRast.lineWidth = 1.0f;
                    mpRast.cullMode = VK_CULL_MODE_NONE;
                    mpRast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

                    VkPipelineMultisampleStateCreateInfo mpMS{};
                    mpMS.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
                    mpMS.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

                    VkPipelineDepthStencilStateCreateInfo mpDS{};
                    mpDS.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
                    mpDS.depthTestEnable = VK_TRUE;
                    mpDS.depthWriteEnable = VK_TRUE;
                    mpDS.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

                    VkPipelineColorBlendAttachmentState mpCBA{};
                    mpCBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
                    // Enable alpha blending for opacity texture and material alpha support.
                    // Cutout materials (opacity <= 0.02) are handled via discard in the shader;
                    // partial transparency (e.g. glass, foliage edges) uses standard src-over.
                    mpCBA.blendEnable         = VK_TRUE;
                    mpCBA.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                    mpCBA.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    mpCBA.colorBlendOp        = VK_BLEND_OP_ADD;
                    mpCBA.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                    mpCBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    mpCBA.alphaBlendOp        = VK_BLEND_OP_ADD;

                    VkPipelineColorBlendStateCreateInfo mpCB{};
                    mpCB.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
                    mpCB.attachmentCount = 1;
                    mpCB.pAttachments = &mpCBA;

                    VkDynamicState mpDynStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
                    VkPipelineDynamicStateCreateInfo mpDyn{};
                    mpDyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
                    mpDyn.dynamicStateCount = 2;
                    mpDyn.pDynamicStates = mpDynStates;

                    VkGraphicsPipelineCreateInfo mpPCI{};
                    mpPCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                    mpPCI.stageCount = 2;
                    mpPCI.pStages = mpStages;
                    mpPCI.pVertexInputState = &mpVertInput;
                    mpPCI.pInputAssemblyState = &mpIA;
                    mpPCI.pViewportState = &mpVpState;
                    mpPCI.pRasterizationState = &mpRast;
                    mpPCI.pMultisampleState = &mpMS;
                    mpPCI.pDepthStencilState = &mpDS;
                    mpPCI.pColorBlendState = &mpCB;
                    mpPCI.pDynamicState = &mpDyn;
                    mpPCI.layout = m_interactiveViewport.materialPreviewPipelineLayout;
                    mpPCI.renderPass = m_interactiveViewport.renderPass;
                    mpPCI.subpass = 0;

                    // Guard: if layout or pool creation failed (e.g. device limits on older GPU),
                    // skip pipeline creation entirely to avoid passing null layout to the driver.
                    VkResult mpResult = VK_ERROR_INITIALIZATION_FAILED;
                    if (m_interactiveViewport.materialPreviewPipelineLayout != VK_NULL_HANDLE &&
                        m_interactiveViewport.materialPreviewDescPool != VK_NULL_HANDLE) {
                       /* SCENE_LOG_INFO(std::string("[MP-init] step=preCreatePipeline layout=") +
                            std::to_string((uintptr_t)m_interactiveViewport.materialPreviewPipelineLayout) +
                            " renderPass=" + std::to_string((uintptr_t)m_interactiveViewport.renderPass) +
                            " descLayout=" + std::to_string((uintptr_t)m_interactiveViewport.materialPreviewDescLayout) +
                            " pool=" + std::to_string((uintptr_t)m_interactiveViewport.materialPreviewDescPool));*/
                        mpResult = safeCreateMaterialPreviewPipeline(vkDevice, &mpPCI,
                                                                     &m_interactiveViewport.materialPreviewPipeline);
                       /* SCENE_LOG_INFO(std::string("[MP-init] step=postCreatePipeline result=") +
                            std::to_string((int)mpResult) +
                            " pipeline=" + std::to_string((uintptr_t)m_interactiveViewport.materialPreviewPipeline));*/
                    } else {
                        SCENE_LOG_WARN("[VulkanViewportBackend] Material preview pipeline skipped: layout or pool null (device limit exceeded or extension unsupported).");
                    }
                    if (mpResult != VK_SUCCESS) {
                        SCENE_LOG_WARN("[VulkanViewportBackend] Material preview pipeline creation failed — falling back to solid/matcap permanently for this session (likely NVIDIA Maxwell driver bug).");
                        m_interactiveViewport.materialPreviewPipeline = VK_NULL_HANDLE;
                        m_materialPreviewPipelineGaveUp = true;
                        if (m_statusCallback && !m_materialPreviewUnsupportedNotified) {
                            m_materialPreviewUnsupportedNotified = true;
                            m_statusCallback("Material Preview unavailable on this GPU — falling back to Matcap mode.", 1);
                        }
                    } else {
                        VkDescriptorSetAllocateInfo mpDsai{};
                        mpDsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                        mpDsai.descriptorPool = m_interactiveViewport.materialPreviewDescPool;
                        mpDsai.descriptorSetCount = 1;
                        mpDsai.pSetLayouts = &m_interactiveViewport.materialPreviewDescLayout;
                        VkResult mpAllocRes = vkAllocateDescriptorSets(vkDevice, &mpDsai, &m_interactiveViewport.materialPreviewDescSet);
                       /* SCENE_LOG_INFO(std::string("[MP-init] step=allocDescSet result=") +
                            std::to_string((int)mpAllocRes) +
                            " descSet=" + std::to_string((uintptr_t)m_interactiveViewport.materialPreviewDescSet));*/

                        const VkBuffer materialBuffer = m_externalMaterialBuffer
                            ? m_externalMaterialBuffer
                            : m_device->m_materialBuffer.buffer;
                        const VkDeviceSize materialRange = m_externalMaterialBuffer
                            ? ((m_externalMaterialBufferSize > 0) ? m_externalMaterialBufferSize : VK_WHOLE_SIZE)
                            : VK_WHOLE_SIZE;
                        if (materialBuffer != VK_NULL_HANDLE && m_interactiveViewport.materialPreviewDescSet != VK_NULL_HANDLE) {
                            VkDescriptorBufferInfo matBufInfo{};
                            matBufInfo.buffer = materialBuffer;
                            matBufInfo.offset = 0;
                            matBufInfo.range = materialRange;

                            VkWriteDescriptorSet mpWds{};
                            mpWds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                            mpWds.dstSet = m_interactiveViewport.materialPreviewDescSet;
                            mpWds.dstBinding = 0;
                            mpWds.descriptorCount = 1;
                            mpWds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                            mpWds.pBufferInfo = &matBufInfo;
                            vkUpdateDescriptorSets(vkDevice, 1, &mpWds, 0, nullptr);

                            // Populate binding 1 with all textures already uploaded.
                            // Textures uploaded later are handled by updateMaterialPreviewTextureDescriptor.
                            for (auto& [texID, texImg] : m_uploadedImages) {
                                if (texID <= 0 || texID >= VULKAN_TEXTURE_CAPACITY) continue;
                                if (!texImg.view || !texImg.sampler) continue;
                                VkDescriptorImageInfo tii{};
                                tii.sampler     = texImg.sampler;
                                tii.imageView   = texImg.view;
                                tii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                                VkWriteDescriptorSet twds{};
                                twds.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                twds.dstSet          = m_interactiveViewport.materialPreviewDescSet;
                                twds.dstBinding      = 1;
                                twds.dstArrayElement = (uint32_t)texID;
                                twds.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                                twds.descriptorCount = 1;
                                twds.pImageInfo      = &tii;
                                vkUpdateDescriptorSets(vkDevice, 1, &twds, 0, nullptr);
                            }

                            // ── Binding 2: baked env maps for specular reflection ──
                            // Upload studio + outdoor 128×64 RGBA32F maps (one-time, ~256 KB total).
                            if (m_interactiveViewport.envMapStudioID == 0) {
                                auto studioPixels  = bakeEnvMap(false);
                                auto outdoorPixels = bakeEnvMap(true);
                                m_interactiveViewport.envMapStudioID  = this->uploadTexture2D(
                                    studioPixels.data(), 128, 64, 4, false, true);
                                m_interactiveViewport.envMapOutdoorID = this->uploadTexture2D(
                                    outdoorPixels.data(), 128, 64, 4, false, true);
                            }
                            // Write them into binding 2 array slots 0 and 1.
                            auto writeEnvSlot = [&](int64_t id, uint32_t slot) {
                                auto it = m_uploadedImages.find(id);
                                if (id <= 0 || it == m_uploadedImages.end()) return;
                                const auto& img = it->second;
                                if (!img.view || !img.sampler) return;
                                VkDescriptorImageInfo eii{};
                                eii.sampler     = img.sampler;
                                eii.imageView   = img.view;
                                eii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                                VkWriteDescriptorSet ewds{};
                                ewds.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                ewds.dstSet          = m_interactiveViewport.materialPreviewDescSet;
                                ewds.dstBinding      = 2;
                                ewds.dstArrayElement = slot;
                                ewds.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                                ewds.descriptorCount = 1;
                                ewds.pImageInfo      = &eii;
                                vkUpdateDescriptorSets(vkDevice, 1, &ewds, 0, nullptr);
                            };
                            writeEnvSlot(m_interactiveViewport.envMapStudioID,  0);
                            writeEnvSlot(m_interactiveViewport.envMapOutdoorID, 1);

                            // ── Binding 3: terrain layer SSBO ──
                            // Falls back to material buffer (a valid non-null SSBO) when no
                            // terrain data has been uploaded yet — the shader guards on FLAG_TERRAIN
                            // so it will never dereference stale data.
                            const VkBuffer terrainBuf = (m_device && m_device->m_terrainLayerBuffer.buffer)
                                ? m_device->m_terrainLayerBuffer.buffer
                                : (m_externalMaterialBuffer ? m_externalMaterialBuffer : m_device->m_materialBuffer.buffer);
                            if (terrainBuf != VK_NULL_HANDLE) {
                                VkDescriptorBufferInfo tLayerBufInfo{};
                                tLayerBufInfo.buffer = terrainBuf;
                                tLayerBufInfo.offset = 0;
                                tLayerBufInfo.range  = (m_device && m_device->m_terrainLayerBuffer.buffer)
                                    ? (m_device->m_terrainLayerBuffer.size > 0 ? m_device->m_terrainLayerBuffer.size : VK_WHOLE_SIZE)
                                    : VK_WHOLE_SIZE;
                                VkWriteDescriptorSet twds3{};
                                twds3.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                twds3.dstSet          = m_interactiveViewport.materialPreviewDescSet;
                                twds3.dstBinding      = 3;
                                twds3.descriptorCount = 1;
                                twds3.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                                twds3.pBufferInfo     = &tLayerBufInfo;
                                vkUpdateDescriptorSets(vkDevice, 1, &twds3, 0, nullptr);
                            }
                        }
                        SCENE_LOG_INFO("[VulkanViewportBackend] Material preview pipeline created successfully (texture + env map + terrain support enabled).");
                    }
                }
                if (mpVertModule) vkDestroyShaderModule(vkDevice, mpVertModule, nullptr);
                if (mpFragModule) vkDestroyShaderModule(vkDevice, mpFragModule, nullptr);
            }
        }
    }

  //  SCENE_LOG_INFO("[MP-init] step=enterHairPipeline");
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

   // SCENE_LOG_INFO("[MP-init] step=postAllPipelines");
    if (m_interactiveViewport.width == width &&
        m_interactiveViewport.height == height &&
        m_interactiveViewport.framebuffer != VK_NULL_HANDLE &&
        m_interactiveViewport.colorImage.image != VK_NULL_HANDLE &&
        m_interactiveViewport.depthImage.image != VK_NULL_HANDLE &&
        m_interactiveViewport.stagingBuffer.buffer != VK_NULL_HANDLE) {
        return true;
    }

    destroyInteractiveViewportResourcesImpl(true);
   // SCENE_LOG_INFO("[MP-init] step=createImages");

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
  //  SCENE_LOG_INFO("[MP-init] step=done");
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
        const double pixelCount = (std::max)(1.0, static_cast<double>(width) * static_cast<double>(height));
        const double referencePixels = 1920.0 * 1080.0;
        const double resolutionScale = std::sqrt(referencePixels / pixelCount);
        const double feedbackScale = allowAdaptiveRasterBudget
            ? (std::clamp)(static_cast<double>(m_rasterScatterBudgetScale), 0.35, 1.0)
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

    if (!m_interactiveViewport.identityInstanceBuffer.buffer) {
        // Buffer allocation failed (e.g. old driver / memory pressure) — skip raster frame
        return;
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
    const VkBuffer materialBuffer = m_externalMaterialBuffer
        ? m_externalMaterialBuffer
        : m_device->m_materialBuffer.buffer;
    const bool useMaterialPreview = (m_viewportMode == ViewportMode::MaterialPreview) &&
        !m_rasterInstances.empty() &&
        m_interactiveViewport.materialPreviewPipeline != VK_NULL_HANDLE &&
        m_interactiveViewport.materialPreviewDescSet != VK_NULL_HANDLE &&
        materialBuffer != VK_NULL_HANDLE;

    if (useMaterialPreview) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.materialPreviewPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_interactiveViewport.materialPreviewPipelineLayout,
                                0, 1, &m_interactiveViewport.materialPreviewDescSet, 0, nullptr);
    } else {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.solidPipeline);
        if (m_interactiveViewport.matcapDescSet != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.pipelineLayout, 0, 1, &m_interactiveViewport.matcapDescSet, 0, nullptr);
        }
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

            if (useMaterialPreview) {
                struct MaterialPreviewPushConstants {
                    float viewProj[16];
                    float view[16];
                    float cameraPos[4];
                    float lightDir0[4];
                    float lightDir1[4];
                    float lightDir2[4];
                    uint32_t materialMeta[4];
                };
                MaterialPreviewPushConstants mpPush{};
                matrixToGL(viewProj, mpPush.viewProj);
                matrixToGL(view, mpPush.view);
                mpPush.cameraPos[0] = m_camera.origin.x;
                mpPush.cameraPos[1] = m_camera.origin.y;
                mpPush.cameraPos[2] = m_camera.origin.z;
                mpPush.cameraPos[3] = 0.0f;
                mpPush.lightDir0[0] = 0.45f; mpPush.lightDir0[1] = 0.8f; mpPush.lightDir0[2] = 0.35f; mpPush.lightDir0[3] = 2.5f;
                mpPush.lightDir1[0] = -0.6f; mpPush.lightDir1[1] = 0.3f; mpPush.lightDir1[2] = 0.4f; mpPush.lightDir1[3] = 0.8f;
                mpPush.lightDir2[0] = -0.2f; mpPush.lightDir2[1] = 0.4f; mpPush.lightDir2[2] = -0.8f; mpPush.lightDir2[3] = 1.2f;
                uint32_t previewQuality = 2u;
                switch (::render_settings.raster_viewport_quality_preset) {
                    case ::RasterViewportQualityPreset::Performance: previewQuality = 1u; break;
                    case ::RasterViewportQualityPreset::Balanced: previewQuality = 2u; break;
                    case ::RasterViewportQualityPreset::Quality: previewQuality = 3u; break;
                    case ::RasterViewportQualityPreset::Auto:
                    default: previewQuality = 2u; break;
                }
                const uint32_t materialCount = m_externalMaterialBuffer
                    ? m_externalMaterialCount
                    : (m_device ? m_device->m_materialCount : 0u);
                mpPush.materialMeta[0] = materialCount;
                mpPush.materialMeta[1] = previewQuality;
                mpPush.materialMeta[2] = static_cast<uint32_t>(::render_settings.material_preview_lighting_preset);
                mpPush.materialMeta[3] = 0;
                vkCmdPushConstants(cmd, m_interactiveViewport.materialPreviewPipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(MaterialPreviewPushConstants), &mpPush);

                VkBuffer mpVertexBuffers[5] = {
                    rmb.vertexBuffer.buffer,
                    rmb.normalBuffer.buffer  ? rmb.normalBuffer.buffer  : rmb.vertexBuffer.buffer,
                    rmb.matIdBuffer.buffer ? rmb.matIdBuffer.buffer : rmb.vertexBuffer.buffer,
                    rmb.instanceBuffer.buffer,
                    rmb.uvBuffer.buffer ? rmb.uvBuffer.buffer : rmb.vertexBuffer.buffer,
                };
                VkDeviceSize mpOffsets[5] = { 0, 0, 0, 0, 0 };
                vkCmdBindVertexBuffers(cmd, 0, 5, mpVertexBuffers, mpOffsets);
            } else {
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
            }

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
            if (!m_interactiveViewport.identityInstanceBuffer.buffer) continue;

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
            uint32_t vend = (std::min)((uint32_t)positions.size(), vstart + (std::min)(seg2Count, (uint32_t)6) * 3u);
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

    if (m_interactiveViewport.gridVertexBuffer.buffer &&
        m_interactiveViewport.gridNormalBuffer.buffer &&
        m_interactiveViewport.identityInstanceBuffer.buffer &&
        m_interactiveViewport.gridVertexCount > 0) {
        if (useMaterialPreview) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.solidPipeline);
            if (m_interactiveViewport.matcapDescSet != VK_NULL_HANDLE) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        m_interactiveViewport.pipelineLayout,
                                        0, 1, &m_interactiveViewport.matcapDescSet, 0, nullptr);
            }
        }
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
            m_rasterScatterBudgetScale = (std::max)(0.35f, m_rasterScatterBudgetScale * 0.55f);
        } else if (frameMs > 60.0) {
            m_rasterScatterBudgetScale = (std::max)(0.35f, m_rasterScatterBudgetScale * 0.75f);
        } else if (frameMs > 40.0) {
            m_rasterScatterBudgetScale = (std::max)(0.35f, m_rasterScatterBudgetScale * 0.88f);
        } else if (frameMs < 20.0) {
            m_rasterScatterBudgetScale = (std::min)(1.0f, m_rasterScatterBudgetScale * 1.04f);
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

    auto findTargetKey = [&]() -> std::string {
        if (nodeName.empty()) return {};
        for (const auto& ri : m_rasterInstances) {
            if (ri.nodeName == nodeName) return ri.meshKey;
        }
        for (const auto& ri : m_rasterInstances) {
            if (viewportMatchesNodeNameForInstance(ri.nodeName, nodeName)) return ri.meshKey;
        }
        for (const auto& ri : m_rasterInstances) {
            if (viewportMatchesNodeNameForInstance(nodeName, ri.nodeName)) return ri.meshKey;
        }
        return {};
    };
    const std::string targetKey = findTargetKey();
    if (targetKey.empty()) return false;

    auto meshIt = m_rasterMeshes.find(targetKey);
    if (meshIt == m_rasterMeshes.end()) return false;

    auto& rmb = meshIt->second;
    const size_t vertCount = triangles.size() * 3;
    const size_t floatCount = vertCount * 3;
    const size_t uvFloatCount = vertCount * 2;

    auto ensurePreviewAttributeBuffers = [&]() {
        if (!rmb.uvBuffer.buffer) {
            VulkanRT::BufferCreateInfo uci{};
            uci.size = uvFloatCount * sizeof(float);
            uci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            uci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            uci.initialData = nullptr;
            rmb.uvBuffer = m_device->createBuffer(uci);
        }
        if (!rmb.matIdBuffer.buffer) {
            VulkanRT::BufferCreateInfo mci{};
            mci.size = vertCount * sizeof(uint32_t);
            mci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            mci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            mci.initialData = nullptr;
            rmb.matIdBuffer = m_device->createBuffer(mci);
        }
    };

    std::vector<float> newPositions(floatCount);
    std::vector<float> newNormals(floatCount);
    std::vector<float> newUVs(uvFloatCount);
    std::vector<uint32_t> newMatIds(vertCount);

    size_t idx = 0;
    size_t uvIdx = 0;
    size_t matIdx = 0;
    for (const auto& tri : triangles) {
        if (!tri) continue;
        const bool hasSharedTransform = (tri->getTransformHandle() != nullptr);
        auto [uv0, uv1, uv2] = tri->getUVCoordinates();
        const Vec2 triUvs[3] = { uv0, uv1, uv2 };
        const uint32_t triMatId = static_cast<uint32_t>(tri->getMaterialID());
        for (int v = 0; v < 3; ++v) {
            Vec3 p = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
            Vec3 n = hasSharedTransform ? tri->getOriginalVertexNormal(v) : tri->getOriginalVertexNormal(v);
            newPositions[idx]   = p.x; newPositions[idx + 1] = p.y; newPositions[idx + 2] = p.z;
            newNormals[idx]     = n.x; newNormals[idx + 1]   = n.y; newNormals[idx + 2]   = n.z;
            newUVs[uvIdx]       = triUvs[v].x; newUVs[uvIdx + 1] = triUvs[v].y;
            newMatIds[matIdx]   = triMatId;
            idx += 3;
            uvIdx += 2;
            ++matIdx;
        }
    }

    auto logMaterialPreviewMatIds = [&](const char* stage,
                                        const std::string& key,
                                        const std::vector<uint32_t>& matIds) {
        static std::unordered_set<std::string> loggedKeys;
        const std::string logKey = std::string(stage) + ":" + key;
        if (!loggedKeys.insert(logKey).second) {
            return;
        }
        uint32_t minMat = 0;
        uint32_t maxMat = 0;
        std::unordered_set<uint32_t> uniqueMatIds;
        std::string sampleList;
        if (!matIds.empty()) {
            minMat = matIds[0];
            maxMat = matIds[0];
            const size_t previewCount = std::min<size_t>(matIds.size(), 8);
            for (size_t i = 0; i < matIds.size(); ++i) {
                const uint32_t matId = matIds[i];
                minMat = (std::min)(minMat, matId);
                maxMat = (std::max)(maxMat, matId);
                uniqueMatIds.insert(matId);
                if (i < previewCount) {
                    if (!sampleList.empty()) sampleList += ",";
                    sampleList += std::to_string(matId);
                }
            }
        }
        SCENE_LOG_INFO("[MaterialPreview] matIds " + std::string(stage) +
                       " mesh='" + key +
                       "' verts=" + std::to_string(matIds.size()) +
                       " unique=" + std::to_string(uniqueMatIds.size()) +
                       " min=" + std::to_string(minMat) +
                       " max=" + std::to_string(maxMat) +
                       " sample=[" + sampleList + "]");
    };
    logMaterialPreviewMatIds("update", targetKey, newMatIds);

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

        VulkanRT::BufferCreateInfo uci{};
        uci.size = uvFloatCount * sizeof(float);
        uci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        uci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        uci.initialData = nullptr;
        rmb.uvBuffer = m_device->createBuffer(uci);

        VulkanRT::BufferCreateInfo mci{};
        mci.size = newMatIds.size() * sizeof(uint32_t);
        mci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        mci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        mci.initialData = nullptr;
        rmb.matIdBuffer = m_device->createBuffer(mci);

        if (rmb.vertexBuffer.buffer) {
            m_device->uploadBuffer(rmb.vertexBuffer, newPositions.data(), floatCount * sizeof(float), 0);
        }
        if (rmb.normalBuffer.buffer) {
            m_device->uploadBuffer(rmb.normalBuffer, newNormals.data(), floatCount * sizeof(float), 0);
        }
        if (rmb.uvBuffer.buffer) {
            m_device->uploadBuffer(rmb.uvBuffer, newUVs.data(), uvFloatCount * sizeof(float), 0);
        }
        if (rmb.matIdBuffer.buffer) {
            m_device->uploadBuffer(rmb.matIdBuffer, newMatIds.data(), newMatIds.size() * sizeof(uint32_t), 0);
        }

        rmb.cpuPositions = std::move(newPositions);
        rmb.cpuNormals = std::move(newNormals);
        rmb.cpuMatIds = std::move(newMatIds);
        if (!rmb.instanceIndices.empty()) {
            uploadRasterInstanceBuffer(rmb);
        }
    } else if (rmb.cpuPositions.size() == floatCount) {
        ensurePreviewAttributeBuffers();
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

        if (rmb.uvBuffer.buffer) {
            m_device->uploadBuffer(rmb.uvBuffer, newUVs.data(), uvFloatCount * sizeof(float), 0);
        }
        if (rmb.matIdBuffer.buffer) {
            m_device->uploadBuffer(rmb.matIdBuffer, newMatIds.data(), newMatIds.size() * sizeof(uint32_t), 0);
        }
        rmb.cpuMatIds = std::move(newMatIds);
    } else {
        ensurePreviewAttributeBuffers();
        m_device->uploadBuffer(rmb.vertexBuffer, newPositions.data(), floatCount * sizeof(float));
        m_device->uploadBuffer(rmb.normalBuffer, newNormals.data(), floatCount * sizeof(float));
        if (rmb.uvBuffer.buffer) {
            m_device->uploadBuffer(rmb.uvBuffer, newUVs.data(), uvFloatCount * sizeof(float), 0);
        }
        if (rmb.matIdBuffer.buffer) {
            m_device->uploadBuffer(rmb.matIdBuffer, newMatIds.data(), newMatIds.size() * sizeof(uint32_t), 0);
        }
        rmb.cpuPositions = std::move(newPositions);
        rmb.cpuNormals = std::move(newNormals);
        rmb.cpuMatIds = std::move(newMatIds);
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

    auto findTargetKey = [&]() -> std::string {
        if (nodeName.empty()) return {};
        for (const auto& ri : m_rasterInstances) {
            if (ri.nodeName == nodeName) return ri.meshKey;
        }
        for (const auto& ri : m_rasterInstances) {
            if (viewportMatchesNodeNameForInstance(ri.nodeName, nodeName)) return ri.meshKey;
        }
        for (const auto& ri : m_rasterInstances) {
            if (viewportMatchesNodeNameForInstance(nodeName, ri.nodeName)) return ri.meshKey;
        }
        return {};
    };
    const std::string targetKey = findTargetKey();
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

    std::vector<size_t> sortedDirty = dirtyIndices;
    std::sort(sortedDirty.begin(), sortedDirty.end());
    sortedDirty.erase(std::unique(sortedDirty.begin(), sortedDirty.end()), sortedDirty.end());

    size_t dirtyMinFloat = expectedFloatCount;
    size_t dirtyMaxFloat = 0;

    for (const size_t triIdx : sortedDirty) {
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

    std::vector<std::pair<size_t, size_t>> dirtyRuns;
    dirtyRuns.reserve((std::min)(sortedDirty.size(), size_t(64)));
    if (!sortedDirty.empty()) {
        size_t runStart = sortedDirty.front();
        size_t prev = sortedDirty.front();
        for (size_t i = 1; i < sortedDirty.size(); ++i) {
            const size_t triIdx = sortedDirty[i];
            if (triIdx == prev + 1) {
                prev = triIdx;
                continue;
            }
            dirtyRuns.emplace_back(runStart, prev);
            runStart = prev = triIdx;
        }
        dirtyRuns.emplace_back(runStart, prev);
    }

    const size_t dirtyTriCount = sortedDirty.size();
    const size_t spanTriCount =
        (dirtyMinFloat <= dirtyMaxFloat)
        ? ((dirtyMaxFloat - dirtyMinFloat + 1) / 9)
        : 0;
    const bool smallBrushLikeUpdate = dirtyTriCount > 0 && dirtyTriCount <= 96;
    const bool spanIsReasonable =
        dirtyTriCount > 0 &&
        spanTriCount > 0 &&
        spanTriCount <= dirtyTriCount * 6;
    const bool shouldUploadAsSingleSpan =
        dirtyRuns.empty() ||
        dirtyRuns.size() > 64 ||
        (smallBrushLikeUpdate && spanIsReasonable);

    if (shouldUploadAsSingleSpan && dirtyMinFloat <= dirtyMaxFloat) {
        dirtyMinFloat = (dirtyMinFloat / 3) * 3;
        dirtyMaxFloat = ((dirtyMaxFloat / 3) + 1) * 3;
        if (dirtyMaxFloat > expectedFloatCount) dirtyMaxFloat = expectedFloatCount;

        const uint64_t byteOffset = dirtyMinFloat * sizeof(float);
        const uint64_t byteSize = (dirtyMaxFloat - dirtyMinFloat) * sizeof(float);

        m_device->uploadBuffer(rmb.vertexBuffer, &rmb.cpuPositions[dirtyMinFloat], byteSize, byteOffset);
        m_device->uploadBuffer(rmb.normalBuffer, &rmb.cpuNormals[dirtyMinFloat], byteSize, byteOffset);
    } else {
        for (const auto& [runStart, runEnd] : dirtyRuns) {
            const size_t runStartFloat = runStart * 9;
            size_t runEndFloat = runEnd * 9 + 9;
            if (runStartFloat >= expectedFloatCount) {
                continue;
            }
            if (runEndFloat > expectedFloatCount) {
                runEndFloat = expectedFloatCount;
            }

            const uint64_t byteOffset = runStartFloat * sizeof(float);
            const uint64_t byteSize = (runEndFloat - runStartFloat) * sizeof(float);
            if (byteSize == 0) {
                continue;
            }

            m_device->uploadBuffer(rmb.vertexBuffer, &rmb.cpuPositions[runStartFloat], byteSize, byteOffset);
            m_device->uploadBuffer(rmb.normalBuffer, &rmb.cpuNormals[runStartFloat], byteSize, byteOffset);
        }
    }

    m_interactiveViewport.dirty = true;
    return true;
}

bool VulkanViewportBackend::cloneRasterObjectByNodeName(
    const std::string& sourceNodeName,
    const std::string& newNodeName,
    const Matrix4x4& transform) {
    if (!m_device || !m_device->isInitialized() || sourceNodeName.empty() || newNodeName.empty()) {
        return false;
    }
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    std::vector<uint32_t> sourceIndices;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_rasterInstances.size()); ++i) {
        const auto& ri = m_rasterInstances[i];
        if (ri.nodeName == sourceNodeName ||
            viewportMatchesNodeNameForInstance(ri.nodeName, sourceNodeName) ||
            viewportMatchesNodeNameForInstance(sourceNodeName, ri.nodeName)) {
            sourceIndices.push_back(i);
        }
    }
    if (sourceIndices.empty()) {
        return false;
    }

    std::unordered_set<std::string> dirtyMeshKeys;
    for (uint32_t sourceIndex : sourceIndices) {
        if (sourceIndex >= m_rasterInstances.size()) continue;

        RasterInstance clone = m_rasterInstances[sourceIndex];
        clone.nodeName = newNodeName;
        clone.transform = transform;
        updateRasterInstanceWorldBBox(clone);

        const uint32_t newIndex = static_cast<uint32_t>(m_rasterInstances.size());
        m_rasterInstances.push_back(std::move(clone));

        auto meshIt = m_rasterMeshes.find(m_rasterInstances.back().meshKey);
        if (meshIt != m_rasterMeshes.end()) {
            meshIt->second.instanceIndices.push_back(newIndex);
            dirtyMeshKeys.insert(meshIt->first);
        }
    }

    for (const auto& meshKey : dirtyMeshKeys) {
        auto meshIt = m_rasterMeshes.find(meshKey);
        if (meshIt != m_rasterMeshes.end()) {
            uploadRasterInstanceBuffer(meshIt->second);
        }
    }

    m_interactiveViewport.dirty = true;
    return !dirtyMeshKeys.empty();
}

void VulkanViewportBackend::buildRasterGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (!m_device || !m_device->isInitialized()) {
        SCENE_LOG_WARN("[ViewportRaster] buildRasterGeometry skipped: device not initialized. objects=" +
                       std::to_string(objects.size()));
        return;
    }

    // Skip rebuild if raster cache is still valid for the current scene generation.
    {
        extern std::atomic<uint64_t> g_scene_geometry_generation;
        const uint64_t curGen = g_scene_geometry_generation.load(std::memory_order_acquire);
        const uint64_t prevGen = m_rasterBuiltGeometryGeneration;
        if (!m_rasterMeshes.empty() && m_rasterBuiltGeometryGeneration == curGen) {
            SCENE_LOG_INFO("[ViewportRaster] buildRasterGeometry early-out: cache valid. gen=" +
                           std::to_string(curGen) + " meshes=" + std::to_string(m_rasterMeshes.size()) +
                           " objects=" + std::to_string(objects.size()));
            m_rasterGeometryDirty = false;
            return;
        }
        SCENE_LOG_INFO("[ViewportRaster] buildRasterGeometry starting: gen " +
                       std::to_string(prevGen) + " -> " + std::to_string(curGen) +
                       " objects=" + std::to_string(objects.size()) +
                       " prevMeshes=" + std::to_string(m_rasterMeshes.size()));
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

        // Filter nulls into a compact raw-pointer list so the parallel extraction can
        // index output slots directly (9 floats pos + 9 normals + 6 uvs + 3 matIds per
        // triangle). High-poly foliage sources (50-100K tri) used to burn a single core
        // here during raster prep.
        std::vector<const Triangle*> valid;
        valid.reserve(triangles.size());
        for (const auto& t : triangles) {
            if (t) valid.push_back(t.get());
        }
        if (valid.empty()) return;

        const size_t validCount = valid.size();
        std::vector<float> positions(validCount * 9);
        std::vector<float> normals(validCount * 9);
        std::vector<float> uvs(validCount * 6);
        std::vector<uint32_t> matIds(validCount * 3);

        struct LocalBBox { Vec3 bMin; Vec3 bMax; };

        auto extractRange = [&valid, &positions, &normals, &uvs, &matIds]
                            (size_t start, size_t end) -> LocalBBox {
            Vec3 bMin(1e18f, 1e18f, 1e18f), bMax(-1e18f, -1e18f, -1e18f);
            for (size_t i = start; i < end; ++i) {
                const Triangle* t = valid[i];
                auto [uv0, uv1, uv2] = t->getUVCoordinates();
                const Vec2 triUvs[3] = { uv0, uv1, uv2 };
                const uint32_t triMatId = static_cast<uint32_t>(t->getMaterialID());
                const size_t posBase = i * 9;
                const size_t uvBase  = i * 6;
                const size_t matBase = i * 3;
                for (int v = 0; v < 3; ++v) {
                    Vec3 p = t->getOriginalVertexPosition(v);
                    Vec3 n = t->getOriginalVertexNormal(v);
                    positions[posBase + v * 3 + 0] = p.x;
                    positions[posBase + v * 3 + 1] = p.y;
                    positions[posBase + v * 3 + 2] = p.z;
                    normals[posBase + v * 3 + 0] = n.x;
                    normals[posBase + v * 3 + 1] = n.y;
                    normals[posBase + v * 3 + 2] = n.z;
                    uvs[uvBase + v * 2 + 0] = triUvs[v].x;
                    uvs[uvBase + v * 2 + 1] = triUvs[v].y;
                    matIds[matBase + v] = triMatId;
                    bMin.x = (std::min)(bMin.x, p.x); bMin.y = (std::min)(bMin.y, p.y); bMin.z = (std::min)(bMin.z, p.z);
                    bMax.x = (std::max)(bMax.x, p.x); bMax.y = (std::max)(bMax.y, p.y); bMax.z = (std::max)(bMax.z, p.z);
                }
            }
            return { bMin, bMax };
        };

        Vec3 bMin(1e18f, 1e18f, 1e18f), bMax(-1e18f, -1e18f, -1e18f);
        constexpr size_t kExtractParallelThreshold = 4096;
        unsigned extract_threads = std::thread::hardware_concurrency();
        if (extract_threads == 0) extract_threads = 4;

        if (validCount < kExtractParallelThreshold || extract_threads < 2) {
            LocalBBox lbb = extractRange(0, validCount);
            bMin = lbb.bMin;
            bMax = lbb.bMax;
        } else {
            const size_t chunk = (validCount + extract_threads - 1) / extract_threads;
            std::vector<std::future<LocalBBox>> futures;
            futures.reserve(extract_threads);
            for (unsigned t = 0; t < extract_threads; ++t) {
                const size_t s = t * chunk;
                const size_t e = (std::min)(s + chunk, validCount);
                if (s >= e) break;
                futures.push_back(std::async(std::launch::async, extractRange, s, e));
            }
            for (auto& f : futures) {
                LocalBBox lbb = f.get();
                bMin.x = (std::min)(bMin.x, lbb.bMin.x); bMin.y = (std::min)(bMin.y, lbb.bMin.y); bMin.z = (std::min)(bMin.z, lbb.bMin.z);
                bMax.x = (std::max)(bMax.x, lbb.bMax.x); bMax.y = (std::max)(bMax.y, lbb.bMax.y); bMax.z = (std::max)(bMax.z, lbb.bMax.z);
            }
        }

        static std::unordered_set<std::string> loggedBuildKeys;
        if (loggedBuildKeys.insert(meshKey).second) {
            uint32_t minMat = 0;
            uint32_t maxMat = 0;
            std::unordered_set<uint32_t> uniqueMatIds;
            std::string sampleList;
            if (!matIds.empty()) {
                minMat = matIds[0];
                maxMat = matIds[0];
                const size_t previewCount = std::min<size_t>(matIds.size(), 8);
                for (size_t i = 0; i < matIds.size(); ++i) {
                    const uint32_t matId = matIds[i];
                    minMat = (std::min)(minMat, matId);
                    maxMat = (std::max)(maxMat, matId);
                    uniqueMatIds.insert(matId);
                    if (i < previewCount) {
                        if (!sampleList.empty()) sampleList += ",";
                        sampleList += std::to_string(matId);
                    }
                }
            }
            SCENE_LOG_INFO("[MaterialPreview] matIds build mesh='" + meshKey +
                           "' verts=" + std::to_string(matIds.size()) +
                           " unique=" + std::to_string(uniqueMatIds.size()) +
                           " min=" + std::to_string(minMat) +
                           " max=" + std::to_string(maxMat) +
                           " sample=[" + sampleList + "]");
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

        // UV buffer — required for material-preview texture sampling
        VulkanRT::BufferCreateInfo uci{};
        uci.size = uvs.size() * sizeof(float);
        uci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        uci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        uci.initialData = nullptr;
        rmb.uvBuffer = m_device->createBuffer(uci);

        // Material-ID buffer — per-vertex uint32
        VulkanRT::BufferCreateInfo mci{};
        mci.size = matIds.size() * sizeof(uint32_t);
        mci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        mci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        mci.initialData = nullptr;
        rmb.matIdBuffer = m_device->createBuffer(mci);

        if (rmb.vertexBuffer.buffer) {
            m_device->uploadBuffer(rmb.vertexBuffer, positions.data(), positions.size() * sizeof(float), 0);
        }
        if (rmb.normalBuffer.buffer) {
            m_device->uploadBuffer(rmb.normalBuffer, normals.data(), normals.size() * sizeof(float), 0);
        }
        if (rmb.uvBuffer.buffer && !uvs.empty()) {
            m_device->uploadBuffer(rmb.uvBuffer, uvs.data(), uvs.size() * sizeof(float), 0);
        }
        if (rmb.matIdBuffer.buffer && !matIds.empty()) {
            m_device->uploadBuffer(rmb.matIdBuffer, matIds.data(), matIds.size() * sizeof(uint32_t), 0);
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

        // Move vertex-attribute buffers into CPU shadow copies — updateRasterMeshFromTriangles
        // uses these to detect dirty ranges and re-upload matIds. Moving (instead of copying)
        // avoids a deep duplication of large per-source vectors for high-poly foliage.
        rmb.cpuPositions = std::move(positions);
        rmb.cpuNormals   = std::move(normals);
        rmb.cpuMatIds    = std::move(matIds);

        m_rasterMeshes[meshKey] = std::move(rmb);
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
        const float height = (std::max)(maxY - minY, 1e-4f);
        const float fallbackRadius = (std::max)({ maxX - minX, maxZ - minZ, 1e-3f }) * 0.5f;

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
                        plane.extents[slice] = (std::max)(plane.extents[slice], extent);
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
                    plane.extents[i] = (std::max)(plane.extents[i] * 0.96f, minExtent);
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
            proxyMin.x = (std::min)(proxyMin.x, positions[i + 0]);
            proxyMin.y = (std::min)(proxyMin.y, positions[i + 1]);
            proxyMin.z = (std::min)(proxyMin.z, positions[i + 2]);
            proxyMax.x = (std::max)(proxyMax.x, positions[i + 0]);
            proxyMax.y = (std::max)(proxyMax.y, positions[i + 1]);
            proxyMax.z = (std::max)(proxyMax.z, positions[i + 2]);
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
        std::vector<float> uvs;
        std::vector<uint32_t> matIds;
        std::vector<std::shared_ptr<Triangle>> triangles;
        Matrix4x4 transform;
        uint8_t mask = 0xFF;
    };

    std::vector<RasterTriGroup> groups;
    std::unordered_map<std::string, size_t> groupByKey;

    std::function<void(const std::shared_ptr<Hittable>&)> processObj;
    processObj = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj) return;

        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (!inst->visible || !inst->source_triangles || inst->source_triangles->empty()) return;
            if (hasInstancePrefix(inst->node_name)) return;

            const auto srcPtr = reinterpret_cast<uintptr_t>(inst->source_triangles.get());
            const std::string instanceNodeName = inst->node_name.empty()
                ? ("[ViewportInst-" + std::to_string(m_rasterInstances.size()) + "]")
                : inst->node_name;
            std::unordered_map<std::string, std::vector<std::shared_ptr<Triangle>>> trianglesByNode;
            trianglesByNode.reserve(inst->source_triangles->size());
            for (const auto& tri : *inst->source_triangles) {
                if (!tri) continue;
                const std::string triNodeName = tri->getNodeName().empty() ? instanceNodeName : tri->getNodeName();
                trianglesByNode[triNodeName].push_back(tri);
            }

            for (const auto& [triNodeName, groupedTriangles] : trianglesByNode) {
                if (groupedTriangles.empty()) continue;
                // Keep viewport raster meshes instance-local and node-local so material preview
                // matches the same object grouping used by UI selection/material slots.
                std::string meshKey = "[Viewport-Raster]-" + triNodeName +
                                      "-src-" + std::to_string(srcPtr) +
                                      "-tris-" + std::to_string(groupedTriangles.size());

                ensureRasterMeshForTriangles(meshKey, groupedTriangles);

                RasterInstance ri;
                ri.meshKey = meshKey;
                ri.nodeName = triNodeName;
                ri.transform = inst->transform;
                ri.mask = 0xFF;
                m_rasterInstances.push_back(ri);
            }

        } else if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
            for (auto& child : list->objects) processObj(child);
        } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            processObj(bvh->left);
            processObj(bvh->right);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (!tri->visible) return;

            auto triTransformHandle = tri->getTransformHandle();
            const bool hasSharedTransform = (triTransformHandle != nullptr);
            std::string nodeName = tri->getNodeName();
            if (nodeName.empty()) nodeName = "[Solo-" + std::to_string(groups.size()) + "]";
            const uintptr_t transformKey = triTransformHandle
                ? reinterpret_cast<uintptr_t>(triTransformHandle.get())
                : reinterpret_cast<uintptr_t>(tri.get());
            const std::string groupKey = nodeName + "#th=" + std::to_string(transformKey);

            auto found = groupByKey.find(groupKey);
            if (found == groupByKey.end()) {
                RasterTriGroup g;
                g.meshKey = "[Raster-Solo]-" + nodeName;
                g.nodeName = nodeName;
                g.transform = hasSharedTransform ? tri->getTransformMatrix() : Matrix4x4::identity();
                groups.push_back(std::move(g));
                found = groupByKey.emplace(groupKey, groups.size() - 1).first;
            }

            auto& grp = groups[found->second];
            grp.triangles.push_back(tri);
            auto [uv0, uv1, uv2] = tri->getUVCoordinates();
            const Vec2 triUvs[3] = { uv0, uv1, uv2 };
            const uint32_t triMatId = static_cast<uint32_t>(tri->getMaterialID());
            for (int v = 0; v < 3; ++v) {
                Vec3 p = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
                Vec3 n = hasSharedTransform ? tri->getOriginalVertexNormal(v) : tri->getOriginalVertexNormal(v);
                grp.positions.push_back(p.x); grp.positions.push_back(p.y); grp.positions.push_back(p.z);
                grp.normals.push_back(n.x); grp.normals.push_back(n.y); grp.normals.push_back(n.z);
                grp.uvs.push_back(triUvs[v].x); grp.uvs.push_back(triUvs[v].y);
                grp.matIds.push_back(triMatId);
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
            bMin.x = (std::min)(bMin.x, px); bMin.y = (std::min)(bMin.y, py); bMin.z = (std::min)(bMin.z, pz);
            bMax.x = (std::max)(bMax.x, px); bMax.y = (std::max)(bMax.y, py); bMax.z = (std::max)(bMax.z, pz);
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

        if (!grp.uvs.empty()) {
            VulkanRT::BufferCreateInfo uci{};
            uci.size = grp.uvs.size() * sizeof(float);
            uci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            uci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            uci.initialData = nullptr;
            rmb.uvBuffer = m_device->createBuffer(uci);
        }

        if (!grp.matIds.empty()) {
            VulkanRT::BufferCreateInfo mci{};
            mci.size = grp.matIds.size() * sizeof(uint32_t);
            mci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            mci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            mci.initialData = nullptr;
            rmb.matIdBuffer = m_device->createBuffer(mci);
        }

        if (rmb.vertexBuffer.buffer) {
            m_device->uploadBuffer(rmb.vertexBuffer, grp.positions.data(), grp.positions.size() * sizeof(float), 0);
        }
        if (rmb.normalBuffer.buffer) {
            m_device->uploadBuffer(rmb.normalBuffer, grp.normals.data(), grp.normals.size() * sizeof(float), 0);
        }
        if (rmb.uvBuffer.buffer && !grp.uvs.empty()) {
            m_device->uploadBuffer(rmb.uvBuffer, grp.uvs.data(), grp.uvs.size() * sizeof(float), 0);
        }
        if (rmb.matIdBuffer.buffer && !grp.matIds.empty()) {
            m_device->uploadBuffer(rmb.matIdBuffer, grp.matIds.data(), grp.matIds.size() * sizeof(uint32_t), 0);
        }

        rmb.cpuPositions = grp.positions;
        rmb.cpuNormals   = grp.normals;
        rmb.cpuMatIds    = grp.matIds;

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

    // ============================================================================
    // Scatter-instance expansion — parallelized per-group inner loop.
    //
    // Serial pre-pass: per (group, srcIdx) we resolve the meshKey once and call
    // ensureRasterMeshForTriangles / ensureScatterProxyMesh (these mutate
    // m_rasterMeshes / m_rasterMeshBBoxes so they must stay single-threaded).
    //
    // Parallel pass: for each group fan out matrix composition + per-instance
    // RasterInstance construction across threads using std::async chunks.
    // Large foliage groups burned a single core here with inst.toMatrix() +
    // nodeName concatenation + vector reallocation.
    // ============================================================================
    struct GroupSrcMeta {
        std::vector<std::string> meshKeyBySrc;  // indexed by srcIdx; empty = invalid source
    };
    std::vector<GroupSrcMeta> groupMeta(instanceGroups.size());

    size_t totalValidScatterInstances = 0;
    for (size_t gi = 0; gi < instanceGroups.size(); ++gi) {
        const auto& group = instanceGroups[gi];
        if (group.instances.empty() || group.sources.empty()) continue;
        auto& meta = groupMeta[gi];
        meta.meshKeyBySrc.resize(group.sources.size());

        for (size_t si = 0; si < group.sources.size(); ++si) {
            const auto& source = group.sources[si];
            const auto* triSource = source.centered_triangles_ptr ? source.centered_triangles_ptr.get() : nullptr;
            if ((!triSource || triSource->empty()) && source.triangles.empty()) continue;

            std::string meshKey;
            if (triSource) {
                const auto srcPtr = reinterpret_cast<uintptr_t>(triSource);
                meshKey = "[Raster-Group]-" + std::to_string(group.id) + "-" + std::to_string(si) +
                          "-" + std::to_string(srcPtr) + "-" + std::to_string(triSource->size());
                ensureRasterMeshForTriangles(meshKey, *triSource);
            } else {
                const auto srcPtr = reinterpret_cast<uintptr_t>(&source.triangles);
                meshKey = "[Raster-Group]-" + std::to_string(group.id) + "-" + std::to_string(si) +
                          "-" + std::to_string(srcPtr) + "-" + std::to_string(source.triangles.size());
                ensureRasterMeshForTriangles(meshKey, source.triangles);
            }

            auto rasterMeshIt = m_rasterMeshes.find(meshKey);
            if (rasterMeshIt != m_rasterMeshes.end()) {
                rasterMeshIt->second.isScatterGroup = true;
                ensureScatterProxyMesh(meshKey, triSource ? triSource : &source.triangles);
            }
            meta.meshKeyBySrc[si] = std::move(meshKey);
        }

        for (const auto& inst : group.instances) {
            int srcIdx = inst.source_index;
            if (srcIdx < 0 || srcIdx >= static_cast<int>(group.sources.size())) srcIdx = 0;
            if (srcIdx < static_cast<int>(meta.meshKeyBySrc.size()) &&
                !meta.meshKeyBySrc[srcIdx].empty()) {
                ++totalValidScatterInstances;
            }
        }
    }

    m_rasterInstances.reserve(m_rasterInstances.size() + totalValidScatterInstances);

    unsigned num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    const size_t kParallelThreshold = 1024;

    for (size_t gi = 0; gi < instanceGroups.size(); ++gi) {
        const auto& group = instanceGroups[gi];
        if (group.instances.empty() || group.sources.empty()) continue;
        const auto& meshKeyBySrc = groupMeta[gi].meshKeyBySrc;
        if (meshKeyBySrc.empty()) continue;

        const size_t count = group.instances.size();
        std::vector<RasterInstance> localInstances(count);

        auto fillRange = [&group, &meshKeyBySrc, &localInstances](size_t start, size_t end) {
            const std::string nodePrefix = "_inst_gid" + std::to_string(group.id) + "_";
            for (size_t i = start; i < end; ++i) {
                const auto& inst = group.instances[i];
                int srcIdx = inst.source_index;
                if (srcIdx < 0 || srcIdx >= static_cast<int>(group.sources.size())) srcIdx = 0;
                if (srcIdx >= static_cast<int>(meshKeyBySrc.size()) ||
                    meshKeyBySrc[srcIdx].empty()) {
                    continue;  // invalid src — leave default-constructed (meshKey stays empty)
                }
                auto& ri = localInstances[i];
                ri.meshKey = meshKeyBySrc[srcIdx];
                ri.nodeName = nodePrefix + std::to_string(i);
                ri.transform = inst.toMatrix();
                ri.mask = 0xFF;
            }
        };

        if (count < kParallelThreshold || num_threads < 2) {
            fillRange(0, count);
        } else {
            const size_t chunk = (count + num_threads - 1) / num_threads;
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads);
            for (unsigned t = 0; t < num_threads; ++t) {
                const size_t s = t * chunk;
                const size_t e = (std::min)(s + chunk, count);
                if (s >= e) break;
                futures.push_back(std::async(std::launch::async, fillRange, s, e));
            }
            for (auto& f : futures) f.get();
        }

        // Serial compact — preserves instance order so nodeName's encoded index
        // still matches its position conceptually. Skip invalid slots.
        for (auto& ri : localInstances) {
            if (ri.meshKey.empty()) continue;
            m_rasterInstances.push_back(std::move(ri));
        }
    }

    // Assign localBBox from cached mesh AABB and compute worldBBox per instance.
    // Parallelized — m_rasterMeshBBoxes is read-only from here on; per-instance
    // writes touch disjoint RasterInstance objects.
    {
        const size_t total = m_rasterInstances.size();
        auto bboxRange = [this](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                auto& ri = m_rasterInstances[i];
                auto bboxIt = m_rasterMeshBBoxes.find(ri.meshKey);
                if (bboxIt != m_rasterMeshBBoxes.end()) {
                    ri.localBBox = bboxIt->second;
                    updateRasterInstanceWorldBBox(ri);
                }
            }
        };

        if (total < kParallelThreshold || num_threads < 2) {
            bboxRange(0, total);
        } else {
            const size_t chunk = (total + num_threads - 1) / num_threads;
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads);
            for (unsigned t = 0; t < num_threads; ++t) {
                const size_t s = t * chunk;
                const size_t e = (std::min)(s + chunk, total);
                if (s >= e) break;
                futures.push_back(std::async(std::launch::async, bboxRange, s, e));
            }
            for (auto& f : futures) f.get();
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
        SCENE_LOG_INFO("[ViewportRaster] buildRasterGeometry done: gen=" +
                       std::to_string(m_rasterBuiltGeometryGeneration) +
                       " meshes=" + std::to_string(m_rasterMeshes.size()) +
                       " instances=" + std::to_string(m_rasterInstances.size()));
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
    if (m_viewportMode != ViewportMode::Solid &&
        m_viewportMode != ViewportMode::Matcap &&
        m_viewportMode != ViewportMode::MaterialPreview) return;

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
