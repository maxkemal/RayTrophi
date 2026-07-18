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
#include <sstream>
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

static uint64_t estimateViewportTextureBytes(const Texture* tex, uint32_t uploadChannels) {
    if (!tex || tex->width <= 0 || tex->height <= 0) return 0;
    const uint64_t pixelCount = static_cast<uint64_t>(tex->width) * static_cast<uint64_t>(tex->height);
    if (tex->is_hdr) {
        return pixelCount * (uploadChannels <= 1 ? 4ull : 16ull);
    }
    return pixelCount * static_cast<uint64_t>((std::max)(1u, uploadChannels));
}

static std::string buildViewportSceneTextureKey(const Texture* tex,
                                                TextureType textureType,
                                                bool forceLinear,
                                                bool preferSingleChannel) {
    if (!tex) return {};

    std::ostringstream oss;
    if (!tex->name.empty()) {
        oss << "tex:" << tex->name;
    } else {
        oss << "tex_ptr:" << reinterpret_cast<uintptr_t>(tex);
    }
    oss << "|type=" << static_cast<uint32_t>(textureType)
        << "|forceLinear=" << (forceLinear ? 1 : 0)
        << "|preferSingle=" << (preferSingleChannel ? 1 : 0)
        << "|hdr=" << (tex->is_hdr ? 1 : 0);
    return oss.str();
}

static TextureHandle registerViewportTerrainSplatTexture(SceneTextureManager* manager,
                                                         const Texture* tex) {
    if (!manager || !tex || !tex->is_loaded()) {
        return {};
    }

    return manager->registerTextureKey(
        buildViewportSceneTextureKey(tex, TextureType::Unknown, true, false),
        TextureConsumer::RasterPreview,
        static_cast<uint32_t>((std::max)(0, tex->width)),
        static_cast<uint32_t>((std::max)(0, tex->height)),
        estimateViewportTextureBytes(tex, 4u));
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
                TextureHandle sceneHandle{};
                if (m_sceneTextureManager) {
                    sceneHandle = registerViewportTerrainSplatTexture(
                        m_sceneTextureManager.get(),
                        splatTex);
                }
                if (splatTex->vulkan_dirty) {
                    if (sceneHandle.isValid() && m_sceneTextureManager) {
                        int64_t pooledId = 0;
                        if (m_sceneTextureManager->tryGetVulkanTextureId(
                                sceneHandle, sceneTextureOwnerScope(), pooledId) &&
                            pooledId != 0) {
                            m_sceneTextureManager->clearVulkanBacking(sceneTextureOwnerScope(), pooledId);
                            this->destroyTexture(pooledId);
                        }
                    }
                    auto oldIt = m_uploadedImageIDs.find(cacheKey);
                    if (oldIt != m_uploadedImageIDs.end()) {
                        int64_t oldId = oldIt->second;
                        if (oldId) this->destroyTexture(oldId);
                    }
                    splatTex->clearVulkanDirty();
                }
                if (sceneHandle.isValid() && m_sceneTextureManager) {
                    int64_t existingSceneTextureId = 0;
                    VulkanRT::ImageHandle existingSceneImage{};
                    if (m_sceneTextureManager->tryGetVulkanTextureId(sceneHandle, sceneTextureOwnerScope(), existingSceneTextureId) &&
                        existingSceneTextureId != 0 &&
                        tryGetUploadedImageHandle(existingSceneTextureId, existingSceneImage)) {
                        m_uploadedImageIDs[cacheKey] = existingSceneTextureId;
                        m_textureIdToCacheKey[existingSceneTextureId] = cacheKey;
                    }
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
                        if (id > 0) {
                            m_uploadedImageIDs[cacheKey] = id;
                            m_textureIdToCacheKey[id] = cacheKey;
                            registerSceneTextureUpload(sceneHandle, id);
                            gld.splat_map_tex = static_cast<uint32_t>(id);
                        }
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
        (size > 0) ? (size / sizeof(VulkanRT::VkGpuMaterialCore)) : 0);
    m_interactiveViewport.dirty = true;
    if (!m_device) return;

    VkDevice vkDevice = m_device->getDevice();
    if (vkDevice == VK_NULL_HANDLE) return;

    if (m_interactiveViewport.materialPreviewDescSet != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo matBufInfo{};
        matBufInfo.buffer = m_externalMaterialBuffer;
        matBufInfo.offset = 0;
        matBufInfo.range = (m_externalMaterialBufferSize > 0) ? m_externalMaterialBufferSize : VK_WHOLE_SIZE;

        // binding 4 (cold MaterialExt): external callers hand over only the core
        // buffer — fall back to it so the binding is never left unwritten.
        VkDescriptorBufferInfo matExtBufInfo{};
        matExtBufInfo.buffer = m_device->m_materialExtBuffer.buffer
                             ? m_device->m_materialExtBuffer.buffer
                             : m_externalMaterialBuffer;
        matExtBufInfo.offset = 0;
        matExtBufInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet mpWds[2]{};
        mpWds[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        mpWds[0].dstSet = m_interactiveViewport.materialPreviewDescSet;
        mpWds[0].dstBinding = 0;
        mpWds[0].dstArrayElement = 0;
        mpWds[0].descriptorCount = 1;
        mpWds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        mpWds[0].pBufferInfo = &matBufInfo;
        mpWds[1] = mpWds[0];
        mpWds[1].dstBinding = 4;
        mpWds[1].pBufferInfo = &matExtBufInfo;

        vkUpdateDescriptorSets(vkDevice, 2, mpWds, 0, nullptr);
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
    if (m_interactiveViewport.particleAddPipeline != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipeline(vkDevice, m_interactiveViewport.particleAddPipeline, nullptr);
        m_interactiveViewport.particleAddPipeline = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.particleAlphaPipeline != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipeline(vkDevice, m_interactiveViewport.particleAlphaPipeline, nullptr);
        m_interactiveViewport.particleAlphaPipeline = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.editLinePipeline != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipeline(vkDevice, m_interactiveViewport.editLinePipeline, nullptr);
        m_interactiveViewport.editLinePipeline = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.editFacePipeline != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipeline(vkDevice, m_interactiveViewport.editFacePipeline, nullptr);
        m_interactiveViewport.editFacePipeline = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.editPointPipeline != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipeline(vkDevice, m_interactiveViewport.editPointPipeline, nullptr);
        m_interactiveViewport.editPointPipeline = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.editOverlayPipelineLayout != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipelineLayout(vkDevice, m_interactiveViewport.editOverlayPipelineLayout, nullptr);
        m_interactiveViewport.editOverlayPipelineLayout = VK_NULL_HANDLE;
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
    if (m_interactiveViewport.particleAddVertexBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.particleAddVertexBuffer);
        m_interactiveViewport.particleAddVertexCount = 0;
    }
    if (m_interactiveViewport.particleAlphaVertexBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.particleAlphaVertexBuffer);
        m_interactiveViewport.particleAlphaVertexCount = 0;
    }
    // Edit overlay buffers are framebuffer-size independent: keep them alive
    // across viewport resizes (keepPipeline=true), otherwise the UI-side sync
    // state still believes they are uploaded and the overlay silently
    // disappears until the next topology/selection change.
    if (!keepPipeline) {
        if (m_interactiveViewport.editPositionBuffer.buffer) {
            m_device->destroyBuffer(m_interactiveViewport.editPositionBuffer);
            m_interactiveViewport.editVertexCount = 0;
        }
        if (m_interactiveViewport.editFlagBuffer.buffer) {
            m_device->destroyBuffer(m_interactiveViewport.editFlagBuffer);
        }
        if (m_interactiveViewport.editEdgeIndexBuffer.buffer) {
            m_device->destroyBuffer(m_interactiveViewport.editEdgeIndexBuffer);
            m_interactiveViewport.editEdgeIndexCount = 0;
        }
        if (m_interactiveViewport.editFaceIndexBuffer.buffer) {
            m_device->destroyBuffer(m_interactiveViewport.editFaceIndexBuffer);
            m_interactiveViewport.editFaceIndexCount = 0;
        }
        if (m_interactiveViewport.editSelEdgeIndexBuffer.buffer) {
            m_device->destroyBuffer(m_interactiveViewport.editSelEdgeIndexBuffer);
            m_interactiveViewport.editSelEdgeIndexCount = 0;
        }
        if (m_interactiveViewport.editSelFaceIndexBuffer.buffer) {
            m_device->destroyBuffer(m_interactiveViewport.editSelFaceIndexBuffer);
            m_interactiveViewport.editSelFaceIndexCount = 0;
        }
        m_interactiveViewport.editOverlayParams = Backend::EditMeshOverlayParams{};
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
    // Selection outline: mask image + framebuffers track the viewport size
    // (destroyed on resize); render passes/pipelines/descriptors follow the
    // other pipelines' keepPipeline lifecycle.
    if (m_interactiveViewport.selectionMaskFramebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(vkDevice, m_interactiveViewport.selectionMaskFramebuffer, nullptr);
        m_interactiveViewport.selectionMaskFramebuffer = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.selectionCompositeFramebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(vkDevice, m_interactiveViewport.selectionCompositeFramebuffer, nullptr);
        m_interactiveViewport.selectionCompositeFramebuffer = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.selectionMaskImage.image) {
        m_device->destroyImage(m_interactiveViewport.selectionMaskImage);
        m_interactiveViewport.selectionMaskImage = {};
    }
    if (!keepPipeline) {
        if (m_interactiveViewport.selectionMaskFullPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(vkDevice, m_interactiveViewport.selectionMaskFullPipeline, nullptr);
            m_interactiveViewport.selectionMaskFullPipeline = VK_NULL_HANDLE;
        }
        if (m_interactiveViewport.selectionMaskVisiblePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(vkDevice, m_interactiveViewport.selectionMaskVisiblePipeline, nullptr);
            m_interactiveViewport.selectionMaskVisiblePipeline = VK_NULL_HANDLE;
        }
        if (m_interactiveViewport.selectionCompositePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(vkDevice, m_interactiveViewport.selectionCompositePipeline, nullptr);
            m_interactiveViewport.selectionCompositePipeline = VK_NULL_HANDLE;
        }
        if (m_interactiveViewport.selectionMaskPipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(vkDevice, m_interactiveViewport.selectionMaskPipelineLayout, nullptr);
            m_interactiveViewport.selectionMaskPipelineLayout = VK_NULL_HANDLE;
        }
        if (m_interactiveViewport.selectionCompositePipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(vkDevice, m_interactiveViewport.selectionCompositePipelineLayout, nullptr);
            m_interactiveViewport.selectionCompositePipelineLayout = VK_NULL_HANDLE;
        }
        if (m_interactiveViewport.selectionMaskRenderPass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(vkDevice, m_interactiveViewport.selectionMaskRenderPass, nullptr);
            m_interactiveViewport.selectionMaskRenderPass = VK_NULL_HANDLE;
        }
        if (m_interactiveViewport.selectionCompositeRenderPass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(vkDevice, m_interactiveViewport.selectionCompositeRenderPass, nullptr);
            m_interactiveViewport.selectionCompositeRenderPass = VK_NULL_HANDLE;
        }
        if (m_interactiveViewport.selectionCompositeDescPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(vkDevice, m_interactiveViewport.selectionCompositeDescPool, nullptr);
            m_interactiveViewport.selectionCompositeDescPool = VK_NULL_HANDLE;
            m_interactiveViewport.selectionCompositeDescSet = VK_NULL_HANDLE;
        }
        if (m_interactiveViewport.selectionCompositeDescLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(vkDevice, m_interactiveViewport.selectionCompositeDescLayout, nullptr);
            m_interactiveViewport.selectionCompositeDescLayout = VK_NULL_HANDLE;
        }
        if (m_interactiveViewport.selectionMaskSampler != VK_NULL_HANDLE) {
            vkDestroySampler(vkDevice, m_interactiveViewport.selectionMaskSampler, nullptr);
            m_interactiveViewport.selectionMaskSampler = VK_NULL_HANDLE;
        }
        if (m_interactiveViewport.selectionInstanceBuffer.buffer) {
            m_device->destroyBuffer(m_interactiveViewport.selectionInstanceBuffer);
            m_interactiveViewport.selectionInstanceBuffer = {};
        }
        m_interactiveViewport.selectionOutlineParams = Backend::SelectionOutlineParams{};
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
        // STORE (not DONT_CARE): the selection-outline mask pass re-loads this
        // depth right after the main pass to depth-test the visible channel.
        // Store/load ops don't affect render-pass compatibility, so existing
        // pipelines are unaffected.
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
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
        // Alpha blend so the grid distance fade can dissolve lines; mesh draws
        // output alpha 1.0 so their result is unchanged.
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

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
                    m_interactiveViewport.materialPreviewTextureArrayLen = mpTextureArrayLen;

                    VkDescriptorSetLayoutBinding mpDslBindings[5]{};
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
                    // binding 4: cold MaterialExt half of the split material record —
                    // material_preview_frag.spv statically uses it (micro-detail/sheen/SSS).
                    mpDslBindings[4].binding = 4;
                    mpDslBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    mpDslBindings[4].descriptorCount = 1;
                    mpDslBindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

                    VkDescriptorSetLayoutCreateInfo mpDslci{};
                    mpDslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                    mpDslci.bindingCount = 5;
                    mpDslci.pBindings = mpDslBindings;

                    // Binding 1 is a sparse sampler2D array: only the slots whose texture IDs
                    // were actually uploaded get written. Without PARTIALLY_BOUND, unwritten
                    // slots are "undefined" per spec and stricter drivers (non-RT Intel/AMD
                    // ICDs observed crashing in vkCmdBindDescriptorSets/first draw) dereference
                    // them unconditionally. PARTIALLY_BOUND is core in Vulkan 1.2 and only
                    // requires VK_EXT_descriptor_indexing — already gated by `hasDescIdx`.
                    VkDescriptorBindingFlags mpBindingFlags[5] = {
                        0,
                        hasDescIdx ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : VkDescriptorBindingFlags{0},
                        0,
                        0,
                        0
                    };
                    VkDescriptorSetLayoutBindingFlagsCreateInfo mpBindingFlagsCI{};
                    mpBindingFlagsCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
                    mpBindingFlagsCI.bindingCount = 5;
                    mpBindingFlagsCI.pBindingFlags = mpBindingFlags;
                    if (hasDescIdx) {
                        mpDslci.pNext = &mpBindingFlagsCI;
                    }
                    vkCreateDescriptorSetLayout(vkDevice, &mpDslci, nullptr,
                                                &m_interactiveViewport.materialPreviewDescLayout);

                    if (m_interactiveViewport.materialPreviewDescLayout != VK_NULL_HANDLE) {
                    VkDescriptorPoolSize mpPoolSizes[2]{};
                    mpPoolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    mpPoolSizes[0].descriptorCount = 3; // binding 0 (materials) + binding 3 (terrain) + binding 4 (material ext)
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

                            VkDescriptorBufferInfo matExtBufInfo{};
                            matExtBufInfo.buffer = m_device->m_materialExtBuffer.buffer
                                                 ? m_device->m_materialExtBuffer.buffer
                                                 : materialBuffer;
                            matExtBufInfo.offset = 0;
                            matExtBufInfo.range = VK_WHOLE_SIZE;

                            VkWriteDescriptorSet mpWds[2]{};
                            mpWds[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                            mpWds[0].dstSet = m_interactiveViewport.materialPreviewDescSet;
                            mpWds[0].dstBinding = 0;
                            mpWds[0].descriptorCount = 1;
                            mpWds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                            mpWds[0].pBufferInfo = &matBufInfo;
                            mpWds[1] = mpWds[0];
                            mpWds[1].dstBinding = 4;
                            mpWds[1].pBufferInfo = &matExtBufInfo;
                            vkUpdateDescriptorSets(vkDevice, 2, mpWds, 0, nullptr);

                            if (!m_interactiveViewport.matcapImage.image) {
                                std::vector<uint8_t> white(4 * 2 * 2, 255);
                                const int64_t id = this->uploadTexture2D(white.data(), 2, 2, 4, false, false);
                                auto it = m_uploadedImages.find(id);
                                if (it != m_uploadedImages.end()) {
                                    m_interactiveViewport.matcapImage = it->second;
                                }
                            }
                            if (m_interactiveViewport.matcapImage.view && m_interactiveViewport.matcapImage.sampler) {
                                VkDescriptorImageInfo dummyInfo{};
                                dummyInfo.sampler = m_interactiveViewport.matcapImage.sampler;
                                dummyInfo.imageView = m_interactiveViewport.matcapImage.view;
                                dummyInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                                std::vector<VkDescriptorImageInfo> dummyInfos(mpTextureArrayLen, dummyInfo);
                                VkWriteDescriptorSet dummyWds{};
                                dummyWds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                dummyWds.dstSet = m_interactiveViewport.materialPreviewDescSet;
                                dummyWds.dstBinding = 1;
                                dummyWds.descriptorCount = mpTextureArrayLen;
                                dummyWds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                                dummyWds.pImageInfo = dummyInfos.data();
                                vkUpdateDescriptorSets(vkDevice, 1, &dummyWds, 0, nullptr);
                            }

                            // Populate binding 1 with all textures already uploaded.
                            // Textures uploaded later are handled by updateMaterialPreviewTextureDescriptor.
                            for (auto& [texID, texImg] : m_uploadedImages) {
                                if (texID <= 0 || static_cast<uint32_t>(texID) >= mpTextureArrayLen) continue;
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

    // --- Particle Billboard Pipelines (additive + alpha) ---
    if (m_interactiveViewport.particleAddPipeline == VK_NULL_HANDLE) {
        const std::string pVertPath = shaderDir + "/particle_viewport.spv";
        const std::string pFragPath = shaderDir + "/particle_viewport_frag.spv";
        if (std::filesystem::exists(pVertPath) && std::filesystem::exists(pFragPath)) {
            std::vector<uint32_t> pvSPV = loadViewportSPV(pVertPath);
            std::vector<uint32_t> pfSPV = loadViewportSPV(pFragPath);
            if (!pvSPV.empty() && !pfSPV.empty()) {
                VkShaderModuleCreateInfo smci{};
                smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

                smci.codeSize = pvSPV.size() * sizeof(uint32_t);
                smci.pCode = pvSPV.data();
                VkShaderModule pVert = VK_NULL_HANDLE;
                vkCreateShaderModule(vkDevice, &smci, nullptr, &pVert);

                smci.codeSize = pfSPV.size() * sizeof(uint32_t);
                smci.pCode = pfSPV.data();
                VkShaderModule pFrag = VK_NULL_HANDLE;
                vkCreateShaderModule(vkDevice, &smci, nullptr, &pFrag);

                if (pVert && pFrag) {
                    VkPipelineShaderStageCreateInfo pStages[2]{};
                    pStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    pStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
                    pStages[0].module = pVert;
                    pStages[0].pName  = "main";
                    pStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    pStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
                    pStages[1].module = pFrag;
                    pStages[1].pName  = "main";

                    // Vertex: {vec3 position, vec2 uv, vec4 rgba} = 36 bytes
                    VkVertexInputBindingDescription pBinding{};
                    pBinding.binding   = 0;
                    pBinding.stride    = sizeof(float) * 9;
                    pBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

                    VkVertexInputAttributeDescription pAttribs[3]{};
                    pAttribs[0].location = 0; pAttribs[0].binding = 0;
                    pAttribs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;    pAttribs[0].offset = 0;
                    pAttribs[1].location = 1; pAttribs[1].binding = 0;
                    pAttribs[1].format   = VK_FORMAT_R32G32_SFLOAT;       pAttribs[1].offset = sizeof(float) * 3;
                    pAttribs[2].location = 2; pAttribs[2].binding = 0;
                    pAttribs[2].format   = VK_FORMAT_R32G32B32A32_SFLOAT; pAttribs[2].offset = sizeof(float) * 5;

                    VkPipelineVertexInputStateCreateInfo pVI{};
                    pVI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
                    pVI.vertexBindingDescriptionCount   = 1;
                    pVI.pVertexBindingDescriptions      = &pBinding;
                    pVI.vertexAttributeDescriptionCount = 3;
                    pVI.pVertexAttributeDescriptions    = pAttribs;

                    VkPipelineInputAssemblyStateCreateInfo pIA{};
                    pIA.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
                    pIA.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

                    VkPipelineViewportStateCreateInfo pVP{};
                    pVP.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
                    pVP.viewportCount = 1;
                    pVP.scissorCount  = 1;

                    VkPipelineRasterizationStateCreateInfo pRas{};
                    pRas.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                    pRas.polygonMode = VK_POLYGON_MODE_FILL;
                    pRas.lineWidth   = 1.0f;
                    pRas.cullMode    = VK_CULL_MODE_NONE;
                    pRas.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

                    VkPipelineMultisampleStateCreateInfo pMS{};
                    pMS.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
                    pMS.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

                    VkPipelineDepthStencilStateCreateInfo pDS{};
                    pDS.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
                    pDS.depthTestEnable  = VK_TRUE;
                    pDS.depthWriteEnable = VK_FALSE; // transparent: test against scene, don't occlude
                    pDS.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

                    // Common blend skeleton; src = SRC_ALPHA. dst differs per mode.
                    VkPipelineColorBlendAttachmentState pCBA{};
                    pCBA.blendEnable         = VK_TRUE;
                    pCBA.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                    pCBA.colorBlendOp        = VK_BLEND_OP_ADD;
                    pCBA.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                    pCBA.alphaBlendOp        = VK_BLEND_OP_ADD;
                    pCBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

                    VkPipelineColorBlendStateCreateInfo pCB{};
                    pCB.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
                    pCB.attachmentCount = 1;
                    pCB.pAttachments    = &pCBA;

                    VkDynamicState pDyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
                    VkPipelineDynamicStateCreateInfo pDynState{};
                    pDynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
                    pDynState.dynamicStateCount = 2;
                    pDynState.pDynamicStates    = pDyn;

                    VkGraphicsPipelineCreateInfo pPI{};
                    pPI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                    pPI.stageCount          = 2;
                    pPI.pStages             = pStages;
                    pPI.pVertexInputState   = &pVI;
                    pPI.pInputAssemblyState = &pIA;
                    pPI.pViewportState      = &pVP;
                    pPI.pRasterizationState = &pRas;
                    pPI.pMultisampleState   = &pMS;
                    pPI.pDepthStencilState  = &pDS;
                    pPI.pColorBlendState    = &pCB;
                    pPI.pDynamicState       = &pDynState;
                    pPI.layout              = m_interactiveViewport.pipelineLayout;
                    pPI.renderPass          = m_interactiveViewport.renderPass;
                    pPI.subpass             = 0;

                    // Additive: dst = ONE (colors accumulate -> glow).
                    pCBA.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                    pCBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                    vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pPI, nullptr,
                                              &m_interactiveViewport.particleAddPipeline);

                    // Alpha: dst = ONE_MINUS_SRC_ALPHA (standard transparency).
                    pCBA.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    pCBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pPI, nullptr,
                                              &m_interactiveViewport.particleAlphaPipeline);
                }
                if (pVert) vkDestroyShaderModule(vkDevice, pVert, nullptr);
                if (pFrag) vkDestroyShaderModule(vkDevice, pFrag, nullptr);
            }
        }
    }

    // --- Edit-Mesh Overlay Pipelines (wireframe / face fill / vertex markers) ---
    if (m_interactiveViewport.editOverlayPipelineLayout == VK_NULL_HANDLE) {
        // Push-constant-only layout: mat4 mvp + 4x vec4 = 128 bytes, vertex stage.
        VkPushConstantRange ePCR{};
        ePCR.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        ePCR.offset = 0;
        ePCR.size = sizeof(float) * 32;
        VkPipelineLayoutCreateInfo ePLCI{};
        ePLCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ePLCI.pushConstantRangeCount = 1;
        ePLCI.pPushConstantRanges = &ePCR;
        vkCreatePipelineLayout(vkDevice, &ePLCI, nullptr, &m_interactiveViewport.editOverlayPipelineLayout);
    }
    if (m_interactiveViewport.editOverlayPipelineLayout != VK_NULL_HANDLE &&
        (m_interactiveViewport.editLinePipeline == VK_NULL_HANDLE ||
         m_interactiveViewport.editFacePipeline == VK_NULL_HANDLE ||
         m_interactiveViewport.editPointPipeline == VK_NULL_HANDLE)) {

        auto loadModule = [&](const std::string& path) -> VkShaderModule {
            if (!std::filesystem::exists(path)) return VK_NULL_HANDLE;
            std::vector<uint32_t> spv = loadViewportSPV(path);
            if (spv.empty()) return VK_NULL_HANDLE;
            VkShaderModuleCreateInfo smci{};
            smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = spv.size() * sizeof(uint32_t);
            smci.pCode = spv.data();
            VkShaderModule mod = VK_NULL_HANDLE;
            vkCreateShaderModule(vkDevice, &smci, nullptr, &mod);
            return mod;
        };
        VkShaderModule eVert  = loadModule(shaderDir + "/edit_overlay.spv");
        VkShaderModule eFrag  = loadModule(shaderDir + "/edit_overlay_frag.spv");
        VkShaderModule epVert = loadModule(shaderDir + "/edit_overlay_point.spv");
        VkShaderModule epFrag = loadModule(shaderDir + "/edit_overlay_point_frag.spv");

        // Shared fixed state. Two vertex streams: 0 = vec3 position, 1 = uint flags.
        // Line/face pipelines consume them per-vertex (indexed); the point
        // pipeline re-binds the same buffers per-instance and expands a quad
        // from gl_VertexIndex.
        auto buildPipeline = [&](VkShaderModule vert, VkShaderModule frag,
                                 VkPrimitiveTopology topology,
                                 VkVertexInputRate inputRate,
                                 VkPipeline* outPipeline) {
            if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return;

            VkPipelineShaderStageCreateInfo eStages[2]{};
            eStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            eStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            eStages[0].module = vert;
            eStages[0].pName  = "main";
            eStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            eStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            eStages[1].module = frag;
            eStages[1].pName  = "main";

            VkVertexInputBindingDescription eBindings[2]{};
            eBindings[0].binding = 0;
            eBindings[0].stride = sizeof(float) * 3;
            eBindings[0].inputRate = inputRate;
            eBindings[1].binding = 1;
            eBindings[1].stride = sizeof(uint32_t);
            eBindings[1].inputRate = inputRate;

            VkVertexInputAttributeDescription eAttribs[2]{};
            eAttribs[0].location = 0; eAttribs[0].binding = 0;
            eAttribs[0].format = VK_FORMAT_R32G32B32_SFLOAT; eAttribs[0].offset = 0;
            eAttribs[1].location = 1; eAttribs[1].binding = 1;
            eAttribs[1].format = VK_FORMAT_R32_UINT; eAttribs[1].offset = 0;

            VkPipelineVertexInputStateCreateInfo eVI{};
            eVI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            eVI.vertexBindingDescriptionCount = 2;
            eVI.pVertexBindingDescriptions = eBindings;
            eVI.vertexAttributeDescriptionCount = 2;
            eVI.pVertexAttributeDescriptions = eAttribs;

            VkPipelineInputAssemblyStateCreateInfo eIA{};
            eIA.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            eIA.topology = topology;

            VkPipelineViewportStateCreateInfo eVP{};
            eVP.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            eVP.viewportCount = 1;
            eVP.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo eRas{};
            eRas.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            eRas.polygonMode = VK_POLYGON_MODE_FILL;
            eRas.lineWidth = 1.0f;
            eRas.cullMode = VK_CULL_MODE_NONE;
            eRas.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineMultisampleStateCreateInfo eMS{};
            eMS.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            eMS.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo eDS{};
            eDS.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            eDS.depthTestEnable = VK_TRUE;
            eDS.depthWriteEnable = VK_FALSE; // overlay: test against scene, never occlude it
            eDS.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkPipelineColorBlendAttachmentState eCBA{};
            eCBA.blendEnable = VK_TRUE;
            eCBA.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            eCBA.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            eCBA.colorBlendOp = VK_BLEND_OP_ADD;
            eCBA.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            eCBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            eCBA.alphaBlendOp = VK_BLEND_OP_ADD;
            eCBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

            VkPipelineColorBlendStateCreateInfo eCB{};
            eCB.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            eCB.attachmentCount = 1;
            eCB.pAttachments = &eCBA;

            VkDynamicState eDyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo eDynState{};
            eDynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            eDynState.dynamicStateCount = 2;
            eDynState.pDynamicStates = eDyn;

            VkGraphicsPipelineCreateInfo ePI{};
            ePI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            ePI.stageCount = 2;
            ePI.pStages = eStages;
            ePI.pVertexInputState = &eVI;
            ePI.pInputAssemblyState = &eIA;
            ePI.pViewportState = &eVP;
            ePI.pRasterizationState = &eRas;
            ePI.pMultisampleState = &eMS;
            ePI.pDepthStencilState = &eDS;
            ePI.pColorBlendState = &eCB;
            ePI.pDynamicState = &eDynState;
            ePI.layout = m_interactiveViewport.editOverlayPipelineLayout;
            ePI.renderPass = m_interactiveViewport.renderPass;
            ePI.subpass = 0;

            vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &ePI, nullptr, outPipeline);
        };

        if (m_interactiveViewport.editLinePipeline == VK_NULL_HANDLE) {
            buildPipeline(eVert, eFrag, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                          VK_VERTEX_INPUT_RATE_VERTEX, &m_interactiveViewport.editLinePipeline);
        }
        if (m_interactiveViewport.editFacePipeline == VK_NULL_HANDLE) {
            buildPipeline(eVert, eFrag, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                          VK_VERTEX_INPUT_RATE_VERTEX, &m_interactiveViewport.editFacePipeline);
        }
        if (m_interactiveViewport.editPointPipeline == VK_NULL_HANDLE) {
            buildPipeline(epVert, epFrag, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                          VK_VERTEX_INPUT_RATE_INSTANCE, &m_interactiveViewport.editPointPipeline);
        }

        if (eVert)  vkDestroyShaderModule(vkDevice, eVert, nullptr);
        if (eFrag)  vkDestroyShaderModule(vkDevice, eFrag, nullptr);
        if (epVert) vkDestroyShaderModule(vkDevice, epVert, nullptr);
        if (epFrag) vkDestroyShaderModule(vkDevice, epFrag, nullptr);
    }

    // ── Selection outline: render passes + pipelines (created once) ─────────
    // Mask pass re-draws the selected instances into an R8G8 target
    // (G = full silhouette via depth-test-off, R = visible via depth-test
    // against the scene depth left by the main pass), then the composite pass
    // edge-detects the mask over the finished color image. Everything here is
    // optional: when the SPIR-V is missing the pipelines stay null and the
    // draw path simply skips, leaving the UI on its ImGui fallback.
    if (m_interactiveViewport.selectionMaskRenderPass == VK_NULL_HANDLE) {
        VkAttachmentDescription mAtt[2]{};
        mAtt[0].format = VK_FORMAT_R8G8_UNORM;
        mAtt[0].samples = VK_SAMPLE_COUNT_1_BIT;
        mAtt[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        mAtt[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        mAtt[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        mAtt[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        mAtt[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        mAtt[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        // Scene depth: loaded read-only (depthWriteEnable=false in both
        // pipelines); contents are not needed after this pass.
        mAtt[1].format = VK_FORMAT_D32_SFLOAT;
        mAtt[1].samples = VK_SAMPLE_COUNT_1_BIT;
        mAtt[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        mAtt[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        mAtt[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        mAtt[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        mAtt[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        mAtt[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference mColorRef{};
        mColorRef.attachment = 0;
        mColorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentReference mDepthRef{};
        mDepthRef.attachment = 1;
        mDepthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription mSub{};
        mSub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        mSub.colorAttachmentCount = 1;
        mSub.pColorAttachments = &mColorRef;
        mSub.pDepthStencilAttachment = &mDepthRef;

        VkSubpassDependency mDeps[2]{};
        // Main pass depth writes -> our depth test reads.
        mDeps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        mDeps[0].dstSubpass = 0;
        mDeps[0].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        mDeps[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        mDeps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        mDeps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        // Mask writes -> composite fragment shader sampling.
        mDeps[1].srcSubpass = 0;
        mDeps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        mDeps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        mDeps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        mDeps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        mDeps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo mRPCI{};
        mRPCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        mRPCI.attachmentCount = 2;
        mRPCI.pAttachments = mAtt;
        mRPCI.subpassCount = 1;
        mRPCI.pSubpasses = &mSub;
        mRPCI.dependencyCount = 2;
        mRPCI.pDependencies = mDeps;
        vkCreateRenderPass(vkDevice, &mRPCI, nullptr, &m_interactiveViewport.selectionMaskRenderPass);
    }
    if (m_interactiveViewport.selectionCompositeRenderPass == VK_NULL_HANDLE) {
        // Composite over the finished color image. The main pass leaves the
        // color image in GENERAL; keep it there so the staging copy that
        // follows sees the layout it expects.
        VkAttachmentDescription cAtt{};
        cAtt.format = VK_FORMAT_R8G8B8A8_UNORM;
        cAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        cAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        cAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        cAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        cAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        cAtt.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        cAtt.finalLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkAttachmentReference cColorRef{};
        cColorRef.attachment = 0;
        cColorRef.layout = VK_IMAGE_LAYOUT_GENERAL;

        VkSubpassDescription cSub{};
        cSub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        cSub.colorAttachmentCount = 1;
        cSub.pColorAttachments = &cColorRef;

        VkSubpassDependency cDep{};
        cDep.srcSubpass = VK_SUBPASS_EXTERNAL;
        cDep.dstSubpass = 0;
        cDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        cDep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        cDep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        cDep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo cRPCI{};
        cRPCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        cRPCI.attachmentCount = 1;
        cRPCI.pAttachments = &cAtt;
        cRPCI.subpassCount = 1;
        cRPCI.pSubpasses = &cSub;
        cRPCI.dependencyCount = 1;
        cRPCI.pDependencies = &cDep;
        vkCreateRenderPass(vkDevice, &cRPCI, nullptr, &m_interactiveViewport.selectionCompositeRenderPass);
    }
    if (m_interactiveViewport.selectionMaskSampler == VK_NULL_HANDLE) {
        // Bilinear so the composite's fractional kernel offsets give cheap AA.
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(vkDevice, &sci, nullptr, &m_interactiveViewport.selectionMaskSampler);
    }
    if (m_interactiveViewport.selectionMaskPipelineLayout == VK_NULL_HANDLE) {
        // mat4 viewProj + vec4 maskValue = 80 bytes, vertex stage.
        VkPushConstantRange sPCR{};
        sPCR.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        sPCR.offset = 0;
        sPCR.size = sizeof(float) * 20;
        VkPipelineLayoutCreateInfo sPLCI{};
        sPLCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        sPLCI.pushConstantRangeCount = 1;
        sPLCI.pPushConstantRanges = &sPCR;
        vkCreatePipelineLayout(vkDevice, &sPLCI, nullptr, &m_interactiveViewport.selectionMaskPipelineLayout);
    }
    if (m_interactiveViewport.selectionCompositeDescLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding maskBinding{};
        maskBinding.binding = 0;
        maskBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        maskBinding.descriptorCount = 1;
        maskBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 1;
        dslci.pBindings = &maskBinding;
        vkCreateDescriptorSetLayout(vkDevice, &dslci, nullptr, &m_interactiveViewport.selectionCompositeDescLayout);

        if (m_interactiveViewport.selectionCompositeDescLayout != VK_NULL_HANDLE) {
            VkDescriptorPoolSize poolSize{};
            poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            poolSize.descriptorCount = 1;
            VkDescriptorPoolCreateInfo dpci{};
            dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.poolSizeCount = 1;
            dpci.pPoolSizes = &poolSize;
            dpci.maxSets = 1;
            if (vkCreateDescriptorPool(vkDevice, &dpci, nullptr, &m_interactiveViewport.selectionCompositeDescPool) == VK_SUCCESS) {
                VkDescriptorSetAllocateInfo dsai{};
                dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                dsai.descriptorPool = m_interactiveViewport.selectionCompositeDescPool;
                dsai.descriptorSetCount = 1;
                dsai.pSetLayouts = &m_interactiveViewport.selectionCompositeDescLayout;
                vkAllocateDescriptorSets(vkDevice, &dsai, &m_interactiveViewport.selectionCompositeDescSet);
            }
        }
    }
    if (m_interactiveViewport.selectionCompositePipelineLayout == VK_NULL_HANDLE &&
        m_interactiveViewport.selectionCompositeDescLayout != VK_NULL_HANDLE) {
        // 4x vec4 (primary/secondary/occluded colors + params) fragment stage.
        VkPushConstantRange cPCR{};
        cPCR.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        cPCR.offset = 0;
        cPCR.size = sizeof(float) * 16;
        VkPipelineLayoutCreateInfo cPLCI{};
        cPLCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        cPLCI.pushConstantRangeCount = 1;
        cPLCI.pPushConstantRanges = &cPCR;
        cPLCI.setLayoutCount = 1;
        cPLCI.pSetLayouts = &m_interactiveViewport.selectionCompositeDescLayout;
        vkCreatePipelineLayout(vkDevice, &cPLCI, nullptr, &m_interactiveViewport.selectionCompositePipelineLayout);
    }
    if (m_interactiveViewport.selectionMaskRenderPass != VK_NULL_HANDLE &&
        m_interactiveViewport.selectionMaskPipelineLayout != VK_NULL_HANDLE &&
        (m_interactiveViewport.selectionMaskFullPipeline == VK_NULL_HANDLE ||
         m_interactiveViewport.selectionMaskVisiblePipeline == VK_NULL_HANDLE)) {

        auto loadSelModule = [&](const std::string& path) -> VkShaderModule {
            if (!std::filesystem::exists(path)) return VK_NULL_HANDLE;
            std::vector<uint32_t> spv = loadViewportSPV(path);
            if (spv.empty()) return VK_NULL_HANDLE;
            VkShaderModuleCreateInfo smci{};
            smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = spv.size() * sizeof(uint32_t);
            smci.pCode = spv.data();
            VkShaderModule mod = VK_NULL_HANDLE;
            vkCreateShaderModule(vkDevice, &smci, nullptr, &mod);
            return mod;
        };
        VkShaderModule smVert = loadSelModule(shaderDir + "/selection_mask.spv");
        VkShaderModule smFrag = loadSelModule(shaderDir + "/selection_mask_frag.spv");

        auto buildMaskPipeline = [&](bool depthTest, VkColorComponentFlags writeMask,
                                     VkPipeline* outPipeline) {
            if (smVert == VK_NULL_HANDLE || smFrag == VK_NULL_HANDLE) return;

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = smVert;
            stages[0].pName  = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = smFrag;
            stages[1].pName  = "main";

            // Stream 0: raster mesh position buffer (vec3). Stream 1: one
            // mat4 per selected instance (per-instance rate).
            VkVertexInputBindingDescription bind[2]{};
            bind[0].binding = 0;
            bind[0].stride = sizeof(float) * 3;
            bind[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            bind[1].binding = 1;
            bind[1].stride = sizeof(float) * 16;
            bind[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

            VkVertexInputAttributeDescription attr[5]{};
            attr[0].location = 0; attr[0].binding = 0;
            attr[0].format = VK_FORMAT_R32G32B32_SFLOAT; attr[0].offset = 0;
            for (int i = 0; i < 4; ++i) {
                attr[1 + i].location = 1 + i;
                attr[1 + i].binding = 1;
                attr[1 + i].format = VK_FORMAT_R32G32B32A32_SFLOAT;
                attr[1 + i].offset = sizeof(float) * 4 * i;
            }

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount = 2;
            vi.pVertexBindingDescriptions = bind;
            vi.vertexAttributeDescriptionCount = 5;
            vi.pVertexAttributeDescriptions = attr;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo ras{};
            ras.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            ras.polygonMode = VK_POLYGON_MODE_FILL;
            ras.lineWidth = 1.0f;
            ras.cullMode = VK_CULL_MODE_NONE; // silhouette must include backfaces (open meshes)
            ras.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable = depthTest ? VK_TRUE : VK_FALSE;
            ds.depthWriteEnable = VK_FALSE; // scene depth is read-only here
            ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            // MAX blend so overlapping primary (1.0) and secondary (0.5)
            // silhouettes keep the stronger tier per channel.
            VkPipelineColorBlendAttachmentState cba{};
            cba.blendEnable = VK_TRUE;
            cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cba.colorBlendOp = VK_BLEND_OP_MAX;
            cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cba.alphaBlendOp = VK_BLEND_OP_MAX;
            cba.colorWriteMask = writeMask;

            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 1;
            cb.pAttachments = &cba;

            VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynState{};
            dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynState.dynamicStateCount = 2;
            dynState.pDynamicStates = dyn;

            VkGraphicsPipelineCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pi.stageCount = 2;
            pi.pStages = stages;
            pi.pVertexInputState = &vi;
            pi.pInputAssemblyState = &ia;
            pi.pViewportState = &vp;
            pi.pRasterizationState = &ras;
            pi.pMultisampleState = &ms;
            pi.pDepthStencilState = &ds;
            pi.pColorBlendState = &cb;
            pi.pDynamicState = &dynState;
            pi.layout = m_interactiveViewport.selectionMaskPipelineLayout;
            pi.renderPass = m_interactiveViewport.selectionMaskRenderPass;
            pi.subpass = 0;
            vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pi, nullptr, outPipeline);
        };

        if (m_interactiveViewport.selectionMaskFullPipeline == VK_NULL_HANDLE) {
            buildMaskPipeline(false, VK_COLOR_COMPONENT_G_BIT,
                              &m_interactiveViewport.selectionMaskFullPipeline);
        }
        if (m_interactiveViewport.selectionMaskVisiblePipeline == VK_NULL_HANDLE) {
            buildMaskPipeline(true, VK_COLOR_COMPONENT_R_BIT,
                              &m_interactiveViewport.selectionMaskVisiblePipeline);
        }

        if (smVert) vkDestroyShaderModule(vkDevice, smVert, nullptr);
        if (smFrag) vkDestroyShaderModule(vkDevice, smFrag, nullptr);
    }
    if (m_interactiveViewport.selectionCompositePipeline == VK_NULL_HANDLE &&
        m_interactiveViewport.selectionCompositeRenderPass != VK_NULL_HANDLE &&
        m_interactiveViewport.selectionCompositePipelineLayout != VK_NULL_HANDLE) {

        auto loadSelModule = [&](const std::string& path) -> VkShaderModule {
            if (!std::filesystem::exists(path)) return VK_NULL_HANDLE;
            std::vector<uint32_t> spv = loadViewportSPV(path);
            if (spv.empty()) return VK_NULL_HANDLE;
            VkShaderModuleCreateInfo smci{};
            smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = spv.size() * sizeof(uint32_t);
            smci.pCode = spv.data();
            VkShaderModule mod = VK_NULL_HANDLE;
            vkCreateShaderModule(vkDevice, &smci, nullptr, &mod);
            return mod;
        };
        VkShaderModule soVert = loadSelModule(shaderDir + "/selection_outline.spv");
        VkShaderModule soFrag = loadSelModule(shaderDir + "/selection_outline_frag.spv");

        if (soVert != VK_NULL_HANDLE && soFrag != VK_NULL_HANDLE) {
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = soVert;
            stages[0].pName  = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = soFrag;
            stages[1].pName  = "main";

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo ras{};
            ras.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            ras.polygonMode = VK_POLYGON_MODE_FILL;
            ras.lineWidth = 1.0f;
            ras.cullMode = VK_CULL_MODE_NONE;
            ras.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

            VkPipelineColorBlendAttachmentState cba{};
            cba.blendEnable = VK_TRUE;
            cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cba.colorBlendOp = VK_BLEND_OP_ADD;
            cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cba.alphaBlendOp = VK_BLEND_OP_ADD;
            cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 1;
            cb.pAttachments = &cba;

            VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynState{};
            dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynState.dynamicStateCount = 2;
            dynState.pDynamicStates = dyn;

            VkGraphicsPipelineCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pi.stageCount = 2;
            pi.pStages = stages;
            pi.pVertexInputState = &vi;
            pi.pInputAssemblyState = &ia;
            pi.pViewportState = &vp;
            pi.pRasterizationState = &ras;
            pi.pMultisampleState = &ms;
            pi.pDepthStencilState = &ds;
            pi.pColorBlendState = &cb;
            pi.pDynamicState = &dynState;
            pi.layout = m_interactiveViewport.selectionCompositePipelineLayout;
            pi.renderPass = m_interactiveViewport.selectionCompositeRenderPass;
            pi.subpass = 0;
            vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pi, nullptr,
                                      &m_interactiveViewport.selectionCompositePipeline);
        }
        if (soVert) vkDestroyShaderModule(vkDevice, soVert, nullptr);
        if (soFrag) vkDestroyShaderModule(vkDevice, soFrag, nullptr);
    }

   // SCENE_LOG_INFO("[MP-init] step=postAllPipelines");
    // NOTE: selection-outline mask resources are intentionally NOT part of
    // this early-return: they are created in the same resize pass as the
    // main framebuffer below, and if their creation ever failed (memory
    // pressure) re-trying every frame would thrash the whole viewport.
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

    // Selection outline targets share the viewport size; failure here only
    // disables the outline (the main viewport stays functional).
    if (m_interactiveViewport.selectionMaskRenderPass != VK_NULL_HANDLE &&
        m_interactiveViewport.selectionCompositeRenderPass != VK_NULL_HANDLE) {
        m_interactiveViewport.selectionMaskImage = m_device->createImage2D(
            (uint32_t)width, (uint32_t)height, VK_FORMAT_R8G8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        if (m_interactiveViewport.selectionMaskImage.image) {
            VkImageView maskAttachments[2] = {
                m_interactiveViewport.selectionMaskImage.view,
                m_interactiveViewport.depthImage.view
            };
            VkFramebufferCreateInfo maskFBI{};
            maskFBI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            maskFBI.renderPass = m_interactiveViewport.selectionMaskRenderPass;
            maskFBI.attachmentCount = 2;
            maskFBI.pAttachments = maskAttachments;
            maskFBI.width = (uint32_t)width;
            maskFBI.height = (uint32_t)height;
            maskFBI.layers = 1;
            vkCreateFramebuffer(vkDevice, &maskFBI, nullptr, &m_interactiveViewport.selectionMaskFramebuffer);

            VkFramebufferCreateInfo compFBI{};
            compFBI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            compFBI.renderPass = m_interactiveViewport.selectionCompositeRenderPass;
            compFBI.attachmentCount = 1;
            compFBI.pAttachments = &m_interactiveViewport.colorImage.view;
            compFBI.width = (uint32_t)width;
            compFBI.height = (uint32_t)height;
            compFBI.layers = 1;
            vkCreateFramebuffer(vkDevice, &compFBI, nullptr, &m_interactiveViewport.selectionCompositeFramebuffer);

            // (Re)point the composite descriptor at the fresh mask image.
            if (m_interactiveViewport.selectionCompositeDescSet != VK_NULL_HANDLE &&
                m_interactiveViewport.selectionMaskSampler != VK_NULL_HANDLE) {
                VkDescriptorImageInfo di{};
                di.sampler = m_interactiveViewport.selectionMaskSampler;
                di.imageView = m_interactiveViewport.selectionMaskImage.view;
                di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkWriteDescriptorSet wds{};
                wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wds.dstSet = m_interactiveViewport.selectionCompositeDescSet;
                wds.dstBinding = 0;
                wds.descriptorCount = 1;
                wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                wds.pImageInfo = &di;
                vkUpdateDescriptorSets(vkDevice, 1, &wds, 0, nullptr);
            }
        }
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
        mix(c.orthographic ? 1.0f : 0.0f);
        mix(c.orthoHeight);
        mix((float)c.gridPlane);
        return h;
    };
    uint64_t camHash = hashCamera(m_camera);
    {
        // Grid settings join the change hash so slider edits invalidate the cached frame.
        auto mixGrid = [&](float v) { uint32_t bits; std::memcpy(&bits, &v, 4); camHash ^= bits; camHash *= 1099511628211ull; };
        mixGrid(::render_settings.grid_fade_distance);
        mixGrid(::render_settings.grid_opacity);
    }
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
            return;
        }
    }
    m_lastCameraHash = camHash;
    m_interactiveViewport.dirty = false;

    struct SolidPushConstants {
        float viewProj[16];
        float view[16];
        int useMatcap;
        float overrideR, overrideG, overrideB;
        // Grid distance fade (world units, around fadeCenter). fadeEnd <= fadeStart disables.
        float fadeCenterX, fadeCenterY, fadeCenterZ;
        float fadeStart, fadeEnd;
        float overrideA; // base opacity for flat-color draws (grid)
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

    auto makeOrthoMatrix = [&](float orthoHeight, float aspect, float zNear, float zFar) {
        const float h = (orthoHeight > 1e-4f) ? orthoHeight : 10.0f;
        const float w = h * aspect;
        Matrix4x4 proj = Matrix4x4::zero();
        proj.m[0][0] = 2.0f / w;
        proj.m[1][1] = -2.0f / h;                  // Y-flip, matching the perspective convention
        proj.m[2][2] = 1.0f / (zNear - zFar);      // maps z_eye[-near,-far] -> NDC z[0,1]
        proj.m[2][3] = zNear / (zNear - zFar);
        proj.m[3][3] = 1.0f;
        return proj;
    };

    Matrix4x4 view = makeViewMatrix(m_camera.origin, m_camera.lookAt, m_camera.up);
    const float aspect = (height > 0) ? ((float)width / (float)height) : 1.0f;
    Matrix4x4 proj = m_camera.orthographic
        ? makeOrthoMatrix(m_camera.orthoHeight, aspect, 0.01f, 1000000.0f)
        : makePerspectiveMatrix(
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
        double baseBudgetMillions = 72.0;
        uint64_t minBudget = 24ull * 1000ull * 1000ull;
        uint64_t maxBudget = 96ull * 1000ull * 1000ull;
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
        } else if (m_viewportMode == ViewportMode::MaterialPreview) {
            baseBudgetMillions = 36.0;
            minBudget = 8ull * 1000ull * 1000ull;
            maxBudget = 48ull * 1000ull * 1000ull;
            switch (::render_settings.raster_viewport_quality_preset) {
                case ::RasterViewportQualityPreset::Performance:
                    rasterQualityScale = 0.45;
                    break;
                case ::RasterViewportQualityPreset::Balanced:
                    rasterQualityScale = 0.65;
                    break;
                case ::RasterViewportQualityPreset::Quality:
                    rasterQualityScale = 1.0;
                    break;
                case ::RasterViewportQualityPreset::Auto:
                default:
                    rasterQualityScale = 0.72;
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
        const uint64_t adaptiveBudget = static_cast<uint64_t>(
            baseBudgetMillions * 1000.0 * 1000.0 * resolutionScale * rasterQualityScale * feedbackScale);
        m_rasterScatterTriangleBudget = std::clamp<uint64_t>(adaptiveBudget, minBudget, maxBudget);
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
        m_interactiveViewport.dirty = true;
        return;
    }

    // ── Selection outline: resolve node names → raster instances and upload
    // their matrices BEFORE command recording starts (uploadBuffer submits its
    // own transfer command buffer).
    std::vector<SelectionOutlineDrawItem> selectionOutlineDraws;
    {
        const Backend::SelectionOutlineParams& sop = m_interactiveViewport.selectionOutlineParams;
        const bool selectionOutlineReady =
            sop.enabled && !sop.nodeNames.empty() &&
            m_interactiveViewport.selectionMaskFullPipeline != VK_NULL_HANDLE &&
            m_interactiveViewport.selectionMaskVisiblePipeline != VK_NULL_HANDLE &&
            m_interactiveViewport.selectionCompositePipeline != VK_NULL_HANDLE &&
            m_interactiveViewport.selectionMaskFramebuffer != VK_NULL_HANDLE &&
            m_interactiveViewport.selectionCompositeFramebuffer != VK_NULL_HANDLE &&
            m_interactiveViewport.selectionCompositeDescSet != VK_NULL_HANDLE;
        if (selectionOutlineReady) {
            resolveSelectionOutlineDraws(sop.nodeNames, selectionOutlineDraws);
        }
    }

    VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        m_interactiveViewport.dirty = true;
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

            const bool drawMaterialPreview =
                useMaterialPreview &&
                !rmb.isScatterProxy &&
                rmb.matIdBuffer.buffer &&
                rmb.uvBuffer.buffer;

            if (drawMaterialPreview) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.materialPreviewPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        m_interactiveViewport.materialPreviewPipelineLayout,
                                        0, 1, &m_interactiveViewport.materialPreviewDescSet, 0, nullptr);
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
                mpPush.materialMeta[3] = m_interactiveViewport.materialPreviewTextureArrayLen;
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
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.solidPipeline);
                if (m_interactiveViewport.matcapDescSet != VK_NULL_HANDLE) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            m_interactiveViewport.pipelineLayout,
                                            0, 1, &m_interactiveViewport.matcapDescSet, 0, nullptr);
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

    // Rebuild the reference grid when it doesn't exist yet, when the active plane changed
    // (standard view switched), or when the adaptive spacing bucket changed (zoom). The grid
    // lays out on the plane the camera faces so it stays a usable alignment reference, and the
    // spacing follows the zoom so it never collapses to a few lines or fuses into a mass.
    const int activeGridPlane = m_camera.gridPlane;
    float viewScale;
    if (m_camera.orthographic) {
        viewScale = (m_camera.orthoHeight > 1e-4f) ? m_camera.orthoHeight : 10.0f;
    } else {
        // Use the camera's perpendicular distance to the active grid plane.
        // This is much more stable than orbit distance (origin-lookAt) because:
        //  - It doesn't change during pan (camera slides parallel to the grid)
        //  - It changes smoothly during orbit/zoom
        //  - It reflects the actual visible grid extent
        float planeDist;
        switch (activeGridPlane) {
            case 1:  planeDist = std::abs(m_camera.origin.z); break; // XY plane
            case 2:  planeDist = std::abs(m_camera.origin.x); break; // YZ plane
            default: planeDist = std::abs(m_camera.origin.y); break; // XZ floor
        }
        // Scale by FOV so wider lenses get coarser grid (more visible area)
        const float fovRad = (m_camera.fov > 1.0f ? m_camera.fov : 60.0f) * 3.14159265f / 180.0f;
        const float fovScale = 2.0f * std::tan(fovRad * 0.5f); // visible height at unit distance
        viewScale = planeDist * fovScale;
        // Fallback: when camera is ON the grid plane (planeDist ≈ 0), use orbit distance
        if (viewScale < 0.5f) {
            const float orbitDist = (m_camera.origin - m_camera.lookAt).length();
            viewScale = (std::max)(orbitDist * fovScale, 1.0f);
        }
    }
    // Finer 1-2-5 sub-steps to reduce visible grid "pop" (max ratio ≈ 1.5×).
    auto niceStep = [](float x) -> float {
        if (x <= 1e-6f) return 1.0f;
        const float p = std::pow(10.0f, std::floor(std::log10(x)));
        const float n = x / p; // 1..10
        float m;
        if      (n < 1.25f) m = 1.0f;
        else if (n < 1.75f) m = 1.5f;
        else if (n < 2.5f)  m = 2.0f;
        else if (n < 4.0f)  m = 3.0f;
        else if (n < 6.0f)  m = 5.0f;
        else if (n < 8.5f)  m = 7.0f;
        else                m = 10.0f;
        return m * p;
    };
    // Hysteresis: only switch spacing when the new value differs by >15%
    // from the currently built spacing. This prevents rapid flip-flopping
    // at the boundary of two nice-step buckets during camera pan.
    float candidateSpacing = niceStep(viewScale * 0.1f);
    const float builtSpacing = m_interactiveViewport.gridBuiltSpacing;
    if (builtSpacing > 0.0f) {
        const float ratio = candidateSpacing / builtSpacing;
        if (ratio > 0.87f && ratio < 1.15f) {
            candidateSpacing = builtSpacing; // keep current
        }
    }
    const float gridSpacing = candidateSpacing;

    // The grid stays ORIGIN-centred (the world axes are the visual anchor), but its extent must
    // not be cropped to the zoom alone: gridHalf ~ viewScale meant a floor close-up far from the
    // origin shrank the patch away from under the camera and the grid seemed to vanish. The
    // camera's planar distance from the origin now sets a lower bound on the extent instead.
    float camU, camV;
    switch (activeGridPlane) {
        case 1:  camU = m_camera.origin.x; camV = m_camera.origin.y; break; // XY (Front/Back)
        case 2:  camU = m_camera.origin.z; camV = m_camera.origin.y; break; // YZ (Left/Right)
        default: camU = m_camera.origin.x; camV = m_camera.origin.z; break; // XZ floor
    }
    const float camPlanarDist = (std::max)(std::abs(camU), std::abs(camV));
    // User knob: scales the fog horizon (and with it the built extents below). Growth re-triggers
    // a rebuild via extentStale; shrink may keep oversized geometry, which the draw-side fade
    // bands simply dissolve earlier.
    const float gridFadeScale = std::clamp(::render_settings.grid_fade_distance, 0.25f, 4.0f);
    // 16x viewScale: the major lattice must reach past the fade horizon (~19x viewScale, see
    // drawSeg fade bands below) so lines dissolve into the fog instead of ending at a visible
    // geometric edge. With the 1.25x build margin the edge sits at >=20x viewScale.
    const float requiredHalf = viewScale * 16.0f * gridFadeScale + camPlanarDist;
    // Minor (fine) lattice is camera-centred (see rebuild below); snap its centre to the major
    // (10x) lattice so minor lines always land on the global grid as the patch follows the camera.
    const float coarseSpacing = gridSpacing * 10.0f;
    const float fineCenterU = std::round(camU / coarseSpacing) * coarseSpacing;
    const float fineCenterV = std::round(camV / coarseSpacing) * coarseSpacing;
    // Hysteresis: rebuild when the camera outgrows the built extent, or when the built extent
    // is so oversized (>2.5x) that the grid should shrink back. The 1.25x build margin below
    // keeps small camera moves from re-triggering this every frame.
    const bool extentStale =
        requiredHalf > m_interactiveViewport.gridBuiltHalf ||
        requiredHalf < m_interactiveViewport.gridBuiltHalf * 0.4f;

    if (!m_interactiveViewport.gridVertexBuffer.buffer ||
        m_interactiveViewport.gridBuiltPlane != activeGridPlane ||
        m_interactiveViewport.gridBuiltSpacing != gridSpacing ||
        m_interactiveViewport.gridBuiltCenterU != fineCenterU ||
        m_interactiveViewport.gridBuiltCenterV != fineCenterV ||
        extentStale) {
        if (m_interactiveViewport.gridVertexBuffer.buffer) {
            m_device->destroyBuffer(m_interactiveViewport.gridVertexBuffer);
            m_interactiveViewport.gridVertexBuffer = {};
        }
        if (m_interactiveViewport.gridNormalBuffer.buffer) {
            m_device->destroyBuffer(m_interactiveViewport.gridNormalBuffer);
            m_interactiveViewport.gridNormalBuffer = {};
        }

        // Two-tier lattice. MINOR lines (current spacing) only span the visible area around the
        // camera footprint — building them across the whole extended extent made distant lines
        // collapse below a pixel and moiré badly in perspective (especially at 720p, no MSAA).
        // MAJOR lines (10x spacing, 10x thickness = every 10th minor) span the full extent from
        // the origin out past the camera, so the far field stays referenced with far fewer,
        // fatter lines. Axes are unchanged (origin-anchored, full extent). 25% growth margin so
        // small camera moves don't immediately re-trigger a rebuild.
        const float desiredHalf = requiredHalf * 1.25f;
        const int   coarseEachSide = std::clamp((int)std::ceil(desiredHalf / coarseSpacing), 4, 2048);
        const float gridHalf = coarseSpacing * (float)coarseEachSide;
        // 18x viewScale: minor lines reach well past the working area (10x still read as an
        // early cutoff); the distance fade below dissolves them before the geometric edge and
        // before they collapse sub-pixel and moiré (~40x viewScale at 720p).
        const int   fineEachSide = std::clamp((int)std::ceil((viewScale * 18.0f * gridFadeScale) / gridSpacing), 10, 1024);
        const float fineHalf = gridSpacing * (float)fineEachSide;
        const float step = gridSpacing;
        const float thin = gridSpacing * 0.008f; // thicker so 720p (no MSAA) doesn't break lines up
        const float coarseThin = coarseSpacing * 0.008f;
        const float axisThin = gridSpacing * 0.022f;

        // Plane basis: ax = horizontal axis, ay = vertical axis, nrm = plane normal.
        Vec3 ax, ay, nrm;
        switch (activeGridPlane) {
            case 1: ax = Vec3(1, 0, 0); ay = Vec3(0, 1, 0); nrm = Vec3(0, 0, 1); break; // XY (Front/Back)
            case 2: ax = Vec3(0, 0, 1); ay = Vec3(0, 1, 0); nrm = Vec3(1, 0, 0); break; // YZ (Left/Right)
            default: ax = Vec3(1, 0, 0); ay = Vec3(0, 0, 1); nrm = Vec3(0, 1, 0); break; // XZ (Top/Persp floor)
        }
        const Vec3 nOff = nrm * (gridSpacing * 0.0002f); // lift axis lines so they win the depth test

        std::vector<float> positions, normals;
        auto addLineQuad = [&](Vec3 a, Vec3 b, Vec3 widthDir) {
            Vec3 w = widthDir;
            Vec3 p0 = a - w, p1 = a + w, p2 = b + w, p3 = b - w;
            Vec3 n = nrm;
            // Emit triangles (front-facing) and their reversed winding (back-facing)
            for (const Vec3& p : { p0, p1, p2, p0, p2, p3, p2, p1, p0, p3, p2, p0 }) {
                positions.push_back(p.x); positions.push_back(p.y); positions.push_back(p.z);
                normals.push_back(n.x); normals.push_back(n.y); normals.push_back(n.z);
            }
        };

        // Major lattice (full extent, origin-centred; i==0 is covered by the axis quads).
        // Kept in its own segment so it can fade at the far horizon while the minor
        // lattice fades earlier (separate fade bands per drawSeg call).
        uint32_t segMajorStart = 0;
        for (int i = -coarseEachSide; i <= coarseEachSide; ++i) {
            if (i == 0) continue;
            const float u = coarseSpacing * (float)i;
            addLineQuad(ax * u - ay * gridHalf, ax * u + ay * gridHalf, ax * coarseThin);
        }
        for (int i = -coarseEachSide; i <= coarseEachSide; ++i) {
            if (i == 0) continue;
            const float v = coarseSpacing * (float)i;
            addLineQuad(ay * v - ax * gridHalf, ay * v + ax * gridHalf, ay * coarseThin);
        }
        uint32_t segMajorCount = (uint32_t)(positions.size() / 3) - segMajorStart;

        // Minor lattice (visible area, centred on the camera footprint snapped to the major
        // lattice — every 10th index lands on a major line / the axes and is skipped).
        uint32_t segMinorStart = (uint32_t)(positions.size() / 3);
        for (int i = -fineEachSide; i <= fineEachSide; ++i) {
            if (((int)std::lround(fineCenterU / step) + i) % 10 == 0) continue;
            const float u = fineCenterU + step * (float)i;
            addLineQuad(ax * u + ay * (fineCenterV - fineHalf), ax * u + ay * (fineCenterV + fineHalf), ax * thin);
        }
        for (int i = -fineEachSide; i <= fineEachSide; ++i) {
            if (((int)std::lround(fineCenterV / step) + i) % 10 == 0) continue;
            const float v = fineCenterV + step * (float)i;
            addLineQuad(ay * v + ax * (fineCenterU - fineHalf), ay * v + ax * (fineCenterU + fineHalf), ay * thin);
        }
        uint32_t segMinorCount = (uint32_t)(positions.size() / 3) - segMinorStart;

        uint32_t segAxisUStart = (uint32_t)(positions.size() / 3);
        addLineQuad(nOff, ax * gridHalf + nOff, ay * axisThin);
        uint32_t segAxisUCount = (uint32_t)(positions.size() / 3) - segAxisUStart;

        uint32_t segAxisVStart = (uint32_t)(positions.size() / 3);
        addLineQuad(nOff, ay * gridHalf + nOff, ax * axisThin);
        uint32_t segAxisVCount = (uint32_t)(positions.size() / 3) - segAxisVStart;

        uint32_t segNegStart = (uint32_t)(positions.size() / 3);
        addLineQuad(ax * -gridHalf + nOff, nOff, ay * thin);
        addLineQuad(ay * -gridHalf + nOff, nOff, ax * thin);
        uint32_t segNegCount = (uint32_t)(positions.size() / 3) - segNegStart;

        m_interactiveViewport.gridBuiltPlane = activeGridPlane;
        m_interactiveViewport.gridBuiltSpacing = gridSpacing;
        // Store the UNCLAMPED target so a hit line-cap doesn't re-trigger a rebuild every frame.
        m_interactiveViewport.gridBuiltHalf = (std::max)(gridHalf, desiredHalf);
        m_interactiveViewport.gridBuiltCenterU = fineCenterU;
        m_interactiveViewport.gridBuiltCenterV = fineCenterV;
        m_interactiveViewport.gridBuiltFineHalf = fineHalf;

        m_interactiveViewport.gridVertexCount = (uint32_t)(positions.size() / 3);
        m_interactiveViewport.gridSegments[0] = segMajorStart;
        m_interactiveViewport.gridSegments[1] = segMajorCount;
        m_interactiveViewport.gridSegments[2] = segMinorStart;
        m_interactiveViewport.gridSegments[3] = segMinorCount;
        m_interactiveViewport.gridSegments[4] = segAxisUStart;
        m_interactiveViewport.gridSegments[5] = segAxisUCount;
        m_interactiveViewport.gridSegments[6] = segAxisVStart;
        m_interactiveViewport.gridSegments[7] = segAxisVCount;
        m_interactiveViewport.gridSegments[8] = segNegStart;
        m_interactiveViewport.gridSegments[9] = segNegCount;

        //if (segAxisVCount > 0) {
        //    std::string sample = "[Grid] +Z sample:";
        //    uint32_t vstart = segAxisVStart * 3;
        //    uint32_t vend = (std::min)((uint32_t)positions.size(), vstart + (std::min)(segAxisVCount, (uint32_t)6) * 3u);
        //    for (uint32_t vi = vstart; vi + 2 < vend; vi += 3) {
        //        sample += " (" + std::to_string(positions[vi]) + "," + std::to_string(positions[vi+1]) + "," + std::to_string(positions[vi+2]) + ")";
        //    }
        //    SCENE_LOG_INFO(sample);
        //}

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

        // Distance fade is measured IN-PLANE from the built minor-patch centre (not 3D camera
        // distance — that would wipe the whole grid in ortho, where the camera sits far away).
        Vec3 fadeAx, fadeAy;
        switch (m_interactiveViewport.gridBuiltPlane) {
            case 1:  fadeAx = Vec3(1, 0, 0); fadeAy = Vec3(0, 1, 0); break; // XY
            case 2:  fadeAx = Vec3(0, 0, 1); fadeAy = Vec3(0, 1, 0); break; // YZ
            default: fadeAx = Vec3(1, 0, 0); fadeAy = Vec3(0, 0, 1); break; // XZ
        }
        const Vec3 fadeCenter = fadeAx * m_interactiveViewport.gridBuiltCenterU +
                                fadeAy * m_interactiveViewport.gridBuiltCenterV;
        // Minor lines dissolve just inside their built extent so the geometric edge is never
        // visible; majors/axes fog out at the far horizon (kept below the >=20x viewScale
        // major-lattice edge guaranteed by requiredHalf above). The built extent caps the minor
        // band because shrinking grid_fade_distance doesn't shrink already-built geometry.
        const float gridOpacity = std::clamp(::render_settings.grid_opacity, 0.0f, 1.0f);
        const float builtCoarse = m_interactiveViewport.gridBuiltSpacing * 10.0f;
        const float minorGeomLimit = (std::max)(m_interactiveViewport.gridBuiltFineHalf - 2.0f * builtCoarse, builtCoarse);
        const float minorFadeEnd = (std::min)(minorGeomLimit, viewScale * 16.0f * gridFadeScale);
        const float minorFadeStart = minorFadeEnd * 0.45f;
        const float majorFadeStart = viewScale * 12.0f * gridFadeScale;
        const float majorFadeEnd = viewScale * 19.0f * gridFadeScale;

        auto drawSeg = [&](uint32_t first, uint32_t count, float r, float g, float b,
                           float fadeStart, float fadeEnd) {
            if (!count) return;
            SolidPushConstants gp{};
            matrixToGL(gridMvp, gp.viewProj);
            matrixToGL(identity, gp.view);
            gp.useMatcap = -1;
            gp.overrideR = r; gp.overrideG = g; gp.overrideB = b;
            gp.fadeCenterX = fadeCenter.x; gp.fadeCenterY = fadeCenter.y; gp.fadeCenterZ = fadeCenter.z;
            gp.fadeStart = fadeStart; gp.fadeEnd = fadeEnd;
            gp.overrideA = gridOpacity;
            vkCmdPushConstants(cmd, m_interactiveViewport.pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SolidPushConstants), &gp);
            vkCmdDraw(cmd, count, 1, first, 0);
        };

        if (gridOpacity > 0.01f) {
            const auto* segments = m_interactiveViewport.gridSegments;
            drawSeg(segments[2], segments[3], 0.38f, 0.38f, 0.38f, minorFadeStart, minorFadeEnd); // minor lattice
            drawSeg(segments[0], segments[1], 0.38f, 0.38f, 0.38f, majorFadeStart, majorFadeEnd); // major lattice
            drawSeg(segments[4], segments[5], 0.75f, 0.15f, 0.15f, majorFadeStart, majorFadeEnd); // +X axis
            // TEMP: brighten +Z axis for debugging
            drawSeg(segments[6], segments[7], 0.20f, 0.45f, 0.95f, majorFadeStart, majorFadeEnd); // +Z axis
            drawSeg(segments[8], segments[9], 0.30f, 0.30f, 0.30f, majorFadeStart, majorFadeEnd); // negative axes
        }
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

    // --- Particle billboard overlay (alpha first, then additive on top) ---
    {
        SolidPushConstants pp{};
        matrixToGL(viewProj, pp.viewProj);
        matrixToGL(view, pp.view);
        pp.useMatcap = 0;

        auto drawParticleGroup = [&](VkPipeline pipeline,
                                     const VulkanRT::BufferHandle& vbuf,
                                     uint32_t vcount) {
            if (pipeline == VK_NULL_HANDLE || !vbuf.buffer || vcount == 0) {
                return;
            }
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdPushConstants(cmd, m_interactiveViewport.pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(SolidPushConstants), &pp);
            VkBuffer pb = vbuf.buffer;
            VkDeviceSize pOff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &pb, &pOff);
            vkCmdDraw(cmd, vcount, 1, 0, 0);
        };

        drawParticleGroup(m_interactiveViewport.particleAlphaPipeline,
                          m_interactiveViewport.particleAlphaVertexBuffer,
                          m_interactiveViewport.particleAlphaVertexCount);
        drawParticleGroup(m_interactiveViewport.particleAddPipeline,
                          m_interactiveViewport.particleAddVertexBuffer,
                          m_interactiveViewport.particleAddVertexCount);
    }

    // --- Edit-mesh overlay (face fills -> edges -> vertex markers) ---
    {
        const Backend::EditMeshOverlayParams& eop = m_interactiveViewport.editOverlayParams;
        if (eop.enabled &&
            m_interactiveViewport.editOverlayPipelineLayout != VK_NULL_HANDLE &&
            m_interactiveViewport.editPositionBuffer.buffer &&
            m_interactiveViewport.editFlagBuffer.buffer &&
            m_interactiveViewport.editVertexCount > 0) {

            struct EditOverlayPushConstants {
                float mvp[16];
                float baseColor[4];
                float selectColor[4];
                float params[4];   // x=pointRadiusPx, y=viewportW, z=viewportH, w=mode
                float params2[4];  // x=depthBiasNDC, y=softHighlight, z/w reserved
            } epc{};

            const Matrix4x4 mvp = viewProj * eop.model;
            matrixToGL(mvp, epc.mvp);
            epc.params[0] = eop.pointRadiusPx;
            epc.params[1] = (float)width;
            epc.params[2] = (float)height;
            epc.params2[0] = eop.depthBias;
            epc.params2[2] = eop.xray ? 1.0f : 0.0f;

            VkBuffer eVbs[2] = {
                m_interactiveViewport.editPositionBuffer.buffer,
                m_interactiveViewport.editFlagBuffer.buffer
            };
            VkDeviceSize eOffs[2] = { 0, 0 };
            vkCmdBindVertexBuffers(cmd, 0, 2, eVbs, eOffs);

            // softAlphaBase mirrors the ImGui weightToColor alpha bases per
            // element type (faces 110/255, edges 230/255, points 235/255).
            auto pushEPC = [&](const float base[4], const float select[4], float mode, float softAlphaBase) {
                std::memcpy(epc.baseColor, base, sizeof(float) * 4);
                std::memcpy(epc.selectColor, select, sizeof(float) * 4);
                epc.params[3] = mode;
                epc.params2[1] = eop.softHighlight ? softAlphaBase : 0.0f;
                vkCmdPushConstants(cmd, m_interactiveViewport.editOverlayPipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(epc), &epc);
            };
            auto drawIndexed = [&](VkPipeline pipeline, const VulkanRT::BufferHandle& indexBuffer,
                                   uint32_t indexCount) {
                if (pipeline == VK_NULL_HANDLE || !indexBuffer.buffer || indexCount == 0) return;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
            };

            // Dim fill of every face (Face select mode), then selected faces on top.
            if (eop.drawFaces) {
                pushEPC(eop.faceColor, eop.selectColor, 0.0f, 0.43f);
                drawIndexed(m_interactiveViewport.editFacePipeline,
                            m_interactiveViewport.editFaceIndexBuffer,
                            m_interactiveViewport.editFaceIndexCount);
            }
            pushEPC(eop.selectFaceColor, eop.selectFaceColor, 1.0f, 0.0f);
            drawIndexed(m_interactiveViewport.editFacePipeline,
                        m_interactiveViewport.editSelFaceIndexBuffer,
                        m_interactiveViewport.editSelFaceIndexCount);

            // Wireframe, then selected edges on top.
            if (eop.drawEdges) {
                pushEPC(eop.edgeColor, eop.selectColor, 0.0f, 0.90f);
                drawIndexed(m_interactiveViewport.editLinePipeline,
                            m_interactiveViewport.editEdgeIndexBuffer,
                            m_interactiveViewport.editEdgeIndexCount);
            }
            pushEPC(eop.edgeColor, eop.selectColor, 1.0f, 0.0f);
            drawIndexed(m_interactiveViewport.editLinePipeline,
                        m_interactiveViewport.editSelEdgeIndexBuffer,
                        m_interactiveViewport.editSelEdgeIndexCount);

            // Vertex markers: one billboard disc instance per editable vertex.
            if (eop.drawPoints && m_interactiveViewport.editPointPipeline != VK_NULL_HANDLE) {
                pushEPC(eop.pointColor, eop.selectColor, 0.0f, 0.92f);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  m_interactiveViewport.editPointPipeline);
                vkCmdDraw(cmd, 6, m_interactiveViewport.editVertexCount, 0, 0);
            }
        }
    }

    vkCmdEndRenderPass(cmd);

    // ── Selection outline: mask pass (selected instances only) + fullscreen
    // edge composite over the finished color image. Render-pass dependencies
    // order main-pass depth → mask test → composite sampling.
    if (!selectionOutlineDraws.empty()) {
        const Backend::SelectionOutlineParams& sop = m_interactiveViewport.selectionOutlineParams;

        recordSelectionOutlineMaskPass(cmd, selectionOutlineDraws, viewProj, width, height);

        VkRenderPassBeginInfo compRPBI{};
        compRPBI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        compRPBI.renderPass = m_interactiveViewport.selectionCompositeRenderPass;
        compRPBI.framebuffer = m_interactiveViewport.selectionCompositeFramebuffer;
        compRPBI.renderArea.offset = { 0, 0 };
        compRPBI.renderArea.extent = { (uint32_t)width, (uint32_t)height };
        vkCmdBeginRenderPass(cmd, &compRPBI, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          m_interactiveViewport.selectionCompositePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_interactiveViewport.selectionCompositePipelineLayout,
                                0, 1, &m_interactiveViewport.selectionCompositeDescSet, 0, nullptr);

        struct SelectionOutlinePush {
            float primary[4];
            float secondary[4];
            float occluded[4];
            float params[4];
        };
        SelectionOutlinePush compPush{};
        std::memcpy(compPush.primary, sop.primaryColor, sizeof(compPush.primary));
        std::memcpy(compPush.secondary, sop.secondaryColor, sizeof(compPush.secondary));
        std::memcpy(compPush.occluded, sop.occludedColor, sizeof(compPush.occluded));
        compPush.params[0] = sop.thicknessPx;
        compPush.params[1] = (width > 0) ? (1.0f / (float)width) : 0.0f;
        compPush.params[2] = (height > 0) ? (1.0f / (float)height) : 0.0f;
        compPush.params[3] = 0.0f;
        vkCmdPushConstants(cmd, m_interactiveViewport.selectionCompositePipelineLayout,
                           VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SelectionOutlinePush), &compPush);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
    }

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

    const size_t numTriangles = triangles.size();
    #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static)
    for (int t = 0; t < (int)numTriangles; ++t) {
        const auto& tri = triangles[t];
        if (!tri) continue;

        const size_t local_idx = t * 9;
        const size_t local_uvIdx = t * 6;
        const size_t local_matIdx = t * 3;

        const bool hasSharedTransform = (tri->getTransformPtr() != nullptr);
        
        Vec2 triUvs[3] = { Vec2(0, 0), Vec2(0, 0), Vec2(0, 0) };
        uint32_t triMatId = 0;
        Vec3 verts[3];
        Vec3 norms[3];
        bool resolved = false;

        if (tri->parentMesh && tri->parentMesh->geometry) {
            TriangleMesh* parentMesh = tri->parentMesh.get();
            const Vec3* cachedPositions = parentMesh->geometry->get_attribute_data<Vec3>("P");
            const Vec3* cachedNormals = parentMesh->geometry->get_attribute_data<Vec3>("N");
            const Vec3* cachedOrigPositions = parentMesh->geometry->get_attribute_data<Vec3>("P_orig");
            const Vec3* cachedOrigNormals = parentMesh->geometry->get_attribute_data<Vec3>("N_orig");
            const Vec2* cachedUvs = parentMesh->geometry->get_attribute_data<Vec2>("uv");
            const uint16_t* cachedMatIDs = parentMesh->geometry->get_attribute_data<uint16_t>("materialID");
            const std::vector<uint32_t, DNA::AlignedAllocator<uint32_t, 32>>* cachedIndices = &parentMesh->geometry->indices;

            if (cachedPositions && cachedIndices && !cachedIndices->empty()) {
                uint32_t faceIdx = tri->faceIndex;
                uint32_t baseIdx = faceIdx * 3;
                if (baseIdx + 2 < cachedIndices->size()) {
                    uint32_t i0 = (*cachedIndices)[baseIdx + 0];
                    uint32_t i1 = (*cachedIndices)[baseIdx + 1];
                    uint32_t i2 = (*cachedIndices)[baseIdx + 2];

                    triUvs[0] = cachedUvs ? cachedUvs[i0] : Vec2(0, 0);
                    triUvs[1] = cachedUvs ? cachedUvs[i1] : Vec2(0, 0);
                    triUvs[2] = cachedUvs ? cachedUvs[i2] : Vec2(0, 0);

                    triMatId = cachedMatIDs ? cachedMatIDs[i0] : static_cast<uint32_t>(tri->getMaterialID());

                    if (hasSharedTransform) {
                        if (tri->hasSkinData()) {
                            verts[0] = tri->getOriginalVertexPosition(0);
                            verts[1] = tri->getOriginalVertexPosition(1);
                            verts[2] = tri->getOriginalVertexPosition(2);
                        } else {
                            verts[0] = cachedOrigPositions ? cachedOrigPositions[i0] : cachedPositions[i0];
                            verts[1] = cachedOrigPositions ? cachedOrigPositions[i1] : cachedPositions[i1];
                            verts[2] = cachedOrigPositions ? cachedOrigPositions[i2] : cachedPositions[i2];
                        }
                        norms[0] = cachedOrigNormals ? cachedOrigNormals[i0] : (cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0));
                        norms[1] = cachedOrigNormals ? cachedOrigNormals[i1] : (cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0));
                        norms[2] = cachedOrigNormals ? cachedOrigNormals[i2] : (cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0));
                    } else {
                        verts[0] = cachedPositions[i0];
                        verts[1] = cachedPositions[i1];
                        verts[2] = cachedPositions[i2];
                        norms[0] = cachedOrigNormals ? cachedOrigNormals[i0] : (cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0));
                        norms[1] = cachedOrigNormals ? cachedOrigNormals[i1] : (cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0));
                        norms[2] = cachedOrigNormals ? cachedOrigNormals[i2] : (cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0));
                    }
                    resolved = true;
                }
            }
        }

        if (!resolved) {
            auto [uv0, uv1, uv2] = tri->getUVCoordinates();
            triUvs[0] = uv0; triUvs[1] = uv1; triUvs[2] = uv2;
            triMatId = static_cast<uint32_t>(tri->getMaterialID());
            for (int v = 0; v < 3; ++v) {
                verts[v] = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
                norms[v] = tri->getOriginalVertexNormal(v);
            }
        }

        for (int v = 0; v < 3; ++v) {
            newPositions[local_idx + v * 3 + 0] = verts[v].x;
            newPositions[local_idx + v * 3 + 1] = verts[v].y;
            newPositions[local_idx + v * 3 + 2] = verts[v].z;

            newNormals[local_idx + v * 3 + 0] = norms[v].x;
            newNormals[local_idx + v * 3 + 1] = norms[v].y;
            newNormals[local_idx + v * 3 + 2] = norms[v].z;

            newUVs[local_uvIdx + v * 2 + 0] = triUvs[v].x;
            newUVs[local_uvIdx + v * 2 + 1] = triUvs[v].y;

            newMatIds[local_matIdx + v] = triMatId;
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

bool VulkanViewportBackend::updateRasterMeshFromMeshSoA(const std::string& nodeName,
                                                        const TriangleMesh* mesh) {
    if (!m_device || !m_device->isInitialized() || !mesh || !mesh->geometry) return false;
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

    const DNA::GeometryDetail* g = mesh->geometry.get();
    const auto& indices = g->indices;
    const size_t numFaces = indices.size() / 3;
    const size_t vertCount = numFaces * 3;
    const size_t floatCount = vertCount * 3;
    const size_t uvFloatCount = vertCount * 2;
    // The raster buffer was built 3N (non-indexed) from this same SoA, so the slot count must
    // match for an in-place refit; a structural change goes through a full buildRasterGeometry.
    if (vertCount != rmb.vertexCount) return false;

    const Vec3* Porig = g->get_attribute_data<Vec3>("P_orig");
    if (!Porig) Porig = g->get_attribute_data<Vec3>("P");
    const Vec3* Norig = g->get_attribute_data<Vec3>("N_orig");
    if (!Norig) Norig = g->get_attribute_data<Vec3>("N");
    const Vec2* UV = g->get_attribute_data<Vec2>("uv");
    const uint16_t* matIDs = g->get_attribute_data<uint16_t>("materialID");
    if (!Porig) return false;

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

    // Flat raster verts are LOCAL (P_orig) — the raster instance carries the mesh transform — so
    // this mirrors updateRasterMeshFromTriangles' shared-transform branch.
    #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static)
    for (int f = 0; f < (int)numFaces; ++f) {
        for (int c = 0; c < 3; ++c) {
            const uint32_t soaV = indices[static_cast<size_t>(f) * 3 + c];
            const Vec3 p = Porig[soaV];
            const Vec3 n = Norig ? Norig[soaV] : Vec3(0.0f, 1.0f, 0.0f);
            const Vec2 uv = UV ? UV[soaV] : Vec2(0.0f, 0.0f);
            const uint32_t m = matIDs ? static_cast<uint32_t>(matIDs[soaV]) : 0u;
            const size_t vbase = (static_cast<size_t>(f) * 3 + c) * 3;
            const size_t uvbase = (static_cast<size_t>(f) * 3 + c) * 2;
            const size_t mbase = static_cast<size_t>(f) * 3 + c;
            newPositions[vbase + 0] = p.x; newPositions[vbase + 1] = p.y; newPositions[vbase + 2] = p.z;
            newNormals[vbase + 0] = n.x;   newNormals[vbase + 1] = n.y;   newNormals[vbase + 2] = n.z;
            newUVs[uvbase + 0] = uv.x;     newUVs[uvbase + 1] = uv.y;
            newMatIds[mbase] = m;
        }
    }

    // Same upload path as updateRasterMeshFromTriangles: when the slot count matches and a CPU
    // mirror exists, only the changed vertex range is re-uploaded (realtime per-dab), else a full
    // re-upload. Sizes always match here (early-out above), so the incremental branch is taken.
    if (rmb.cpuPositions.size() == floatCount) {
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
        if (rmb.vertexBuffer.buffer) m_device->uploadBuffer(rmb.vertexBuffer, newPositions.data(), floatCount * sizeof(float));
        if (rmb.normalBuffer.buffer) m_device->uploadBuffer(rmb.normalBuffer, newNormals.data(), floatCount * sizeof(float));
        if (rmb.uvBuffer.buffer) m_device->uploadBuffer(rmb.uvBuffer, newUVs.data(), uvFloatCount * sizeof(float), 0);
        if (rmb.matIdBuffer.buffer) m_device->uploadBuffer(rmb.matIdBuffer, newMatIds.data(), newMatIds.size() * sizeof(uint32_t), 0);
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

    TriangleMesh* lastParentMesh = nullptr;
    const Vec3* cachedPositions = nullptr;
    const Vec3* cachedNormals = nullptr;
    const Vec3* cachedOrigPositions = nullptr;
    const Vec3* cachedOrigNormals = nullptr;
    const std::vector<uint32_t, DNA::AlignedAllocator<uint32_t, 32>>* cachedIndices = nullptr;
    bool hasGeometry = false;

    for (const size_t triIdx : sortedDirty) {
        if (triIdx >= meshEntries.size()) continue;
        const auto& tri = meshEntries[triIdx].second;
        if (!tri) continue;

        const size_t baseFloat = triIdx * 9;
        if (baseFloat + 8 >= expectedFloatCount) continue;

        const bool hasSharedTransform = (tri->getTransformPtr() != nullptr);
        
        Vec3 verts[3];
        Vec3 norms[3];
        bool resolved = false;

        if (tri->parentMesh) {
            if (tri->parentMesh.get() != lastParentMesh) {
                lastParentMesh = tri->parentMesh.get();
                if (lastParentMesh->geometry) {
                    cachedPositions = lastParentMesh->geometry->get_attribute_data<Vec3>("P");
                    cachedNormals = lastParentMesh->geometry->get_attribute_data<Vec3>("N");
                    cachedOrigPositions = lastParentMesh->geometry->get_attribute_data<Vec3>("P_orig");
                    cachedOrigNormals = lastParentMesh->geometry->get_attribute_data<Vec3>("N_orig");
                    cachedIndices = &lastParentMesh->geometry->indices;
                    hasGeometry = (cachedPositions != nullptr) && (cachedIndices != nullptr) && (!cachedIndices->empty());
                } else {
                    hasGeometry = false;
                }
            }

            if (hasGeometry) {
                uint32_t faceIdx = tri->faceIndex;
                uint32_t baseIdx = faceIdx * 3;
                uint32_t i0 = (*cachedIndices)[baseIdx + 0];
                uint32_t i1 = (*cachedIndices)[baseIdx + 1];
                uint32_t i2 = (*cachedIndices)[baseIdx + 2];

                if (hasSharedTransform) {
                    if (tri->hasSkinData()) {
                        verts[0] = tri->getOriginalVertexPosition(0);
                        verts[1] = tri->getOriginalVertexPosition(1);
                        verts[2] = tri->getOriginalVertexPosition(2);
                    } else {
                        verts[0] = cachedOrigPositions ? cachedOrigPositions[i0] : cachedPositions[i0];
                        verts[1] = cachedOrigPositions ? cachedOrigPositions[i1] : cachedPositions[i1];
                        verts[2] = cachedOrigPositions ? cachedOrigPositions[i2] : cachedPositions[i2];
                    }
                    norms[0] = cachedOrigNormals ? cachedOrigNormals[i0] : (cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0));
                    norms[1] = cachedOrigNormals ? cachedOrigNormals[i1] : (cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0));
                    norms[2] = cachedOrigNormals ? cachedOrigNormals[i2] : (cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0));
                } else {
                    verts[0] = cachedPositions[i0];
                    verts[1] = cachedPositions[i1];
                    verts[2] = cachedPositions[i2];
                    norms[0] = cachedOrigNormals ? cachedOrigNormals[i0] : (cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0));
                    norms[1] = cachedOrigNormals ? cachedOrigNormals[i1] : (cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0));
                    norms[2] = cachedOrigNormals ? cachedOrigNormals[i2] : (cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0));
                }
                resolved = true;
            }
        }

        if (!resolved) {
            for (int v = 0; v < 3; ++v) {
                verts[v] = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
                norms[v] = tri->getOriginalVertexNormal(v);
            }
        }

        for (int v = 0; v < 3; ++v) {
            const size_t idx = baseFloat + static_cast<size_t>(v) * 3;
            rmb.cpuPositions[idx]     = verts[v].x;
            rmb.cpuPositions[idx + 1] = verts[v].y;
            rmb.cpuPositions[idx + 2] = verts[v].z;
            rmb.cpuNormals[idx]       = norms[v].x;
            rmb.cpuNormals[idx + 1]   = norms[v].y;
            rmb.cpuNormals[idx + 2]   = norms[v].z;
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
        clone.scatterGroupId = -1;
        clone.scatterInstanceIndex = UINT32_MAX;
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
          /*  SCENE_LOG_INFO("[ViewportRaster] buildRasterGeometry early-out: cache valid. gen=" +
                           std::to_string(curGen) + " meshes=" + std::to_string(m_rasterMeshes.size()) +
                           " objects=" + std::to_string(objects.size()));*/
            m_rasterGeometryDirty = false;
            return;
        }
       /* SCENE_LOG_INFO("[ViewportRaster] buildRasterGeometry starting: gen " +
                       std::to_string(prevGen) + " -> " + std::to_string(curGen) +
                       " objects=" + std::to_string(objects.size()) +
                       " prevMeshes=" + std::to_string(m_rasterMeshes.size()));*/
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

    auto buildSkinBuffersForTrianglePtrs = [](const std::vector<const Triangle*>& triangles,
                                              uint32_t vertexCount,
                                              std::vector<int32_t>& boneIndices,
                                              std::vector<float>& boneWeights) {
        boneIndices.assign(static_cast<size_t>(vertexCount) * 4, -1);
        boneWeights.assign(static_cast<size_t>(vertexCount) * 4, 0.0f);

        auto fillRange = [&triangles, &boneIndices, &boneWeights](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                const Triangle* tri = triangles[i];
                if (!tri) continue;
                for (int v = 0; v < 3; ++v) {
                    const auto& weights = tri->getSkinBoneWeights(v);
                    const size_t base = (i * 3 + static_cast<size_t>(v)) * 4;
                    for (int w = 0; w < 4; ++w) {
                        if (w < static_cast<int>(weights.size())) {
                            boneIndices[base + static_cast<size_t>(w)] = weights[w].first;
                            boneWeights[base + static_cast<size_t>(w)] = weights[w].second;
                        }
                    }
                }
            }
        };

        constexpr size_t kSkinParallelThreshold = 4096;
        unsigned skinThreads = std::thread::hardware_concurrency();
        if (skinThreads == 0) skinThreads = 4;

        const size_t count = triangles.size();
        if (count < kSkinParallelThreshold || skinThreads < 2) {
            fillRange(0, count);
        } else {
            const size_t chunk = (count + skinThreads - 1) / skinThreads;
            std::vector<std::future<void>> futures;
            futures.reserve(skinThreads);
            for (unsigned t = 0; t < skinThreads; ++t) {
                const size_t s = t * chunk;
                const size_t e = (std::min)(s + chunk, count);
                if (s >= e) break;
                futures.push_back(std::async(std::launch::async, fillRange, s, e));
            }
            for (auto& f : futures) f.get();
        }
    };

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
        for (const Triangle* tri : valid) {
            if (tri && tri->hasSkinData()) {
                hasSkinning = true;
                break;
            }
        }
        if (hasSkinning) {
            std::vector<int32_t> boneIndices;
            std::vector<float> boneWeights;
            buildSkinBuffersForTrianglePtrs(valid, rmb.vertexCount, boneIndices, boneWeights);

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

    // Pointer-based overload: accept a list of raw Triangle pointers to avoid
    // copying/shared_ptr refcount increments when grouping instance source meshes.
    auto ensureRasterMeshForTrianglePtrs = [&](const std::string& meshKey,
                                               const std::vector<const Triangle*>& triangles) {
        if (triangles.empty()) return;
        if (m_rasterMeshes.find(meshKey) != m_rasterMeshes.end()) return;

        std::vector<const Triangle*> valid;
        valid.reserve(triangles.size());
        for (const Triangle* t : triangles) if (t) valid.push_back(t);
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

        // Cache bbox and upload buffers (same as shared_ptr variant)
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

        VulkanRT::BufferCreateInfo uci{};
        uci.size = uvs.size() * sizeof(float);
        uci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        uci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        uci.initialData = nullptr;
        rmb.uvBuffer = m_device->createBuffer(uci);

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
        for (const Triangle* tri : valid) {
            if (tri && tri->hasSkinData()) { hasSkinning = true; break; }
        }
        if (hasSkinning) {
            std::vector<int32_t> boneIndices;
            std::vector<float> boneWeights;
            buildSkinBuffersForTrianglePtrs(valid, rmb.vertexCount, boneIndices, boneWeights);

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
        std::vector<int32_t> boneIndices;
        std::vector<float> boneWeights;
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

            // Group by node name using raw Triangle* to avoid incrementing shared_ptr refcounts
            std::unordered_map<std::string, std::vector<const Triangle*>> trianglesByNode;
            trianglesByNode.reserve(inst->source_triangles->size());
            for (const auto& tri_sp : *inst->source_triangles) {
                if (!tri_sp) continue;
                const Triangle* tri = tri_sp.get();
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

                // Call pointer-based mesh builder to avoid refcount churn
                ensureRasterMeshForTrianglePtrs(meshKey, groupedTriangles);

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

            Transform* triTransformPtr = tri->getTransformPtr();
            const bool hasSharedTransform = (triTransformPtr != nullptr);
            std::string nodeName = tri->getNodeName();
            if (nodeName.empty()) nodeName = "[Solo-" + std::to_string(groups.size()) + "]";
            const uintptr_t transformKey = triTransformPtr
                ? reinterpret_cast<uintptr_t>(triTransformPtr)
                : reinterpret_cast<uintptr_t>(tri.get());
            const std::string groupKey = nodeName + "#th=" + std::to_string(transformKey);

            auto found = groupByKey.find(groupKey);
            if (found == groupByKey.end()) {
                RasterTriGroup g;
                // Keyed by groupKey (nodeName + transform), not just nodeName: multi-material
                // imports create several sibling objects sharing one nodeName (one per material),
                // and a plain nodeName key would collide in m_rasterMeshes, silently overwriting
                // all but the last material's raster buffer (Solid view showed only one material
                // while Vulkan RT, which keys BLAS per-object-pointer, rendered fine).
                g.meshKey = "[Raster-Solo]-" + groupKey;
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
        // Flat/proxy flip: a dense mesh placed directly in world.objects as a TriangleMesh (no
        // facades). Expand its SoA faces into a raster group (local P_orig + getFinal() transform,
        // matching the facade path above) so Solid/Matcap viewport shows it.
        else if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (!mesh->visible || !mesh->geometry || mesh->geometry->indices.empty()) return;
            const auto* geom = mesh->geometry.get();
            const Vec3* Po = geom->get_positions_orig();
            const Vec3* No = geom->get_normals_orig();
            const Vec2* UV = geom->get_uvs();
            const uint16_t* M = geom->get_material_ids();
            if (!Po) Po = geom->get_positions();
            if (!No) No = geom->get_normals();
            if (!Po) return;

            std::string nodeName = mesh->nodeName.empty() ? ("[Solo-" + std::to_string(groups.size()) + "]") : mesh->nodeName;
            const std::string groupKey = nodeName + "#mesh=" + std::to_string(reinterpret_cast<uintptr_t>(mesh.get()));
            auto found = groupByKey.find(groupKey);
            if (found == groupByKey.end()) {
                RasterTriGroup g;
                // See the facade-Triangle branch above: key by groupKey (includes the mesh
                // pointer), not bare nodeName, so sibling per-material TriangleMesh objects with
                // the same nodeName don't collide and overwrite each other in m_rasterMeshes.
                g.meshKey = "[Raster-Solo]-" + groupKey;
                g.nodeName = nodeName;
                g.transform = mesh->transform ? mesh->transform->getFinal() : Matrix4x4::identity();
                groups.push_back(std::move(g));
                found = groupByKey.emplace(groupKey, groups.size() - 1).first;
            }
            auto& grp = groups[found->second];
            const auto& idx = geom->indices;
            const size_t triCount = idx.size() / 3;
            grp.positions.reserve(grp.positions.size() + triCount * 9);
            grp.normals.reserve(grp.normals.size() + triCount * 9);
            grp.uvs.reserve(grp.uvs.size() + triCount * 6);
            grp.matIds.reserve(grp.matIds.size() + triCount * 3);
            const bool meshHasSkinning = mesh->hasSkinWeights();
            if (meshHasSkinning) {
                grp.boneIndices.reserve(grp.boneIndices.size() + triCount * 12);
                grp.boneWeights.reserve(grp.boneWeights.size() + triCount * 12);
            }
            for (size_t f = 0; f < triCount; ++f) {
                for (int c = 0; c < 3; ++c) {
                    const uint32_t vi = idx[f * 3 + c];
                    const Vec3& p = Po[vi];
                    const Vec3 n = No ? No[vi] : Vec3(0.0f, 1.0f, 0.0f);
                    const Vec2 uv = UV ? UV[vi] : Vec2(0.0f, 0.0f);
                    grp.positions.push_back(p.x); grp.positions.push_back(p.y); grp.positions.push_back(p.z);
                    grp.normals.push_back(n.x); grp.normals.push_back(n.y); grp.normals.push_back(n.z);
                    grp.uvs.push_back(uv.x); grp.uvs.push_back(uv.y);
                    grp.matIds.push_back(M ? static_cast<uint32_t>(M[vi]) : 0u);
                    if (meshHasSkinning) {
                        const auto* influences = vi < geom->skin_weights.size() ? &geom->skin_weights[vi] : nullptr;
                        for (size_t influence = 0; influence < 4; ++influence) {
                            grp.boneIndices.push_back(influences && influence < influences->size() ? (*influences)[influence].first : -1);
                            grp.boneWeights.push_back(influences && influence < influences->size() ? (*influences)[influence].second : 0.0f);
                        }
                    }
                }
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

        bool hasSkinning = !grp.boneIndices.empty();
        for (const auto& tri : grp.triangles) {
            if (tri && tri->hasSkinData()) {
                hasSkinning = true;
                break;
            }
        }
        if (hasSkinning) {
            std::vector<int32_t> boneIndices = std::move(grp.boneIndices);
            std::vector<float> boneWeights = std::move(grp.boneWeights);
            if (boneIndices.empty()) {
                std::vector<const Triangle*> validSkinTriangles;
                validSkinTriangles.reserve(grp.triangles.size());
                for (const auto& tri : grp.triangles) {
                    if (tri) validSkinTriangles.push_back(tri.get());
                }
                buildSkinBuffersForTrianglePtrs(validSkinTriangles, rmb.vertexCount, boneIndices, boneWeights);
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

        rmb.cpuPositions = std::move(grp.positions);
        rmb.cpuNormals   = std::move(grp.normals);
        rmb.cpuMatIds    = std::move(grp.matIds);

        m_rasterMeshes[grp.meshKey] = std::move(rmb);

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
                ri.scatterGroupId = group.id;
                ri.scatterInstanceIndex = static_cast<uint32_t>(i);
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
    m_hasPresentedRenderedFrame = false;
    m_lastCameraHash = 0;

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
    transformMap.reserve(objects.size());

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
            auto th = tri->getTransformPtr();
            std::string name = tri->getNodeName();
            if (!name.empty() && th) {
                transformMap[name] = tri->getTransformMatrix();
            }
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            // Flat (direct SoA) mesh: drives its world transform through its own handle. Without this
            // the Solid/Matcap viewport never refreshed a keyframed/physics-driven flat mesh per
            // frame — it froze during playback (RT updated correctly, raster did not).
            if (!tm->nodeName.empty() && tm->transform) {
                transformMap[tm->nodeName] = tm->transform->getFinal();
            }
        }
    };

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

    for (size_t i = 0; i < baseObjectCount; ++i) {
        collectTransforms(objects[i]);
    }

    const auto& instanceGroups = InstanceManager::getInstance().getGroups();
    std::unordered_map<int, const InstanceGroup*> scatterGroupsById;
    scatterGroupsById.reserve(instanceGroups.size());
    for (const auto& group : instanceGroups) {
        if (!group.instances.empty()) {
            scatterGroupsById.emplace(group.id, &group);
        }
    }

    bool changed = false;
    std::unordered_set<std::string> dirtyMeshKeys;
    const size_t kParallelThreshold = 2048;
    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    auto syncRange = [this, &transformMap, &scatterGroupsById]
                     (size_t start, size_t end) {
        std::unordered_set<std::string> localDirty;
        for (size_t i = start; i < end; ++i) {
            auto& ri = m_rasterInstances[i];
            Matrix4x4 newTransform;
            bool hasTransform = false;

            if (ri.scatterGroupId >= 0 && ri.scatterInstanceIndex != UINT32_MAX) {
                auto groupIt = scatterGroupsById.find(ri.scatterGroupId);
                if (groupIt != scatterGroupsById.end()) {
                    const auto* group = groupIt->second;
                    if (ri.scatterInstanceIndex < group->instances.size()) {
                        newTransform = group->instances[ri.scatterInstanceIndex].toMatrix();
                        hasTransform = true;
                    }
                }
            } else {
                auto it = transformMap.find(ri.nodeName);
                if (it != transformMap.end()) {
                    newTransform = it->second;
                    hasTransform = true;
                }
            }

            if (hasTransform && !(ri.transform == newTransform)) {
                ri.transform = newTransform;
                updateRasterInstanceWorldBBox(ri);
                localDirty.insert(ri.meshKey);
            }
        }
        return localDirty;
    };

    if (m_rasterInstances.size() < kParallelThreshold || numThreads < 2) {
        dirtyMeshKeys = syncRange(0, m_rasterInstances.size());
    } else {
        const size_t chunk = (m_rasterInstances.size() + numThreads - 1) / numThreads;
        std::vector<std::future<std::unordered_set<std::string>>> futures;
        futures.reserve(numThreads);
        for (unsigned t = 0; t < numThreads; ++t) {
            const size_t s = t * chunk;
            const size_t e = (std::min)(s + chunk, m_rasterInstances.size());
            if (s >= e) break;
            futures.push_back(std::async(std::launch::async, syncRange, s, e));
        }
        for (auto& f : futures) {
            auto localDirty = f.get();
            dirtyMeshKeys.insert(localDirty.begin(), localDirty.end());
        }
    }
    changed = !dirtyMeshKeys.empty();

    if (changed) {
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
        std::shared_ptr<TriangleMesh> mesh;
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
            auto th = tri->getTransformPtr();
            void* groupKey = th ? (void*)th : (void*)tri.get();
            auto& grp = skinnedGroups[groupKey];
            if (grp.meshKey.empty()) {
                std::string nodeName = tri->getNodeName();
                auto it = nodeToMeshKey.find(nodeName);
                if (it != nodeToMeshKey.end()) {
                    grp.meshKey = it->second;
                }
            }
            grp.triangles.push_back(tri);
        } else if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (!mesh->visible || !mesh->hasSkinWeights()) return;
            void* groupKey = mesh->transform ? static_cast<void*>(mesh->transform.get()) : static_cast<void*>(mesh.get());
            auto& grp = skinnedGroups[groupKey];
            auto it = nodeToMeshKey.find(mesh->nodeName);
            if (it != nodeToMeshKey.end()) grp.meshKey = it->second;
            grp.mesh = mesh;
        }
    };
    for (const auto& obj : objects) collectSkinned(obj);

    if (skinnedGroups.empty()) return;

    for (auto& [key, grp] : skinnedGroups) {
        if (grp.meshKey.empty() || (grp.triangles.empty() && !grp.mesh)) continue;
        auto meshIt = m_rasterMeshes.find(grp.meshKey);
        if (meshIt == m_rasterMeshes.end()) continue;

        auto& rmb = meshIt->second;
        const size_t vertCount = grp.mesh && grp.mesh->geometry
            ? grp.mesh->geometry->indices.size()
            : grp.triangles.size() * 3;
        const size_t floatCount = vertCount * 3;
        if (rmb.vertexCount != static_cast<uint32_t>(vertCount)) continue;

        std::vector<float> newPositions(floatCount);
        std::vector<float> newNormals(floatCount);
        size_t idx = 0;

        if (grp.mesh && grp.mesh->geometry) {
            grp.mesh->applySkinning(boneMatrices);
            const Vec3* positions = grp.mesh->geometry->get_positions();
            const Vec3* normals = grp.mesh->geometry->get_normals();
            for (uint32_t vertexIndex : grp.mesh->geometry->indices) {
                const Vec3 position = positions ? positions[vertexIndex] : Vec3(0.0f);
                const Vec3 normal = normals ? normals[vertexIndex] : Vec3(0.0f, 1.0f, 0.0f);
                newPositions[idx] = position.x; newPositions[idx + 1] = position.y; newPositions[idx + 2] = position.z;
                newNormals[idx] = normal.x; newNormals[idx + 1] = normal.y; newNormals[idx + 2] = normal.z;
                idx += 3;
            }
        } else {
            for (const auto& tri : grp.triangles) {
                for (int v = 0; v < 3; ++v) {
                    Vec3 p = tri->apply_bone_to_vertex(v, boneMatrices);
                    Vec3 n = tri->apply_bone_to_normal(
                        tri->getOriginalVertexNormal(v),
                        tri->getSkinBoneWeights(v),
                        boneMatrices);
                    newPositions[idx] = p.x; newPositions[idx + 1] = p.y; newPositions[idx + 2] = p.z;
                    newNormals[idx] = n.x; newNormals[idx + 1] = n.y; newNormals[idx + 2] = n.z;
                    idx += 3;
                }
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

void VulkanBackendAdapter::uploadParticleBillboards(const std::vector<float>& addData, uint32_t addVertexCount,
                                                    const std::vector<float>& alphaData, uint32_t alphaVertexCount) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;

    auto uploadGroup = [&](const std::vector<float>& data, uint32_t vertexCount,
                           VulkanRT::BufferHandle& buffer, uint32_t& countOut) {
        if (data.empty() || vertexCount == 0) {
            countOut = 0;
            return;
        }
        const uint64_t byteSize = data.size() * sizeof(float);
        // Reallocate only when current buffer is too small.
        if (!buffer.buffer || buffer.size < byteSize) {
            if (buffer.buffer) {
                m_device->destroyBuffer(buffer);
            }
            VulkanRT::BufferCreateInfo bci{};
            bci.size     = byteSize;
            bci.usage    = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            bci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            buffer = m_device->createBuffer(bci);
        }
        if (buffer.buffer) {
            m_device->uploadBuffer(buffer, data.data(), byteSize, 0);
            countOut = vertexCount;
        } else {
            countOut = 0;
        }
    };

    const uint32_t prevAdd = m_interactiveViewport.particleAddVertexCount;
    const uint32_t prevAlpha = m_interactiveViewport.particleAlphaVertexCount;

    uploadGroup(addData, addVertexCount,
                m_interactiveViewport.particleAddVertexBuffer,
                m_interactiveViewport.particleAddVertexCount);
    uploadGroup(alphaData, alphaVertexCount,
                m_interactiveViewport.particleAlphaVertexBuffer,
                m_interactiveViewport.particleAlphaVertexCount);

    // Re-render only when there is something to show or something just cleared.
    if (m_interactiveViewport.particleAddVertexCount || m_interactiveViewport.particleAlphaVertexCount ||
        prevAdd || prevAlpha) {
        m_interactiveViewport.dirty = true;
        m_currentSamples = 0;
    }
}

// ============================================================================
// Edit-Mesh Overlay (raster viewport GPU wireframe / vertex / face passes)
// ============================================================================

namespace {
// Grow-only upload shared by every edit-overlay buffer: reallocates only when
// the existing buffer is too small, otherwise writes in place.
template <typename T>
void uploadEditOverlayBuffer(VulkanRT::VulkanDevice* device,
                             VulkanRT::BufferHandle& buffer,
                             const std::vector<T>& data,
                             VulkanRT::BufferUsage usage) {
    if (data.empty()) {
        return;
    }
    const uint64_t byteSize = data.size() * sizeof(T);
    if (!buffer.buffer || buffer.size < byteSize) {
        if (buffer.buffer) {
            device->destroyBuffer(buffer);
        }
        VulkanRT::BufferCreateInfo bci{};
        bci.size     = byteSize;
        bci.usage    = usage | VulkanRT::BufferUsage::TRANSFER_DST;
        // Use CPU_TO_GPU memory for frequently-updated overlay buffers to map and copy directly.
        // This avoids staging buffer allocation, command recording, queue submission, and fence waits
        // on the main thread, completely preventing OpenMP / Vulkan queue deadlocks.
        bci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
        buffer = device->createBuffer(bci);
    }
    if (buffer.buffer) {
        device->uploadBuffer(buffer, data.data(), byteSize, 0);
    }
}
} // namespace

void VulkanBackendAdapter::uploadEditMeshOverlayGeometry(const std::vector<float>& positions, uint32_t vertexCount) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;

    if (positions.empty() || vertexCount == 0) {
        m_interactiveViewport.editVertexCount = 0;
        m_interactiveViewport.dirty = true;
        return;
    }
    uploadEditOverlayBuffer(m_device.get(), m_interactiveViewport.editPositionBuffer,
                            positions, VulkanRT::BufferUsage::VERTEX);
    m_interactiveViewport.editVertexCount =
        m_interactiveViewport.editPositionBuffer.buffer ? vertexCount : 0;
    m_interactiveViewport.dirty = true;
    m_currentSamples = 0;
}

void VulkanBackendAdapter::uploadEditMeshOverlayFlags(const std::vector<uint32_t>& flags) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;

    uploadEditOverlayBuffer(m_device.get(), m_interactiveViewport.editFlagBuffer,
                            flags, VulkanRT::BufferUsage::VERTEX);
    m_interactiveViewport.dirty = true;
    m_currentSamples = 0;
}

void VulkanBackendAdapter::uploadEditMeshOverlayTopology(const std::vector<uint32_t>& edgeIndices,
                                                         const std::vector<uint32_t>& faceIndices) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;

    uploadEditOverlayBuffer(m_device.get(), m_interactiveViewport.editEdgeIndexBuffer,
                            edgeIndices, VulkanRT::BufferUsage::INDEX);
    m_interactiveViewport.editEdgeIndexCount =
        m_interactiveViewport.editEdgeIndexBuffer.buffer ? (uint32_t)edgeIndices.size() : 0;

    uploadEditOverlayBuffer(m_device.get(), m_interactiveViewport.editFaceIndexBuffer,
                            faceIndices, VulkanRT::BufferUsage::INDEX);
    m_interactiveViewport.editFaceIndexCount =
        m_interactiveViewport.editFaceIndexBuffer.buffer ? (uint32_t)faceIndices.size() : 0;

    m_interactiveViewport.dirty = true;
    m_currentSamples = 0;
}

void VulkanBackendAdapter::uploadEditMeshOverlaySelectionIndices(const std::vector<uint32_t>& selEdgeIndices,
                                                                 const std::vector<uint32_t>& selFaceIndices) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;

    uploadEditOverlayBuffer(m_device.get(), m_interactiveViewport.editSelEdgeIndexBuffer,
                            selEdgeIndices, VulkanRT::BufferUsage::INDEX);
    m_interactiveViewport.editSelEdgeIndexCount =
        m_interactiveViewport.editSelEdgeIndexBuffer.buffer ? (uint32_t)selEdgeIndices.size() : 0;

    uploadEditOverlayBuffer(m_device.get(), m_interactiveViewport.editSelFaceIndexBuffer,
                            selFaceIndices, VulkanRT::BufferUsage::INDEX);
    m_interactiveViewport.editSelFaceIndexCount =
        m_interactiveViewport.editSelFaceIndexBuffer.buffer ? (uint32_t)selFaceIndices.size() : 0;

    m_interactiveViewport.dirty = true;
    m_currentSamples = 0;
}

void VulkanBackendAdapter::setEditMeshOverlayParams(const EditMeshOverlayParams& params) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Called every frame the overlay is active — only force a re-render when
    // something actually changed, otherwise the viewport would never settle.
    // Field-wise compare (NOT memcmp: struct padding is indeterminate).
    const EditMeshOverlayParams& cur = m_interactiveViewport.editOverlayParams;
    auto sameColor = [](const float a[4], const float b[4]) {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
    };
    const bool unchanged =
        cur.enabled == params.enabled &&
        cur.drawEdges == params.drawEdges &&
        cur.drawPoints == params.drawPoints &&
        cur.drawFaces == params.drawFaces &&
        cur.softHighlight == params.softHighlight &&
        cur.xray == params.xray &&
        cur.model == params.model &&
        cur.pointRadiusPx == params.pointRadiusPx &&
        cur.depthBias == params.depthBias &&
        sameColor(cur.edgeColor, params.edgeColor) &&
        sameColor(cur.pointColor, params.pointColor) &&
        sameColor(cur.faceColor, params.faceColor) &&
        sameColor(cur.selectColor, params.selectColor) &&
        sameColor(cur.selectFaceColor, params.selectFaceColor);
    if (unchanged) {
        return;
    }
    m_interactiveViewport.editOverlayParams = params;
    m_interactiveViewport.dirty = true;
    m_currentSamples = 0;
}

void VulkanBackendAdapter::clearEditMeshOverlay() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;

    const bool hadContent =
        m_interactiveViewport.editOverlayParams.enabled ||
        m_interactiveViewport.editVertexCount > 0;

    if (m_interactiveViewport.editPositionBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.editPositionBuffer);
    }
    if (m_interactiveViewport.editFlagBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.editFlagBuffer);
    }
    if (m_interactiveViewport.editEdgeIndexBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.editEdgeIndexBuffer);
    }
    if (m_interactiveViewport.editFaceIndexBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.editFaceIndexBuffer);
    }
    if (m_interactiveViewport.editSelEdgeIndexBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.editSelEdgeIndexBuffer);
    }
    if (m_interactiveViewport.editSelFaceIndexBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.editSelFaceIndexBuffer);
    }
    m_interactiveViewport.editVertexCount = 0;
    m_interactiveViewport.editEdgeIndexCount = 0;
    m_interactiveViewport.editFaceIndexCount = 0;
    m_interactiveViewport.editSelEdgeIndexCount = 0;
    m_interactiveViewport.editSelFaceIndexCount = 0;
    m_interactiveViewport.editOverlayParams = EditMeshOverlayParams{};

    if (hadContent) {
        m_interactiveViewport.dirty = true;
        m_currentSamples = 0;
    }
}

void VulkanBackendAdapter::setSelectionOutlineParams(const SelectionOutlineParams& params) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Called every frame while a selection exists — only force a re-render
    // when something actually changed. Instance transforms are re-resolved
    // from m_rasterInstances at draw time, so a gizmo drag doesn't need to
    // flow through here (the transform sync already dirties the viewport).
    const SelectionOutlineParams& cur = m_interactiveViewport.selectionOutlineParams;
    auto sameColor = [](const float a[4], const float b[4]) {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
    };
    const bool unchanged =
        cur.enabled == params.enabled &&
        cur.nodeNames == params.nodeNames &&
        cur.thicknessPx == params.thicknessPx &&
        sameColor(cur.primaryColor, params.primaryColor) &&
        sameColor(cur.secondaryColor, params.secondaryColor) &&
        sameColor(cur.occludedColor, params.occludedColor);
    if (unchanged) {
        return;
    }
    m_interactiveViewport.selectionOutlineParams = params;
    m_interactiveViewport.dirty = true;
    m_currentSamples = 0;
}

namespace {
inline void selectionMatrixToGL(const Matrix4x4& mat, float out[16]) {
    Matrix4x4 t = mat.transpose();
    int k = 0;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out[k++] = t.m[r][c];
        }
    }
}
} // namespace

void VulkanBackendAdapter::resolveSelectionOutlineDraws(
        const std::vector<std::string>& nodeNames,
        std::vector<SelectionOutlineDrawItem>& outDraws) {
    outDraws.clear();
    if (!m_device || nodeNames.empty()) return;

    // One mat4 per matched (mesh, instance) pair; the per-draw byte offset
    // into selectionInstanceBuffer selects the matrix. The LAST name is the
    // primary selection (mask value 1.0), the rest are secondary (0.5).
    constexpr size_t kMaxSelectionOutlineInstances = 256;
    std::vector<float> selMats;
    for (size_t ni = 0; ni < nodeNames.size(); ++ni) {
        const std::string& selName = nodeNames[ni];
        if (selName.empty()) continue;
        const float maskValue = (ni + 1 == nodeNames.size()) ? 1.0f : 0.5f;
        for (const auto& ri : m_rasterInstances) {
            if (ri.mask == 0) continue;
            const bool match =
                ri.nodeName == selName ||
                viewportMatchesNodeNameForInstance(ri.nodeName, selName) ||
                viewportMatchesNodeNameForInstance(selName, ri.nodeName);
            if (!match) continue;
            auto meshIt = m_rasterMeshes.find(ri.meshKey);
            if (meshIt == m_rasterMeshes.end()) continue;
            const RasterMeshBuffer& srmb = meshIt->second;
            // Scatter pools are huge and have their own proxy logic — out of
            // scope for the outline (matches the CPU path's semantics).
            if (srmb.isScatterGroup || srmb.isScatterProxy) continue;
            if (!srmb.vertexBuffer.buffer || srmb.vertexCount == 0) continue;

            SelectionOutlineDrawItem d;
            d.mesh = &srmb;
            d.instanceByteOffset = selMats.size() * sizeof(float);
            d.maskValue = maskValue;
            outDraws.push_back(d);

            float gl[16];
            selectionMatrixToGL(ri.transform, gl);
            selMats.insert(selMats.end(), gl, gl + 16);
            if (outDraws.size() >= kMaxSelectionOutlineInstances) break;
        }
        if (outDraws.size() >= kMaxSelectionOutlineInstances) break;
    }

    if (selMats.empty()) {
        outDraws.clear();
        return;
    }
    const uint64_t byteSize = selMats.size() * sizeof(float);
    VulkanRT::BufferHandle& selBuf = m_interactiveViewport.selectionInstanceBuffer;
    if (!selBuf.buffer || selBuf.size < byteSize) {
        if (selBuf.buffer) {
            m_device->destroyBuffer(selBuf);
        }
        VulkanRT::BufferCreateInfo bci{};
        bci.size = byteSize;
        bci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        // Host-visible: uploadBuffer then maps + memcpys directly instead of
        // submitting a staging copy — this resolve runs per recompute and the
        // buffer is tiny, so skipping a blocking GPU submit matters more than
        // device-local read speed. Safe: only the synchronous single-time
        // renders read it, never an async trace.
        bci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
        selBuf = m_device->createBuffer(bci);
    }
    if (selBuf.buffer) {
        m_device->uploadBuffer(selBuf, selMats.data(), byteSize, 0);
    } else {
        outDraws.clear();
    }
}

void VulkanBackendAdapter::recordSelectionOutlineMaskPass(
        VkCommandBuffer cmd,
        const std::vector<SelectionOutlineDrawItem>& draws,
        const Matrix4x4& viewProj, int width, int height) {
    VkClearValue maskClear[2]{};
    maskClear[0].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    maskClear[1].depthStencil = { 1.0f, 0 }; // ignored (loadOp = LOAD)

    VkRenderPassBeginInfo maskRPBI{};
    maskRPBI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    maskRPBI.renderPass = m_interactiveViewport.selectionMaskRenderPass;
    maskRPBI.framebuffer = m_interactiveViewport.selectionMaskFramebuffer;
    maskRPBI.renderArea.offset = { 0, 0 };
    maskRPBI.renderArea.extent = { (uint32_t)width, (uint32_t)height };
    maskRPBI.clearValueCount = 2;
    maskRPBI.pClearValues = maskClear;
    vkCmdBeginRenderPass(cmd, &maskRPBI, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{};
    vp.width = (float)width;
    vp.height = (float)height;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{};
    sc.extent = { (uint32_t)width, (uint32_t)height };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    struct SelectionMaskPush {
        float viewProj[16];
        float maskValue[4];
    };
    SelectionMaskPush maskPush{};
    selectionMatrixToGL(viewProj, maskPush.viewProj);

    const VkPipeline maskPipelines[2] = {
        m_interactiveViewport.selectionMaskFullPipeline,    // G: full silhouette
        m_interactiveViewport.selectionMaskVisiblePipeline  // R: depth-tested visible
    };
    for (VkPipeline maskPipeline : maskPipelines) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, maskPipeline);
        for (const SelectionOutlineDrawItem& d : draws) {
            maskPush.maskValue[0] = d.maskValue;
            vkCmdPushConstants(cmd, m_interactiveViewport.selectionMaskPipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SelectionMaskPush), &maskPush);
            VkBuffer vbs[2] = {
                d.mesh->vertexBuffer.buffer,
                m_interactiveViewport.selectionInstanceBuffer.buffer
            };
            VkDeviceSize vbOffsets[2] = { 0, d.instanceByteOffset };
            vkCmdBindVertexBuffers(cmd, 0, 2, vbs, vbOffsets);
            if (d.mesh->indexBuffer.buffer && d.mesh->indexCount > 0) {
                vkCmdBindIndexBuffer(cmd, d.mesh->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, d.mesh->indexCount, 1, 0, 0, 0);
            } else {
                vkCmdDraw(cmd, d.mesh->vertexCount, 1, 0, 0);
            }
        }
    }
    vkCmdEndRenderPass(cmd);
}

void VulkanBackendAdapter::setRasterInstanceTransformsForNodes(
        const std::vector<std::pair<std::string, Matrix4x4>>& nodes) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (nodes.empty() || m_rasterInstances.empty()) return;
    rebuildTargetedTransformIndex();

    auto applyToInstance = [&](RasterInstance& ri, const Matrix4x4& mtx) {
        if (ri.transform == mtx) return;
        ri.transform = mtx;
        updateRasterInstanceWorldBBox(ri);
        auto meshIt = m_rasterMeshes.find(ri.meshKey);
        if (meshIt != m_rasterMeshes.end()) {
            // Next uploadVisibleRasterInstances re-uploads just this mesh's
            // instance buffer — the depth prepass then sees the fresh pose.
            meshIt->second.visibleInstancesDirty = true;
        }
    };

    for (const auto& node : nodes) {
        if (node.first.empty()) continue;
        auto idxIt = m_rasterNodeIndex.find(node.first);
        if (idxIt != m_rasterNodeIndex.end()) {
            for (uint32_t i : idxIt->second) {
                if (i < m_rasterInstances.size()) {
                    applyToInstance(m_rasterInstances[i], node.second);
                }
            }
            continue;
        }
        // Exact name missing from the index — fall back to the same prefix
        // matching the outline draw resolution uses.
        for (auto& ri : m_rasterInstances) {
            if (ri.nodeName == node.first ||
                viewportMatchesNodeNameForInstance(ri.nodeName, node.first) ||
                viewportMatchesNodeNameForInstance(node.first, ri.nodeName)) {
                applyToInstance(ri, node.second);
            }
        }
    }
}

bool VulkanBackendAdapter::renderSelectionOutlineMaskReadback(
        const std::vector<std::string>& nodeNames,
        const Vec3& camEye, const Vec3& camLookAt,
        const Vec3& camUp, float camFovDeg,
        float camAspect,
        int fullWidth, int fullHeight,
        int maskWidth, int maskHeight,
        std::vector<uint8_t>& outMaskRG) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device || !m_device->isInitialized() || !m_device->supportsGraphicsQueue()) return false;
    if (nodeNames.empty() || fullWidth <= 0 || fullHeight <= 0) return false;
    if (maskWidth <= 0 || maskHeight <= 0 || maskWidth > fullWidth || maskHeight > fullHeight) return false;
    if (m_rasterMeshes.empty() || m_rasterInstances.empty()) return false;

    // Same shader-dir fallback chain as the interactive render path.
    std::string shaderDir = "shaders";
    if (!std::filesystem::exists(shaderDir + "/solid.spv")) shaderDir = "source/shaders";
    if (!std::filesystem::exists(shaderDir + "/solid.spv")) shaderDir = "../shaders";
    if (!std::filesystem::exists(shaderDir + "/solid.spv")) {
        shaderDir = std::filesystem::current_path().string() + "/shaders";
    }
    // Targets stay at full size; the mask renders into the top-left
    // maskWidth x maskHeight sub-region. This keeps resource sizes stable
    // when the caller alternates between coarse (interactive) and full
    // (settled refine) masks — a resize here would destroy and recreate the
    // whole interactive viewport every transition.
    if (!ensureInteractiveViewportResources(shaderDir, fullWidth, fullHeight)) return false;
    if (m_interactiveViewport.solidPipeline == VK_NULL_HANDLE ||
        m_interactiveViewport.pipelineLayout == VK_NULL_HANDLE ||
        m_interactiveViewport.framebuffer == VK_NULL_HANDLE ||
        m_interactiveViewport.selectionMaskFullPipeline == VK_NULL_HANDLE ||
        m_interactiveViewport.selectionMaskVisiblePipeline == VK_NULL_HANDLE ||
        m_interactiveViewport.selectionMaskFramebuffer == VK_NULL_HANDLE ||
        m_interactiveViewport.selectionMaskImage.image == VK_NULL_HANDLE ||
        !m_interactiveViewport.stagingBuffer.buffer) {
        return false;
    }

    // Camera matrices — PERSPECTIVE always: the Rendered viewport path-traces
    // in perspective even when the viewport camera is orthographic, and this
    // overlay must align with that image.
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
    // Aspect comes from the caller (render-image aspect, the same convention
    // the gizmo overlay projection uses) — NOT the readback target's aspect:
    // the path-traced image is produced at render resolution and stretched
    // onto the screen, and the overlay must line up with that.
    const float aspect = (camAspect > 1e-4f)
        ? camAspect
        : ((fullHeight > 0) ? ((float)fullWidth / (float)fullHeight) : 1.0f);
    const float fovDeg = camFovDeg > 1.0f ? camFovDeg : 60.0f;
    const float zNear = 0.01f, zFar = 1000000.0f;
    const float f = 1.0f / std::tan(fovDeg * 0.5f * 3.14159265358979f / 180.0f);
    Matrix4x4 proj = Matrix4x4::zero();
    proj.m[0][0] = f / aspect;
    proj.m[1][1] = -f;
    proj.m[2][2] = zFar / (zNear - zFar);
    proj.m[2][3] = (zFar * zNear) / (zNear - zFar);
    proj.m[3][2] = -1.0f;
    Matrix4x4 view = makeViewMatrix(camEye, camLookAt, camUp);
    Matrix4x4 viewProj = proj * view;

    // Refresh per-mesh visible-instance buffers for THIS camera — in Rendered
    // mode the per-frame raster culling upload doesn't run. Scatter pools are
    // excluded outright: re-culling + re-uploading hundreds of thousands of
    // foliage instances per recompute caused visible stalls, scatter objects
    // are not selectable anyway, and their only contribution here would be
    // the occluded-gray tint behind foliage.
    extractFrustumPlanes(viewProj);
    for (auto& [meshKey, mesh] : m_rasterMeshes) {
        (void)meshKey;
        if (mesh.isScatterGroup || mesh.isScatterProxy) continue;
        uploadVisibleRasterInstances(mesh);
    }

    std::vector<SelectionOutlineDrawItem> draws;
    resolveSelectionOutlineDraws(nodeNames, draws);
    if (draws.empty()) return false;

    VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return false;

    // ── Depth prepass: rasterize the whole scene into depthImage via the
    // main render pass + solid pipeline. The color result is irrelevant here
    // (Rendered mode displays the path-traced image); the raster viewport
    // fully re-renders on its next dirty frame anyway.
    {
        VkClearValue clearValues[2]{};
        clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
        clearValues[1].depthStencil = { 1.0f, 0 };
        VkRenderPassBeginInfo rpbi{};
        rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass = m_interactiveViewport.renderPass;
        rpbi.framebuffer = m_interactiveViewport.framebuffer;
        rpbi.renderArea.offset = { 0, 0 };
        rpbi.renderArea.extent = { (uint32_t)maskWidth, (uint32_t)maskHeight };
        rpbi.clearValueCount = 2;
        rpbi.pClearValues = clearValues;
        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp{};
        vp.width = (float)maskWidth;
        vp.height = (float)maskHeight;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{};
        sc.extent = { (uint32_t)maskWidth, (uint32_t)maskHeight };
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.solidPipeline);
        if (m_interactiveViewport.matcapDescSet != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_interactiveViewport.pipelineLayout,
                                    0, 1, &m_interactiveViewport.matcapDescSet, 0, nullptr);
        }
        // Must match the layout pushed by the interactive render path.
        struct SolidPushConstants {
            float viewProj[16];
            float view[16];
            int useMatcap;
            float overrideR, overrideG, overrideB;
            float fadeCenterX, fadeCenterY, fadeCenterZ;
            float fadeStart, fadeEnd;
            float overrideA;
        };
        SolidPushConstants push{};
        selectionMatrixToGL(viewProj, push.viewProj);
        selectionMatrixToGL(view, push.view);
        vkCmdPushConstants(cmd, m_interactiveViewport.pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(SolidPushConstants), &push);

        for (const auto& [meshKey, rmb] : m_rasterMeshes) {
            (void)meshKey;
            // Scatter pools skipped (see the culling loop above) — they don't
            // occlude the outline, but they also don't stall the recompute.
            if (rmb.isScatterGroup || rmb.isScatterProxy) continue;
            if (!rmb.vertexBuffer.buffer || !rmb.instanceBuffer.buffer ||
                rmb.vertexCount == 0 || rmb.instanceCount == 0) {
                continue;
            }
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
        vkCmdEndRenderPass(cmd);
    }

    recordSelectionOutlineMaskPass(cmd, draws, viewProj, maskWidth, maskHeight);

    // ── Mask sub-region → staging buffer (R8G8, rows tightly packed at
    // maskWidth; the full-size RGBA8 staging buffer is always large enough).
    // The mask pass left the image in SHADER_READ_ONLY_OPTIMAL.
    {
        VkImageMemoryBarrier toSrc{};
        toSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toSrc.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toSrc.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.image = m_interactiveViewport.selectionMaskImage.image;
        toSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toSrc.subresourceRange.levelCount = 1;
        toSrc.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &toSrc);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { (uint32_t)maskWidth, (uint32_t)maskHeight, 1 };
        vkCmdCopyImageToBuffer(cmd, m_interactiveViewport.selectionMaskImage.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               m_interactiveViewport.stagingBuffer.buffer, 1, &region);
        // No restore barrier: the next mask pass enters with initialLayout
        // UNDEFINED and the composite only samples after a fresh mask pass.
    }
    m_device->endSingleTimeCommands(cmd);

    const size_t maskBytes = (size_t)maskWidth * (size_t)maskHeight * 2;
    outMaskRG.resize(maskBytes);
    m_device->downloadBuffer(m_interactiveViewport.stagingBuffer, outMaskRG.data(), maskBytes);

    // We trashed colorImage/depthImage; make sure the raster viewport fully
    // re-renders instead of trusting any cached-frame state on mode switch.
    m_interactiveViewport.dirty = true;
    return true;
}

bool VulkanBackendAdapter::hasGpuSelectionOutline() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    return m_interactiveViewport.selectionMaskFullPipeline != VK_NULL_HANDLE &&
           m_interactiveViewport.selectionMaskVisiblePipeline != VK_NULL_HANDLE &&
           m_interactiveViewport.selectionCompositePipeline != VK_NULL_HANDLE &&
           m_interactiveViewport.selectionMaskFramebuffer != VK_NULL_HANDLE &&
           m_interactiveViewport.selectionCompositeFramebuffer != VK_NULL_HANDLE &&
           m_interactiveViewport.selectionCompositeDescSet != VK_NULL_HANDLE;
}

void VulkanBackendAdapter::clearSelectionOutline() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_interactiveViewport.selectionOutlineParams.enabled &&
        m_interactiveViewport.selectionOutlineParams.nodeNames.empty()) {
        return;
    }
    m_interactiveViewport.selectionOutlineParams = SelectionOutlineParams{};
    m_interactiveViewport.dirty = true;
    m_currentSamples = 0;
}

} // namespace Backend
