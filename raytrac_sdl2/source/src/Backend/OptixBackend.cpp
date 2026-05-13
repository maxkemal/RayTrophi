/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          OptixBackend.cpp
 * Description:   NVIDIA OptiX implementation of IBackend
 * =========================================================================
 */
#include "Backend/OptixBackend.h"
#include "Backend/VulkanBackend.h"
#include "Core/RenderStateManager.h"
#include "globals.h"
#include "OptixWrapper.h"
#include "Camera.h"
#include "Texture.h"
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

namespace Backend {
namespace {
    inline void trimCudaMemoryPool() {
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess || device < 0) return;

        // Ensure all pending work is done before trimming allocator caches.
        cudaDeviceSynchronize();

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
        cudaMemPool_t pool = nullptr;
        if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess && pool) {
            // Release cached pages back to the driver/OS.
            cudaMemPoolTrimTo(pool, 0);
        }
#endif
    }

    std::string buildOptixSceneTextureKey(const Texture* tex) {
        if (!tex) return {};
        std::string key = tex->name.empty()
            ? "tex_ptr:" + std::to_string(reinterpret_cast<uintptr_t>(tex))
            : "tex:" + tex->name;
        key += (tex->is_hdr ? "|backend=optix|hdr=1" : "|backend=optix|hdr=0");
        return key;
    }

    uint64_t estimateOptixTextureBytes(const Texture* tex) {
        if (!tex || tex->width <= 0 || tex->height <= 0) return 0;
        const uint64_t pixelCount = static_cast<uint64_t>(tex->width) * static_cast<uint64_t>(tex->height);
        return tex->is_hdr ? pixelCount * 16ull : pixelCount * 4ull;
    }

    cudaTextureObject_t ensureOptixTextureResident(Texture* tex) {
        if (!tex || !tex->is_loaded()) return 0;
        if ((!tex->is_gpu_uploaded || tex->get_cuda_texture() == 0) && g_hasOptix) {
            if (tex->get_cuda_texture() == 0) {
                tex->is_gpu_uploaded = false;
            }
            tex->upload_to_gpu();
        }
        return tex->get_cuda_texture();
    }

    cudaTextureObject_t resolveOptixTexture(SceneTextureManager* manager, int64_t rawHandle) {
        if (!rawHandle) return 0;

        // Small positive handles are already backend/runtime ids in several
        // transitional paths. Do not reinterpret them as host Texture pointers.
        if (rawHandle > 0 && static_cast<uint64_t>(rawHandle) < (1ull << 32)) {
            return static_cast<cudaTextureObject_t>(rawHandle);
        }

        Texture* tex = reinterpret_cast<Texture*>(rawHandle);
        if (!tex || !tex->is_loaded()) {
            return static_cast<cudaTextureObject_t>(rawHandle);
        }

        if (manager) {
            int64_t resolvedTextureId = 0;
            if (manager->resolveOrCreateOptixTextureId(
                buildOptixSceneTextureKey(tex),
                TextureConsumer::Optix,
                static_cast<uint32_t>(std::max(0, tex->width)),
                static_cast<uint32_t>(std::max(0, tex->height)),
                estimateOptixTextureBytes(tex),
                [tex](TextureHandle) -> int64_t {
                    return static_cast<int64_t>(ensureOptixTextureResident(tex));
                },
                resolvedTextureId)) {
                const cudaTextureObject_t currentTexture = ensureOptixTextureResident(tex);
                if (currentTexture == 0) {
                    manager->clearOptixTextureId(resolvedTextureId);
                    return 0;
                }
                if (currentTexture != static_cast<cudaTextureObject_t>(resolvedTextureId)) {
                    manager->clearOptixTextureId(resolvedTextureId);
                    return currentTexture;
                }
                return static_cast<cudaTextureObject_t>(resolvedTextureId);
            }
        }

        return ensureOptixTextureResident(tex);
    }
}

OptixBackend::OptixBackend()
    : m_optix_owned(std::make_unique<OptixWrapper>()),
      m_optix(m_optix_owned.get()),
      m_sceneTextureManager(getSharedSceneTextureManager()) {}

OptixBackend::OptixBackend(OptixWrapper* existingWrapper)
    : m_optix_owned(nullptr),
      m_optix(existingWrapper),
      m_sceneTextureManager(getSharedSceneTextureManager()) {}

OptixBackend::~OptixBackend() = default;

bool OptixBackend::initialize() {
    try {
        if (m_sceneTextureManager) {
            m_sceneTextureManager->initialize(captureRuntimeRenderCapabilities(), "OptixBackend");
        }
        m_optix->initialize();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[OptixBackend] Init failed: " << e.what() << std::endl;
        return false;
    }
}

void OptixBackend::shutdown() {
    if (m_optix) {
        m_optix->cleanup();
    }
    // Important for OptiX -> Vulkan switches:
    // release CUDA allocator cache so shared GPU memory pressure does not
    // accumulate across backend toggles in long sessions.
    trimCudaMemoryPool();
}

void OptixBackend::loadShaders(const ShaderProgramData& data) {
    OptixWrapper::PtxData ptx;
    ptx.raygen_ptx = data.raygen.c_str();
    ptx.miss_ptx = data.miss.c_str();
    ptx.hitgroup_ptx = data.hitgroup.c_str();
    m_optix->setupPipeline(ptx);
}

BackendInfo OptixBackend::getInfo() const {
    BackendInfo info;
    info.type = BackendType::OPTIX;
    info.name = "NVIDIA OptiX";
    
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    info.deviceName = prop.name;
    info.hasHardwareRT = (prop.major > 7) || (prop.major == 7 && prop.minor >= 5);
    info.vramBytes = prop.totalGlobalMem;
    
    return info;
}

GpuMemoryStats OptixBackend::getMemoryStats() const {
    GpuMemoryStats stats;

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess && totalBytes > 0) {
        stats.totalBytes = static_cast<uint64_t>(totalBytes);
        stats.freeBytes = static_cast<uint64_t>(freeBytes);
        stats.usedBytes = stats.totalBytes - stats.freeBytes;
        stats.hasDeviceUsage = true;
    } else {
        const BackendInfo info = getInfo();
        stats.totalBytes = info.vramBytes;
    }

    if (m_sceneTextureManager) {
        stats.trackedTextureBytes = m_sceneTextureManager->totalResidentTextureBytes();
        stats.trackedTextureBytesThisBackend = m_sceneTextureManager->estimatedOptixTextureBytes();
        stats.hasTrackedTextures = true;
    }

    return stats;
}

uint32_t OptixBackend::uploadTriangles(const std::vector<TriangleData>& triangles, const std::string& meshName) {
    return 0; 
}

uint32_t OptixBackend::uploadHairStrands(const std::vector<HairStrandData>& strands, const std::string& groomName) {
    return 0;
}

void OptixBackend::updateMeshTransform(uint32_t meshHandle, const Matrix4x4& transform) {
}

void OptixBackend::rebuildAccelerationStructure() {
    m_optix->rebuildTLAS();
}

void OptixBackend::showAllInstances() {
    m_optix->showAllInstances();
}

void OptixBackend::updateSceneGeometry(const std::vector<std::shared_ptr<Hittable>>& objects, const std::vector<Matrix4x4>& boneMatrices) {
    m_optix->updateTLASGeometry(objects, boneMatrices);
}

void OptixBackend::updateInstanceMaterialBinding(const std::string& nodeName, int oldMatID, int newMatID) {
    m_optix->updateMeshMaterialBinding(nodeName, oldMatID, newMatID);
}

void OptixBackend::setVisibilityByNodeName(const std::string& nodeName, bool visible) {
    m_optix->setVisibilityByNodeName(nodeName, visible);
}

void OptixBackend::updateGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) {
    m_optix->updateGeometry(objects);
}

void OptixBackend::uploadMaterials(const std::vector<MaterialData>& materials) {
    if (!m_optix || !g_hasOptix) return;
    ScopedCudaTextureUpload allowCudaTextureUpload;
    std::vector<GpuMaterial> gpuMaterials;
    gpuMaterials.reserve(materials.size());
    
    for (const auto& mat : materials) {
        GpuMaterial gpuMat = {};
        gpuMat.albedo = make_float3(mat.albedo.x, mat.albedo.y, mat.albedo.z);
        gpuMat.roughness = mat.roughness;
        gpuMat.metallic = mat.metallic;
        gpuMat.specular = mat.specular;
        gpuMat.emission = make_float3(mat.emission.x * mat.emissionStrength, mat.emission.y * mat.emissionStrength, mat.emission.z * mat.emissionStrength);
        gpuMat.ior = mat.ior;
        gpuMat.transmission = mat.transmission;
        gpuMat.opacity = mat.opacity;
        gpuMat.subsurface = mat.subsurface;
        gpuMat.subsurface_color = make_float3(mat.subsurfaceColor.x, mat.subsurfaceColor.y, mat.subsurfaceColor.z);
        gpuMat.subsurface_radius = make_float3(mat.subsurfaceRadius.x, mat.subsurfaceRadius.y, mat.subsurfaceRadius.z);
        gpuMat.subsurface_scale = mat.subsurfaceScale;
        gpuMat.subsurface_anisotropy = mat.subsurfaceAnisotropy;
        gpuMat.clearcoat = mat.clearcoat;
        gpuMat.clearcoat_roughness = mat.clearcoatRoughness;
        gpuMat.sheen = mat.sheen;
        gpuMat.sheen_tint = mat.sheenTint;
        gpuMat.anisotropic = mat.anisotropic;
        gpuMat.flags = static_cast<int>(mat.flags);
        gpuMat.tile_break_strength = mat.tile_break_strength;
        gpuMat.micro_detail_strength = mat.micro_detail_strength;
        gpuMat.micro_detail_scale = mat.micro_detail_scale;
        gpuMat.fft_ocean_size = mat.fft_ocean_size;
        gpuMat.fft_choppiness = mat.fft_choppiness;
        gpuMat.fft_wind_speed = mat.fft_wind_speed;
        gpuMat.fft_wind_direction = mat.fft_wind_direction;
        gpuMat.fft_amplitude = mat.fft_amplitude;
        gpuMat.fft_time_scale = mat.fft_time_scale;
        gpuMat.micro_anim_speed = mat.micro_anim_speed;
        gpuMat.micro_morph_speed = mat.micro_morph_speed;
        gpuMat.foam_noise_scale = mat.foam_noise_scale;
        gpuMat.foam_threshold = mat.foam_threshold;
        gpuMat.normal_strength = mat.normalStrength;
        gpuMat.uv_scale_x = static_cast<float>(mat.uvScale.x);
        gpuMat.uv_scale_y = static_cast<float>(mat.uvScale.y);
        gpuMat.uv_offset_x = static_cast<float>(mat.uvOffset.x);
        gpuMat.uv_offset_y = static_cast<float>(mat.uvOffset.y);
        gpuMat.uv_rotation_degrees = mat.uvRotationDegrees;
        gpuMat.uv_tiling_x = static_cast<float>(mat.uvTiling.x);
        gpuMat.uv_tiling_y = static_cast<float>(mat.uvTiling.y);
        gpuMat.uv_wrap_mode = static_cast<int>(mat.uvWrapMode);
        
        SceneTextureManager* textureManager = m_sceneTextureManager.get();
        gpuMat.albedo_tex = resolveOptixTexture(textureManager, mat.albedoTexture);
        gpuMat.normal_tex = resolveOptixTexture(textureManager, mat.normalTexture);
        gpuMat.roughness_tex = resolveOptixTexture(textureManager, mat.roughnessTexture);
        gpuMat.metallic_tex = resolveOptixTexture(textureManager, mat.metallicTexture);
        gpuMat.specular_tex = resolveOptixTexture(textureManager, mat.specularTexture);
        gpuMat.emission_tex = resolveOptixTexture(textureManager, mat.emissionTexture);
        gpuMat.transmission_tex = resolveOptixTexture(textureManager, mat.transmissionTexture);
        gpuMat.opacity_tex = resolveOptixTexture(textureManager, mat.opacityTexture);
        gpuMat.height_tex = resolveOptixTexture(textureManager, mat.heightTexture);
        
        gpuMaterials.push_back(gpuMat);
    }
    
    m_optix->updateMaterialBuffer(gpuMaterials);
}

void OptixBackend::uploadHairMaterials(const std::vector<HairMaterialData>& materials) {
    if (materials.empty() || !m_optix || !g_hasOptix) return;
    ScopedCudaTextureUpload allowCudaTextureUpload;
    const auto& mat = materials[0];
    m_optix->setHairMaterial(
        make_float3(mat.color.x, mat.color.y, mat.color.z),
        make_float3(mat.absorption.x, mat.absorption.y, mat.absorption.z),
        mat.melanin,
        mat.melaninRedness,
        mat.roughness,
        mat.radialRoughness,
        mat.ior,
        mat.coat,
        mat.cuticleAngle,
        mat.randomHue,
        mat.randomValue
    );
    m_optix->setHairColorMode(mat.colorMode);

    // Textures are carried as Texture* handles and resolved through the shared pool.
    SceneTextureManager* textureManager = m_sceneTextureManager.get();
    cudaTextureObject_t albedoTex = (mat.albedoTexture != -1) ? resolveOptixTexture(textureManager, mat.albedoTexture) : 0;
    cudaTextureObject_t roughnessTex = (mat.roughnessTexture != -1) ? resolveOptixTexture(textureManager, mat.roughnessTexture) : 0;
    cudaTextureObject_t scalpAlbedoTex = (mat.scalpAlbedoTexture != -1) ? resolveOptixTexture(textureManager, mat.scalpAlbedoTexture) : 0;

    m_optix->setHairTextures(
        albedoTex, albedoTex != 0,
        roughnessTex, roughnessTex != 0,
        scalpAlbedoTex, scalpAlbedoTex != 0,
        make_float3(mat.scalpBaseColor.x, mat.scalpBaseColor.y, mat.scalpBaseColor.z)
    );
}

int64_t OptixBackend::uploadTexture2D(const void* data, uint32_t width, uint32_t height, uint32_t channels, bool sRGB, bool isFloat) {
    return 0; 
}

void OptixBackend::destroyTexture(int64_t textureHandle) {
    if (textureHandle) {
        // Clear cache BEFORE destroying the CUDA object so concurrent resolves
        // never receive a stale handle that CUDA may have recycled to another texture.
        if (m_sceneTextureManager) {
            m_sceneTextureManager->clearOptixTextureId(textureHandle);
        }
        cudaDestroyTextureObject((cudaTextureObject_t)textureHandle);
    }
}

void OptixBackend::setRenderParams(const RenderParams& params) {
    m_optix->resetBuffers(params.imageWidth, params.imageHeight);
}

void OptixBackend::setCamera(const CameraParams& params) {
    Camera cam;
    cam.lookfrom = params.origin;
    cam.lookat = params.lookAt;
    cam.vup = params.up;
    cam.vfov = params.fov;
    cam.aspect_ratio = params.aspectRatio; 
    cam.aperture = params.aperture;
    cam.lens_radius = params.aperture * 0.5f; // [DOF FIX] Ensure lens_radius is set for OptiX kernel
    cam.focus_dist = params.focusDistance;
    // Ensure exposure-related fields are forwarded to the CPU Camera struct
    cam.ev_compensation = params.ev_compensation;
    cam.auto_exposure = params.autoAE;
    cam.use_physical_exposure = params.usePhysicalExposure;
    cam.iso = 100 * (int)std::pow(2.0f, (float)params.isoPresetIndex); // Approx for display
    cam.iso_preset_index = params.isoPresetIndex;
    cam.shutter_preset_index = params.shutterPresetIndex;
    cam.fstop_preset_index = params.fstopPresetIndex;
    cam.auto_exposure = params.autoAE;
    cam.use_physical_exposure = params.usePhysicalExposure;
    cam.enable_motion_blur = params.motionBlurEnabled;
    cam.enable_vignetting = params.vignettingEnabled;
    cam.enable_chromatic_aberration = params.chromaticAberrationEnabled;
    
    // Pro Features
    cam.distortion = params.distortion;
    cam.lens_quality = params.lens_quality;
    cam.vignetting_amount = params.vignetting_amount;
    cam.vignetting_falloff = params.vignetting_falloff;
    cam.chromatic_aberration = params.chromatic_aberration;
    cam.chromatic_aberration_r = params.chromatic_aberration_r;
    cam.chromatic_aberration_b = params.chromatic_aberration_b;
    cam.camera_mode = (CameraMode)params.camera_mode;
    cam.blade_count = params.blade_count;
    
    // Shake / Handheld
    cam.enable_camera_shake = params.shake_enabled;
    cam.shake_intensity = params.shake_intensity;
    cam.shake_frequency = params.shake_frequency;
    cam.handheld_sway_amplitude = params.handheld_sway_amplitude;
    cam.handheld_sway_frequency = params.handheld_sway_frequency;
    cam.breathing_amplitude = params.breathing_amplitude;
    cam.breathing_frequency = params.breathing_frequency;
    cam.enable_focus_drift = params.enable_focus_drift;
    cam.focus_drift_amount = params.focus_drift_amount;
    cam.operator_skill = (Camera::OperatorSkill)params.operator_skill;
    cam.ibis_enabled = params.ibis_enabled;
    cam.ibis_effectiveness = params.ibis_effectiveness;
    cam.rig_mode = (Camera::RigMode)params.rig_mode;

    cam.update_camera_vectors();
    
    m_optix->setCameraParams(cam, params.exposureFactor);
    m_optix->resetAccumulation(); 
}

void OptixBackend::setTime(float time, float deltaTime) {
    (void)deltaTime;
    // OptiX launch params expect absolute water time here, not frame delta.
    // The generic IBackend signature is (time, deltaTime), so translate it to
    // OptiX's (time, water_time) convention explicitly.
    m_optix->setTime(time, time);
}

void OptixBackend::updateInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects) {
    m_optix->updateTLASMatricesOnly(objects);
}

void OptixBackend::updateInstanceTransform(int instance_id, const float transform[12]) {
    m_optix->updateInstanceTransform(instance_id, transform);
}

void OptixBackend::setStatusCallback(std::function<void(const std::string&, int)> callback) {
    m_optix->setAccelManagerStatusCallback(callback);
}

void* OptixBackend::getNativeCommandQueue() {
    return (void*)m_optix->getStream();
}

void OptixBackend::setWindParams(const Vec3& direction, float strength, float speed, float time) {
    m_optix->setWindParams(direction, strength, speed, time);
}

void OptixBackend::renderPass(bool accumulate) {
    if (!accumulate) m_optix->resetAccumulation();
    m_optix->launch(m_optix->getImageWidth(), m_optix->getImageHeight());
}

void OptixBackend::renderProgressive(void* outSurface, void* outWindow, void* outRenderer, 
                                    int width, int height, void* outFramebuffer, void* outTexture) {
    m_optix->launch_random_pixel_mode_progressive(
        static_cast<SDL_Surface*>(outSurface),
        static_cast<SDL_Window*>(outWindow),
        static_cast<SDL_Renderer*>(outRenderer),
        width, height,
        outFramebuffer,
        static_cast<SDL_Texture*>(outTexture)
    );
}

void OptixBackend::setViewportMode(ViewportMode mode) {
    if (m_viewportMode == mode) return;
    m_viewportMode = mode;
    Core::RenderStateManager::instance().setViewportMode(mode);
}

ViewportMode OptixBackend::getViewportMode() const {
    return m_viewportMode;
}

bool OptixBackend::supportsViewportMode(ViewportMode mode) const {
    // Foundation only for now: non-rendered interactive modes still fall back
    // to the existing OptiX rendered viewport path.
    return mode == ViewportMode::Rendered;
}

bool OptixBackend::updateInteractiveMesh(const std::string& nodeName,
                                         const std::vector<std::shared_ptr<Triangle>>& triangles) {
    (void)nodeName;
    (void)triangles;
    return false;
}

void OptixBackend::downloadImage(void* outPixels) {
    if (!outPixels) return;
    m_optix->downloadFramebuffer(static_cast<uchar4*>(outPixels), m_optix->getImageWidth(), m_optix->getImageHeight());
}

bool OptixBackend::getDenoiserFrame(DenoiserFrameData& frame, bool useAuxiliary) {
    if (!m_optix) return false;

    static std::vector<float> color;
    static std::vector<float> albedo;
    static std::vector<float> normal;
    if (!m_optix->downloadDenoiserBuffers(color, albedo, normal, useAuxiliary)) {
        return false;
    }

    frame.width = m_optix->getImageWidth();
    frame.height = m_optix->getImageHeight();
    frame.color = color.data();
    frame.albedo = useAuxiliary ? albedo.data() : nullptr;
    frame.normal = useAuxiliary ? normal.data() : nullptr;
    return true;
}

bool OptixBackend::getDenoiserFrameGPU(DenoiserFrameDataGPU& frame, bool useAuxiliary) {
    if (!m_optix) return false;
    float4* col = m_optix->getAccumulationDevicePtr();
    float4* alb = useAuxiliary ? m_optix->getDenoiserAlbedoDevicePtr() : nullptr;
    float4* nrm = useAuxiliary ? m_optix->getDenoiserNormalDevicePtr() : nullptr;
    const int w = m_optix->getImageWidth();
    const int h = m_optix->getImageHeight();
    if (!col || w <= 0 || h <= 0) return false;

    frame.width = w;
    frame.height = h;
    frame.colorDevPtr = col;
    frame.albedoDevPtr = alb;
    frame.normalDevPtr = nrm;
    frame.pixelByteStride = sizeof(float4);
    frame.rowByteStride = static_cast<size_t>(w) * sizeof(float4);
    frame.cudaStream = static_cast<void*>(m_optix->getStream());
    frame.cudaDeviceOrdinal = -1;  // OptiX uses current CUDA device; Renderer resolves via cudaGetDevice
    return true;
}

int OptixBackend::getCurrentSampleCount() const {
    return m_optix->getAccumulatedSamples();
}

bool OptixBackend::isAccumulationComplete() const {
    return m_optix->isAccumulationComplete();
}

void OptixBackend::setEnvironmentMap(int64_t hdrTextureHandle) {
}

void OptixBackend::setSkyParams() {
}

void OptixBackend::setLights(const std::vector<std::shared_ptr<Light>>& lights) {
    m_optix->setLightParams(lights);
}

void OptixBackend::setWorldData(const void* worldData) {
    if (worldData) {
        m_optix->setWorld(*(static_cast<const WorldData*>(worldData)));
    }
}

void OptixBackend::updateVDBVolumes(const std::vector<GpuVDBVolume>& volumes) {
    m_optix->updateVDBVolumeBuffer(volumes);
}

void OptixBackend::updateGasVolumes(const std::vector<GpuGasVolume>& volumes) {
    m_optix->updateGasVolumeBuffer(volumes);
}

void OptixBackend::waitForCompletion() {
    cudaStreamSynchronize(m_optix->getStream());
}

void OptixBackend::resetAccumulation() {
    m_optix->resetAccumulation();
}

float OptixBackend::getMillisecondsPerSample() const {
    return 0.0f; 
}

// ========================================================================
// Extended IBackend methods (delegated to OptixWrapper)
// ========================================================================

bool OptixBackend::isUsingTLAS() const {
    return m_optix ? m_optix->isUsingTLAS() : false;
}

std::vector<int> OptixBackend::getInstancesByNodeName(const std::string& nodeName) const {
    return m_optix ? m_optix->getInstancesByNodeName(nodeName) : std::vector<int>{};
}

void OptixBackend::updateObjectTransform(const std::string& nodeName, const Matrix4x4& transform) {
    if (m_optix) m_optix->updateObjectTransform(nodeName, transform);
}

void OptixBackend::syncCamera(const Camera& cam) {
    if (m_optix) m_optix->setCameraParams(cam, cam.getPhysicalExposureMultiplier());
}

void OptixBackend::hideInstancesByNodeName(const std::string& nodeName) {
    if (m_optix) m_optix->hideInstancesByNodeName(nodeName);
}

int OptixBackend::getPickedObjectId(int x, int y, int viewport_width, int viewport_height) {
    return m_optix ? m_optix->getPickedObjectId(x, y, viewport_width, viewport_height) : -1;
}

std::string OptixBackend::getPickedObjectName(int x, int y, int viewport_width, int viewport_height) {
    return m_optix ? m_optix->getPickedObjectName(x, y, viewport_width, viewport_height) : "";
}

// Factory implementation
std::unique_ptr<IBackend> createBackend(BackendType preferredType) {
    // Try OptiX first for AUTO or explicit OPTIX
    if (preferredType == BackendType::OPTIX || preferredType == BackendType::AUTO) {
        auto backend = std::make_unique<OptixBackend>();
        if (backend->initialize()) return backend;
        if (preferredType == BackendType::OPTIX) return nullptr;
        // AUTO: fall through to Vulkan
    }

    // Try Vulkan for VULKAN_RT, VULKAN_COMPUTE, or AUTO fallback
    if (preferredType == BackendType::VULKAN_RT || preferredType == BackendType::VULKAN_COMPUTE || preferredType == BackendType::AUTO) {
        auto backend = std::make_unique<VulkanBackendAdapter>();
        if (backend->initialize()) {
            std::cout << "[BackendFactory] Vulkan backend initialized successfully" << std::endl;
            return backend;
        }
    }

    std::cerr << "[BackendFactory] No backend available!" << std::endl;
    return nullptr;
}

std::vector<BackendInfo> enumerateBackends() {
    std::vector<BackendInfo> backends;
    
    // Check OptiX availability
    try {
        auto optix = std::make_unique<OptixBackend>();
        if (optix->initialize()) {
            backends.push_back(optix->getInfo());
            optix->shutdown();
        }
    } catch (...) {}

    // Check Vulkan availability
    try {
        auto vulkan = std::make_unique<VulkanBackendAdapter>();
        if (vulkan->initialize()) {
            backends.push_back(vulkan->getInfo());
            vulkan->shutdown();
        }
    } catch (...) {}

    return backends;
}

bool isBackendAvailable(BackendType type) {
    if (type == BackendType::OPTIX) {
        try {
            auto optix = std::make_unique<OptixBackend>();
            bool ok = optix->initialize();
            if (ok) optix->shutdown();
            return ok;
        } catch (...) { return false; }
    }
    if (type == BackendType::VULKAN_RT || type == BackendType::VULKAN_COMPUTE) {
        try {
            auto vulkan = std::make_unique<VulkanBackendAdapter>();
            bool ok = vulkan->initialize();
            if (ok) vulkan->shutdown();
            return ok;
        } catch (...) { return false; }
    }
    return false;
}

} // namespace Backend
