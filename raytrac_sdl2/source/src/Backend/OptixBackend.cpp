/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          OptixBackend.cpp
 * Description:   NVIDIA OptiX implementation of IBackend
 * =========================================================================
 */
#include "Backend/OptixBackend.h"
#include "Backend/VulkanBackend.h"
#include "OptixWrapper.h"
#include "Camera.h"
#include <iostream>
#include <cuda_runtime.h>

namespace Backend {

OptixBackend::OptixBackend() 
    : m_optix_owned(std::make_unique<OptixWrapper>()), m_optix(m_optix_owned.get()) {}

OptixBackend::OptixBackend(OptixWrapper* existingWrapper) 
    : m_optix_owned(nullptr), m_optix(existingWrapper) {}

OptixBackend::~OptixBackend() = default;

bool OptixBackend::initialize() {
    try {
        m_optix->initialize();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[OptixBackend] Init failed: " << e.what() << std::endl;
        return false;
    }
}

void OptixBackend::shutdown() {
    m_optix->cleanup();
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
    std::vector<GpuMaterial> gpuMaterials;
    gpuMaterials.reserve(materials.size());
    
    for (const auto& mat : materials) {
        GpuMaterial gpuMat = {};
        gpuMat.albedo = make_float3(mat.albedo.x, mat.albedo.y, mat.albedo.z);
        gpuMat.roughness = mat.roughness;
        gpuMat.metallic = mat.metallic;
        gpuMat.emission = make_float3(mat.emission.x * mat.emissionStrength, mat.emission.y * mat.emissionStrength, mat.emission.z * mat.emissionStrength);
        gpuMat.ior = mat.ior;
        gpuMat.transmission = mat.transmission;
        
        gpuMat.albedo_tex = (cudaTextureObject_t)mat.albedoTexture;
        
        gpuMaterials.push_back(gpuMat);
    }
    
    m_optix->updateMaterialBuffer(gpuMaterials);
}

void OptixBackend::uploadHairMaterials(const std::vector<HairMaterialData>& materials) {
    if (materials.empty()) return;
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

    // Textures
    cudaTextureObject_t albedoTex = (mat.albedoTexture != -1) ? (cudaTextureObject_t)mat.albedoTexture : 0;
    cudaTextureObject_t roughnessTex = (mat.roughnessTexture != -1) ? (cudaTextureObject_t)mat.roughnessTexture : 0;
    cudaTextureObject_t scalpAlbedoTex = (mat.scalpAlbedoTexture != -1) ? (cudaTextureObject_t)mat.scalpAlbedoTexture : 0;

    m_optix->setHairTextures(
        albedoTex, mat.albedoTexture != -1,
        roughnessTex, mat.roughnessTexture != -1,
        scalpAlbedoTex, mat.scalpAlbedoTexture != -1,
        make_float3(mat.scalpBaseColor.x, mat.scalpBaseColor.y, mat.scalpBaseColor.z)
    );
}

int64_t OptixBackend::uploadTexture2D(const void* data, uint32_t width, uint32_t height, uint32_t channels, bool sRGB, bool isFloat) {
    return 0; 
}

void OptixBackend::destroyTexture(int64_t textureHandle) {
    if (textureHandle) {
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
    m_optix->setTime(time, deltaTime);
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
        *static_cast<std::vector<uchar4>*>(outFramebuffer),
        static_cast<SDL_Texture*>(outTexture)
    );
}

void OptixBackend::downloadImage(void* outPixels) {
    if (!outPixels) return;
    m_optix->downloadFramebuffer(static_cast<uchar4*>(outPixels), m_optix->getImageWidth(), m_optix->getImageHeight());
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
