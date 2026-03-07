/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          IBackend.h
 * Description:   Abstract Render Backend Interface
 *                Allows switching between OptiX (CUDA), Vulkan, Metal
 * =========================================================================
 */
#ifndef I_BACKEND_H
#define I_BACKEND_H

#include "Vec3.h"
#include "Vec2.h"
#include "Matrix4x4.h"
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include "Hittable.h" // For Hittable definition
#include "light.h"
class Camera;  // Forward declaration for syncCamera
struct GpuVDBVolume;
struct GpuGasVolume;

namespace Backend {

// ============================================================================
// Backend Types
// ============================================================================

enum class BackendType : uint8_t {
    AUTO,           // Auto-detect best available
    OPTIX,          // NVIDIA OptiX (CUDA)
    VULKAN_RT,      // Vulkan with ray tracing
    VULKAN_COMPUTE, // Vulkan compute-only fallback
    METAL,          // Apple Metal (future)
    CPU_EMBREE,     // CPU fallback with Embree
    CPU_CUSTOM      // Custom CPU BVH
};

struct BackendInfo {
    BackendType type;
    std::string name;
    std::string deviceName;
    std::string driverVersion;
    bool hasHardwareRT;
    uint64_t vramBytes;
};

// ============================================================================
// Common Data Structures
// ============================================================================

struct RenderParams {
    int imageWidth;
    int imageHeight;
    int samplesPerPixel;
    int maxBounces;
    int currentPass;
    int frameNumber;
    bool useAdaptiveSampling;
    float adaptiveThreshold;
};

struct CameraParams {
    Vec3 origin;
    Vec3 lookAt;
    Vec3 up;
    float fov;
    float aperture;
    float focusDistance;
    float aspectRatio;
    float exposureFactor;
    float ev_compensation;
    int isoPresetIndex;
    int shutterPresetIndex;
    int fstopPresetIndex;
    bool autoAE;
    bool usePhysicalExposure;
    bool motionBlurEnabled;
    bool vignettingEnabled;
    bool chromaticAberrationEnabled;
    
    // Pro Features
    float distortion;
    float lens_quality;
    float vignetting_amount;
    float vignetting_falloff;
    float chromatic_aberration;
    float chromatic_aberration_r;
    float chromatic_aberration_b;
    int camera_mode;
    int blade_count;
    
    // Shake / Handheld
    bool shake_enabled;
    float shake_intensity;
    float shake_frequency;
    float handheld_sway_amplitude;
    float handheld_sway_frequency;
    float breathing_amplitude;
    float breathing_frequency;
    bool enable_focus_drift;
    float focus_drift_amount;
    int operator_skill;
    bool ibis_enabled;
    float ibis_effectiveness;
    int rig_mode;
};

// Triangle data for upload
struct TriangleData {
    Vec3 v0, v1, v2;        // Positions
    Vec3 n0, n1, n2;        // Normals
    Vec2 uv0, uv1, uv2;     // Texture coords
    uint16_t materialID;
    
    // Optional skinning data
    bool hasSkinData = false;
    int32_t boneIndices_v0[4] = {-1, -1, -1, -1};
    int32_t boneIndices_v1[4] = {-1, -1, -1, -1};
    int32_t boneIndices_v2[4] = {-1, -1, -1, -1};
    float boneWeights_v0[4] = {0, 0, 0, 0};
    float boneWeights_v1[4] = {0, 0, 0, 0};
    float boneWeights_v2[4] = {0, 0, 0, 0};
};

// Hair strand data for upload
struct HairStrandData {
    std::vector<Vec3> points;
    std::vector<float> radii;
    uint16_t materialID;
    Vec2 rootUV;
};

struct ShaderProgramData {
    std::string raygen;
    std::string miss;
    std::string hitgroup;
};

// ============================================================================
// Abstract Backend Interface
// ============================================================================

class IBackend {
public:
    virtual ~IBackend() = default;
    
    // ========================================================================
    // Initialization
    // ========================================================================
    
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual void loadShaders(const ShaderProgramData& data) = 0;
    virtual BackendInfo getInfo() const = 0;
    
    // ========================================================================
    // Geometry Upload
    // ========================================================================
    
    /**
     * @brief Upload triangle mesh geometry
     * @param triangles Vector of triangle data
     * @param meshName Unique identifier for this mesh
     * @return Mesh handle/index
     */
    virtual uint32_t uploadTriangles(
        const std::vector<TriangleData>& triangles,
        const std::string& meshName
    ) = 0;
    
    /**
     * @brief Upload hair strand geometry
     * @param strands Vector of hair strands
     * @param groomName Unique identifier
     * @return Groom handle/index
     */
    virtual uint32_t uploadHairStrands(
        const std::vector<HairStrandData>& strands,
        const std::string& groomName
    ) = 0;
    
    /**
     * @brief Update mesh transform (for instances)
     */
    virtual void updateMeshTransform(uint32_t meshHandle, const Matrix4x4& transform) = 0;
    
    /**
     * @brief Rebuild acceleration structure after geometry changes
     */
    virtual void rebuildAccelerationStructure() = 0;

    /**
     * @brief Show all instances in the scene
     */
    virtual void showAllInstances() = 0;
    
    /**
     * @brief Update scene geometry (fast path for animation/skinning)
     */
    virtual void updateSceneGeometry(const std::vector<std::shared_ptr<Hittable>>& objects, const std::vector<Matrix4x4>& boneMatrices) = 0;

    /**
     * @brief Update material for a specific instance
     */
    virtual void updateInstanceMaterialBinding(const std::string& nodeName, int oldMatID, int newMatID) = 0;
    
    /**
     * @brief Set visibility for all instances matching a node name
     */
    virtual void setVisibilityByNodeName(const std::string& nodeName, bool visible) = 0;

    // ========================================================================
    // Extended Methods (default implementations for backward compatibility)
    // These bridge the gap between OptixWrapper-specific and generic backend.
    // ========================================================================
    
    /**
     * @brief Check if hardware acceleration structure (TLAS) is active
     */
    virtual bool isUsingTLAS() const { return false; }
    
    /**
     * @brief Get instance IDs that match a given node name
     */
    virtual std::vector<int> getInstancesByNodeName(const std::string& nodeName) const { (void)nodeName; return {}; }
    
    /**
     * @brief Update a named object's transform (by node name)
     */
    virtual void updateObjectTransform(const std::string& nodeName, const Matrix4x4& transform) { (void)nodeName; (void)transform; }
    
    /**
     * @brief Set camera from Camera object (convenience overload)
     */
    virtual void syncCamera(const Camera& cam) { (void)cam; }
    
    /**
     * @brief Set lights from scene light list (convenience alias)
     */
    virtual void syncLights(const std::vector<std::shared_ptr<Light>>& lights) { setLights(lights); }
    
    /**
     * @brief Hide all instances matching a node name  
     */
    virtual void hideInstancesByNodeName(const std::string& nodeName) { setVisibilityByNodeName(nodeName, false); }

    /**
     * @brief GPU object picking: get object ID at screen position
     */
    virtual int getPickedObjectId(int x, int y, int viewport_width = 0, int viewport_height = 0) {
        (void)x; (void)y; (void)viewport_width; (void)viewport_height; return -1;
    }
    
    /**
     * @brief GPU object picking: get object name at screen position
     */
    virtual std::string getPickedObjectName(int x, int y, int viewport_width = 0, int viewport_height = 0) {
        (void)x; (void)y; (void)viewport_width; (void)viewport_height; return "";
    }

    /**
     * @brief Update geometry for the entire scene (optimized path)
     */
    virtual void updateGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) = 0;
    
    // ========================================================================
    // Material Upload
    // ========================================================================
    
    struct MaterialData {
        Vec3 albedo = Vec3(0.8f);
        float roughness = 0.5f;
        float metallic = 0.0f;
        Vec3 emission = Vec3(0.0f);
        float emissionStrength = 0.0f;
        float ior = 1.5f;
        float transmission = 0.0f;
        float opacity = 1.0f;

        // Subsurface / SSS
        float subsurface = 0.0f;
        Vec3 subsurfaceColor = Vec3(1.0f);
        Vec3 subsurfaceRadius = Vec3(1.0f);
        float subsurfaceScale = 1.0f;
        float subsurfaceAnisotropy = 0.0f;
        float subsurfaceIOR = 1.33f;
        // SSS behavior controls
        bool useRandomWalkSSS = true; // enable bounded multi-scatter by default
        int  sssMaxSteps = 6;         // default bounded steps for random-walk

        // Clearcoat / Sheen / Anisotropy
        float clearcoat = 0.0f;
        float clearcoatRoughness = 0.0f;
        float translucent = 0.0f;
        float anisotropic = 0.0f;
        float sheen = 0.0f;
        float sheenTint = 0.0f;

        // Texture handles (backend-specific opaque handles)
        int64_t albedoTexture = 0;
        int64_t normalTexture = 0;
        int64_t roughnessTexture = 0;
        int64_t metallicTexture = 0;
        int64_t emissionTexture = 0;
        int64_t transmissionTexture = 0;
        int64_t opacityTexture = 0;
        int64_t heightTexture = 0;

        // Padding/flags for future use
        uint32_t flags = 0;
        uint32_t terrainLayerIdx = 0; // Index into terrain layer buffer (valid when MAT_FLAG_TERRAIN set)

        // Water-specific parameters (maps to VkGpuMaterial Block 8 & Block 9)
        float micro_detail_strength = 0.0f;
        float micro_detail_scale    = 0.0f;
        float foam_threshold        = 0.0f;
        float fft_ocean_size        = 0.0f;
        float fft_choppiness        = 0.0f;
        float fft_wind_speed        = 0.0f;
        float fft_amplitude         = 0.0f;
        float fft_time_scale        = 0.0f;
    };

    // Flag bits for MaterialData::flags
    static constexpr uint32_t MAT_FLAG_TERRAIN = (1u << 16); // Splat-blended terrain material

    /**
     * @brief Per-terrain layer descriptor for splat-map based blending.
     *        Used by the Vulkan backend (binding 12).
     */
    struct TerrainLayerData {
        uint32_t layer_mat_id[4]   = {0, 0, 0, 0};  // Material indices for layers 0-3
        float    layer_uv_scale[4] = {1, 1, 1, 1};  // UV tiling for layers 0-3
        int64_t  splatMapTexture   = 0;              // Texture pointer/handle for RGBA splat map
        uint32_t layer_count       = 0;              // Active layer count (0 = no terrain)
    };

    virtual void uploadMaterials(const std::vector<MaterialData>& materials) = 0;

    /**
     * @brief Upload terrain layer descriptors for splat-map blending (Vulkan path).
     *        Default no-op for backends that handle terrain differently (e.g. OptiX).
     */
    virtual void uploadTerrainLayerMaterials(const std::vector<TerrainLayerData>& /*layers*/) {}
    
    // Hair-specific material (synced with all rendering backends)
    struct HairMaterialData {
        Vec3 color;
        Vec3 absorption;
        float melanin;
        float melaninRedness;
        float roughness;
        float radialRoughness;
        float ior;
        float coat;
        float cuticleAngle;
        float randomHue;
        float randomValue;
        int colorMode;
        // Artistic controls
        float tint = 0.0f;
        Vec3 tintColor = Vec3(1, 1, 1);
        float specularTint = 0.0f;
        float diffuseSoftness = 0.5f;
        Vec3 coatTint = Vec3(1, 1, 1);
        // Emission
        Vec3 emission = Vec3(0, 0, 0);
        float emissionStrength = 0.0f;
        // Root-tip gradient
        bool enableRootTipGradient = false;
        Vec3 tipColor = Vec3(0.6f, 0.4f, 0.25f);
        float rootTipBalance = 0.5f;
        // Textures
        int64_t albedoTexture = -1;
        int64_t roughnessTexture = -1;
        int64_t scalpAlbedoTexture = -1;
        Vec3 scalpBaseColor = Vec3(0.5f);
    };
    
    virtual void uploadHairMaterials(const std::vector<HairMaterialData>& materials) = 0;
    
    // ========================================================================
    // Texture Upload
    // ========================================================================
    
    virtual int64_t uploadTexture2D(
        const void* data,
        uint32_t width,
        uint32_t height,
        uint32_t channels,
        bool sRGB,
        bool isFloat = false
    ) = 0;

    /**
     * @brief Upload 3D texture (e.g. NanoVDB grid as VkImage or SSBO for volume rendering)
     * @param data    Raw voxel data (float or uint8)
     * @param width/height/depth  Grid resolution
     * @param channels  1=density, 4=RGBA
     * @param isFloat   true → float32 per channel
     * @return Opaque texture handle (backend-specific)
     *
     * Default no-op for backends that don't support 3D textures (OptiX uses NanoVDB SSBO path).
     * Vulkan backend overrides this for VkImage3D allocation.
     */
    virtual int64_t uploadTexture3D(
        const void* data,
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        uint32_t channels,
        bool isFloat = false
    ) {
        (void)data; (void)width; (void)height; (void)depth; (void)channels; (void)isFloat;
        return 0;
    }

    virtual void destroyTexture(int64_t textureHandle) = 0;
    
    // ========================================================================
    // Rendering
    // ========================================================================
    
    /**
     * @brief Set render parameters
     */
    virtual void setRenderParams(const RenderParams& params) = 0;
    
    /**
     * @brief Set camera for rendering
     */
    virtual void setCamera(const CameraParams& camera) = 0;
    
    /**
     * @brief Set current simulation time (for wind/animation)
     */
    virtual void setTime(float time, float deltaTime) = 0;

    /**
     * @brief Update instance transforms (optimized path)
     */
    virtual void updateInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects) = 0;

    /**
     * @brief Update a single instance transform by ID (for wind/foliage animation)
     */
    virtual void updateInstanceTransform(int instance_id, const float transform[12]) { (void)instance_id; (void)transform; }

    /**
     * @brief Set status callback for background operations
     */
    virtual void setStatusCallback(std::function<void(const std::string&, int)> callback) = 0;

    /**
     * @brief Get native command queue (CUstream for OptiX, VkQueue for Vulkan)
     */
    virtual void* getNativeCommandQueue() = 0;
    
    /**
     * @brief Set wind shader parameters (direction, strength, speed, time)
     */
    virtual void setWindParams(const Vec3& direction, float strength, float speed, float time) {
        (void)direction; (void)strength; (void)speed; (void)time;
    }

    /**
     * @brief Execute one render pass
     * @param accumulate If true, blend with previous passes
     */
    virtual void renderPass(bool accumulate = true) = 0;

    /**
     * @brief Execute a progressive render pass (Commonly used for viewport interactive rendering)
     */
    virtual void renderProgressive(void* outSurface, void* outWindow, void* outRenderer, 
                                  int width, int height, void* outFramebuffer, void* outTexture) = 0;
    
    /**
     * @brief Download rendered image to CPU
     * @param outPixels Output buffer (RGBA float or uint8 depending on format)
     */
    virtual void downloadImage(void* outPixels) = 0;
    
    virtual int getCurrentSampleCount() const = 0;
    
    /**
     * @brief Check if target sample count is reached
     */
    virtual bool isAccumulationComplete() const = 0;
    
    // ========================================================================
    // Environment
    // ========================================================================
    
    virtual void setEnvironmentMap(int64_t hdrTextureHandle) = 0;
    virtual void setSkyParams() = 0;
    
    // Add light parameters
    virtual void setLights(const std::vector<std::shared_ptr<Light>>& lights) = 0;
    
    // Set environment/world data (generic properties like sun direction, etc.)
    virtual void setWorldData(const void* worldData) = 0;
    
    // Updates VDB volume buffer for GPU ray marching
    virtual void updateVDBVolumes(const std::vector<GpuVDBVolume>& volumes) = 0;

    // Updates Gas volume buffer (Legacy path)
    virtual void updateGasVolumes(const std::vector<GpuGasVolume>& volumes) = 0;
    
    // ========================================================================
    // Utility
    // ========================================================================
    
    virtual void waitForCompletion() = 0;
    virtual void resetAccumulation() = 0;
    
    /**
     * @brief Get estimated render time per sample (for UI feedback)
     */
    virtual float getMillisecondsPerSample() const = 0;
};

// ============================================================================
// Backend Factory
// ============================================================================

/**
 * @brief Create render backend
 * @param preferredType Preferred backend (AUTO for best available)
 * @return Unique pointer to backend instance
 */
std::unique_ptr<IBackend> createBackend(BackendType preferredType = BackendType::AUTO);

/**
 * @brief Get list of available backends
 */
std::vector<BackendInfo> enumerateBackends();

/**
 * @brief Check if specific backend is available
 */
bool isBackendAvailable(BackendType type);

} // namespace Backend

#endif // I_BACKEND_H
