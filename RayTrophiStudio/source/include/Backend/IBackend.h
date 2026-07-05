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

// GPU stylize request types (StylizeKernel.h / StylizeCore.h). Forward declared
// so this widely-included interface does not pull CUDA headers; the OptiX
// backend includes the full definitions in its .cpp.
namespace StylizeGPU { struct KernelParams; }
namespace StylizeCore { struct StyleProfileCore; }

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

enum class ViewportMode : uint8_t {
    Rendered = 0,
    Solid,
    MaterialPreview,
    Matcap
};

struct BackendInfo {
    BackendType type;
    std::string name;
    std::string deviceName;
    std::string driverVersion;
    bool hasHardwareRT;
    uint64_t vramBytes;
};

struct GpuMemoryStats {
    uint64_t totalBytes = 0;
    uint64_t freeBytes = 0;
    uint64_t usedBytes = 0;
    uint64_t trackedTextureBytes = 0;
    uint64_t trackedTextureBytesThisBackend = 0;
    bool hasDeviceUsage = false;
    bool hasTrackedTextures = false;
};

// ============================================================================
// Common Data Structures
// ============================================================================

struct RenderParams {
    int imageWidth;
    int imageHeight;
    int samplesPerPixel;
    int minSamples;
    int maxBounces;
    int diffuseBounces = 4;
    int transmissionBounces = 8;
    int currentPass;
    int frameNumber;
    bool useAdaptiveSampling;
    float adaptiveThreshold;
    // Photon caustics (Faz 2 — Vulkan RT only for now)
    bool  causticsEnabled = false;
    bool  causticsDebug = false;       // visualize the photon grid instead of shading
    int   causticsPhotons = 262144;    // photons per frame
    float causticsCellSize = 0.05f;    // hash-grid cell size (world units)
    float causticsEnergy = 1.0f;       // photon power calibration knob
    bool  causticsVolumetric = false;  // Faz 2V: volumetric caustic shafts in media
    bool  causticsVolDebug = false;    // debug: march-visualize the volume grid
    float causticsVolStrength = 1.0f;  // sigma_s scatter coefficient knob
    bool  causticsVolDirect = false;   // also deposit the light->glass leg (direct shafts)
    float causticsVolNoise = 0.0f;     // heterogeneous dust turbulence amount (0..1)
};

struct DenoiserFrameData {
    int width = 0;
    int height = 0;
    const float* color = nullptr;
    const float* albedo = nullptr;
    const float* normal = nullptr;
    // Stylize AOV (optional): stride 4 — x,y,z = primary-hit world position, w = encoded
    // material id (0 = miss, 1 = hit/unknown material, >=2 → material index = w - 2).
    // Linear depth is reconstructed host-side from world position + camera origin.
    // Bottom-up layout, matching color/albedo/normal. Non-null only when the backend
    // produces it (Vulkan RT with the position image).
    const float* position = nullptr;
};

// GPU-side denoiser input. Pointers are device memory on the backend's CUDA device.
// Used by Renderer to run OIDN (CUDA) with zero host round-trips on inputs.
struct DenoiserFrameDataGPU {
    int width = 0;
    int height = 0;
    void* colorDevPtr = nullptr;         // float4* (pixel stride = sizeof(float4))
    void* albedoDevPtr = nullptr;        // optional
    void* normalDevPtr = nullptr;        // optional
    size_t pixelByteStride = 0;          // bytes per pixel in device layout
    size_t rowByteStride = 0;            // bytes per row (pixelByteStride * width for packed)
    void* cudaStream = nullptr;          // cudaStream_t (nullable; caller ensures ordering)
    int cudaDeviceOrdinal = -1;          // -1 = use current CUDA device
};

struct CameraParams {
    Vec3 origin;
    Vec3 lookAt;
    Vec3 up = Vec3(0, 1, 0);
    float fov = 60.0f;
    float aperture = 0.0f;
    float focusDistance = 1.0f;
    float aspectRatio = 1.777f;

    // Orthographic / standard-view state (viewport alignment).
    bool  orthographic = false;      // true = parallel projection in the raster preview
    float orthoHeight = 10.0f;       // full vertical world-units visible (used when orthographic)
    int   gridPlane = 0;             // active grid plane: 0=XZ (floor), 1=XY (front), 2=YZ (side)

    float exposureFactor = 1.0f;
    float ev_compensation = 0.0f;
    int isoPresetIndex = -1;
    int shutterPresetIndex = -1;
    int fstopPresetIndex = -1;
    bool autoAE = false;
    bool usePhysicalExposure = false;
    bool motionBlurEnabled = false;
    bool vignettingEnabled = false;
    bool chromaticAberrationEnabled = false;
    
    // Pro Features
    float distortion = 0.0f;
    float lens_quality = 1.0f;
    float vignetting_amount = 0.0f;
    float vignetting_falloff = 1.0f;
    float chromatic_aberration = 0.0f;
    float chromatic_aberration_r = 1.0f;
    float chromatic_aberration_b = 1.0f;
    int camera_mode = 0;
    int blade_count = 6;
    
    // Shake / Handheld
    bool shake_enabled = false;
    float shake_intensity = 0.0f;
    float shake_frequency = 1.0f;
    float handheld_sway_amplitude = 0.0f;
    float handheld_sway_frequency = 1.0f;
    float breathing_amplitude = 0.0f;
    float breathing_frequency = 1.0f;
    bool enable_focus_drift = false;
    float focus_drift_amount = 0.0f;
    int operator_skill = 0;
    bool ibis_enabled = false;
    float ibis_effectiveness = 0.0f;
    int rig_mode = 0;
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
    virtual GpuMemoryStats getMemoryStats() const { return {}; }
    
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
        float specular = 0.5f;
        Vec3 emission = Vec3(0.0f);
        float emissionStrength = 0.0f;
        float ior = 1.5f;
        float transmission = 0.0f;
        float dispersion = 0.0f;  // spectral dispersion strength (0 = off)
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
        float normalStrength = 1.0f;
        float sheen = 0.0f;
        float sheenTint = 0.0f;

        // Texture handles (backend-specific opaque handles)
        int64_t albedoTexture = 0;
        int64_t normalTexture = 0;
        int64_t roughnessTexture = 0;
        int64_t metallicTexture = 0;
        int64_t specularTexture = 0;
        int64_t emissionTexture = 0;
        int64_t transmissionTexture = 0;
        int64_t opacityTexture = 0;
        int64_t heightTexture = 0;

        // Packed-texture channel override for metallic/roughness slots:
        // 0 = Auto (ORM: rough .g / metal .b, BC4/R8 → .r), 1 = R, 2 = G, 3 = B.
        int metallicTexChannel = 0;
        int roughnessTexChannel = 0;

        // Padding/flags for future use
        uint32_t flags = 0;
        uint32_t terrainLayerIdx = 0; // Index into terrain layer buffer (valid when MAT_FLAG_TERRAIN set)

        // Thin-shell BUBBLE (champagne / soda / soap-foam close-up). Carried from
        // the host gpuMaterial so the backend can set GPU_MAT_FLAG_BUBBLE +
        // bubble_ior/bubble_film on the final GpuMaterial.
        bool  is_bubble   = false;
        float bubble_ior  = 1.33f;
        float bubble_film = 0.0f;

        // Iridescent clearcoat (thin-film tint on the clearcoat lobe). 0 = plain white.
        float clearcoat_iridescence = 0.0f;
        float clearcoat_film_thickness = 0.55f;

        // Transmission interior absorption density (thick resin / glass-marble depth).
        // 0 = legacy constant-thickness glass tint; >0 = Beer-Lambert over real distance.
        float transmission_density = 0.0f;
        Vec3  resin_color = Vec3(1.0f, 1.0f, 1.0f); // resin absorption tint (separate from albedo)
        float resin_roughness = 0.1f;              // resin coat gloss (reflect lobe), independent of base
        float resin_inclusion = 0.0f;              // dust cloudiness (heterogeneous absorption)
        float resin_dirt = 0.0f;                   // opaque dirt-speck amount (early-return)
        float resin_inclusion_scale = 8.0f;        // procedural feature size
        Vec3  resin_dirt_color = Vec3(0.18f, 0.14f, 0.10f);
        bool  glass_marble_volume = false;         // full-volume marble medium march (raygen)

        // Procedural surface detail
        float micro_detail_strength = 0.0f;
        float micro_detail_scale    = 0.0f;
        float tile_break_strength   = 0.0f;
        // Water-specific parameters (maps to VkGpuMaterial Block 8 & Block 9)
        float foam_threshold        = 0.0f;
        float fft_ocean_size        = 0.0f;
        float fft_choppiness        = 0.0f;
        float fft_wind_speed        = 0.0f;
        float fft_wind_direction    = 0.0f;
        float fft_amplitude         = 0.0f;
        float fft_time_scale        = 0.0f;
        float micro_anim_speed      = 0.0f;
        float micro_morph_speed     = 0.0f;
        float foam_noise_scale      = 0.0f;

        // Material-space UV transform
        Vec2 uvScale = Vec2(1.0f, 1.0f);
        Vec2 uvOffset = Vec2(0.0f, 0.0f);
        Vec2 uvTiling = Vec2(1.0f, 1.0f);
        float uvRotationDegrees = 0.0f;
        uint32_t uvWrapMode = 0;
    };

    // Flag bits for MaterialData::flags
    static constexpr uint32_t MAT_FLAG_TERRAIN = (1u << 16); // Splat-blended terrain material
    static constexpr uint32_t MAT_FLAG_WATER   = (1u << 17); // Explicit water surface material
    static constexpr uint32_t MAT_FLAG_WATER_FFT_READY = (1u << 18); // Vulkan height/normal slots contain FFT textures

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
    virtual bool updateMaterial(uint32_t materialIndex, const MaterialData& material) {
        (void)materialIndex;
        (void)material;
        return false;
    }

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

    /**
     * @brief Set the interactive viewport matcap by uploaded texture ID (backend-specific)
     * @param textureID Opaque texture handle returned by uploadTexture2D
     */
    virtual void setInteractiveViewportMatcap(int64_t textureID) { (void)textureID; }
    /**
     * @brief Select a built-in matcap preset for the interactive viewport (0..9)
     * Preset mapping: 0=Solid clay, 1=User texture (unused here), 2..9=procedural presets
     */
    virtual void setInteractiveViewportMatcapPreset(int preset) { (void)preset; }
    /* custom matcap support removed */
    
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
     * @brief Select the viewport shading path used for interactive display.
     *        Backends may fall back to Rendered until a lighter path is implemented.
     */
    virtual void setViewportMode(ViewportMode mode) { (void)mode; }
    virtual ViewportMode getViewportMode() const { return ViewportMode::Rendered; }
    virtual bool supportsViewportMode(ViewportMode mode) const {
        return mode == ViewportMode::Rendered;
    }
    virtual bool updateInteractiveMesh(const std::string& nodeName,
                                       const std::vector<std::shared_ptr<class Triangle>>& triangles) {
        (void)nodeName;
        (void)triangles;
        return false;
    }
    
    /**
     * @brief Download rendered image to CPU
     * @param outPixels Output buffer (RGBA float or uint8 depending on format)
     */
    virtual void downloadImage(void* outPixels) = 0;
    // includeColor=false skips the (full-res) color image copy/download for callers that
    // only need the AOVs (e.g. the Stylize position/albedo/normal pull).
    virtual bool getDenoiserFrame(DenoiserFrameData& frame, bool useAuxiliary = true, bool includeColor = true) {
        (void)frame;
        (void)useAuxiliary;
        (void)includeColor;
        return false;
    }

    // GPU-direct variant: fills device pointers so Renderer can run OIDN (CUDA) without
    // downloading. Return false → caller must use getDenoiserFrame() host path.
    virtual bool getDenoiserFrameGPU(DenoiserFrameDataGPU& frame, bool useAuxiliary = true) {
        (void)frame;
        (void)useAuxiliary;
        return false;
    }

    // GPU stylize: run the stylize post-process on device (AOVs already resident,
    // graded color uploaded/downloaded), in place on the SDL surface (passed as
    // void* to keep SDL out of this header). Return false → caller falls back to
    // the CPU stylize path. Default no-op until a backend implements it.
    virtual bool applyStylizeGPU(void* surface,
                                 const StylizeGPU::KernelParams& params,
                                 const StylizeCore::StyleProfileCore& profile) {
        (void)surface;
        (void)params;
        (void)profile;
        return false;
    }
    
    virtual int getCurrentSampleCount() const = 0;
    
    /**
     * @brief Check if target sample count is reached
     */
    virtual bool isAccumulationComplete() const = 0;

    /**
     * @brief Returns true when the backend wants another viewport render pass
     * even if no new user input arrived yet.
     */
    virtual bool needsViewportRender() const { return false; }
    
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
