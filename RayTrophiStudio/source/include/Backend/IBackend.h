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
#include "RayFusion/ProbeBounce.h"
#include "RayFusion/ProbeField.h"
#include "RayFusion/ScreenGi.h"
#include "RayFusion/Reflection.h"
#include "Vec2.h"
#include "Matrix4x4.h"
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include "Hittable.h" // For Hittable definition
#include "light.h"
#include "Viewport/RasterFrameTelemetry.h"
#include "Viewport/RasterStageTimings.h"
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

struct MaterialPreviewIblStatus {
    bool supported = false;
    bool ready = false;
    // Which producer filled the irradiance/prefilter slots: "none", "hdri" or
    // "sky". A reader that only sees ready=true cannot tell whether Physical
    // Sky took the prefiltered path or fell back to the per-fragment cone.
    std::string source = "none";
    bool sky_capture_supported = false;
    // ★★★★ ARKA PLANIN GERCEKTE CIZDIGI SEY -- `ready` degil. Sky shader'inin
    //   kapisi ile AYNI girdilerden hesaplanir (`m_envTexID` + yuklu goruntu,
    //   atmosfer LUT view/sampler'lari), yani "HDRI secili" ile "HDRI cizildi"
    //   ayrisirsa BU alan soyler. Degerler:
    //     "color"                 — dunya modu duz renk (istenen sonuc)
    //     "hdri"                  — HDRI dokusu bagli ve ciziliyor
    //     "sky"                   — Physical Sky LUT'lari bagli ve ciziliyor
    //     "solid_fallback"        — mod HDRI ama doku YOK -> duz renk ciziliyor
    //     "sky_analytic_fallback" — mod Physical Sky ama LUT YOK -> kaba gradyan
    //   Son ikisi arizanin adidir; "ready=false" onlari ayirt edemiyordu.
    // ★ Adapter GIRDILERE gore raporlar. Preset kapisi (three_point) bu
    //   gecisi tamamen kapatabilir; onu API katmani "not_drawn" olarak
    //   soyler, cunku preset bu adapter'in bilgisi degil.
    std::string background_source = "color";
};

// RayFusion probe field, as the GPU consumer sees it. Reported so the swap
// from "one global sky texture" to "a probe field" is measurable rather than
// believed: the render looks the same either way, on purpose.
struct RayFusionProbeStatus {
    bool overlay_requested = false, overlay_ready = false, follow_camera = false;
    uint32_t overlay_markers = 0;
    std::string overlay_reason;
    RayFusion::BounceStatus bounce;
    bool supported = false;   // GPU buffers exist
    bool configured = false;  // the CPU field has a grid and a revision
    bool uploaded = false;    // at least one slot has reached the GPU
    bool bound = false;       // the preview descriptor set points at them
    // What actually RAN, never what was requested -- an instrument that echoes
    // the request cannot show the request being refused. "sky_bake" (step 1a),
    // "traced" (step 1b), or "none".
    std::string producer = "none";
    bool producer_traced_requested = false;
    std::string producer_reason;       // why a traced request was not honoured
    std::string budget_preset;         // which quality budget is scheduling
    uint64_t producer_signature = 0;
    // Traced-producer measurements. hit_fraction is the share of probe rays
    // that hit anything: if it stays 0 while geometry is clearly in range, the
    // rays are not seeing the scene at all -- and the result then looks exactly
    // like the sky-only producer, which is the failure nobody reports.
    float hit_fraction = 0.0f;
    uint32_t rejected_inside = 0;      // probes born inside geometry
    // ★★★★ "Katinin icinde" kararini ARKA YUZ ORANI tek basina veremez.
    //   Olculdu 2026-09-15: bu odanin arka yuz orani %84,5 -- yani esik %35
    //   iken odanin TAM ORTASINDAKI probe da "katinin icinde" sayilip
    //   atiliyordu, ve belirti "problar kirmizi" oluyordu. Duvara gomulu bir
    //   probe ile oda ortasindaki probe arka yuz oraniyla AYNI gorunur;
    //   ayiran sey MESAFE: gomulunun etrafindaki geometri santimetrelerde,
    //   odadakinin metrelerde.
    // ★★★ backface_enclosed duzeltmenin KANITIDIR: "kirmizi probe kalmadi"
    //   hem "kapi calisti" hem "hic probe yayinlanmadi" demek olabilirdi.
    //   Bu sayac ikisini ayirir -- yuksekse kapi gercekten ates ediyor.
    uint32_t backface_enclosed = 0;    // cok arka yuz AMA uzak: ic mekan, kati degil
    float mean_hit_distance = 0.0f;    // dunya birimi; esigi olcuyle karsilastir
    double trace_ms = 0.0;
    uint64_t traced_publishes = 0;
    uint32_t total = 0, valid = 0, pending = 0, in_flight = 0;
    uint64_t accepted = 0, rejected = 0;
    // The APPLIED window, read back from the field itself rather than from the
    // request that asked for it. counts/minimum are cells, spacing world units.
    uint32_t counts[3]{};
    int32_t minimum[3]{};
    float spacing = 0.0f;
    uint32_t max_slots = 0;   // ceiling on counts.x*y*z, fixed by the GPU buffer
    // ★★★★ Izgara artik SAHNEYE otomatik oturuyor. `auto_fit_mode` HANGI
    //   REJIMIN kostugunu soyler ("scene" | "camera_local" | "off") ve bu bir
    //   ayrinti degil: ayni kaba izgara iki farkli sebepten dogabilir, ve
    //   ikisinin duzeltmesi farklidir. Sebep metni de tasinir, cunku moda
    //   bakip "neden" diye tahmin etmek tam olarak kacindigimiz sey.
    bool auto_fit = false;
    std::string auto_fit_mode = "off";
    std::string auto_fit_reason;
};

// The acceleration structure the raster viewport keeps resident. This is the
// bill RayFusion pays whether or not a ray is traced this frame, so it is
// reported as bytes and milliseconds rather than as a boolean.
struct RayFusionSceneASStatus {
    bool hardware_rt = false;
    bool ready = false;
    uint32_t blas_count = 0;
    uint32_t instance_count = 0;
    uint32_t instances_skipped = 0;
    uint32_t meshes_skipped = 0;
    uint64_t as_bytes = 0;
    double last_build_ms = 0.0;
    uint64_t builds = 0;
    uint64_t built_geometry_generation = 0;
    // What the gate actually watches. The global generation is reported too,
    // but it is NOT the gate: scene.delete leaves it unchanged.
    uint64_t geometry_signature = 0;
    uint64_t instance_signature = 0;
    double signature_ms = 0.0;
    uint64_t tlas_only_refreshes = 0;
    // ★★★★★ TRACE EDILEN sahnenin dunya kutusu -- cizilen sahnenin degil.
    //   Ayni donguden, ayni `continue`'lardan SONRA toplanir: gizli (mask==0)
    //   ve cap ile elenen instance'lar TLAS'ta yoksa bu kutuda da yoktur.
    //   Ayri bir dongu yazmak, probe izgarasini ISINLARIN GORMEDIGI bir
    //   hacme oturtur ve belirti "izgara dogru yerde ama problar bos" olurdu.
    // ★★★ Gecerlilik AYRI bir alan: (0,0,0)-(0,0,0) mesru bir kutudur (tek
    //   nokta), yani "olculmedi"yi deger kumesine kodlamak olurdu.
    bool world_bounds_valid = false;
    float world_min[3]{};
    float world_max[3]{};
    // Instances the raster draw hides (mask 0) and the AS therefore excludes.
    // scene.delete produces these: it hides, it does not erase.
    uint32_t instances_hidden = 0;
    // ★★★★ Of blas_count, how many were built from a WELDED (indexed) raster
    //   mesh. This exists because the failure it guards is invisible: building
    //   a welded mesh without its index buffer produces a full, "ready" AS
    //   whose triangles are fabricated from storage order. Every counter stayed
    //   green while the traced world was wrong. A scene whose large static
    //   meshes are welded (terrain, imported props) MUST report a non-zero
    //   number here; 0 in such a scene means the triangulation was dropped
    //   again. blas_flat is its complement -- genuine flat SoA, one vertex per
    //   corner -- so the two must sum to blas_count.
    uint32_t blas_indexed = 0;
    uint32_t blas_flat = 0;
    // ★★★★ Of blas_count, how many were built from a GPU-SKINNED raster mesh
    //   and are therefore re-fit as the character deforms. This exists because
    //   the failure it guards is invisible in exactly the same way blas_indexed
    //   guards its own: GPU skinning rewrites the CONTENTS of the borrowed
    //   vertex buffer while its handle, address and vertex count all stay put,
    //   so a "ready" AS with every counter green can be describing the pose of
    //   the frame it was built in. A scene with an animated character MUST
    //   report a non-zero number here; 0 in such a scene means the deforming
    //   mesh is casting a frozen shadow again.
    //
    //   skin_refits is the number that shows it is actually HAPPENING, not just
    //   possible: it has to climb while the timeline plays and stay put while
    //   it is paused. Frozen at 0 with blas_skinned > 0 means the generation
    //   never reached the gate. IPC WRITE-THEN-MEASURE applies -- put a frame
    //   between stepping the timeline and reading this.
    uint32_t blas_skinned = 0;
    uint64_t skin_refits = 0;
    uint64_t skin_refit_failures = 0;
    // The total, and the three parts it is made of. Read the PARTS: the total
    // cannot be acted on because each part has a different fix -- the drain is
    // the GPU still owing us a frame, the BLAS figure is one submit plus a
    // fence wait around the refits, and the TLAS figure is a full top-level
    // rebuild that drains a second time.
    double last_skin_refit_ms = 0.0;
    double last_skin_drain_ms = 0.0;
    double last_skin_blas_ms = 0.0;
    double last_skin_tlas_ms = 0.0;
    // Released because the viewport shows Rendered (see yieldRayFusionSceneAS).
    bool yielded = false;
    uint64_t yields = 0;
    // Driver-reported device-local VRAM for the whole PROCESS (all devices).
    // vram_measured=false means the zeros are not a reading.
    bool vram_measured = false;
    uint64_t vram_usage_bytes = 0;
    uint64_t vram_budget_bytes = 0;
    std::string inactive_reason;
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
    // Debug Visualizer (Vulkan RT) — exclusive false-color views; mode table in raygen CameraPC
    int   debugView = 0;
    float debugExposure = 1.0f;
    float debugOverlay = 0.0f;         // 0 = pure debug … 1 = pure beauty
    // ★★★ Realtime (raster) depth of field. The STRENGTH is not here: the blur
    //   radius comes from the camera's own aperture and focus distance, exactly
    //   as the path tracer's lens does, so Rendered and Realtime cannot drift.
    //   These three are the COST knobs plus the master switch.
    //   ★ With aperture 0 (the default camera) the pass early-outs to a plain
    //   tone map, so leaving this enabled costs nothing until a lens is dialled.
    bool  realtimeDepthOfField = true;
    float realtimeDofMaxCoC = 24.0f;   // pixels; caps both blur size and cost
    int   realtimeDofMaxTaps = 32;     // gather samples at the maximum radius
    // ★★★★ Realtime raster TAA. `Samples` is a STOPPING CONDITION, not a
    //   quality dial: the viewport keeps asking for frames until it has that
    //   many jittered samples, then stops. That is also the honest place to
    //   read its cost -- a still scene converges and then goes idle.
    bool  realtimeTaa = true;
    int   realtimeTaaSamples = 16;
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
    // Cubic B-spline tessellation level (Vulkan RT). 0 = raw chord per span (linear,
    // kinked curls); >0 = each span sampled into (1<<subdivisions) short LSS sub-segments
    // so the silhouette follows the curve. Set from HairGenerationParams.subdivisions.
    uint32_t subdivisions = 0;
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
     * @param cornerAttribs Optional Attribute-node channels for this EXPANDED soup:
     *        triangles.size() * 3 * kMatAttribSlots floats, corner-major, i.e. the value
     *        for corner c of triangle t at [(t*3 + c) * kMatAttribSlots + slot]. Corner-major
     *        because the soup BLAS has no index buffer — its vertex id IS the corner id.
     *        Null (the default, and what a source with no SoA link can supply) uploads no
     *        block, and the shader then reads every channel as 0 = unpainted — which is
     *        exactly what the CPU reads for the same geometry, so the two stay in step.
     * @return Mesh handle/index
     */
    virtual uint32_t uploadTriangles(
        const std::vector<TriangleData>& triangles,
        const std::string& meshName,
        const std::vector<float>* cornerAttribs = nullptr
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
    
    virtual void updateSceneGeometry(const std::vector<std::shared_ptr<Hittable>>& objects, const std::vector<Matrix4x4>& boneMatrices) = 0;
    virtual bool updateFlatMeshBLAS(const std::string& /*nodeName*/, const class TriangleMesh* /*mesh*/) { return false; }

    /**
     * @brief Was the uploaded geometry built with the per-vertex pointiness attribute?
     *
     * Only meaningful where a material program can read the Geometry node's Pointiness
     * output (Vulkan RT). A graph that newly reads it needs one geometry re-upload; the
     * node editor asks here instead of rebuilding on every edit. Backends without the
     * attribute answer true so nothing tries to re-upload for them.
     */
    virtual bool geometryHasPointiness() const { return true; }

    /**
     * @brief Was the uploaded geometry built with the per-vertex named-attribute block?
     * Same one-time re-upload gate as geometryHasPointiness, for the Attribute node.
     */
    virtual bool geometryHasAttributes() const { return true; }

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
        int  sssMethod = 0;           // 0 = random walk, 1 = fast (Lambert, no walk)
        int  sssWalkMaxSteps = 64;    // walk hard cap, [8, 256]

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

        // Closed-mesh volume material.  The triangle boundary remains in the
        // regular BLAS; renderers treat front faces as medium entry and back
        // faces as exit, integrating over the real segment length.
        bool  isVolume = false;
        float volumeDensity = 0.0f;
        float volumeAbsorption = 0.0f;
        float volumeScattering = 0.0f;
        float volumeAnisotropy = 0.0f;
        float volumeStepSize = 0.05f;
        int   volumeMaxSteps = 128;
        float volumeNoiseScale = 1.0f;
        float volumeMultiScatter = 0.0f;
        int   volumeLightSteps = 4;
        float volumeShadowStrength = 0.8f;

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
        float resin_shard = 0.0f;                  // colored glass-shard amount
        float resin_shard_hue = -1.0f;             // base hue 0..1; <0 = rainbow palette
        bool  resin_object_space = true;           // interior anchored to the object (vs world)
        int   dust_style = 0;                      // 0=Nebula(auto) 1=Billow 2=Wispy 3=Paint swirl
        Vec3  dust_color_a = Vec3(1.0f, 1.0f, 1.0f);
        Vec3  dust_color_b = Vec3(1.0f, 1.0f, 1.0f);
        int   shard_shape = 0;                     // 0=round chips 1=faceted crystals
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
    static constexpr uint32_t MAT_FLAG_WATER_LAKE = (1u << 22); // Fetch-limited inland water profile
    static constexpr uint32_t MAT_FLAG_WATER_RIVER = (1u << 23); // UV-flow-aligned river profile
    static constexpr uint32_t MAT_FLAG_VOLUME = (1u << 24); // Closed triangle mesh bounds a volume medium

    /**
     * @brief Per-terrain layer descriptor for splat-map based blending.
     *        Used by the Vulkan backend (binding 12).
     */
    /// Eight layer slots: 0-3 splat-weighted (normalized partition of the
    /// surface), 4-7 semantic overlays (Flow/Wetness/Ice/Hardness) composited
    /// over that blend by their own unnormalized weight.
    struct TerrainLayerData {
        uint32_t layer_mat_id[8]   = {0, 0, 0, 0, 0, 0, 0, 0};
        float    layer_uv_scale[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        float    overlayStrength[4] = {1, 1, 1, 1};  // Artist dial for slots 4-7
        /// Bit s set means semantic slot 4+s carries a material. Material id 0
        /// is valid, so a bound slot cannot be detected from layer_mat_id.
        uint32_t overlayMask       = 0;
        int64_t  splatMapTexture   = 0;              // Texture pointer/handle for RGBA splat map
        uint32_t layer_count       = 0;              // Active splat layer count (0 = no terrain)
        int64_t  macroColorTexture = 0;              // Texture pointer/handle for macro color map
        float    macroColorStrength= 0.0f;           // Blend strength [0.0 - 1.0]
        int64_t  semanticMapTexture= 0;              // R=Flow G=Wetness B=Ice A=Hardness
        float    semanticWetDarkening = 0.28f;
        float    semanticWetRoughness = 0.65f;
    };

    virtual void uploadMaterials(const std::vector<MaterialData>& materials) = 0;

    /// Faz 2b: upload the flattened per-pixel material-program buffer (produced by
    /// MaterialNodesV2::flattenMaterialPrograms) indexed 1:1 with the material
    /// list just uploaded. `words` is the raw uint stream; empty clears programs.
    /// Default no-op — only the Vulkan RT backend interprets it (OptiX/CPU ignore).
    virtual void uploadMaterialPrograms(const std::vector<uint32_t>& words) { (void)words; }

    virtual bool updateMaterial(uint32_t materialIndex, const MaterialData& material) {
        (void)materialIndex;
        (void)material;
        return false;
    }

    /// Monotonic counter bumped whenever the backend purges its uploaded-texture
    /// cache (every GPU texture id handed out before the purge becomes invalid).
    /// Callers that upload tables containing texture ids (material SSBO, program
    /// stream) compare it across calls to detect a mid-sequence purge and re-upload.
    virtual uint64_t textureCacheGeneration() const { return 0; }

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
        float selfShadow = 1.0f;   // deep hair self-shadow strength (Vulkan RT): 0=off, 1=full
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

    /**
     * @brief Force the raster/Realtime viewport to publish the frame it just
     *        rendered instead of the newest already-completed one.
     *
     * ★★★ The asynchronous frame ring (Realtime roadmap Faz 0.5a) presents an
     * older completed slot so the host never blocks. That is right for a human
     * dragging the camera and WRONG for an automated probe: the pixels a script
     * reads would belong to a frame recorded before its last scene edit, and
     * nothing in the image would say so. rtapi turns this on whenever viewport
     * capture is enabled, trading latency for a defensible measurement.
     */
    virtual void setInteractiveViewportSynchronousPresent(bool enabled) { (void)enabled; }

    /**
     * @brief Read the last raster viewport frame's presentation telemetry.
     * @return false when this backend has no raster frame ring at all. The out
     *         parameter is then absence, NOT a measurement of zero.
     */
    virtual bool getInteractiveViewportFrameTelemetry(RasterFrameTelemetry& out) const {
        (void)out;
        return false;
    }
    virtual bool getRayFusionSceneASStatus(RayFusionSceneASStatus& out) const {
        (void)out;
        return false;
    }
    virtual bool getRayFusionProbeStatus(RayFusionProbeStatus& out) const {
        (void)out;
        return false;
    }
    /**
     * @brief Choose the probe producer: traced (step 1b) or the sky bake (1a).
     * This exists so the two can be compared on the SAME scene. A producer that
     * cannot be switched off cannot be measured against the one it replaced,
     * and "it looks different" is not a measurement.
     */
    virtual void setRayFusionProbeProducer(bool traced) { (void)traced; }
    virtual bool setRayFusionProbeBounce(bool enabled) { (void)enabled; return false; }
    virtual RayFusion::ScreenGiStatus screenGiStatus() const { return {}; }
    virtual bool setScreenGi(const RayFusion::ScreenGiSettings&, std::string& error) { error = "no screen GI backend"; return false; }
    // ★★★ Piksel basina spekuler yansima. `metallic` ile SINIRLI DEGIL: kapi
    //   split-sum agirligina (F0*brdf.x + brdf.y) kurulu, o da bir Fresnel
    //   terimi -- verniklenmis ahsap, boyali zemin, seramik ve plastik ayni
    //   yoldan gecer.
    virtual RayFusion::ReflectionStatus reflectionStatus() const { return {}; }
    virtual bool setReflection(const RayFusion::ReflectionSettings&, std::string& error) { error = "no reflection backend"; return false; }
    virtual bool setRayFusionProbeOverlay(bool enabled) { (void)enabled; return false; }
    virtual bool setRayFusionProbeFollowCamera(bool enabled) { (void)enabled; return false; }
    /**
     * @brief A/B lever for the global instance buffer + GPU culling path.
     * False forces the per-mesh fallback: no GPU culling, no scatter LOD
     * proxies, and a full frame-ring drain on every visible-set change. That
     * fallback is what the realtime viewport ran unconditionally until
     * 2026-09-08, so it has to stay selectable or the fix cannot be measured
     * against the thing it replaced. Read the outcome from
     * viewport.frame_telemetry: gpu_culling, visible_triangles, resource_drains.
     */
    virtual bool setRasterGpuInstancing(bool enabled) { (void)enabled; return false; }
    virtual bool rasterGpuInstancingAllowed() const { return false; }
    // Alfa-test'li derinlik on gecisi. Iki isi birden yapar: overdraw
    // golgelendirmesini erken-Z ile eler, ve RT golge isininin cikis noktasi
    // olan piksel basina derinligi uretir.
    // ★ Kapatilabilir birakildi: kapatilamayan bir duzeltme, kendisini
    //   yargilayacak olcumu de oldurur.
    virtual bool setRasterDepthPrepass(bool enabled) { (void)enabled; return false; }
    virtual bool rasterDepthPrepassAllowed() const { return false; }
    // Same-frame directional RT shadows with alpha cutouts and cascade fallback.
    virtual bool setRtShadow(bool enabled) { (void)enabled; return false; }
    virtual bool rtShadowAllowed() const { return false; }
    // `cascadesReplaced` is how many cascade shadow VIEWS the ray pass took
    // over this frame -- the only number that can show the cost actually moved.
    // Rays rising while this stays 0 means both paths are running, which is what
    // "the cost did not drop" looked like before the handoff existed.
    virtual bool getRtShadowStatus(bool& supported, bool& ready, uint32_t& rays,
                                   uint32_t& cascadesReplaced,
                                   std::string& reason) const {
        supported = false; ready = false; rays = 0; cascadesReplaced = 0;
        reason = "backend has no RT shadow pass";
        return false;
    }
    // ── Per-pass raster frame timing ────────────────────────────────────────
    // ★★★ The window is a MEASUREMENT, so it is reset and read explicitly
    //   rather than accumulating for the life of the session: a lifetime mean
    //   silently averages every configuration the session passed through, and
    //   this app's settings change what a frame does, not just how fast it is.
    //   Drive the camera between the reset and the read -- the viewport renders
    //   only when it is marked dirty, so a still camera measures nothing and
    //   says so.
    virtual bool getRasterStageTimings(RasterStageTimings& out) const {
        (void)out; return false;
    }
    virtual bool resetRasterStageTimings() { return false; }

    /**
     * @brief Move or reshape the probe window at RUNTIME.
     * Coverage is the one input every RayFusion image question so far has been
     * confounded by (a field nailed to the world origin, half of it outdoors),
     * and a parameter that needs a rebuild to change cannot be A/B'd at all.
     * Partial: unsent fields keep their current value. Fail-closed -- on a
     * rejected request nothing changes and @p error says which field was wrong.
     */
    virtual bool setRayFusionProbeGrid(const RayFusion::GridRequest& request,
                                       std::string& error) {
        (void)request;
        error = "no viewport backend owns a probe field";
        return false;
    }
    virtual bool getMaterialPreviewIblStatus(MaterialPreviewIblStatus& out) const {
        out = {};
        return false;
    }

    // ★★★★★ TAA, ayarindan degil BIRIKEN ORNEK sayisindan olculur.
    //   "enabled" yalnizca istegi soyler; goruntunun gercekten yumusayip
    //   yumusamadigini soyleyen tek sayi `accumulated`. Ikisi ayrildi cunku
    //   hat dort ayri yerde sessizce kapanabiliyor: shader yok, pipeline
    //   kurulamadi, gecmis tahsisi basarisiz, ya da mod raster degil --
    //   dordu de "enabled = true" ile birlikte yasayabilir.
    struct RasterTaaStatus {
        bool supported = false;     // pipeline + gecmis goruntuleri var
        bool enabled = false;       // istenen
        uint32_t target_samples = 0;
        uint32_t accumulated_samples = 0; // OLCUM: su ana kadar biriken
        bool converged = false;
        double last_ms = 0.0;       // gecen karedeki dispatch + kopya
        std::string inactive_reason;
    };
    virtual bool getRasterTaaStatus(RasterTaaStatus& out) const {
        out = {};
        out.inactive_reason = "this backend has no raster viewport";
        return false;
    }
    
    /**
     * @brief Resolve a CPU texture pointer to a GPU bindless texture array index.
     *        Used by material program flattening to embed hardware texture IDs into VM instructions.
     */
    virtual uint32_t resolveTextureHandle(int64_t texturePtr, int textureType = 0, bool forceLinear = false, bool preferSingleChannel = false) { 
        (void)texturePtr; (void)textureType; (void)forceLinear; (void)preferSingleChannel; return 0; 
    }
    
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
