/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          OptixWrapper.h
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

// ===========================================================================
// THIS PATH IS FROZEN (2026-09-16). See docs/dev/OPTIX_DONDURULDU.md
//
//   The OptiX/CUDA path is in MAINTENANCE MODE: it takes no new features and
//   only preserves current behaviour. A capability that works only here is
//   INCOMPLETE -- Vulkan is the primary GPU path.
//
//   BUT IT CANNOT BE REMOVED, and the reason is counter-intuitive: a TDR does
//   NOT kill OptiX. A TDR resets only the Vulkan driver; raster and Vulkan RT
//   crash while the CUDA/OptiX context SURVIVES. That makes OptiX the real
//   lifeboat in this product. The assumption "same GPU, so it dies too" is
//   WRONG, and it is exactly the first guess anyone makes without reading this.
// ===========================================================================


#include <optix.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <mutex>
#include "Vec3.h"
#include "Ray.h"
#include "Hittable.h"
#include <Camera.h>
#include <Light.h>
#include "params.h"
#include "World.h"
#include <AreaLight.h>
#include <PointLight.h>
#include <DirectionalLight.h>
#include <SDL_surface.h>
#include <SDL_image.h>
#include <SDL.h>
#include "material_gpu.h"
#include "sbt_record.h"
#include <ColorProcessingParams.h>
#include <OpenImageDenoise/oidn.hpp>
#include <sbt_data.h>
#include "Matrix4x4.h"
#include <OptixAccelManager.h>

#include <OptixTypes.h>

// Forward declarations for TLAS/BLAS support
class Triangle;
class OptixAccelManager;
namespace Hair { class HairSystem; }

// GPU stylize (StylizeKernel.h / StylizeCore.h) — forward declared to keep this
// widely-included header light; the full headers are pulled in OptixWrapper.cpp.
namespace StylizeGPU { struct KernelParams; }
namespace StylizeCore { struct StyleProfileCore; }


class OptixWrapper {
public:
    OptixWrapper();
    void partialCleanup();
    void clearScene(); // Clears traversable handle for empty scenes
    ~OptixWrapper();

    void resizeBuffers(int w, int h);

    void initialize();
    bool isCudaAvailable();
    // OIDN denoising is now handled by Renderer::applyOIDNDenoising
    void validateMaterialIndices(const OptixGeometryData& data);
    struct PtxData {
        const char* raygen_ptx;
        const char* miss_ptx;
        const char* hitgroup_ptx;
    };
    void setupPipeline(const PtxData& ptx);
    void destroyTextureObjects();
    void buildFromData(const OptixGeometryData& data);
    void buildFromDataTLAS(const OptixGeometryData& data, const std::vector<std::shared_ptr<Hittable>>& objects);
    void updateGeometry(const std::vector<std::shared_ptr<Hittable>>& objects); // Auto-decides based on mode
    void updateTLASGeometry(const std::vector<std::shared_ptr<Hittable>>& objects, const std::vector<Matrix4x4>& boneMatrices = {}); // BLAS+TLAS update
    void updateTLASMatricesOnly(const std::vector<std::shared_ptr<Hittable>>& objects); // Transform-only update
    void refitSphereGroups(); // Foam point-sphere GAS in-place refit (motion path)
    // Rendering enters through OptixBackend, which drives this pair. The old
    // SDL-surface launch overload and launch_tile_based_progressive were removed:
    // the first only forwarded to launch(w, h) and had no callers left, the second
    // never had a definition at all — it was a declaration nothing could link to.
    void launch(int w, int h);
    void launch_random_pixel_mode_progressive(SDL_Surface* surface,
        SDL_Window* window,
        SDL_Renderer* renderer,
        int width,
        int height,
        void* framebuffer,
        SDL_Texture* raytrace_texture);
   
    // Removed: applyOIDNDenoising - use Renderer::applyOIDNDenoising instead
  
   
    bool trace(const Ray& ray, HitRecord& rec) const;
    void cleanup();
    void setCameraParams(const Camera& camera, float exposure_override = -1.0f);
    // void setLightParams(const std::shared_ptr<Light>& light);
    void setWorld(const WorldData& world);

    void setLightParams(const std::vector<std::shared_ptr<Light>>& lights);
    void setTime(float time, float water_time);
    void setWindParams(const Vec3& direction, float strength, float speed, float time);
    bool SaveSurface(SDL_Surface* surface, const char* file_path);
    void resetBuffers(int width, int height);
    
    // Cycles-style accumulation status
    bool isAccumulationComplete() const;
    int getAccumulatedSamples() const { return accumulated_samples; }
    // Accumulation-wipe instrument; see the note next to accum_wipe_count.
    uint64_t getAccumWipeCount() const { return accum_wipe_count; }
    uint64_t getAccumWipeResolutionCount() const { return accum_wipe_resolution; }
    uint64_t getAccumWipeCameraCount() const { return accum_wipe_camera; }
    uint64_t getResetBuffersCalls() const { return accum_reset_buffers_calls; }
    // Is the accumulation buffer's .w (sample count) actually PERSISTENT across
    // launches? If the counter says 35 while the image carries one-sample
    // noise, the kernel is taking the `prev_samples == 0` branch every frame --
    // and the only way to know is to read the BUFFER ITSELF. The host counter
    // is not evidence here; it is the thing that is lying.
    // Samples the middle row: cheap and diagnostic.
    bool sampleAccumulationW(float& meanW, float& maxW, float& centerW) const;
    // How many pixels the kernel saw with "no previous samples" in the last
    // launch, plus the dimensions of the region we read -- the latter exists to
    // validate the INSTRUMENT ITSELF.
    unsigned int getAccumPrevZeroLastLaunch() const { return accum_prev_zero_host; }
    void getAccumSampleGeometry(int& readW, int& readH, int& imgW, int& imgH) const {
        readW = prev_width; readH = prev_height;
        imgW = params.image_width; imgH = params.image_height;
    }
    // Separates the "it stops at 35" half in a single read: the adaptive
    // sampling gate and its thresholds. If these are sane and the image is
    // still noisy, the fault is somewhere else.
    bool  getUseAdaptiveSampling() const { return params.use_adaptive_sampling != 0; }
    int   getMinSamplesParam() const { return params.min_samples; }
    float getVarianceThresholdParam() const { return params.variance_threshold; }
    int   getSamplesPerPixelParam() const { return params.samples_per_pixel; }
    void getAccumLastWipeSizes(int& fromW, int& fromH, int& toW, int& toH) const {
        fromW = accum_last_wipe_from[0]; fromH = accum_last_wipe_from[1];
        toW = accum_last_wipe_to[0];     toH = accum_last_wipe_to[1];
    }
    void resetAccumulation();  // Reset accumulation for new frame (animation)
    
    // Stream access for synchronized GPU updates
    CUstream getStream() const { return stream; }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // OPTIMIZED MATERIAL UPDATES (No Geometry Rebuild)
    // ═══════════════════════════════════════════════════════════════════════════
    // Use these instead of rebuildOptiXGeometry when only material properties change
    
    // Updates d_materials buffer only - for material property changes (color, roughness, etc.)
    void updateMaterialBuffer(const std::vector<GpuMaterial>& materials);
    bool updateMaterialAt(uint32_t material_index, const GpuMaterial& material);
    
    // Synchronize material properties (emission, textures) into existing SBT
    void syncSBTMaterialData(const std::vector<GpuMaterial>& materials, bool sync_terrain = true);

    // Updates instance transform by Node Name (e.g. for Gizmo/UI updates)
    void updateObjectTransform(const std::string& node_name, const Matrix4x4& transform);
    
    // Updates d_material_indices buffer - for material slot reassignment (different material assigned)
    void updateSBTMaterialBindings(const std::vector<int>& material_indices);
    // [NEW] Update specific mesh material binding (uses AccelManager fast path)
    void updateMeshMaterialBinding(const std::string& node_name, int old_mat_id, int new_mat_id);
    
    // Updates SBT hitgroup records with new volumetric parameters - for volumetric material changes
    // Updates SBT hitgroup records with new volumetric parameters - for volumetric material changes
    void updateSBTVolumetricData(const std::vector<OptixGeometryData::VolumetricInfo>& volumetric_info);
    
    // Updates VDB volume buffer for GPU ray marching
    void updateVDBVolumeBuffer(const std::vector<GpuVDBVolume>& volumes);

    // Set callback for OptixAccelManager status messages (for HUD)
    void setAccelManagerStatusCallback(std::function<void(const std::string&, int)> callback);
    // Rebuild TLAS only (call after transform updates)
    void rebuildTLAS();
    // Refreshes native foliage instances while preserving BLAS/SBT allocations.
    // Returns false when a source BLAS is not resident and a full rebuild is required.
    bool refreshScatterInstances();
    // Build scene using TLAS/BLAS structure (new method - replaces buildFromData for TLAS mode)
  
    bool isUsingTLAS() const { return use_tlas_mode; }
    // Update TLAS geometry (update all BLAS vertex buffers and refit)
   
    // Returns a vector because one object might be split into multiple instances (multi-material)
    std::vector<int> getInstancesByNodeName(const std::string& nodeName) const {
        auto it = node_to_instance.find(nodeName);
        return (it != node_to_instance.end()) ? it->second : std::vector<int>{};
    }
    void updateInstanceTransform(int instance_id, const float transform[12]);
    
    // Buffer accessors for IBackend
    uchar4* getFramebufferDevicePtr() const { return d_framebuffer; }
    float4* getAccumulationDevicePtr() const { return d_accumulation_float4; }
    float4* getDenoiserAlbedoDevicePtr() const { return d_denoiser_albedo; }
    float4* getDenoiserNormalDevicePtr() const { return d_denoiser_normal; }
    float4* getStylizePositionDevicePtr() const { return d_stylize_position; }
    void* getParamsDevicePtr() const { return (void*)d_params; }
    int getImageWidth() const { return Image_width; }
    int getImageHeight() const { return Image_height; }

    // GPU stylize — runs StylizeCore on the device AOV buffers + the already-graded
    // SDL surface (uploaded, stylized, downloaded). AOVs stay on device (no
    // readback) and the costly per-pixel CPU compute moves to the GPU. Returns
    // false when unavailable (size mismatch, missing AOVs, rebuild in progress, or
    // a CUDA error) so the caller can fall back to the CPU stylize path.
    bool applyStylizeGPU(SDL_Surface* surface,
                         const StylizeGPU::KernelParams& params,
                         const StylizeCore::StyleProfileCore& profile);

    
    // Set visibility by node name (uses OptiX visibility masks)
    void setVisibilityByNodeName(const std::string& nodeName, bool visible);
    void updateInstanceVisibility(int instance_id, bool visible);
    void showAllInstances();

    // Targeted BLAS Update for Terrain Sculpting (Avoids full scene rebuild)
    bool updateMeshBLASFromTriangles(const std::string& node_name, const std::vector<std::shared_ptr<Triangle>>& triangles);
    bool updateTerrainBLASPartial(const std::string& node_name, class TerrainObject* terrain);
    bool updateFlatMeshBLAS(const std::string& node_name, const class TriangleMesh* mesh);

    // ═══════════════════════════════════════════════════════════════════════
    // INCREMENTAL UPDATES (Fast delete/duplicate without BLAS rebuild)
    // ═══════════════════════════════════════════════════════════════════════
    
    // Hide all instances with matching node name (for delete - instant!)
    void hideInstancesByNodeName(const std::string& nodeName);
    
    // Clone all instances with matching node name (for duplicate - instant!)
    // Clone all instances with matching node name (for duplicate - instant!)
    std::vector<int> cloneInstancesByNodeName(const std::string& sourceName, const std::string& newName);
    
    // Get AccelManager for advanced operations
    OptixAccelManager* getAccelManager() { return accel_manager.get(); }
    // Updates Gas Volume buffer for GPU ray marching
    void updateGasVolumeBuffer(const std::vector<GpuGasVolume>& volumes);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // HAIR RENDERING (OptiX Curve Primitives)
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Build hair geometry acceleration structure
     * @param vertices float4 array (x, y, z, radius)
     * @param indices Segment start indices
     * @param tangents Pre-computed tangent per segment
     * @param vertex_count Number of vertices
     * @param segment_count Number of curve segments
     */
    void buildHairGeometry(
        const float4* vertices,
        const unsigned int* indices,
        const uint32_t* strand_ids,
        const float3* tangents,
        const float2* root_uvs,
        const float* strand_v,
        size_t vertex_count,
        size_t segment_count,
        const GpuHairMaterial& material,
        const std::string& groomName = "default",
        int materialID = 0,
        int meshMaterialID = -1,
        bool useBSpline = false,
        bool clearPrevious = true
    );
    
    /**
     * @brief Fast update for hair geometry (Refit)
     * Used for GPU grooming to avoid full GAS rebuild
     */
    void updateHairGeometryRefit(
        const std::string& groomName,
        const float3* d_vertices,
        const float* d_widths,
        const float3* d_tangents
    );
    
    /**
     * @brief Update hair material parameters
     */
    void setHairMaterial(float3 color, float3 absorption, float melanin, float melaninRedness, float roughness, float radialRoughness, float ior, float coat, float alpha, float random_hue, float random_value);
    
    /**
     * @brief Set hair color mode (0=Direct, 1=Melanin, 2=Absorption, 3=Root UV Map)
     */
    void setHairColorMode(int colorMode);
    
    /**
     * @brief Set hair custom textures (albedo, roughness) and scalp mesh texture
     */
    void setHairTextures(
        cudaTextureObject_t albedoTex, bool hasAlbedo,
        cudaTextureObject_t roughnessTex, bool hasRoughness,
        cudaTextureObject_t scalpAlbedoTex, bool hasScalpAlbedo,
        float3 scalpBaseColor
    );
    
    /**
     * @brief Set the material ID for hair geometry
     */
    void setHairMaterialID(int materialID) { m_hairMaterialID = materialID; }
    
    /**
     * @brief Check if hair geometry is present
     */
    bool hasHairGeometry() const { return m_hairHandle != 0; }
    
    /**
     * @brief Clear hair geometry
     */
    void clearHairGeometry();
    void updateHairMaterialsOnly(const Hair::HairSystem& hairSystem);
    
    // Download image from GPU to host memory
    void downloadFramebuffer(uchar4* host_ptr, int width, int height);
    bool downloadDenoiserBuffers(std::vector<float>& color, std::vector<float>& albedo, std::vector<float>& normal, bool useAuxiliary = true);
    // Stylize AOV: downloads the world-position + encoded-material buffer as stride-4
    // (x,y,z,w). Bottom-up (matches the renderer's GPU buffer convention). Returns false
    // when the buffer isn't allocated (Stylize/denoiser off).
    bool downloadStylizePositionBuffer(std::vector<float>& position);

private:
    // Scene rebuilds can be requested by both the central dirty-state sync and
    // the deferred OptiX rebuild path. Serialize ownership of scene CUDA
    // allocations so cleanup cannot free the same event/buffer concurrently.
    std::recursive_mutex scene_resource_mutex_;
    // Hair rendering members
    OptixTraversableHandle m_hairHandle = 0;
    CUdeviceptr m_d_hairVertices = 0;
    CUdeviceptr m_d_hairIndices = 0;
    CUdeviceptr m_d_hairTangents = 0;
    CUdeviceptr m_d_hairStrandIDs = 0;
    CUdeviceptr m_d_hairGas = 0;
    size_t m_hairVertexCount = 0;
    size_t m_hairSegmentCount = 0;
    int m_hairMaterialID = 0;
    
    // Mapping for per-groom material updates
    std::unordered_map<std::string, int> m_groomToCurveID; 

  
    
    std::function<void(const std::string&, int)> m_accelStatusCallback;
    
    // OptiX context
    // ═══════════════════════════════════════════════════════════════════════════
    // TLAS/BLAS ACCELERATION STRUCTURE (Two-Level AS for efficient updates)
    // ═══════════════════════════════════════════════════════════════════════════
    
   
    
    // Update instance transform without rebuilding BLAS (fast path for animation)
    void updateInstanceTransform(int instance_id, const Vec3& position, 
                                  const Vec3& rotation_deg, const Vec3& scale);
    
    // Enable TLAS mode (call before first buildFromDataTLAS)
    void enableTLASMode(bool enable) { use_tlas_mode = enable; }

   
    

    
    ColorProcessor color_processor;


    Camera prev_camera;
    bool first_frame_camera = true;
    // OIDN members removed - denoising handled by Renderer class
    cudaDeviceProp props;
    // OptiX context
    OptixDeviceContext context = nullptr;
    // in the header file
    std::vector<SbtRecord<HitGroupData>> hitgroup_records;
    
    //  Texture CUDA array tracking (memory leak fix)
    std::vector<cudaArray_t> texture_arrays;
    
    // TLAS/BLAS acceleration structure manager
    std::unique_ptr<OptixAccelManager> accel_manager;
    bool use_tlas_mode = true;  // Set to true to use TLAS/BLAS instead of single GAS
    
    // Node name to instance mapping (for per-object transform updates)
    // One node (e.g. "Car") can map to MULTIPLE instances (e.g. "Car_mat_0", "Car_mat_1")
    std::unordered_map<std::string, std::vector<int>> node_to_instance;  // nodeName → vector<instance_id>
    std::unordered_map<int, std::string> instance_to_node;  // instance_id → nodeName
    
    // Per-BLAS data storage (geometry buffers for each mesh)
    struct PerBLASData {
        CUdeviceptr d_vertices = 0;
        CUdeviceptr d_indices = 0;
        CUdeviceptr d_normals = 0;
        CUdeviceptr d_uvs = 0;
        CUdeviceptr d_tangents = 0;
        CUdeviceptr d_material_indices = 0;
        CUdeviceptr d_gas_output = 0;
        OptixTraversableHandle handle = 0;
        size_t triangle_count = 0;
        std::string node_name;
    };
    std::vector<PerBLASData> per_blas_data;
    
    // Helper: Extract mesh geometry from triangles for BLAS building
    MeshGeometry extractMeshGeometry(
        const std::vector<std::shared_ptr<Triangle>>& all_triangles,
        const MeshData& mesh);

    
    CUstream stream = nullptr;
      int Image_width;
    int Image_height;
    // BVH ve bufferlar
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices = 0;
    CUdeviceptr d_bvh_output = 0;
    CUdeviceptr d_normals=0;
    CUdeviceptr d_uvs=0;
    CUdeviceptr d_material_indices=0;
    CUdeviceptr d_tangents=0;
    CUdeviceptr d_temp_buffer=0, d_output_buffer=0, d_compacted_size=0;
    CUdeviceptr d_params=0;
    CUdeviceptr d_coords_x=0, d_coords_y=0;
    OptixTraversableHandle traversable_handle = 0;
    OptixModule raygen_module = nullptr;
    OptixModule miss_module = nullptr;
    OptixModule hitgroup_module = nullptr;
    // Built-in intersection modules are context-owned OptiX objects too. Keep
    // their handles so backend teardown can release them with the pipeline.
    OptixModule curve_is_module = nullptr;
    OptixModule sphere_is_module = nullptr;
    // Value-initialized on the HOST so newly added fields start at 0 without
    // giving RayGenParams a default member initializer — the type is also
    // declared as a CUDA __constant__, and nvcc rejects any dynamic
    // initialization in such a type.
    RayGenParams params{};
    OptixProgramGroup raygen_pg = nullptr;
    OptixProgramGroup miss_pg = nullptr;
    OptixProgramGroup miss_shadow_pg = nullptr;
    OptixProgramGroup hit_pg = nullptr;
    OptixProgramGroup hit_shadow_pg = nullptr;
    OptixProgramGroup hair_hit_pg = nullptr;      // Hair curve radiance
    OptixProgramGroup hair_shadow_pg = nullptr;   // Hair curve shadow
    OptixProgramGroup sphere_hit_pg = nullptr;    // Point-sphere (foam) radiance
    OptixProgramGroup sphere_shadow_pg = nullptr; // Point-sphere (foam) shadow
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    GpuMaterial* d_materials = nullptr;
    GpuVolumetricInfo* d_volumetric_infos = nullptr;
    
    // VDB Volume Objects (independent objects with NanoVDB grids)
    GpuVDBVolume* d_vdb_volumes = nullptr;
    size_t d_vdb_volumes_capacity = 0;
    
    // Gas Volume Objects (Dense 3D Textures)
    GpuGasVolume* d_gas_volumes = nullptr;
    size_t d_gas_volumes_capacity = 0;
    
   

    float* d_accumulation_buffer = nullptr;
    float* d_variance_buffer = nullptr;
    int* d_sample_count_buffer = nullptr;
    int* d_converged_count = nullptr;  // Atomic counter for adaptive sampling debug
    std::vector<uchar4> partial_framebuffer;

    // Accumulates the coordinates of pixels processed since the last screen update.
    std::vector<std::pair<int, int>> accumulated_coords;
    int prev_width = 0;
    int prev_height = 0;
    int frame_counter = 1;
    uchar4* host_output_buffers[2] = { nullptr, nullptr }; // GPU'dan CPU'ya indirilen bufferlar
    cudaEvent_t host_output_copy_events[2] = { nullptr, nullptr };
    bool host_output_copy_pending[2] = { false, false };
    size_t host_output_buffer_capacity = 0;
    int host_output_display_slot = -1;
    uint32_t host_output_write_slot = 0;
    int* host_converged_count_buffers[2] = { nullptr, nullptr };
    cudaEvent_t host_converged_count_events[2] = { nullptr, nullptr };
    bool host_converged_count_pending[2] = { false, false };
    int host_converged_count_latest = 0;
    uint32_t host_converged_count_write_slot = 0;
    uchar4* d_framebuffer = nullptr;
    // (pipeline and SBT will live here later)
    struct Tile {
        int x, y;
        int width, height;
        int samples;
        float variance;
        bool completed;
    };

    // Optimization state tracking
    bool is_gas_built_as_soup = false;
    size_t allocated_vertex_byte_size = 0;
    size_t allocated_normal_byte_size = 0;
    size_t last_vertex_count = 0;
    
    // Cycles-style accumulative rendering state
    uint64_t last_camera_hash = 0;           // Camera state hash for change detection
    // Signature of the display/post values last baked into the presented frame.
    // A change while accumulation is already converged triggers a resolve-only
    // launch (params.display_resolve_only) instead of nothing at all — see
    // renderFrame's converged early-return.
    uint64_t last_display_post_signature = 0;
    // Hash of every value that changes how the accumulation is turned into
    // pixels. NOTE: OptiX currently honours only camera exposure and vignette
    // here — make_color() in vec3_utils.cuh still hardcodes Reinhard + gamma 2.2
    // and never reads g_display_post, so tone-map type / gamma / saturation /
    // colour temperature do not reach the OptiX image at all. Those fields are
    // mixed in anyway so this signature does not have to change when that is
    // fixed (see docs/dev — "single display transform").
    uint64_t computeDisplayPostSignature() const;
    int accumulated_samples = 0;              // Total samples accumulated so far
    bool accumulation_valid = false;          // Is current accumulation buffer valid?
    // ACCUMULATION-WIPE INSTRUMENT (2026-09-16). Symptom: "the counter climbs
    // but the image does not accumulate". The mechanism is the accumulation
    // buffer being REALLOCATED AND ZEROED whenever `accumulation_valid` is
    // false; that branch does NOT reset `accumulated_samples`, so the counter
    // keeps climbing and reports progress the pixels do not have.
    // When two numbers contradict each other, the only way to tell which one
    // is right is to count the WIPES.
    uint64_t accum_wipe_count = 0;         // how many times the buffer was zeroed
    uint64_t accum_wipe_resolution = 0;    // how many of those were resolution changes
    // The camera branch is counted SEPARATELY because it is a different
    // mechanism: cudaMemsetAsync, not a realloc. Counting only the realloc
    // would have shown "wipe_count 0" while the camera path re-zeroed every
    // single frame -- an instrument going quiet mistaken for proof of absence.
    uint64_t accum_wipe_camera = 0;
    unsigned int* d_accum_prev_zero = nullptr;   // device-side counter
    unsigned int  accum_prev_zero_host = 0u;     // last value read back
    // The unconditional memset inside resetBuffers() destroys accumulation;
    // counting the calls settles "is it running every frame" in one read.
    uint64_t accum_reset_buffers_calls = 0;
    int accum_last_wipe_from[2]{0, 0};     // resolution BEFORE the last wipe
    int accum_last_wipe_to[2]{0, 0};       // and AFTER -- if these two keep
                                           // swapping, that is the root
    float4* d_accumulation_float4 = nullptr;  // High precision accumulation buffer (float4)
    float4* d_denoiser_albedo = nullptr;
    float4* d_denoiser_normal = nullptr;
    float4* d_stylize_position = nullptr;   // Stylize AOV: world pos (.xyz) + encoded matid (.w)
    void* d_stylize_color = nullptr;        // uint32 staging buffer for the GPU stylize round-trip
    int   stylize_color_w = 0;
    int   stylize_color_h = 0;
    std::vector<float> host_denoiser_color;
    std::vector<float> host_denoiser_albedo;
    std::vector<float> host_denoiser_normal;
    float frozen_water_time = 0.0f;           // Water time frozen at accumulation start
    
    // FFT Ocean (Tessendorf) state
    void* fft_ocean_state = nullptr;          // FFTOceanState* (opaque to avoid header dependency)
    cudaTextureObject_t fft_height_tex = 0;   // Height map texture for shaders
    cudaTextureObject_t fft_normal_tex = 0;   // Normal map texture for shaders
    
    // ═══════════════════════════════════════════════════════════════════════════
    // PERSISTENT GPU BUFFERS (Animation Performance Optimization)
    // ═══════════════════════════════════════════════════════════════════════════
    // These buffers are allocated once and reused across frames to avoid
    // expensive per-frame cudaMalloc/cudaFree overhead.
    
    CUdeviceptr d_params_persistent = 0;      // Persistent RayGenParams buffer
    CUdeviceptr d_lights_persistent = 0;      // Persistent lights buffer
    size_t d_lights_capacity = 0;             // Current capacity of lights buffer (bytes)
    
    // Helper function to compute camera hash
    uint64_t computeCameraHash() const;
    
    // Flag for parameter uploads
    bool params_dirty = true;
    bool hasDirtyParams() const { return params_dirty; }
    size_t d_temp_buffer_size = 0; // Usage tracking for temp buffer optimization
    
    // Cached scene data for incremental SBT/TLAS updates
    std::vector<GpuMaterial> m_cached_materials;
    std::vector<OptixGeometryData::TextureBundle> m_cached_textures;
    std::vector<OptixGeometryData::VolumetricInfo> m_cached_volumetrics;
    int m_material_count = 0;
};
