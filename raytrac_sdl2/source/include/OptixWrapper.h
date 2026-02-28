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

#include <optix.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
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
    void launch(SDL_Surface* surface, SDL_Window* window, int w, int h);
    void launch(int w, int h); // Overload for internal use (renderer)

  
    void launch_tile_based_progressive(SDL_Surface* surface, SDL_Window* window, int width, int height, std::vector<uchar4>& framebuffer, SDL_Texture* raytrace_texture);
    void launch_random_pixel_mode_progressive(SDL_Surface* surface,
        SDL_Window* window,
        SDL_Renderer* renderer, // YENİ PARAMETRE
        int width,
        int height,
        std::vector<uchar4>& framebuffer,
        SDL_Texture* raytrace_texture);
   
    // Removed: applyOIDNDenoising - use Renderer::applyOIDNDenoising instead
  
   
   // void launch(uchar4* output_buffer, int width, int height);
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
    void resetAccumulation();  // Reset accumulation for new frame (animation)
    
    // Stream access for synchronized GPU updates
    CUstream getStream() const { return stream; }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // OPTIMIZED MATERIAL UPDATES (No Geometry Rebuild)
    // ═══════════════════════════════════════════════════════════════════════════
    // Use these instead of rebuildOptiXGeometry when only material properties change
    
    // Updates d_materials buffer only - for material property changes (color, roughness, etc.)
    void updateMaterialBuffer(const std::vector<GpuMaterial>& materials);
    
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
    void* getParamsDevicePtr() const { return (void*)d_params; }
    int getImageWidth() const { return Image_width; }
    int getImageHeight() const { return Image_height; }

    
    // Set visibility by node name (uses OptiX visibility masks)
    void setVisibilityByNodeName(const std::string& nodeName, bool visible);
    void updateInstanceVisibility(int instance_id, bool visible);
    void showAllInstances();

    // Targeted BLAS Update for Terrain Sculpting (Avoids full scene rebuild)
    void updateMeshBLASFromTriangles(const std::string& node_name, const std::vector<std::shared_ptr<Triangle>>& triangles);
    void updateTerrainBLASPartial(const std::string& node_name, class TerrainObject* terrain);

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

private:
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
    // header dosyasında
    std::vector<SbtRecord<HitGroupData>> hitgroup_records;
    
    //  Texture CUDA array tracking (memory leak fix)
    std::vector<cudaArray_t> texture_arrays;
    
    // TLAS/BLAS acceleration structure managerstum süper
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
    RayGenParams params;
    OptixProgramGroup raygen_pg = nullptr;
    OptixProgramGroup miss_pg = nullptr;
    OptixProgramGroup miss_shadow_pg = nullptr;
    OptixProgramGroup hit_pg = nullptr;
    OptixProgramGroup hit_shadow_pg = nullptr;
    OptixProgramGroup hair_hit_pg = nullptr;      // Hair curve radiance
    OptixProgramGroup hair_shadow_pg = nullptr;   // Hair curve shadow
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

    // Son ekran güncellemesinden bu yana işlenen piksellerin koordinatlarını biriktirir.
    std::vector<std::pair<int, int>> accumulated_coords;
    int prev_width = 0;
    int prev_height = 0;
    int frame_counter = 1;
     uchar4* host_output_buffer = nullptr; // GPU'dan CPU'ya indirilen buffer
    uchar4* d_framebuffer = nullptr;
    // (İleride pipeline ve SBT buraya gelir)
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
    int accumulated_samples = 0;              // Total samples accumulated so far
    bool accumulation_valid = false;          // Is current accumulation buffer valid?
    float4* d_accumulation_float4 = nullptr;  // High precision accumulation buffer (float4)
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
    
    // ═══════════════════════════════════════════════════════════════════════════
    // GPU PICKING (Object ID buffer for viewport selection)
    // ═══════════════════════════════════════════════════════════════════════════
    int* d_pick_buffer = nullptr;          // Per-pixel object ID (-1 = no hit)
    float* d_pick_depth_buffer = nullptr;  // Per-pixel hit distance
    size_t pick_buffer_size = 0;           // Current allocation size
    
    // Cached scene data for incremental SBT/TLAS updates
    std::vector<GpuMaterial> m_cached_materials;
    std::vector<OptixGeometryData::TextureBundle> m_cached_textures;
    std::vector<OptixGeometryData::VolumetricInfo> m_cached_volumetrics;
    int m_material_count = 0;
    
public:
    // Get object ID at screen coordinates (returns -1 if no hit or buffer not ready)
    // viewport_width/height = screen size for coordinate scaling (0 = no scaling)
    int getPickedObjectId(int x, int y, int viewport_width = 0, int viewport_height = 0);
    
    // Get object name at screen coordinates (returns empty string if no hit)
    std::string getPickedObjectName(int x, int y, int viewport_width = 0, int viewport_height = 0);
    
    // Ensure pick buffers are allocated for current resolution
    void ensurePickBuffers(int width, int height);
};
