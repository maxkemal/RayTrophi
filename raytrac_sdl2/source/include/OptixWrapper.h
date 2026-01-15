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

// Forward declarations for TLAS/BLAS support
class Triangle;
struct MeshGeometry;
struct MeshData;

struct OptixGeometryData {
    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<float3> normals;
    std::vector<float3> tangents;
    std::vector<float2> uvs;
    std::vector<GpuMaterial> materials;
    std::vector<int> material_indices;

    struct TextureBundle {
        cudaTextureObject_t albedo_tex = 0;
        cudaTextureObject_t roughness_tex = 0;
        cudaTextureObject_t normal_tex = 0;
        cudaTextureObject_t metallic_tex = 0;
        cudaTextureObject_t transmission_tex = 0;
        cudaTextureObject_t opacity_tex = 0;
        cudaTextureObject_t emission_tex=0;

        int has_albedo_tex = 0;
        int has_roughness_tex = 0;
        int has_normal_tex = 0;
        int has_metallic_tex = 0;
        int has_transmission_tex = 0;
        int has_opacity_tex = 0;
        int has_emission_tex = 0;
    };
    
    // Volumetric material info for GPU
    struct VolumetricInfo {
        int is_volumetric = 0;
        float density = 1.0f;
        float absorption = 0.1f;
        float scattering = 0.5f;
        float3 albedo = {1.0f, 1.0f, 1.0f};
        float3 emission = {0.0f, 0.0f, 0.0f};
        float g = 0.0f;
        float step_size = 0.1f;
        int max_steps = 100;
        float noise_scale = 1.0f;  // Noise frequency multiplier
        
        // Multi-Scattering Parameters
        float multi_scatter = 0.3f;
        float g_back = -0.3f;
        float lobe_mix = 0.7f;
        int light_steps = 4;
        float shadow_strength = 0.8f;
        
        float3 aabb_min = {0.0f, 0.0f, 0.0f};
        float3 aabb_max = {1.0f, 1.0f, 1.0f};
        
        // NanoVDB GPU grid pointer
        void* nanovdb_grid = nullptr;
        int has_nanovdb = 0;
    };

    std::vector<TextureBundle> textures;
    std::vector<VolumetricInfo> volumetric_info;  // Parallel to materials
};

// Forward declare the acceleration manager (full include in OptixWrapper.cpp)
class OptixAccelManager;

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
    void setupPipeline(const char* raygen_ptx);
    void destroyTextureObjects();
    void buildFromData(const OptixGeometryData& data);
    void buildFromDataTLAS(const OptixGeometryData& data, const std::vector<std::shared_ptr<Hittable>>& objects);
    void updateGeometry(const std::vector<std::shared_ptr<Hittable>>& objects); // Auto-decides based on mode
    void updateTLASGeometry(const std::vector<std::shared_ptr<Hittable>>& objects, const std::vector<Matrix4x4>& boneMatrices = {}); // BLAS+TLAS update
    void updateTLASMatricesOnly(const std::vector<std::shared_ptr<Hittable>>& objects); // Transform-only update
    void launch(SDL_Surface* surface, SDL_Window* window, int w, int h);

  
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
    void setCameraParams(const Camera& camera);
    // void setLightParams(const std::shared_ptr<Light>& light);
    void setWorld(const WorldData& world);

    void setLightParams(const std::vector<std::shared_ptr<Light>>& lights);
    void setTime(float time, float water_time);
    bool SaveSurface(SDL_Surface* surface, const char* file_path);
    void resetBuffers(int width, int height);
    
    // Cycles-style accumulation status
    bool isAccumulationComplete() const;
    int getAccumulatedSamples() const { return accumulated_samples; }
    void resetAccumulation();  // Reset accumulation for new frame (animation)
    
    // ═══════════════════════════════════════════════════════════════════════════
    // OPTIMIZED MATERIAL UPDATES (No Geometry Rebuild)
    // ═══════════════════════════════════════════════════════════════════════════
    // Use these instead of rebuildOptiXGeometry when only material properties change
    
    // Updates d_materials buffer only - for material property changes (color, roughness, etc.)
    void updateMaterialBuffer(const std::vector<GpuMaterial>& materials);
    
    // Updates instance transform by Node Name (e.g. for Gizmo/UI updates)
    void updateObjectTransform(const std::string& node_name, const Matrix4x4& transform);
    
    // Updates d_material_indices buffer - for material slot reassignment (different material assigned)
    void updateSBTMaterialBindings(const std::vector<int>& material_indices);
    
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
    
    // Targeted BLAS Update for Terrain Sculpting (Avoids full scene rebuild)
    void updateMeshBLASFromTriangles(const std::string& node_name, const std::vector<std::shared_ptr<Triangle>>& triangles);

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
    
private:
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
    OptixModule module = nullptr;
    RayGenParams params;
    OptixProgramGroup raygen_pg = nullptr;
    OptixProgramGroup miss_pg = nullptr;
    OptixProgramGroup hit_pg = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    GpuMaterial* d_materials = nullptr;
    GpuVolumetricInfo* d_volumetric_infos = nullptr;
    
    // VDB Volume Objects (independent objects with NanoVDB grids)
    GpuVDBVolume* d_vdb_volumes = nullptr;
    size_t d_vdb_volumes_capacity = 0;
    float* d_accumulation_buffer = nullptr;
    float* d_variance_buffer = nullptr;
    int* d_sample_count_buffer = nullptr;
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
    size_t d_temp_buffer_size = 0; // Usage tracking for temp buffer optimization
};
