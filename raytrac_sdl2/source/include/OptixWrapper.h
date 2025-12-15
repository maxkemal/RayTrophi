#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <vector>
#include "Vec3.h"
#include "Ray.h"
#include "Hittable.h"
#include <Camera.h>
#include <Light.h>
#include "params.h"
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

    std::vector<TextureBundle> textures;
};




class OptixWrapper {
public:
    OptixWrapper();
    void partialCleanup();
    ~OptixWrapper();

    void resizeBuffers(int w, int h);

    void initialize();
    bool isCudaAvailable();
    // OIDN denoising is now handled by Renderer::applyOIDNDenoising
    void validateMaterialIndices(const OptixGeometryData& data);
    void setupPipeline(const char* raygen_ptx);
    void destroyTextureObjects();
    void buildFromData(const OptixGeometryData& data);
    void updateGeometry(const std::vector<std::shared_ptr<Hittable>>& objects); // Dynamic update for animation
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
    void setBackgroundColor(const Vec3& color);

    void setLightParams(const std::vector<std::shared_ptr<Light>>& lights);
    bool SaveSurface(SDL_Surface* surface, const char* file_path);
    void resetBuffers(int width, int height);
    
    // Cycles-style accumulation status
    bool isAccumulationComplete() const;
    int getAccumulatedSamples() const { return accumulated_samples; }
    void resetAccumulation();  // Reset accumulation for new frame (animation)
  
      ColorProcessor color_processor;



private:
    // OIDN members removed - denoising handled by Renderer class
    cudaDeviceProp props;
    // OptiX context
    OptixDeviceContext context = nullptr;
    // header dosyasında
    std::vector<SbtRecord<HitGroupData>> hitgroup_records;
    
    //  Texture CUDA array tracking (memory leak fix)
    std::vector<cudaArray_t> texture_arrays;
    
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
    
    // Helper function to compute camera hash
    uint64_t computeCameraHash() const;
};
