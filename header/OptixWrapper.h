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
struct OptixGeometryData {
    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<float3> normals;
    std::vector<float3> tangents;   // ✅ EKLENDİ — buraya geçiyoruz
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
    ~OptixWrapper();

    void initialize();
    bool isCudaAvailable();
    void applyOIDNDenoising(SDL_Surface* surface, int numThreads, bool denoise, float blend);
    void validateMaterialIndices(const OptixGeometryData& data);
    void setupPipeline(const char* raygen_ptx);
    void buildFromData(const OptixGeometryData& data);
    void launch(SDL_Surface* surface, SDL_Window* window, int w, int h);

    void launch_random_pixel_mode(SDL_Surface* surface, SDL_Window* window, int width, int height, std::vector<uchar4>& framebuffer);
    uchar4* host_output_buffer = nullptr; // GPU'dan CPU'ya indirilen buffer

    void applyOIDNDenoising(SDL_Surface* surface, float blend_factor = 1.0f, bool use_albedo = false, float sharpness = 0.95f);
   // void launch_random_pixel_mode(SDL_Surface* surface, SDL_Window* window, int width, int height, std::vector<uchar4>& framebuffer);

    void launch_batch_random_pixel_mode(
        SDL_Surface* surface, SDL_Window* window, int width, int height,
        std::vector<uchar4>& framebuffer
    );

   // void launch_batch_random_pixel_mode(SDL_Surface* surface, SDL_Window* window, int width, int height);

    void launch_single_pixel_mode(SDL_Surface* surface, SDL_Window* window, int width, int height);
   
   // void launch(uchar4* output_buffer, int width, int height);
    bool trace(const Ray& ray, HitRecord& rec) const;
    void cleanup();
    void setCameraParams(const Camera& camera);
   // void setLightParams(const std::shared_ptr<Light>& light);
    void setBackgroundColor(const Vec3& color);

    void setLightParams(const std::vector<std::shared_ptr<Light>>& lights);
    bool SaveSurface(SDL_Surface* surface, const char* file_path);
      ColorProcessor color_processor;



private:
    // OptiX context
    OptixDeviceContext context = nullptr;
    CUstream stream = nullptr;
      int image_width;
    int image_height;
    // BVH ve bufferlar
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices = 0;
    CUdeviceptr d_bvh_output = 0;
    OptixTraversableHandle traversable_handle = 0;
    OptixModule module = nullptr;
    RayGenParams params;
    OptixProgramGroup raygen_pg = nullptr;
    OptixProgramGroup miss_pg = nullptr;
    OptixProgramGroup hit_pg = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    GpuMaterial* d_materials = nullptr;


    // (İleride pipeline ve SBT buraya gelir)
};
