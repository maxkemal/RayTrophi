/* 1.07.20024
* 
  Ray Tracing Project

  This file contains an implementation of a ray tracing application written in C++,
  designed to render 3D scenes realistically using ray tracing techniques. It includes
  optimizations using SIMD and multi-threading for performance.

  Created by Kemal DEMİRTAŞ and licensed under the MIT License.
*/


/*
  Ray Tracing Projesi

  Bu dosya, ray tracing tekniğini kullanarak 3D sahneleri foto-gerçekçi bir şekilde render etmek için
  C++ dilinde yazılmış bir uygulamayı içerir. SIMD kullanımı ve çoklu iş parçacığı desteği ile performans
  optimizasyonu sağlanmıştır.

  Proje, Kemal DEMİRTAŞ tarafından oluşturulmuştur ve MIT Lisansı altında lisanslanmıştır.
*/


#ifndef RENDERER_H
#define RENDERER_H

#include <SDL.h>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "HittableList.h"
#include "light.h"
#include "Vec3.h"
#include "Camera.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "AreaLight.h"
#include "Volumetric.h"
#include "matrix4x4.h"
#include "PrincipledBSDF.h"
#include "Dielectric.h"
#include "Material.h"
#include "Triangle.h"
#include "Vec2.h"
#include "Vec3SIMD.h"
#include "Mesh.h"
#include "AABB.h"
#include "Ray.h"
#include "Hittable.h"
#include "World.h"
#include "ParallelBVHNode.h"
#include <OpenImageDenoise/oidn.hpp>
#include "AnimatedObject.h"
#include <ColorProcessingParams.h>
#include <scene_data.h>


enum class BVHType {
    CustomCPU,
    Embree,
    OptixGPU
};

class OptixWrapper;

class Renderer {
public:

    void updatePixel(SDL_Surface* surface,int x, int y, const Vec3& c);

    void init(SDL_Surface* surface);

    static bool isCudaAvailable();

    void applyOIDNDenoising(SDL_Surface* surface, int numThreads, bool denoise, float blend);

    Renderer(int image_width, int image_height, int max_depth, int samples_per_pixel);
    void resetResolution(int w, int h);
    ~Renderer();
   
    void precompute_halton(int max_halton_index);
    float get_halton_value(size_t index, int dimension);
    static float halton(int index, int base);
    Vec2 stratified_halton(int pixel_x, int pixel_y, int sample_index, int samples_per_pixel);
    float compute_ambient_occlusion( HitRecord& rec, const ParallelBVHNode* bvh);
    Vec3 fresnel_schlick(float cosTheta, Vec3 F0);
   
    static void create_coordinate_system(const Vec3& N, Vec3& T, Vec3& B);
    float sobol(int index, int dimension);
    void initialize_sobol_cache();
    void precompute_sobol(int max_index);
    float get_sobol_value(int index, int dimension);
    Vec2 stratified_sobol(int x, int y, int sample_index, int samples_per_pixel);
    void initialize_halton_cache();
    Vec3 adaptive_sample_pixel(int i, int j, const Camera& cam, const ParallelBVHNode* bvh, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, int max_samples_per_pixel, double variance_threshold, int variance_check_interval);
     //void render_chunk(int start_row, int end_row, SDL_Surface* surface, const HittableList& world, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, const ParallelBVHNode* bvh, const int samples_per_pass, const int current_sample);

    void update_pixel_color(SDL_Surface* surface, int i, int j, const Vec3 color, int current_sample, int new_samples);
  
    void set_window(SDL_Window* win);

    void draw_progress_bar(SDL_Surface* surface, float progress);
    bool SaveSurface(SDL_Surface* surface, const char* file_path);
    void render_image(SDL_Surface* surface, SDL_Window* window, SDL_Texture* raytrace_texture, SDL_Renderer* renderer,
        const int total_samples_per_pixel, const int samples_per_pass, SceneData& scene);
    Matrix4x4 calculateAnimationTransform(const AnimationData& animation, float currentTime);
   // void updateBVHForAnimatedObjects(ParallelBVHNode& bvh, const std::vector<std::shared_ptr<Hittable>>& animatedObjects);
    void render_Animation(SDL_Surface* surface, SDL_Window* window, SDL_Texture* raytrace_texture, SDL_Renderer* renderer, 
        const int total_samples_per_pixel, const int samples_per_pass, float fps, float duration, SceneData& scene,
        const std::string& output_folder = "", bool use_denoiser = false, float denoiser_blend = 0.9f,
        OptixWrapper* optix_gpu = nullptr, bool use_optix = false);
    bool updateAnimationState(SceneData& scene, float time);
   // void create_scene(SceneData& scene,OptixWrapper* optix_gpu_ptr = nullptr);

    void create_scenefromMesh(const std::string& filename);

    void rebuildBVH(SceneData& scene, bool use_embree);

    void create_scene(SceneData& scene, OptixWrapper* optix_gpu_ptr, const std::string& model_path);

   
    void initializeBuffers(int image_width, int image_height);
    World world;
    static std::vector<Vec3> normalMapBuffer;
    SDL_PixelFormat* pixelFormat;
    Uint8 Rshift, Gshift, Bshift;
    Uint32 Rmask, Gmask, Bmask;
private:
    std::vector<float> variance_buffer;
    static constexpr size_t CACHE_SIZE = 8;
    static constexpr size_t DIMENSION_COUNT = 2;
    AssimpLoader assimpLoader;
    // Cache yapısı
    struct SobolCache {
        std::vector<std::vector<float>> values;
        std::atomic<size_t> last_computed_index;

        SobolCache() : values(DIMENSION_COUNT), last_computed_index(0) {
            for (auto& dim : values) {
                dim.resize(CACHE_SIZE);
            }
        }
    };
   
    OptixWrapper* optix_gpu_ptr = nullptr; // Set externally via set_optix()
    static SobolCache cache;
    std::mutex cache_mutex;  // Header'da tanımlama
    ColorProcessor color_processor;
   
    int image_width;
    int image_height;
    double aspect_ratio;
    int samples_per_pixel;
    int max_depth;
    std::vector<Vec3> frame_buffer;
    std::vector<int> sample_counts;

    std::atomic<int> next_pixel{ 0 };
    const int total_pixels=0;
   
    static const int MAX_DIMENSIONS = 2; // Halton dizisi için maksimum boyut
     size_t MAX_SAMPLES_HALTON = 1024;  // 8K için yeterli olacak şekilde
    int max_halton_index ; // Örnek bir değer, ihtiyaca göre ayarlayın
     const int MAX_SAMPLES_SOBOL = 4;
   
    std::vector<std::vector<float>> sobol_cache;
    std::unique_ptr<float[]> halton_cache;  // Tek boyutlu array olarak tutacağız
    std::atomic<int> next_row{ 0 };
    std::atomic<bool> rendering_complete{ false };
    // Rastgele sıralama (shuffle)
    std::mt19937 gen; // Mersenne Twister 19937 generator
    std::uniform_real_distribution<float> dis;
     SDL_Surface* front_buffer;  // Display için
     SDL_Surface* back_buffer;   // Render için
     std::mutex buffer_mutex;
    std::mutex mtx;
    std::atomic<bool> frame_ready = false;
    float max(float a, float b) const { return a > b ? a : b; }
    // Adaptive sampling için ekstra bufferlar
    std::vector<Vec3> variance_map;


    SDL_Renderer* sdlRenderer; // SDL_Renderer pointer'ı ekleyin
    std::shared_ptr<Texture> background_texture;
    Vec3 sample_directional_light(const ParallelBVHNode* bvh, const DirectionalLight* light, const HitRecord& rec, const Vec3& light_contribution);
    Vec3 sample_point_light(const ParallelBVHNode* bvh, const PointLight* light, const HitRecord& rec, const Vec3& light_contribution);
    Vec3 sample_area_light(const ParallelBVHNode* bvh, const AreaLight* light, const HitRecord& rec, const Vec3& light_contribution, int num_samples);
    void removeFireflies(SDL_Surface* surface);
    Vec3 getColorFromSurface(SDL_Surface* surface, int i, int j);
    void update_variance_map_from_surface(SDL_Surface* surface);
    void update_variance_map_hybrid(SDL_Surface* surface);
   // void render_worker(int image_height, SDL_Surface* surface, const HittableList& world, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, const ParallelBVHNode* bvh, const int samples_per_pass, const int current_sample);
    
    void render_worker(SDL_Surface* surface, const std::vector<std::pair<int, int>>& shuffled_pixel_list, std::atomic<int>& next_pixel_index, const HittableList& world, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, const   Hittable* bvh, const std::shared_ptr<Camera>& camera, const int samples_per_pass, const int current_sample);

    void update_display(SDL_Window* window, SDL_Texture* raytrace_texture, SDL_Surface* surface, SDL_Renderer* renderer);
   
  
    void apply_normal_map( HitRecord& rec);
    void render_chunk_adaptive(SDL_Surface* surface, const std::vector<std::pair<int, int>>& shuffled_pixel_list, std::atomic<int>& next_pixel_index, const HittableList& world, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, const Hittable* bvh, const std::shared_ptr<Camera>& camera, const int total_samples_per_pixel);

    void render_chunk_fixed_sampling(SDL_Surface* surface, const std::vector<std::pair<int, int>>& shuffled_pixel_list, std::atomic<int>& next_pixel_index, const HittableList& world, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, const Hittable* bvh, const std::shared_ptr<Camera>& camera, const int total_samples_per_pixel);
  
    void render_chunk(SDL_Surface* surface, const std::vector<std::pair<int, int>>& shuffled_pixel_list, std::atomic<int>& next_pixel_index, const HittableList& world, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, const   Hittable* bvh, const std::shared_ptr<Camera>& camera, const int samples_per_pass, const int current_sample);
  
    Vec3 ray_color(const Ray& r, const   Hittable* bvh, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, int depth, int sample_index=0);
    float radical_inverse(unsigned int bits);
    Vec3 calculate_volumetric_lighting(const ParallelBVHNode* bvh, const std::vector<std::shared_ptr<Light>>& lights, const HitRecord& rec, const Ray& ray);
   

    float luminance(const Vec3& color);

    int pick_smart_light(const std::vector<std::shared_ptr<Light>>& lights, const Vec3& hit_position);

    Vec3 calculate_direct_lighting_single_light(const Hittable* bvh, const std::shared_ptr<Light>& light, const HitRecord& rec, const Vec3& normal, const Ray& r_in);

    Vec3 calculate_brdf_mis_single_light(const std::shared_ptr<Light>& light, const HitRecord& rec, const Ray& scattered, const Ray& ray_in);
  

    Vec3 calculate_specular(const Vec3& intensity, const Vec3& normal, const Vec3& to_light, const Vec3& view_direction, float shininess);
 
    Vec3 calculate_diffuse(const Vec3& intensity, float cos_theta, float metallic);
    Vec3 calculate_global_illumination(const ParallelBVHNode* bvh, const std::vector<std::shared_ptr<Light>>& lights, const HitRecord& rec, const Vec3& normal, const Vec3& view_direction, const Vec3& background_color);
   
    
    // OIDN Members
    oidn::DeviceRef oidnDevice;
    oidn::FilterRef oidnFilter;
    std::mutex oidnMutex;
    bool oidnInitialized = false;

    // OIDN Buffer Cache - performans optimizasyonu
    // Boyut değişmedikçe buffer'lar yeniden kullanılır
    oidn::BufferRef oidnColorBuffer;
    oidn::BufferRef oidnOutputBuffer;
    std::vector<float> oidnColorData;    // CPU tarafı color buffer cache
    int oidnCachedWidth = 0;
    int oidnCachedHeight = 0;

    void initOIDN(); // Helper to init device
    float lastAnimationUpdateTime = -1.0f; // Track animation time
    SDL_Window* window;
   
public:
    // ============ CYCLES-STYLE ACCUMULATIVE RENDERING (CPU) ============
    struct Vec4 { float x, y, z, w; };  // For accumulation buffer (RGB + sample count)
    
    // Accumulation state
    std::vector<Vec4> cpu_accumulation_buffer;
    int cpu_accumulated_samples = 0;
    uint64_t cpu_last_camera_hash = 0;
    bool cpu_accumulation_valid = false;
    
    // Progressive render functions
    void render_progressive_pass(SDL_Surface* surface, SDL_Window* window, SceneData& scene, int samples_this_pass = 1);
    void resetCPUAccumulation();
    bool isCPUAccumulationComplete() const;
    int getCPUAccumulatedSamples() const { return cpu_accumulated_samples; }
    uint64_t computeCPUCameraHash(const Camera& cam) const;
    
};
#endif // RENDERER_H