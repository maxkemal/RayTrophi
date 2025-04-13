/* 1.07.20024
* 
  Ray Tracing Project

  This file contains an implementation of a ray tracing application written in C++,
  designed to render 3D scenes realistically using ray tracing techniques. It includes
  optimizations using SIMD and multi-threading for performance.

  Created by Kemal DEM�RTA� and licensed under the MIT License.
*/


/*
  Ray Tracing Projesi

  Bu dosya, ray tracing tekni�ini kullanarak 3D sahneleri foto-ger�ek�i bir �ekilde render etmek i�in
  C++ dilinde yaz�lm�� bir uygulamay� i�erir. SIMD kullan�m� ve �oklu i� par�ac��� deste�i ile performans
  optimizasyonu sa�lanm��t�r.

  Proje, Kemal DEM�RTA� taraf�ndan olu�turulmu�tur ve MIT Lisans� alt�nda lisanslanm��t�r.
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
#include "AtmosphericEffects.h"
#include "ParallelBVHNode.h"
#include <OpenImageDenoise/oidn.hpp>
#include "AnimatedObject.h"
#include <ColorProcessingParams.h>

struct SceneData {
    HittableList world;
    std::shared_ptr<Hittable> bvh;
    std::vector<AnimationData> animationDataList;
    std::vector<std::shared_ptr<AnimatedObject>> animatedObjects;
    std::shared_ptr<Camera> camera;
    std::vector<std::shared_ptr<Light>> lights;
    Vec3 background_color;
};

class Renderer {
public:

    static bool isCudaAvailable();

    static void applyOIDNDenoising(SDL_Surface* surface, int numThreads, bool denoise, float blend);

    Renderer(int image_width, int image_height, int max_depth, int samples_per_pixel);
    ~Renderer();
   
    void precompute_halton(int max_halton_index);
    float get_halton_value(size_t index, int dimension);
    static float halton(int index, int base);
    Vec2 stratified_halton(int pixel_x, int pixel_y, int sample_index, int samples_per_pixel);
    float compute_ambient_occlusion( HitRecord& rec, const ParallelBVHNode* bvh);
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
    void render_image(SDL_Surface* surface, SDL_Window* window, const int total_samples_per_pixel, const int samples_per_pass);
    Matrix4x4 calculateAnimationTransform(const AnimationData& animation, float currentTime);
   // void updateBVHForAnimatedObjects(ParallelBVHNode& bvh, const std::vector<std::shared_ptr<Hittable>>& animatedObjects);
    void render_Animation(SDL_Surface* surface, SDL_Window* window, const int total_samples_per_pixel, const int samples_per_pass, float fps, float duration);
    SceneData create_scene(bool use_embree);

    void initializeBuffers(int image_width, int image_height);
    static std::vector<Vec3> normalMapBuffer;
   
private:
    static constexpr size_t CACHE_SIZE = 8;
    static constexpr size_t DIMENSION_COUNT = 2;
    AssimpLoader assimpLoader;
    // Cache yap�s�
    struct SobolCache {
        std::vector<std::vector<float>> values;
        std::atomic<size_t> last_computed_index;

        SobolCache() : values(DIMENSION_COUNT), last_computed_index(0) {
            for (auto& dim : values) {
                dim.resize(CACHE_SIZE);
            }
        }
    };
   

    static SobolCache cache;
    std::mutex cache_mutex;  // Header'da tan�mlama
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
   
    static const int MAX_DIMENSIONS = 2; // Halton dizisi i�in maksimum boyut
     size_t MAX_SAMPLES_HALTON = 1024;  // 8K i�in yeterli olacak �ekilde
    int max_halton_index ; // �rnek bir de�er, ihtiyaca g�re ayarlay�n
     const int MAX_SAMPLES_SOBOL = 4;
   
    std::vector<std::vector<float>> sobol_cache;
    std::unique_ptr<float[]> halton_cache;  // Tek boyutlu array olarak tutaca��z
    std::atomic<int> next_row{ 0 };
    std::atomic<bool> rendering_complete{ false };
    // Rastgele s�ralama (shuffle)
    std::mt19937 gen; // Mersenne Twister 19937 generator
    std::uniform_real_distribution<float> dis;

    std::mutex mtx;
    float max(float a, float b) const { return a > b ? a : b; }
    // Adaptive sampling i�in ekstra bufferlar
    std::vector<Vec3> variance_map;
    std::vector<int> sample_budget_map; // �imdilik opsiyonel

    AtmosphericEffects atmosphericEffects;
    SDL_Renderer* sdlRenderer; // SDL_Renderer pointer'� ekleyin
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

    void update_display(SDL_Window* window, SDL_Surface* surface);
   
  
    void apply_normal_map( HitRecord& rec);
    void render_chunk_lowpass(SDL_Surface* surface, const std::vector<std::pair<int, int>>& shuffled_pixel_list, std::atomic<int>& next_pixel_index, const HittableList& world, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, const   Hittable* bvh, const std::shared_ptr<Camera>& camera, const int min_samples);
    void render_chunk(SDL_Surface* surface, const std::vector<std::pair<int, int>>& shuffled_pixel_list, std::atomic<int>& next_pixel_index, const HittableList& world, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, const   Hittable* bvh, const std::shared_ptr<Camera>& camera, const int samples_per_pass, const int current_sample);
    void render_chunk_lowpass(SDL_Surface* surface, const std::vector<std::pair<int, int>>& shuffled_pixel_list, std::atomic<int>& next_pixel_index, const HittableList& world, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, const   Hittable* bvh, const std::shared_ptr<Camera>& camera);
    void render_chunk_adaptive_pass(SDL_Surface* surface, const std::vector<std::pair<int, int>>& shuffled_pixel_list, std::atomic<int>& next_pixel_index, const HittableList& world, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, const   Hittable* bvh, const std::shared_ptr<Camera>& camera, const int max_samples, const int current_sample);
    Vec3 ray_color(const Ray& r, const   Hittable* bvh, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, int depth, int sample_index=0);
    float radical_inverse(unsigned int bits);
    Vec3 calculate_volumetric_lighting(const ParallelBVHNode* bvh, const std::vector<std::shared_ptr<Light>>& lights, const HitRecord& rec, const Ray& ray);
   // Vec3 calculate_light_contribution(const std::shared_ptr<Light>& light, const Vec3& point, const Vec3& shading_normal, const Vec3& view_direction, bool is_global=true);
    Vec3 calculate_light_contribution(const std::shared_ptr<Light>& light, const Vec3& point, const Vec3& shading_normal,  bool is_global, const HitRecord& rec);
    Vec3 calculate_direct_lighting(const  Hittable* bvh, const std::vector<std::shared_ptr<Light>>& lights, const HitRecord& rec, const Vec3& normal, float ao_factor);
  
    Vec3 apply_atmospheric_effects(const Vec3& intensity, float distance, bool is_global);
    Vec3 calculate_specular(const Vec3& intensity, const Vec3& normal, const Vec3& to_light, const Vec3& view_direction, float shininess);
 
    Vec3 calculate_diffuse(const Vec3& intensity, float cos_theta, float metallic);
    Vec3 calculate_global_illumination(const ParallelBVHNode* bvh, const std::vector<std::shared_ptr<Light>>& lights, const HitRecord& rec, const Vec3& normal, const Vec3& view_direction, const Vec3& background_color);
   
    
      SDL_Window* window;
  
    
};
#endif // RENDERER_H