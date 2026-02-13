/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Renderer.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
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
#include "Vec3.h"
#include "matrix4x4.h"
#include "Vec2.h"
#include "Vec3SIMD.h"
#include "World.h"
#include <OpenImageDenoise/oidn.hpp>
#include <ColorProcessingParams.h>
#include <functional>
#include "AssimpLoader.h"
#include "AnimationController.h"  // New animation management system
#include "Hair/HairSystem.h"      // Hair/Fur rendering system

// Forward Declarations
class HittableList;
class Light;
class Camera;
class PointLight;
class DirectionalLight;
class AreaLight;
class Volumetric;
class PrincipledBSDF;
class Dielectric;
class Material;
class Triangle;
class Mesh;
class AABB;
class Ray;
class Hittable;
struct HitRecord;
class ParallelBVHNode;
class AnimatedObject;
struct SceneData;
struct AnimationData;



enum class BVHType {
    CustomCPU,
    Embree,
    OptixGPU
};

class OptixWrapper;
struct UIContext;

class Renderer {
public:

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
   
    void initialize_halton_cache();
    Vec3 adaptive_sample_pixel(int i, int j, const Camera& cam, const ParallelBVHNode* bvh, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, int max_samples_per_pixel, float variance_threshold, int variance_check_interval);
   
    bool SaveSurface(SDL_Surface* surface, const char* file_path);
    void render_Animation(SDL_Surface* surface, SDL_Window* window, SDL_Texture* raytrace_texture, SDL_Renderer* renderer, 
        const int total_samples_per_pixel, const int samples_per_pass, float fps, float duration, int start_frame, int end_frame, SceneData& scene,
        const std::string& output_folder = "", bool use_denoiser = false, float denoiser_blend = 0.9f,
        OptixWrapper* optix_gpu = nullptr, bool use_optix = false, UIContext* ui_ctx = nullptr);
    bool updateAnimationState(SceneData& scene, float time, bool apply_cpu_skinning = true);
    std::vector<Matrix4x4> finalBoneMatrices; // Stores computed bone matrices for the current frame
    
    // Wind Animation System
    void updateWind(SceneData& scene, float time);

    // ============ NEW ANIMATION SYSTEM ============
    // Initialize animation controller with scene clips
    void initializeAnimationSystem(SceneData& scene);
    
    // Update animation using the new node graph system (returns true if geometry changed)
    bool updateAnimationWithGraph(SceneData& scene, float deltaTime, bool apply_cpu_skinning = true);
    
    // Flag to use new animation system (can be toggled for compatibility)
    bool useNewAnimationSystem = false;
    // void create_scene(SceneData& scene,OptixWrapper* optix_gpu_ptr = nullptr);

    void setOptixWrapper(OptixWrapper* ptr) { optix_gpu_ptr = ptr; }

    void rebuildBVH(SceneData& scene, bool use_embree);
    void updateBVH(SceneData& scene, bool use_embree);
   
    void create_scene(SceneData& scene, OptixWrapper* optix_gpu_ptr, const std::string& model_path,
        std::function<void(int progress, const std::string& stage)> progress_callback = nullptr,
        bool append = false, const std::string& import_prefix = "");  // If true, don't clear scene before loading

    // Rebuild OptiX geometry after scene modifications (deletion/addition)
    void rebuildOptiXGeometry(SceneData& scene, OptixWrapper* optix_gpu_ptr);
    
    // VARIANT: Rebuild OptiX geometry with a specific object list (to avoid race conditions)
    void rebuildOptiXGeometryWithList(const std::vector<std::shared_ptr<Hittable>>& objects, OptixWrapper* optix_gpu_ptr);

    // Sync all volumetric data (VDB, Gas) to GPU - Moved to VolumetricRenderer

    // Update OptiX materials only (fast path - no geometry rebuild)
    // Use when only material properties change (color, roughness, volumetric params, etc.)
    void updateOptiXMaterialsOnly(SceneData& scene, OptixWrapper* optix_gpu_ptr);

    // Update OptiX Gas Volumes (fast path - no geometry rebuild)
    // Updates texture handles, transforms, and shader parameters for gas volumes
    void updateOptiXGasVolumes(SceneData& scene, OptixWrapper* optix_gpu_ptr);
   
    void initializeBuffers(int image_width, int image_height);
    World world;
    static std::vector<Vec3> normalMapBuffer;
    SDL_PixelFormat* pixelFormat;

    // Hair/Fur System
    Hair::HairSystem hairSystem;
    Hair::HairMaterialParams hairMaterial;  // Current hair material from UI
    
    // Hair system accessors
    Hair::HairSystem& getHairSystem() { return hairSystem; }
    const Hair::HairSystem& getHairSystem() const { return hairSystem; }
    
    // Hair material setter (called from UI) - updates both CPU and GPU
    void setHairMaterial(const Hair::HairMaterialParams& mat);
    const Hair::HairMaterialParams& getHairMaterial() const { return hairMaterial; }
    
    // Upload hair to GPU (call after hair system changes)
    void uploadHairToGPU();
    
    // Fast update hair geometry (uses refit if possible)
    void updateHairGeometryOnGPU(bool forceRebuild = false);

    bool hideInterpolatedHair = false; // [NEW] Toggle to hide child hairs (interpolated) for performance during grooming

    OptixWrapper* optix_gpu_ptr = nullptr; // Set externally via set_optix()   
private:
    std::vector<float> variance_buffer;
    static constexpr size_t CACHE_SIZE = 8;
    static constexpr size_t DIMENSION_COUNT = 2;
    AssimpLoader assimpLoader;

   
    std::mutex cache_mutex;  // Header'da tanımlama
    ColorProcessor color_processor;
   
    int image_width;
    int image_height;
    float aspect_ratio = 1.0f;
    int samples_per_pixel;
    int max_depth;
    std::vector<Vec3> frame_buffer;
    std::vector<int> sample_counts;

    std::atomic<int> next_pixel{ 0 };
    const int total_pixels=0;
    static const int MAX_DIMENSIONS = 2; // Halton dizisi için maksimum boyut
     size_t MAX_SAMPLES_HALTON = 1024;  // 8K için yeterli olacak şekilde
    int max_halton_index ; // Örnek bir değer, ihtiyaca göre ayarlayın

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
   
    Vec3 getColorFromSurface(SDL_Surface* surface, int i, int j);


    void apply_normal_map( HitRecord& rec);
  
    Vec3 ray_color(const Ray& r, const   Hittable* bvh, const std::vector<std::shared_ptr<Light>>& lights, const Vec3& background_color, int depth, int sample_index, const SceneData& scene);
    float radical_inverse(unsigned int bits);
    // God Rays (CPU) - Moved to VolumetricRenderer

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
    std::vector<float> oidnColorData;      // CPU tarafı color buffer cache
    std::vector<float> oidnOriginalData;   // Original pixel cache (blend için, double-read eliminasyonu)
    int oidnCachedWidth = 0;
    int oidnCachedHeight = 0;

    void initOIDN(); // Helper to init device
    float lastAnimationUpdateTime = -1.0f; // Track animation time
    struct AnimatableGroup {
        std::string nodeName;
        bool isSkinned;
        std::shared_ptr<Transform> transformHandle;
        std::vector<std::shared_ptr<Triangle>> triangles;
    };
    std::vector<AnimatableGroup> animation_groups;
    bool animation_groups_dirty = true;
    SDL_Window* window;
   
public:
    // ============ CYCLES-STYLE ACCUMULATIVE RENDERING (CPU) ============
    struct Vec4 { float x, y, z, w; };  // For accumulation buffer (RGB + sample count)
    
    // Accumulation state
    std::vector<Vec4> cpu_accumulation_buffer;
    int cpu_accumulated_samples = 0;
    uint64_t cpu_last_camera_hash = 0;
    bool cpu_accumulation_valid = false;
    
    // Variance buffer for adaptive sampling (tracks per-pixel noise level)
    std::vector<float> cpu_variance_buffer;
    
    // Cached pixel list for progressive rendering (avoids per-pass allocation + shuffle)
    std::vector<std::pair<int, int>> cpu_cached_pixel_list;
    bool cpu_pixel_list_valid = false;  // Reset when resolution or camera changes
    
    // Progressive render functions
    void render_progressive_pass(SDL_Surface* surface, SDL_Window* window, SceneData& scene, int samples_this_pass = 1, int override_target_samples = 0);
    void resetCPUAccumulation();
    bool isCPUAccumulationComplete() const;
    int getCPUAccumulatedSamples() const { return cpu_accumulated_samples; }
    uint64_t computeCPUCameraHash(const Camera& cam) const;
    
    // === PRO CAMERA HUD ACCESSORS ===
    const std::vector<Vec3>& getFrameBuffer() const { return frame_buffer; }
    int getImageWidth() const { return image_width; }
    int getImageHeight() const { return image_height; }

    // === RENDERING CONTROL ===
    std::atomic<bool> force_stop_rendering{ false };
    void stopRendering() { force_stop_rendering = true; }
    void resumeRendering() { force_stop_rendering = false; }
    
};
#endif // RENDERER_H

