#ifndef GLOBALS_H
#define GLOBALS_H

#include <mutex>
#include <atomic>
#include <limits>
#include <cmath>
#include <vector>
#include "Vec3.h"
#include <filesystem>

struct RenderSettings {
    // Sampling
    int samples_per_pixel;
    int samples_per_pass;
    int max_bounces;

    // Adaptive Sampling
    bool use_adaptive_sampling;
    int min_samples;
    int max_samples;
    float variance_threshold;

    // Denoiser
    bool use_denoiser;
    float denoiser_blend_factor;

    // Backend
    bool use_optix;

    // Animation
    float animation_duration;
    int animation_fps;
    bool start_animation_render = false;
    bool save_image_requested = false;
};


// Yalnýzca bildirim:
extern RenderSettings render_settings;
extern std::mutex mtx;
extern std::atomic<int> completed_pixels;
extern std::atomic<bool> rendering_complete;
extern const double min_distance;
extern const double max_distance;
extern  float aspect_ratio; // Sabit olarak double türünde tanýmlýyoruz
extern  int image_width;
extern  int image_height;
extern const double EPSILON;
extern std::atomic<int> next_row;
extern const double infinity;
extern  std::string baseDirectory;
extern bool atmosferic_effect_enabled;
extern const float max_normal_distance;
extern const float max_normal_strength;
extern const float aperture;
extern const float focusdistance;
extern const float gamma;
extern const float exposure ;
extern const float saturation ;
const float BASE_EPSILON = 1e-6f;
const float MIN_EPSILON = 1e-8f;
const float MAX_EPSILON = 1e-4f;
extern bool globalreflectance ;
extern bool is_normal_map;
extern  int hitcount;
extern bool use_embree;
extern float last_render_time_ms;
extern bool pending_resolution_change;
extern int pending_width;
extern int pending_height;
extern float pending_aspect_ratio;

#endif // GLOBALS_H
