#include "globals.h"

std::mutex mtx;
std::atomic<int> completed_pixels(0);
std::atomic<bool> rendering_complete(false);
constexpr double min_distance = 0.1;  // Minimum mesafe
constexpr double max_distance =10000.0;  // Maksimum mesafe
 float aspect_ratio = 16.0 / 9.0; // Sabit olarak float türünde tanýmlýyoruz
 int image_width = 1680*1;
 int image_height = static_cast<int>(image_width / aspect_ratio);
constexpr double EPSILON = 1e-4f;
constexpr float max_normal_distance = 10.0f;
constexpr float max_normal_strength = 1.0f;
std::atomic<int> next_row(0);
//constexpr double infinity = std::numeric_limits<double>::max();
std::string baseDirectory="";
bool atmosferic_effect_enabled = false;
constexpr float gamma= 1.0f;
constexpr float exposure= 1.0f;
constexpr float saturation=1.0f;
constexpr float aperture = 0.5;
constexpr float focusdistance = 1.573f;
int hitcount=0;
bool is_normal_map = false;
bool globalreflectance = false;
bool use_embree = true;
float last_render_time_ms = 0.0f;  // Render süresi buraya yazýlacak
int pending_width = 1680;
int pending_height = 950;
float pending_aspect_ratio = 16 / 9;
bool pending_resolution_change=false;
RenderSettings render_settings = {
    1,       // samples_per_pixel
    1,       // samples_per_pass
    16,      // max_bounces

    true,    // use_adaptive_sampling
    1,       // min_samples
    1,       // max_samples
    0.0002f, // variance_threshold

    false,    // use_denoiser
    1.0f,    // denoiser_blend_factor

    true,    // use_optix

    1.0f,    // animation_duration
    24,      // animation_fps
    false    // start_animation_render
};
