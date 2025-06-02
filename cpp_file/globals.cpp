#include "globals.h"

std::mutex mtx;
std::atomic<int> completed_pixels(0);
std::atomic<bool> rendering_complete(false);
constexpr double min_distance = 0.1;  // Minimum mesafe
constexpr double max_distance =10000.0;  // Maksimum mesafe
constexpr double aspect_ratio = 16.0 / 9.0; // Sabit olarak double türünde tanýmlýyoruz
constexpr int image_width = 1680*1;
constexpr int image_height = static_cast<int>(image_width / aspect_ratio);
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
constexpr float aperture = 0.0;
constexpr float focusdistance = 11.42f;
int hitcount=0;
bool is_normal_map = false;
bool globalreflectance = false;
bool use_embree = true;
RenderSettings render_settings = {
    10,  // samples_per_pixel
    4,   // samples_per_pass
    12,   // max_bounces
    true,  // use_adaptive_sampling
    2,     // min_samples
    6,    // max_samples
    0.0002f, // variance_threshold	
    true,  // use_optix
    5.0f,  // animation_duration
    24.0f  // animation_fps
};
