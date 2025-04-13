#ifndef GLOBALS_H
#define GLOBALS_H

#include <mutex>
#include <atomic>
#include <limits>
#include <cmath>
#include <vector>
#include "Vec3.h"


extern std::mutex mtx;
extern std::atomic<int> completed_pixels;
extern std::atomic<bool> rendering_complete;
extern const double min_distance;
extern const double max_distance;
extern const double aspect_ratio; // Sabit olarak double türünde tanımlıyoruz
extern const int image_width;
extern const int image_height;
extern const double EPSILON;
extern  const int MAX_DEPTH;
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
#endif // GLOBALS_H
