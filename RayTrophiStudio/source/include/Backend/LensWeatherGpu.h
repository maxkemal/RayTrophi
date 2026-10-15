#pragma once

#include "Atmosphere/LensWeatherSettings.h"
#include "Vec3.h"
#include <cstdint>

class Camera;
namespace Backend {
struct CameraParams;
void copyLensWeatherCamera(const Camera& camera, CameraParams& params);
struct LensWeatherCamera {
    atmosphere::LensWeatherSettings settings;
    Vec3 origin;
    Vec3 forward = Vec3(0, 0, -1);
    Vec3 up = Vec3(0, 1, 0);
    float blur_uv = 0.01f;
    std::uint64_t identity = 0;
};

// std430: nearState is copied directly from the canonical cloud params buffer.
struct alignas(16) LensWeatherHeader {
    float near_state[8][4]{};
    std::uint32_t meta[4]{}; // current step, live count, initialized, reset key
    float stats[4]{};       // impact/s, coverage, film, overflow count
    float previous[4]{};    // origin xyz, time
    std::uint32_t history[4]{}; // oldest/newest step, history miss count, epoch step
};
static_assert(sizeof(LensWeatherHeader) == 192, "Lens header std430 ABI");
} // namespace Backend
