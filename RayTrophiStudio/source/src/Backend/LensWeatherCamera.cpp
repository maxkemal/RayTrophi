#include "Backend/LensWeatherGpu.h"
#include "Backend/IBackend.h"
#include "Camera.h"
#include <algorithm>

namespace Backend {
void copyLensWeatherCamera(const Camera& camera, CameraParams& params) {
    params.lens_weather = camera.lens_weather;
    params.lens_weather_identity = reinterpret_cast<std::uintptr_t>(&camera) ^ camera.lens_weather.identity;
    // Match the canonical camera sampling gate. fNumber() has a useful UI
    // fallback even for aperture == 0; it must not invent an f/2 pupil for a
    // pinhole camera and dilute every glass drop across a large blur disk.
    const float pupil = camera.effectiveLensRadius() > 0.0f ?
        camera.focal_length_mm / std::max(camera.fNumber(), 0.5f) : 0.0f;
    params.lens_weather_blur_uv = std::clamp(
        pupil / std::max(camera.sensor_height_mm, 1.0f) * 0.25f, 0.0f, 0.15f);
}
} // namespace Backend
