#include "Atmosphere/LensWeatherService.h"
#include "Backend/VulkanBackend.h"
#include "Camera.h"
#include "ProjectManager.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "globals.h"
#include <stdexcept>

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern std::unique_ptr<Backend::IBackend> g_backend;

namespace atmosphere {
namespace {
void publish(UIContext& context, Camera& camera) {
    context.apply_tonemap = true;
    if (context.scene.camera.get() != &camera) {
        return;
    }
    for (auto* backend : {g_backend.get(),
            static_cast<Backend::IBackend*>(g_viewport_backend.get())}) {
        if (auto* vulkan = dynamic_cast<Backend::VulkanBackendAdapter*>(backend)) {
            vulkan->updateLensWeather(camera);
        }
    }
}
} // namespace

nlohmann::json setLensWeather(UIContext& context, Camera& camera,
                              const nlohmann::json& patch) {
    const auto candidate = patchLensWeather(camera.lens_weather, patch);
    if (lensWeatherJson(candidate) != lensWeatherJson(camera.lens_weather)) {
        camera.lens_weather = candidate;
        publish(context, camera);
        ProjectManager::getInstance().markModified();
    }
    return lensWeatherJson(camera.lens_weather);
}

nlohmann::json resetLensWeather(UIContext& context, Camera& camera) {
    ++camera.lens_weather.revision;
    publish(context, camera);
    return lensWeatherJson(camera.lens_weather);
}

nlohmann::json sampleLensWeather(UIContext& context, const std::string& role) {
    if (role != "viewport" && role != "render") {
        throw std::invalid_argument("backend must be viewport or render");
    }
    if (!context.scene.camera) {
        throw std::invalid_argument("No active camera");
    }
    auto* backend = role == "viewport" ?
        static_cast<Backend::IBackend*>(g_viewport_backend.get()) : g_backend.get();
    auto* vulkan = dynamic_cast<Backend::VulkanBackendAdapter*>(backend);
    if (!vulkan) {
        throw std::invalid_argument("Lens weather requires a Vulkan backend");
    }
    auto result = vulkan->sampleLensWeather();
    result["backend"] = role;
    return result;
}
} // namespace atmosphere
