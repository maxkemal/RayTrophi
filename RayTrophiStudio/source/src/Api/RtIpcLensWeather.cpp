#include "RtLensWeatherBindings.h"
#include "RtApiInternal.h"
#include "Atmosphere/LensWeatherService.h"
#include "Camera.h"
#include <stdexcept>

namespace rt_lens_weather {
using json = nlohmann::json;
json invoke(UIContext& context, const std::string& method, const json& params) {
    if (!context.scene.camera) {
        throw std::invalid_argument("No active camera");
    }
    if (!params.is_object()) {
        throw std::invalid_argument("Lens weather parameters must be an object");
    }
    auto& camera = *context.scene.camera;
    if (method == "camera.set_lens_weather") {
        if (params.size() != 1 || !params.contains("patch")) {
            throw std::invalid_argument("set_lens_weather requires one patch object");
        }
        return atmosphere::setLensWeather(context, camera, params.at("patch"));
    }
    if (method == "camera.sample_lens_weather") {
        if (params.size() > 1 || (!params.empty() && !params.contains("backend"))) {
            throw std::invalid_argument("sample_lens_weather only accepts backend");
        }
        return atmosphere::sampleLensWeather(context,
            params.value("backend", std::string("viewport")));
    }
    if (!params.empty()) {
        throw std::invalid_argument("Lens weather operation takes no parameters");
    }
    if (method == "camera.reset_lens_weather") {
        return atmosphere::resetLensWeather(context, camera);
    }
    if (method == "camera.get_lens_weather") {
        return atmosphere::lensWeatherJson(camera.lens_weather);
    }
    throw std::invalid_argument("Unknown lens weather operation");
}

} // namespace

bool dispatchLensWeatherIpc(const std::string& method, const nlohmann::json& params,
                           const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result) {
    if (method != "camera.get_lens_weather" && method != "camera.set_lens_weather" &&
        method != "camera.reset_lens_weather" && method != "camera.sample_lens_weather") {
        return false;
    }
    result = enqueue([method, params](UIContext& context) {
        return rt_lens_weather::invoke(context, method, params);
    });
    return true;
}
