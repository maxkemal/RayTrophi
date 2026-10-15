#include "RtLensWeatherBindings.h"
#include "RtApiInternal.h"
#include "Atmosphere/LensWeatherService.h"
#include "Camera.h"
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace {
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
        return invoke(context, method, params);
    });
    return true;
}

namespace rtpy {
void registerLensWeatherBindings(pybind11::module_& camera) {
    namespace py = pybind11;
    for (const char* operation : {"get_lens_weather", "set_lens_weather",
                                "reset_lens_weather", "sample_lens_weather"}) {
        const std::string method = std::string("camera.") + operation;
        camera.def(operation, [method](const py::kwargs& kwargs) {
            if (!rtapi::g_ctx) {
                throw std::runtime_error("No active UI context");
            }
            const auto params = json::parse(py::cast<std::string>(
                py::module_::import("json").attr("dumps")(kwargs)));
            const auto request = method == "camera.set_lens_weather" ?
                json{{"patch", params}} : params;
            const auto result = invoke(*rtapi::g_ctx, method, request);
            return py::module_::import("json").attr("loads")(result.dump());
        });
    }
}
} // namespace rtpy
