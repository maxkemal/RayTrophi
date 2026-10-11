#include "RtWeatherEffectsBindings.h"
#include "Api/RtApiWeatherEffects.h"
#include "RtIpcMethodRegistry.h"

#include <stdexcept>

namespace {
void requireWeatherResult(const rtapi::Result& result) {
    if (!result.ok) {
        throw std::runtime_error(result.error);
    }
}

const MethodParam weatherPatch[] = {
    {"patch", "object", true, "Weather effect fields; wind is climate-owned", nullptr, nullptr}
};
const MethodDescriptor weatherGet = {
    "world.get_weather_effects", "world", "Read fast weather and surface cover controls",
    "Wind values are read-only. Rain=1, snow=2, dust=3, mist=4.",
    "read", "Read", false, "object", "weather|rain|snow|dust", "world.set_weather_effects",
    nullptr, nullptr, nullptr, nullptr, nullptr, 0, true
};
const MethodDescriptor weatherSet = {
    "world.set_weather_effects", "world", "Patch fast weather and surface cover controls",
    "Rejects malformed or out-of-range values atomically; wind reads climate.",
    "write", "SceneWrite", false, "object", "weather|rain|snow|dust", "world.get_weather_effects",
    nullptr, nullptr, "world.get_weather_effects", "accumulation",
    weatherPatch, 1, true
};
const MethodRegistration registerWeatherGet(weatherGet);
const MethodRegistration registerWeatherSet(weatherSet);
} // namespace

namespace rtpy {
void registerWeatherEffectsBindings(pybind11::module_& world) {
    namespace py = pybind11;
    world.def("get_weather_effects", [] {
        std::string out;
        requireWeatherResult(rtapi::getWeatherEffectsJson(out));
        return py::module_::import("json").attr("loads")(out);
    });
    world.def("set_weather_effects", [](const py::kwargs& kwargs) {
        const std::string patch = py::cast<std::string>(
            py::module_::import("json").attr("dumps")(kwargs));
        requireWeatherResult(rtapi::setWeatherEffectsJson(patch));
    }, "Partial patch of fast rain/snow/dust/mist and surface cover controls.");
}
} // namespace rtpy

bool dispatchWeatherEffectsIpc(const std::string& method, const nlohmann::json& params,
                              const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result) {
    if (method == "world.get_weather_effects") {
        result = enqueue([](UIContext&) {
            std::string out;
            requireWeatherResult(rtapi::getWeatherEffectsJson(out));
            return nlohmann::json::parse(out);
        });
        return true;
    }
    if (method == "world.set_weather_effects") {
        if (!params.is_object() || params.size() != 1 || !params.contains("patch") ||
            !params["patch"].is_object()) {
            throw std::invalid_argument("requires one object parameter: patch");
        }
        const std::string patch = params["patch"].dump();
        result = enqueue([patch](UIContext&) {
            requireWeatherResult(rtapi::setWeatherEffectsJson(patch));
            std::string out;
            requireWeatherResult(rtapi::getWeatherEffectsJson(out));
            return nlohmann::json::parse(out);
        });
        return true;
    }
    return false;
}
