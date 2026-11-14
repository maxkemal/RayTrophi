// Studio-only Python transport. Shared services and IPC live in the original module.
#include "RtWeatherEffectsBindings.h"
#include "Api/RtApiWeatherEffects.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

using nlohmann::json;
namespace py = pybind11;

using rt_weather_effects::requireWeatherResult;

namespace rtpy {
void registerWeatherEffectsBindings(pybind11::module_& world) {
    namespace py = pybind11;
    world.def("get_weather_effects", [] {
        std::string out;
        requireWeatherResult(rtapi::getWeatherEffectsJson(out));
        return py::module_::import("json").attr("loads")(out);
    });
    world.def(
        "set_weather_effects",
        [](const py::kwargs& kwargs) {
            const std::string patch =
                py::cast<std::string>(py::module_::import("json").attr("dumps")(kwargs));
            requireWeatherResult(rtapi::setWeatherEffectsJson(patch));
        },
        "Partial patch of the manual surface cover (wet/snow/dust).");
}
} // namespace rtpy
