// Studio-only Python transport. Shared services and IPC live in the original module.
#include "RtLensWeatherBindings.h"
#include "RtApiInternal.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

using nlohmann::json;
namespace py = pybind11;

using rt_lens_weather::invoke;

namespace rtpy {
void registerLensWeatherBindings(pybind11::module_& camera) {
    namespace py = pybind11;
    for (const char* operation :
         {"get_lens_weather", "set_lens_weather", "reset_lens_weather", "sample_lens_weather"}) {
        const std::string method = std::string("camera.") + operation;
        camera.def(operation, [method](const py::kwargs& kwargs) {
            if (!rtapi::g_ctx) {
                throw std::runtime_error("No active UI context");
            }
            const auto params = json::parse(
                py::cast<std::string>(py::module_::import("json").attr("dumps")(kwargs)));
            const auto request =
                method == "camera.set_lens_weather" ? json{{"patch", params}} : params;
            const auto result = invoke(*rtapi::g_ctx, method, request);
            return py::module_::import("json").attr("loads")(result.dump());
        });
    }
}
} // namespace rtpy
