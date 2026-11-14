// Studio-only Python transport. Shared services and IPC live in the original module.
#include "RtRayGpuTimingBindings.h"
#include "Api/RayGpuTimingApi.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

using nlohmann::json;
namespace py = pybind11;

void bindRayGpuTimingPython(pybind11::module_& render) {
    namespace py = pybind11;
    render.def(
        "set_gpu_timing",
        [](bool enabled) {
            const auto result = rtapi::setRayGpuTimingCapture(enabled);
            if (!result.ok) {
                throw std::runtime_error(result.error);
            }
        },
        py::arg("enabled"));
    render.def("gpu_timing_stats", [] {
        return py::module_::import("json").attr("loads")(rtapi::rayGpuTimingCaptureStats().dump());
    });
}
