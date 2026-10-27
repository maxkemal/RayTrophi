#include "RtRealtimeDenoiserBindings.h"
#include "Api/RtApi.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bindRealtimeDenoiser(py::module_& render) {
    render.def("realtime_denoiser_stats", [](bool synchronize) {
        return realtimeDenoiserDictionary<py::dict>(synchronize);
    }, py::arg("synchronize") = true,
        "SVGF settings, last GPU pass costs and measured mean history length. "
        "synchronize=True waits for GPU completion; False never stalls.");
    render.def("set_realtime_denoiser", [](py::object mode, py::object iterations,
        py::object history, py::object clip, py::object still, py::object debug,
        py::object scene) {
        std::optional<std::string> m;
        std::optional<int> i, h;
        std::optional<float> c, st;
        if (!mode.is_none()) {
            m = mode.cast<std::string>();
        }
        auto integer = [](const py::object& value) {
            if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value)) {
                throw py::value_error("SVGF iterations/history must be integers");
            }
            return value.cast<int>();
        };
        if (!iterations.is_none()) {
            i = integer(iterations);
        }
        if (!history.is_none()) {
            h = integer(history);
        }
        if (!clip.is_none()) {
            if (py::isinstance<py::bool_>(clip) ||
                (!py::isinstance<py::int_>(clip) && !py::isinstance<py::float_>(clip))) {
                throw py::value_error("firefly_clip must be numeric");
            }
            c = clip.cast<float>();
        }
        if (!still.is_none()) {
            if (py::isinstance<py::bool_>(still) ||
                (!py::isinstance<py::int_>(still) && !py::isinstance<py::float_>(still))) {
                throw py::value_error("still_noise must be numeric (0 = off)");
            }
            st = still.cast<float>();
        }
        std::optional<bool> dbg;
        if (!debug.is_none()) {
            if (!py::isinstance<py::bool_>(debug)) {
                throw py::value_error("debug_history must be a bool");
            }
            dbg = debug.cast<bool>();
        }
        std::optional<bool> sm;
        if (!scene.is_none()) {
            if (!py::isinstance<py::bool_>(scene)) {
                throw py::value_error("scene_motion must be a bool");
            }
            sm = scene.cast<bool>();
        }
        const auto result = rtapi::setRealtimeDenoiser(m, i, h, c, st, dbg, sm);
        if (!result.ok) {
            throw py::value_error(result.error);
        }
    }, py::arg("mode") = py::none(), py::arg("atrous_iterations") = py::none(),
        py::arg("history_max") = py::none(), py::arg("firefly_clip") = py::none(),
        py::arg("still_noise") = py::none(), py::arg("debug_history") = py::none(),
        py::arg("scene_motion") = py::none(),
        "Session-only Vulkan SVGF: mode off|svgf, iterations 1..5, history 1..64, "
        "firefly_clip 0.1..10, still_noise 0..0.5 (0 = off; still-camera spatial "
        "filtering while viewport OIDN is off, faded out per pixel once the raw "
        "accumulation's relative standard error reaches this value), scene_motion "
        "bool (animation that resets accumulation with the camera still counts as "
        "motion for SVGF and FSR). Validates the entire update before changing settings.");
}
