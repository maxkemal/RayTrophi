#include "RtUpscaleBindings.h"
#include "Api/RtApi.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
void bindUpscale(py::module_& render) {
    render.def("upscale_stats", [](py::object synchronize) {
        if (!py::isinstance<py::bool_>(synchronize)) {
            throw py::value_error("synchronize must be boolean");
        }
        return upscaleDictionary<py::dict>(synchronize.cast<bool>());
    }, py::arg("synchronize") = py::bool_(true),
        "Requested FSR settings, observed capability/active extent, provider and GPU timings. "
        "synchronize=True waits for the GPU; False is suitable for polling.");
    render.def("set_upscale", [](py::object mode, py::object quality, py::object sharpness,
        py::object rejection) {
        std::optional<std::string> m, q;
        std::optional<float> s;
        if (!mode.is_none()) {
            m = mode.cast<std::string>();
        }
        if (!quality.is_none()) {
            q = quality.cast<std::string>();
        }
        if (!sharpness.is_none()) {
            if (py::isinstance<py::bool_>(sharpness) ||
                (!py::isinstance<py::float_>(sharpness) && !py::isinstance<py::int_>(sharpness))) {
                throw py::value_error("sharpness must be numeric");
            }
            s = sharpness.cast<float>();
        }
        std::optional<bool> r;
        if (!rejection.is_none()) {
            if (!py::isinstance<py::bool_>(rejection)) {
                throw py::value_error("rejection_reactive must be a bool");
            }
            r = rejection.cast<bool>();
        }
        const auto result = rtapi::setUpscale(m, q, s, r);
        if (!result.ok) {
            throw py::value_error(result.error);
        }
    }, py::arg("mode") = py::none(), py::arg("quality") = py::none(),
        py::arg("sharpness") = py::none(), py::arg("rejection_reactive") = py::none(),
        "Session-only moving Vulkan FSR: mode off|fsr|auto, quality "
        "quality|balanced|performance|ultra_performance, sharpness 0..1, "
        "rejection_reactive bool (clustered motion-history rejections drive the "
        "reactive mask; stops ghosting on content without vectors). "
        "Omitted fields remain unchanged; updates are validated atomically.");
}
