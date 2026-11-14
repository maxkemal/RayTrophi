// Studio-only Python transport. Shared services and IPC live in the original module.
#include "RtViewportCutoutBindings.h"
#include "Api/RtApiViewportCutout.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

using nlohmann::json;
namespace py = pybind11;

void registerViewportCutoutPython(pybind11::module_& viewport) {
    viewport.def("automatic_cutout", &rtapi::viewportAutomaticCutout);
    viewport.def(
        "set_automatic_cutout",
        [](pybind11::object value) {
            if (!pybind11::isinstance<pybind11::bool_>(value))
                throw pybind11::type_error("enabled must be a boolean");
            const auto r = rtapi::setViewportAutomaticCutout(value.cast<bool>());
            if (!r.ok)
                throw std::runtime_error(r.error);
        },
        pybind11::arg("enabled"));
}
