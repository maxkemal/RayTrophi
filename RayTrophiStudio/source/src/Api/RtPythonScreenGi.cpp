#include "RtPythonScreenGi.h"
#include "RtScreenGiBindings.h"
#include <stdexcept>
namespace rtpy {
void registerScreenGiBindings(pybind11::module_& module) {
    namespace py=pybind11;
    module.def("screen_gi",[](){return screenGiDictionary<py::dict>();});
    module.def("set_screen_gi",[](bool enabled,py::object samples,py::object radius,py::object distance) {
        // bool is an int in Python; reject it explicitly for numeric controls.
        if (PyBool_Check(samples.ptr()) || !PyLong_Check(samples.ptr()) ||
            PyBool_Check(radius.ptr()) || !PyLong_Check(radius.ptr()))
            throw py::value_error("samples and filter_radius must be integers");
        if (PyBool_Check(distance.ptr()) || (!PyFloat_Check(distance.ptr()) && !PyLong_Check(distance.ptr())))
            throw py::value_error("max_distance must be numeric");
        RayFusion::ScreenGiSettings s;
        s.enabled=enabled;
        const auto count=samples.cast<int64_t>(),filter=radius.cast<int64_t>();
        if ((count!=1 && count!=2 && count!=4) || filter<0 || filter>2)
            throw py::value_error("samples must be 1/2/4; filter_radius must be 0/1/2");
        s.samples=uint32_t(count);s.filterRadius=uint32_t(filter);s.maxDistance=distance.cast<float>();
        std::string error;
        if (!RayFusion::validateScreenGi(s,error)) throw py::value_error(error);
        if (!rtapi::setScreenGi(s,error)) throw std::runtime_error(error);
        return screenGiDictionary<py::dict>();
    },py::arg("enabled").noconvert(),py::arg("samples")=1,py::arg("filter_radius")=2,py::arg("max_distance")=100.0,
        "Replace same-frame diffuse RTGI settings; no temporal accumulation. Requires RT shadow depth.");
}
}
