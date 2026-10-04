#include "RtPhaseGrid.h"

#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace rtpy {
void registerPhaseGridBindings(py::module_& fluid) {
    const auto dictionary = [](const nlohmann::json& value) {
        return py::module_::import("json").attr("loads")(value.dump());
    };
    fluid.def("get_phase_grids", [dictionary](const std::string& domain) {
        return dictionary(rtapi::getPhaseGrids(domain));
    }, py::arg("domain"));
    fluid.def("set_phase_grid", [dictionary](const std::string& domain,
        const std::string& phase, bool inherit, const py::object& lo,
        const py::object& hi, float voxel) {
        const auto json = py::module_::import("json");
        const auto minimum = nlohmann::json::parse(
            json.attr("dumps")(lo).cast<std::string>());
        const auto maximum = nlohmann::json::parse(
            json.attr("dumps")(hi).cast<std::string>());
        return dictionary(rtapi::setPhaseGrid(domain, phase, inherit, minimum, maximum, voxel));
    }, py::arg("domain"), py::arg("phase"), py::arg("inherit") = false,
       py::arg("bounds_min") = py::none(), py::arg("bounds_max") = py::none(),
       py::arg("voxel") = 0.1f);
}
} // namespace rtpy
