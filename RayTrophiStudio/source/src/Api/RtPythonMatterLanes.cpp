#include "RtMatterLanes.h"
#include <pybind11/stl.h>

namespace rtpy {
void registerMatterLaneBindings(pybind11::module_& fluid) {
    namespace py = pybind11;
    const auto dictionary = [](const nlohmann::json& value) {
        return py::module_::import("json").attr("loads")(value.dump());
    };
    fluid.def("get_lanes", [dictionary](const std::string& domain) {
        return dictionary(rtapi::getMatterDomainLanes(domain));
    }, py::arg("domain"));
    fluid.def("set_lanes", [dictionary](const std::string& domain, bool gas, bool liquid) {
        return dictionary(rtapi::setMatterDomainLanes(domain, gas, liquid));
    }, py::arg("domain"), py::arg("gas_lane_enabled"), py::arg("liquid_lane_enabled"));
}
} // namespace rtpy
