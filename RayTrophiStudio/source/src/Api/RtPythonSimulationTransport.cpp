#include "RtSimulationTransport.h"
#include <pybind11/stl.h>

namespace rtpy {
void registerSimulationTransportBindings(pybind11::module_& timeline,
                                          pybind11::module_& sim) {
    const auto dictionary = [](const nlohmann::json& value) {
        return pybind11::module_::import("json").attr("loads")(value.dump());
    };
    timeline.def("play", [dictionary]() {
        return dictionary(rtapi::setTimelinePlaying(true));
    });
    timeline.def("pause", [dictionary]() {
        return dictionary(rtapi::setTimelinePlaying(false));
    });
    timeline.def("transport", [dictionary]() {
        return dictionary(rtapi::timelineTransport());
    });
    sim.def("get_mode", [dictionary]() {
        return dictionary(rtapi::timelineTransport());
    });
    sim.def("set_mode", [dictionary](const std::string& mode) {
        return dictionary(rtapi::setSimulationMode(mode));
    }, pybind11::arg("mode"));
}
}
