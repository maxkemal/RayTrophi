#include "RtMatterModels.h"

#include <pybind11/stl.h>

namespace rtpy {

void registerMatterModelBindings(pybind11::module_& fluid) {
    fluid.def("grain_settings", [](const std::string& domain) {
        return pybind11::module_::import("json").attr("loads")(
            rtapi::matterGrainSettings(domain, nlohmann::json::object(), false).dump());
    }, pybind11::arg("domain"));
    fluid.def("set_grain_settings", [](const std::string& domain, const pybind11::kwargs& kwargs) {
        const auto patch = nlohmann::json::parse(pybind11::module_::import("json")
            .attr("dumps")(kwargs).cast<std::string>());
        return pybind11::module_::import("json").attr("loads")(
            rtapi::matterGrainSettings(domain, patch, true).dump());
    }, pybind11::arg("domain"));
    fluid.def("grain_reference", [](const pybind11::kwargs& kwargs) {
        const auto params = nlohmann::json::parse(pybind11::module_::import("json")
            .attr("dumps")(kwargs).cast<std::string>());
        return pybind11::module_::import("json").attr("loads")(
            rtapi::runGrainReferenceProbe(params).dump());
    }, "Bounded transient CPU grain experiment; does not mutate the scene or MPM solver.");
    fluid.def("set_pore_exchange", [](const std::string& domain, const pybind11::kwargs& kwargs) {
        auto patch = nlohmann::json::parse(pybind11::module_::import("json").attr("dumps")(
            kwargs).cast<std::string>());
        return pybind11::module_::import("json").attr("loads")(
            rtapi::setMatterPoreExchange(domain, patch).dump());
    }, pybind11::arg("domain"));
    fluid.def("matter_models", [](const std::string& domain, bool include_transfer) {
        return pybind11::module_::import("json").attr("loads")(
            rtapi::getMatterModels(domain, include_transfer).dump());
    }, pybind11::arg("domain"), pybind11::arg("include_transfer") = false);
}

} // namespace rtpy
