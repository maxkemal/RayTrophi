#include "RtPythonDomainWalls.h"
#include "Api/DomainWallsApi.h"
#include <stdexcept>

namespace rtpy {
namespace {
pybind11::object wallResult(const rtapi::Result& status, const nlohmann::json& result) {
    if (!status.ok) {
        throw std::runtime_error(status.error);
    }
    return pybind11::module_::import("json").attr("loads")(result.dump());
}
} // namespace

void registerDomainWallBindings(pybind11::module_& fluid) {
    fluid.def("get_wall_contact", [](const std::string& domain) {
        nlohmann::json result;
        const auto status = rtapi::getDomainWallContact(domain, result);
        return wallResult(status, result);
    }, pybind11::arg("domain"));
    fluid.def("set_wall_contact", [](const std::string& domain, const pybind11::dict& settings) {
        const auto json = pybind11::module_::import("json");
        const auto patch = nlohmann::json::parse(
            json.attr("dumps")(settings).cast<std::string>());
        nlohmann::json result;
        const auto status = rtapi::setDomainWallContact(domain, patch, result);
        return wallResult(status, result);
    }, pybind11::arg("domain"), pybind11::arg("settings"),
        "Set shared liquid/MPM/DEM domain-wall friction [0,2] restitution [0,1], and adhesion_pa [0,1e9] Pa.");
}

} // namespace rtpy
