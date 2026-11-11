#include "RtPythonMatterSeedSources.h"
#include "Api/DomainCreationApi.h"
#include "Api/MatterSeedSourcesApi.h"
#include <stdexcept>

namespace rtpy {
namespace {
pybind11::object call(const std::string &domain, const std::string &operation, uint64_t id,
                      const pybind11::dict &settings) {
    auto json = pybind11::module_::import("json");
    auto patch = nlohmann::json::parse(json.attr("dumps")(settings).cast<std::string>());
    nlohmann::json result;
    const auto status = rtapi::matterSeedSources(domain, operation, id, patch, result);
    if (!status.ok) {
        throw std::runtime_error(status.error);
    }
    return json.attr("loads")(result.dump());
}
} // namespace

void registerMatterSeedSources(pybind11::module_ &fluid) {
    namespace py = pybind11;
    fluid.def(
        "create_placed_domain",
        [](const std::string &name, const std::string &source) {
            nlohmann::json result;
            const auto status = rtapi::createPlacedMatterDomain(name, source, result);
            if (!status.ok) {
                throw std::runtime_error(status.error);
            }
            return py::module_::import("json").attr("loads")(result.dump());
        },
        py::arg("name") = "", py::arg("source_object") = "");
    fluid.def(
        "get_seed_sources",
        [](const std::string &domain) { return call(domain, "list", 0, py::dict()); },
        py::arg("domain"));
    fluid.def(
        "select_seed_source",
        [](const std::string &domain, uint64_t id) {
            return call(domain, "select", id, py::dict());
        },
        py::arg("domain"), py::arg("id") = 0);
    fluid.def(
        "add_seed_source",
        [](const std::string &domain, const py::dict &settings) {
            return call(domain, "add", 0, settings);
        },
        py::arg("domain"), py::arg("settings"));
    fluid.def(
        "update_seed_source",
        [](const std::string &domain, uint64_t id, const py::dict &settings) {
            return call(domain, "update", id, settings);
        },
        py::arg("domain"), py::arg("id"), py::arg("settings"));
    fluid.def(
        "remove_seed_source",
        [](const std::string &domain, uint64_t id) {
            return call(domain, "remove", id, py::dict());
        },
        py::arg("domain"), py::arg("id"));
    fluid.def(
        "apply_seed_sources",
        [](const std::string &domain) { return call(domain, "apply", 0, py::dict()); },
        py::arg("domain"));
    fluid.def(
        "set_seed_source_settings",
        [](const std::string &domain, const py::dict &settings) {
            return call(domain, "settings", 0, settings);
        },
        py::arg("domain"), py::arg("settings"));
}
} // namespace rtpy
