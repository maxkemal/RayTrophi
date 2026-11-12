#include "RtDomainVisualBindings.h"
#include "Api/DomainVisualApi.h"
#include <pybind11/stl.h>

namespace rtpy {
namespace py = pybind11;
namespace {
py::object result(const rtapi::Result& status, const nlohmann::json& value) {
    if (!status.ok) {
        throw std::runtime_error(status.error);
    }
    return py::module_::import("json").attr("loads")(value.dump());
}
}

void registerDomainVisualBindings(py::module_& fluid) {
    fluid.def("visual_samples", [](const std::string& domain, uint64_t offset, uint64_t limit) {
        nlohmann::json output;
        const auto status = rtapi::getDomainVisualSamples(domain, output, offset, limit);
        return result(status, output);
    }, py::arg("domain"), py::arg("offset") = 0, py::arg("limit") = 0);
    fluid.def("set_visual_samples", [](const std::string& domain, const py::kwargs& kwargs) {
        const auto patch = nlohmann::json::parse(py::module_::import("json")
            .attr("dumps")(kwargs).cast<std::string>());
        nlohmann::json output;
        auto status = rtapi::setDomainVisualSamples(domain, patch);
        if (status.ok) {
            status = rtapi::getDomainVisualSamples(domain, output);
        }
        return result(status, output);
    }, py::arg("domain"));
    fluid.def("reset_visual_samples", [](const std::string& domain) {
        nlohmann::json output;
        auto status = rtapi::resetDomainVisualSamples(domain);
        if (status.ok) {
            status = rtapi::getDomainVisualSamples(domain, output);
        }
        return result(status, output);
    }, py::arg("domain"));
}
} // namespace rtpy
