// Studio-only Python transport. Shared services and IPC live in the original module.
#include "RtVolumeMaterialBindings.h"
#include "RtApiInternal.h"
#include "VolumeMaterialAuthoring.h"
#include "VolumeMaterialService.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

using nlohmann::json;
namespace py = pybind11;

namespace {
py::object toPython(const json& result) {
    if (result.contains("__error")) {
        throw std::runtime_error(result["__error"].get<std::string>());
    }
    return py::module_::import("json").attr("loads")(result.dump());
}
} // namespace

void registerVolumeMaterialPython(py::module_& module) {
    auto volume = module.def_submodule("volume", "Shared gas/VDB material authoring");
    volume.def("list_materials", [] {
        if (!rtapi::g_ctx) {
            throw std::runtime_error("rtapi is not bound");
        }
        return toPython(volume_material::execute(*rtapi::g_ctx, "list", ""));
    });
    volume.def(
        "get_material",
        [](const std::string& target) {
            if (!rtapi::g_ctx) {
                throw std::runtime_error("rtapi is not bound");
            }
            return toPython(volume_material::execute(*rtapi::g_ctx, "get", target));
        },
        py::arg("target"));
    volume.def(
        "set_material",
        [](const std::string& target, const py::kwargs& kwargs) {
            if (!rtapi::g_ctx) {
                throw std::runtime_error("rtapi is not bound");
            }
            const auto text = py::module_::import("json")
                                  .attr("dumps")(kwargs, py::arg("allow_nan") = false)
                                  .cast<std::string>();
            return toPython(
                volume_material::execute(*rtapi::g_ctx, "set", target, json::parse(text)));
        },
        py::arg("target"));
}
