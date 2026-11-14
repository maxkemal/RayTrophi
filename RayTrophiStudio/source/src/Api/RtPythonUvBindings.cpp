// Studio-only Python transport. Shared services and IPC live in the original module.
#include "RtUvBindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

using nlohmann::json;
namespace py = pybind11;

using rt_uv_binding::operations;

void registerUvPython(pybind11::module_& module) {
    namespace py = pybind11;
    auto uv = module.def_submodule("uv", "Flat mesh UV authoring (main thread)");
    for (const char* operation : operations) {
        uv.def(
            operation,
            [operation = std::string(operation)](py::kwargs kwargs) {
                const auto codec = py::module_::import("json");
                const std::string encoded =
                    codec.attr("dumps")(kwargs, py::arg("allow_nan") = false).cast<std::string>();
                const json result = invokeUv(operation, json::parse(encoded));
                if (result.is_object() && result.contains("__error")) {
                    throw std::runtime_error(result.at("__error").get<std::string>());
                }
                return codec.attr("loads")(result.dump());
            },
            "Keyword arguments match the uv.* IPC method schema.");
    }
}
