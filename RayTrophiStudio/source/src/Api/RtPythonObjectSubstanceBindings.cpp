// Studio-only Python transport. Shared services and IPC live in the original module.
#include "RtObjectSubstanceBindings.h"
#include "RtApiInternal.h"
#include "ObjectSubstance.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

using nlohmann::json;
namespace py = pybind11;

namespace rtpy {
void registerObjectSubstanceBindings(pybind11::module_& substance) {
    namespace py = pybind11;
    using json = nlohmann::json;
    for (const char* operation :
         {"list_objects", "get_object", "set_object", "clear_object", "adopt_object"}) {
        substance.def(operation, [name = std::string(operation)](const py::kwargs& kwargs) {
            if (!rtapi::g_ctx) {
                throw std::runtime_error("No active scene context");
            }
            const auto params = json::parse(
                py::cast<std::string>(py::module_::import("json").attr("dumps")(kwargs)));
            const auto result = matter::invokeObjectSubstance(*rtapi::g_ctx, name, params);
            return py::module_::import("json").attr("loads")(result.dump());
        });
    }
}
} // namespace rtpy
