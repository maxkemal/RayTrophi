// Studio-only Python transport. Shared services and IPC live in the original module.
#include "RtPostBindings.h"
#include "RtApiInternal.h"
#include "PostProcess/PostService.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

using nlohmann::json;
namespace py = pybind11;

namespace {
json unbound() {
    return {{"__error", "rtapi is not bound to a UIContext"}};
}

py::object toPython(const json& j) {
    if (j.contains("__error"))
        throw std::runtime_error(j["__error"].get<std::string>());
    return py::module_::import("json").attr("loads")(j.dump());
}
} // namespace

void registerPostExposurePython(py::module_& post) {
    post.def("get_exposure",
             []() { return toPython(rtapi::g_ctx ? rtpost::inspect(*rtapi::g_ctx) : unbound()); });
    post.def("configure_exposure", [](const py::kwargs& kw) {
        // allow_nan=false rejects non-finite Python inputs before JSON conversion.
        const auto text = py::module_::import("json")
                              .attr("dumps")(kw, py::arg("allow_nan") = false)
                              .cast<std::string>();
        return toPython(rtapi::g_ctx ? rtpost::configure(*rtapi::g_ctx, json::parse(text))
                                     : unbound());
    });
    post.def("reset_exposure",
             []() { return toPython(rtapi::g_ctx ? rtpost::reset(*rtapi::g_ctx) : unbound()); });
}
