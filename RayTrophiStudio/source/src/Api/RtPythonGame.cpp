#include "RtPythonGame.h"
#include "Api/RtApiGame.h"
#include "RtPyCommon.h"

namespace py = pybind11;
namespace {
nlohmann::json fromPython(const py::dict &value) {
    const auto text = py::cast<std::string>(py::module_::import("json").attr("dumps")(value));
    return nlohmann::json::parse(text);
}
py::object toPython(const nlohmann::json &value) {
    return py::module_::import("json").attr("loads")(value.dump());
}
} // namespace

namespace rtpy {
void registerGameBindings(py::module_ &module) {
    auto game = module.def_submodule("game", "Shared room rules and session state");
    game.def(
        "session",
        [](const std::string &operation, const py::dict &config) {
            nlohmann::json output;
            requireResult(rtapi::gameSessionCommand(operation, fromPython(config), output));
            return toPython(output);
        },
        py::arg("operation") = "status", py::arg("config") = py::dict());
    game.def(
        "post_event",
        [](const py::dict &event) { requireResult(rtapi::postGameEvent(fromPython(event))); },
        py::arg("event"));
    game.def("rules", [] {
        nlohmann::json output;
        requireResult(rtapi::getGameRules(output));
        return toPython(output);
    });
}
} // namespace rtpy
