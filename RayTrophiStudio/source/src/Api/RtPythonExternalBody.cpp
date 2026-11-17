#include "RtPythonExternalBody.h"
#include "Api/RtApiExternalBody.h"
#include "Api/RtApiCharacter.h"
#include "RtPyCommon.h"

namespace py = pybind11;
namespace {
nlohmann::json requestJson(const py::dict &request) {
    return nlohmann::json::parse(
        py::cast<std::string>(py::module_::import("json").attr("dumps")(request)));
}
} // namespace
namespace rtpy {
void registerExternalBodyBindings(py::module_ &module) {
    auto anim = module.def_submodule("anim");
    anim.def("delete_character", [](const std::string &character) {
        requireResult(rtapi::deleteCharacter(character));
    }, py::arg("character"));
    anim.def(
        "set_external_pose",
        [](const py::object &character, const py::object &person_id, const py::dict &config) {
            py::dict request;
            request["character"] = character;
            request["person_id"] = person_id;
            request["config"] = config;
            requireResult(rtapi::setExternalBodyPose(requestJson(request)));
        },
        py::arg("character"), py::arg("person_id"), py::arg("config"));
    anim.def("external_poses", [] {
        nlohmann::json output;
        requireResult(rtapi::getExternalBodyPoses(output));
        return py::module_::import("json").attr("loads")(output.dump());
    });
    anim.def(
        "clear_external_pose",
        [](const py::object &character) {
            py::dict request;
            request["character"] = character;
            requireResult(rtapi::clearExternalBodyPose(requestJson(request)));
        },
        py::arg("character"));
}
} // namespace rtpy
