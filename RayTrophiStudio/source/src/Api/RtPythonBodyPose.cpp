#include "RtPythonBodyPose.h"
#include "Api/RtApiBodyPose.h"
#include "RtPyCommon.h"

namespace py = pybind11;

namespace rtpy {
void registerBodyPoseBindings(py::module_ &module) {
    auto input = module.def_submodule("input", "Transient room-space body input");
    input.def(
        "set_body_poses",
        [](const py::object &t, const py::list &persons) {
            const auto codec = py::module_::import("json");
            py::dict packet;
            packet["t"] = t;
            packet["persons"] = persons;
            const auto text = py::cast<std::string>(codec.attr("dumps")(packet));
            requireResult(rtapi::setBodyPoses(nlohmann::json::parse(text)));
        },
        py::arg("t"), py::arg("persons"));
    input.def("body_poses", [] {
        nlohmann::json value;
        requireResult(rtapi::getBodyPoses(value));
        return py::module_::import("json").attr("loads")(value.dump());
    });
    input.def("clear_body_poses", [] { requireResult(rtapi::clearBodyPoses()); });
}
} // namespace rtpy
