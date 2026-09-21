#include "Api/RtApi.h"
#include "RtRigIKBindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
namespace py = pybind11;
namespace {
void require(const rtapi::Result &r) {
    if (!r.ok)
        throw std::invalid_argument(r.error);
}
Vec3 point(const std::vector<float> &p) {
    if (p.size() != 3)
        throw std::invalid_argument("rig_ik_invalid_target");
    return Vec3(p[0], p[1], p[2]);
}
} // namespace
void registerRigIKPython(py::module_ &rig) {
    auto recipe = [](float fps, float cadence, int cycles, float stride, float stepHeight,
                     float bodyBounce, float armSwing, float bodyMotion) {
        RigAuthoring::HumanWalkRecipe value;
        value.fps = fps;
        value.cadence = cadence;
        value.cycles = cycles;
        value.stride = stride;
        value.stepHeight = stepHeight;
        value.bodyBounce = bodyBounce;
        value.armSwing = armSwing;
        value.bodyMotion = bodyMotion;
        return value;
    };
    rig.def(
        "preview_human_walk",
        [recipe](const std::string &c, float fps, float cadence, int cycles, float stride,
                 float stepHeight, float bodyBounce, float armSwing, float bodyMotion) {
            nlohmann::json out;
            require(rtapi::previewRigHumanWalk(
                c,
                recipe(fps, cadence, cycles, stride, stepHeight, bodyBounce, armSwing, bodyMotion),
                out));
            return py::module_::import("json").attr("loads")(out.dump());
        },
        py::arg("character"), py::arg("fps") = 30.f, py::arg("cadence") = 100.f,
        py::arg("cycles") = 2, py::arg("stride") = .35f, py::arg("step_height") = .06f,
        py::arg("body_bounce") = .02f, py::arg("arm_swing") = .7f, py::arg("body_motion") = .75f);
    rig.def(
        "create_human_walk_clip",
        [recipe](const std::string &c, const std::string &name, uint64_t revision, float fps,
                 float cadence, int cycles, float stride, float stepHeight, float bodyBounce,
                 float armSwing, float bodyMotion) {
            require(rtapi::createRigHumanWalkClip(
                c, name,
                recipe(fps, cadence, cycles, stride, stepHeight, bodyBounce, armSwing, bodyMotion),
                revision));
        },
        py::arg("character"), py::arg("name"), py::arg("rig_revision").noconvert(),
        py::arg("fps") = 30.f, py::arg("cadence") = 100.f, py::arg("cycles") = 2,
        py::arg("stride") = .35f, py::arg("step_height") = .06f, py::arg("body_bounce") = .02f,
        py::arg("arm_swing") = .7f, py::arg("body_motion") = .75f);
    rig.def(
        "set_ik_spline",
        [](const std::string &c, const std::string &control,
           const std::vector<std::vector<float>> &values, bool enabled, uint64_t revision) {
            if (values.size() > 2) {
                throw std::invalid_argument("rig_ik_invalid_spline");
            }
            std::vector<Vec3> points;
            for (const auto &value : values) {
                if (value.size() != 3) {
                    throw std::invalid_argument("rig_ik_invalid_spline");
                }
                points.emplace_back(value[0], value[1], value[2]);
            }
            require(rtapi::setRigIKSpline(c, control, points, enabled, revision));
        },
        py::arg("character"), py::arg("control"), py::arg("points_world"),
        py::arg("enabled").noconvert(), py::arg("rig_revision").noconvert());

    rig.def(
        "get_ik_channels",
        [](const std::string &c) {
            nlohmann::json out;
            require(rtapi::getRigIKChannels(c, out));
            return py::module_::import("json").attr("loads")(out.dump());
        },
        py::arg("character"));
    rig.def(
        "insert_ik_key",
        [](const std::string &c, const std::string &control, uint64_t revision) {
            require(rtapi::insertRigIKKey(c, control, revision));
        },
        py::arg("character"), py::arg("control"), py::arg("rig_revision").noconvert());
    rig.def(
        "clear_ik_channels",
        [](const std::string &c, const std::string &control, uint64_t revision) {
            require(rtapi::clearRigIKChannels(c, control, revision));
        },
        py::arg("character"), py::arg("control"), py::arg("rig_revision").noconvert());
    rig.def(
        "set_ik_contact_interval",
        [](const std::string &c, const std::string &control, int start, int end,
           uint64_t revision) {
            require(rtapi::setRigIKContactInterval(c, control, start, end, revision));
        },
        py::arg("character"), py::arg("control"), py::arg("start_frame").noconvert(),
        py::arg("end_frame").noconvert(), py::arg("rig_revision").noconvert());
    rig.def(
        "bake_ik_channels",
        [](const std::string &c, const std::string &name, int start, int end, uint64_t revision) {
            require(rtapi::bakeRigIKChannels(c, name, start, end, revision));
        },
        py::arg("character"), py::arg("name"), py::arg("start_frame").noconvert(),
        py::arg("end_frame").noconvert(), py::arg("rig_revision").noconvert());
    rig.def(
        "create_chain_control",
        [](const std::string &c, const std::string &chain, uint64_t revision) {
            require(rtapi::createRigChainControl(c, chain, revision));
        },
        py::arg("character"), py::arg("chain"), py::arg("rig_revision").noconvert());
    rig.def(
        "create_aim_control",
        [](const std::string &c, const std::string &role, uint64_t revision) {
            require(rtapi::createRigAimControl(c, role, revision));
        },
        py::arg("character"), py::arg("role"), py::arg("rig_revision").noconvert());
    rig.def(
        "get_controls",
        [](const std::string &c) {
            nlohmann::json out;
            require(rtapi::getRigControls(c, out));
            return py::module_::import("json").attr("loads")(out.dump());
        },
        py::arg("character"));
    rig.def(
        "create_controls",
        [](const std::string &c, uint64_t revision, const py::object &controls) {
            auto value = nlohmann::json::parse(
                py::module_::import("json").attr("dumps")(controls).cast<std::string>(), nullptr,
                false);
            if (value.is_discarded())
                throw std::invalid_argument("rig_ik_invalid_controls");
            require(rtapi::createRigControls(c, value, revision));
        },
        py::arg("character"), py::arg("rig_revision").noconvert(),
        py::arg("controls") = py::none());
    rig.def(
        "select_control",
        [](const std::string &c, const std::string &control, const std::string &handle) {
            require(rtapi::selectRigControl(c, control, handle));
        },
        py::arg("character"), py::arg("control"), py::arg("handle") = "target");
    rig.def(
        "set_ik_target",
        [](const std::string &c, const std::string &control, const std::vector<float> &target,
           const std::vector<float> &pole, uint64_t revision) {
            require(rtapi::setRigIKTarget(c, control, point(target), point(pole), revision));
        },
        py::arg("character"), py::arg("control"), py::arg("target_world"), py::arg("pole_world"),
        py::arg("rig_revision").noconvert());
    rig.def(
        "set_ik_orientation",
        [](const std::string &c, const std::string &control, const std::vector<float> &q,
           bool enabled, uint64_t revision) {
            if (q.size() != 4)
                throw std::invalid_argument("rig_ik_invalid_orientation");
            require(rtapi::setRigIKOrientation(c, control, Quaternion(q[0], q[1], q[2], q[3]),
                                               enabled, revision));
        },
        py::arg("character"), py::arg("control"), py::arg("orientation_world"),
        py::arg("enabled").noconvert(), py::arg("rig_revision").noconvert());
    rig.def(
        "set_ik_fk",
        [](const std::string &c, const std::string &control, float blend, uint64_t revision) {
            require(rtapi::setRigIKFK(c, control, blend, revision));
        },
        py::arg("character"), py::arg("control"), py::arg("blend"),
        py::arg("rig_revision").noconvert());
    rig.def(
        "match_ik_to_fk",
        [](const std::string& c, const std::string& control, uint64_t revision) {
            require(rtapi::matchRigIKToFK(c, control, revision));
        },
        py::arg("character"), py::arg("control"), py::arg("rig_revision").noconvert());
    rig.def(
        "remove_ik_key",
        [](const std::string &c, const std::string &control, uint64_t revision) {
            require(rtapi::removeRigIKKey(c, control, revision));
        },
        py::arg("character"), py::arg("control"), py::arg("rig_revision").noconvert());
    rig.def(
        "match_fk_to_ik",
        [](const std::string& c, const std::string& control, uint64_t revision) {
            require(rtapi::matchRigFKToIK(c, control, revision));
        },
        py::arg("character"), py::arg("control"), py::arg("rig_revision").noconvert());
    rig.def(
        "set_ik_contact",
        [](const std::string &c, const std::string &control, bool enabled, uint64_t revision) {
            require(rtapi::setRigIKContact(c, control, enabled, revision));
        },
        py::arg("character"), py::arg("control"), py::arg("enabled").noconvert(),
        py::arg("rig_revision").noconvert());
}
