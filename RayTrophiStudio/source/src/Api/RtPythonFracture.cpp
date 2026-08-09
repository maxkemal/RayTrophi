#include "RtPythonFracture.h"

#include "Api/RtApi.h"

#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {
void require(const rtapi::Result& result) {
    if (!result.ok) throw std::runtime_error(result.error);
}

py::dict toDict(const rtapi::FractureGroupInfo& info) {
    py::dict out;
    out["group"] = info.group;
    out["shard_count"] = info.shard_count;
    out["broken_count"] = info.broken_count;
    out["base_break_impulse"] = info.base_break_impulse;
    out["effective_break_impulse"] = info.effective_break_impulse;
    out["integrity_weakening"] = info.integrity_weakening;
    out["integrity_exponent"] = info.integrity_exponent;
    out["minimum_threshold_scale"] = info.minimum_threshold_scale;
    out["mean_integrity"] = info.mean_integrity;
    out["minimum_integrity"] = info.minimum_integrity;
    out["remaining_support_ratio"] = info.remaining_support_ratio;
    return out;
}
} // namespace

namespace rtpy {
void registerFractureBindings(py::module_& physics) {
    physics.def("make_fracture_group",
        [](const std::string& group, const std::vector<std::string>& shards,
           float threshold, bool weakening, float exponent, float minimum) {
            rtapi::FractureGroupInfo info;
            require(rtapi::makePhysicsFractureGroup(
                group, shards, threshold, weakening, exponent, minimum, info));
            return toDict(info);
        }, py::arg("group"), py::arg("shard_objects"),
           py::arg("break_impulse") = 5.0f,
           py::arg("integrity_weakening") = true,
           py::arg("integrity_exponent") = 1.5f,
           py::arg("minimum_threshold_scale") = 0.15f);

    physics.def("fracture_group", [](const std::string& group) {
        rtapi::FractureGroupInfo info;
        require(rtapi::getPhysicsFractureGroup(group, info));
        return toDict(info);
    }, py::arg("group"));

    physics.def("break_fracture_group", [](const std::string& group, float strength) {
        require(rtapi::breakPhysicsFractureGroup(group, strength));
    }, py::arg("group"), py::arg("strength") = 6.0f);

    physics.def("apply_fracture_impulse",
        [](const std::string& group, const py::tuple& point,
           const py::tuple& direction, float impulse) {
            if (point.size() != 3 || direction.size() != 3)
                throw std::runtime_error("point and direction must contain three values");
            const Vec3 p(py::cast<float>(point[0]), py::cast<float>(point[1]),
                         py::cast<float>(point[2]));
            const Vec3 d(py::cast<float>(direction[0]), py::cast<float>(direction[1]),
                         py::cast<float>(direction[2]));
            bool triggered = false;
            require(rtapi::applyPhysicsFractureImpulse(group, p, d, impulse, triggered));
            return triggered;
        }, py::arg("group"), py::arg("point") = py::make_tuple(0, 0, 0),
           py::arg("direction") = py::make_tuple(0, 1, 0),
           py::arg("impulse") = 1.0f);
}
} // namespace rtpy
