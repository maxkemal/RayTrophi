#include "Input/ExternalBodyRigMath.h"
#include "Animation/RigBindMath.h"
#include "Animation/RigPosePreview.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace Input {
namespace {
bool jointName(const std::string &name) {
    return std::any_of(kBodyJoints.begin(), kBodyJoints.end(),
                       [&](const char *known) { return name == known; });
}
bool scalar(const nlohmann::json &value, double low, double high) {
    return value.is_number() && std::isfinite(value.get<double>()) && value.get<double>() >= low &&
           value.get<double>() <= high;
}
Vec3 position(const std::array<double, 4> &joint) {
    return Vec3(static_cast<float>(joint[0]), static_cast<float>(joint[1]),
                static_cast<float>(joint[2]));
}
} // namespace

bool parseExternalRigMapping(const nlohmann::json &config, const RayTrophi::NodeHierarchy &rest,
                             const std::vector<RigAuthoring::IKControl> &controls,
                             ExternalRigMapping &output, std::string &error) {
    if (!config.is_object()) {
        error = "external_pose_config_requires_object";
        return false;
    }
    const std::set<std::string> keys = {"anchor_bone",    "anchor_joint", "controls",
                                        "min_confidence", "blend",        "teleport_metres"};
    for (const auto &entry : config.items()) {
        if (!keys.count(entry.key())) {
            error = "unknown_external_pose_field: " + entry.key();
            return false;
        }
    }
    if (!config.contains("anchor_bone") || !config["anchor_bone"].is_string() ||
        !rest.find(config["anchor_bone"].get<std::string>()) || rest.empty() ||
        (config.contains("anchor_joint") && !config["anchor_joint"].is_string()) ||
        !config.contains("controls") || !config["controls"].is_object() ||
        config["controls"].empty() || config["controls"].size() > 16) {
        error = "external_pose_requires_anchor_bone_and_controls";
        return false;
    }
    ExternalRigMapping staged;
    staged.anchor_bone = config["anchor_bone"].get<std::string>();
    staged.anchor_joint = config.value("anchor_joint", std::string("pelvis"));
    if (!jointName(staged.anchor_joint) ||
        (config.contains("min_confidence") && !scalar(config["min_confidence"], 0.1, 1)) ||
        (config.contains("blend") && !scalar(config["blend"], 0, 1)) ||
        (config.contains("teleport_metres") && !scalar(config["teleport_metres"], 0.1, 10))) {
        error = "invalid_external_pose_joint_confidence_or_blend";
        return false;
    }
    staged.min_confidence = config.value("min_confidence", 0.5);
    staged.blend = config.value("blend", 1.0f);
    staged.teleport_metres = config.value("teleport_metres", 1.0);
    std::vector<RigAuthoring::IKControl> selected;
    for (const auto &entry : config["controls"].items()) {
        const auto control =
            std::find_if(controls.begin(), controls.end(),
                         [&](const auto &candidate) { return candidate.name == entry.key(); });
        const auto &value = entry.value();
        if (control == controls.end() || !value.is_object() || !value.contains("joint") ||
            !value["joint"].is_string() || (value.contains("pole") && !value["pole"].is_string()) ||
            value.size() != 1u + static_cast<std::size_t>(value.contains("pole"))) {
            error = "invalid_external_pose_control: " + entry.key();
            return false;
        }
        ExternalRigTarget target{value["joint"].get<std::string>(), value.value("pole", "")};
        if (!jointName(target.joint) || (!target.pole.empty() && !jointName(target.pole))) {
            error = "unknown_external_pose_joint";
            return false;
        }
        selected.push_back(*control);
        staged.controls.emplace(entry.key(), std::move(target));
    }
    if (!RigAuthoring::validateIKControls(selected, rest, error)) {
        return false;
    }
    output = std::move(staged);
    return true;
}

bool solveExternalBodyRig(const RayTrophi::NodeHierarchy &rest,
                          const std::vector<RigAuthoring::IKControl> &controls,
                          const std::vector<RigAuthoring::JointRule> &limits,
                          const Matrix4x4 &placement, const BodyPerson &body,
                          const ExternalRigMapping &mapping, RayTrophi::NodeHierarchy &output,
                          std::vector<std::string> &limit_hits, std::string &error) {
    const auto anchor = body.joints.find(mapping.anchor_joint);
    if (body.confidence < mapping.min_confidence || anchor == body.joints.end() ||
        anchor->second[3] < mapping.min_confidence) {
        error = "external_pose_low_confidence";
        return false;
    }
    Matrix4x4 inverse;
    if (!RigAuthoring::bindAffineInverse(placement, inverse)) {
        error = "external_pose_invalid_placement";
        return false;
    }
    std::vector<RigAuthoring::PreviewJoint> rest_joints;
    if (!RigAuthoring::sampleRigPose(rest, nullptr, 0, rest_joints, error)) {
        return false;
    }
    const auto source =
        std::find_if(rest_joints.begin(), rest_joints.end(),
                     [&](const auto &joint) { return joint.name == mapping.anchor_bone; });
    if (source == rest_joints.end() || rest.rootIndex() < 0) {
        error = "external_pose_anchor_unresolved";
        return false;
    }
    auto translated = rest;
    const Vec3 delta =
        (position(anchor->second) - placement.transform_point(source->world.getTranslation())) *
        mapping.blend;
    const Vec3 local_delta = inverse.transform_vector(delta);
    auto &root = translated.nodes[static_cast<std::size_t>(translated.rootIndex())];
    root.localBind = Matrix4x4::translation(local_delta) * root.localBind;

    std::vector<RigAuthoring::IKControl> selected;
    RigAuthoring::IKPoses poses;
    for (const auto &control : controls) {
        const auto binding = mapping.controls.find(control.name);
        if (binding == mapping.controls.end()) {
            continue;
        }
        selected.push_back(control);
        const auto point = body.joints.find(binding->second.joint);
        if (point == body.joints.end() || point->second[3] < mapping.min_confidence) {
            continue; // low-confidence control stays in rest, never invent a target
        }
        RigAuthoring::IKPose pose;
        if (!RigAuthoring::matchIKPose(translated, control, placement, pose, error)) {
            return false;
        }
        pose.enabled = true;
        pose.blend = mapping.blend;
        pose.target = position(point->second);
        if (!binding->second.pole.empty()) {
            const auto pole = body.joints.find(binding->second.pole);
            if (pole == body.joints.end() || pole->second[3] < mapping.min_confidence) {
                continue;
            }
            pose.pole = position(pole->second);
        }
        poses.emplace(control.name, pose);
    }
    RayTrophi::NodeHierarchy solved;
    if (!RigAuthoring::solveIKPose(translated, selected, poses, placement, solved, error)) {
        return false;
    }
    // Translation follows the tracked anchor; rotation/limb limits stay relative
    // to the translated rest. A root lock must not cancel the room translation.
    return RigAuthoring::constrainJointPose(translated, limits, solved, output, limit_hits, error);
}
} // namespace Input
