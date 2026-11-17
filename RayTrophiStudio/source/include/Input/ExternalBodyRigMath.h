#pragma once

#include "Animation/RigIK.h"
#include "Animation/RigJointRules.h"
#include "Input/BodyPoseRuntime.h"
#include <map>

namespace Input {
struct ExternalRigTarget {
    std::string joint;
    std::string pole;
};

struct ExternalRigMapping {
    std::string anchor_bone;
    std::string anchor_joint = "pelvis";
    std::map<std::string, ExternalRigTarget> controls;
    double min_confidence = 0.5;
    float blend = 1.0f;
    double teleport_metres = 1.0;
};

bool parseExternalRigMapping(const nlohmann::json &config, const RayTrophi::NodeHierarchy &rest,
                             const std::vector<RigAuthoring::IKControl> &controls,
                             ExternalRigMapping &output, std::string &error);

// Uses the engine's existing IK solvers and joint limits. No face geometry or
// renderer state enters this independently testable retarget step.
bool solveExternalBodyRig(const RayTrophi::NodeHierarchy &rest,
                          const std::vector<RigAuthoring::IKControl> &controls,
                          const std::vector<RigAuthoring::JointRule> &limits,
                          const Matrix4x4 &placement, const BodyPerson &body,
                          const ExternalRigMapping &mapping, RayTrophi::NodeHierarchy &output,
                          std::vector<std::string> &limit_hits, std::string &error);
} // namespace Input
