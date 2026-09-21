#pragma once
#include "Animation/RigIK.h"

namespace RigAuthoring {
bool matchAimIKPose(const RayTrophi::NodeHierarchy& hierarchy, const IKControl& control,
                    const Matrix4x4& placement, IKPose& output, std::string& error);
bool solveAimIKPose(const RayTrophi::NodeHierarchy& input, const IKControl& control,
                    const IKPose& pose, const Matrix4x4& placement,
                    RayTrophi::NodeHierarchy& output, std::string& error);
float aimIKErrorDegrees(const RayTrophi::NodeHierarchy& hierarchy, const IKControl& control,
                        const IKPose& pose, const Matrix4x4& placement);
}
