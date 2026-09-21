#pragma once
#include "Animation/RigIK.h"

namespace RigAuthoring {
bool validateSplineIKPose(const IKPose& pose);
std::vector<Vec3> splineIKWorldCurve(const Vec3& anchor, const IKPose& pose, int samples);
bool seedSplineIKChain(const IKPose& pose, const Matrix4x4& placement,
                       const std::vector<float>& lengths, std::vector<Vec3>& points,
                       std::string& error);
}
