#pragma once
#include "Animation/NodeHierarchy.h"
#include "Vec3.h"
#include "Quaternion.h"
#include "json.hpp"
#include <unordered_map>
#include <vector>
#include <string>
namespace RigAuthoring {
struct IKControl {
    std::string name, root, mid, tip;
    std::vector<std::string> chain;
    std::string solver = "two_bone";
    Vec3 aimAxis = Vec3(0, 1, 0);
    Vec3 upAxis = Vec3(0, 0, 1);
};
std::vector<std::string> ikControlBones(const IKControl&);
struct IKPose {
    bool enabled = false, contact = false;
    float blend = 1;
    bool splineEnabled = false;
    std::vector<Vec3> splineWorld; // Two cubic Bezier interior control points, in world space.
    bool orientationEnabled = false;
    Quaternion orientationWorld;                       // Unit quaternion, w/x/y/z.
    Vec3 target = Vec3(0, 0, 0), pole = Vec3(0, 0, 0); // World-space authoring handles.
};
using IKPoses = std::unordered_map<std::string, IKPose>;
bool validateIKControls(const std::vector<IKControl>&, const RayTrophi::NodeHierarchy&,
                        std::string& error);
nlohmann::json serializeIKControls(const std::vector<IKControl>&);
bool deserializeIKControls(const nlohmann::json&, const RayTrophi::NodeHierarchy&,
                           std::vector<IKControl>&, std::string& error);
bool matchIKPose(const RayTrophi::NodeHierarchy&, const IKControl&, const Matrix4x4& placement,
                 IKPose&, std::string& error);
bool solveIKPose(const RayTrophi::NodeHierarchy&, const std::vector<IKControl>&, const IKPoses&,
                 const Matrix4x4& placement, RayTrophi::NodeHierarchy& output, std::string& error);
nlohmann::json inspectIKPose(const RayTrophi::NodeHierarchy&, const std::vector<IKControl>&,
                             const IKPoses&, const Matrix4x4& placement);
bool hasEnabledIK(const IKPoses&);
bool ikDrivesBone(const std::vector<IKControl>&, const IKPoses&, const std::string& bone);
bool sameIKPoses(const IKPoses&, const IKPoses&);
}
