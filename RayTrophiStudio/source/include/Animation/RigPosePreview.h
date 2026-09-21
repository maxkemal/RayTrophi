#pragma once
#include "Animation/ClipBinding.h"
namespace RigAuthoring {
struct PreviewJoint {
    std::string name, parent;
    Matrix4x4 world;
};
struct ClipPosePreview {
    ClipBindingReport binding;
    double time_seconds = 0, duration_seconds = 0;
    std::string source_pose_source, target_pose_source;
    std::vector<PreviewJoint> source, target;
};
// Model-space preview; scene placement and runtime/controller state are untouched.
bool sampleRigPose(const RayTrophi::NodeHierarchy&, const AnimationData* clip,
                   double timeSeconds, std::vector<PreviewJoint>&, std::string& error);
}
