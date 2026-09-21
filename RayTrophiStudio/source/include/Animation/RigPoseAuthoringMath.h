#pragma once
#include "Animation/RigPosePreview.h"
#include "Animation/RigView.h"
namespace RigAuthoring {
bool poseHierarchy(const RayTrophi::NodeHierarchy& rest,const AnimationData* clip,double seconds,const JointGlobals& locals,RayTrophi::NodeHierarchy& output,std::string& error);
bool insertPoseKeys(AnimationData& clip,const RayTrophi::NodeHierarchy& pose,const std::vector<std::string>& bones,double seconds,std::string& error);
bool removePoseKeys(AnimationData& clip, const RayTrophi::NodeHierarchy& pose,
                    const std::vector<std::string>& bones, double seconds, std::string& error);
}
