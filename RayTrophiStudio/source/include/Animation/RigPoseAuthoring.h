#pragma once
#include "Animation/RigPoseAuthoringMath.h"
#include "scene_data.h"
namespace RigAuthoring {
bool canAuthorPose(const SceneData&,const std::string& character,std::string& error);
bool enterPose(SceneData&,const std::string& character,std::string& error);
void leavePose(SceneData&);
void synchronizePoseFrame(SceneData&);
bool needsIKPlacementEvaluation(const SceneData&);
bool currentPoseHierarchy(const SceneData&,const std::string& character,RayTrophi::NodeHierarchy&,std::string& error,bool preview=false,std::vector<std::string>* limitHits=nullptr,bool evaluateIK=true);
bool applyAuthoringPose(SceneData&,SceneData::ImportedModelContext&,std::vector<Matrix4x4>& skin,bool cpu,bool& changed);
}
