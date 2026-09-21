#pragma once
#include "scene_data.h"
namespace RigAuthoring {
void initializeRestPoseDefaults(SceneData&);
// Synchronize canonical flat vertices before CPU rendering/picking BVH rebuild.
bool synchronizeCpuSkinning(SceneData&,const std::vector<Matrix4x4>& skin);
bool needsFileAnimationEvaluation(const SceneData&);
bool isRestPoseView(const SceneData&,const std::string& character);
bool setPoseView(SceneData&,const std::string& character,const std::string& mode,std::string& error);
bool applyRestPoseView(SceneData&,SceneData::ImportedModelContext&,std::vector<Matrix4x4>& skin,bool cpu,bool& changed);
bool applyPoseViewResume(SceneData&,SceneData::ImportedModelContext&,const std::vector<Matrix4x4>& skin,bool cpu);
}
