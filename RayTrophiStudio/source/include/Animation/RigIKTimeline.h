#pragma once
#include "scene_data.h"
namespace RigAuthoring {
const AnimationData* authoredIKClip(const SceneData&,const std::string& character);
bool validateIKClipChannels(const SceneData&,const AnimationData&,std::string& error);
bool effectiveIKControls(const SceneData&,const std::string& character,bool preview,IKPoses&,std::string& error);
bool bakeIKChannels(const SceneData&,const SceneData::ImportedModelContext&,const AnimationData&,int start,int end,AnimationData& output,std::string& error);
}
