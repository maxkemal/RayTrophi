#pragma once
#include "json.hpp"
#include "Animation/NodeHierarchy.h"
#include <string>
struct SceneData;
namespace RigAuthoring {
bool fitSetup(const SceneData&,const std::string& character,const std::string& mesh,nlohmann::json&,std::string& error);
bool previewFit(const SceneData&,const std::string& character,const std::string& mesh,const nlohmann::json& landmarks,bool axesConfirmed,nlohmann::json&,std::string& error);
bool fittedHierarchy(const SceneData&,const std::string& character,const std::string& mesh,const nlohmann::json& preview,RayTrophi::NodeHierarchy&,std::string& error);
}
