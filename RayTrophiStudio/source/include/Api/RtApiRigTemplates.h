#pragma once
#include <string>
#include <cstdint>
#include "json.hpp"
namespace rtapi {
struct Result;
Result listRigFitTargets(nlohmann::json& output);
Result getRigFitSetup(const std::string& character,const std::string& mesh,nlohmann::json& output);
Result previewRigFit(const std::string& character,const std::string& mesh,const nlohmann::json& landmarks,bool axesConfirmed,nlohmann::json& output);
Result commitRigFit(const std::string& character,const std::string& mesh,const nlohmann::json& preview);
Result getRigVertexWeights(const std::string& object,uint64_t vertex,nlohmann::json& report);
Result getRigWeightStats(const std::string& mesh,nlohmann::json& report);
Result preflightRigMesh(const std::string& mesh,nlohmann::json& report);
Result listRigTemplates(nlohmann::json& output);
Result getRigTemplate(const std::string& id,float height,nlohmann::json& output);
}
