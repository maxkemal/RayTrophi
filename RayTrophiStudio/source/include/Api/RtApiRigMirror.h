#pragma once
#include "json.hpp"
#include <string>
#include <vector>
#include <cstdint>
namespace rtapi {
struct Result;
Result mirrorRigRest(const std::string& character,const std::vector<std::string>& bones,const std::string& direction,const std::string& axis,float offset,uint64_t revision);
Result createMirroredRigBone(const std::string& character,const std::string& bone,const std::string& name,const std::string& sourceSide,const std::string& axis,float offset,uint64_t revision);
Result mirrorRigLandmarks(const std::string& character,const nlohmann::json& landmarks,const std::vector<std::string>& bones,const std::string& direction,const std::string& axis,float offset,uint64_t revision,nlohmann::json& output);
}
