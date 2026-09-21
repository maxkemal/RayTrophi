#pragma once
#include "json.hpp"
#include <string>
#include <cstdint>
namespace rtapi {
struct Result;
Result getRigJointLimitView(const std::string& character,const std::string& bone,nlohmann::json& output);
Result setRigJointLimits(const std::string& character,const std::string& bone,float minimum,float maximum,float swing,uint64_t revision);
Result getRigJointLimitOverlay(nlohmann::json& output);
Result setRigJointLimitOverlay(bool visible,bool edit);
}
