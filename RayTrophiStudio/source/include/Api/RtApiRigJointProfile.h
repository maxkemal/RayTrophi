#pragma once
#include "json.hpp"
#include "Api/RtApiRigJointLimits.h"
#include <string>
#include <cstdint>
namespace rtapi {
struct Result;
Result getRigJointProfile(const std::string& character,nlohmann::json& output);
Result suggestRigJointProfile(const std::string& character,nlohmann::json& output);
Result setRigJointProfile(const std::string& character,const nlohmann::json& profile,uint64_t revision);
}
