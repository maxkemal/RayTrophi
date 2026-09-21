#pragma once
#include <string>
#include "json.hpp"
namespace rtapi {
struct Result;
Result getRigAnatomy(const std::string& character,nlohmann::json& output);
Result setRigAnatomy(const std::string& character,const nlohmann::json& anatomy);
}
