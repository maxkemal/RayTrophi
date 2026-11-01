#pragma once
#include <string>
#include "json.hpp"
namespace rtapi {
struct Result;
Result getRigAnatomy(const std::string& character,nlohmann::json& output);
Result setRigAnatomy(const std::string& character,const nlohmann::json& anatomy);
// Adds missing (or, with overwrite, all) leg foot pivots and limb bend hints; one undo step.
Result deriveRigLimbHints(const std::string& character,bool overwrite,nlohmann::json& report);
}
