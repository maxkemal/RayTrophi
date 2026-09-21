#pragma once
#include "json.hpp"
#include <string>
#include <vector>
namespace rtapi {
struct Result;
Result selectRigBones(const std::string& character,const std::vector<std::string>& bones,const std::string& active="",const std::string& mode="replace",const std::string& anchor="");
Result getRigSelection(nlohmann::json& output);
Result setRigSelectionPivot(const std::string& mode);
}
