#pragma once
#include "json.hpp"
#include <string>
namespace rtapi {
struct Result;
Result setRigWeightMapVisible(bool visible);
Result getRigWeightMapVisible(bool& visible);
Result getRigWeightMap(const std::string& mesh,const std::string& character,const std::string& bone,nlohmann::json& output);
}
