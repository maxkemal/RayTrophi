#pragma once

#include "Animation/NodeHierarchy.h"
#include "Animation/RigAnatomy.h"
#include <map>
#include <string>
#include <vector>

namespace RigAuthoring {

bool evaluateRigDrivenControls(const RayTrophi::NodeHierarchy& input,
                               const std::vector<RigDrivenControl>& definitions,
                               const std::map<std::string, float>& values,
                               RayTrophi::NodeHierarchy& output,
                               std::vector<std::string>& affectedBones,
                               std::string& error);
nlohmann::json serializeRigDrivenControls(const std::vector<RigDrivenControl>& controls);

} // namespace RigAuthoring
