#pragma once
#include "Animation/RigWalkRecipe.h"
#include <cstdint>

namespace rtapi {
struct Result;
Result previewRigHumanWalk(const std::string &character,
                           const RigAuthoring::HumanWalkRecipe &recipe, nlohmann::json &output);
Result createRigHumanWalkClip(const std::string &character, const std::string &name,
                              const RigAuthoring::HumanWalkRecipe &recipe, uint64_t revision);
} // namespace rtapi
