#pragma once
#include "Animation/RigRomTest.h"
#include "Animation/RigWalkRecipe.h"
#include <cstdint>

namespace rtapi {
struct Result;
Result previewRigHumanWalk(const std::string &character,
                           const RigAuthoring::HumanWalkRecipe &recipe, nlohmann::json &output);
Result createRigHumanWalkClip(const std::string &character, const std::string &name,
                              const RigAuthoring::HumanWalkRecipe &recipe, uint64_t revision);
// Range-of-motion deformation report for the authored binding; read-only, any mode.
Result previewRigRomTest(const std::string &character, const RigAuthoring::RomTestSettings &settings,
                         nlohmann::json &output);
// Same motions baked into an editable rig clip (Pose mode, matching revision, one undo step).
Result createRigRomClip(const std::string &character, const std::string &name,
                        const RigAuthoring::RomTestSettings &settings, uint64_t revision);
} // namespace rtapi
