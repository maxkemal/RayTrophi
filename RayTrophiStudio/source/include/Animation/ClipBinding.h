#pragma once
#include "Animation/AnimationData.h"
#include "Animation/NodeHierarchy.h"
#include <memory>
#include <map>

namespace RigAuthoring {
struct ClipNodeMatch { std::string source, target, authored_name; };
struct ClipBindingReport {
    bool ready = false;
    std::string mode = "same_rig";
    float translation_scale = 1.f;
    std::string source_character, source_clip, target_character, output_clip;
    std::vector<ClipNodeMatch> matches;
    std::vector<std::string> unmapped, ambiguous, hierarchy_mismatches;
    int rest_difference_count = 0;
};
// same_rig copies local TRS; rest_basis transports rest-frame motion deltas.
// Both require matching mapped parent chains; neither solves IK/contact.
bool buildSameRigClip(const AnimationData&, const RayTrophi::NodeHierarchy& source,
                      const RayTrophi::NodeHierarchy& target, const std::string& targetCharacter,
                      const std::string& outputName, ClipBindingReport&,
                      std::shared_ptr<AnimationData>& output, std::string& error,
                      const std::map<std::string, std::string>& nodeMap = {},
                      const std::string& mode = "same_rig", float translationScale = 1.f);
}
