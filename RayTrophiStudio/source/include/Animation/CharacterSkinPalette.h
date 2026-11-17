#pragma once

#include "Animation/AnimationData.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace Animation {

// Publish evaluated hierarchy globals into the scene's canonical bone slots.
// Both graph and controller use this boundary before the shared skin service.
inline bool publishCharacterSkinPalette(
    const BoneData& bones,
    const std::string& character,
    const std::unordered_map<std::string, Matrix4x4>& globals,
    std::vector<Matrix4x4>& palette) {
    const std::string prefix = character + "_";
    Matrix4x4 inverse = bones.globalInverseTransform;
    const auto modelInverse = bones.perModelInverses.find(character);
    if (modelInverse != bones.perModelInverses.end()) {
        inverse = modelInverse->second;
    }

    // Validate before publishing: no half-written weighted skeleton.
    for (const auto& entry : bones.boneNameToIndex) {
        if (entry.first.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        if (bones.weightedBoneNames.count(entry.first) &&
            (globals.find(entry.first) == globals.end() ||
             bones.boneOffsetMatrices.find(entry.first) == bones.boneOffsetMatrices.end())) {
            return false;
        }
    }

    if (palette.size() < bones.getBoneIndexCapacity()) {
        palette.resize(bones.getBoneIndexCapacity(), Matrix4x4::identity());
    }
    for (const auto& entry : bones.boneNameToIndex) {
        if (entry.first.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        const auto world = globals.find(entry.first);
        if (world == globals.end()) {
            continue;
        }
        const auto offset = bones.boneOffsetMatrices.find(entry.first);
        const Matrix4x4 bindInverse = offset == bones.boneOffsetMatrices.end()
            ? Matrix4x4::identity() : offset->second;
        palette[entry.second] = inverse * world->second * bindInverse;
    }
    return true;
}

} // namespace Animation
