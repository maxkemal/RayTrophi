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
    struct PendingBoneWrite {
        size_t slot;
        const Matrix4x4* world;
        const Matrix4x4* bindInverse;
    };

    const std::string prefix = character + "_";
    Matrix4x4 inverse = bones.globalInverseTransform;
    const auto modelInverse = bones.perModelInverses.find(character);
    if (modelInverse != bones.perModelInverses.end()) {
        inverse = modelInverse->second;
    }

    // Single pass: validate all weighted bones and gather resolved pointers so
    // we never half-write a skeleton and avoid duplicate map/hash lookups.
    thread_local std::vector<PendingBoneWrite> pending;
    pending.clear();
    if (pending.capacity() < globals.size()) {
        pending.reserve(globals.size());
    }

    for (const auto& entry : bones.boneNameToIndex) {
        if (entry.first.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        const auto world = globals.find(entry.first);
        const auto offset = bones.boneOffsetMatrices.find(entry.first);
        const bool hasWorld = (world != globals.end());
        const bool hasOffset = (offset != bones.boneOffsetMatrices.end());

        if ((!hasWorld || !hasOffset) && bones.weightedBoneNames.count(entry.first)) {
            return false;
        }
        if (!hasWorld) {
            continue;
        }
        pending.push_back({
            static_cast<size_t>(entry.second),
            &world->second,
            hasOffset ? &offset->second : nullptr
        });
    }

    if (palette.size() < bones.getBoneIndexCapacity()) {
        palette.resize(bones.getBoneIndexCapacity(), Matrix4x4::identity());
    }
    for (const auto& item : pending) {
        if (item.slot >= palette.size()) {
            continue;
        }
        if (item.bindInverse) {
            palette[item.slot] = inverse * (*item.world) * (*item.bindInverse);
        } else {
            palette[item.slot] = inverse * (*item.world);
        }
    }
    return true;
}

// Fast O(totalBones) contiguous scatter for Ozz runtime joints, avoiding
// per-bone unordered_map iteration and string prefix allocations.
inline void publishOzzModelPalette(
    const std::vector<int>& sceneBoneToRuntimeJoint,
    const std::vector<Matrix4x4>& modelBoneMatrices,
    std::vector<Matrix4x4>& palette) {
    const size_t limit = (palette.size() < sceneBoneToRuntimeJoint.size())
        ? palette.size()
        : sceneBoneToRuntimeJoint.size();
    const size_t runtimeCount = modelBoneMatrices.size();
    for (size_t globalIdx = 0; globalIdx < limit; ++globalIdx) {
        const int runtimeJointIndex = sceneBoneToRuntimeJoint[globalIdx];
        if (runtimeJointIndex < 0 ||
            static_cast<size_t>(runtimeJointIndex) >= runtimeCount) {
            continue;
        }
        palette[globalIdx] = modelBoneMatrices[static_cast<size_t>(runtimeJointIndex)];
    }
}

} // namespace Animation
