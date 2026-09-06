/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          Animation/NodeHierarchy.cpp
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*
* The per-frame node walk, moved off aiNode. See Animation/NodeHierarchy.h for
* why it had to move.
* =========================================================================
*/
#include "Animation/NodeHierarchy.h"

#include "Animation/AnimationData.h"   // AnimationData::calculateAnimationTransform

namespace RayTrophi {

namespace {

void walk(const NodeHierarchy& hierarchy,
          int index,
          const Matrix4x4& parentGlobal,
          const std::map<std::string, std::shared_ptr<AnimationData>>& animationMap,
          float currentTime,
          std::unordered_map<std::string, Matrix4x4>& out) {
    if (index < 0 || index >= static_cast<int>(hierarchy.nodes.size())) return;
    const SceneNode& node = hierarchy.nodes[static_cast<size_t>(index)];

    // Default: the node's static (bind pose) local transform.
    Matrix4x4 local = node.localBind;

    // animationMap is keyed by the PREFIXED name; uniqueName was resolved when
    // the hierarchy was built, precisely so this hot loop needs no loader.
    auto it = animationMap.find(node.uniqueName);
    if (it != animationMap.end() && it->second) {
        const AnimationData& anim = *it->second;
        local = anim.calculateAnimationTransform(anim, currentTime, node.uniqueName, local);
    }

    const Matrix4x4 global = parentGlobal * local;
    out[node.uniqueName] = global;

    for (int child : node.children) {
        walk(hierarchy, child, global, animationMap, currentTime, out);
    }
}

} // namespace

void computeAnimatedGlobalTransforms(
    const NodeHierarchy& hierarchy,
    const std::map<std::string, std::shared_ptr<AnimationData>>& animationMap,
    float currentTime,
    std::unordered_map<std::string, Matrix4x4>& out) {
    if (hierarchy.empty()) return;
    walk(hierarchy, hierarchy.rootIndex(), Matrix4x4::identity(), animationMap, currentTime, out);
}

} // namespace RayTrophi
