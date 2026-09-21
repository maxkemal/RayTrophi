#pragma once

#include "../../../external/ufbx/ufbx.h"
#include <unordered_set>

namespace rtimport {

// FBX joints exist independently of skin deformers and animation tracks.
// Ancestors carry axis/unit conversions but are not all deform joints.
inline void collectExplicitFbxJoints(
        const ufbx_scene& scene,
        std::unordered_set<const ufbx_node*>& technical,
        std::unordered_set<const ufbx_node*>& indexed) {
    for (const ufbx_node* node : scene.nodes) {
        if (!node || !node->bone) continue;
        indexed.insert(node);
        for (const ufbx_node* ancestor = node; ancestor; ancestor = ancestor->parent) {
            // Existing closure already contains this ancestor and its parents.
            if (!technical.insert(ancestor).second) break;
        }
    }
}

} // namespace rtimport
