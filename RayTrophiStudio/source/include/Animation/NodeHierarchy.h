/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Animation/NodeHierarchy.h
* Author:        Kemal Demirtas
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*
* THE SCENE NODE TREE, OWNED BY RAYTROPHI.
*
* WHY THIS EXISTS (Assimp import replacement, Faz 0.5)
* ---------------------------------------------------
* Faz 0 got Assimp's TYPES out of the animation keys. This gets Assimp's
* OBJECT GRAPH out of the animation runtime, which turned out to be the bigger
* blocker and the brief had missed it:
*
*     Renderer::updateAnimationWithGraph / updateAnimationState called
*         modelCtx.loader->calculateAnimatedNodeTransformsRecursive(
*             modelCtx.loader->getScene()->mRootNode, ...)
*
* EVERY FRAME. `ImportedModelContext::loader` existed largely to keep the
* aiScene alive for that walk, for the whole session. A model loaded by any
* non-Assimp reader has no aiScene, so it would simply not animate — no error,
* no warning, just a static mesh.
*
* ★ Note what the walk actually needed from aiNode: a name, a local bind
* transform, and children. That is all. The dependency was never on Assimp,
* only on the fact that nobody had written down the three fields.
*
* DESIGN
* ------
* Flat array + child indices, not a pointer tree. Copyable, serialisable, no
* ownership questions, and cache-friendly for the per-frame walk. Node 0 is the
* root. Both names are stored at BUILD time so the walk needs no loader:
*   - `name`       as authored in the file (what animation channels key on
*                  before prefixing)
*   - `uniqueName` import-prefixed (what animationMap and the transform store
*                  are actually keyed by)
* Deriving uniqueName during the walk is what forced a loader pointer into a
* hot path in the first place.
* =========================================================================
*/
#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Matrix4x4.h"

struct AnimationData;   // Animation/AnimationData.h — only the walk's .cpp needs it

namespace RayTrophi {

struct SceneNode {
    std::string name;        // original, as authored in the file
    std::string uniqueName;  // import-prefixed; the key animation data uses
    Matrix4x4   localBind = Matrix4x4::identity();
    int         parent = -1;
    std::vector<int> children;
};

class NodeHierarchy {
public:
    std::vector<SceneNode> nodes;   // nodes[0] is the root when non-empty

    bool   empty()     const { return nodes.empty(); }
    size_t size()      const { return nodes.size(); }
    int    rootIndex() const { return nodes.empty() ? -1 : 0; }

    // Appends a node and links it to `parent` (-1 for the root). Returns its index.
    int addNode(std::string name, std::string uniqueName,
                const Matrix4x4& localBind, int parent) {
        const int index = static_cast<int>(nodes.size());
        SceneNode n;
        n.name = std::move(name);
        n.uniqueName = std::move(uniqueName);
        n.localBind = localBind;
        n.parent = parent;
        nodes.push_back(std::move(n));
        if (parent >= 0 && parent < index) nodes[parent].children.push_back(index);
        return index;
    }

    const SceneNode* find(const std::string& uniqueName) const {
        for (const auto& n : nodes) if (n.uniqueName == uniqueName) return &n;
        return nullptr;
    }
};

// ---------------------------------------------------------------------------
// Per-frame walk: local bind transform (or the animated one, when a channel
// exists for this node) composed down the tree into world-space transforms,
// keyed by uniqueName.
//
// The reader owns identity resolution. Direct glTF imports use the same unique
// node key for geometry, skeleton and animation channels; the legacy Assimp
// reader retains its existing naming policy.
// ---------------------------------------------------------------------------
void computeAnimatedGlobalTransforms(
    const NodeHierarchy& hierarchy,
    const std::map<std::string, std::shared_ptr<AnimationData>>& animationMap,
    float currentTime,
    std::unordered_map<std::string, Matrix4x4>& out);

} // namespace RayTrophi
