#pragma once
#include "Animation/AnimationSource.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

struct SceneData;

namespace AnimationGraph {
    class AnimationNodeGraph;
}

namespace RayTrophi {

class AnimationGraphComposition {
public:
    static AnimationGraph::AnimationNodeGraph* getOrCreateInfo(SceneData& scene, const std::string& characterId);

    // High-Level Graph Assembly (Level 2 Semantic -> Level 3 Composition Graph)
    static bool applyMotionRecipeToGraph(
        SceneData& scene,
        const std::string& characterId,
        const std::string& recipeId,
        const std::unordered_map<std::string, float>& modifiers,
        float speed = 1.0f
    );

    static bool applyImportedClipToGraph(
        SceneData& scene,
        const std::string& characterId,
        const std::string& clipId,
        const std::unordered_map<std::string, float>& modifiers
    );

    static bool bakeGraphToKeyframes(
        SceneData& scene,
        const std::string& characterId,
        int startFrame,
        int endFrame
    );
};

} // namespace RayTrophi
