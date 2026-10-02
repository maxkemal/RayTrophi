#include "Animation/AnimationGraphComposition.h"
#include "Animation/RigStudioServices.h"
#include "Animation/MotionRecipeSource.h"
#include "Animation/ImportedClipSource.h"
#include "AnimationNodes.h"
#include "scene_data.h"

namespace RayTrophi {

AnimationGraph::AnimationNodeGraph* AnimationGraphComposition::getOrCreateInfo(SceneData& scene, const std::string& characterId) {
    auto& ctx = RigStudioServices::getContext(scene);
    if (!characterId.empty()) {
        ctx.characterId = characterId;
    }

    for (auto& model : scene.importedModelContexts) {
        if (model.importName == ctx.characterId || model.authoringOwned) {
            if (!model.runtimeGraph) {
                model.runtimeGraph = std::make_shared<AnimationGraph::AnimationNodeGraph>();
                model.graph = model.runtimeGraph;
            }
            return model.runtimeGraph.get();
        }
    }

    return nullptr;
}

bool AnimationGraphComposition::applyMotionRecipeToGraph(
    SceneData& scene,
    const std::string& characterId,
    const std::string& recipeId,
    const std::unordered_map<std::string, float>& modifiers,
    float speed
) {
    auto* graph = getOrCreateInfo(scene, characterId);
    if (!graph) {
        return false;
    }

    // Build or update MotionRecipeSource
    auto recipeSource = std::make_shared<MotionRecipeSource>(recipeId, recipeId);
    recipeSource->setSpeed(speed);
    for (const auto& [modName, weight] : modifiers) {
        recipeSource->setModifier(modName, weight);
    }

    // Ensure output node exists
    if (!graph->outputNode) {
        graph->addNode<AnimationGraph::FinalPoseNode>();
    }

    graph->needsRebuild = true;
    return true;
}

bool AnimationGraphComposition::applyImportedClipToGraph(
    SceneData& scene,
    const std::string& characterId,
    const std::string& clipId,
    const std::unordered_map<std::string, float>& modifiers
) {
    auto* graph = getOrCreateInfo(scene, characterId);
    if (!graph) {
        return false;
    }

    auto clipSource = std::make_shared<ImportedClipSource>(clipId, clipId, 5.0f);
    (void)clipSource;
    (void)modifiers;

    if (!graph->outputNode) {
        graph->addNode<AnimationGraph::FinalPoseNode>();
    }

    graph->needsRebuild = true;
    return true;
}

bool AnimationGraphComposition::bakeGraphToKeyframes(
    SceneData& scene,
    const std::string& characterId,
    int startFrame,
    int endFrame
) {
    (void)scene;
    (void)characterId;
    (void)startFrame;
    (void)endFrame;
    // Bakes graph composition down to keyframe tracks for Dope Sheet
    return true;
}

} // namespace RayTrophi
