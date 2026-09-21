#include "Animation/RigClipRuntimeSync.h"

#include "AnimationController.h"
#include "OzzRuntime.h"
#include "scene_data.h"

#include <memory>
#include <vector>

namespace RigAuthoring {

bool synchronizeClipRuntime(SceneData& scene, const std::string& character) {
    SceneData::ImportedModelContext* model = nullptr;
    for (auto& candidate : scene.importedModelContexts) {
        if (candidate.importName == character) {
            model = &candidate;
            break;
        }
    }
    if (!model) {
        return false;
    }

    std::vector<std::shared_ptr<AnimationData>> clips;
    for (const auto& clip : scene.animationDataList) {
        if (clip && clip->modelName == character) {
            clips.push_back(clip);
        }
    }

    if (!model->animator) {
        model->animator = std::make_shared<AnimationController>();
    }
    model->animator->registerClips(clips);
    model->ozzAnimationSet = OzzRuntime::buildStubAnimationSet(
        character, scene.boneData, clips);
    model->hasAnimation = !clips.empty();
    model->restPoseApplied = false;
    model->rigJointGlobals.clear();

    // A few non-character animation tools still consult the scene controller.
    // Keep it pointed at the same canonical snapshots as the per-model runtime.
    AnimationController::getInstance().registerClips(scene.animationDataList);
    return true;
}

} // namespace RigAuthoring
