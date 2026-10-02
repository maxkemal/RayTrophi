#include "Animation/RigStudioServices.h"
#include "Animation/RigView.h"
#include "scene_data.h"

namespace RayTrophi {

static RigStudioContext s_globalContext;

RigStudioContext& RigStudioServices::getContext(SceneData& scene) {
    syncContextWithRigView(scene);
    return s_globalContext;
}

void RigStudioServices::syncContextWithRigView(SceneData& scene) {
    const auto& view = scene.rigView;
    if (view.edit_mode) {
        s_globalContext.characterId = view.edit_character;
        s_globalContext.mode = RigStudioMode::Rest;
    } else if (view.pose.active) {
        s_globalContext.characterId = view.pose.character;
        s_globalContext.mode = RigStudioMode::Pose;
    } else {
        if (!view.character.empty()) {
            s_globalContext.characterId = view.character;
        }
    }

    s_globalContext.rigId = s_globalContext.characterId;
    s_globalContext.selectedBoneIds = view.selected_bones;
    s_globalContext.activeBoneId = view.bone;
    s_globalContext.showBones = view.visible;
    s_globalContext.showEnvelopes = view.envelope_overlay_visible;
    s_globalContext.showWeightHeatmap = view.weight_map_visible;
}

void RigStudioServices::setMode(SceneData& scene, RigStudioMode mode) {
    s_globalContext.mode = mode;
    switch (mode) {
    case RigStudioMode::Rest:
        scene.rigView.edit_mode = true;
        scene.rigView.pose.active = false;
        break;
    case RigStudioMode::Pose:
        scene.rigView.edit_mode = false;
        scene.rigView.pose.active = true;
        break;
    case RigStudioMode::Animate:
        scene.rigView.edit_mode = false;
        break;
    case RigStudioMode::Skin:
        scene.rigView.edit_mode = false;
        scene.rigView.weight_map_visible = true;
        break;
    }
}

void RigStudioServices::setManipulationMode(SceneData& scene, ManipulationMode mode) {
    (void)scene;
    s_globalContext.manipulationMode = mode;
}

void RigStudioServices::setActiveCharacter(SceneData& scene, const std::string& characterId) {
    s_globalContext.characterId = characterId;
    s_globalContext.rigId = characterId;
    scene.rigView.character = characterId;
    if (scene.rigView.edit_mode) {
        scene.rigView.edit_character = characterId;
    }
    if (scene.rigView.pose.active) {
        scene.rigView.pose.character = characterId;
    }
}

} // namespace RayTrophi
