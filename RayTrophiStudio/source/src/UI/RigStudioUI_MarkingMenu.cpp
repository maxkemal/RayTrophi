#include "UI/RigStudioUI_MarkingMenu.h"
#include "Animation/RigStudioServices.h"
#include "Api/RtApi.h"
#include "Api/RtApiRigPoseAuthoring.h"
#include "Api/RtApiRigIK.h"
#include "scene_ui.h"
#include "imgui.h"

namespace RayTrophi {

void RigStudioUI_MarkingMenu::draw(UIContext& ctx) {
    auto& studioCtx = RigStudioServices::getContext(ctx.scene);
    if (studioCtx.activeBoneId.empty()) {
        return;
    }

    if (ImGui::BeginPopupContextItem("RigStudioMarkingMenu")) {
        ImGui::Text("Bone: %s", studioCtx.activeBoneId.c_str());
        ImGui::Separator();

        if (ImGui::MenuItem("KEYFRAME", "K")) {
            std::vector<std::string> bones = { studioCtx.activeBoneId };
            rtapi::insertRigPoseKeys(studioCtx.characterId, bones);
        }
        if (ImGui::MenuItem("RESET REST")) {
            rtapi::cancelRigPosePreview(studioCtx.characterId);
        }
        if (ImGui::MenuItem("MIRROR POSE")) {
            std::vector<std::string> bones = { studioCtx.activeBoneId };
            rtapi::mirrorRigPose(studioCtx.characterId, bones, 0, "selected", "x");
        }
        if (ImGui::MenuItem("SNAP FK <-> IK")) {
            rtapi::setRigIKFK(studioCtx.characterId, studioCtx.activeBoneId, 1.0f, 0);
        }

        ImGui::EndPopup();
    }
}

} // namespace RayTrophi
