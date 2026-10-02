#include "UI/RigStudioUI_Header.h"
#include "Animation/RigStudioServices.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "imgui.h"

namespace RayTrophi {

void RigStudioUI_Header::draw(UIContext& ctx) {
    auto& studioCtx = RigStudioServices::getContext(ctx.scene);

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6, 4));

    // --- ROW 1: Character Selector & Quick Toggles ---
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("Character:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(170.0f);
    if (ImGui::BeginCombo("##CharacterSelectHeader", studioCtx.characterId.empty() ? "-- Select Character --" : studioCtx.characterId.c_str())) {
        for (const auto& model : ctx.scene.importedModelContexts) {
            if (ImGui::Selectable(model.importName.c_str(), studioCtx.characterId == model.importName)) {
                RigStudioServices::setActiveCharacter(ctx.scene, model.importName);
            }
        }
        ImGui::EndCombo();
    }

    if (!studioCtx.characterId.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.4f, 0.85f, 0.55f, 1.0f), "[ Active: %s ]", studioCtx.characterId.c_str());
    } else {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.9f, 0.55f, 0.3f, 1.0f), "(No Character Selected)");
    }

    // Right-aligned quick toggles
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 210.0f);
    
    bool showBones = studioCtx.showBones;
    if (ImGui::Checkbox("Skeleton Overlay", &showBones)) {
        studioCtx.showBones = showBones;
        rtapi::setRigOverlayVisible(showBones);
    }
    ImGui::SameLine();
    bool autoKey = studioCtx.autoKey;
    if (ImGui::Checkbox("AutoKey", &autoKey)) {
        studioCtx.autoKey = autoKey;
    }

    ImGui::Separator();

    // --- ROW 2: Mode Segmented Buttons Synchronized with Workspace Tabs ---
    const RigStudioMode currentMode = studioCtx.mode;
    
    auto drawModeButton = [&](const char* iconLabel, RigStudioMode targetMode, int targetTab) {
        bool active = (currentMode == targetMode);
        if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.52f, 0.82f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.60f, 0.92f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.22f, 0.27f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.32f, 0.38f, 1.0f));
        }

        if (ImGui::Button(iconLabel, ImVec2(105, 25))) {
            RigStudioServices::setMode(ctx.scene, targetMode);
            studioCtx.requestedTab = targetTab;
        }

        ImGui::PopStyleColor(2);
    };

    drawModeButton("REST", RigStudioMode::Rest, 1);
    ImGui::SameLine();
    drawModeButton("POSE", RigStudioMode::Pose, 2);
    ImGui::SameLine();
    drawModeButton("ANIMATE", RigStudioMode::Animate, 0);
    ImGui::SameLine();
    drawModeButton("SKIN", RigStudioMode::Skin, 3);

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    // Manipulation Mode Segmented Buttons: FK | IK | Aim
    const ManipulationMode manipMode = studioCtx.manipulationMode;
    auto drawManipButton = [&](const char* label, ManipulationMode modeVal) {
        bool active = (manipMode == modeVal);
        if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.60f, 0.40f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.30f, 0.68f, 0.46f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.20f, 0.24f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.28f, 0.34f, 1.0f));
        }
        if (ImGui::Button(label, ImVec2(45, 25))) {
            RigStudioServices::setManipulationMode(ctx.scene, modeVal);
        }
        ImGui::PopStyleColor(2);
    };

    drawManipButton("FK", ManipulationMode::FK);
    ImGui::SameLine();
    drawManipButton("IK", ManipulationMode::IK);
    ImGui::SameLine();
    drawManipButton("Aim", ManipulationMode::Aim);

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    bool mirror = studioCtx.mirrorMode;
    if (ImGui::Checkbox("Mirror X", &mirror)) {
        studioCtx.mirrorMode = mirror;
    }

    ImGui::PopStyleVar(2);
}

} // namespace RayTrophi

