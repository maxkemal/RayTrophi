#include "UI/RigStudioUI_MotionStrip.h"
#include "Animation/RigStudioServices.h"
#include "Animation/AnimationGraphComposition.h"
#include "Animation/RigWalkRecipe.h"
#include "Api/RtApi.h"
#include "Api/RtApiRigMotion.h"
#include "scene_ui.h"
#include "imgui.h"
#include <unordered_map>
#include <string>

namespace RayTrophi {

static char s_selectedRecipe[64] = "Walk";
static float s_speed = 1.0f;
static float s_stride = 1.0f;
static float s_drunkWeight = 0.0f;
static float s_tiredWeight = 0.0f;
static float s_sneakWeight = 0.0f;
static float s_limpWeight = 0.0f;
static std::string s_statusMessage;

void RigStudioUI_MotionStrip::draw(UIContext& ctx) {
    auto& studioCtx = RigStudioServices::getContext(ctx.scene);
    if (studioCtx.characterId.empty()) {
        ImGui::TextDisabled("Select a character from the header above to configure Motion Strip.");
        return;
    }

    ImGui::TextDisabled("MOTION TILES:");
    ImGui::SameLine();

    const char* recipes[] = { "Idle", "Walk", "Run", "Sit", "Lie Down", "Jump", "Turn" };
    for (int i = 0; i < 7; ++i) {
        bool selected = (std::string(s_selectedRecipe) == recipes[i]);
        if (selected) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.65f, 0.45f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.72f, 0.50f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.20f, 0.25f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.28f, 0.35f, 1.0f));
        }

        if (ImGui::Button(recipes[i], ImVec2(72, 28))) {
            std::snprintf(s_selectedRecipe, sizeof(s_selectedRecipe), "%s", recipes[i]);
        }
        ImGui::PopStyleColor(2);

        if (i < 6) ImGui::SameLine();
    }

    ImGui::Spacing();

    // Two Columns for Parameters
    ImGui::Columns(2, "MotionParamsCol", true);
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.48f);

    ImGui::Text("Base Locomotion (%s)", s_selectedRecipe);
    ImGui::Separator();
    ImGui::SliderFloat("Speed Multiplier", &s_speed, 0.2f, 3.0f, "%.2fx");
    ImGui::SliderFloat("Stride Scale", &s_stride, 0.2f, 2.0f, "%.2fx");

    ImGui::NextColumn();

    ImGui::Text("Expressive Recipe Modifiers");
    ImGui::Separator();
    ImGui::SliderFloat("Drunk", &s_drunkWeight, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Tired", &s_tiredWeight, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Sneak", &s_sneakWeight, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Limp", &s_limpWeight, 0.0f, 1.0f, "%.2f");

    ImGui::Columns(1);
    ImGui::Spacing();
    ImGui::Separator();

    // Action buttons
    if (ImGui::Button(" CREATE & APPLY MOTION ", ImVec2(220, 32))) {
        std::unordered_map<std::string, float> mods;
        if (s_drunkWeight > 0.0f) mods["Drunk"] = s_drunkWeight;
        if (s_tiredWeight > 0.0f) mods["Tired"] = s_tiredWeight;
        if (s_sneakWeight > 0.0f) mods["Sneak"] = s_sneakWeight;
        if (s_limpWeight > 0.0f) mods["Limp"] = s_limpWeight;

        // Apply to Composition Graph
        AnimationGraphComposition::applyMotionRecipeToGraph(
            ctx.scene, studioCtx.characterId, s_selectedRecipe, mods, s_speed
        );

        // Build HumanWalkRecipe
        RigAuthoring::HumanWalkRecipe walkRecipe;
        walkRecipe.fps = static_cast<float>(ctx.render_settings.animation_fps);
        const std::string recStr = s_selectedRecipe;
        if (recStr == "Run") {
            walkRecipe.cadence = 160.0f * s_speed;
            walkRecipe.stride = 0.55f * s_stride;
            walkRecipe.armSwing = 0.9f;
        } else if (recStr == "Idle") {
            walkRecipe.cadence = 40.0f * s_speed;
            walkRecipe.stride = 0.05f * s_stride;
            walkRecipe.armSwing = 0.2f;
        } else { // Walk
            walkRecipe.cadence = 100.0f * s_speed;
            walkRecipe.stride = 0.35f * s_stride;
            walkRecipe.armSwing = 0.7f;
        }

        // Apply Modifiers to Recipe
        if (s_drunkWeight > 0.0f) {
            walkRecipe.bodyMotion = 0.95f;
            walkRecipe.bodyBounce = 0.03f + s_drunkWeight * 0.05f;
        }
        if (s_tiredWeight > 0.0f) {
            walkRecipe.armSwing = 0.2f;
            walkRecipe.cadence *= 0.7f;
        }
        if (s_sneakWeight > 0.0f) {
            walkRecipe.stepHeight = 0.12f;
            walkRecipe.stride *= 0.7f;
        }
        if (s_limpWeight > 0.0f) {
            walkRecipe.bodyBounce = 0.07f;
        }

        const std::string clipName = studioCtx.characterId + "_" + s_selectedRecipe + "_Clip";
        rtapi::Result res = rtapi::createRigHumanWalkClip(studioCtx.characterId, clipName, walkRecipe, 0);
        if (res.ok) {
            s_statusMessage = "Successfully generated motion clip: " + clipName;
            ctx.start_render = true;
        } else {
            s_statusMessage = "Error generating motion clip: " + res.error;
        }
    }

    ImGui::SameLine();
    if (ImGui::Button(" Open Graph Node Editor ", ImVec2(200, 32))) {
        studioCtx.activeAnimGraphId = studioCtx.characterId;
        studioCtx.requestedTab = 4; // Switch to Anim Graph tab
    }

    if (!s_statusMessage.empty()) {
        ImGui::Spacing();
        if (s_statusMessage.rfind("Successfully", 0) == 0) {
            ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.0f), "%s", s_statusMessage.c_str());
        } else {
            ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.4f, 1.0f), "%s", s_statusMessage.c_str());
        }
    }
}

} // namespace RayTrophi

