#include "UI/RigMotionRecipeUI.h"
#include "Api/RtApi.h"
#include "imgui.h"
#include "scene_ui.h"

namespace RigUI {
void drawRigMotionRecipes(UIContext &ctx, const std::string &character, uint64_t revision) {
    if (!UIWidgets::CollapsingHeader("Quick Motion")) {
        return;
    }
    static char name[129] = "Walk_Loop";
    static RigAuthoring::HumanWalkRecipe recipe;
    static nlohmann::json plan;
    static std::string message;
    static std::string planCharacter;
    static uint64_t planRevision = 0;
    if (planCharacter != character || planRevision != revision) {
        plan = nullptr;
        message.clear();
        planCharacter = character;
        planRevision = revision;
    }
    auto report = [&](const rtapi::Result &result) { message = result.ok ? "" : result.error; };

    ImGui::TextUnformatted("Human Walk (in place)");
    ImGui::InputText("Clip name", name, sizeof(name));
    bool recipeChanged = false;
    recipeChanged |= ImGui::SliderFloat("Cadence (steps/min)", &recipe.cadence, 40, 220, "%.0f");
    recipeChanged |= ImGui::SliderInt("Cycles", &recipe.cycles, 1, 8);
    recipeChanged |= ImGui::SliderFloat("Stride (% height)", &recipe.stride, .05f, .8f, "%.2f");
    recipeChanged |=
        ImGui::SliderFloat("Foot lift (% height)", &recipe.stepHeight, 0, .25f, "%.2f");
    recipeChanged |=
        ImGui::SliderFloat("Body bounce (% height)", &recipe.bodyBounce, 0, .15f, "%.2f");
    recipeChanged |= ImGui::SliderFloat("Arm swing", &recipe.armSwing, 0, 1, "%.2f");
    recipeChanged |= ImGui::SliderFloat("Body motion", &recipe.bodyMotion, 0, 1, "%.2f");
    const float sceneFps = static_cast<float>(ctx.render_settings.animation_fps);
    recipeChanged |= recipe.fps != sceneFps;
    recipe.fps = sceneFps;
    if (recipeChanged) {
        plan = nullptr;
        message.clear();
    }

    if (ImGui::Button("Check recipe")) {
        report(rtapi::previewRigHumanWalk(character, recipe, plan));
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(ctx.scene.rigView.pose.hasPreview);
    if (ImGui::Button("Generate / Update clip")) {
        const auto result = rtapi::createRigHumanWalkClip(character, name, recipe, revision);
        report(result);
        if (result.ok) {
            plan = nullptr;
            message = "Walk clip generated or updated.";
        }
    }
    ImGui::EndDisabled();

    if (plan.is_object()) {
        ImGui::Text("%d frames | %.2f s | stride %.3g scene units", plan["frames"].get<int>(),
                    plan["duration_seconds"].get<double>(), plan["stride_world"].get<double>());
        ImGui::Text("Foot lift %.3g | body bounce %.3g", plan["step_height_world"].get<double>(),
                    plan["body_bounce_world"].get<double>());
    }
    ImGui::TextWrapped("Generates a loopable, in-place bone clip from humanoid anatomy, limb IK "
                       "and joint limits. Play or scrub the new clip, then correct individual "
                       "bones with the existing Pose tools. Path and root motion come next.");
    if (!message.empty()) {
        ImGui::TextWrapped("%s", message.c_str());
    }
}
} // namespace RigUI
