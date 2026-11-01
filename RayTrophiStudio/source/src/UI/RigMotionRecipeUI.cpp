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
    recipeChanged |= ImGui::SliderFloat("Leg reach", &recipe.legReach, .85f, 1, "%.3f");
    recipeChanged |=
        ImGui::SliderFloat("Stance knee (deg)", &recipe.stanceKnee, 0, 35, "%.0f");
    recipeChanged |= ImGui::SliderFloat("Heel strike (deg)", &recipe.heelStrike, 0, 30, "%.0f");
    recipeChanged |= ImGui::SliderFloat("Toe off (deg)", &recipe.toeOff, 0, 60, "%.0f");
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
        const auto &metrics = plan["metrics"];
        ImGui::Text("Max leg reach %.3f | foot error %.2f%% height",
                    metrics["max_reach_ratio"].get<double>(),
                    metrics["foot_target_error_fraction"].get<double>() * 100.0);
        ImGui::Text("Pelvis drop %.3g..%.3g | ground penetration %.3g",
                    metrics["pelvis_drop_min"].get<double>(),
                    metrics["pelvis_drop_max"].get<double>(),
                    metrics["ground_penetration_max"].get<double>());
        if (metrics["unreachable_frames"].get<int>() > 0) {
            ImGui::TextWrapped("%d frames cannot reach the foot target; reduce stride.",
                               metrics["unreachable_frames"].get<int>());
        }
    }
    ImGui::TextWrapped("Generates a loopable, in-place bone clip from humanoid anatomy, limb IK "
                       "and joint limits. The supporting knee flexes by Stance knee after contact, which "
                       "sets the pelvis height; Leg reach only caps it. "
                       "and the foot rolls over heel and ball pivots derived from the toe roles. "
                       "Play or scrub the new clip, then correct individual bones with the "
                       "existing Pose tools. Path and root motion come next.");
    if (!message.empty()) {
        ImGui::TextWrapped("%s", message.c_str());
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Range of motion test");
    static char romName[129] = "ROM_Test";
    static RigAuthoring::RomTestSettings rom;
    static nlohmann::json romReport;
    static std::string romMessage;
    static std::string romCharacter;
    static uint64_t romRevision = 0;
    if (romCharacter != character || romRevision != revision) {
        romReport = nullptr;
        romMessage.clear();
        romCharacter = character;
        romRevision = revision;
    }
    ImGui::InputText("ROM clip name", romName, sizeof(romName));
    bool romChanged = ImGui::SliderFloat("Angle scale", &rom.scale, .1f, 1.5f, "%.2f");
    romChanged |= ImGui::SliderInt("Frames per motion", &rom.framesPerMotion, 4, 120);
    rom.fps = sceneFps;
    if (romChanged) {
        romReport = nullptr;
    }
    if (ImGui::Button("Check deformation")) {
        const auto result = rtapi::previewRigRomTest(character, rom, romReport);
        romMessage = result.ok ? "" : result.error;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(ctx.scene.rigView.pose.hasPreview);
    if (ImGui::Button("Generate ROM clip")) {
        const auto result = rtapi::createRigRomClip(character, romName, rom, revision);
        romMessage = result.ok ? "ROM clip generated; scrub it to see each motion." : result.error;
    }
    ImGui::EndDisabled();
    if (romReport.is_object()) {
        const auto &summary = romReport["summary"];
        ImGui::Text("%d motions | min area ratio %.2f", romReport["motions"].get<int>(),
                    summary["min_area_ratio"].get<double>());
        ImGui::Text("Pinched %d | stretched %d | folded %d | limited %d",
                    summary["pinched_motions"].get<int>(), summary["stretched_motions"].get<int>(),
                    summary["folded_motions"].get<int>(), summary["limited_motions"].get<int>());
        if (ImGui::BeginTable("RomWorst", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Motion");
            ImGui::TableSetupColumn("Min");
            ImGui::TableSetupColumn("Folds");
            ImGui::TableSetupColumn("Worst bone");
            ImGui::TableHeadersRow();
            for (const auto &row : romReport["worst"]) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(row["motion"].get<std::string>().c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", row["min_area_ratio"].get<double>());
                ImGui::TableNextColumn();
                ImGui::Text("%d", row["folded_triangles"].get<int>());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(row["worst"]["bone"].get<std::string>().c_str());
            }
            ImGui::EndTable();
        }
    }
    ImGui::TextWrapped("Bends every limb and spine joint and measures the bound skin at each "
                       "peak: area ratio below .5 is a pinch, above 2 a stretch, a turned "
                       "normal a fold. The clip plays the same motions for a visual check.");
    if (!romMessage.empty()) {
        ImGui::TextWrapped("%s", romMessage.c_str());
    }
}
} // namespace RigUI
