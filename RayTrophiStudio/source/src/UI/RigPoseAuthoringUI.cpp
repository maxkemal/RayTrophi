#include "UI/RigPoseAuthoringUI.h"
#include "Animation/RigPoseAuthoring.h"
#include "Animation/RigSelection.h"
#include "Api/RtApi.h"
#include "UI/RigIKUI.h"
#include "UI/RigMotionRecipeUI.h"
#include "imgui.h"
#include "scene_ui.h"
#include <cmath>
namespace RigUI {
void drawRigPoseAuthoring(UIContext &ctx) {
    static std::string message;
    static char name[129] = "PoseClip";
    const auto &session = ctx.scene.rigView.pose;
    const auto character = session.active ? session.character : ctx.scene.rigView.character;
    if (!ImGui::CollapsingHeader("Pose / Bone Keys",
                                 session.active ? ImGuiTreeNodeFlags_DefaultOpen : 0))
        return;
    std::string error;
    const bool valid = RigAuthoring::canAuthorPose(ctx.scene, character, error);
    auto report = [&](const rtapi::Result &r) { message = r.ok ? "" : r.error; };
    if (!session.active) {
        ImGui::BeginDisabled(!valid);
        if (ImGui::Button("Enter Pose"))
            report(rtapi::setRigMode("pose", character));
        ImGui::EndDisabled();
        if (!valid)
            ImGui::TextWrapped("%s", error.c_str());
        if (!message.empty())
            ImGui::TextWrapped("%s", message.c_str());
        return;
    }
    if (ImGui::Button("Return to Scene")) {
        report(rtapi::setRigMode("scene"));
        return;
    }
    nlohmann::json state;
    const auto result = rtapi::getRigPoseState(character, state);
    if (!result.ok) {
        report(result);
        ImGui::TextWrapped("%s", message.c_str());
        return;
    }
    ImGui::TextWrapped("Pose: %s | Rotate: R, Move: G | Rest and weights preserved",
                       character.c_str());
    if (ImGui::IsKeyPressed(ImGuiKey_Escape) && state.value("preview", false))
        report(rtapi::cancelRigPosePreview(character));
    ImGui::SetNextItemWidth(190);
    if (ImGui::BeginCombo("Editable clip", state["clip"].get<std::string>().empty()
                                               ? "Create a pose clip"
                                               : state["clip"].get<std::string>().c_str())) {
        for (const auto &c : state["clips"]) {
            const auto label = c["name"].get<std::string>();
            if (ImGui::Selectable(label.c_str(), state["clip"] == label))
                report(rtapi::selectRigPoseClip(character, label));
        }
        ImGui::EndCombo();
    }
    ImGui::SetNextItemWidth(180);
    ImGui::InputText("New clip", name, sizeof(name));
    ImGui::SameLine();
    if (ImGui::Button("Create"))
        report(rtapi::createRigPoseClip(character, name,
                                        static_cast<float>(ctx.render_settings.animation_fps)));
    int frame = ctx.scene.timeline.current_frame;
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputInt("Frame", &frame))
        report(rtapi::setRigPoseFrame(frame));
    bool autoKey = session.autoKey;
    if (ImGui::Checkbox("Auto Key", &autoKey))
        report(rtapi::setRigPoseAutoKey(autoKey));
    if (!state["limit_hits"].empty())
        ImGui::TextDisabled("Joint rule limits this preview.");
    ImGui::TextDisabled(
        "Auto Key writes on gesture release. Unkeyed pose resets when frame changes.");
    const auto selected = RigAuthoring::selectedBones(ctx.scene);
    drawRigIKControls(ctx, character);
    ImGui::BeginDisabled(selected.empty() || state.value("preview", false));
    if (ImGui::Button("Mirror selected pose"))
        report(rtapi::mirrorRigPose(character, selected, state["rig_revision"].get<uint64_t>()));
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Copy paired bones' motion across the rig X axis, relative to each "
                          "side's rest. Select one side of each pair, then Apply or Cancel.");
    if (ctx.scene.rigView.pose.hasPreview) {
        if (ImGui::Button("Apply pose preview"))
            report(rtapi::applyRigPosePreview(character));
        ImGui::SameLine();
        if (ImGui::Button("Cancel##pose"))
            report(rtapi::cancelRigPosePreview(character));
    }
    if (!ctx.scene.rigView.bone.empty() &&
        state["local_transforms"].contains(ctx.scene.rigView.bone)) {
        ImGui::SeparatorText("Selected bone FK");
        const auto bone = ctx.scene.rigView.bone;
        const auto values = state["local_transforms"][bone];
        Matrix4x4 local;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                local.m[r][c] = values[r * 4 + c].get<float>();
        Vec3 p, rotation, scale;
        local.decompose(p, rotation, scale);
        float xyz[3] = {p.x, p.y, p.z}, angles[3] = {rotation.x, rotation.y, rotation.z};
        ImGui::TextWrapped("Active: %s", bone.c_str());
        const bool moved = ImGui::InputFloat3("Local position", xyz);
        const bool moveDone = ImGui::IsItemDeactivatedAfterEdit();
        const bool rotated = ImGui::InputFloat3("Local rotation", angles);
        const bool rotateDone = ImGui::IsItemDeactivatedAfterEdit();
        if (moved || rotated) {
            const float rad = 3.14159265359f / 180.f;
            const auto transform = Matrix4x4::translation(Vec3(xyz[0], xyz[1], xyz[2])) *
                                   Matrix4x4::rotationZ(angles[2] * rad) *
                                   Matrix4x4::rotationY(angles[1] * rad) *
                                   Matrix4x4::rotationX(angles[0] * rad);
            nlohmann::json matrix = nlohmann::json::array();
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    matrix.push_back(transform.m[r][c]);
            report(rtapi::previewRigPoseLocals(character, {{bone, matrix}},
                                               state["rig_revision"].get<uint64_t>()));
        }
        if ((moveDone || rotateDone) && ctx.scene.rigView.pose.hasPreview)
            report(rtapi::applyRigPosePreview(character));
    }
    ImGui::SeparatorText("Bone keys");
    size_t keyedSelected = 0;
    for (const auto& bone : selected)
        for (const auto& keyed : state["keyed_bones_at_frame"])
            if (keyed == bone) {
                ++keyedSelected;
                break;
            }
    ImGui::Text("Current frame: %zu / %zu selected bones keyed", keyedSelected,
                selected.size());
    ImGui::BeginDisabled(state["clip"].get<std::string>().empty());
    ImGui::BeginDisabled(selected.empty());
    if (ImGui::Button("Insert / Update selected (I)"))
        report(rtapi::insertRigPoseKeys(character, selected));
    ImGui::SameLine();
    ImGui::BeginDisabled(keyedSelected == 0);
    if (ImGui::Button("Remove selected key"))
        report(rtapi::removeRigPoseKeys(character, selected));
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    if (ImGui::Button("Key all bones")) {
        std::vector<std::string> all;
        for (const auto &p : state["local_transforms"].items())
            all.push_back(p.key());
        report(rtapi::insertRigPoseKeys(character, all));
    }
    ImGui::EndDisabled();
    ImGui::TextDisabled("Keys store local position and quaternion rotation at this frame.");
    drawRigMotionRecipes(ctx, character, state["rig_revision"].get<uint64_t>());
    if (ImGui::CollapsingHeader("Deformation diagnostics")) {
        if (ImGui::Button("Inspect bind coverage")) {
            nlohmann::json coverage;
            const auto r = rtapi::getRigPoseCoverage(character, coverage);
            message =
                r.ok ? "Vertices: " +
                           std::to_string(coverage["vertex_count"].get<size_t>()) +
                           " | Unweighted: " +
                           std::to_string(coverage["unweighted_vertices"].get<size_t>()) +
                           " (stay at bind position)"
                     : r.error;
        }
        bool map = ctx.scene.rigView.weight_map_visible;
        if (ImGui::Checkbox("Selected bone weight map", &map))
            report(rtapi::setRigWeightMapVisible(map));
    }
    if (!message.empty())
        ImGui::TextWrapped("%s", message.c_str());
}
} // namespace RigUI
