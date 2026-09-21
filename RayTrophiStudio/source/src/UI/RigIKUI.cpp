#include "UI/RigIKUI.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "imgui.h"
namespace RigUI {
void drawRigIKControls(UIContext& ctx, const std::string& character) {
    if (!ImGui::CollapsingHeader("IK Controls / Contacts", ImGuiTreeNodeFlags_DefaultOpen))
        return;
    static std::string message;
    auto report = [&](const rtapi::Result& r) { message = r.ok ? "" : r.error; };
    nlohmann::json data;
    auto result = rtapi::getRigControls(character, data);
    if (!result.ok) {
        ImGui::TextWrapped("%s", result.error.c_str());
        return;
    }
    const uint64_t revision = data["rig_revision"].get<uint64_t>();
    ImGui::BeginDisabled(ctx.scene.rigView.pose.hasPreview);
    if (ImGui::Button("Create limb controls"))
        report(rtapi::createRigControls(character, nullptr, revision));
    ImGui::SameLine();
    if (ImGui::Button("Bone FK gizmo"))
        report(rtapi::selectRigControl(character, ""));
    nlohmann::json anatomy;
    const bool anatomyReady = rtapi::getRigAnatomy(character, anatomy).ok;
    if (ImGui::BeginCombo("Add chain IK", "Choose anatomy chain")) {
        bool eligible = false;
        if (anatomyReady) {
            for (const auto& chain : anatomy["chains"]) {
                if (chain["bones"].size() < 4 || chain["bones"].size() > 64)
                    continue;
                eligible = true;
                const auto name = chain["name"].get<std::string>();
                bool exists = false;
                for (const auto& row : data["controls"])
                    if (row["name"] == name)
                        exists = true;
                if (ImGui::Selectable(name.c_str(), false,
                                      exists ? ImGuiSelectableFlags_Disabled : 0))
                    report(rtapi::createRigChainControl(character, name, revision));
            }
        }
        if (!eligible)
            ImGui::TextDisabled("Define an anatomy chain with 4 to 64 bones first.");
        ImGui::EndCombo();
    }
    if (ImGui::BeginCombo("Add aim control", "Choose anatomy role")) {
        if (!anatomyReady || anatomy["roles"].empty())
            ImGui::TextDisabled("Define an anatomy role first.");
        if (anatomyReady) {
            for (const auto& role : anatomy["roles"]) {
                const auto roleName = role["role"].get<std::string>();
                const auto controlName = roleName + ".aim";
                bool exists = false;
                for (const auto& row : data["controls"])
                    if (row["name"] == controlName)
                        exists = true;
                if (ImGui::Selectable(roleName.c_str(), false,
                                      exists ? ImGuiSelectableFlags_Disabled : 0))
                    report(rtapi::createRigAimControl(character, roleName, revision));
            }
        }
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    const auto selected = data["selected_control"].get<std::string>();
    if (ImGui::BeginCombo("IK control",
                          selected.empty() ? "Select IK control" : selected.c_str())) {
        for (const auto& row : data["controls"]) {
            const auto name = row["name"].get<std::string>();
            if (ImGui::Selectable(name.c_str(), name == selected))
                report(rtapi::selectRigControl(character, name));
        }
        ImGui::EndCombo();
    }
    for (const auto& row : data["controls"]) {
        const auto name = row["name"].get<std::string>();
        if (name != selected)
            continue;
        const bool aim = row["solver"] == "aim";
        if (aim)
            ImGui::TextWrapped("Aim bone: %s", row["root"].get<std::string>().c_str());
        else
            ImGui::TextWrapped("%s -> %s -> %s", row["root"].get<std::string>().c_str(),
                               row["mid"].get<std::string>().c_str(),
                               row["tip"].get<std::string>().c_str());
        ImGui::Text("Solver: %s | %zu bones", row["solver"].get<std::string>().c_str(),
                    row["bones"].size());
        bool enabled = row["enabled"].get<bool>();
        if (ImGui::Checkbox("IK enabled", &enabled)) {
            auto r = rtapi::setRigIKFK(character, name, enabled ? 1.f : 0.f, revision);
            report(r);
            if (r.ok)
                report(rtapi::applyRigPosePreview(character));
        }
        const auto handle = data["handle"].get<std::string>();
        ImGui::BeginDisabled(ctx.scene.rigView.pose.hasPreview);
        if (ImGui::RadioButton(aim ? "Aim target" : "Target handle", handle == "target"))
            report(rtapi::selectRigControl(character, name, "target"));
        ImGui::SameLine();
        if (ImGui::RadioButton(aim ? "Roll up" : "Pole handle", handle == "pole"))
            report(rtapi::selectRigControl(character, name, "pole"));
        ImGui::EndDisabled();
        if (!aim) {
            ImGui::BeginDisabled(ctx.scene.rigView.pose.hasPreview);
            if (ImGui::RadioButton("Orientation handle", handle == "orientation"))
                report(rtapi::selectRigControl(character, name, "orientation"));
            ImGui::EndDisabled();
        }
        float blend = row["blend"].get<float>();
        if (ImGui::SliderFloat("IK / FK blend", &blend, 0, 1))
            report(rtapi::setRigIKFK(character, name, blend, revision));
        if (ImGui::IsItemDeactivatedAfterEdit() && ctx.scene.rigView.pose.hasPreview)
            report(rtapi::applyRigPosePreview(character));
        if (row["solver"] == "two_bone") {
            const bool pending = ctx.scene.rigView.pose.hasPreview;
            ImGui::BeginDisabled(pending);
            if (ImGui::Button("Match IK <- FK")) {
                auto matched = rtapi::matchRigIKToFK(character, name, revision);
                report(matched);
                if (matched.ok) {
                    report(rtapi::applyRigPosePreview(character));
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Match FK <- IK")) {
                auto matched = rtapi::matchRigFKToIK(character, name, revision);
                report(matched);
                if (matched.ok) {
                    report(rtapi::applyRigPosePreview(character));
                }
            }
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Match the destination to the visible pose, then switch without a pop.");
            }
        }
        float target[3], pole[3];
        for (int i = 0; i < 3; ++i) {
            target[i] = row["target_world"][i].get<float>();
            pole[i] = row["pole_world"][i].get<float>();
        }
        const bool moved = ImGui::InputFloat3("Target world", target);
        const bool targetDone = ImGui::IsItemDeactivatedAfterEdit();
        const bool bent = ImGui::InputFloat3("Pole world", pole);
        const bool poleDone = ImGui::IsItemDeactivatedAfterEdit();
        if (moved || bent)
            report(rtapi::setRigIKTarget(character, name, Vec3(target[0], target[1], target[2]),
                                         Vec3(pole[0], pole[1], pole[2]), revision));
        if ((targetDone || poleDone) && ctx.scene.rigView.pose.hasPreview)
            report(rtapi::applyRigPosePreview(character));
        if (!aim) {
            bool orientationEnabled = row["orientation_enabled"].get<bool>();
            float q[4];
            for (int i = 0; i < 4; ++i)
                q[i] = row["orientation_world"][i].get<float>();
            if (ImGui::Checkbox("Hold tip orientation", &orientationEnabled)) {
                if (orientationEnabled)
                    for (int i = 0; i < 4; ++i)
                        q[i] = row["tip_orientation_world"][i].get<float>();
                auto r =
                    rtapi::setRigIKOrientation(character, name, Quaternion(q[0], q[1], q[2], q[3]),
                                               orientationEnabled, revision);
                report(r);
                if (r.ok)
                    report(rtapi::applyRigPosePreview(character));
            }
            if (ImGui::InputFloat4("World quaternion (wxyz)", q))
                report(rtapi::setRigIKOrientation(character, name,
                                                  Quaternion(q[0], q[1], q[2], q[3]),
                                                  orientationEnabled, revision));
            if (ImGui::IsItemDeactivatedAfterEdit() && ctx.scene.rigView.pose.hasPreview)
                report(rtapi::applyRigPosePreview(character));
            if (orientationEnabled)
                ImGui::Text("Orientation error: %.2f deg",
                            row["orientation_error_degrees"].get<double>());
        }
        if (!aim) {
            bool contact = row["contact"].get<bool>();
            if (ImGui::Checkbox("Pin position in world", &contact)) {
                auto r = rtapi::setRigIKContact(character, name, contact, revision);
                report(r);
                if (r.ok)
                    report(rtapi::applyRigPosePreview(character));
            }
        }
        if (aim)
            ImGui::Text("Aim error: %.3f deg", row["aim_error_degrees"].get<double>());
        else
            ImGui::Text("Target error: %.5g", row["target_error_world"].get<double>());
        ImGui::TextDisabled(
            aim ? "Aim keys use the selected clip."
                : "Live contacts hold while scrubbing; IK keys use the selected clip.");
        if (!aim)
            drawRigSplineControls(ctx, character, name, revision, row);
        drawRigIKTimeline(ctx, character, name, revision, !aim);
    }
    if (!message.empty())
        ImGui::TextWrapped("%s", message.c_str());
}
}
