#include "UI/RigIKUI.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "imgui.h"
namespace RigUI {
void drawRigIKTimeline(UIContext& ctx, const std::string& character, const std::string& control,
                       uint64_t revision, bool contactsSupported) {
    if (!ImGui::TreeNode("IK clip channels"))
        return;
    static std::string error;
    static int start = 0, end = 30;
    static char name[129] = "IK_Baked";
    auto report = [&](const rtapi::Result& r) { error = r.ok ? "" : r.error; };
    nlohmann::json data;
    auto result = rtapi::getRigIKChannels(character, data);
    if (!result.ok) {
        ImGui::TextWrapped("%s", result.error.c_str());
        ImGui::TreePop();
        return;
    }
    const auto& channels = data["channels"];
    if (channels.contains(control)) {
        const auto& c = channels[control];
        ImGui::Text("%zu IK keys | %zu contact intervals", c["keys"].size(), c["contacts"].size());
        for (const auto& k : c["contacts"])
            ImGui::Text("Contact: %.3f to %.3f s", k["start_seconds"].get<double>(),
                        k["end_seconds"].get<double>());
    } else
        ImGui::TextDisabled("No IK channels for this control.");
    ImGui::BeginDisabled(ctx.scene.rigView.pose.hasPreview);
    if (ImGui::Button("Insert / Update IK key (I)"))
        report(rtapi::insertRigIKKey(character, control, revision));
    ImGui::SameLine();
    if (ImGui::Button("Remove IK key here"))
        report(rtapi::removeRigIKKey(character, control, revision));
    ImGui::InputInt("Start frame", &start);
    ImGui::InputInt(contactsSupported ? "End frame (exclusive contact)" : "End frame", &end);
    if (contactsSupported && ImGui::Button("Capture contact interval"))
        report(rtapi::setRigIKContactInterval(character, control, start, end, revision));
    if (ImGui::Button("Clear this control's IK channels"))
        report(rtapi::clearRigIKChannels(character, control, revision));
    ImGui::InputText("Baked clip name", name, sizeof(name));
    if (ImGui::Button("Bake all IK to new bone clip"))
        report(rtapi::bakeRigIKChannels(character, name, start, end, revision));
    ImGui::EndDisabled();
    if (contactsSupported)
        ImGui::TextWrapped("Contact captures the current achieved pose for [start, end). Bake "
                           "samples every frame, including end, using saved clip channels and "
                           "current actor placement. Undo restores changes.");
    else
        ImGui::TextWrapped("Aim target and roll-up handles can be keyed or baked to bone keys. "
                           "Bake samples every frame, including end. Undo restores changes.");
    if (!error.empty())
        ImGui::TextWrapped("%s", error.c_str());
    ImGui::TreePop();
}
}
