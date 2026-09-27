#include "UI/KinematicColliderAuthoringUI.h"

#include "Api/RtApi.h"
#include "imgui.h"
#include "scene_ui.h"

#include <algorithm>
#include <string>
#include <vector>

namespace KinematicColliderUI {
namespace {

struct AuthoringState {
    uint64_t selected_set = 0;
    uint64_t selected_proxy = 0;
    int character_index = 0;
    int bone_index = 0;
    std::string message;
    bool message_is_error = false;
};

AuthoringState state;

void report(const rtapi::Result& result, const char* success) {
    state.message_is_error = !result.ok;
    state.message = result.ok ? success : result.error;
}

bool consumerCheckbox(const char* label,
                      uint32_t bit,
                      uint32_t& mask) {
    bool enabled = (mask & bit) != 0u;
    if (!ImGui::Checkbox(label, &enabled)) {
        return false;
    }
    if (enabled) {
        mask |= bit;
    } else {
        mask &= ~bit;
    }
    return true;
}

void drawSetList(const std::vector<RayTrophiSim::KinematicProxySet>& sets) {
    if (sets.empty()) {
        ImGui::TextDisabled("No kinematic proxy sets.");
        state.selected_set = 0;
        state.selected_proxy = 0;
        return;
    }
    if (!ImGui::BeginTable(
            "KinematicProxySetTable",
            3,
            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
                ImGuiTableFlags_SizingStretchProp)) {
        return;
    }
    ImGui::TableSetupColumn("Set", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Rig", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Proxies", ImGuiTableColumnFlags_WidthFixed, 54.0f);
    for (const auto& set : sets) {
        ImGui::PushID(static_cast<int>(set.id));
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        if (ImGui::Selectable(
                set.name.c_str(),
                state.selected_set == set.id,
                ImGuiSelectableFlags_SpanAllColumns)) {
            state.selected_set = set.id;
            state.selected_proxy = 0;
        }
        ImGui::TableSetColumnIndex(1);
        ImGui::TextUnformatted(set.target_character.c_str());
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%zu", set.proxies.size());
        ImGui::PopID();
    }
    ImGui::EndTable();
}

void drawProxyList(const RayTrophiSim::KinematicProxySet& set) {
    if (set.proxies.empty()) {
        ImGui::TextDisabled("No proxies. Use Auto Fit or add one below.");
        state.selected_proxy = 0;
        return;
    }
    if (!ImGui::BeginTable(
            "KinematicProxyTable",
            3,
            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
                ImGuiTableFlags_SizingStretchProp)) {
        return;
    }
    ImGui::TableSetupColumn("Proxy", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Bone", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Shape", ImGuiTableColumnFlags_WidthFixed, 64.0f);
    for (const auto& proxy : set.proxies) {
        ImGui::PushID(static_cast<int>(proxy.id));
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        if (ImGui::Selectable(
                proxy.name.c_str(),
                state.selected_proxy == proxy.id,
                ImGuiSelectableFlags_SpanAllColumns)) {
            state.selected_proxy = proxy.id;
        }
        ImGui::TableSetColumnIndex(1);
        ImGui::TextUnformatted(proxy.bone.c_str());
        ImGui::TableSetColumnIndex(2);
        ImGui::TextUnformatted(
            RayTrophiSim::kinematicProxyShapeName(proxy.shape));
        ImGui::PopID();
    }
    ImGui::EndTable();
}

} // namespace

void drawAuthoringPanel(UIContext&) {
    ImGui::SeparatorText("Kinematic Collider Sources");
    ImGui::TextWrapped(
        "Bone-local analytic proxies shared by fluid, gas, granular and "
        "particle solvers. They do not deform or recook the character mesh.");

    std::vector<std::string> characters;
    const rtapi::Result characters_result =
        rtapi::listRigCharacters(characters);
    if (!characters_result.ok) {
        report(characters_result, "");
    }
    if (state.character_index >= static_cast<int>(characters.size())) {
        state.character_index = std::max(0, static_cast<int>(characters.size()) - 1);
    }
    const char* character_preview = characters.empty()
        ? "No rigged character"
        : characters[static_cast<std::size_t>(state.character_index)].c_str();
    ImGui::SetNextItemWidth(-120.0f);
    if (ImGui::BeginCombo("##KinematicCharacter", character_preview)) {
        for (int index = 0; index < static_cast<int>(characters.size()); ++index) {
            const bool selected = index == state.character_index;
            if (ImGui::Selectable(characters[index].c_str(), selected)) {
                state.character_index = index;
                state.bone_index = 0;
            }
            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(characters.empty());
    if (ImGui::Button("Add Set", ImVec2(110.0f, 0.0f))) {
        const std::string& character =
            characters[static_cast<std::size_t>(state.character_index)];
        RayTrophiSim::KinematicProxySet requested;
        requested.name = character + " Proxies";
        requested.target_character = character;
        RayTrophiSim::KinematicProxySet created;
        const rtapi::Result result =
            rtapi::createKinematicProxySet(requested, created);
        report(result, "Proxy set created.");
        if (result.ok) {
            state.selected_set = created.id;
            state.selected_proxy = 0;
        }
    }
    ImGui::EndDisabled();

    std::vector<RayTrophiSim::KinematicProxySet> sets;
    const rtapi::Result list_result = rtapi::listKinematicProxySets(sets);
    if (!list_result.ok) {
        report(list_result, "");
        return;
    }
    if (state.selected_set == 0 && !sets.empty()) {
        state.selected_set = sets.front().id;
    }
    drawSetList(sets);

    auto set_it = std::find_if(
        sets.begin(), sets.end(), [](const auto& set) {
            return set.id == state.selected_set;
        });
    if (set_it == sets.end()) {
        if (!state.message.empty()) {
            ImGui::TextColored(
                state.message_is_error
                    ? ImVec4(1.0f, 0.35f, 0.28f, 1.0f)
                    : ImVec4(0.35f, 0.85f, 0.45f, 1.0f),
                "%s",
                state.message.c_str());
        }
        return;
    }

    RayTrophiSim::KinematicProxySet edited = *set_it;
    bool set_changed = ImGui::Checkbox("Enabled##KinematicSet", &edited.enabled);
    ImGui::SameLine();
    set_changed |= ImGui::Checkbox(
        "Show in Viewport##KinematicSet", &edited.viewport_visible);
    set_changed |= ImGui::DragFloat(
        "Friction##KinematicSet", &edited.friction, 0.01f, 0.0f, 4.0f);
    set_changed |= ImGui::DragFloat(
        "Restitution##KinematicSet", &edited.restitution, 0.01f, 0.0f, 1.0f);
    set_changed |= ImGui::DragFloat(
        "Thickness##KinematicSet", &edited.thickness, 0.002f, 0.0f, 1.0f);
    ImGui::TextDisabled("Consumers");
    set_changed |= consumerCheckbox(
        "Fluid", RayTrophiSim::KinematicConsumerFluid, edited.consumer_mask);
    ImGui::SameLine();
    set_changed |= consumerCheckbox(
        "Gas", RayTrophiSim::KinematicConsumerGas, edited.consumer_mask);
    ImGui::SameLine();
    set_changed |= consumerCheckbox(
        "Granular", RayTrophiSim::KinematicConsumerGranular, edited.consumer_mask);
    set_changed |= consumerCheckbox(
        "Particles", RayTrophiSim::KinematicConsumerParticles, edited.consumer_mask);
    ImGui::SameLine();
    set_changed |= consumerCheckbox(
        "MSF", RayTrophiSim::KinematicConsumerMaterialState, edited.consumer_mask);
    if (set_changed) {
        report(
            rtapi::updateKinematicProxySet(edited.id, edited),
            "Proxy set updated.");
    }

    if (ImGui::Button("Auto Fit Skeleton", ImVec2(-1.0f, 0.0f))) {
        RayTrophiSim::KinematicAutoFitOptions options;
        uint32_t created_count = 0;
        const rtapi::Result result = rtapi::autoFitKinematicProxySet(
            edited.id, options, created_count);
        report(
            result,
            result.ok ? "Skeleton proxies generated." : "");
        state.selected_proxy = 0;
    }

    std::vector<RigAuthoring::BoneView> bones;
    const rtapi::Result bones_result =
        rtapi::listRigBones(edited.target_character, bones);
    if (state.bone_index >= static_cast<int>(bones.size())) {
        state.bone_index = std::max(0, static_cast<int>(bones.size()) - 1);
    }

    ImGui::SeparatorText("Proxies");
    drawProxyList(edited);
    const char* bone_preview = bones.empty()
        ? "No bones"
        : bones[static_cast<std::size_t>(state.bone_index)].name.c_str();
    ImGui::SetNextItemWidth(-120.0f);
    if (ImGui::BeginCombo("##KinematicBone", bone_preview)) {
        for (int index = 0; index < static_cast<int>(bones.size()); ++index) {
            const bool selected = index == state.bone_index;
            if (ImGui::Selectable(bones[index].name.c_str(), selected)) {
                state.bone_index = index;
            }
            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(bones.empty());
    if (ImGui::Button("Add Proxy", ImVec2(110.0f, 0.0f))) {
        RayTrophiSim::KinematicProxyDesc requested;
        requested.bone = bones[static_cast<std::size_t>(state.bone_index)].name;
        requested.name = requested.bone;
        RayTrophiSim::KinematicProxyDesc stored;
        const rtapi::Result result =
            rtapi::setKinematicProxy(edited.id, requested, stored);
        report(result, "Proxy added.");
        if (result.ok) {
            state.selected_proxy = stored.id;
        }
    }
    ImGui::EndDisabled();

    const auto proxy_it = std::find_if(
        edited.proxies.begin(), edited.proxies.end(), [](const auto& proxy) {
            return proxy.id == state.selected_proxy;
        });
    if (proxy_it != edited.proxies.end()) {
        RayTrophiSim::KinematicProxyDesc proxy = *proxy_it;
        bool proxy_changed = ImGui::Checkbox(
            "Proxy Enabled##KinematicProxy", &proxy.enabled);
        int proxy_bone_index = -1;
        for (int index = 0; index < static_cast<int>(bones.size()); ++index) {
            if (bones[static_cast<std::size_t>(index)].name == proxy.bone) {
                proxy_bone_index = index;
                break;
            }
        }
        const char* proxy_bone_preview = proxy_bone_index >= 0
            ? bones[static_cast<std::size_t>(proxy_bone_index)].name.c_str()
            : proxy.bone.c_str();
        if (ImGui::BeginCombo(
                "Bone##KinematicProxy", proxy_bone_preview)) {
            for (int index = 0; index < static_cast<int>(bones.size()); ++index) {
                const bool selected = index == proxy_bone_index;
                if (ImGui::Selectable(bones[index].name.c_str(), selected)) {
                    proxy.bone = bones[index].name;
                    proxy_bone_index = index;
                    proxy_changed = true;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        int shape = static_cast<int>(proxy.shape);
        const char* shapes[] = {"Sphere", "Capsule", "Box"};
        if (ImGui::Combo("Shape##KinematicProxy", &shape, shapes, 3)) {
            proxy.shape = static_cast<RayTrophiSim::KinematicProxyShape>(shape);
            proxy_changed = true;
        }
        proxy_changed |= ImGui::DragFloat3(
            "Local Position", &proxy.local_position.x, 0.005f);
        proxy_changed |= ImGui::DragFloat3(
            "Local Rotation", &proxy.local_rotation_degrees.x, 0.5f);
        if (proxy.shape == RayTrophiSim::KinematicProxyShape::Capsule) {
            proxy_changed |= ImGui::DragFloat3(
                "Local Axis", &proxy.local_axis.x, 0.01f, -1.0f, 1.0f);
            proxy_changed |= ImGui::DragFloat(
                "Radius", &proxy.radius, 0.002f, 0.001f, 10.0f);
            proxy_changed |= ImGui::DragFloat(
                "Half Length", &proxy.half_length, 0.005f, 0.0f, 50.0f);
        } else if (proxy.shape == RayTrophiSim::KinematicProxyShape::Sphere) {
            proxy_changed |= ImGui::DragFloat(
                "Radius", &proxy.radius, 0.002f, 0.001f, 10.0f);
        } else {
            proxy_changed |= ImGui::DragFloat3(
                "Half Extents", &proxy.half_extents.x, 0.005f, 0.001f, 50.0f);
        }
        if (proxy_changed) {
            RayTrophiSim::KinematicProxyDesc stored;
            report(
                rtapi::setKinematicProxy(edited.id, proxy, stored),
                "Proxy updated.");
        }
        if (ImGui::Button("Remove Selected Proxy", ImVec2(-1.0f, 0.0f))) {
            report(
                rtapi::removeKinematicProxy(edited.id, proxy.id),
                "Proxy removed.");
            state.selected_proxy = 0;
        }
    }

    if (ImGui::Button("Delete Proxy Set", ImVec2(-1.0f, 0.0f))) {
        report(
            rtapi::removeKinematicProxySet(edited.id),
            "Proxy set deleted.");
        state.selected_set = 0;
        state.selected_proxy = 0;
    }
    if (!state.message.empty()) {
        ImGui::TextColored(
            state.message_is_error
                ? ImVec4(1.0f, 0.35f, 0.28f, 1.0f)
                : ImVec4(0.35f, 0.85f, 0.45f, 1.0f),
            "%s",
            state.message.c_str());
    }
}

} // namespace KinematicColliderUI
