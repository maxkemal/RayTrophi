#pragma once

// Substance editor shown under every substance picker (flow source, collider).
// Built-ins are read-only: "Derive" makes a project substance from the shown
// one and assigns it to the picker. A project substance edits only through
// rtapi, the same calls IPC and Python make, so the panel cannot hold a value
// the library does not (CLAUDE.md §1).

#include "Api/RtApi.h"
#include "SubstanceLibrary.h"
#include "imgui.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace SubstanceEditorUI {

// One substance picker for every surface (domain default, flow source,
// collider): grouped by category, project substances labelled with their base.
// `empty_label` non-null offers an empty choice with that label. Returns true
// when `name` changed.
inline bool drawPicker(const char* id, std::string& name, const char* empty_label = nullptr) {
    std::vector<rtapi::SubstanceSummary> substances;
    if (!rtapi::listSubstances(substances).ok) return false;
    bool changed = false;
    const char* preview = name.empty() ? (empty_label ? empty_label : "") : name.c_str();
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo(id, preview)) {
        if (empty_label && ImGui::Selectable(empty_label, name.empty())) {
            changed = !name.empty();
            name.clear();
        }
        static const std::pair<const char*, const char*> kGroups[] = {
            {"liquid", "Liquids"}, {"fuel", "Fuels"},
            {"granular", "Granular"}, {"solid", "Solids"}};
        for (const auto& [key, title] : kGroups) {
            ImGui::SeparatorText(title);
            for (const auto& row : substances) {
                if (row.category != key) continue;
                const bool selected = name == row.name;
                const std::string label = row.builtin
                    ? row.name
                    : row.name + "  (from " + row.based_on + ")";
                if (ImGui::Selectable(label.c_str(), selected) && !selected) {
                    name = row.name;
                    changed = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

// Returns true when the library or `assigned` changed (the caller re-renders).
inline bool draw(const char* id, std::string& assigned) {
    if (assigned.empty()) return false;
    std::string text;
    if (!rtapi::getSubstance(assigned, text).ok) {
        ImGui::TextDisabled("Substance '%s' is not in the library", assigned.c_str());
        return false;
    }
    const nlohmann::json info = nlohmann::json::parse(text, nullptr, false);
    if (info.is_discarded()) return false;
    const bool builtin = info.value("builtin", true);
    const nlohmann::json& fields = info["fields"];
    std::vector<std::string> overridden = info.value("overridden", std::vector<std::string>{});
    const auto is_overridden = [&](const char* key) {
        return std::find(overridden.begin(), overridden.end(), key) != overridden.end();
    };

    bool changed = false;
    static std::string last_error;
    ImGui::PushID(id);
    const std::string header = builtin
        ? "Substance: " + assigned + " (built-in, read-only)"
        : "Substance: " + assigned + " (from " + info.value("based_on", std::string()) + ")";
    if (!ImGui::TreeNode(header.c_str())) {
        ImGui::PopID();
        return false;
    }

    // Derive: always offered, from a built-in or from a project substance.
    {
        static char derive_name[64] = {0};
        ImGui::SetNextItemWidth(-90.0f);
        ImGui::InputTextWithHint("##derive_name", "New substance name", derive_name,
                                 sizeof(derive_name));
        ImGui::SameLine();
        if (ImGui::Button("Derive")) {
            const auto result = rtapi::deriveSubstance(derive_name, assigned);
            if (result.ok) {
                assigned = derive_name;
                derive_name[0] = '\0';
                last_error.clear();
                changed = true;
            } else {
                last_error = result.error;
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Create a project substance that inherits every value from this one\n"
                "and assign it here. Only the fields you then change are stored, so\n"
                "a later fix to the base still reaches it. Saved with the project.");
        }
    }
    if (!last_error.empty()) ImGui::TextWrapped("%s", last_error.c_str());
    if (changed) {
        ImGui::TreePop();
        ImGui::PopID();
        return true;
    }

    const auto apply = [&](const char* key, const nlohmann::json& value) {
        nlohmann::json patch = nlohmann::json::object();
        patch[key] = value;
        const auto result = rtapi::setSubstanceFields(assigned, patch.dump());
        if (result.ok) {
            last_error.clear();
            changed = true;
        } else {
            last_error = result.error;
        }
    };

    ImGui::BeginDisabled(builtin);
    const char* group = "";
    for (const auto& spec : RayTrophiSim::substanceFieldSpecs()) {
        if (std::string(group) != spec.group) {
            group = spec.group;
            ImGui::SeparatorText(group);
        }
        ImGui::PushID(spec.key);
        const bool marked = is_overridden(spec.key);
        std::string label = std::string(marked ? "* " : "") + spec.key;
        if (spec.unit && spec.unit[0]) label += std::string(" (") + spec.unit + ")";
        const nlohmann::json& value = fields[spec.key];
        switch (spec.kind) {
            case RayTrophiSim::SubstanceFieldKind::Float: {
                float v = value.get<float>();
                const float speed = std::max(std::fabs(v) * 0.005f, 1.0e-4f);
                ImGui::SetNextItemWidth(160.0f);
                // Applied on release: dragging would republish the library
                // and invalidate the bake on every pixel of the drag.
                ImGui::DragFloat("##v", &v, speed, spec.min, spec.max, "%.5g",
                                 ImGuiSliderFlags_AlwaysClamp);
                if (ImGui::IsItemDeactivatedAfterEdit()) apply(spec.key, v);
                break;
            }
            case RayTrophiSim::SubstanceFieldKind::Bool: {
                bool v = value.get<bool>();
                if (ImGui::Checkbox("##v", &v)) apply(spec.key, v);
                break;
            }
            case RayTrophiSim::SubstanceFieldKind::Model: {
                static const char* models[] = {"fluid", "granular", "elastic"};
                int idx = 0;
                for (int i = 0; i < 3; ++i) if (value == models[i]) idx = i;
                ImGui::SetNextItemWidth(160.0f);
                if (ImGui::Combo("##v", &idx, models, 3)) apply(spec.key, models[idx]);
                break;
            }
            case RayTrophiSim::SubstanceFieldKind::Category: {
                static const char* categories[] = {"liquid", "granular", "solid", "fuel"};
                int idx = 2;
                for (int i = 0; i < 4; ++i) if (value == categories[i]) idx = i;
                ImGui::SetNextItemWidth(160.0f);
                if (ImGui::Combo("##v", &idx, categories, 4)) apply(spec.key, categories[idx]);
                break;
            }
            case RayTrophiSim::SubstanceFieldKind::Color: {
                float c[3] = {value[0].get<float>(), value[1].get<float>(), value[2].get<float>()};
                ImGui::SetNextItemWidth(160.0f);
                ImGui::ColorEdit3("##v", c, ImGuiColorEditFlags_NoInputs);
                if (ImGui::IsItemDeactivatedAfterEdit())
                    apply(spec.key, nlohmann::json::array({c[0], c[1], c[2]}));
                break;
            }
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(label.c_str());
        if (marked) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Revert")) apply(spec.key, nullptr);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Drop this override: follow the base substance again");
        }
        ImGui::PopID();
    }
    ImGui::EndDisabled();

    if (!builtin) {
        ImGui::Spacing();
        if (ImGui::Button("Remove substance")) {
            // This picker is itself a reference; release it first and put it
            // back if anything else still holds the substance.
            const std::string name = assigned;
            assigned.clear();
            const auto result = rtapi::removeSubstance(name);
            if (result.ok) {
                last_error.clear();
                changed = true;
            } else {
                assigned = name;
                last_error = result.error;
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Releases this picker, then removes the substance. Refused (and the\n"
                "picker kept) while another flow source, collider, domain binding or\n"
                "substance still uses it.");
        }
    }
    ImGui::TreePop();
    ImGui::PopID();
    return changed;
}

} // namespace SubstanceEditorUI
