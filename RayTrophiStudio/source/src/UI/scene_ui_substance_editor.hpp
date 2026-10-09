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
#include <cctype>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>

namespace SubstanceEditorUI {

// Readable label for a field key; the key itself is the tooltip (it is what
// scripts write). "granular_friction_degrees" -> "Friction".
inline std::string fieldLabel(const char* key) {
    std::string k = key;
    if (k == "default_constitutive_model") return "Solid behavior";
    if (k == "granular_transport") return "Solver (dem = grains, mpm = continuum)";
    for (const char* prefix : {"granular_", "grain_", "solver_", "liquid_"}) {
        const std::string p = prefix;
        if (k.rfind(p, 0) == 0) { k = k.substr(p.size()); break; }
    }
    // The unit is printed after the label, so a unit suffix is dropped;
    // "_kelvin" reads as "temperature" ("Melt temperature (K)").
    using Suffix = std::pair<const char*, const char*>;
    for (const auto& [suffix, word] : {Suffix{"_degrees", ""}, Suffix{"_deg", ""},
                                       Suffix{"_kelvin", "_temperature"}, Suffix{"_per_s", ""},
                                       Suffix{"_n_m", ""}, Suffix{"_m", ""}}) {
        const std::string x = suffix;
        if (k.size() > x.size() && k.compare(k.size() - x.size(), x.size(), x) == 0) {
            k = k.substr(0, k.size() - x.size()) + word;
            break;
        }
    }
    std::replace(k.begin(), k.end(), '_', ' ');
    if (!k.empty()) k[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(k[0])));
    return k;
}

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
        static char filter[48] = {0};
        if (ImGui::IsWindowAppearing()) {
            filter[0] = '\0';
            ImGui::SetKeyboardFocusHere();
        }
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputTextWithHint("##substance_filter", "Search", filter, sizeof(filter));
        std::string needle = filter;
        std::transform(needle.begin(), needle.end(), needle.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const auto matches = [&](const std::string& text) {
            if (needle.empty()) return true;
            std::string lower = text;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return lower.find(needle) != std::string::npos;
        };
        if (empty_label && ImGui::Selectable(empty_label, name.empty())) {
            changed = !name.empty();
            name.clear();
        }
        static const std::pair<const char*, const char*> kGroups[] = {
            {"liquid", "Liquids"}, {"fuel", "Fuels"},
            {"granular", "Granular"}, {"solid", "Solids"}};
        for (const auto& [key, title] : kGroups) {
            bool any = false;
            for (const auto& row : substances) {
                any |= row.category == key && (matches(row.name) || matches(row.based_on));
            }
            if (!any) continue;
            ImGui::SeparatorText(title);
            for (const auto& row : substances) {
                if (row.category != key || !(matches(row.name) || matches(row.based_on))) {
                    continue;
                }
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
    // Field being dragged ("substance/key") and its in-flight value.
    static std::string pending_key;
    static float pending_value[3] = {0.0f, 0.0f, 0.0f};
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
    const auto draw_field = [&](const RayTrophiSim::SubstanceFieldSpec& spec) {
        ImGui::PushID(spec.key);
        const std::string edit_key = assigned + "/" + spec.key;
        const bool marked = is_overridden(spec.key);
        std::string label = std::string(marked ? "* " : "") + fieldLabel(spec.key);
        if (spec.unit && spec.unit[0]) label += std::string(" (") + spec.unit + ")";
        const nlohmann::json& value = fields[spec.key];
        switch (spec.kind) {
            case RayTrophiSim::SubstanceFieldKind::Float: {
                // The library value is re-read every frame but only written on
                // release, so the dragged value lives here until then; reading
                // the library mid-drag snapped the field back each frame.
                const bool editing = pending_key == edit_key;
                float v = editing ? pending_value[0] : value.get<float>();
                const float speed = std::max(std::fabs(v) * 0.005f, 1.0e-4f);
                ImGui::SetNextItemWidth(160.0f);
                // Applied on release: dragging would republish the library
                // and invalidate the bake on every pixel of the drag.
                ImGui::DragFloat("##v", &v, speed, spec.min, spec.max, "%.5g",
                                 ImGuiSliderFlags_AlwaysClamp);
                if (ImGui::IsItemActive()) {
                    pending_key = edit_key;
                    pending_value[0] = v;
                }
                if (ImGui::IsItemDeactivatedAfterEdit()) apply(spec.key, v);
                if (!ImGui::IsItemActive() && pending_key == edit_key) pending_key.clear();
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
            case RayTrophiSim::SubstanceFieldKind::Transport: {
                static const char* transports[] = {"mpm", "dem"};
                int index = value == "dem" ? 1 : 0;
                ImGui::SetNextItemWidth(160.0f);
                if (ImGui::Combo("##v", &index, transports, 2)) {
                    apply(spec.key, transports[index]);
                }
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
                const bool editing = pending_key == edit_key;
                float c[3] = {value[0].get<float>(), value[1].get<float>(), value[2].get<float>()};
                if (editing) std::copy(pending_value, pending_value + 3, c);
                ImGui::SetNextItemWidth(160.0f);
                ImGui::ColorEdit3("##v", c, ImGuiColorEditFlags_NoInputs);
                if (ImGui::IsItemActive()) {
                    pending_key = edit_key;
                    std::copy(c, c + 3, pending_value);
                }
                if (ImGui::IsItemDeactivatedAfterEdit())
                    apply(spec.key, nlohmann::json::array({c[0], c[1], c[2]}));
                if (!ImGui::IsItemActive() && pending_key == edit_key) pending_key.clear();
                break;
            }
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(label.c_str());
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Script key: %s", spec.key);
        }
        if (marked) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Revert")) apply(spec.key, nullptr);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Drop this override: follow the base substance again");
        }
        ImGui::PopID();
    };
    const auto field_text = [&](const char* key) {
        return fields.contains(key) && fields[key].is_string()
            ? fields[key].get<std::string>() : std::string();
    };
    const auto field_flag = [&](const char* key) {
        return fields.contains(key) && fields[key].is_boolean() && fields[key].get<bool>();
    };
    const std::string category = field_text("category");
    const std::string model = field_text("default_constitutive_model");
    const bool granular = model == "granular" || category == "granular";
    const bool dem = granular && field_text("granular_transport") == "dem";
    // A section is drawn when this substance can use it; its switch (combustible,
    // meltable) is always drawn so the section can be turned on. Nothing is
    // unreachable: "Show unused sections" draws the rest, and IPC sets any key.
    struct Section { const char* group; const char* title; bool used; const char* gate; };
    const Section sections[] = {
        {"Identity", "Identity", true, nullptr},
        {"Thermal", "Thermal", true, nullptr},
        {"Phase change", "Melting and boiling", field_flag("meltable"), "meltable"},
        {"Liquid", "Liquid", category == "liquid" || category == "fuel" || model == "fluid" ||
            field_flag("meltable"), nullptr},
        {"Granular", dem ? "Granular (grains: DEM)" : "Granular (continuum: MPM)", granular,
            nullptr},
        {"Grains (DEM)", "Grain material (DEM)", dem, nullptr},
        {"Moisture", "Moisture", category == "solid" || category == "granular", nullptr},
        {"Combustion", "Burning (solid)", field_flag("combustible"), "combustible"},
        {"Liquid fuel", "Liquid fuel", category == "fuel" || field_flag("fluid_flammable") ||
            field_flag("fluid_extinguishing"), nullptr},
        {"Optical", "Look", true, nullptr},
    };
    static bool show_unused = false;
    std::string hidden;
    for (const auto& section : sections) {
        const bool open = section.used || show_unused;
        if (!open) {
            hidden += (hidden.empty() ? "" : ", ") + std::string(section.title);
        }
        if (!open && !section.gate) continue;
        ImGui::SeparatorText(section.title);
        const bool granular_section = std::string(section.group) == "Granular";
        if (granular_section && dem) {
            // These continuum fields feed MPM only; editing them on a DEM
            // substance changed nothing, which is the panel lying.
            ImGui::TextWrapped("Grains (DEM) use the grain material below; the continuum fields "
                "are used only when the solver is mpm (Show unused sections lists them).");
        }
        for (const auto& spec : RayTrophiSim::substanceFieldSpecs()) {
            if (std::string(spec.group) != section.group) continue;
            const bool is_gate = section.gate && std::string(spec.key) == section.gate;
            const bool is_transport = std::string(spec.key) == "granular_transport";
            if (!open && !is_gate) continue;
            if (granular_section && dem && !is_transport && !show_unused) continue;
            draw_field(spec);
        }
    }
    // Viewing controls stay usable on a read-only built-in; only fields lock.
    ImGui::EndDisabled();
    // Solver numerics are tuned per substance but are not material physics.
    const bool advanced_open = ImGui::TreeNode("Advanced: solver numerics");
    if (advanced_open) {
        ImGui::BeginDisabled(builtin);
        for (const auto& spec : RayTrophiSim::substanceFieldSpecs()) {
            if (std::string(spec.group) == "Solver hints") draw_field(spec);
        }
        ImGui::EndDisabled();
        ImGui::TreePop();
    }
    ImGui::Checkbox("Show unused sections", &show_unused);
    if (ImGui::IsItemHovered()) {
        const std::string tip = hidden.empty() ? std::string("Every section is in use.")
            : "Not used by this substance: " + hidden;
        ImGui::SetTooltip("%s", tip.c_str());
    }
    ImGui::BeginDisabled(builtin);
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
