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
// collider): grouped by canonical family, with detailed examples opt-in.
// `empty_label` non-null offers an empty choice with that label. Returns true
// when `name` changed.
inline bool drawPicker(const char* id, std::string& name, const char* empty_label = nullptr) {
    std::vector<rtapi::SubstanceSummary> substances;
    if (!rtapi::listSubstances(substances).ok) {
        return false;
    }
    bool changed = false;
    const char* preview = name.empty() ? (empty_label ? empty_label : "") : name.c_str();
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo(id, preview)) {
        static char filter[48] = {0};
        static bool show_details = false;
        if (ImGui::IsWindowAppearing()) {
            filter[0] = '\0';
            ImGui::SetKeyboardFocusHere();
        }
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputTextWithHint("##substance_filter", "Search all profiles", filter, sizeof(filter));
        ImGui::Checkbox("Detailed examples / tests", &show_details);
        std::string needle = filter;
        std::transform(needle.begin(), needle.end(), needle.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const auto matches = [&](std::string text) {
            std::transform(text.begin(), text.end(), text.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return needle.empty() || text.find(needle) != std::string::npos;
        };
        if (empty_label && ImGui::Selectable(empty_label, name.empty())) {
            changed = !name.empty();
            name.clear();
        }
        std::vector<nlohmann::json> metadata;
        std::vector<std::pair<std::string, std::string>> groups;
        for (const auto& row : substances) {
            const auto info = nlohmann::json::parse(row.catalogue_json);
            metadata.push_back(info);
            const auto tier = info.at("tier").get<std::string>();
            const bool visible = tier == "foundation" || tier == "project" ||
                show_details || !needle.empty() || row.name == name;
            const auto key = info.at("family").get<std::string>();
            if (visible && matches(row.name + " " + row.based_on + " " +
                    info.at("family_label").get<std::string>()) &&
                std::none_of(groups.begin(), groups.end(),
                    [&](const auto& group) { return group.first == key; })) {
                groups.emplace_back(key, info.at("family_label").get<std::string>());
            }
        }
        for (const auto& [key, title] : groups) {
            ImGui::SeparatorText(title.c_str());
            for (std::size_t index = 0; index < substances.size(); ++index) {
                const auto& row = substances[index];
                const auto& info = metadata[index];
                const auto tier = info.at("tier").get<std::string>();
                const bool selected = name == row.name;
                if (info.at("family") != key ||
                    (!show_details && needle.empty() && !selected &&
                        tier != "foundation" && tier != "project") ||
                    !matches(row.name + " " + row.based_on + " " + title)) {
                    continue;
                }
                const auto label = row.name + (tier == "project" ? " (project)" :
                    tier == "foundation" ? " (starting profile)" :
                    tier == "test" ? " (test)" : " (example)");
                if (ImGui::Selectable(label.c_str(), selected) && !selected) {
                    name = row.name;
                    changed = true;
                }
                if (ImGui::IsItemHovered()) {
                    const auto representative = info.at("representative").get<std::string>();
                    ImGui::SetTooltip("%s\n%s%s", row.category.c_str(),
                        representative.empty() ? "Base: " : "Starting values from: ",
                        representative.empty() ? info.at("root_profile").get_ref<const std::string&>().c_str() :
                            representative.c_str());
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

// Returns true when the library or `assigned` changed (the caller re-renders).
inline bool draw(const char* id, std::string& assigned,
                 const char* initial_tab = "General") {
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
    const auto& catalogue = info.at("catalogue");
    ImGui::TextWrapped("%s | %s", catalogue.at("family_label").get_ref<const std::string&>().c_str(),
        builtin ? "Built-in / read only" : "Shared project profile");
    if (!builtin) {
        ImGui::TextWrapped("Inherits: %s | %zu overridden fields",
            info.at("based_on").get_ref<const std::string&>().c_str(), overridden.size());
    }
    ImGui::TextWrapped("Shared scope: %zu recorded references. Derive to isolate this assignment.",
        info.at("affected_references").size());
    if (!ImGui::TreeNode("Profile details")) {
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

    const auto draw_field = [&](const RayTrophiSim::SubstanceFieldSpec& spec) {
        ImGui::PushID(spec.key);
        ImGui::BeginDisabled(builtin);
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
        ImGui::EndDisabled();
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
        {"Liquid fuel", "Liquid fuel", category == "fuel" || field_flag("fluid_flammable"),
            nullptr},
        {"Acoustics", "Acoustics", true, nullptr},
        {"Acoustic medium (fluid)", "Acoustic medium (equivalent fluid)", true, nullptr},
        {"Optical", "Look", true, nullptr},
    };
    static bool show_unused = false;
    const auto tab_for = [](const std::string& group) {
        if (group == "Identity") {
            return std::string("General");
        }
        if (group == "Liquid" || group == "Granular" || group == "Grains (DEM)") {
            return std::string("Mechanical");
        }
        if (group == "Acoustics" || group == "Acoustic medium (fluid)") {
            return std::string("Acoustics");
        }
        if (group == "Optical" || group == "Solver hints") {
            return std::string("Advanced");
        }
        return std::string("Thermal");
    };
    const ImGuiID initialized_key = ImGui::GetID("##SubstanceTabsInitialized");
    const bool initialize = ImGui::GetStateStorage()->GetInt(initialized_key, 0) == 0;
    if (ImGui::BeginTabBar("SubstanceSections", ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (const char* tab : {"General", "Mechanical", "Thermal", "Acoustics", "Advanced"}) {
            const auto flags = initialize && std::string(tab) == initial_tab ?
                ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
            if (!ImGui::BeginTabItem(tab, nullptr, flags)) {
                continue;
            }
            ImGui::BeginChild("ProfileFields", ImVec2(0,
                ImGui::GetTextLineHeightWithSpacing() * 14), false);
            if (std::string(tab) == "General") {
                const auto representative = catalogue.at("representative").get<std::string>();
                if (!representative.empty()) {
                    ImGui::TextWrapped("Representative starting values: %s. "
                        "This is not a universal measurement for the entire family.",
                        representative.c_str());
                }
                ImGui::TextWrapped("Surface acoustics: %s | Medium: %s",
                    field_flag("acoustic_defined") ? "Defined" : "Unconfigured",
                    field_flag("acoustic_medium_defined") ? "Defined" : "Unconfigured");
                const auto& references = info.at("affected_references");
                ImGui::TextWrapped("Editing this shared profile affects its references "
                    "and inheriting profiles (%zu recorded references).", references.size());
                if (ImGui::TreeNode("Affected references")) {
                    for (const auto& reference : references) {
                        ImGui::BulletText("%s", reference.get_ref<const std::string&>().c_str());
                    }
                    ImGui::TreePop();
                }
            }
            if (std::string(tab) == "Advanced") {
                ImGui::Checkbox("Show unused sections", &show_unused);
                ImGui::SeparatorText("Solver numerics");
                for (const auto& spec : RayTrophiSim::substanceFieldSpecs()) {
                    if (std::string(spec.group) == "Solver hints") {
                        draw_field(spec);
                    }
                }
            }
            for (const auto& section : sections) {
                if (tab_for(section.group) != tab) {
                    continue;
                }
                const bool open = section.used || show_unused;
                if (!open && !section.gate) {
                    continue;
                }
                if (std::string(tab) == "Acoustics") {
                    const bool configured = field_flag(std::string(section.group) == "Acoustics" ?
                        "acoustic_defined" : "acoustic_medium_defined");
                    ImGui::TextWrapped("%s: %s", section.title,
                        configured ? "Defined" : "Unconfigured (stored values are inactive)");
                    if (!ImGui::CollapsingHeader(section.title)) {
                        continue;
                    }
                } else {
                    ImGui::SeparatorText(section.title);
                }
                const bool granular_section = std::string(section.group) == "Granular";
                if (granular_section && dem) {
                    ImGui::TextWrapped("DEM uses grain fields. MPM continuum fields are "
                        "available through Show unused sections in Advanced.");
                }
                for (const auto& spec : RayTrophiSim::substanceFieldSpecs()) {
                    if (std::string(spec.group) != section.group) {
                        continue;
                    }
                    const bool gate = section.gate && std::string(spec.key) == section.gate;
                    if ((!open && !gate) || (granular_section && dem &&
                            std::string(spec.key) != "granular_transport" && !show_unused)) {
                        continue;
                    }
                    draw_field(spec);
                }
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
        ImGui::GetStateStorage()->SetInt(initialized_key, 1);
    }

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
