#include "UI/GameDesignUI.h"

#include "Api/RtApi.h"
#include "Api/RtApiPaintWet.h"
#include "Api/RtApiGame.h"
#include "Audio/AudioService.h"
#include "UI/AudioSourceLabel.h"
#include "UIContext.h"
#include "scene_data.h"
#include "imgui.h"
#include "json.hpp"
#include "ui_modern.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <sstream>
#include <utility>
#include <vector>

namespace GameDesignUI {
namespace {

using json = nlohmann::json;

// The last failed call, shown until the next one succeeds. A message, not state.
std::string last_error;

void report(const rtapi::Result& result) {
    last_error = result.ok ? std::string() : result.error;
}

bool addPaintMaterial(const std::string& object, rtapi::MaterialInfo material, int resolution) {
    for (const auto& other : rtapi::listObjects()) {
        if (other == object) {
            continue;
        }
        const auto slots = rtapi::objectMaterials(other);
        if (std::find(slots.begin(), slots.end(), material.name) == slots.end()) {
            continue;
        }
        std::string clone;
        auto result = rtapi::createMaterial(material.type, object + "_Paint_" + material.name,
                                            clone, material.name);
        if (result.ok) {
            result = rtapi::assignMaterial(object, clone, material.name);
        }
        if (result.ok) {
            result = rtapi::getMaterial(clone, material);
        }
        if (!result.ok) {
            report(result);
            return false;
        }
        break;
    }
    rtapi::WetPaintSurfaceRuntimeInfo info;
    const auto result = rtapi::addWetPaintSurface(object, material.id, resolution, info);
    report(result);
    return result.ok;
}

// Unsent demo form. Re-read from the live demo whenever it is (re)configured
// elsewhere (IPC, project load), so the form never contradicts the core.
struct DemoDraft {
    int balloons = 12;
    int particles_per_balloon = 1800;
    int seed = 417;
    float interval_min = 2.5f;
    float interval_max = 5.0f;
    bool autostart = false;
    std::string interaction_mode = "carry";
    bool body_controls = false;
    bool jolt_motion = true;
    bool impact_soft_body = true;
    float wind_strength = 0.25f;
    float size_min = 1.0f;
    float size_max = 1.0f;
    float idle_reset_seconds = 0.0f;   // 0 = off
    std::vector<std::array<float, 3>> colors;   // empty: the core's default palette
    std::string pop_sound;   // audio source id; empty = silent pops
    std::string surprise_sound;   // empty: the pop sound, brighter
    float surprise_chance = 0.1f;
    int sparkles = 90;            // firework sparks per surprise pop / milestone burst, 0 = off
    bool sparkle_trails = true;
    float drop_sparks = 0.25f;    // chance a sampled paint drop contact sparks, 0 = off
    bool drop_trails = true;      // paint drops leave trails in their colour
    float burst_clumpiness = 0.7f;  // pop shape: 0 = even fine spray, 1 = ligaments/clumps/blobs
    int burst_jets_max = 10;        // ligaments per pop drawn from 3..this
    float room_goal = 0.5f;              // 0 = no shared goal
    float finale_reset_seconds = 10.0f;  // 0 = no auto reset after the finale
    std::string celebrate_sound;
    float float_amplitude = 0.08f;   // m, 0 = still
    float drift_speed = 0.12f;       // m/s, 0 = no wandering
    bool set_transmission = false;   // off = the balloon materials keep their own value
    float transmission = 0.6f;
    bool spawn_auto = true;          // from the paint surfaces
    float spawn_min[3] = {-3.0f, 0.8f, -3.0f};
    float spawn_max[3] = {3.0f, 2.8f, 2.8f};
    std::vector<std::string> visual_nodes;   // empty: particle balloons
    std::string synced_from;   // config dump the draft was last loaded from
};
DemoDraft draft;

json demoCommand(const char* operation, const json& config = json::object(),
                 bool report_result = true) {
    std::string output;
    const auto result = rtapi::wetPaintDemoCommand(operation, config.dump(), output);
    if (report_result) {
        report(result);   // the per-frame status read must not wipe the last error
    }
    if (!result.ok) {
        return json();
    }
    try {
        return json::parse(output);
    } catch (...) {
        return json();
    }
}

void syncDraft(const json& config) {
    if (!config.is_object() || config.empty()) {
        return;
    }
    const std::string dump = config.dump();
    if (dump == draft.synced_from) {
        return;
    }
    draft.synced_from = dump;
    draft.balloons = config.value("balloons", draft.balloons);
    draft.particles_per_balloon = config.value("particles_per_balloon", draft.particles_per_balloon);
    draft.seed = config.value("seed", draft.seed);
    draft.interval_min = config.value("interval_min", draft.interval_min);
    draft.interval_max = config.value("interval_max", draft.interval_max);
    draft.autostart = config.value("autostart", draft.autostart);
    draft.body_controls = config.value("body_controls", false);
    draft.jolt_motion = config.value("jolt_motion", true);
    draft.impact_soft_body = config.value("impact_soft_body", true);
    draft.wind_strength = config.value("wind_strength", 0.25f);
    draft.interaction_mode = config.value("interaction_mode",
        config.contains("visual_nodes") ? std::string("carry") : std::string("pop"));
    draft.size_min = config.value("size_min", 1.0f);
    draft.size_max = config.value("size_max", 1.0f);
    draft.idle_reset_seconds = config.value("idle_reset_seconds", 0.0f);
    draft.colors.clear();
    if (config.contains("colors") && config["colors"].is_array()) {
        for (const auto& c : config["colors"]) {
            if (c.is_array() && c.size() == 3 && c[0].is_number() && c[1].is_number() &&
                c[2].is_number()) {
                draft.colors.push_back({c[0].get<float>(), c[1].get<float>(), c[2].get<float>()});
            }
        }
    }
    draft.pop_sound = config.value("pop_sound", std::string());
    draft.surprise_sound = config.value("surprise_sound", std::string());
    draft.surprise_chance = config.value("surprise_chance", 0.1f);
    draft.sparkles = config.value("sparkles", 90);
    draft.sparkle_trails = config.value("sparkle_trails", true);
    draft.drop_sparks = config.value("drop_sparks", 0.25f);
    draft.drop_trails = config.value("drop_trails", true);
    draft.burst_clumpiness = config.value("burst_clumpiness", 0.7f);
    draft.burst_jets_max = config.value("burst_jets_max", 10);
    draft.room_goal = config.value("room_goal", 0.5f);
    draft.finale_reset_seconds = config.value("finale_reset_seconds", 10.0f);
    draft.celebrate_sound = config.value("celebrate_sound", std::string());
    draft.float_amplitude = config.value("float_amplitude", 0.08f);
    draft.drift_speed = config.value("drift_speed", 0.12f);
    draft.set_transmission = config.contains("transmission");
    draft.transmission = config.value("transmission", 0.6f);
    draft.spawn_auto = !config.contains("spawn_min");
    if (!draft.spawn_auto && config["spawn_min"].is_array() && config["spawn_max"].is_array()) {
        for (int k = 0; k < 3; ++k) {
            draft.spawn_min[k] = config["spawn_min"][k].get<float>();
            draft.spawn_max[k] = config["spawn_max"][k].get<float>();
        }
    }
    draft.visual_nodes.clear();
    if (config.contains("visual_nodes") && config["visual_nodes"].is_array()) {
        for (const auto& node : config["visual_nodes"]) {
            if (node.is_string()) draft.visual_nodes.push_back(node.get<std::string>());
        }
    }
}

json draftConfig() {
    json config = {{"balloons", draft.balloons},
                   {"particles_per_balloon", draft.particles_per_balloon},
                   {"seed", draft.seed},
                   {"interval_min", draft.interval_min},
                   {"interval_max", draft.interval_max},
                   {"autostart", draft.autostart},
                   {"interaction_mode", draft.visual_nodes.empty() ? "pop" : draft.interaction_mode},
                   {"body_controls", !draft.visual_nodes.empty() && draft.body_controls},
                   {"size_min", draft.size_min},
                   {"size_max", draft.size_max},
                   // 0 = off; 1..9 is not a valid setting, round it up.
                   {"idle_reset_seconds", draft.idle_reset_seconds <= 0.0f
                                              ? 0.0f : std::max(10.0f, draft.idle_reset_seconds)}};
    if (!draft.pop_sound.empty()) {
        config["pop_sound"] = draft.pop_sound;
    }
    if (!draft.surprise_sound.empty()) {
        config["surprise_sound"] = draft.surprise_sound;
    }
    config["surprise_chance"] = draft.surprise_chance;
    config["sparkles"] = std::clamp(draft.sparkles, 0, 400);
    config["sparkle_trails"] = draft.sparkle_trails;
    config["drop_sparks"] = std::clamp(draft.drop_sparks, 0.0f, 1.0f);
    config["drop_trails"] = draft.drop_trails;
    config["burst_clumpiness"] = std::clamp(draft.burst_clumpiness, 0.0f, 1.0f);
    config["burst_jets_max"] = std::clamp(draft.burst_jets_max, 3, 24);
    // Below the valid minimum means off, never an out-of-range config.
    config["room_goal"] = draft.room_goal < 0.05f ? 0.0f : draft.room_goal;
    config["finale_reset_seconds"] =
        draft.finale_reset_seconds < 3.0f ? 0.0f : draft.finale_reset_seconds;
    if (!draft.celebrate_sound.empty()) {
        config["celebrate_sound"] = draft.celebrate_sound;
    }
    config["float_amplitude"] = draft.float_amplitude;
    config["drift_speed"] = draft.drift_speed;
    if (draft.set_transmission) {
        config["transmission"] = draft.transmission;
    }
    if (!draft.spawn_auto) {
        // Keep at least the 0.2 m the core requires on every axis.
        json mn = json::array(), mx = json::array();
        for (int k = 0; k < 3; ++k) {
            mn.push_back(draft.spawn_min[k]);
            mx.push_back(std::max(draft.spawn_max[k], draft.spawn_min[k] + 0.2f));
        }
        config["spawn_min"] = mn;
        config["spawn_max"] = mx;
    }
    if (draft.colors.size() >= 2) {
        json colors = json::array();
        for (const auto& c : draft.colors) colors.push_back({c[0], c[1], c[2]});
        config["colors"] = colors;
    }
    if (!draft.visual_nodes.empty()) {
        // One mesh per balloon: the list decides the count.
        config["balloons"] = static_cast<int>(draft.visual_nodes.size());
        config["visual_nodes"] = draft.visual_nodes;   // absent: particle balloons
        config["jolt_motion"] = draft.jolt_motion;
        config["impact_soft_body"] = draft.impact_soft_body;
        config["wind_strength"] = draft.wind_strength;
    }
    return config;
}

// Long object lists scroll inside a box of at most `rows` lines instead of
// growing the panel.
bool beginListBox(const char* id, size_t count, int rows = 5) {
    const float line = ImGui::GetFrameHeightWithSpacing();
    const float height = line * static_cast<float>(std::min<size_t>(std::max<size_t>(count, 1),
                                                                     static_cast<size_t>(rows))) +
                         ImGui::GetStyle().WindowPadding.y;
    return ImGui::BeginChild(id, ImVec2(0.0f, height), ImGuiChildFlags_Borders);
}

// A combo listing the scene's objects (rtapi::listObjects, read only while the
// list is open). Returns the picked name, empty when nothing was picked.
std::string pickSceneObject(const char* label, const std::vector<std::string>& exclude) {
    std::string picked;
    if (ImGui::BeginCombo(label, "Pick from scene...", ImGuiComboFlags_HeightLarge)) {
        static char filter[64] = "";
        ImGui::InputTextWithHint("##filter", "filter", filter, sizeof(filter));
        std::string needle(filter);
        std::transform(needle.begin(), needle.end(), needle.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        for (const auto& name : rtapi::listObjects()) {
            if (std::find(exclude.begin(), exclude.end(), name) != exclude.end()) continue;
            if (!needle.empty()) {
                std::string lower = name;
                std::transform(lower.begin(), lower.end(), lower.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (lower.find(needle) == std::string::npos) continue;
            }
            if (ImGui::Selectable(name.c_str())) picked = name;
        }
        ImGui::EndCombo();
    }
    return picked;
}

// The balloon mesh list: add the selected object or one from the scene list,
// remove with x. Names are never typed.
void drawBalloonMeshes(const std::string& selected_object,
                       const std::vector<std::string>& surfaces) {
    ImGui::Text("Balloon meshes (%zu)", draft.visual_nodes.size());
    if (draft.visual_nodes.empty()) {
        ImGui::TextDisabled("None: particle balloons.");
    }
    if (!draft.visual_nodes.empty()) {
        if (beginListBox("##balloon_list", draft.visual_nodes.size())) {
            for (size_t i = 0; i < draft.visual_nodes.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::SmallButton("x")) {
                    draft.visual_nodes.erase(draft.visual_nodes.begin() +
                                             static_cast<std::ptrdiff_t>(i));
                    ImGui::PopID();
                    break;
                }
                ImGui::SameLine();
                ImGui::TextUnformatted(draft.visual_nodes[i].c_str());
                ImGui::PopID();
            }
        }
        ImGui::EndChild();
    }
    // A paint surface cannot also be a balloon (the core refuses it).
    std::vector<std::string> exclude = draft.visual_nodes;
    exclude.insert(exclude.end(), surfaces.begin(), surfaces.end());
    const bool full = draft.visual_nodes.size() >= 24;
    const bool can_add_selected = !full && !selected_object.empty() &&
        std::find(exclude.begin(), exclude.end(), selected_object) == exclude.end();
    ImGui::BeginDisabled(!can_add_selected);
    if (ImGui::Button("Add selected")) {
        draft.visual_nodes.push_back(selected_object);
    }
    ImGui::EndDisabled();
    if (!full) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1.0f);
        const std::string picked = pickSceneObject("##balloon_pick", exclude);
        if (!picked.empty()) draft.visual_nodes.push_back(picked);
    }
    if (!draft.visual_nodes.empty() && ImGui::SmallButton("Clear list")) {
        draft.visual_nodes.clear();
    }
}

// Every field paint.wet_set_settings writes; same ranges as the API.
void drawSurfaceSettings(const rtapi::WetPaintSurfaceRuntimeInfo& surface) {
    rtapi::WetPaintSurfaceSettingsInfo s = surface.settings;
    bool changed = false;
    changed |= ImGui::SliderFloat("Wetness", &s.wetness, 0.0f, 1.0f);
    changed |= ImGui::DragFloat("Open time (s)", &s.lifetime_seconds, 0.1f, 0.05f, 600.0f, "%.2f");
    changed |= ImGui::SliderFloat("Diffusion", &s.diffusion, 0.0f, 2.0f);
    changed |= ImGui::SliderFloat("Runoff", &s.runoff, 0.0f, 2.0f);
    changed |= ImGui::SliderFloat("Absorption", &s.absorption, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Drip head", &s.drip_head, 0.0f, 1.5f);
    changed |= ImGui::SliderFloat("Falloff", &s.falloff, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Paint load", &s.paint_load, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Deposit rate", &s.deposit_rate, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Pickup rate", &s.pickup_rate, 0.0f, 1.0f);
    changed |= ImGui::DragFloat("Trail radius (px@1024)", &s.brush_radius, 0.1f, 0.01f, 256.0f, "%.2f");
    changed |= ImGui::Checkbox("Oil paint", &s.oil);
    ImGui::SeparatorText("Material channels");
    changed |= ImGui::Checkbox("Reveal object with paint", &s.reveal_opacity);
    if (s.reveal_opacity) {
        ImGui::TextWrapped("Unpainted areas are invisible. Clear paint hides the object again. "
                           "This mode replaces the opacity map on solid geometry.");
    }
    changed |= ImGui::Checkbox("Roughness while wet", &s.roughness_channel);
    if (s.roughness_channel) {
        changed |= ImGui::SliderFloat("Wet roughness", &s.wet_roughness, 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Dry roughness", &s.dry_roughness, 0.0f, 1.0f);
    }
    changed |= ImGui::Checkbox("Raised paint (height + normal)", &s.height_normal_channel);
    if (s.height_normal_channel) {
        changed |= ImGui::SliderFloat("Height", &s.height_contribution, 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Normal strength", &s.normal_strength, 0.0f, 8.0f);
    }
    if (changed) {
        report(rtapi::setWetPaintSurfaceSettings(surface.object, surface.material_id, s));
    }
}

void drawSplat(const rtapi::WetPaintSplatInfo& current) {
    rtapi::WetPaintSplatInfo s = current;
    bool changed = false;
    changed |= ImGui::Checkbox("Particles paint on contact", &s.consume_particles);
    char system[128];
    std::snprintf(system, sizeof(system), "%s", s.particle_system.c_str());
    if (ImGui::InputText("Only system (empty: all)", system, sizeof(system),
                         ImGuiInputTextFlags_EnterReturnsTrue)) {
        s.particle_system = system;
        changed = true;
    }
    changed |= ImGui::DragFloat("Splat radius (m)", &s.radius, 0.005f, 0.001f, 10.0f, "%.3f");
    changed |= ImGui::SliderFloat("Splat amount", &s.amount, 0.0f, 1.0f);
    changed |= ImGui::DragFloat("Min impact speed", &s.min_speed, 0.05f, 0.0f, 1000.0f, "%.2f");
    changed |= ImGui::SliderFloat("Min facing", &s.facing_min, -1.0f, 1.0f);
    int max_events = static_cast<int>(std::min<uint32_t>(s.max_events_per_tick, 1u << 20));
    if (ImGui::DragInt("Max contacts / tick", &max_events, 64.0f, 1, 1 << 20)) {
        s.max_events_per_tick = static_cast<uint32_t>(std::clamp(max_events, 1, 1 << 20));
        changed = true;
    }
    if (changed) {
        report(rtapi::setWetPaintSplat(s));
    }
}

// Audio sources (id, name) read from the scene's audio service. Values only.
std::vector<std::pair<std::string, std::string>> audioSources() {
    std::vector<std::pair<std::string, std::string>> out;
    auto* context = rtapi::boundContext();
    if (!context || !context->scene.audio) return out;
    try {
        const json data = context->scene.audio->invoke("list", json::object());
        for (const auto& e : data.value("emitters", json::array())) {
            if (!e.is_object() || !e.contains("id") || !e["id"].is_string()) continue;
            const std::string name = AudioUI::sourceLabel(e, data);
            out.emplace_back(e["id"].get<std::string>(), name);
        }
    } catch (...) {
    }
    return out;
}

// Balloon/paint palette (linear RGB, like material base_color). 2..8 colours.
void drawPalette(const json& demo_status) {
    if (draft.colors.empty() && demo_status.is_object() && demo_status.contains("palette")) {
        // The live palette (the core default when the config has none).
        for (const auto& c : demo_status["palette"]) {
            if (c.is_array() && c.size() == 3) {
                draft.colors.push_back({c[0].get<float>(), c[1].get<float>(), c[2].get<float>()});
            }
        }
    }
    ImGui::Text("Colours (%zu)", draft.colors.size());
    for (size_t i = 0; i < draft.colors.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        if (i % 4 != 0) ImGui::SameLine();
        ImGui::ColorEdit3("##colour", draft.colors[i].data(),
                          ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float);
        if (draft.colors.size() > 2 && ImGui::BeginPopupContextItem("##colour_menu")) {
            if (ImGui::MenuItem("Remove colour")) {
                draft.colors.erase(draft.colors.begin() + static_cast<std::ptrdiff_t>(i));
                ImGui::EndPopup();
                ImGui::PopID();
                break;
            }
            ImGui::EndPopup();
        }
        ImGui::PopID();
    }
    if (draft.colors.size() < 8) {
        if (!draft.colors.empty()) ImGui::SameLine();
        if (ImGui::Button("+##add_colour")) draft.colors.push_back({1.0f, 0.05f, 0.4f});
    }
    ImGui::Checkbox("Same translucency for all balloons", &draft.set_transmission);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Every balloon has its own material (its colour changes), so a\n"
                          "hand-edited value drifts between them. On: the demo writes this\n"
                          "transmission on each balloon when it appears.");
    }
    if (draft.set_transmission) {
        ImGui::SliderFloat("Transmission", &draft.transmission, 0.0f, 1.0f, "%.2f");
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Linear colour (as material base colour). Right-click a colour to remove it.\n"
                          "Keep other channels near 0 for vivid paint; overlapping splats blend.");
    }
}

void drawSoundPicker(const char* label, std::string& selected, const char* none_label) {
    std::string preview = selected.empty() ? std::string(none_label) : selected;
    const auto sources = audioSources();
    for (const auto& [id, name] : sources) {
        if (id == selected) preview = name;
    }
    if (ImGui::BeginCombo(label, preview.c_str())) {
        if (ImGui::Selectable(none_label, selected.empty())) selected.clear();
        for (const auto& [id, name] : sources) {
            if (ImGui::Selectable((name + "##" + id).c_str(), id == selected)) {
                selected = id;
            }
        }
        ImGui::EndCombo();
    }
}

void drawPopSound() {
    drawSoundPicker("Pop sound", draft.pop_sound, "None (silent)");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("An Audio source (mono asset). Every popped balloon plays it once at\n"
                          "its own position; pops at the same moment do not cut each other.");
    }
}

// Droplet colliders: the existing particle collider assignment of the four
// droplet systems. The selected object's collider is set here for all four.
void drawColliders(const std::string& selected_object, const json& demo_status) {
    const json rows = demo_status.is_object() ? demo_status.value("colliders", json::array())
                                              : json::array();
    if (!rows.empty() && beginListBox("##collider_list", rows.size())) {
        for (const auto& row : rows) {
            const bool ready = row.value("ready", true);
            ImGui::Text("%s  [%s%s]%s", row.value("object", std::string("-")).c_str(),
                        row.value("mode", std::string("?")).c_str(),
                        ready ? "" : ", cooking",
                        row.value("in_all_systems", true) ? "" : "  (not in all 4 systems)");
        }
    }
    if (!rows.empty()) ImGui::EndChild();
    if (rows.empty()) {
        ImGui::TextDisabled("No droplet collider: drops fly through everything.");
    }
    if (selected_object.empty()) {
        ImGui::TextDisabled("Select an object to give the droplets a collider on it.");
        return;
    }
    // Current managed collider of the selected object, if any.
    static const char* modes[] = {"none", "aabb", "obb", "mesh_sdf"};
    int current = 0;
    int sdf_res = 1;
    for (const auto& row : rows) {
        if (row.value("managed", false) && row.value("object", std::string()) == selected_object) {
            const std::string mode = row.value("mode", std::string("none"));
            for (int i = 0; i < 4; ++i) if (mode == modes[i]) current = i;
            sdf_res = row.value("sdf_resolution_mode", 1);
        }
    }
    int chosen = current;
    ImGui::Text("Collider for %s", selected_object.c_str());
    bool changed = ImGui::Combo("Shape##collider", &chosen, modes, 4);
    if (chosen == 3) {
        static const char* resolutions[] = {"32^3", "64^3", "128^3"};
        changed |= ImGui::Combo("SDF resolution", &sdf_res, resolutions, 3);
    }
    if (changed) {
        demoCommand("set_collider", json{{"object", selected_object}, {"mode", modes[chosen]},
                                         {"sdf_resolution_mode", sdf_res}});
    }
}

void drawDemo(const std::string& selected_object, const std::vector<std::string>& surfaces) {
    const bool has_surfaces = !surfaces.empty();
    const json status = demoCommand("status", json::object(), false);
    const bool prepared = status.is_object() && status.value("prepared", false);
    const bool running = status.is_object() && status.value("running", false);
    if (prepared) {
        syncDraft(status.value("config", json::object()));
        ImGui::Text("Prepared, %s | %s balloons | events %llu",
                    running ? "running" : "stopped",
                    status.value("visual_mode", std::string("?")).c_str(),
                    static_cast<unsigned long long>(status.value("events_total", uint64_t{0})));
    } else {
        ImGui::TextDisabled("Not prepared");
    }
    if (status.is_object() && !status.value("error", std::string()).empty()) {
        ImGui::TextWrapped("Demo: %s", status["error"].get<std::string>().c_str());
    }
    if (draft.visual_nodes.empty()) {
        ImGui::DragInt("Balloons", &draft.balloons, 0.1f, 3, 24);
    } else {
        ImGui::TextDisabled("Balloons: %zu (one per mesh, 3..24)", draft.visual_nodes.size());
    }
    ImGui::DragInt("Drops per balloon", &draft.particles_per_balloon, 8.0f, 32, 8192);
    ImGui::SliderFloat("Pop clumpiness", &draft.burst_clumpiness, 0.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("How the paint leaves a popped balloon (drawn anew every pop).\n"
                          "0 = even fine spray; 1 = uneven ligaments, clumps and heavy blobs.");
    }
    ImGui::SliderInt("Pop ligaments (max)", &draft.burst_jets_max, 3, 24);
    ImGui::DragInt("Seed", &draft.seed, 1.0f, 0, 2147483647);
    ImGui::DragFloatRange2("Pop interval (s)", &draft.interval_min, &draft.interval_max,
                           0.05f, 2.0f, 30.0f, "%.1f", "%.1f");
    ImGui::Checkbox("Start automatically", &draft.autostart);
    ImGui::BeginDisabled(draft.visual_nodes.empty());
    ImGui::Checkbox("Two-hand body balloon controls", &draft.body_controls);
    ImGui::EndDisabled();
    drawPalette(status);
    ImGui::DragFloatRange2("Balloon size", &draft.size_min, &draft.size_max, 0.01f, 0.3f, 2.0f,
                           "%.2fx", "%.2fx");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Each respawn picks a size in this range (x the mesh).\n"
                          "Paint drops scale with the surface (size^2); bigger pops sound lower.");
    }
    ImGui::DragFloat("Idle reset (s)", &draft.idle_reset_seconds, 1.0f, 0.0f, 3600.0f,
                     draft.idle_reset_seconds <= 0.0f ? "off" : "%.0f s");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Clean room after this long without a player pop (0 = off, min 10 s).\n"
                          "Player: F9 resets now. A game session resets at its end.");
    }
    drawPopSound();
    ImGui::SliderFloat("Surprise chance", &draft.surprise_chance, 0.0f, 0.5f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Share of balloons that come back as a gold surprise:\n"
                          "rainbow paint (all colours, 1.5x drops), confetti, its own sound.");
    }
    if (draft.surprise_chance > 0.0f) {
        drawSoundPicker("Surprise sound", draft.surprise_sound, "Pop sound, brighter");
    }
    ImGui::SliderInt("Sparkles", &draft.sparkles, 0, 400);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Firework sparks per surprise pop and per milestone burst (0 = off).\n"
                          "They fade slowly (never flash) and bounce off the room like the confetti.");
    }
    if (draft.sparkles > 0) {
        ImGui::Checkbox("Sparkle trails", &draft.sparkle_trails);
    }
    ImGui::Checkbox("Paint drop trails", &draft.drop_trails);
    ImGui::SliderFloat("Paint drop sparks", &draft.drop_sparks, 0.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Sparks where paint drops land (0 = off): chance per sampled contact.\n"
                          "Up to 64 contacts per droplet system are sampled each frame.");
    }
    ImGui::SeparatorText("Floating");
    ImGui::BeginDisabled(draft.visual_nodes.empty());
    ImGui::Checkbox("Jolt balloon motion", &draft.jolt_motion);
    ImGui::BeginDisabled(!draft.jolt_motion);
    ImGui::Checkbox("Simulate soft impact shells", &draft.impact_soft_body);
    ImGui::EndDisabled();
    if (!draft.visual_nodes.empty()) {
        ImGui::TextDisabled("Motion changes take effect when the demo is prepared again.");
    }
    ImGui::EndDisabled();
    if (status.contains("jolt_motion") && status["jolt_motion"].is_object()) {
        const auto& jolt = status["jolt_motion"];
        if (jolt.value("enabled", false)) {
            ImGui::Text("Jolt update %.2f ms (peak %.2f ms), shell step peak %.2f ms",
                        jolt.value("last_ms", 0.0), jolt.value("max_ms", 0.0),
                        jolt.value("shell_step_ms_max", 0.0));
        }
    }
    ImGui::BeginDisabled(draft.jolt_motion && !draft.visual_nodes.empty());
    ImGui::SliderFloat("Bob height (m)", &draft.float_amplitude, 0.0f, 0.5f, "%.2f");
    ImGui::SliderFloat("Drift speed (m/s)", &draft.drift_speed, 0.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Balloons wander slowly (big ones slower), keep off the walls and\n"
                          "each other. Mesh balloons only.");
    }
    ImGui::EndDisabled();
    bool spawn_auto = draft.spawn_auto;
    if (ImGui::Checkbox("Balloon area from the room", &spawn_auto)) {
        draft.spawn_auto = spawn_auto;
        if (!spawn_auto && status.is_object() && status.contains("spawn_min")) {
            for (int k = 0; k < 3; ++k) {   // start from what the room gave
                draft.spawn_min[k] = status["spawn_min"][k].get<float>();
                draft.spawn_max[k] = status["spawn_max"][k].get<float>();
            }
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Automatic: the paint surfaces' bounds, 0.9 m in from the walls,\n"
                          "0.8 m above the floor, 1 m under the ceiling.");
    }
    if (!draft.spawn_auto) {
        ImGui::DragFloat3("Area min", draft.spawn_min, 0.05f);
        ImGui::DragFloat3("Area max", draft.spawn_max, 0.05f);
    } else if (prepared && status.contains("spawn_min")) {
        ImGui::TextDisabled("Area y %.1f..%.1f m, x %.1f..%.1f, z %.1f..%.1f",
                            status["spawn_min"][1].get<float>(), status["spawn_max"][1].get<float>(),
                            status["spawn_min"][0].get<float>(), status["spawn_max"][0].get<float>(),
                            status["spawn_min"][2].get<float>(), status["spawn_max"][2].get<float>());
    }
    if (prepared) {
        ImGui::TextDisabled("Motion cost %.2f ms max per 30 Hz push",
                            status.value("motion_ms_max", 0.0));
    }

    ImGui::SeparatorText("Shared room goal");
    ImGui::SliderFloat("Room goal", &draft.room_goal, 0.0f, 1.0f,
                       draft.room_goal < 0.05f ? "off" : "%.2f of the walls");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Children fill the room together. At 25/50/75 %% of this goal: confetti\n"
                          "bursts; at 100 %%: confetti rain, every balloon pops in a cascade, then\n"
                          "the room cleans itself for the next group. No flashes.");
    }
    if (draft.room_goal >= 0.05f) {
        ImGui::DragFloat("Clean after finale (s)", &draft.finale_reset_seconds, 0.5f, 0.0f, 120.0f,
                         draft.finale_reset_seconds < 3.0f ? "never" : "%.0f s");
        drawSoundPicker("Celebrate sound", draft.celebrate_sound, "None");
        if (prepared) {
            const float progress = status.value("room_progress", 0.0f);
            char label[48];
            std::snprintf(label, sizeof(label), "%.0f %%  (%d/4)", progress * 100.0f,
                          status.value("milestones", 0));
            ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), label);
        }
    }
    drawBalloonMeshes(selected_object, surfaces);
    ImGui::BeginDisabled(!has_surfaces);
    if (ImGui::Button(prepared ? "Reconfigure" : "Prepare demo")) {
        // No surface list: the demo paints this scene's wet paint surfaces as set above.
        demoCommand("configure", draftConfig());
    }
    ImGui::EndDisabled();
    if (!has_surfaces && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Make at least one object a wet paint surface first.");
    }
    if (prepared) {
        ImGui::SameLine();
        if (ImGui::Button(running ? "Stop" : "Start")) {
            demoCommand(running ? "stop" : "start");
        }
        ImGui::SameLine();
        if (ImGui::Button("Pop one")) {
            demoCommand("burst", json{{"count", 1}});
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset room")) {
            demoCommand("reset");
        }
    }
}

} // namespace

void drawSection(const std::string& selected_object) {
    if (!UIWidgets::BeginSection("Game Design", ImVec4(0.55f, 0.75f, 0.95f, 1.0f), true)) {
        return;
    }
    rtapi::GameDesignInfo design;
    if (rtapi::getGameDesign(design).ok) {
        const std::string missing_surface = "flat paint surface not found: ";
        if (last_error.rfind(missing_surface, 0) == 0) {
            const auto object = last_error.substr(missing_surface.size());
            auto* context = rtapi::boundContext();
            if (context && (context->scene.isEditorPendingDeleteObjectName(object) ||
                            !context->scene.getFlatNodeMesh(object))) {
                // The shared service has removed this obsolete registration.
                last_error.clear();
            }
        }
        if (design.pending) {
            ImGui::TextDisabled("Loading the project's game design...");
        } else if (!design.status.empty()) {
            ImGui::TextWrapped("%s", design.status.c_str());
        }
    }
    ImGui::TextDisabled("Saved with the project. Paint itself is not saved.");

    // Play test: the same core as IPC game.playtest. The panel holds no state.
    {
        nlohmann::json play;
        if (rtapi::gamePlaytest("status", play).ok) {
            const bool active = play.value("active", false);
            if (ImGui::Button(active ? "Stop play test" : "Play in Studio")) {
                nlohmann::json ignored;
                const auto result = rtapi::gamePlaytest(active ? "stop" : "start", ignored);
                if (!result.ok) last_error = result.error;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Runs the room the way the Player does: live simulation, the demo\n"
                                  "started when it is set to start automatically, particles drawn as\n"
                                  "billboards only, and it keeps running when Studio is in the\n"
                                  "background. Stop returns to the previous simulation mode.");
            }
            if (active) {
                ImGui::SameLine();
                ImGui::TextDisabled("playing %.0f s", play.value("seconds", 0.0));
            }
        }
    }

    rtapi::WetPaintStatusInfo status;
    const bool have_status = rtapi::getWetPaintStatus(status).ok;
    ImGui::SeparatorText("Wet paint surfaces");
    // Start from the actual size; the selection is a request until Apply is pressed.
    static int resolution_index = 1;
    static bool resolution_touched = false;
    if (!resolution_touched && have_status && !status.surface_list.empty()) {
        const int existing = status.surface_list.front().resolution;
        resolution_index = existing <= 512 ? 0 : (existing <= 1024 ? 1 : 2);
    }
    std::vector<std::string> surface_names;
    if (have_status) {
        for (const auto& surface : status.surface_list) surface_names.push_back(surface.object);
    }
    const std::vector<rtapi::WetPaintSurfaceRuntimeInfo> no_surfaces;
    const auto& listed = have_status ? status.surface_list : no_surfaces;
    const auto materials = rtapi::listMaterials();
    const auto is_registered = [&](const std::string& object, int material_id) {
        return std::any_of(listed.begin(), listed.end(), [&](const auto& surface) {
            return surface.object == object && surface.material_id == material_id;
        });
    };
    const auto missing_materials = [&](const std::string& object) {
        std::vector<rtapi::MaterialInfo> missing;
        for (const auto& name : rtapi::objectMaterials(object)) {
            rtapi::MaterialInfo material;
            if (rtapi::getMaterial(name, material).ok && !is_registered(object, material.id)) {
                missing.push_back(std::move(material));
            }
        }
        return missing;
    };
    const auto add_missing = [](const std::string& object, const auto& missing) {
        for (const auto& material : missing) {
            if (!addPaintMaterial(object, material, 512 << resolution_index)) {
                break;
            }
        }
    };
    if (!listed.empty() && beginListBox("##surface_list", listed.size())) {
    for (const auto& surface : listed) {
        ImGui::PushID(surface.object.c_str());
        ImGui::PushID(surface.material_id);
        if (ImGui::SmallButton("x")) {
            report(rtapi::removeWetPaintSurface(surface.object, surface.material_id));
        }
        ImGui::SameLine();
        const auto material = std::find_if(materials.begin(), materials.end(), [&](const auto& m) {
            return m.id == surface.material_id;
        });
        const std::string label = surface.object + " / " +
            (material == materials.end() ? std::to_string(surface.material_id) : material->name);
        if (surface.object == selected_object) {
            ImGui::TextColored(ImGui::GetStyle().Colors[ImGuiCol_CheckMark], "%s (selected)",
                               label.c_str());
        } else {
            ImGui::TextUnformatted(label.c_str());
        }
        ImGui::PopID();
        ImGui::PopID();
    }
    }
    if (!listed.empty()) ImGui::EndChild();
    if (surface_names.empty()) {
        ImGui::TextDisabled("None yet. Droplets paint only these surfaces.");
    }
    const char* resolutions[] = {"512", "1024", "2048"};
    resolution_touched |= ImGui::Combo("Paint texture size", &resolution_index, resolutions, 3);
    if (!listed.empty()) {
        const int first = listed.front().resolution;
        const bool mixed = std::any_of(listed.begin(), listed.end(), [first](const auto& s) {
            return s.resolution != first;
        });
        if (mixed) {
            ImGui::TextDisabled("Current surfaces: mixed sizes");
        } else {
            ImGui::TextDisabled("Current surfaces: %d px", first);
        }
        ImGui::TextDisabled("Apply resizes all surfaces and clears wet paint.");
        if (ImGui::Button("Apply to all surfaces")) {
            report(rtapi::resizeWetPaintSurfaces(512 << resolution_index));
            UIWidgets::EndSection();
            return; // Refresh status on the next frame.
        }
    }
    ImGui::TextDisabled("New surfaces also use the selected size.");
    const bool selected_is_target = !selected_object.empty() &&
        std::find(draft.visual_nodes.begin(), draft.visual_nodes.end(), selected_object) ==
            draft.visual_nodes.end();
    const auto missing = selected_is_target ? missing_materials(selected_object)
                                            : std::vector<rtapi::MaterialInfo>{};
    ImGui::BeginDisabled(!have_status || missing.empty());
    const bool add_selected = ImGui::Button("Add missing slots");
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Copies shared materials for this object. Clears wet paint.");
    }
    if (add_selected) {
        add_missing(selected_object, missing);
        UIWidgets::EndSection();
        return;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1.0f);
    {
        // Balloon meshes cannot also be paint surfaces (the core refuses it).
        std::vector<std::string> exclude = surface_names;
        exclude.insert(exclude.end(), draft.visual_nodes.begin(), draft.visual_nodes.end());
        const std::string picked = pickSceneObject("##surface_pick", exclude);
        if (!picked.empty() && have_status) {
            add_missing(picked, missing_materials(picked));
            UIWidgets::EndSection();
            return;
        }
    }
    if (!surface_names.empty() && ImGui::Button("Clear paint")) {
        report(rtapi::resetWetPaint());
    }
    if (have_status && selected_is_target) {
        ImGui::SeparatorText("Selected object's paint slots");
        ImGui::PushID(selected_object.c_str());
        for (const auto& name : rtapi::objectMaterials(selected_object)) {
            rtapi::MaterialInfo material;
            if (!rtapi::getMaterial(name, material).ok) {
                continue;
            }
            const auto slot = std::find_if(listed.begin(), listed.end(), [&](const auto& surface) {
                return surface.object == selected_object && surface.material_id == material.id;
            });
            ImGui::PushID(material.id);
            if (slot == listed.end()) {
                if (ImGui::SmallButton("Add slot")) {
                    addPaintMaterial(selected_object, material, 512 << resolution_index);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Copies shared materials for this object. Clears wet paint.");
                }
                ImGui::SameLine();
                ImGui::TextUnformatted(name.c_str());
            } else if (ImGui::TreeNode("settings", "%s (%d px)", name.c_str(), slot->resolution)) {
                drawSurfaceSettings(*slot);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        ImGui::PopID();
    } else if (!surface_names.empty()) {
        ImGui::TextDisabled("Select a surface in the scene to edit its paint model.");
    }

    if (have_status && !status.surface_list.empty()) {
        ImGui::SeparatorText("Particle contacts");
        ImGui::PushID("splat");
        drawSplat(status.splat);
        ImGui::PopID();
    }

    ImGui::SeparatorText("Droplet colliders");
    ImGui::PushID("colliders");
    drawColliders(selected_object, demoCommand("status", json::object(), false));
    ImGui::PopID();

    ImGui::SeparatorText("Balloon demo");
    ImGui::PushID("demo");
    drawDemo(selected_object, surface_names);
    ImGui::PopID();

    if (!last_error.empty()) {
        ImGui::TextWrapped("Error: %s", last_error.c_str());
    }
    UIWidgets::EndSection();
}

} // namespace GameDesignUI
