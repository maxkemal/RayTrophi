// Status bar alert for held Matter steps.
//
// When the mixed step is held (the grain damping bound needs more substeps
// than the contact substep limit, among others) the grains freeze where they
// are. The only place that said so was the DEM line in the solver panel, so a
// user working in any other panel saw a simulation that had silently stopped.
// This draws a persistent pill in the status bar while any domain is held,
// fires one toast when a domain becomes held, and selects that domain in the
// Physics panel on click. The flag itself is also read over IPC
// (fluid.matter_models -> mixed_execution.step_held).
// A second, amber pill says when the grain step did NOT hold but softened its
// contact springs to fit the substep limit (grain_diagnostics.cfl_terms
// .stiffness_scale < 1): the scene runs, with softer grains than authored.
#include "scene_ui.h"
#include "scene_data.h"
#include "SceneSelection.h"
#include "ParticleSimulation.h"
#include "imgui.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

struct HeldDomain {
    int system_index = -1;
    int domain_index = -1;
    std::string name;
    std::string reason;
    float stiffness_scale = 1.0f;
    double withheld_kg_step = 0.0;
    double withheld_kg_total = 0.0;
    std::size_t particles = 0;
    std::size_t max_particles = 0;
};

enum class DomainAlert {
    Held,       // mixed step held: grains frozen
    Softened,   // running, contact springs softened to fit the substep limit
    Throttled   // liquid sources held back by the domain's Max Particles ceiling
};

std::vector<HeldDomain> collectHeldDomains(const SceneData& scene,
                                           DomainAlert alert = DomainAlert::Held) {
    std::vector<HeldDomain> held;
    for (std::size_t s = 0; s < scene.particle_systems.size(); ++s) {
        const auto& runtime = scene.particle_systems[s].runtime;
        if (!runtime) continue;
        const auto& domains = runtime->gridDomains();
        const auto& states = runtime->gridDomainStates();
        const std::size_t n = std::min(domains.size(), states.size());
        for (std::size_t d = 0; d < n; ++d) {
            const auto& state = states[d];
            const auto& stats = state.fluid_stats;
            if (!domains[d].enabled || !state.valid) continue;
            const bool hit =
                alert == DomainAlert::Held ? stats.mixed_step_held :
                alert == DomainAlert::Softened
                    ? !stats.mixed_step_held && stats.mixed_model_step &&
                      stats.grain_report.cfl.measured &&
                      stats.grain_report.cfl.stiffness_scale < 1.0f
                    : state.emission_withheld_kg_step > 0.0;
            if (!hit) continue;
            HeldDomain h;
            h.system_index = static_cast<int>(s);
            h.domain_index = static_cast<int>(d);
            h.name = domains[d].name;
            h.reason = stats.gpu_status;
            h.stiffness_scale = stats.grain_report.cfl.stiffness_scale;
            h.withheld_kg_step = state.emission_withheld_kg_step;
            h.withheld_kg_total = state.emission_withheld_kg_total;
            h.particles = state.particles.size();
            h.max_particles = domains[d].fluid_max_particles;
            held.push_back(std::move(h));
        }
    }
    return held;
}

std::string alertKey(const HeldDomain& h) {
    return std::to_string(h.system_index) + ":" + std::to_string(h.domain_index);
}

void openDomain(SceneUI& ui, UIContext& ctx, const HeldDomain& h) {
    ctx.scene.setActiveParticleSystemObject(static_cast<std::size_t>(h.system_index));
    ctx.selection.selectSimulationDomain(h.system_index, h.domain_index, h.name);
    const auto& domain =
        ctx.scene.particle_systems[static_cast<std::size_t>(h.system_index)]
            .runtime->gridDomains()[static_cast<std::size_t>(h.domain_index)];
    const Vec3 mn = Vec3::min(domain.bounds_min, domain.bounds_max);
    const Vec3 mx = Vec3::max(domain.bounds_min, domain.bounds_max);
    ctx.selection.selected.position = (mn + mx) * 0.5f;
    ctx.selection.selected.scale = mx - mn;
    ui.show_forcefield_tab = true;
    ui.tab_to_focus = "Simulation";
}

// A clickable status-bar pill; returns true when clicked.
bool statusPill(const char* id, const std::string& label, ImU32 fill, ImU32 hover_fill,
                const std::string& tooltip) {
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
    const ImVec2 size(text_size.x + 16.0f, 18.0f);
    ImGui::InvisibleButton(id, size);
    const bool hovered = ImGui::IsItemHovered();
    const bool clicked = ImGui::IsItemClicked();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), hovered ? hover_fill : fill, 9.0f);
    dl->AddText(ImVec2(p.x + 8.0f, p.y + 2.0f), IM_COL32(255, 255, 255, 255), label.c_str());
    if (hovered) ImGui::SetTooltip("%s", tooltip.c_str());
    // Same advance as the status bar's other pills.
    ImGui::SetCursorScreenPos(ImVec2(p.x + size.x + 6.0f, p.y));
    return clicked;
}

} // namespace

void SceneUI::drawSimulationHeldStatus(UIContext& ctx) {
    const std::vector<HeldDomain> held = collectHeldDomains(ctx.scene);

    // Toast on the transition only; the pill stays for as long as it is held.
    std::vector<std::string> keys;
    keys.reserve(held.size());
    for (const auto& h : held) {
        const std::string key = alertKey(h);
        if (std::find(held_domain_alert_keys.begin(), held_domain_alert_keys.end(), key) ==
            held_domain_alert_keys.end()) {
            addViewportMessage("Simulation stopped: " + h.name +
                                   " - grains are frozen. See the status bar.",
                               8.0f, ImVec4(1.0f, 0.42f, 0.35f, 1.0f));
        }
        keys.push_back(key);
    }
    held_domain_alert_keys = std::move(keys);

    if (!held.empty()) {
        const HeldDomain& first = held.front();
        std::string label = "SIM STOPPED: " + first.name;
        if (held.size() > 1) label += " (+" + std::to_string(held.size() - 1) + ")";
        std::string tip;
        for (const auto& h : held) {
            tip += h.name + ": " + h.reason + "\n";
        }
        tip += "\nGrains are frozen; heat, boiling and gas keep running.\n"
               "Click to open this domain in the Physics panel.";
        if (statusPill("##SimulationHeldStatus", label, IM_COL32(170, 45, 40, 255),
                       IM_COL32(200, 60, 50, 255), tip)) {
            openDomain(*this, ctx, first);
        }
    }

    // Softened, not held: no toast (it can change every frame), only the pill.
    const std::vector<HeldDomain> softened = collectHeldDomains(ctx.scene, DomainAlert::Softened);
    if (!softened.empty()) {
        const HeldDomain& first = *std::min_element(softened.begin(), softened.end(),
            [](const HeldDomain& a, const HeldDomain& b) {
                return a.stiffness_scale < b.stiffness_scale; });
        char scale[32];
        std::snprintf(scale, sizeof(scale), "%.2f", first.stiffness_scale);
        std::string label = std::string("GRAINS SOFTENED x") + scale + ": " + first.name;
        if (softened.size() > 1) label += " (+" + std::to_string(softened.size() - 1) + ")";
        const std::string tip =
            "The grain contact springs were softened this frame so the stability\n"
            "bound fits the Contact substep limit, instead of stopping the step.\n"
            "Grains overlap more than authored; bulk flow is usually unaffected.\n"
            "Raise the Contact substep limit to keep the authored stiffness.\n"
            "Click to open this domain in the Physics panel.";
        if (statusPill("##SimulationSoftenedStatus", label, IM_COL32(150, 105, 25, 255),
                       IM_COL32(180, 130, 35, 255), tip)) {
            openDomain(*this, ctx, first);
        }
    }

    // Liquid sources held back by Max Particles: a steady condition, so one
    // toast when it starts and the pill while it lasts.
    const std::vector<HeldDomain> throttled =
        collectHeldDomains(ctx.scene, DomainAlert::Throttled);
    std::vector<std::string> throttled_keys;
    for (const auto& h : throttled) {
        const std::string key = alertKey(h);
        if (std::find(throttled_domain_alert_keys.begin(), throttled_domain_alert_keys.end(),
                      key) == throttled_domain_alert_keys.end()) {
            addViewportMessage(h.name + " is at Max Particles: its sources now refill "
                               "what leaves instead of pouring their rate.",
                               8.0f, ImVec4(1.0f, 0.75f, 0.3f, 1.0f));
        }
        throttled_keys.push_back(key);
    }
    throttled_domain_alert_keys = std::move(throttled_keys);
    if (!throttled.empty()) {
        const HeldDomain& first = throttled.front();
        std::string label = "EMITTERS THROTTLED: " + first.name;
        if (throttled.size() > 1) label += " (+" + std::to_string(throttled.size() - 1) + ")";
        std::string tip;
        for (const auto& h : throttled) {
            char line[256];
            std::snprintf(line, sizeof(line),
                "%s: %zu / %zu particles; withheld %.4f kg last step, %.3f kg since reset\n",
                h.name.c_str(), h.particles, h.max_particles, h.withheld_kg_step,
                h.withheld_kg_total);
            tip += line;
        }
        tip += "\nThe domain is at its Max Particles ceiling (a resource limit).\n"
               "Its liquid sources no longer pour their authored rate: they\n"
               "refill whatever evaporates, burns or drains. Raise Max Particles,\n"
               "or bound the source with Limit Emitted Mass (kg).\n"
               "Click to open this domain in the Physics panel.";
        if (statusPill("##SimulationThrottledStatus", label, IM_COL32(150, 105, 25, 255),
                       IM_COL32(180, 130, 35, 255), tip)) {
            openDomain(*this, ctx, first);
        }
    }
}
