#pragma once

#include "ParticleSystemUsage.h"
#include "ProjectManager.h"
#include "scene_ui.h"

namespace ParticleUsageUI {

inline bool draw(UIContext& ctx) {
    const int active_index = ctx.scene.active_particle_system_index;
    if (active_index < 0 ||
        active_index >= static_cast<int>(ctx.scene.particle_systems.size())) {
        return true;
    }

    const std::size_t index = static_cast<std::size_t>(active_index);
    const auto& system = ctx.scene.particle_systems[index];
    int usage = system.render.emitter_only ? 0 : 1;
    const char* usage_names[] = {
        "Emitter Only (Gas / Fluid)",
        "Visible Particles"
    };

    if (ImGui::Combo("Usage##ParticleSystemUsage", &usage,
                     usage_names, IM_ARRAYSIZE(usage_names))) {
        ParticleSystemUsage::setEmitterOnly(ctx.scene, index, usage == 0);
        ProjectManager::getInstance().markModified();
        ctx.renderer.resetCPUAccumulation();
    }

    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Emitter Only keeps simulation particles active as gas/fluid sources, "
            "but hides those carrier particles from RayFusion, Solid and ray-traced "
            "renders. Gas/fluid domain output remains visible.\n\n"
            "Visible Particles enables the separate particle billboard/instance look.");
    }

    if (usage == 0) {
        ImGui::TextWrapped(
            "Carrier particles feed gas/fluid only. Particle billboards, RT geometry "
            "and particle materials are not created.");
    }
    return usage == 0;
}

} // namespace ParticleUsageUI
