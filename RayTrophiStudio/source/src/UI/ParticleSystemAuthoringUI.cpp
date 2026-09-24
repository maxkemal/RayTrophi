#include "UI/ParticleSystemAuthoringUI.h"

#include "ParticleSimulation.h"
#include "ProjectManager.h"
#include "scene_ui.h"

#include <algorithm>
#include <string>

namespace ParticleSystemAuthoringUI {
namespace {

int activeSystemIndex(const UIContext& ctx) {
    return ctx.scene.active_particle_system_index;
}

void selectNewSystem(UIContext& ctx, SceneData::ParticleSystemObject& system) {
    const int index = static_cast<int>(ctx.scene.particle_systems.size()) - 1;
    ctx.scene.setActiveParticleSystemObject(static_cast<std::size_t>(index));
    ctx.selection.selectParticleSystem(index, system.name);
}

} // namespace

void drawCreationBar(UIContext& ctx,
                     int& selected_emitter_index,
                     int& selected_collider_index,
                     int& selected_domain_index) {
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float button_width =
        std::max(1.0f, (ImGui::GetContentRegionAvail().x - spacing) * 0.5f);

    if (ImGui::Button("New Empty System##ParticleNewEmpty",
                      ImVec2(button_width, 28.0f))) {
        auto& system = ctx.scene.addParticleSystemObject();
        selectNewSystem(ctx, system);
        selected_emitter_index = -1;
        selected_collider_index = -1;
        selected_domain_index = -1;
        ctx.scene.clearSimFrameCache();
        ctx.scene.requestSimulationTimelineRenderResync();
        ProjectManager::getInstance().markModified();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Creates an independent particle system with no preset emitters, "
            "domains or colliders.");
    }

    ImGui::SameLine();
    if (ImGui::Button("Add Point Emitter##ParticleAddPoint",
                      ImVec2(button_width, 28.0f))) {
        if (!ctx.scene.activeParticleSystemObject()) {
            auto& system = ctx.scene.addParticleSystemObject();
            selectNewSystem(ctx, system);
        }

        auto runtime = ctx.scene.getParticleSimulationSystem();
        RayTrophiSim::ParticleEmitterDesc emitter;
        const std::size_t number = runtime ? runtime->emitters().size() + 1u : 1u;
        emitter.name = "Particle Emitter " + std::to_string(number);
        if (ctx.scene.camera) {
            emitter.point = ctx.scene.camera->lookat;
        }
        ctx.scene.addParticleEmitter(emitter);
        selected_emitter_index = static_cast<int>(
            ctx.scene.ensureParticleSimulationSystem().emitters().size()) - 1;
        selected_collider_index = -1;
        selected_domain_index = -1;
        ctx.selection.selectParticleEmitter(
            activeSystemIndex(ctx), selected_emitter_index, emitter.name);
        ctx.scene.clearSimFrameCache();
        ctx.scene.requestSimulationTimelineRenderResync();
        ProjectManager::getInstance().markModified();
        ctx.renderer.resetCPUAccumulation();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Adds a standalone point emitter to the active system. It can be "
            "positioned, aimed, keyed and rendered without applying a preset.");
    }
}

} // namespace ParticleSystemAuthoringUI
