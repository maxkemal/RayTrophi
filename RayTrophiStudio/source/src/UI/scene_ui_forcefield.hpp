/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_forcefield.hpp
* Author:        Kemal Demirtaş
* Date:          January 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
/**
 * @file scene_ui_forcefield.hpp
 * @brief UI Panel for Force Fields
 * 
 * Provides controls for:
 * - Force field list management
 * - Force field type and shape selection
 * - Strength, falloff, and noise parameters
 * - Per-system affect masks
 */

#pragma once

#include "scene_ui.h"
#include "ui_modern.h"
#include "ForceField.h"
#include "Backend/IViewportBackend.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "Material.h"
#include "TimelineWidget.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <cstdint>
#include <thread>
#include <functional>

namespace ForceFieldUI {

void drawSimulationDomainControls(
    SceneUI& ui,
    UIContext& ui_ctx,
    SceneData& scene,
    TimelineWidget* timeline,
    int& selected_domain_index,
    const std::function<void()>& drainSimulationMutationBackends,
    const std::function<void()>& clearForceFieldSelection,
    const std::function<void()>& drawSimBakeControls);

// Currently selected force field for UI
inline std::shared_ptr<Physics::ForceField> selected_force_field = nullptr;

// Edit-Mesh vertex selection snapshot, published by SceneUI each frame so the
// (free-function) Bodies panel can offer "Pin Selected Vertices" for cloth/soft
// bodies WITHOUT depending on the full SceneUI type. `active` is true only when
// Edit Mesh > Vertex mode is on for `object_name`; `world_positions` are the
// selected vertices in world space. See SceneUI::publishEditPinSelection().
struct EditPinSelectionSnapshot {
    bool active = false;
    std::string object_name;
    std::vector<Vec3> world_positions;
};
inline EditPinSelectionSnapshot g_edit_pin_selection;

// Total physical system RAM in bytes (0 on failure). Defined in scene_ui.cpp, where
// <windows.h> is available — this header is included before that include, so the
// query can't live here. Used to size the "RAM sim cache is large, bake to disk"
// nudge relative to the actual machine instead of a fixed threshold.
std::uint64_t queryTotalPhysicalRamBytes();

// Fluid baking state variables (defined at namespace scope to avoid MSVC lambda capture errors)
inline bool is_baking = false;
inline int current_bake_frame = 0;
inline float progress = 0.0f;
inline std::unique_ptr<std::thread> bake_thread = nullptr;
inline bool cancel_bake = false;

/**
 * @brief Draw the fluid material-preset combo. Applies physically-motivated
 *        rheology to @p params when a non-Custom preset is picked. Returns true
 *        only when a preset was actually applied (so the caller can re-render).
 *        Label/enum order is kept in sync with APICSolverParams::FluidPreset.
 */
inline bool drawFluidPresetCombo(const char* id, RayTrophiSim::Fluid::APICSolverParams& params) {
    using FluidPreset = RayTrophiSim::Fluid::APICSolverParams::FluidPreset;
    static const char* names[] = {
        "Custom (Manual)", "Water", "Oil", "Mud", "Honey", "Lava", "Sand"
    };
    bool applied = false;
    int idx = static_cast<int>(params.current_preset);
    if (idx < 0 || idx >= IM_ARRAYSIZE(names)) idx = 0;
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo(id, &idx, names, IM_ARRAYSIZE(names))) {
        FluidPreset chosen = static_cast<FluidPreset>(idx);
        if (chosen == FluidPreset::Custom) {
            params.current_preset = FluidPreset::Custom;
        } else {
            params.applyPreset(chosen);
            applied = true;
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Physically-motivated rheology presets. Overwrites viscosity, friction,\n"
            "FLIP/APIC blend, damping and packing only - domain, gravity, reseed and\n"
            "performance settings are kept.\n\n"
            "Water : thin, splashy Newtonian liquid.\n"
            "Oil   : mildly viscous, less splashy.\n"
            "Mud   : heavy dissipative slurry.\n"
            "Honey : very viscous, slow, sticky.\n"
            "Lava  : extreme viscosity (renderer adds the glow).\n"
            "Sand  : granular approximation - high friction + strong packing.\n"
            "        (liquid-solver approximation, not full MPM granular).");
    }
    return applied;
}

/**
 * @brief Draw the Force Field panel content
 */
inline void drawForceFieldPanel(SceneUI& ui, UIContext& ui_ctx, SceneData& scene, class TimelineWidget* timeline = nullptr) {
    auto& manager = scene.force_field_manager;

    static int simulation_section = 0;
    static int selected_domain_index = -1;

    auto clearForceFieldSelection = [&]() {
        selected_force_field = nullptr;
        if (ui_ctx.selection.selected.type == SelectableType::ForceField) {
            ui_ctx.selection.clearSelection();
        }
    };

    auto drainSimulationMutationBackends = [&]() {
        extern std::unique_ptr<Backend::IBackend> g_backend;
        extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

        int wait_count = 0;
        while (rendering_in_progress.load() && wait_count < 200) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            ++wait_count;
        }

        Backend::IBackend* renderBackend = g_backend
            ? g_backend.get()
            : ((ui_ctx.backend_ptr && dynamic_cast<Backend::IViewportBackend*>(ui_ctx.backend_ptr) == nullptr)
                ? ui_ctx.backend_ptr
                : nullptr);
        Backend::IViewportBackend* viewportBackend = g_viewport_backend
            ? g_viewport_backend.get()
            : dynamic_cast<Backend::IViewportBackend*>(ui_ctx.backend_ptr);

        if (renderBackend) {
            renderBackend->waitForCompletion();
        }
        if (viewportBackend && static_cast<Backend::IBackend*>(viewportBackend) != renderBackend) {
            viewportBackend->waitForCompletion();
        }
    };

    // ─── Global header: always-visible sim mode + GPU toggle. Was previously
    //     buried inside the per-domain inspector — moved here so the user can
    //     switch Live ↔ Timeline without clicking through to a domain.
    ImGui::SeparatorText("Simulation");
    {
        const char* sim_modes[] = { "Timeline (bake/scrub)", "Live (free-run)" };
        int sim_mode = g_sim_timeline_mode ? 0 : 1;
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::Combo("##SimGlobalMode", &sim_mode, sim_modes, IM_ARRAYSIZE(sim_modes))) {
            drainSimulationMutationBackends();
            g_sim_timeline_mode = (sim_mode == 0);
            ui_ctx.renderer.resetCPUAccumulation();
            if (ui_ctx.backend_ptr) {
                ui_ctx.backend_ptr->resetAccumulation();
            }
            ui_ctx.start_render = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Timeline: play to bake, scrub to replay; stopped = frozen.\nLive: continuous free-run preview (heavier).");
        }
    }

    ImGui::SeparatorText("Section");
    if (ImGui::BeginTabBar("##SimulationSectionTabs", ImGuiTabBarFlags_FittingPolicyResizeDown)) {
        if (ImGui::BeginTabItem("Fields"))    { simulation_section = 0; ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Particles")) { simulation_section = 1; ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Domains"))   { simulation_section = 2; ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Collision")) { simulation_section = 3; ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Bodies")) { simulation_section = 4; ImGui::EndTabItem(); }
        ImGui::EndTabBar();
    }
    ImGui::Separator();

    // Disk-bake controls (point/geo cache). Scene-wide: fluid particle systems AND
    // soft/cloth bodies are baked together into <project>.simcache. Shown in both
    // the Domains panel and the Bodies panel so a cloth-only scene (no fluid) can
    // still bake.
    auto drawSimBakeControls = [&](bool soft_only = false) {
        const std::string proj_path = ProjectManager::getInstance().getCurrentFilePath();
        const bool has_project = !proj_path.empty();
        const bool has_systems = !scene.particle_systems.empty() || !scene.rigid_bodies.empty();
        const bool can_bake = has_project && has_systems;
        const bool baking = scene.isSimulationBaking();

        if (baking) {
            const float frac = scene.simBakeProgress();
            char overlay[64];
            std::snprintf(overlay, sizeof(overlay), "Baking  frame %d / %d",
                          scene.simBakeCurrentFrame(), scene.simBakeEndFrame());
            ImGui::ProgressBar(frac, ImVec2(-1, 24), overlay);
            if (ImGui::Button("Cancel Bake##SimPointBakeCancel", ImVec2(-1, 24))) {
                scene.cancelSimulationDiskBake();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Stop the bake and discard the partial cache on disk.");
        } else {
            if (!can_bake) ImGui::BeginDisabled();
            if (ImGui::Button("Bake Simulation to Disk (point cache)##SimPointBake", ImVec2(-1, 30))) {
                const std::string dir = SceneData::simCacheDirForProject(proj_path);
                // Bake the TIMELINE range (single source of truth), not the
                // sequence-render range — a sim cache should cover the whole timeline
                // regardless of render output settings. Fall back to the render range
                // only if the timeline isn't wired (defensive).
                int s, e;
                if (timeline) {
                    s = std::min(timeline->getStartFrame(), timeline->getEndFrame());
                    e = std::max(timeline->getStartFrame(), timeline->getEndFrame());
                } else {
                    s = std::min(ui_ctx.render_settings.animation_start_frame,
                                 ui_ctx.render_settings.animation_end_frame);
                    e = std::max(ui_ctx.render_settings.animation_start_frame,
                                 ui_ctx.render_settings.animation_end_frame);
                }
                if (e <= s) { s = 0; e = 100; }
                const float fps = static_cast<float>(std::max(1, ui_ctx.render_settings.animation_fps));
                drainSimulationMutationBackends();
                if (!scene.beginSimulationDiskBake(dir, s, e, fps)) {
                    SCENE_LOG_INFO("Simulation point cache bake could not start (no systems / invalid range).");
                }
            }
            if (!can_bake) ImGui::EndDisabled();
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(has_project
                    ? "Re-simulates the timeline range and writes fluid particle systems\n"
                      "(particles + foam + gas) AND soft/cloth bodies to <project>.simcache.\n"
                      "Reloading the project restores the bake without re-simulating."
                    : "Save the project first — the cache is written next to the project file.");
            }
        }

        // RAM cache nudge: interactive preview keeps every frame in RAM, which balloons
        // with crowded/long sims. When it grows large, recommend baking to disk (then
        // scrubbing streams from disk and the RAM frame cache is bypassed).
        const bool active = soft_only ? scene.hasValidSoftSimDiskCache() : scene.hasValidParticleSimDiskCache();
        if (!active && !scene.hasValidSimDiskCache()) {
            const double cache_mb = scene.estimateSimCacheBytes() / (1024.0 * 1024.0);
            if (cache_mb >= 1.0) {
                const double total_ram_mb = queryTotalPhysicalRamBytes() / (1024.0 * 1024.0);
                // Size the warning to the actual machine: trip at ~20% of physical RAM
                // (256 MB floor so small machines still get a sensible nudge). Fall back
                // to a fixed ~1 GB if the RAM query failed.
                const double warn_mb = (total_ram_mb > 0.0) ? std::max(256.0, total_ram_mb * 0.20)
                                                            : 1024.0;
                const bool warn = cache_mb >= warn_mb;
                const ImVec4 col = warn ? ImVec4(1.0f, 0.55f, 0.15f, 1.0f)
                                        : ImVec4(0.65f, 0.65f, 0.65f, 1.0f);
                if (total_ram_mb > 0.0)
                    ImGui::TextColored(col, "RAM sim cache: ~%.0f MB / %.1f GB system (%d frames)%s",
                                       cache_mb, total_ram_mb / 1024.0, scene.cachedSimFrameCount(),
                                       warn ? "  - bake to disk" : "");
                else
                    ImGui::TextColored(col, "RAM sim cache: ~%.0f MB (%d frames)%s",
                                       cache_mb, scene.cachedSimFrameCount(),
                                       warn ? "  - bake to disk" : "");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Interactive preview holds every frame in RAM and grows with\n"
                                      "crowded/long sims. Bake to disk (above) — scrubbing then streams\n"
                                      "from disk and the RAM frame cache is bypassed.\n"
                                      "(Estimate covers soft/cloth + particles + rigid; excludes fluid grid.)");
            }
        }
        if (!baking && active) {
            if (soft_only) {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Soft body cache active.");
                ImGui::SameLine();
                if (ImGui::SmallButton("Clear Soft Cache##SimPointBakeClearSoft")) {
                    scene.clearSoftSimDiskCache();
                }
            } else {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Fluid cache active.");
                ImGui::SameLine();
                if (ImGui::SmallButton("Clear Fluid Cache##SimPointBakeClearFluid")) {
                    scene.clearParticleSimDiskCache();
                }
            }
        }
    };

    auto drawParticleControls = [&]() {
        static bool particle_ground_enabled = true;
        static float particle_ground_y = 0.0f;
        static float particle_restitution = 0.32f;
        static float particle_drag = 0.03f;
        static int selected_emitter_index = -1;
        static int selected_collider_index = -1;
        static std::string last_synced_selection_key;

        auto particles = scene.getParticleSimulationSystem();
        if (particles) {
            if (scene.pruneInvalidParticleObjectBindings() > 0) {
                selected_emitter_index = -1;
                selected_collider_index = -1;
            }
            particle_ground_enabled = particles->collisionPlaneEnabled();
            particle_ground_y = particles->collisionPlaneY();
            particle_restitution = particles->collisionRestitution();
            particle_drag = particles->linearDrag();
        }

        auto applyParticleTestSettings = [&]() {
            auto& system = scene.ensureParticleSimulationSystem();
            particle_drag = std::max(0.0f, particle_drag);
            particle_restitution = std::clamp(particle_restitution, 0.0f, 1.0f);
            system.setLinearDrag(particle_drag);
            system.setCollisionPlane(particle_ground_y, particle_ground_enabled, particle_restitution);
            scene.syncActiveParticleSystemObjectFromRuntime();
        };

        const int alive = particles ? static_cast<int>(particles->aliveCount()) : 0;
        const int capacity = particles ? static_cast<int>(particles->capacity()) : 0;
        const int emitter_count = particles ? static_cast<int>(particles->emitters().size()) : 0;
        const int collider_count = particles ? static_cast<int>(particles->colliders().size()) : 0;
        const int domain_count = particles ? static_cast<int>(particles->gridDomains().size()) : 0;

        ImGui::TextColored(ImVec4(0.08f, 0.58f, 0.98f, 1.00f), "Particle Systems");
        ImGui::SameLine();
        ImGui::TextDisabled("%d / %d alive | Emitters %d | Colliders %d | Domains %d",
                            alive, capacity, emitter_count, collider_count, domain_count);
        ImGui::Separator();

        const char* display_modes[] = { "Solid (Billboards)", "Debug (Overlay)", "Render (Preview)" };
        const float controls_width = ImGui::GetContentRegionAvail().x;
        const bool compact_particle_header = controls_width < 460.0f;
        if (!compact_particle_header) {
            ImGui::Columns(2, "ParticleOverviewColumns", false);
            ImGui::SetColumnWidth(0, std::max(220.0f, controls_width * 0.42f));
        }
        {
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (ImGui::Combo("Display Mode##PartDisp", &ui_ctx.particle_display_mode, display_modes, IM_ARRAYSIZE(display_modes))) {
                ui_ctx.start_render = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Sets the viewport visualization style for particle systems:\n\n"
                                  "1. Solid: Renders fast textured billboards/points.\n"
                                  "2. Debug: Renders vector overlays showing velocity, bounds, or particle IDs.\n"
                                  "3. Render: Reconstructs high-fidelity volumetric density previews.");
            }
            if (ui_ctx.particle_display_mode == 1) {
                ImGui::TextDisabled("Debug overlay draws over the viewport.");
            }
        }
        if (!compact_particle_header) {
            ImGui::NextColumn();
        }
        {
            // Preset picker. Applying REPLACES the active system's emitters,
            // domains and flow sources, so it stays behind an explicit Apply
            // rather than firing on every combo change.
            struct PresetEntry {
                const char* label;
                SceneData::ParticleSystemPreset preset;
                const char* help;
            };
            static const PresetEntry kPresets[] = {
                { "Campfire", SceneData::ParticleSystemPreset::Campfire,
                  "Hybrid Vulkan gas fire/smoke domain plus ray-traced ember particles.\n"
                  "Embers carry a little fuel, so they keep the plume alive as they drift." },
                { "Explosion (airburst)", SceneData::ParticleSystemPreset::Explosion,
                  "Short combustion/expansion pulse plus a ray-traced debris burst.\n"
                  "The shrapnel is burning: it drops fuel along its arc, so the fireball\n"
                  "spreads WITH the scatter instead of being a static ball." },
                { "Ground Burst", SceneData::ParticleSystemPreset::GroundBurst,
                  "Ground detonation: the floor clips the blast so it spreads wide, then climbs.\n"
                  "Adds a second heavy-dirt emitter that arcs and falls back." },
                { "Fireball (mushroom)", SceneData::ParticleSystemPreset::Fireball,
                  "Fuel-rich deflagration in a tall domain: slow burn plus strong thermal lift,\n"
                  "so the mass rolls upward into a mushroom instead of punching outward." },
                { "Smoke", SceneData::ParticleSystemPreset::Smoke,
                  "Grid-domain volumetric smoke with fine rising turbulence. No combustion." },
                { "Flamethrower", SceneData::ParticleSystemPreset::Flamethrower,
                  "Fast directional Vulkan fuel jet with a narrow turbulent flame tongue.\n"
                  "Aim it at a collider configured with Ignite on Contact to test surface fire." },
            };
            constexpr int kPresetCount = static_cast<int>(sizeof(kPresets) / sizeof(kPresets[0]));
            static int selected_preset = 0;
            selected_preset = std::clamp(selected_preset, 0, kPresetCount - 1);

            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            if (ImGui::BeginCombo("##ParticlePresetCombo", kPresets[selected_preset].label)) {
                for (int i = 0; i < kPresetCount; ++i) {
                    const bool is_selected = (i == selected_preset);
                    if (ImGui::Selectable(kPresets[i].label, is_selected)) {
                        selected_preset = i;
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("%s", kPresets[i].help);
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", kPresets[selected_preset].help);
            }
            if (ImGui::Button("Add Preset as New System##ParticlePresetApply",
                              ImVec2(ImGui::GetContentRegionAvail().x, 26))) {
                auto& added =
                    scene.addParticleSystemPreset(kPresets[selected_preset].preset);
                const int added_index =
                    static_cast<int>(scene.particle_systems.size()) - 1;
                scene.setActiveParticleSystemObject(
                    static_cast<std::size_t>(added_index));
                ui_ctx.selection.selectParticleSystem(added_index, added.name);
                selected_emitter_index = -1;
                selected_collider_index = -1;
                selected_domain_index = -1;
                ProjectManager::getInstance().markModified();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Creates a separate particle system with its own emitters, "
                    "domains and flow sources.\nExisting systems are preserved.");
            }
        }
        if (!compact_particle_header) {
            ImGui::Columns(1);
        }
        ImGui::Spacing();

        particles = scene.getParticleSimulationSystem();
        if (scene.particle_systems.empty()) {
            if (ImGui::Button("Add Particle System##PartAddSys", ImVec2(-1, 30))) {
                scene.addParticleSystemObject();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Creates and registers a new empty particle system container in the scene.");
            }
            return;
        }

        if (scene.active_particle_system_index < 0 ||
            scene.active_particle_system_index >= static_cast<int>(scene.particle_systems.size())) {
            scene.setActiveParticleSystemObject(0);
        }

        if (!ImGui::BeginTabBar("##ParticleAuthoringTabs", ImGuiTabBarFlags_FittingPolicyResizeDown)) {
            scene.syncActiveParticleSystemObjectFromRuntime();
            return;
        }

        // Group 2: System Selection & Appearance
        if (ImGui::BeginTabItem("System")) {

        const char* preview = "Select System...";
        if (const auto* active_system = scene.activeParticleSystemObject()) {
            preview = active_system->name.c_str();
        }
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::BeginCombo("##ActiveParticleSystemCombo", preview)) {
            for (int i = 0; i < static_cast<int>(scene.particle_systems.size()); ++i) {
                const bool selected = scene.active_particle_system_index == i;
                if (ImGui::Selectable(scene.particle_systems[static_cast<std::size_t>(i)].name.c_str(), selected)) {
                    scene.setActiveParticleSystemObject(static_cast<std::size_t>(i));
                    selected_emitter_index = -1;
                    selected_collider_index = -1;
                    selected_domain_index = -1;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        auto* active_obj = scene.activeParticleSystemObject();
        if (active_obj) {
            ImGui::Spacing();
            int blend = static_cast<int>(active_obj->blend_mode);
            const char* blend_names[] = { "Additive (Fire/Spark Glow)", "Alpha (Smoke/Dust Shadows)" };
            if (ImGui::Combo("Blend Mode##PartBlend", &blend, blend_names, IM_ARRAYSIZE(blend_names))) {
                active_obj->blend_mode = static_cast<SceneData::ParticleBlendMode>(blend);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Sets the rasterized viewport blend behavior:\n\n1. Additive: Spells bright glow for fiery particles.\n2. Alpha: Provides shadowing and transparency sorting for smoke/fog.");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            auto& rs = active_obj->render;
            ImGui::Checkbox("Render in Ray Tracing", &rs.render_in_raytrace);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Draws particles as authentic instanced 3D geometry in Vulkan RT or OptiX instead of simple flat billboards.");
            }

            if (rs.render_in_raytrace) {
                int shape = static_cast<int>(rs.shape);
                const char* shapes[] = { "Sphere Primitive", "Cube Primitive", "Tetrahedron Primitive", "Quad Plane", "Custom Scene Meshes (WIP)" };
                if (ImGui::Combo("Ray Trace Shape##PartRTShape", &shape, shapes, IM_ARRAYSIZE(shapes))) {
                    rs.shape = static_cast<SceneData::ParticleRenderShape>(shape);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Selects the geometric primitive to represent each individual particle in the ray-tracer.");
                }

                if (rs.shape == SceneData::ParticleRenderShape::Sphere) {
                    ImGui::SliderInt("Sphere Subdivisions", &rs.sphere_subdivisions, 0, 3);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Higher subdivision counts yield smoother spheres but increase ray-trace BVH traversal cost.");
                    }
                }
                ImGui::DragFloat("Size Multiplier", &rs.size_multiplier, 0.01f, 0.01f, 20.0f, "%.2f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Scales the visual representation size of instanced ray-traced particles.");
                }

                ImGui::Checkbox("Emissive Spark Glow", &rs.emissive);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Allows particles to act as light sources by emitting visual luminance.");
                }

                ImGui::Checkbox("Inherit Colors from Emitter", &rs.inherit_color_from_emitter);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Ray-traced particle colors will inherit their values from the first attached emitter's start/end color values.");
                }

                ImGui::BeginDisabled(rs.inherit_color_from_emitter);
                float cstart[3] = { rs.base_color.x, rs.base_color.y, rs.base_color.z };
                if (ImGui::ColorEdit3("Color Start##PartRTColS", cstart)) {
                    rs.base_color = Vec3(cstart[0], cstart[1], cstart[2]);
                }
                float cend[3] = { rs.color_end.x, rs.color_end.y, rs.color_end.z };
                if (ImGui::ColorEdit3("Color End##PartRTColE", cend)) {
                    rs.color_end = Vec3(cend[0], cend[1], cend[2]);
                }
                ImGui::EndDisabled();

                ImGui::SliderInt("Color Variations", &rs.color_buckets, 1, 32);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Defines how many distinct color steps are evaluated along the start-end color gradient to create particle variety.");
                }

                if (rs.emissive) {
                    ImGui::DragFloat("Emission Strength", &rs.emission_strength, 0.05f, 0.0f, 100.0f, "%.2f");
                } else {
                    ImGui::SliderFloat("Roughness##PartRTRough", &rs.roughness, 0.0f, 1.0f, "%.2f");
                }

                if (rs.shape == SceneData::ParticleRenderShape::SceneMeshes) {
                    ImGui::Separator();
                    ImGui::TextDisabled("Debris Meshes (Weighted List)");
                    const bool can_add =
                        ui_ctx.selection.selected.type == SelectableType::Object &&
                        ui_ctx.selection.selected.object != nullptr &&
                        !ui_ctx.selection.selected.object->getNodeName().empty();
                    
                    if (!can_add) ImGui::BeginDisabled();
                    if (ImGui::Button("Add Selected Mesh as Debris##PartAddDeb", ImVec2(-1, 24))) {
                        const std::string nn = ui_ctx.selection.selected.object->getNodeName();
                        bool exists = false;
                        for (const auto& m : rs.mesh_sources) {
                            if (m.node_name == nn) { exists = true; break; }
                        }
                        if (!exists) {
                            SceneData::ParticleRenderMeshSource ms;
                            ms.node_name = nn;
                            ms.weight = 1.0f;
                            rs.mesh_sources.push_back(ms);
                        }
                    }
                    if (ImGui::IsItemHovered() && can_add) {
                        ImGui::SetTooltip("Binds the selected scene object as an instanced mesh model representing individual debris shards.");
                    }
                    if (!can_add) ImGui::EndDisabled();

                    int remove_idx = -1;
                    for (int mi = 0; mi < static_cast<int>(rs.mesh_sources.size()); ++mi) {
                        ImGui::PushID(mi);
                        ImGui::SetNextItemWidth(120.0f);
                        ImGui::DragFloat("##w", &rs.mesh_sources[mi].weight, 0.01f, 0.0f, 100.0f, "%.2f");
                        ImGui::SameLine();
                        ImGui::TextUnformatted(rs.mesh_sources[mi].node_name.c_str());
                        ImGui::SameLine();
                        if (ImGui::SmallButton("x")) remove_idx = mi;
                        ImGui::PopID();
                    }
                    if (remove_idx >= 0) {
                        rs.mesh_sources.erase(rs.mesh_sources.begin() + remove_idx);
                    }
                    if (rs.mesh_sources.empty()) {
                        ImGui::TextDisabled("  (Select an object in scene list and click 'Add Selected Mesh')");
                    }
                }
            }
        }
            ImGui::EndTabItem();
        }

        // Group 3: Physics & Simulation Solver Settings
        if (ImGui::BeginTabItem("Physics")) {

        if (particles) {
            ImGui::Spacing();
            auto& physics = particles->physicsSettings();
            const char* physics_modes[] = { "Spark (Newtonian Ballistics)", "Granular (Rigid Sand Friction)", "Fluid (SPH Liquid)", "Gas (Buoyancy / Vortex)" };
            int physics_mode = static_cast<int>(physics.mode);
            if (ImGui::Combo("Physics Mode##PartPhysMode", &physics_mode, physics_modes, IM_ARRAYSIZE(physics_modes))) {
                const auto previous_mode = physics.mode;
                particles->applyPhysicsModePreset(static_cast<RayTrophiSim::ParticlePhysicsMode>(physics_mode));
                if (physics.mode != previous_mode) {
                    particles->resetGridDomainStates();
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Switches SPH solver presets:\n\n"
                                  "1. Spark: Simple Newtonian ballistic points.\n"
                                  "2. Granular: Rigid contacts with friction, simulating sand or debris.\n"
                                  "3. Fluid: SPH density-corrected liquid solver.\n"
                                  "4. Gas: Buoyancy and vorticity-driven smoke.");
            }

            const char* quality_modes[] = { "Realtime (Fast/Approximate)", "Preview (Balanced)", "Offline (Production/Precise)" };
            int quality_mode = static_cast<int>(physics.quality);
            if (ImGui::Combo("Solver Quality##PartQualMode", &quality_mode, quality_modes, IM_ARRAYSIZE(quality_modes))) {
                particles->applyQualityModePreset(static_cast<RayTrophiSim::ParticleQualityMode>(quality_mode));
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Sets neighborhood tracking grid precision and timestep subdivs. Offline is ideal for final bakes.");
            }

            ImGui::DragFloat("Particle Radius", &physics.particle_radius, 0.002f, 0.001f, 10.0f, "%.3f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("The collision and SPH interaction radius of single particles.");
            }

            ImGui::Checkbox("Inter-Particle Self Collision", &physics.self_collision_enabled);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Enables neighborhood checking and SPH collision resolving between particles.");
            }

            if (physics.self_collision_enabled) {
                ImGui::DragInt("Solver Iterations", &physics.solver_iterations, 1.0f, 1, 32);
                ImGui::DragInt("Max Neighbors / Part", &physics.max_neighbors_per_particle, 1.0f, 1, 256);
            }
            if (physics.mode == RayTrophiSim::ParticlePhysicsMode::Fluid ||
                physics.mode == RayTrophiSim::ParticlePhysicsMode::Granular ||
                physics.mode == RayTrophiSim::ParticlePhysicsMode::Gas) {
                ImGui::DragFloat("SPH Viscosity", &physics.viscosity, 0.01f, 0.0f, 10.0f, "%.3f");
            }
            if (physics.mode == RayTrophiSim::ParticlePhysicsMode::Fluid ||
                physics.mode == RayTrophiSim::ParticlePhysicsMode::Granular) {
                ImGui::DragFloat("Cohesion (Surface Tension)", &physics.cohesion, 0.01f, 0.0f, 10.0f, "%.3f");
                ImGui::DragFloat("Pressure Stiffness", &physics.pressure_stiffness, 0.01f, 0.0f, 100.0f, "%.3f");
                ImGui::DragFloat("Rest Density", &physics.rest_density, 1.0f, 0.001f, 10000.0f, "%.1f");
            }
            if (physics.mode == RayTrophiSim::ParticlePhysicsMode::Gas) {
                ImGui::DragFloat("Thermal Buoyancy", &physics.buoyancy, 0.01f, -100.0f, 100.0f, "%.3f");
                ImGui::DragFloat("Gravity Scale", &physics.gravity_scale, 0.01f, -10.0f, 10.0f, "%.3f");
                ImGui::DragFloat("Turbulent Vorticity", &physics.vorticity, 0.01f, 0.0f, 100.0f, "%.3f");
            }

            ImGui::Spacing();
            if (ImGui::CollapsingHeader("Gas Coupling (particles feed the grid)")) {
                ImGui::TextDisabled("Per-second deposit into any Gas domain a particle flies through.");
                ImGui::DragFloat("Smoke Deposit", &physics.grid_density_deposit,
                                 0.05f, 0.0f, 50.0f, "%.2f /s");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Density added to the grid along each particle path.\n"
                                      "This is the smoke trail behind flying debris.");
                }
                ImGui::DragFloat("Heat Deposit", &physics.grid_temperature_deposit,
                                 0.05f, 0.0f, 50.0f, "%.2f /s");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Temperature added along each particle path. Makes the gas\n"
                                      "glow and lift around a hot ember instead of ignoring it.");
                }
                ImGui::DragFloat("Fuel Deposit", &physics.grid_fuel_deposit,
                                 0.05f, 0.0f, 50.0f, "%.2f /s");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Fuel dropped along each particle path. This is what lets burning\n"
                                      "shrapnel IGNITE the gas it passes through, so the fireball spreads\n"
                                      "with the scatter. Needs the domain's Fuel channel + Combustion on.");
                }
                ImGui::Checkbox("Fade Deposit With Age", &physics.grid_deposit_fade_with_age);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Scale the deposit by remaining lifetime, so a cooling ember stops\n"
                                      "igniting things well before it disappears.");
                }
            }
        }
            ImGui::EndTabItem();
        }

        // Group 4: World Boundaries, Collisions, and Spawning
        if (ImGui::BeginTabItem("Actions")) {

        bool settings_changed = false;
        settings_changed |= ImGui::Checkbox("Enable Ground Plane", &particle_ground_enabled);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Enables an infinite collision plane at height Y.");
        }

        if (particle_ground_enabled) {
            settings_changed |= ImGui::DragFloat("Ground Plane Height (Y)", &particle_ground_y, 0.05f, -1000.0f, 1000.0f, "%.2f");
            settings_changed |= ImGui::SliderFloat("Bounce Restitution", &particle_restitution, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Elasticity of collisions against the ground. 0 = no bounce, 1 = fully elastic bounce.");
            }
        }
        settings_changed |= ImGui::DragFloat("Linear Drag Damping", &particle_drag, 0.005f, 0.0f, 10.0f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Friction damping representing air resistance, gradually slowing particles down over time.");
        }

        if (settings_changed && particles) {
            applyParticleTestSettings();
        }

        ImGui::Spacing();
        Vec3 spawn_center(0.0f, 1.0f, 0.0f);
        if (selected_force_field) {
            spawn_center = selected_force_field->position;
        } else if (ui_ctx.scene.camera) {
            spawn_center = ui_ctx.scene.camera->lookat;
        }

        if (ImGui::Button("Spawn Debug Burst (96 particles)##PartBurstBtn", ImVec2(-1, 28))) {
            scene.spawnDebugParticleBurst(spawn_center, 96, 0.25f, 2.5f, 5.0f);
            applyParticleTestSettings();
            scene.updateParticleSimulation(1.0f / 60.0f);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Instantly injects a rapid burst of SPH particles at camera focus point or active force field.");
        }
            ImGui::Spacing();
            ImGui::Separator();
            if (ImGui::Button("Clear Emitters Queue##PartEmitClr", ImVec2(-1, 30))) {
                scene.clearParticleEmitters();
                selected_emitter_index = -1;
            }
            if (ImGui::Button("Wipe All Active Particles##PartWipeClr", ImVec2(-1, 30))) {
                scene.clearParticles();
            }
            ImGui::EndTabItem();
        }

        // Group 5: Spawning Sources & Emitters
        if (ImGui::BeginTabItem("Emitters")) {

        const bool has_object_selection =
            ui_ctx.selection.selected.type == SelectableType::Object &&
            ui_ctx.selection.selected.object != nullptr &&
            !ui_ctx.selection.selected.object->getNodeName().empty();
        const std::string selected_object_name_for_actions =
            has_object_selection ? ui_ctx.selection.selected.object->getNodeName() : std::string();

        if (!selected_force_field) ImGui::BeginDisabled();
        if (ImGui::Button("Add Emitter from Force Field Selection##PartAddF", ImVec2(-1, 26))) {
            scene.addParticleEmitterFromForceField(selected_force_field);
            selected_emitter_index = static_cast<int>(scene.ensureParticleSimulationSystem().emitters().size()) - 1;
            applyParticleTestSettings();
            scene.updateParticleSimulation(1.0f / 60.0f);
        }
        if (ImGui::IsItemHovered() && selected_force_field) {
            ImGui::SetTooltip("Spawns particles bound to the coordinate location of the selected force field.");
        }
        if (!selected_force_field) ImGui::EndDisabled();

        if (!has_object_selection) ImGui::BeginDisabled();
        if (ImGui::Button("Add Emitter from Object Selection##PartAddO", ImVec2(-1, 26))) {
            scene.addParticleEmitterFromObject(selected_object_name_for_actions);
            selected_emitter_index = static_cast<int>(scene.ensureParticleSimulationSystem().emitters().size()) - 1;
            applyParticleTestSettings();
            scene.updateParticleSimulation(1.0f / 60.0f);
        }
        if (ImGui::IsItemHovered() && has_object_selection) {
            ImGui::SetTooltip("Spawns particles bound to the volume or surface AABB of the selected 3D mesh.");
        }
        if (!has_object_selection) ImGui::EndDisabled();

        particles = scene.getParticleSimulationSystem();
        if (particles && !particles->emitters().empty()) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Text("Emitters List:");
            auto& emitters = particles->emitters();
            if (selected_emitter_index >= static_cast<int>(emitters.size())) {
                selected_emitter_index = static_cast<int>(emitters.size()) - 1;
            }
            int emitter_to_remove = -1;

            if (ImGui::BeginTable("ParticleEmitterListTable", 2,
                                  ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Emitter");
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 28.0f);
                for (int i = 0; i < static_cast<int>(emitters.size()); ++i) {
                    ImGui::PushID(i);
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    if (ImGui::Selectable(emitters[i].name.c_str(), selected_emitter_index == i)) {
                        selected_emitter_index = i;
                    }
                    ImGui::TableSetColumnIndex(1);
                    if (ImGui::SmallButton("x")) {
                        emitter_to_remove = i;
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Remove emitter");
                    }
                    ImGui::PopID();
                }
                ImGui::EndTable();
            }
            if (emitter_to_remove >= 0) {
                particles->removeEmitter(static_cast<std::size_t>(emitter_to_remove));
                selected_emitter_index = std::min(emitter_to_remove, static_cast<int>(particles->emitters().size()) - 1);
            }

            if (selected_emitter_index >= 0 && selected_emitter_index < static_cast<int>(emitters.size())) {
                auto& emitter = emitters[static_cast<std::size_t>(selected_emitter_index)];
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Emitter Configuration:");
                
                ImGui::Checkbox("Emitter Enabled", &emitter.enabled);
                ImGui::TextDisabled("Source Binding: %s", emitter.source_name.empty() ? "Point Coordinate" : emitter.source_name.c_str());
                const bool has_gas_domain = std::any_of(
                    particles->gridDomains().begin(),
                    particles->gridDomains().end(),
                    [](const RayTrophiSim::SimulationGridDomainDesc& domain) {
                        return domain.enabled &&
                               domain.type == RayTrophiSim::SimulationDomainType::Gas;
                    });
                if (has_gas_domain) {
                    ImGui::TextColored(
                        ImVec4(0.45f, 0.8f, 1.0f, 1.0f),
                        "Gas coupling: particle trails deposit density/velocity into overlapping gas domains.");
                    ImGui::TextDisabled(
                        "Fuel and heat are authored by Flow Sources; particle-emitter channel mapping is not yet per-emitter.");
                } else {
                    ImGui::TextDisabled(
                        "Particle-only emitter. Add a Gas Domain + Flow Source for volumetric smoke/fire.");
                }
                
                const char* spawn_modes[] = { "Spawn Center Point", "Object AABB Surface", "Mesh Surface Geometry" };
                int spawn_mode = static_cast<int>(emitter.spawn_mode);
                if (ImGui::Combo("Spawn Geometry##PartEmitGeom", &spawn_mode, spawn_modes, IM_ARRAYSIZE(spawn_modes))) {
                    emitter.spawn_mode = static_cast<RayTrophiSim::ParticleEmitterSpawnMode>(spawn_mode);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Sets whether particles sprout from the source pivot point, surface shell bounds, or fine mesh geometry faces.");
                }

                ImGui::DragFloat("Spawn Rate / Sec", &emitter.rate_per_second, 1.0f, 0.0f, 100000.0f, "%.1f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Number of particles injected per second.");
                }
                ImGui::DragFloat("Initial Speed", &emitter.speed, 0.05f, 0.0f, 1000.0f, "%.2f");
                ImGui::DragFloat("Velocity Spread", &emitter.spread, 0.01f, 0.0f, 10.0f, "%.2f");
                if (emitter.spawn_mode == RayTrophiSim::ParticleEmitterSpawnMode::ObjectAABBSurface ||
                    emitter.spawn_mode == RayTrophiSim::ParticleEmitterSpawnMode::MeshSurface) {
                    ImGui::DragFloat("Surface Offset", &emitter.surface_offset, 0.005f, 0.0f, 100.0f, "%.3f");
                }
                ImGui::DragFloat("Particle Lifetime", &emitter.lifetime_seconds, 0.05f, 0.01f, 1000.0f, "%.2f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Duration in seconds before a particle is automatically culled.");
                }
                ImGui::DragFloat("Particle Mass", &emitter.mass, 0.05f, 0.0f, 1000.0f, "%.2f");
                ImGui::DragFloat3("Spit Direction", &emitter.direction.x, 0.05f, -100.0f, 100.0f, "%.2f");
                if (emitter.source_mode == RayTrophiSim::ParticleEmitterSourceMode::Point) {
                    ImGui::DragFloat3("World Position", &emitter.point.x, 0.05f,
                                      -10000.0f, 10000.0f, "%.2f");
                }
                ImGui::DragFloat3("Local Pivot Offset", &emitter.local_offset.x, 0.05f, -1000.0f, 1000.0f, "%.2f");

                if (ImGui::CollapsingHeader("Spawning Appearance Dynamics")) {
                    ImGui::DragFloat("Start Size", &emitter.start_size, 0.005f, 0.0f, 100.0f, "%.3f");
                    ImGui::DragFloat("End Size", &emitter.end_size, 0.005f, 0.0f, 100.0f, "%.3f");
                    ImGui::SliderFloat("Size Jitter", &emitter.size_jitter, 0.0f, 1.0f, "%.2f");
                    ImGui::SliderFloat("Start Opacity", &emitter.start_opacity, 0.0f, 1.0f, "%.2f");
                    ImGui::SliderFloat("End Opacity", &emitter.end_opacity, 0.0f, 1.0f, "%.2f");
                    ImGui::ColorEdit3("Start Color##EmitColS", &emitter.start_color.x);
                    ImGui::ColorEdit3("End Color##EmitColE", &emitter.end_color.x);
                    ImGui::DragFloat("Angular Velocity (rad/s)", &emitter.angular_velocity, 0.05f, -100.0f, 100.0f, "%.2f");
                    ImGui::DragFloat("Angular Velocity Jitter", &emitter.angular_jitter, 0.05f, 0.0f, 100.0f, "%.2f");
                }

                if (ImGui::Button("Trigger Burst (+128 particles)##PartBurstEmit", ImVec2(-1, 26))) {
                    emitter.burst_count += 128;
                    applyParticleTestSettings();
                    scene.updateParticleSimulation(1.0f / 60.0f);
                }
            }
        }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
        scene.syncActiveParticleSystemObjectFromRuntime();
    };

    auto drawFluidControls = [&]() {
        ImGui::TextDisabled("APIC liquid preview");
        if (ImGui::Button("Add Fluid", ImVec2(-1, 0))) {
            scene.addFluidObject("Fluid " + std::to_string(scene.fluid_objects.size() + 1));
        }

        if (scene.fluid_objects.empty()) {
            ImGui::TextDisabled("No fluid objects yet.");
            return;
        }

        if (scene.active_fluid_object_index < 0 ||
            scene.active_fluid_object_index >= static_cast<int>(scene.fluid_objects.size())) {
            scene.active_fluid_object_index = 0;
        }

        const char* preview = scene.activeFluidObject() ? scene.activeFluidObject()->name.c_str() : "Fluid";
        if (ImGui::BeginCombo("Active Fluid", preview)) {
            for (int i = 0; i < static_cast<int>(scene.fluid_objects.size()); ++i) {
                const bool selected = scene.active_fluid_object_index == i;
                if (ImGui::Selectable(scene.fluid_objects[static_cast<std::size_t>(i)].name.c_str(), selected)) {
                    scene.active_fluid_object_index = i;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        auto* fluid = scene.activeFluidObject();
        if (!fluid) {
            return;
        }

        char name_buf[128] = {};
        std::snprintf(name_buf, sizeof(name_buf), "%s", fluid->name.c_str());
        if (ImGui::InputText("Name##Fluid", name_buf, sizeof(name_buf))) {
            fluid->name = name_buf;
        }

        ImGui::Checkbox("Visible", &fluid->visible);
        ImGui::Checkbox("Enabled", &fluid->enabled);

        ImGui::SeparatorText("Disk Bake");
        drawSimBakeControls();

        // ── Render route -----------------------------------------------------
        // Volume     : APIC density splatted to NanoVDB (fog look — default).
        // Particles  : every APIC particle as an instanced sphere (debug/preview).
        // SurfaceSDF : narrow-band level set + isosurface in volume backend
        //              (Phase 3 — placeholder selectable; rendered as Volume
        //              until the SDF builder + isosurface path lands).
        {
            int current_mode_idx = 0; // default to Particles
            if (fluid->render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) {
                current_mode_idx = 1;
            } else if (fluid->render_mode == RayTrophiSim::Fluid::FluidRenderMode::Volume) {
                fluid->render_mode = RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
                current_mode_idx = 1;
            }
            const char* fluid_render_modes[] = { "Particles (Spheres)", "Surface SDF" };
            if (ImGui::Combo("Render Mode##Fluid", &current_mode_idx,
                             fluid_render_modes, 2)) {
                fluid->render_mode = (current_mode_idx == 0)
                    ? RayTrophiSim::Fluid::FluidRenderMode::Particles
                    : RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
                ui_ctx.start_render = true;
            }
            if (fluid->render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) {
                ImGui::ColorEdit3("Particle Color##Fluid", &fluid->particle_render_color.x);
                ImGui::DragFloat("Radius Factor##Fluid", &fluid->particle_render_radius_factor,
                                 0.01f, 0.05f, 1.5f, "%.2f");
                ImGui::DragFloat("Size Mult##Fluid", &fluid->particle_render_size_multiplier,
                                 0.01f, 0.05f, 8.0f, "%.2f");
                ImGui::SliderInt("Sphere Subdivs##Fluid", &fluid->particle_render_subdivisions, 0, 3);
                ImGui::Checkbox("Emissive##Fluid", &fluid->particle_render_emissive);
                if (fluid->particle_render_emissive) {
                    ImGui::DragFloat("Emission##Fluid", &fluid->particle_render_emission,
                                     0.05f, 0.0f, 50.0f, "%.2f");
                }
            }
            if (fluid->render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) {
                ImGui::DragFloat("Kernel Radius (vx)##Fluid",
                                 &fluid->level_set_params.kernel_radius_voxels,
                                 0.05f, 0.5f, 6.0f, "%.2f");
                ImGui::DragFloat("Particle Radius (vx)##Fluid",
                                 &fluid->level_set_params.particle_radius_voxels,
                                 0.02f, 0.05f, 2.0f, "%.2f");
                ImGui::DragFloat("Narrow Band (vx)##Fluid",
                                 &fluid->level_set_params.narrow_band_voxels,
                                 0.05f, 1.0f, 8.0f, "%.2f");
                ImGui::DragFloat("Surface Band (vx)##Fluid",
                                 &fluid->surface_band_voxels,
                                 0.02f, 0.1f, 3.0f, "%.2f");
                ImGui::SliderInt("Smoothing Sweeps##Fluid",
                                 &fluid->level_set_params.smoothing_iterations,
                                 0, 8);
                ImGui::SliderInt("Surface Detail (x sim grid)##Fluid",
                                 &fluid->level_set_params.surface_resolution_multiplier,
                                 1, 4);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Reconstructs the render surface on a grid finer than the simulation\n"
                                      "(1 = same, 2 = half-voxel, ...). Adds sub-voxel detail for wavy/rocky\n"
                                      "coastlines WITHOUT making the sim more expensive. Cost scales x^3 -\n"
                                      "keep modest on large domains. Surface shape is unchanged; only fineness.");
                }
                ImGui::TextDisabled("SDF: %zu active / %zu surface cells (%.2f ms)",
                                    fluid->level_set_stats.active_cells,
                                    fluid->level_set_stats.surface_cells,
                                    fluid->level_set_stats.build_ms);
                if (fluid->level_set_stats.eff_nx > 0 &&
                    fluid->level_set_params.surface_resolution_multiplier > 1) {
                    ImGui::TextDisabled("Surface grid: %dx%dx%d (refined)",
                                        fluid->level_set_stats.eff_nx,
                                        fluid->level_set_stats.eff_ny,
                                        fluid->level_set_stats.eff_nz);
                }
            }

            if (fluid->render_mode != RayTrophiSim::Fluid::FluidRenderMode::Particles) {
                if (!fluid->shader) {
                    fluid->shader = VolumeShader::createSmokePreset();
                    fluid->shader->name = (fluid->render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF)
                        ? "Liquid Surface (SDF)" : "Liquid Volume";
                }
                if (SceneUI::drawVolumeShaderUI(ui_ctx, fluid->shader, nullptr, nullptr)) {
                    scene.refreshFluidSurfaceMaterial();
                    ui_ctx.renderer.updateBackendGasVolumes(scene);
                    ui_ctx.renderer.resetCPUAccumulation();
                    if (ui_ctx.backend_ptr) ui_ctx.backend_ptr->resetAccumulation();
                    ui_ctx.start_render = true;
                }
            }
        }

        {
            int sim_mode = g_sim_timeline_mode ? 0 : 1;
            const char* sim_modes[] = { "Timeline (bake/scrub)", "Live Update (free-run)" };
            if (ImGui::Combo("Simulation Mode##Fluid", &sim_mode, sim_modes, IM_ARRAYSIZE(sim_modes))) {
                drainSimulationMutationBackends();
                g_sim_timeline_mode = (sim_mode == 0);
                ui_ctx.renderer.resetCPUAccumulation();
                if (ui_ctx.backend_ptr) {
                    ui_ctx.backend_ptr->resetAccumulation();
                }
                ui_ctx.start_render = true;
            }
        }

        ImGui::Separator();
        ImGui::Text("Domain");
        bool domain_changed = false;
        domain_changed |= ImGui::DragFloat3("Domain Min", &fluid->domain_min.x, 0.05f, -10000.0f, 10000.0f, "%.2f");
        domain_changed |= ImGui::DragFloat3("Domain Max", &fluid->domain_max.x, 0.05f, -10000.0f, 10000.0f, "%.2f");
        domain_changed |= ImGui::DragFloat("Voxel Size", &fluid->voxel_size, 0.002f, 0.005f, 10.0f, "%.3f");
        if (domain_changed) {
            fluid->voxel_size = std::max(0.005f, fluid->voxel_size);
        }
        int max_grid_cells_ui = static_cast<int>(std::min<size_t>(fluid->max_grid_cells, 64000000));
        if (ImGui::DragInt("Max Grid Cells", &max_grid_cells_ui, 100000.0f, 100000, 64000000)) {
            fluid->max_grid_cells = static_cast<size_t>(std::max(100000, max_grid_cells_ui));
        }
        const Vec3 domain_lo = Vec3::min(fluid->domain_min, fluid->domain_max);
        const Vec3 domain_hi = Vec3::max(fluid->domain_min, fluid->domain_max);
        const Vec3 domain_size = domain_hi - domain_lo;
        const float preview_voxel = std::max(0.005f, fluid->voxel_size);
        const int preview_nx = std::max(1, static_cast<int>(std::round(domain_size.x / preview_voxel)));
        const int preview_ny = std::max(1, static_cast<int>(std::round(domain_size.y / preview_voxel)));
        const int preview_nz = std::max(1, static_cast<int>(std::round(domain_size.z / preview_voxel)));
        const std::size_t preview_cells =
            static_cast<std::size_t>(preview_nx) *
            static_cast<std::size_t>(preview_ny) *
            static_cast<std::size_t>(preview_nz);
        const std::size_t runtime_cells =
            static_cast<std::size_t>(fluid->grid.nx) *
            static_cast<std::size_t>(fluid->grid.ny) *
            static_cast<std::size_t>(fluid->grid.nz);
        ImGui::TextDisabled("Preview: %dx%dx%d  Cells: %zu", preview_nx, preview_ny, preview_nz, preview_cells);
        ImGui::TextDisabled("Runtime: %dx%dx%d  Cells: %zu%s",
                            fluid->grid.nx,
                            fluid->grid.ny,
                            fluid->grid.nz,
                            runtime_cells,
                            fluid->grid_dirty ? "  (dirty)" : "");
        if (preview_cells > fluid->max_grid_cells) {
            ImGui::TextDisabled("Preview exceeds Max Grid Cells; rebuild will clamp voxel size.");
        }
        if (ImGui::Button("Rebuild Fluid Grid", ImVec2(-1, 0))) {
            fluid->particles.clear();
            fluid->grid.clear();
            fluid->grid_dirty = true;
            fluid->ensureGrid();
            fluid->stats = RayTrophiSim::Fluid::APICSolverStats{};
            ui_ctx.start_render = true;
        }

        ImGui::Separator();
        ImGui::Text("Seed Box");
        ImGui::DragFloat3("Seed Min", &fluid->seed_min.x, 0.05f, -10000.0f, 10000.0f, "%.2f");
        ImGui::DragFloat3("Seed Max", &fluid->seed_max.x, 0.05f, -10000.0f, 10000.0f, "%.2f");
        ImGui::SliderInt("Particles / Cell", &fluid->seed_particles_per_cell, 1, 16);
        float estimated_seed_voxel = preview_voxel;
        int estimated_seed_nx = preview_nx;
        int estimated_seed_ny = preview_ny;
        int estimated_seed_nz = preview_nz;
        if (preview_cells > fluid->max_grid_cells) {
            const double scale = std::cbrt(static_cast<double>(preview_cells) /
                                           static_cast<double>(fluid->max_grid_cells));
            estimated_seed_voxel = std::max(0.005f,
                                            static_cast<float>(static_cast<double>(estimated_seed_voxel) * scale));
            estimated_seed_nx = std::max(1, static_cast<int>(std::round(domain_size.x / estimated_seed_voxel)));
            estimated_seed_ny = std::max(1, static_cast<int>(std::round(domain_size.y / estimated_seed_voxel)));
            estimated_seed_nz = std::max(1, static_cast<int>(std::round(domain_size.z / estimated_seed_voxel)));
        }
        const std::size_t estimated_seed_particles = RayTrophiSim::Fluid::estimateSeedBoxParticleCount(
            domain_lo,
            estimated_seed_nx,
            estimated_seed_ny,
            estimated_seed_nz,
            estimated_seed_voxel,
            fluid->seed_min,
            fluid->seed_max,
            fluid->seed_particles_per_cell);
        int max_particles_ui = static_cast<int>(std::min<size_t>(fluid->max_particles, 10000000));
        if (ImGui::DragInt("Max Particles", &max_particles_ui, 1000.0f, 1000, 10000000)) {
            fluid->max_particles = static_cast<size_t>(std::max(1000, max_particles_ui));
        }
        ImGui::TextDisabled("Seed estimate: %zu particles", estimated_seed_particles);
        if (estimated_seed_voxel > preview_voxel + 1e-6f) {
            ImGui::TextDisabled("Estimate uses clamped rebuild voxel: %.3f", estimated_seed_voxel);
        }
        if (estimated_seed_particles > fluid->max_particles) {
            ImGui::TextDisabled("Seed will be capped by Max Particles.");
        }
        ImGui::Checkbox("Seed Replaces Existing", &fluid->replace_on_seed);
        if (ImGui::Button("Seed Fluid", ImVec2(-1, 0))) {
            fluid->grid_dirty = true;
            fluid->ensureGrid();
            if (fluid->replace_on_seed) {
                fluid->particles.clear();
                fluid->grid.clear();
                fluid->ensureGrid();
            }
            RayTrophiSim::Fluid::seedBox(fluid->particles,
                                         fluid->grid,
                                         fluid->seed_min,
                                         fluid->seed_max,
                                         fluid->seed_particles_per_cell,
                                         static_cast<uint32_t>(fluid->id) * 2654435761u,
                                         fluid->particles.size() < fluid->max_particles
                                             ? fluid->max_particles - fluid->particles.size()
                                             : 0u);
            fluid->pending_seed = false;
            fluid->stats = RayTrophiSim::Fluid::APICSolverStats{};
            scene.ensureFluidSimulationSystem();
            ui_ctx.start_render = true;
        }

        ImGui::Separator();
        ImGui::Text("Solver");
        if (drawFluidPresetCombo("Material Preset##FluidSolverPreset", fluid->params)) {
            ui_ctx.start_render = true;
        }
        // Manual edits to any preset-driven rheology field demote the dropdown
        // back to "Custom" so it no longer claims a material it no longer matches.
        bool solver_edited = false;
        ImGui::DragFloat3("Gravity", &fluid->params.gravity.x, 0.05f, -100.0f, 100.0f, "%.2f");
        ImGui::SliderInt("Pressure Iterations", &fluid->params.pressure_iterations, 1, 120);
        ImGui::DragFloat("Pressure Residual Target", &fluid->params.pressure_relative_residual, 1.0e-6f, 1.0e-8f, 1.0e-2f, "%.1e");
        ImGui::Checkbox("Pressure Layer B V-cycle", &fluid->params.pressure_multigrid_preconditioner);
        solver_edited |= ImGui::DragFloat("Density Correction", &fluid->params.density_correction, 0.05f, 0.0f, 10.0f, "%.2f");
        // SOR Omega is dead with PCG+MIC(0); kept in the struct for project
        // file backward compat, hidden here.
        solver_edited |= ImGui::DragFloat("APIC Affine", &fluid->params.apic_blend, 0.01f, 0.0f, 1.0f, "%.2f");
        solver_edited |= ImGui::DragFloat("FLIP Blend",  &fluid->params.flip_blend, 0.01f, 0.0f, 1.0f, "%.2f");
        ImGui::DragFloat("CFL", &fluid->params.cfl, 0.02f, 0.1f, 4.0f, "%.2f");
        ImGui::SliderInt("Max Substeps", &fluid->params.max_substeps, 1, 16);
        solver_edited |= ImGui::DragFloat("Max Velocity", &fluid->params.max_velocity, 1.0f, 1.0f, 5000.0f, "%.0f");
        solver_edited |= ImGui::DragFloat("Velocity Damping", &fluid->params.velocity_damping, 0.001f, 0.0f, 1.0f, "%.3f");
        solver_edited |= ImGui::DragFloat("Internal Friction", &fluid->params.internal_friction, 0.01f, 0.0f, 10.0f, "%.2f");
        solver_edited |= ImGui::DragFloat("Air Drag", &fluid->params.air_drag, 0.01f, 0.0f, 10.0f, "%.2f");
        solver_edited |= ImGui::DragFloat("Wall Damping", &fluid->params.wall_damping, 0.01f, 0.0f, 1.0f, "%.2f");
        solver_edited |= ImGui::DragFloat("Affine Damping", &fluid->params.affine_damping, 0.001f, 0.0f, 1.0f, "%.3f");
        ImGui::DragFloat("Max Affine", &fluid->params.max_affine, 1.0f, 0.0f, 1000.0f, "%.0f");
        if (solver_edited) {
            fluid->params.current_preset = RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Custom;
        }
        ImGui::Checkbox("Free Surface", &fluid->params.free_surface);
        ImGui::Checkbox("Ghost Fluid Method (GFM) Surface", &fluid->params.ghost_fluid_surface);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Ghost Fluid Method (GFM) models sub-cell pressure extrapolation at the air-fluid boundary to eliminate staircasing/aliasing.");
        }
        ImGui::Checkbox("Reseed Enabled", &fluid->params.reseed_enabled);
        if (fluid->params.reseed_enabled) {
            ImGui::DragInt("Reseed Target/Cell", &fluid->params.reseed_target_per_cell, 0.1f, 0, 64);
            ImGui::DragInt("Reseed Min/Cell",    &fluid->params.reseed_min_per_cell,    0.1f, 1, 32);
            ImGui::DragInt("Reseed Max/Cell",    &fluid->params.reseed_max_per_cell,    0.1f, 2, 64);
        }

        ImGui::Separator();
        ImGui::Text("Particles: %zu", fluid->particles.size());
        if (ImGui::CollapsingHeader("Fluid Stats", ImGuiTreeNodeFlags_DefaultOpen)) {
            const auto& stats = fluid->stats;
            ImGui::TextDisabled("Step: %.3f ms  Threads: %d  Substeps: %d",
                                stats.total_ms,
                                stats.cpu_threads,
                                stats.advect_substeps);
            ImGui::TextDisabled("Particles: %zu  Fluid Cells: %zu / %zu",
                                stats.particle_count,
                                stats.active_fluid_cells,
                                stats.grid_cell_count);
            ImGui::Columns(2, "FluidSolverStatsColumns", false);
            ImGui::TextDisabled("Forces"); ImGui::NextColumn();
            ImGui::TextDisabled("%.3f ms", stats.forces_ms); ImGui::NextColumn();
            ImGui::TextDisabled("P2G"); ImGui::NextColumn();
            ImGui::TextDisabled("%.3f ms", stats.p2g_ms); ImGui::NextColumn();
            ImGui::TextDisabled("Boundary"); ImGui::NextColumn();
            ImGui::TextDisabled("%.3f ms", stats.boundary_ms); ImGui::NextColumn();
            ImGui::TextDisabled("Pressure"); ImGui::NextColumn();
            ImGui::TextDisabled("%.3f ms", stats.pressure_ms); ImGui::NextColumn();
            ImGui::TextDisabled("G2P"); ImGui::NextColumn();
            ImGui::TextDisabled("%.3f ms", stats.g2p_ms); ImGui::NextColumn();
            ImGui::TextDisabled("Advect"); ImGui::NextColumn();
            ImGui::TextDisabled("%.3f ms", stats.advect_ms); ImGui::NextColumn();
            ImGui::Columns(1);
        }
        // ── Export & Baking ──────────────────────────────────────────────────
        if (false && UIWidgets::BeginSection("Export & Baking##FluidBake", ImVec4(1.0f, 0.5f, 0.2f, 1.0f), false)) {
            static char export_dir[256] = "";
            static bool export_success = false;
            static bool export_error = false;
            static std::string export_message;
            
            ImGui::Text("Bake / Output Directory:");
            ImGui::InputText("##dir_fluid_bake", export_dir, sizeof(export_dir));
            ImGui::SameLine();
            if (ImGui::Button("Browse##FluidBakeBrowseBtn")) {
                std::string path = SceneUI::selectFolderDialogW(L"Select Fluid Export Directory");
                if (!path.empty()) {
                    strncpy_s(export_dir, path.c_str(), sizeof(export_dir) - 1);
                }
            }
            
            ImGui::Separator();
            
            if (ImGui::Button("Export Current Frame (.vdb)##FluidExportFrame", ImVec2(-1, 30))) {
                if (strlen(export_dir) == 0) {
                    export_error = true;
                    export_message = "Please specify a directory first";
                } else {
                    int current_frame = timeline ? timeline->getCurrentFrame() : 0;
                    std::string full_path = std::string(export_dir) + "/" + fluid->name + "_" + std::to_string(current_frame) + ".vdb";
                    bool result = fluid->exportToVDB(full_path);
                    export_success = result;
                    export_error = !result;
                    export_message = result ? ("Saved: " + full_path) : "Export failed!";
                }
            }
            
            ImGui::Spacing();
            UIWidgets::ColoredHeader("Sequence Baking##FluidSequenceBake", ImVec4(1.0f, 0.6f, 0.4f, 1.0f));
            static int bake_start = 0, bake_end = 100;
            ImGui::DragInt("Start Frame##FluidBakeStart", &bake_start, 1, 0, 1000);
            ImGui::DragInt("End Frame##FluidBakeEnd", &bake_end, 1, 1, 1000);
            
            if (is_baking) {
                progress = static_cast<float>(current_bake_frame - bake_start) / std::max(1, (bake_end - bake_start));
                std::string progress_text = "Baking Frame: " + std::to_string(current_bake_frame) + " (" + std::to_string((int)(progress * 100)) + "%)";
                ImGui::ProgressBar(progress, ImVec2(-1, 0), progress_text.c_str());
                if (ImGui::Button("Cancel Bake##FluidCancelBake", ImVec2(-1, 0))) {
                    cancel_bake = true;
                }
            } else {
                if (ImGui::Button("Start Bake Sequence##FluidStartBake", ImVec2(-1, 35))) {
                    if (strlen(export_dir) == 0) {
                        export_error = true;
                        export_message = "Specify directory first!";
                    } else {
                        is_baking = true;
                        cancel_bake = false;
                        current_bake_frame = bake_start;
                        
                        std::string dir = export_dir;
                        auto f_obj = fluid;
                        int start_f = bake_start;
                        int end_f = bake_end;
                        
                        if (bake_thread && bake_thread->joinable()) bake_thread->join();
                        bake_thread = std::make_unique<std::thread>([dir, start_f, end_f, f_obj]() {
                            f_obj->resetState();
                            f_obj->ensureGrid();
                            
                            // Seed initial particles for sequence baking
                            RayTrophiSim::Fluid::seedBox(
                                f_obj->particles,
                                f_obj->grid,
                                f_obj->seed_min,
                                f_obj->seed_max,
                                f_obj->seed_particles_per_cell,
                                /*seed=*/static_cast<uint32_t>(f_obj->id) * 2654435761u,
                                f_obj->max_particles
                            );
                            
                            float dt = 1.0f / 24.0f; // Bake step dt
                            
                            std::string clean_dir = dir;
                            if (!clean_dir.empty() && (clean_dir.back() == '/' || clean_dir.back() == '\\')) {
                                clean_dir.pop_back();
                            }
                            std::filesystem::create_directories(clean_dir);
                            
                            for (int frame = start_f; frame <= end_f && !cancel_bake; ++frame) {
                                current_bake_frame = frame;
                                
                                if (frame > start_f) {
                                    RayTrophiSim::Fluid::step(
                                        f_obj->particles,
                                        f_obj->grid,
                                        f_obj->params,
                                        dt,
                                        /*force_snapshot=*/nullptr,
                                        /*time_seconds=*/(frame - start_f) * dt,
                                        &f_obj->stats
                                    );
                                }
                                
                                char filename[256];
                                sprintf_s(filename, "%s/%s_%04d.vdb", clean_dir.c_str(), f_obj->name.c_str(), frame);
                                f_obj->exportToVDB(filename);
                            }
                            is_baking = false;
                        });
                    }
                }
            }
            
            if (export_success) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "%s", export_message.c_str());
            } else if (export_error) {
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", export_message.c_str());
            }
            
            UIWidgets::EndSection();
        }

        if (UIWidgets::BeginSection("VDB Export##FluidObjectVDBExport", ImVec4(0.35f, 0.65f, 1.0f, 1.0f), false)) {
            static char vdb_export_dir[512] = "";
            static bool vdb_export_success = false;
            static bool vdb_export_error = false;
            static std::string vdb_export_message;

            ImGui::InputText("Directory##FluidObjectVDBDir", vdb_export_dir, sizeof(vdb_export_dir));
            ImGui::SameLine();
            if (ImGui::Button("Browse##FluidObjectVDBBrowse")) {
                const std::string path = SceneUI::selectFolderDialogW(L"Select VDB Export Directory");
                if (!path.empty()) {
                    std::snprintf(vdb_export_dir, sizeof(vdb_export_dir), "%s", path.c_str());
                }
            }

            const bool can_export_vdb = vdb_export_dir[0] != '\0';
            if (!can_export_vdb) ImGui::BeginDisabled();
            if (ImGui::Button("Export Current Frame (.vdb)##FluidObjectVDBFrame", ImVec2(-1, 28))) {
                std::error_code ec;
                std::filesystem::create_directories(vdb_export_dir, ec);
                const int current_frame = timeline ? timeline->getCurrentFrame() : 0;
                const std::string path =
                    (std::filesystem::path(vdb_export_dir) /
                     (fluid->name + "_" + std::to_string(current_frame) + ".vdb")).string();
                const bool result = fluid->exportToVDB(path);
                vdb_export_success = result;
                vdb_export_error = !result;
                vdb_export_message = result ? ("Saved: " + path) : "Export failed";
            }
            if (!can_export_vdb) ImGui::EndDisabled();
            if (vdb_export_success) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "%s", vdb_export_message.c_str());
            } else if (vdb_export_error) {
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", vdb_export_message.c_str());
            }
            UIWidgets::EndSection();
        }

        if (ImGui::Button("Reset Fluid", ImVec2(-1, 0))) {
            fluid->resetState();
        }
        if (ImGui::Button("Remove Fluid", ImVec2(-1, 0))) {
            const uint32_t id = fluid->id;
            scene.removeFluidObject(id);
        }
    };

    // ─── COLLIDERS TAB ─────────────────────────────────────────────────────
    // Global collider list. Used by both particle physics and the Fluid grid
    // voxelization. Full creation + editing lives here — Particles tab only
    // keeps a deprecation hint pointing here.
    // Dedicated Rigid Bodies (Jolt Physics) panel — ACTIVE dynamics, independent
    // of any sim domain (a rigid body falls/collides on its own via gravity +
    // static colliders). Lives in its own "Rigid Bodies" section tab so the list
    // gets the full panel width/height.
    auto drawRigidBodyControls = [&]() {
        const bool has_obj =
            ui_ctx.selection.selected.type == SelectableType::Object &&
            ui_ctx.selection.selected.object != nullptr &&
            !ui_ctx.selection.selected.object->getNodeName().empty();
        const std::string sel_name =
            has_obj ? ui_ctx.selection.selected.object->getNodeName() : std::string();

        ImGui::TextColored(ImVec4(0.98f, 0.62f, 0.10f, 1.00f), "Physics Bodies");
        ImGui::Separator();
        ImGui::TextDisabled("Rigid, soft & cloth bodies — no simulation domain required.");
        ImGui::Spacing();

        int sel_rb = -1;
        for (int i = 0; i < (int)scene.rigid_bodies.size(); ++i) {
            if (scene.rigid_bodies[i].source_name == sel_name) { sel_rb = i; break; }
        }

        // --- Creation buttons for the selected mesh ---
        if (!has_obj) ImGui::BeginDisabled();
        const float bw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
        if (ImGui::Button(sel_rb >= 0 ? "Reset as Rigid Body##RBAdd" : "Make Rigid Body##RBAdd", ImVec2(bw, 28))) {
            scene.addRigidBodyForObject(sel_name, /*dynamic=*/true);
        }
        if (ImGui::IsItemHovered() && has_obj)
            ImGui::SetTooltip("Selected mesh becomes a dynamic rigid body (falls, collides, tumbles).");
        ImGui::SameLine();
        if (ImGui::Button("Make Static Collider##RBStatic", ImVec2(bw, 28))) {
            scene.addRigidBodyForObject(sel_name, /*dynamic=*/false);
        }
        if (ImGui::IsItemHovered() && has_obj)
            ImGui::SetTooltip("Selected mesh becomes immovable collision geometry (ground/walls).");

        if (ImGui::Button("Make Soft Body##RBSoft", ImVec2(bw, 28))) {
            scene.addSoftBodyForObject(sel_name, RayTrophiSim::BodyKind::SoftBody);
        }
        if (ImGui::IsItemHovered() && has_obj)
            ImGui::SetTooltip("Selected mesh becomes a deformable soft body (falls, deforms, collides).");
        ImGui::SameLine();
        if (ImGui::Button("Make Cloth##RBCloth", ImVec2(bw, 28))) {
            scene.addSoftBodyForObject(sel_name, RayTrophiSim::BodyKind::Cloth);
        }
        if (ImGui::IsItemHovered() && has_obj)
            ImGui::SetTooltip("Selected mesh becomes cloth (surface soft body, two-sided, drapes & collides).");
        if (!has_obj) ImGui::EndDisabled();
        if (!has_obj) ImGui::TextDisabled("Select a mesh object above to add it as a physics body.");

        // ── Destruction: convex Voronoi pre-fracture (Faz 1, geometry only) ──
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Fracture (Destruction)")) {
            ImGui::TextDisabled("Splits the selected mesh into convex shards (Voronoi).");
            ImGui::TextDisabled("Faz 1: geometry only. Faz 2 makes shards rigid bodies.");
            const bool fractured = has_obj && ui.isMeshFractured(sel_name);
            if (!has_obj) ImGui::BeginDisabled();
            ImGui::SliderInt("Shards##frac", &ui.fracture_site_count, 2, 200);
            ImGui::InputInt("Seed##frac", &ui.fracture_seed);
            ImGui::Combo("Pattern##frac", &ui.fracture_pattern, "Uniform\0Impact-clustered\0");
            ImGui::SliderFloat("Preview Gap##frac", &ui.fracture_preview_gap, 0.0f, 0.3f, "%.3f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Shrinks shards toward their centre so the cuts are visible\n"
                                  "before physics. 0 = perfect tiling (looks intact).");
            const float fw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
            if (ImGui::Button(fractured ? "Re-Fracture##frac" : "Generate Shards##frac", ImVec2(fw, 28))) {
                ui.fractureSelectedMesh(ui_ctx, sel_name, ui.fracture_site_count,
                                        static_cast<uint32_t>(ui.fracture_seed), ui.fracture_pattern);
            }
            if (ImGui::IsItemHovered() && has_obj)
                ImGui::SetTooltip("Clip the mesh's convex hull into %d Voronoi shards.", ui.fracture_site_count);
            ImGui::SameLine();
            if (!fractured) ImGui::BeginDisabled();
            if (ImGui::Button("Restore##frac", ImVec2(fw, 28)))
                ui.unfractureMesh(ui_ctx, sel_name);
            if (!fractured) ImGui::EndDisabled();
            if (!has_obj) ImGui::EndDisabled();
            if (fractured) {
                ImGui::TextColored(ImVec4(0.55f, 0.90f, 0.55f, 1.0f), "Fractured into shards.");
                // ── Faz 2: shards → breakable rigid bodies ──
                ImGui::Separator();
                ImGui::TextDisabled("Destruction: shards become rigid bodies, intact until hit.");
                ImGui::SliderFloat("Break Threshold##frac", &ui.fracture_break_threshold,
                                   0.5f, 100.0f, "%.1f");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Impact impulse (kg·m/s) needed to shatter the object.\n"
                                      "Lower = fragile, higher = tough.");
                if (ImGui::Button("Make Breakable##frac", ImVec2(fw, 28))) {
                    auto sit = ui.fracture_shard_nodes_.find(sel_name);
                    if (sit != ui.fracture_shard_nodes_.end())
                        scene.makeFractureGroupBreakable(sel_name, sit->second,
                                                         ui.fracture_break_threshold);
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Register the shards as static rigid bodies that shatter\n"
                                      "into dynamic pieces when something hits them hard enough.");
                ImGui::SameLine();
                if (ImGui::Button("Break Now##frac", ImVec2(fw, 28)))
                    scene.breakFractureGroupNow(sel_name, 6.0f);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Shatter immediately (takes effect during playback).");
            }
        }

        ImGui::Spacing();
        ImGui::Text("Registered Bodies: %zu", scene.rigid_bodies.size());
        ImGui::Separator();
        ImGui::Spacing();

        // Shared handler: rewire the live solver + invalidate the bake cache so
        // any body edit takes effect on the next play.
        auto applyRigidBodyChange = [&]() {
            if (scene.rigid_body_system) {
                scene.rigid_body_system->resetRuntime(true);
                scene.rigid_body_system->setBodies(&scene.rigid_bodies);
            }
            scene.invalidateRigidBodySimulationCache();
        };
        // Unified body "type" across both axes (kind + rigid motion), so the list
        // shows ONE picker: Static / Dynamic / Kinematic / Soft Body / Cloth — the
        // exact set the user reasons about. Index 0..2 = Rigid + that motion type;
        // 3 = SoftBody, 4 = Cloth.
        static const char* kBodyTypeItems = "Static\0Dynamic\0Kinematic\0Soft Body\0Cloth\0";
        auto bodyTypeIndex = [](const RayTrophiSim::RigidBodyObject& body) -> int {
            if (body.kind == RayTrophiSim::BodyKind::SoftBody) return 3;
            if (body.kind == RayTrophiSim::BodyKind::Cloth)    return 4;
            switch (body.motion_type) {
                case RayTrophiSim::RigidBodyMotionType::Static:    return 0;
                case RayTrophiSim::RigidBodyMotionType::Kinematic: return 2;
                case RayTrophiSim::RigidBodyMotionType::Dynamic:
                default:                                           return 1;
            }
        };
        auto bodyTypeLabel = [&](const RayTrophiSim::RigidBodyObject& body) -> const char* {
            switch (bodyTypeIndex(body)) {
                case 0: return "Static";
                case 2: return "Kinematic";
                case 3: return "Soft Body";
                case 4: return "Cloth";
                case 1:
                default: return "Dynamic";
            }
        };
        // Apply a combined-type pick to a body: routes kind + motion + dynamic and
        // forces a Jolt rebuild. Returns true if anything changed.
        auto applyBodyType = [&](RayTrophiSim::RigidBodyObject& body, int idx) -> bool {
            RayTrophiSim::BodyKind new_kind = RayTrophiSim::BodyKind::Rigid;
            RayTrophiSim::RigidBodyMotionType new_motion = body.motion_type;
            switch (idx) {
                case 0: new_kind = RayTrophiSim::BodyKind::Rigid;    new_motion = RayTrophiSim::RigidBodyMotionType::Static; break;
                case 2: new_kind = RayTrophiSim::BodyKind::Rigid;    new_motion = RayTrophiSim::RigidBodyMotionType::Kinematic; break;
                case 3: new_kind = RayTrophiSim::BodyKind::SoftBody; new_motion = RayTrophiSim::RigidBodyMotionType::Dynamic; break;
                case 4: new_kind = RayTrophiSim::BodyKind::Cloth;    new_motion = RayTrophiSim::RigidBodyMotionType::Dynamic; break;
                case 1:
                default: new_kind = RayTrophiSim::BodyKind::Rigid;   new_motion = RayTrophiSim::RigidBodyMotionType::Dynamic; break;
            }
            if (body.kind == new_kind && body.motion_type == new_motion) return false;
            // Restore the mesh to rest using the CURRENT kind's cache BEFORE
            // switching kind. resetRuntime routes by the body's live `kind`; if
            // we change it first, the restore picks the new kind's (nonexistent)
            // cache and the mesh stays deformed.
            scene.restoreBodyMeshToRest(body.source_name, body.kind);
            body.kind = new_kind;
            body.motion_type = new_motion;
            body.dynamic = (new_kind != RayTrophiSim::BodyKind::Rigid) ||
                           (new_motion == RayTrophiSim::RigidBodyMotionType::Dynamic);
            body.created = false;
            body.rest_captured = false;  // force fresh rest capture for the new kind
            scene.syncRigidBodyProxyColliders();  // soft/cloth drop their rigid proxy
            return true;
        };

        int rb_to_remove = -1;
        int rb_to_apply = -1;   // "Apply at Frame": freeze current shape + drop the body
        std::string selection_request_name;  // set when a list row is clicked

        // --- Compact registry list (name | type | remove) ----------------
        // One row per body; selecting a row drives the viewport selection so the
        // list and 3D view stay in lockstep. The editor below targets only the
        // selected body, so the panel no longer grows one giant sub-panel per
        // object. Soft Body slots into the same list/editor when it lands.
        if (scene.rigid_bodies.empty()) {
            ImGui::TextDisabled("No rigid bodies yet. Select a mesh and click \"Make Rigid Body\".");
        } else if (ImGui::BeginTable("RigidBodyRegistryTable", 3,
                                     ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Body");
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 124.0f);
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 28.0f);
            for (int i = 0; i < (int)scene.rigid_bodies.size(); ++i) {
                auto& body = scene.rigid_bodies[i];
                ImGui::PushID(i);
                const bool is_sel = (i == sel_rb);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                if (ImGui::Selectable(body.source_name.c_str(), is_sel)) {
                    selection_request_name = body.source_name;
                }

                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                int row_type = bodyTypeIndex(body);
                if (ImGui::Combo("##rbrowtype", &row_type, kBodyTypeItems)) {
                    if (applyBodyType(body, row_type)) applyRigidBodyChange();
                }

                ImGui::TableSetColumnIndex(2);
                if (ImGui::SmallButton("x")) rb_to_remove = i;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove this body");
                ImGui::PopID();
            }
            ImGui::EndTable();
        }

        // List click -> viewport selection (bidirectional sync; viewport -> list
        // already happens because sel_rb is derived from the active selection).
        if (!selection_request_name.empty()) {
            bool sel_found = false;
            for (size_t i = 0; i < scene.world.objects.size(); ++i) {
                auto tri = std::dynamic_pointer_cast<Triangle>(scene.world.objects[i]);
                if (!tri) continue;
                std::string nn = tri->getNodeName();
                if (nn.empty()) nn = tri->getNodeName();
                if (nn == selection_request_name) {
                    ui_ctx.selection.selectObject(tri, (int)i, nn);
                    sel_found = true;
                    break;
                }
            }
            // Flat (direct SoA) body: no per-face facades in world.objects, so select via the
            // representative facade held in direct_mesh_nodes (the same handle normal selection uses).
            if (!sel_found) {
                auto dit = ui.direct_mesh_nodes.find(selection_request_name);
                if (dit != ui.direct_mesh_nodes.end() && dit->second.rep) {
                    ui_ctx.selection.selectObject(dit->second.rep, dit->second.object_index, selection_request_name);
                }
            }
            for (int i = 0; i < (int)scene.rigid_bodies.size(); ++i) {
                if (scene.rigid_bodies[i].source_name == selection_request_name) { sel_rb = i; break; }
            }
        }

        // --- Editor for the selected body --------------------------------
        ImGui::Spacing();
        if (sel_rb < 0) {
            ImGui::TextDisabled(scene.rigid_bodies.empty()
                ? "Add a body above to begin."
                : "Select a body (in the list or viewport) to edit its properties.");
        } else {
            auto& rb = scene.rigid_bodies[sel_rb];
            ImGui::PushID(sel_rb);
            ImGui::TextColored(ImVec4(0.98f, 0.62f, 0.10f, 1.00f), "%s  [%s]",
                               rb.source_name.c_str(), bodyTypeLabel(rb));
            ImGui::Separator();
            bool rb_changed = false;
            bool rb_rebuild = false;

            const bool is_rigid = (rb.kind == RayTrophiSim::BodyKind::Rigid);
            if (!ImGui::BeginTabBar("##RigidBodyAuthoringTabs", ImGuiTabBarFlags_FittingPolicyResizeDown)) {
                ImGui::PopID();
                return;
            }

                if (ImGui::BeginTabItem("Body")) {
                    int type_idx = bodyTypeIndex(rb);
                    ImGui::SetNextItemWidth(180);
                    if (ImGui::Combo("Type##rbtype", &type_idx, kBodyTypeItems)) {
                        if (applyBodyType(rb, type_idx)) {
                            rb_rebuild = true;
                            rb_changed = true;
                        }
                    }

                    ImGui::Checkbox("Enabled##rbenabled", &rb.enabled);
                    if (ImGui::IsItemEdited()) {
                        rb_rebuild = true;
                        rb_changed = true;
                    }

                    ImGui::TextDisabled("Collider: %s", rb.collider_name.empty() ? "Object bounds fallback" : rb.collider_name.c_str());

                    // Mass/density only drive the rigid path; soft bodies carry
                    // their own total mass in the Soft Body section below.
                    if (is_rigid) {
                        const bool is_dynamic = rb.motion_type == RayTrophiSim::RigidBodyMotionType::Dynamic;
                        ImGui::BeginDisabled(!is_dynamic);
                        if (ImGui::Checkbox("Auto Mass From Density##rbautomass", &rb.auto_mass_from_density)) {
                            rb_rebuild = true;
                            rb_changed = true;
                        }
                        ImGui::SetNextItemWidth(150);
                        if (ImGui::DragFloat("Density (kg/m3)##rbdensity", &rb.density, 5.0f, 0.1f, 20000.0f, "%.1f")) {
                            rb_rebuild = true;
                            rb_changed = true;
                        }
                        ImGui::BeginDisabled(rb.auto_mass_from_density);
                        ImGui::SetNextItemWidth(150);
                        if (ImGui::DragFloat("Mass (kg)##rbmass", &rb.mass, 0.1f, 0.0f, 100000.0f, "%.2f")) {
                            rb_rebuild = true;
                            rb_changed = true;
                        }
                        ImGui::EndDisabled();
                        ImGui::EndDisabled();
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Density below water (~1000 kg/m3) will tend to float once fluid coupling is solved.");
                        }
                    }
                    ImGui::EndTabItem();
                }

                // ---- Soft Body / Cloth section (deformable kinds) ----
                if (ImGui::BeginTabItem("Soft")) {
                    if (is_rigid) {
                        ImGui::TextDisabled("Soft-body controls apply to Soft Body and Cloth body types.");
                    } else {
                    ImGui::TextDisabled("Deformable body — falls, drapes & collides during play.");
                    ImGui::TextDisabled("Heavy: rebuilds mesh geometry each frame; no bake/scrub cache yet.");
                    ImGui::Spacing();

                    const bool is_cloth = (rb.kind == RayTrophiSim::BodyKind::Cloth);
                    float soft_stiffness = rb.getSoftStiffness();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("Stiffness##sbstiff", &soft_stiffness, 0.005f, 0.0f, 1.0f, "%.3f")) {
                        rb.setSoftStiffness(soft_stiffness);
                        rb_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Edge constraint stiffness: 0 = floppy, 1 = rigid-ish.");

                    float soft_compliance = rb.getSoftCompliance();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("Compliance##sbcompl", &soft_compliance, 0.0001f, 0.0f, 1.0f, "%.4f")) {
                        rb.setSoftCompliance(soft_compliance);
                        rb_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("XPBD inverse stiffness; 0 = fully stiff.");

                    float soft_damping = rb.getSoftDamping();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("Damping##sbdamp", &soft_damping, 0.005f, 0.0f, 1.0f, "%.3f")) {
                        rb.setSoftDamping(soft_damping);
                        rb_changed = true;
                    }

                    int soft_iterations = rb.getSoftIterations();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragInt("Iterations##sbiter", &soft_iterations, 0.1f, 1, 64)) {
                        rb.setSoftIterations(soft_iterations);
                        rb_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Constraint solver iterations per step (higher = stiffer/stabler, slower).");

                    float soft_mass = rb.getSoftMass();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("Total Mass (kg)##sbmass", &soft_mass, 0.05f, 0.001f, 100000.0f, "%.3f")) {
                        rb.setSoftMass(soft_mass);
                        rb_changed = true;
                    }

                    float soft_vertex_radius = rb.getSoftVertexRadius();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("Vertex Radius##sbvr", &soft_vertex_radius, 0.001f, 0.0f, 1.0f, "%.4f")) {
                        rb.setSoftVertexRadius(soft_vertex_radius);
                        rb_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Per-vertex collision thickness (m).");

                    bool soft_self_collision = rb.getSoftSelfCollision();
                    if (ImGui::Checkbox("Self Collision##sbselfcol", &soft_self_collision)) {
                        rb.setSoftSelfCollision(soft_self_collision);
                        rb_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                        "Push apart the body's own vertices closer than 2x Vertex Radius so cloth/soft "
                        "meshes stop folding through themselves. Jolt does not do this natively; extra "
                        "cost scales with vertex count and Iterations.");

                    ImGui::BeginDisabled(is_cloth);  // closed-volume pressure is meaningless for open cloth
                    float soft_pressure = rb.getSoftPressure();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("Pressure##sbpress", &soft_pressure, 0.05f, 0.0f, 1000.0f, "%.2f")) {
                        rb.setSoftPressure(soft_pressure);
                        rb_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Closed-volume inflation (balloons / soft solids). N/A for cloth.");
                    ImGui::EndDisabled();

                    float soft_friction = rb.getSoftFriction();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("Friction##sbfric", &soft_friction, 0.01f, 0.0f, 1.0f, "%.3f")) {
                        rb.setSoftFriction(soft_friction);
                        rb_changed = true;
                    }

                    float soft_restitution = rb.getSoftRestitution();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("Restitution##sbrest", &soft_restitution, 0.01f, 0.0f, 1.0f, "%.3f")) {
                        rb.setSoftRestitution(soft_restitution);
                        rb_changed = true;
                    }

                    float soft_gravity_factor = rb.getSoftGravityFactor();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("Gravity Factor##sbgrav", &soft_gravity_factor, 0.01f, -10.0f, 10.0f, "%.2f")) {
                        rb.setSoftGravityFactor(soft_gravity_factor);
                        rb_changed = true;
                    }

                    if (is_cloth) {
                        bool soft_two_sided = rb.getSoftTwoSided();
                        if (ImGui::Checkbox("Two-Sided Collision##sb2s", &soft_two_sided)) {
                            rb.setSoftTwoSided(soft_two_sided);
                            rb_changed = true;
                        }
                    }
                    if (rb_changed) rb_rebuild = true;

                    // ---- Pins: hold rest vertices fixed (hang cloth from corners) ----
                    ImGui::Spacing();
                    ImGui::SeparatorText("Pins");
                    ImGui::TextDisabled("Pinned vertices stay fixed in place during play.");

                    // Check if edit mode is active on this object
                    bool is_editing = ui.mesh_overlay_settings.enabled && ui.mesh_overlay_settings.edit_mode &&
                                      ui.mesh_workspace_mode == SceneUI::MeshWorkspaceMode::Edit &&
                                      ui.active_mesh_edit_object_name == rb.source_name &&
                                      ui_ctx.selection.mesh_element_mode == MeshElementSelectMode::Vertex;

                    if (ImGui::Checkbox("Edit Pin Selection##sbeditpin", &is_editing)) {
                        if (is_editing) {
                            // Ensure the object is selected in the viewport first
                            bool found = false;
                            for (size_t i = 0; i < scene.world.objects.size(); ++i) {
                                auto tri = std::dynamic_pointer_cast<Triangle>(scene.world.objects[i]);
                                if (!tri) continue;
                                std::string nn = tri->getNodeName();
                                if (nn.empty()) nn = tri->getNodeName();
                                if (nn == rb.source_name) {
                                    ui_ctx.selection.selectObject(tri, (int)i, nn);
                                    found = true;
                                    break;
                                }
                            }
                            // Flat (direct SoA) body: no facades — select via the representative facade
                            // (direct_mesh_nodes). Without this the Triangle-only scan never matched, so
                            // the edit workspace never activated and you couldn't enter vertex-pin mode
                            // (and a viewport click selected the OBJECT, appearing to "cancel" the mode).
                            if (!found) {
                                auto dit = ui.direct_mesh_nodes.find(rb.source_name);
                                if (dit != ui.direct_mesh_nodes.end() && dit->second.rep) {
                                    ui_ctx.selection.selectObject(dit->second.rep, dit->second.object_index, rb.source_name);
                                    found = true;
                                }
                            }
                            if (found) {
                                ui.activateEditWorkspace(ui_ctx);
                            }
                        } else {
                            ui.resetMeshEditState(ui_ctx);
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Toggle edit/selection mode for this object's vertices directly in the viewport.");
                    }

                    const bool edit_on_this =
                        ForceFieldUI::g_edit_pin_selection.active &&
                        ForceFieldUI::g_edit_pin_selection.object_name == rb.source_name &&
                        !ForceFieldUI::g_edit_pin_selection.world_positions.empty();
                    const int sel_vcount = edit_on_this
                        ? (int)ForceFieldUI::g_edit_pin_selection.world_positions.size() : 0;

                    static float pin_snap_radius = 0.05f;
                    ImGui::SetNextItemWidth(120);
                    ImGui::DragFloat("Pin Radius##sbpinr", &pin_snap_radius, 0.005f, 0.001f, 10.0f, "%.3f");
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Sphere radius used when pinning selected vertices (world units).");

                    ImGui::BeginDisabled(!edit_on_this);
                    if (ImGui::Button(sel_vcount > 0
                                          ? (std::string("Pin ") + std::to_string(sel_vcount) + " Selected Vertices##sbpinsel").c_str()
                                          : "Pin Selected Vertices##sbpinsel",
                                      ImVec2(-FLT_MIN, 26))) {
                        for (const Vec3& wp : ForceFieldUI::g_edit_pin_selection.world_positions) {
                            RayTrophiSim::SoftPinRegion pin;
                            pin.center = wp;
                            pin.radius = pin_snap_radius;
                            rb.getSoftPinsMut().push_back(pin);
                        }
                        rb_rebuild = true;
                        rb_changed = true;
                    }
                    ImGui::EndDisabled();
                    if (!is_editing) {
                        ImGui::TextDisabled("Toggle 'Edit Pin Selection' above to select vertices,");
                        ImGui::TextDisabled("then click the Pin button.");
                    } else if (sel_vcount == 0) {
                        ImGui::TextDisabled("Select vertices in viewport, then click Pin.");
                    }

                    // Pin region list: per-pin radius/enable/remove + add empty/clear.
                    int pin_to_remove = -1;
                    auto& pins = rb.getSoftPinsMut();
                    for (int pi = 0; pi < (int)pins.size(); ++pi) {
                        ImGui::PushID(pi);
                        auto& pin = pins[pi];
                        if (ImGui::Checkbox("##pinen", &pin.enabled)) { rb_rebuild = true; rb_changed = true; }
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(150);
                        if (ImGui::DragFloat3("##pinc", &pin.center.x, 0.01f, -1e6f, 1e6f, "%.3f")) { rb_rebuild = true; rb_changed = true; }
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(70);
                        if (ImGui::DragFloat("##pinr", &pin.radius, 0.005f, 0.001f, 100.0f, "r %.3f")) { rb_rebuild = true; rb_changed = true; }
                        ImGui::SameLine();
                        if (ImGui::SmallButton("x##pinrm")) pin_to_remove = pi;
                        ImGui::PopID();
                    }
                    if (pin_to_remove >= 0) {
                        pins.erase(pins.begin() + pin_to_remove);
                        rb_rebuild = true; rb_changed = true;
                    }
                    if (!pins.empty()) {
                        if (ImGui::SmallButton("Clear All Pins##sbpinclr")) {
                            pins.clear();
                            rb_rebuild = true; rb_changed = true;
                        }
                        ImGui::SameLine();
                        ImGui::TextDisabled("(%d pinned last play)", rb.dbg_pinned_count);
                    }

                    }
                    ImGui::EndTabItem();
                }

                // Motion / Axis Locks / Fluid Coupling are rigid-body concepts.
                if (is_rigid) {
                if (ImGui::BeginTabItem("Motion")) {
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("Linear Damping##rblinDamp", &rb.linear_damping, 0.01f, 0.0f, 20.0f, "%.3f")) {
                        rb_rebuild = true;
                        rb_changed = true;
                    }
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("Angular Damping##rbangDamp", &rb.angular_damping, 0.01f, 0.0f, 20.0f, "%.3f")) {
                        rb_rebuild = true;
                        rb_changed = true;
                    }
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("Gravity Scale##rbgrav", &rb.gravity_scale, 0.01f, -10.0f, 10.0f, "%.2f")) {
                        rb_rebuild = true;
                        rb_changed = true;
                    }

                    ImGui::SetNextItemWidth(-FLT_MIN);
                    if (ImGui::DragFloat3("Initial Velocity##rblinvel", &rb.initial_linear_velocity.x, 0.05f, -1000.0f, 1000.0f, "%.2f")) {
                        rb_rebuild = true;
                        rb_changed = true;
                    }
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    if (ImGui::DragFloat3("Initial Angular Velocity##rbangvel", &rb.initial_angular_velocity.x, 0.05f, -1000.0f, 1000.0f, "%.2f")) {
                        rb_rebuild = true;
                        rb_changed = true;
                    }

                    if (ImGui::Checkbox("Allow Sleeping##rbsleep", &rb.sleep_enabled)) {
                        rb_rebuild = true;
                        rb_changed = true;
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Locks")) {
                    ImGui::TextDisabled("Stored for the rigid solver roadmap; Jolt axis constraints land next.");
                    rb_changed |= ImGui::Checkbox("Lock X Translation##rbltx", &rb.lock_translation_x); ImGui::SameLine();
                    rb_changed |= ImGui::Checkbox("Lock Y Translation##rblty", &rb.lock_translation_y); ImGui::SameLine();
                    rb_changed |= ImGui::Checkbox("Lock Z Translation##rbltz", &rb.lock_translation_z);
                    rb_changed |= ImGui::Checkbox("Lock X Rotation##rblrx", &rb.lock_rotation_x); ImGui::SameLine();
                    rb_changed |= ImGui::Checkbox("Lock Y Rotation##rblry", &rb.lock_rotation_y); ImGui::SameLine();
                    rb_changed |= ImGui::Checkbox("Lock Z Rotation##rblrz", &rb.lock_rotation_z);
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Fluid")) {
                    rb_changed |= ImGui::Checkbox("Use Fluid Coupling##rbfluid", &rb.fluid_coupling_enabled);
                    ImGui::BeginDisabled(!rb.fluid_coupling_enabled);
                    
                    float fluid_density = rb.getFluidDensity();
                    if (ImGui::DragFloat("Fluid Density##rbfldens", &fluid_density, 5.0f, 0.1f, 20000.0f, "%.1f")) {
                        rb.setFluidDensity(fluid_density);
                        rb_changed = true;
                    }
                    
                    float buoyancy_scale = rb.getBuoyancyScale();
                    if (ImGui::DragFloat("Buoyancy Scale##rbbscale", &buoyancy_scale, 0.01f, 0.0f, 10.0f, "%.2f")) {
                        rb.setBuoyancyScale(buoyancy_scale);
                        rb_changed = true;
                    }
                    
                    float fluid_drag = rb.getFluidDrag();
                    if (ImGui::DragFloat("Fluid Drag##rbfdrag", &fluid_drag, 0.01f, 0.0f, 100.0f, "%.2f")) {
                        rb.setFluidDrag(fluid_drag);
                        rb_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Linear (viscous) drag: -k*v. Damps slow drift.");
                    
                    float fluid_quadratic_drag = rb.getFluidQuadraticDrag();
                    if (ImGui::DragFloat("Form Drag (slam)##rbfqdrag", &fluid_quadratic_drag, 0.01f, 0.0f, 100.0f, "%.2f")) {
                        rb.setFluidQuadraticDrag(fluid_quadratic_drag);
                        rb_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Quadratic form/slam drag: grows with speed (~v^2).\nThis is what stops a body skipping off the water surface; raise it if impacts bounce too much.");
                    
                    float fluid_angular_drag = rb.getFluidAngularDrag();
                    if (ImGui::DragFloat("Angular Fluid Drag##rbfadrag", &fluid_angular_drag, 0.01f, 0.0f, 100.0f, "%.2f")) {
                        rb.setFluidAngularDrag(fluid_angular_drag);
                        rb_changed = true;
                    }
                    
                    float fluid_max_coupling_speed = rb.getFluidMaxCouplingSpeed();
                    if (ImGui::DragFloat("Max Coupling Speed##rbfmaxspd", &fluid_max_coupling_speed, 0.1f, 0.0f, 50.0f, "%.1f m/s")) {
                        rb.setFluidMaxCouplingSpeed(fluid_max_coupling_speed);
                        rb_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Speed cap on the fluid velocity that drives drag.\nDamps the splash a plunging body stamps into the grid so it isn't flung sideways. 0 disables the clamp.");
                    
                    ImGui::EndDisabled();
                    // Always-visible float/sink verdict (the one number that
                    // explains most "won't float/sink" confusion). The rest of
                    // the per-step coupling telemetry is tucked under a collapsed
                    // Debug header so the panel stays clean.
                    if (rb.fluid_coupling_enabled && rb.dbg_coupled) {
                        const bool floats = rb.dbg_body_density < rb.getFluidDensity();
                        ImGui::TextColored(floats ? ImVec4(0.45f, 0.85f, 1.0f, 1.0f)
                                                  : ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                                           "Body %.0f kg/m3 vs fluid %.0f  (%s)",
                                           rb.dbg_body_density, rb.getFluidDensity(),
                                           floats ? "floats" : "sinks");
                        if (ImGui::TreeNodeEx("Coupling Debug##rbcpldbg", ImGuiTreeNodeFlags_None)) {
                            ImGui::Text("Submerged: %d / %d pts   sd_min %.3f m",
                                        rb.dbg_submerged_pts, rb.dbg_sample_count, rb.dbg_min_sd);
                            ImGui::Text("Buoy accel: %+.2f m/s2  (g = 9.81)", rb.dbg_buoy_accel_y);
                            ImGui::Text("Drag accel: %+.2f m/s2", rb.dbg_drag_accel_y);
                            ImGui::Text("Body vel Y: %+.3f m/s", rb.dbg_vel_y);
                            ImGui::TreePop();
                        }
                    }
                    ImGui::EndTabItem();
                }

                // ---- Force-field coupling (rigid bodies) ----
                if (ImGui::BeginTabItem("Forces")) {
                    const bool is_dynamic = rb.motion_type == RayTrophiSim::RigidBodyMotionType::Dynamic;
                    ImGui::BeginDisabled(!is_dynamic);
                    if (ImGui::Checkbox("Affected by Force Fields##rbffen", &rb.force_field_enabled)) rb_changed = true;
                    ImGui::BeginDisabled(!rb.force_field_enabled);
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::DragFloat("FF Influence##rbffscale", &rb.force_field_scale, 0.01f, 0.0f, 20.0f, "%.2f")) rb_changed = true;
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Per-body multiplier on scene force fields. Applied as a force at the body's center of mass.");
                    ImGui::EndDisabled();
                    ImGui::EndDisabled();
                    if (!is_dynamic) ImGui::TextDisabled("Only dynamic bodies react to force fields.");
                    ImGui::EndTabItem();
                }
                } // is_rigid (Motion / Axis Locks / Fluid Coupling / Force Fields)
                else {
                    if (ImGui::BeginTabItem("Fluid")) {
                        rb_changed |= ImGui::Checkbox("Use Fluid Coupling##rbfluid_soft", &rb.fluid_coupling_enabled);
                        ImGui::BeginDisabled(!rb.fluid_coupling_enabled);
                        
                        float fluid_density = rb.getFluidDensity();
                        if (ImGui::DragFloat("Fluid Density##rbfldens_soft", &fluid_density, 5.0f, 0.1f, 20000.0f, "%.1f")) {
                            rb.setFluidDensity(fluid_density);
                            rb_changed = true;
                        }
                        
                        float buoyancy_scale = rb.getBuoyancyScale();
                        if (ImGui::DragFloat("Buoyancy Scale##rbbscale_soft", &buoyancy_scale, 0.01f, 0.0f, 10.0f, "%.2f")) {
                            rb.setBuoyancyScale(buoyancy_scale);
                            rb_changed = true;
                        }
                        
                        float fluid_drag = rb.getFluidDrag();
                        if (ImGui::DragFloat("Fluid Drag##rbfdrag_soft", &fluid_drag, 0.01f, 0.0f, 100.0f, "%.2f")) {
                            rb.setFluidDrag(fluid_drag);
                            rb_changed = true;
                        }
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Linear (viscous) drag: -k*v. Damps slow drift.");
                        
                        float fluid_quadratic_drag = rb.getFluidQuadraticDrag();
                        if (ImGui::DragFloat("Form Drag (slam)##rbfqdrag_soft", &fluid_quadratic_drag, 0.01f, 0.0f, 100.0f, "%.2f")) {
                            rb.setFluidQuadraticDrag(fluid_quadratic_drag);
                            rb_changed = true;
                        }
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Quadratic form/slam drag: grows with speed (~v^2).");
                        
                        float fluid_angular_drag = rb.getFluidAngularDrag();
                        if (ImGui::DragFloat("Angular Fluid Drag##rbfadrag_soft", &fluid_angular_drag, 0.01f, 0.0f, 100.0f, "%.2f")) {
                            rb.setFluidAngularDrag(fluid_angular_drag);
                            rb_changed = true;
                        }
                        
                        float fluid_max_coupling_speed = rb.getFluidMaxCouplingSpeed();
                        if (ImGui::DragFloat("Max Coupling Speed##rbfmaxspd_soft", &fluid_max_coupling_speed, 0.1f, 0.0f, 50.0f, "%.1f m/s")) {
                            rb.setFluidMaxCouplingSpeed(fluid_max_coupling_speed);
                            rb_changed = true;
                        }
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Speed cap on the fluid velocity that drives drag. 0 disables the clamp.");
                        ImGui::EndDisabled();

                        if (rb.fluid_coupling_enabled && rb.dbg_coupled) {
                            const bool floats = rb.dbg_body_density < rb.getFluidDensity();
                            ImGui::TextColored(floats ? ImVec4(0.45f, 0.85f, 1.0f, 1.0f)
                                                      : ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                                               "Body %.0f kg/m3 vs fluid %.0f  (%s)",
                                               rb.dbg_body_density, rb.getFluidDensity(),
                                               floats ? "floats" : "sinks");
                            if (ImGui::TreeNodeEx("Coupling Debug##rbcpldbg_soft", ImGuiTreeNodeFlags_None)) {
                                ImGui::Text("Submerged: %d / %d pts   sd_min %.3f m",
                                            rb.dbg_submerged_pts, rb.dbg_sample_count, rb.dbg_min_sd);
                                ImGui::Text("Buoy accel: %+.2f m/s2  (g = 9.81)", rb.dbg_buoy_accel_y);
                                ImGui::Text("Drag accel: %+.2f m/s2", rb.dbg_drag_accel_y);
                                ImGui::Text("Body vel Y: %+.3f m/s", rb.dbg_vel_y);
                                ImGui::TreePop();
                            }
                        }
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("Forces")) {
                        if (ImGui::Checkbox("Affected by Force Fields##rbffen_soft", &rb.force_field_enabled)) rb_changed = true;
                        ImGui::BeginDisabled(!rb.force_field_enabled);
                        ImGui::SetNextItemWidth(150);
                        if (ImGui::DragFloat("FF Influence##rbffscale_soft", &rb.force_field_scale, 0.01f, 0.0f, 20.0f, "%.2f")) rb_changed = true;
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Per-body multiplier on scene force fields (wind/vortex/turbulence...).");
                        ImGui::EndDisabled();
                        ImGui::EndTabItem();
                    }
                }

                ImGui::EndTabBar();

                // A structural rebuild (shape / mass / soft params / pins) must reset
                // the body to its REST pose BEFORE the Jolt body is recreated. Setting
                // rb.created=false alone made the next ensureBodyCreated re-capture the
                // rest from the CURRENT (mid-sim, deformed) mesh — so editing a param at
                // frame 50 turned that frame's deformed shape into the new "rest" and
                // frame 0 stopped returning to the original. applyRigidBodyChange()
                // (resetRuntime → restore rest + invalidate cache → re-sim from frame 0)
                // keeps the rest clean while still re-simulating up to the current frame.
                if (rb_rebuild) rb_changed = true;
                if (rb_changed) applyRigidBodyChange();

                ImGui::Spacing();
                if (ImGui::SmallButton("Remove##rbrm")) rb_to_remove = sel_rb;
                ImGui::SameLine();
                if (ImGui::SmallButton("Apply at Frame##rbapply")) rb_to_apply = sel_rb;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Freeze the object at its CURRENT simulated shape and remove the body.\n"
                                      "The mesh keeps this pose permanently (no more simulation, no rest restore).\n"
                                      "Other bodies keep simulating. Stop at the frame you want first.");

                ImGui::PopID();
        }

        if (rb_to_remove >= 0 && rb_to_remove < (int)scene.rigid_bodies.size()) {
            scene.removeRigidBodyForObject(scene.rigid_bodies[rb_to_remove].source_name);
        }
        if (rb_to_apply >= 0 && rb_to_apply < (int)scene.rigid_bodies.size()) {
            // applyBodyAtCurrentFrame requests a SceneUI mesh/bbox cache rebuild
            // internally (this free-function panel can't touch SceneUI caches).
            scene.applyBodyAtCurrentFrame(scene.rigid_bodies[rb_to_apply].source_name);
        }

        // Disk bake (works without any fluid — soft/cloth + rigid bake here too).
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled("Simulation Bake (disk cache)");
        drawSimBakeControls(true);
    };

    auto drawColliderControls = [&]() {
        scene.ensureActiveParticleSystemObject();
        auto* p_sim = &scene.ensureParticleSimulationSystem();
        scene.syncRigidBodyProxyColliders();
        const uint64_t collider_sig_before = scene.computeSimConfigSignature();
        static int selected_collider_index_global = -1;

        ImGui::TextColored(ImVec4(0.08f, 0.58f, 0.98f, 1.00f), "Colliders");
        ImGui::SameLine();
        ImGui::TextDisabled("%zu registered", p_sim->colliders().size());
        ImGui::Separator();
        ImGui::TextDisabled("Shared by particles, fluids, and rigid bodies.");

        ImGui::Spacing();
        ImGui::TextDisabled("Add Primitive:");
        {
            const float pw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
            if (ImGui::Button("Add Sphere##CollAddSph", ImVec2(pw, 26))) {
                RayTrophiSim::ParticleColliderDesc desc;
                desc.name = "Sphere Collider";
                desc.source_mode = RayTrophiSim::ParticleColliderSourceMode::Sphere;
                desc.sphere_center = Vec3(0.0f, 1.0f, 0.0f);
                desc.sphere_radius = 1.0f;
                desc.restitution = 0.25f;
                desc.friction = 0.15f;
                desc.thickness = 0.02f;
                scene.addParticleCollider(desc);
                selected_collider_index_global = static_cast<int>(p_sim->colliders().size()) - 1;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Injects a basic 3D analytical sphere collider at origin.");
            }
            ImGui::SameLine();
            if (ImGui::Button("Add Plane Y##CollAddPl", ImVec2(pw, 26))) {
                RayTrophiSim::ParticleColliderDesc desc;
                desc.name = "Ground Plane";
                desc.source_mode = RayTrophiSim::ParticleColliderSourceMode::PlaneY;
                desc.plane_y = 0.0f;
                desc.restitution = 0.32f;
                desc.friction = 0.20f;
                desc.thickness = 0.02f;
                scene.addParticleCollider(desc);
                selected_collider_index_global = static_cast<int>(p_sim->colliders().size()) - 1;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Injects an infinite horizontal plane collision surface.");
            }
        }

        ImGui::Spacing();
        ImGui::TextDisabled("Add From Selection:");
        const bool has_object_selection =
            ui_ctx.selection.selected.type == SelectableType::Object &&
            ui_ctx.selection.selected.object != nullptr &&
            !ui_ctx.selection.selected.object->getNodeName().empty();
        const std::string selected_object_name =
            has_object_selection ? ui_ctx.selection.selected.object->getNodeName() : std::string();

        if (!has_object_selection) ImGui::BeginDisabled();
        if (ImGui::Button("Create Capsule/Box Proxy from Selection##CollProxy", ImVec2(-1, 26))) {
            scene.addParticleProxyColliderFromObject(selected_object_name);
            selected_collider_index_global = static_cast<int>(p_sim->colliders().size()) - 1;
        }
        if (ImGui::IsItemHovered() && has_object_selection) {
            ImGui::SetTooltip("Constructs a lightweight bounding capsule or box proxy encompassing the active scene mesh.");
        }
        if (!has_object_selection) ImGui::EndDisabled();

        auto findExistingObjectObbCollider = [&]() -> int {
            if (selected_object_name.empty()) return -1;
            const auto& list = p_sim->colliders();
            for (int i = 0; i < static_cast<int>(list.size()); ++i) {
                if (list[static_cast<std::size_t>(i)].source_name == selected_object_name &&
                    list[static_cast<std::size_t>(i)].source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectOBB) {
                    return i;
                }
            }
            return -1;
        };
        const int existing_obb_idx = findExistingObjectObbCollider();
        const bool obb_exists = existing_obb_idx >= 0;
        
        if (!has_object_selection || obb_exists) ImGui::BeginDisabled();
        if (ImGui::Button("Add Object OBB Collider##CollOBB", ImVec2(-1, 26))) {
            scene.addParticleColliderFromObject(selected_object_name);
            selected_collider_index_global = static_cast<int>(p_sim->colliders().size()) - 1;
        }
        if (ImGui::IsItemHovered() && has_object_selection && !obb_exists) {
            ImGui::SetTooltip("Attaches a high-fidelity Oriented Bounding Box collision volume tracing the selected scene mesh.");
        }
        if (!has_object_selection || obb_exists) ImGui::EndDisabled();

        auto& colliders = p_sim->colliders();
        if (selected_collider_index_global >= static_cast<int>(colliders.size())) {
            selected_collider_index_global = static_cast<int>(colliders.size()) - 1;
        }
        if (selected_collider_index_global < 0) selected_collider_index_global = 0;

        ImGui::Spacing();
        int collider_to_remove = -1;
        if (colliders.empty()) {
            ImGui::TextDisabled("No colliders yet.");
        } else if (ImGui::BeginTable("ColliderRegistryTable", 3,
                                     ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Collider");
            ImGui::TableSetupColumn("Enabled", ImGuiTableColumnFlags_WidthFixed, 68.0f);
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 28.0f);
            for (int i = 0; i < static_cast<int>(colliders.size()); ++i) {
                ImGui::PushID(i);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                if (ImGui::Selectable(colliders[i].name.c_str(), selected_collider_index_global == i)) {
                    selected_collider_index_global = i;
                }
                if (ImGui::BeginPopupContextItem()) {
                    selected_collider_index_global = i;
                    if (ImGui::MenuItem("Remove Collider")) collider_to_remove = i;
                    ImGui::EndPopup();
                }
                ImGui::TableSetColumnIndex(1);
                ImGui::Checkbox("##enabled", &colliders[i].enabled);
                ImGui::TableSetColumnIndex(2);
                if (ImGui::SmallButton("x")) collider_to_remove = i;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove collider");
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        
        if (collider_to_remove >= 0) {
            p_sim->removeCollider(static_cast<std::size_t>(collider_to_remove));
            scene.syncRigidBodyProxyColliders();
            scene.invalidateRigidBodySimulationCache();
            const int post = static_cast<int>(p_sim->colliders().size());
            selected_collider_index_global = (collider_to_remove < post) ? collider_to_remove : post - 1;
            return;
        }

        if (selected_collider_index_global < 0 ||
            selected_collider_index_global >= static_cast<int>(colliders.size())) {
            return;
        }
        auto& c = colliders[static_cast<std::size_t>(selected_collider_index_global)];

        // Group 2: Selected Collider Bindings
        if (ImGui::BeginTabBar("##ColliderAuthoringTabs", ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (ImGui::BeginTabItem("Binding")) {

            ImGui::Checkbox("Collider Enabled##CollTab", &c.enabled);
            ImGui::TextDisabled("Source Reference: %s", c.source_name.empty() ? "Manual Primitive" : c.source_name.c_str());

            // ObjectConvexDecomp / ObjectMeshBVH are deprecated: the SDF collider
            // (true signed field, filled interior, sub-grid weights, BVH cook)
            // supersedes both. Migrate any legacy collider to SDF on display and
            // drop them from the picker; the enum values remain for project load.
            if (c.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectConvexDecomp ||
                c.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectMeshBVH) {
                c.source_mode = RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF;
                if (!c.source_name.empty()) scene.rebuildSDFColliderAsync(c);
            }
            const char* modes[] = { "Plane Y Height", "Object AABB Volume", "Object OBB (Oriented)", "Sphere Primitive", "Capsule Primitive", "Object Mesh SDF (Voxel)" };
            int mode_idx = static_cast<int>(c.source_mode);
            if (ImGui::Combo("Collision Mode##CollTab", &mode_idx, modes, IM_ARRAYSIZE(modes))) {
                c.source_mode = static_cast<RayTrophiSim::ParticleColliderSourceMode>(mode_idx);
                if (c.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF && !c.source_name.empty()) {
                    scene.rebuildSDFColliderAsync(c);
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Sets the geometric tracking algorithm for this collider.");
            }

            const bool already_bound = has_object_selection && c.source_name == selected_object_name;
            const bool bind_disabled = !has_object_selection || already_bound;
            if (bind_disabled) ImGui::BeginDisabled();
            if (ImGui::Button("Bind Selected Object to Collider##CollBindBtn", ImVec2(-1, 24))) {
                c.source_name = selected_object_name;
                if (!c.source_name.empty()) {
                    c.name = c.source_name + " Collider";
                    scene.fitParticleColliderToObjectBounds(c, c.source_name, true);
                }
            }
            if (bind_disabled) ImGui::EndDisabled();

            const bool fit_disabled = !has_object_selection;
            if (fit_disabled) ImGui::BeginDisabled();
            if (ImGui::Button("Fit to Selected Object Bounds##CollFitBtn", ImVec2(-1, 24))) {
                scene.fitParticleColliderToObjectBounds(c, selected_object_name, already_bound);
            }
            if (fit_disabled) ImGui::EndDisabled();
            
            if (!c.source_name.empty() && ImGui::Button("Clear Object Binding Connection##CollClearBtn", ImVec2(-1, 24))) {
                c.source_name.clear();
            }
            ImGui::EndTabItem();
        }

        // Group 3: Geometric Properties
        if (ImGui::BeginTabItem("Geometry")) {

            if (c.source_mode == RayTrophiSim::ParticleColliderSourceMode::PlaneY) {
                ImGui::DragFloat("Plane Y Height##CollTab", &c.plane_y, 0.05f, -1000.0f, 1000.0f, "%.2f");
            } else if (c.source_mode == RayTrophiSim::ParticleColliderSourceMode::Sphere) {
                float center[3] = { c.sphere_center.x, c.sphere_center.y, c.sphere_center.z };
                const bool is_geom_disabled = !c.source_name.empty();
                if (is_geom_disabled) ImGui::BeginDisabled();
                if (ImGui::DragFloat3("Sphere Center##CollTab", center, 0.05f, -1000.0f, 1000.0f, "%.2f")) {
                    c.sphere_center = Vec3(center[0], center[1], center[2]);
                }
                ImGui::DragFloat("Sphere Radius##CollTab", &c.sphere_radius, 0.02f, 0.001f, 1000.0f, "%.3f");
                if (is_geom_disabled) ImGui::EndDisabled();
            } else if (c.source_mode == RayTrophiSim::ParticleColliderSourceMode::Capsule) {
                Vec3 center = (c.capsule_start + c.capsule_end) * 0.5f;
                Vec3 axis = c.capsule_end - c.capsule_start;
                float length = std::max(0.001f, axis.length());
                Vec3 direction = length > 1e-6f ? axis * (1.0f / length) : Vec3(0.0f, 1.0f, 0.0f);
                float center_values[3] = { center.x, center.y, center.z };
                float direction_values[3] = { direction.x, direction.y, direction.z };
                
                const bool is_geom_disabled = !c.source_name.empty();
                if (is_geom_disabled) ImGui::BeginDisabled();
                bool changed = false;
                if (ImGui::DragFloat3("Capsule Center##CollTab", center_values, 0.05f, -1000.0f, 1000.0f, "%.2f")) {
                    center = Vec3(center_values[0], center_values[1], center_values[2]); changed = true;
                }
                if (ImGui::DragFloat3("Capsule Direction##CollTab", direction_values, 0.01f, -1.0f, 1.0f, "%.2f")) {
                    direction = Vec3(direction_values[0], direction_values[1], direction_values[2]); changed = true;
                }
                ImGui::DragFloat("Capsule Length##CollTab", &length, 0.05f, 0.001f, 1000.0f, "%.3f");
                if (changed) {
                    const float dlen = direction.length();
                    direction = dlen > 1e-6f ? direction * (1.0f / dlen) : Vec3(0.0f, 1.0f, 0.0f);
                    const Vec3 half = direction * (length * 0.5f);
                    c.capsule_start = center - half;
                    c.capsule_end   = center + half;
                }
                ImGui::DragFloat("Capsule Radius##CollTab", &c.capsule_radius, 0.02f, 0.001f, 1000.0f, "%.3f");
                if (is_geom_disabled) ImGui::EndDisabled();
            } else if (c.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectAABB ||
                       c.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectOBB) {
                if (c.source_name.empty()) {
                    ImGui::TextDisabled("Please bind a scene geometry reference above to track bounds.");
                } else {
                    ImGui::TextDisabled("Transform bounds are live-fitted dynamically from '%s'.", c.source_name.c_str());
                }
            } else if (c.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF) {
                if (c.source_name.empty()) {
                    ImGui::TextDisabled("Please bind a scene geometry reference above.");
                } else {
                    ImGui::Text("Mesh Reference: %s", c.source_name.c_str());
                    
                    const char* resolutions[] = { "Low (32x32x32)", "Medium (64x64x64)", "High (128x128x128)" };
                    if (ImGui::Combo("SDF Resolution##CollTab", &c.sdf_resolution_mode, resolutions, IM_ARRAYSIZE(resolutions))) {
                        scene.rebuildSDFColliderAsync(c);
                    }
                    
                    if (ImGui::Button("Force Rebuild SDF Grid##CollTabRec", ImVec2(-1, 24))) {
                        scene.rebuildSDFColliderAsync(c);
                    }
                    
                    ImGui::Separator();
                    ImGui::Checkbox("Show Isosurface Wireframe##CollTab", &c.draw_wireframe);
                    ImGui::Checkbox("Show 2D Kesit Grid##CollTab", &c.draw_slice_preview);
                    if (c.draw_slice_preview) {
                        const char* axes[] = { "Axis X", "Axis Y", "Axis Z" };
                        ImGui::Combo("Slice Axis##CollTab", &c.slice_axis, axes, IM_ARRAYSIZE(axes));
                        ImGui::SliderFloat("Slice Depth##CollTab", &c.slice_plane_distance, 0.0f, 1.0f, "%.2f");
                    }
                    
                    if (!c.sdf_grid_data) {
                        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "SDF Status: Pending cook / Voxelizing...");
                    } else {
                        ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.4f, 1.0f), "SDF Status: Compiled (%d^3 voxels)", c.sdf_nx);
                    }
                }
            }
            ImGui::EndTabItem();
        }

        // Group 4: Physical Materials
        if (ImGui::BeginTabItem("Material")) {

            ImGui::DragFloat("Restitution (Bounce)##CollTab", &c.restitution, 0.01f, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Frictional energy conservation ratio on particle collision bounce. 0.0 = damp rebound, 1.0 = highly elastic.");
            }
            ImGui::DragFloat("Friction Coefficient##CollTab",    &c.friction,    0.01f, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Tangent sliding resistance applied on particle contact events.");
            }
            ImGui::DragFloat("Voxel Inflation Thickness##CollTab",   &c.thickness,   0.005f, 0.0f, 5.0f, "%.3f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Inflates the collision bounds during fluid grid voxelization.\n"
                                  "For sub-voxel thin walls, set this value >= active Fluid voxel_size to prevent fluid leaks.");
            }
            ImGui::SeparatorText("Gas Interaction");
            ImGui::Checkbox("Enable Gas Surface Source##CollTab", &c.gas_interaction_enabled);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Opt-in only. Injects gas channels in a thin band outside this collider; ordinary colliders remain pure boundaries.");
            }
            ImGui::BeginDisabled(!c.gas_interaction_enabled);
            ImGui::DragFloat("Smoke / Density Rate##CollTab", &c.gas_density_rate, 0.01f, 0.0f, 100.0f, "%.3f");
            ImGui::DragFloat("Temperature Rate##CollTab", &c.gas_temperature_rate, 0.01f, 0.0f, 100.0f, "%.3f");
            ImGui::DragFloat("Fuel Rate##CollTab", &c.gas_fuel_rate, 0.01f, 0.0f, 100.0f, "%.3f");
            ImGui::DragFloat("Flame Rate##CollTab", &c.gas_flame_rate, 0.01f, 0.0f, 100.0f, "%.3f");
            ImGui::DragFloat("Surface Band (Voxels)##CollTab", &c.gas_surface_band_voxels, 0.05f, 0.25f, 8.0f, "%.2f");
            ImGui::Checkbox("Ignite on Contact##CollTab", &c.gas_ignite_on_contact);
            ImGui::BeginDisabled(!c.gas_ignite_on_contact);
            ImGui::DragFloat("Ignition Temperature##CollTab", &c.gas_ignition_temperature, 0.01f, 0.0f, 100.0f, "%.3f");
            ImGui::DragFloat("Surface Fuel Capacity##CollTab", &c.gas_surface_fuel_capacity, 0.05f, 0.0f, 100.0f, "%.2f");
            ImGui::DragFloat("Surface Burn Rate##CollTab", &c.gas_surface_burn_rate, 0.01f, 0.0f, 20.0f, "%.3f");
            ImGui::EndDisabled();
            ImGui::EndDisabled();
            ImGui::EndTabItem();
        }

        // General footer actions
        if (ImGui::BeginTabItem("Actions")) {
            if (ImGui::Button("Wipe All Registered Colliders##CollTabWipe", ImVec2(-1, 30))) {
                scene.clearParticleColliders();
                selected_collider_index_global = -1;
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
        }

        if (scene.computeSimConfigSignature() != collider_sig_before) {
            scene.syncRigidBodyProxyColliders();
            scene.invalidateRigidBodySimulationCache();
        }
    };

    if (simulation_section == 1) {
        drawParticleControls();
        return;
    }
    if (simulation_section == 2) {
        clearForceFieldSelection();
        drawSimulationDomainControls(
            ui, ui_ctx, scene, timeline, selected_domain_index,
            drainSimulationMutationBackends,
            clearForceFieldSelection,
            [&]() { drawSimBakeControls(); });
        return;
    }
    if (simulation_section == 3) {
        clearForceFieldSelection();
        drawColliderControls();
        return;
    }
    if (simulation_section == 4) {
        clearForceFieldSelection();
        drawRigidBodyControls();
        return;
    }
    
    ImGui::TextColored(ImVec4(0.08f, 0.58f, 0.98f, 1.00f), "Force Fields");
    ImGui::SameLine();
    ImGui::TextDisabled("%d registered", static_cast<int>(manager.force_fields.size()));
    // Heads-up about cache invalidation: editing a force field resets the sim cache
    // and rewinds to the start. We don't spell this out in a long static line (it
    // crowds the panel and reads as noise) — a transient HUD toast fires at the
    // moment it actually happens (see consumeSimRewindRequest in SceneUI::draw).
    // A small "(i)" marker here carries the detail on hover for the curious.
    if (!scene.fluid_objects.empty() || !manager.force_fields.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(i)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Editing a force field invalidates the simulation cache:\n"
                              "the RAM + on-disk bake is dropped and the timeline rewinds to\n"
                              "the start. Press Play to re-bake, then re-bake to disk when happy.");
        }
    }
    ImGui::Separator();

    // Add new force field dropdown
    if (ImGui::Button("+ Add Force Field##FFAryAdd", ImVec2(-1, 28))) {
        ImGui::OpenPopup("AddForceFieldPopup");
    }
    
    if (ImGui::BeginPopup("AddForceFieldPopup")) {
        ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.42f, 1.0f), "Add Force Field");
        ImGui::Separator();
        
        struct FieldTypeInfo {
            int index;
            const char* name;
            UIWidgets::IconType icon;
            ImU32 color;
        };
        
        auto drawFieldItem = [&](const FieldTypeInfo& info) {
            ImVec2 pos = ImGui::GetCursorScreenPos();
            std::string label = "    " + std::string(info.name);
            if (ImGui::Selectable(label.c_str(), false, 0, ImVec2(0, 22.0f))) {
                auto field = std::make_shared<Physics::ForceField>();
                field->name = std::string(info.name) + " " + std::to_string(manager.force_fields.size() + 1);
                field->type = static_cast<Physics::ForceFieldType>(info.index);
                
                // Set defaults based on type
                switch (field->type) {
                    case Physics::ForceFieldType::Turbulence:
                    case Physics::ForceFieldType::CurlNoise:
                        field->use_noise = true;
                        field->shape = Physics::ForceFieldShape::Sphere;
                        break;
                    case Physics::ForceFieldType::Vortex:
                        field->shape = Physics::ForceFieldShape::Cylinder;
                        field->inward_force = 0.5f;
                        field->upward_force = 0.2f;
                        break;
                    case Physics::ForceFieldType::Drag:
                        field->shape = Physics::ForceFieldShape::Sphere;
                        field->linear_drag = 0.5f;
                        break;
                    default:
                        break;
                }
                
                manager.addForceField(field);
                ui_ctx.selection.selectForceField(field, -1, field->name);
                selected_force_field = field;
                ImGui::CloseCurrentPopup();
            }
            UIWidgets::DrawIcon(info.icon, ImVec2(pos.x + 4.0f, pos.y + 3.0f), 16.0f, info.color, 1.2f);
        };
        
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Basic Forces");
        drawFieldItem({0, "Wind Field", UIWidgets::IconType::Wind, IM_COL32(180, 220, 255, 255)});
        drawFieldItem({1, "Gravity Field", UIWidgets::IconType::Gravity, IM_COL32(255, 120, 120, 255)});
        drawFieldItem({7, "Drag Field", UIWidgets::IconType::Physics, IM_COL32(200, 200, 200, 255)});
        
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.73f, 0.42f, 1.0f), "Attraction & Vortices");
        drawFieldItem({2, "Attractor Field", UIWidgets::IconType::Magnet, IM_COL32(255, 180, 120, 255)});
        drawFieldItem({3, "Repeller Field", UIWidgets::IconType::Magnet, IM_COL32(255, 100, 100, 255)});
        drawFieldItem({4, "Vortex Field", UIWidgets::IconType::Vortex, IM_COL32(220, 150, 255, 255)});
        drawFieldItem({8, "Magnetic Field", UIWidgets::IconType::Magnet, IM_COL32(120, 180, 255, 255)});
        
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.56f, 0.90f, 0.47f, 1.0f), "Turbulence & Noise");
        drawFieldItem({5, "Turbulence Field", UIWidgets::IconType::Noise, IM_COL32(150, 255, 180, 255)});
        drawFieldItem({6, "Curl Noise Field", UIWidgets::IconType::Noise, IM_COL32(120, 255, 220, 255)});
        
        ImGui::EndPopup();
    }

    // List existing force fields
    ImGui::Spacing();
    std::shared_ptr<Physics::ForceField> field_to_remove = nullptr;
    std::shared_ptr<Physics::ForceField> field_to_duplicate = nullptr;
    if (manager.force_fields.empty()) {
        ImGui::TextDisabled("No force fields yet.");
    } else if (ImGui::BeginTable("ForceFieldRegistryTable", 4,
                                 ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 24.0f);
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("Enabled", ImGuiTableColumnFlags_WidthFixed, 68.0f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 54.0f);
        for (size_t i = 0; i < manager.force_fields.size(); ++i) {
            auto& row_field = manager.force_fields[i];
            if (!row_field) continue;

            const bool is_selected = selected_force_field == row_field;
            UIWidgets::IconType icon_type = UIWidgets::IconType::Force;
            switch (row_field->type) {
                case Physics::ForceFieldType::Wind:      icon_type = UIWidgets::IconType::Wind; break;
                case Physics::ForceFieldType::Gravity:   icon_type = UIWidgets::IconType::Gravity; break;
                case Physics::ForceFieldType::Vortex:    icon_type = UIWidgets::IconType::Vortex; break;
                case Physics::ForceFieldType::Turbulence:
                case Physics::ForceFieldType::CurlNoise: icon_type = UIWidgets::IconType::Noise; break;
                case Physics::ForceFieldType::Magnetic:  icon_type = UIWidgets::IconType::Magnet; break;
                case Physics::ForceFieldType::Attractor:
                case Physics::ForceFieldType::Repeller:
                case Physics::ForceFieldType::Drag:      icon_type = UIWidgets::IconType::Physics; break;
                default: break;
            }

            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImVec2 pos = ImGui::GetCursorScreenPos();
            UIWidgets::DrawIcon(icon_type, ImVec2(pos.x, pos.y + 2.0f), 16,
                is_selected ? ImGui::ColorConvertFloat4ToU32(ImVec4(0.1f, 0.9f, 0.8f, 1.0f))
                            : ImGui::ColorConvertFloat4ToU32(ImVec4(0.7f, 0.7f, 0.7f, 1.0f)), 1.0f);
            ImGui::Dummy(ImVec2(18.0f, 20.0f));

            ImGui::TableSetColumnIndex(1);
            if (!row_field->enabled) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            }
            if (ImGui::Selectable(row_field->name.c_str(), is_selected)) {
                ui_ctx.selection.selectForceField(row_field, -1, row_field->name);
                selected_force_field = row_field;
            }
            if (!row_field->enabled) {
                ImGui::PopStyleColor();
            }
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Delete Field")) {
                    field_to_remove = row_field;
                }
                if (ImGui::MenuItem("Duplicate Field")) {
                    field_to_duplicate = row_field;
                }
                ImGui::Separator();
                if (ImGui::MenuItem(row_field->enabled ? "Disable" : "Enable")) {
                    row_field->enabled = !row_field->enabled;
                }
                ImGui::EndPopup();
            }

            ImGui::TableSetColumnIndex(2);
            ImGui::Checkbox("##enabled", &row_field->enabled);
            ImGui::TableSetColumnIndex(3);
            if (ImGui::SmallButton("+")) {
                field_to_duplicate = row_field;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Duplicate field");
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("x")) {
                field_to_remove = row_field;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Delete field");
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (field_to_duplicate) {
        auto copy = std::make_shared<Physics::ForceField>(*field_to_duplicate);
        copy->name += " Copy";
        manager.addForceField(copy);
    }
    if (field_to_remove) {
        manager.removeForceField(field_to_remove);
        if (selected_force_field == field_to_remove) {
            selected_force_field = nullptr;
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // SELECTED FORCE FIELD PROPERTIES
    // ═══════════════════════════════════════════════════════════════════════
    if (!selected_force_field) {
        ImGui::TextDisabled("Select a force field from the registry to edit properties.");
        return;
    }
    
    auto& field = selected_force_field;
    bool ff_changed = false;
    
    if (!ImGui::BeginTabBar("##ForceFieldAuthoringTabs", ImGuiTabBarFlags_FittingPolicyResizeDown)) {
        return;
    }

    // Group 1: General & Transform Settings
    if (ImGui::BeginTabItem("General")) {

    // Name
    char name_buf[128];
    strncpy_s(name_buf, field->name.c_str(), sizeof(name_buf) - 1);
    if (ImGui::InputText("Field Name##FFName", name_buf, sizeof(name_buf))) {
        field->name = name_buf;
        if (ui_ctx.selection.selected.type == SelectableType::ForceField && ui_ctx.selection.selected.force_field == field) {
            ui_ctx.selection.selected.name = field->name;
        }
    }
    
    if (ImGui::Checkbox("Field Enabled##FFActive", &field->enabled)) ff_changed = true;
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Enables/disables the physical forces generated by this field.");
    }
    ImGui::SameLine(0.0f, 15.0f);
    ImGui::Checkbox("Draw Viewport Gizmo##FFGizmo", &field->visible);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Draws interactive 3D gizmos in the viewport indicating bounds and direction.");
    }
    
    ImGui::Spacing();
    
    float pos[3] = { field->position.x, field->position.y, field->position.z };
    if (ImGui::DragFloat3("Position##FFPos", pos, 0.05f, -10000.0f, 10000.0f, "%.2f")) {
        field->position = Vec3(pos[0], pos[1], pos[2]);
        if (ui_ctx.selection.selected.type == SelectableType::ForceField && ui_ctx.selection.selected.force_field == field) {
            ui_ctx.selection.selected.position = field->position;
        }
        ff_changed = true;
    }
    
    float rot[3] = { field->rotation.x, field->rotation.y, field->rotation.z };
    if (ImGui::DragFloat3("Rotation##FFRot", rot, 1.0f, -360.0f, 360.0f, "%.1f")) {
        field->rotation = Vec3(rot[0], rot[1], rot[2]);
        if (ui_ctx.selection.selected.type == SelectableType::ForceField && ui_ctx.selection.selected.force_field == field) {
            ui_ctx.selection.selected.rotation = field->rotation;
        }
        ff_changed = true;
    }
    
    float scale[3] = { field->scale.x, field->scale.y, field->scale.z };
    if (ImGui::DragFloat3("Scale##FFScale", scale, 0.05f, 0.001f, 10000.0f, "%.3f")) {
        field->scale = Vec3(scale[0], scale[1], scale[2]);
        if (ui_ctx.selection.selected.type == SelectableType::ForceField && ui_ctx.selection.selected.force_field == field) {
            ui_ctx.selection.selected.scale = field->scale;
        }
        ff_changed = true;
    }
        ImGui::EndTabItem();
    }

    // Group 2: Dynamics & Shape Settings
    if (ImGui::BeginTabItem("Dynamics")) {

    const char* types[] = { 
        "Wind Field", "Gravity Field", "Attractor Field", "Repeller Field", 
        "Vortex Field", "Turbulence Field", "Curl Noise Field", "Drag Field", "Magnetic Field", "Directional Noise"
    };
    int current_type = static_cast<int>(field->type);
    if (ImGui::Combo("Force Type##FFType", &current_type, types, IM_ARRAYSIZE(types))) {
        field->type = static_cast<Physics::ForceFieldType>(current_type);
        ff_changed = true;
    }
    
    const char* shapes[] = { "Infinite (Global)", "Sphere (Radial)", "Box (Oriented)", "Cylinder", "Cone" };
    int current_shape = static_cast<int>(field->shape);
    if (ImGui::Combo("Force Bounds Shape##FFShape", &current_shape, shapes, IM_ARRAYSIZE(shapes))) {
        field->shape = static_cast<Physics::ForceFieldShape>(current_shape);
        ff_changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Limits the volumetric boundary shape within which this force field is active.");
    }

    if (ImGui::DragFloat("Force Strength##FFStrength", &field->strength, 0.1f, -1000.0f, 1000.0f, "%.2f")) ff_changed = true;
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Luminance or speed acceleration strength applied to affected particles/smoke.");
    }
    
    // Direction for directional wind/gravity/magnetic
    if (field->type == Physics::ForceFieldType::Wind || 
        field->type == Physics::ForceFieldType::Gravity ||
        field->type == Physics::ForceFieldType::Magnetic) {
        float dir[3] = { field->direction.x, field->direction.y, field->direction.z };
        if (ImGui::DragFloat3("Force Direction##FFDir", dir, 0.01f, -1.0f, 1.0f, "%.2f")) {
            field->direction = Vec3(dir[0], dir[1], dir[2]);
            float len = field->direction.length();
            if (len > 0.001f) field->direction = field->direction * (1.0f / len);
            ff_changed = true;
        }
    }
    
    // Wind→fluid coupling (liquid only). Other systems always use the body force.
    if (field->type == Physics::ForceFieldType::Wind) {
        ImGui::Separator();
        if (ImGui::Checkbox("Fluid Surface Drag##FFWindFluidDrag", &field->fluid_surface_drag)) ff_changed = true;
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Liquid only: drive water as a relative-velocity surface drag\n"
                              "instead of a uniform body push. Water gains 'weight' — it\n"
                              "accelerates toward the wind speed and saturates there, and the\n"
                              "push fades with depth so deep water stays calm.\n"
                              "With this ON, Strength is read as the target surface speed (m/s).");
        }
        if (field->fluid_surface_drag) {
            if (ImGui::DragFloat("Drag Coupling (1/s)##FFWindCoupling", &field->fluid_drag_coupling, 0.05f, 0.0f, 50.0f, "%.2f")) ff_changed = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("How fast surface water reaches the wind speed. Higher = snappier.");
            if (ImGui::DragFloat("Surface Depth (m)##FFWindDepth", &field->fluid_surface_depth, 0.01f, 0.01f, 50.0f, "%.2f")) ff_changed = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("How far below the free surface the wind still pushes the liquid.");
            if (ImGui::DragFloat("Curl Detail##FFWindCurl", &field->fluid_curl_detail, 0.01f, 0.0f, 1.0f, "%.2f")) ff_changed = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Mix divergence-free curl-noise turbulence onto the wind so the\n"
                                  "surface flow swirls instead of moving in a dead-straight line.\n"
                                  "Uses this field's Noise settings (frequency/octaves/speed).");
        }
    }

    // Vortex-specific core properties
    if (field->type == Physics::ForceFieldType::Vortex) {
        float axis[3] = { field->axis.x, field->axis.y, field->axis.z };
        if (ImGui::DragFloat3("Vortex Core Axis##FFVortAxis", axis, 0.01f, -1.0f, 1.0f, "%.2f")) {
            field->axis = Vec3(axis[0], axis[1], axis[2]);
            float len = field->axis.length();
            if (len > 0.001f) field->axis = field->axis * (1.0f / len);
            ff_changed = true;
        }
        if (ImGui::DragFloat("Inward Pull Force##FFVortInward", &field->inward_force, 0.05f, -100.0f, 100.0f, "%.2f")) ff_changed = true;
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Centripetal inward pull strength (attracts particles to the vortex core).");
        }
        if (ImGui::DragFloat("Upward Lift Force##FFVortUpward", &field->upward_force, 0.05f, -100.0f, 100.0f, "%.2f")) ff_changed = true;
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Upward lift axial velocity along the vortex spine.");
        }
    }
    
    // Drag-specific properties
    if (field->type == Physics::ForceFieldType::Drag) {
        if (ImGui::DragFloat("Linear Drag Coeff##FFDragLin", &field->linear_drag, 0.01f, 0.0f, 50.0f, "%.2f")) ff_changed = true;
        if (ImGui::DragFloat("Quadratic Drag Coeff##FFDragQuad", &field->quadratic_drag, 0.001f, 0.0f, 10.0f, "%.3f")) ff_changed = true;
    }

    // Falloff values inside volumetric bounds
    if (field->shape != Physics::ForceFieldShape::Infinite) {
        ImGui::Separator();
        const char* falloff_types[] = { 
            "None (Constant)", "Linear Decay", "Smooth Step Decay", "Spherical Decay", "Inverse Square Decay", "Exponential Decay", "Custom Curve" 
        };
        int current_falloff = static_cast<int>(field->falloff_type);
        if (ImGui::Combo("Falloff Blend Mode##FFFalloff", &current_falloff, falloff_types, IM_ARRAYSIZE(falloff_types))) {
            field->falloff_type = static_cast<Physics::FalloffType>(current_falloff);
            ff_changed = true;
        }
        
        if (ImGui::DragFloat("Inner Radius Core##FFFalloffInner", &field->inner_radius, 0.05f, 0.0f, field->falloff_radius, "%.2f")) ff_changed = true;
        if (ImGui::DragFloat("Outer Falloff Radius##FFFalloffOuter", &field->falloff_radius, 0.05f, field->inner_radius, 1000.0f, "%.2f")) ff_changed = true;
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Distance from pivot where the force begins to decay (Inner) and drops to zero (Outer).");
        }
    }
        ImGui::EndTabItem();
    }

    // Group 3: Noise & Turbulence Settings
    if (ImGui::BeginTabItem("Noise")) {
        const bool supports_noise = field->type == Physics::ForceFieldType::Turbulence ||
                                    field->type == Physics::ForceFieldType::CurlNoise ||
                                    field->type == Physics::ForceFieldType::Wind;

        ImGui::BeginDisabled(!supports_noise);
        if (ImGui::Checkbox("Enable FBM Noise Modulation##FFUseNoise", &field->use_noise)) ff_changed = true;
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Applies fractal Brownian motion noise field to produce turbulence fluctuations.");
        }
        
        if (field->use_noise) {
            if (ImGui::DragFloat("Frequency##FFNoiseFreq", &field->noise.frequency, 0.01f, 0.001f, 50.0f, "%.3f")) ff_changed = true;
            if (ImGui::DragFloat("Amplitude##FFNoiseAmp", &field->noise.amplitude, 0.05f, 0.0f, 100.0f, "%.2f")) ff_changed = true;
            if (ImGui::DragInt("Octaves Detail##FFNoiseOct", &field->noise.octaves, 0.1f, 1, 8)) ff_changed = true;
            if (ImGui::DragFloat("Lacunarity##FFNoiseLac", &field->noise.lacunarity, 0.05f, 1.0f, 6.0f, "%.2f")) ff_changed = true;
            if (ImGui::DragFloat("Persistence##FFNoisePer", &field->noise.persistence, 0.01f, 0.0f, 1.0f, "%.2f")) ff_changed = true;
            if (ImGui::DragFloat("Evolution Speed##FFNoiseSpd", &field->noise.speed, 0.01f, 0.0f, 10.0f, "%.2f")) ff_changed = true;
            if (ImGui::DragInt("Random Seed##FFNoiseSeed", &field->noise.seed, 1, 0, 99999)) ff_changed = true;
        }
        ImGui::EndDisabled();
        if (!supports_noise) {
            ImGui::TextDisabled("Noise controls are used by Wind, Turbulence, and Curl Noise fields.");
        }
        ImGui::EndTabItem();
    }
    
    // Group 4: Activation Bounds & Mask Bindings
    if (ImGui::BeginTabItem("Activation")) {

    if (ImGui::DragFloat("Start Frame Limit##FFTimeStart", &field->start_frame, 1.0f, 0.0f, 100000.0f, "%.0f")) ff_changed = true;
    if (ImGui::DragFloat("End Frame Limit##FFTimeEnd", &field->end_frame, 1.0f, -1.0f, 100000.0f, "%.0f")) ff_changed = true;
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Timeline frame range during which the field exerts forces. Set End Frame to -1 for infinite duration.");
    }
    if (ImGui::DragFloat("Phase Velocity##FFTimePhase", &field->phase, 0.01f, 0.0f, 100.0f, "%.2f")) ff_changed = true;
    
    ImGui::Separator();
    ImGui::TextDisabled("Affected Targets:");
    if (ImGui::Checkbox("Gas/Smoke##FFAffectGas", &field->affects_gas)) ff_changed = true; ImGui::SameLine(120);
    if (ImGui::Checkbox("Particles##FFAffectPart", &field->affects_particles)) ff_changed = true; ImGui::SameLine(240);
    if (ImGui::Checkbox("Cloth##FFAffectCloth", &field->affects_cloth)) ff_changed = true; ImGui::SameLine(360);
    if (ImGui::Checkbox("Rigid Bodies##FFAffectRigid", &field->affects_rigidbody)) ff_changed = true;
        ImGui::EndTabItem();
    }

    ImGui::EndTabBar();

    if (ff_changed) {
        scene.invalidateRigidBodySimulationCache();
        ui_ctx.renderer.resetCPUAccumulation();
        if (ui_ctx.backend_ptr) {
            ui_ctx.backend_ptr->resetAccumulation();
        }
    }
}

} // namespace ForceFieldUI
