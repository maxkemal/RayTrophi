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

namespace ForceFieldUI {

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
            const float pw = (ImGui::GetContentRegionAvail().x - 2.0f * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
            if (ImGui::Button("Campfire##PresetCamp", ImVec2(pw, 26))) {
                scene.addParticleSystemPreset(SceneData::ParticleSystemPreset::Campfire);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Spawns a ready-to-run campfire setup featuring fire, rising smoke, and ember sparks.");
            }
            ImGui::SameLine();
            if (ImGui::Button("Explosion##PresetExpl", ImVec2(pw, 26))) {
                scene.addParticleSystemPreset(SceneData::ParticleSystemPreset::Explosion);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Spawns a highly dynamic explosion burst utilizing spherical force fields.");
            }
            ImGui::SameLine();
            if (ImGui::Button("Smoke##PresetSmoke", ImVec2(pw, 26))) {
                scene.addParticleSystemPreset(SceneData::ParticleSystemPreset::Smoke);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Spawns a grid-domain volumetric smoke system with fine rising turbulence.");
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

    auto drawDomainControls = [&]() {
        auto particles = scene.getParticleSimulationSystem();
        const int domain_count = particles ? static_cast<int>(particles->gridDomains().size()) : 0;
        ImGui::Text("Domains: %d", domain_count);

        // With no object selected, spawn the domain centred on the world origin
        // (a predictable, reproducible spot) rather than wherever the camera
        // happens to be looking — camera->lookat drifts as the user orbits, so
        // a "default" domain would otherwise land at an arbitrary off-centre
        // coordinate. Y stays slightly above the origin so the box straddles
        // the ground plane the same way it always has.
        Vec3 center(0.0f, 1.0f, 0.0f);

        const bool has_object_selection =
            ui_ctx.selection.selected.type == SelectableType::Object &&
            ui_ctx.selection.selected.object != nullptr &&
            !ui_ctx.selection.selected.object->getNodeName().empty();
        if (has_object_selection) {
            Vec3 selected_min;
            Vec3 selected_max;
            if (scene.resolveObjectBoundsForSimulation(ui_ctx.selection.selected.object->getNodeName(), selected_min, selected_max)) {
                center = (Vec3::min(selected_min, selected_max) + Vec3::max(selected_min, selected_max)) * 0.5f;
            }
        }

        if (ImGui::Button("Add Grid Domain##SimulationPanel", ImVec2(-1, 0))) {
            RayTrophiSim::SimulationGridDomainDesc desc;
            // Unique default name so the list is distinguishable at a glance
            // (the [Gas]/[Fluid] tag in the list is derived from .type, not the
            // name, so it stays correct after a type switch).
            const std::size_t domain_count = particles ? particles->gridDomains().size() : 0;
            desc.name = "Grid Domain " + std::to_string(domain_count + 1);
            desc.source_mode = RayTrophiSim::SimulationGridDomainSourceMode::ManualBox;
            desc.bounds_min = center + Vec3(-2.5f, -2.5f, -2.5f);
            desc.bounds_max = center + Vec3(2.5f, 2.5f, 2.5f);
            scene.addSimulationGridDomain(desc);
            particles = scene.getParticleSimulationSystem();
            selected_domain_index = particles ? static_cast<int>(particles->gridDomains().size()) - 1 : -1;
            if (selected_domain_index >= 0) {
                ui_ctx.selection.selectSimulationDomain(scene.active_particle_system_index, selected_domain_index,
                    particles->gridDomains()[static_cast<std::size_t>(selected_domain_index)].name);
                const auto& selected_domain = particles->gridDomains()[static_cast<std::size_t>(selected_domain_index)];
                const Vec3 mn = Vec3::min(selected_domain.bounds_min, selected_domain.bounds_max);
                const Vec3 mx = Vec3::max(selected_domain.bounds_min, selected_domain.bounds_max);
                ui_ctx.selection.selected.position = (mn + mx) * 0.5f;
                ui_ctx.selection.selected.scale = mx - mn;
            }
        }

        if (!has_object_selection) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Add Domain From Selection##SimulationPanel", ImVec2(-1, 0))) {
            const std::string source_name = ui_ctx.selection.selected.object->getNodeName();
            scene.addSimulationGridDomainFromObject(source_name);
            particles = scene.getParticleSimulationSystem();
            selected_domain_index = particles ? static_cast<int>(particles->gridDomains().size()) - 1 : -1;
            if (selected_domain_index >= 0) {
                ui_ctx.selection.selectSimulationDomain(scene.active_particle_system_index, selected_domain_index,
                    particles->gridDomains()[static_cast<std::size_t>(selected_domain_index)].name);
                const auto& selected_domain = particles->gridDomains()[static_cast<std::size_t>(selected_domain_index)];
                const Vec3 mn = Vec3::min(selected_domain.bounds_min, selected_domain.bounds_max);
                const Vec3 mx = Vec3::max(selected_domain.bounds_min, selected_domain.bounds_max);
                ui_ctx.selection.selected.position = (mn + mx) * 0.5f;
                ui_ctx.selection.selected.scale = mx - mn;
            }
        }
        if (!has_object_selection) {
            ImGui::EndDisabled();
        }

        if (ImGui::Button("Clear Domains##SimulationPanel", ImVec2(-1, 0))) {
            scene.clearSimulationGridDomains();
            if (ui_ctx.selection.selected.type == SelectableType::SimulationDomain &&
                ui_ctx.selection.selected.particle_system_index == scene.active_particle_system_index) {
                ui_ctx.selection.clearSelection();
            }
            selected_domain_index = -1;
        }

        // Diagnostic + cleanup for "stuck default emitter" complaints. Shows
        // counts of all stale sources that could be spawning particles or
        // drawing gizmos, with one-click clear buttons. Once the user
        // identifies which list is non-empty, the corresponding button wipes
        // it. Faz 2 will consolidate these paths.
        {
            auto p_sim = scene.getParticleSimulationSystem();
            const std::size_t legacy_fluid_n = scene.fluid_objects.size();
            const std::size_t soa_emitters_n = p_sim ? p_sim->emitters().size() : 0u;
            const std::size_t flow_sources_n = p_sim ? p_sim->flowSources().size() : 0u;
            if (legacy_fluid_n + soa_emitters_n + flow_sources_n > 0) {
                ImGui::TextDisabled("Sim sources — LegacyFluid:%zu  ParticleEmitters:%zu  FlowSources:%zu",
                                     legacy_fluid_n, soa_emitters_n, flow_sources_n);
            }
            if (legacy_fluid_n > 0) {
                if (ImGui::Button("Remove Legacy Fluid Objects##SimulationPanel", ImVec2(-1, 0))) {
                    scene.fluid_objects.clear();
                    if (scene.fluid_simulation_system) {
                        scene.fluid_simulation_system->setObjects(&scene.fluid_objects);
                    }
                    scene.active_fluid_object_index = -1;
                    ui_ctx.start_render = true;
                }
            }
            if (soa_emitters_n > 0 && p_sim) {
                if (ImGui::Button("Clear Particle Emitters##SimulationPanel", ImVec2(-1, 0))) {
                    p_sim->clearEmitters();
                    ui_ctx.start_render = true;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Wipes legacy SoA particle emitters (Spark/Smoke/Fire presets).\nFluid grid-domain sources are unaffected.");
                }
            }
            if (flow_sources_n > 0 && p_sim) {
                if (ImGui::Button("Clear Flow Sources##SimulationPanel", ImVec2(-1, 0))) {
                    p_sim->clearFlowSources();
                    ui_ctx.start_render = true;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Wipes all flow sources (gas inject + fluid emit). Re-add per domain.");
                }
            }
        }

        particles = scene.getParticleSimulationSystem();
        if (!particles || particles->gridDomains().empty()) {
            ImGui::Spacing();
            ImGui::TextDisabled("No simulation domains yet.");
            return;
        }

        auto& domains = particles->gridDomains();
        if (ui_ctx.selection.selected.type == SelectableType::SimulationDomain &&
            ui_ctx.selection.selected.particle_system_index == scene.active_particle_system_index &&
            ui_ctx.selection.selected.simulation_domain_index >= 0 &&
            ui_ctx.selection.selected.simulation_domain_index < static_cast<int>(domains.size())) {
            selected_domain_index = ui_ctx.selection.selected.simulation_domain_index;
        }
        if (selected_domain_index >= static_cast<int>(domains.size())) {
            selected_domain_index = static_cast<int>(domains.size()) - 1;
        }
        if (selected_domain_index < 0) {
            selected_domain_index = 0;
        }

        ImGui::SeparatorText("Grid Domain List");
        if (ImGui::BeginListBox("##SimulationGridDomainListStandalone", ImVec2(-1, 110))) {
            for (int i = 0; i < static_cast<int>(domains.size()); ++i) {
                char label[256];
                // Type tag is DERIVED from the live domain type, not stored in
                // the name — so it follows a Gas<->Fluid switch automatically and
                // the user can still rename the domain freely.
                const char* type_tag =
                    (domains[i].type == RayTrophiSim::SimulationDomainType::Fluid) ? "Fluid" : "Gas";
                std::snprintf(label, sizeof(label), "%s  [%s]##domain_standalone%d",
                              domains[i].name.c_str(), type_tag, i);
                if (ImGui::Selectable(label, selected_domain_index == i)) {
                    selected_domain_index = i;
                    clearForceFieldSelection();
                    ui_ctx.selection.selectSimulationDomain(scene.active_particle_system_index, i, domains[i].name);
                    const Vec3 mn = Vec3::min(domains[i].bounds_min, domains[i].bounds_max);
                    const Vec3 mx = Vec3::max(domains[i].bounds_min, domains[i].bounds_max);
                    ui_ctx.selection.selected.position = (mn + mx) * 0.5f;
                    ui_ctx.selection.selected.scale = mx - mn;
                }
            }
            ImGui::EndListBox();
        }

        if (selected_domain_index < 0 || selected_domain_index >= static_cast<int>(domains.size())) {
            return;
        }

        auto& domain = domains[static_cast<std::size_t>(selected_domain_index)];
        ImGui::SeparatorText("Selected Domain");
        ImGui::Checkbox("Domain Enabled", &domain.enabled);

        // Auto-reseed accumulator: any seed-OR-shape param whose edit settles this
        // frame sets `seed_settled`. The toggle checkbox lives in the Fluid
        // Seeding header; the actual reseed (rewind to frame 0 + re-seed all fluid
        // domains) runs once at the very end of this panel so it covers BOTH the
        // Setup&Grid tab (resolution/voxel/bounds) and the Fluid seeding tab.
        static bool s_fluid_auto_reseed = true;
        bool seed_settled = false;

        // Solver type (Gas vs. Fluid Segmented Buttons)
        {
            ImGui::Text("Domain Solver Type:");
            ImGui::Spacing();
            
            const float button_width = 140.0f;
            const ImVec4 active_color = ImVec4(0.08f, 0.48f, 0.88f, 1.00f); // Sleek modern royal blue
            const ImVec4 inactive_color = ImGui::GetStyleColorVec4(ImGuiCol_Button);
            
            const bool is_gas = (domain.type == RayTrophiSim::SimulationDomainType::Gas);
            const bool is_fluid = (domain.type == RayTrophiSim::SimulationDomainType::Fluid);
            
            // --- Gas Button ---
            if (is_gas) {
                ImGui::PushStyleColor(ImGuiCol_Button, active_color);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.12f, 0.54f, 0.94f, 1.00f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.06f, 0.42f, 0.82f, 1.00f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, inactive_color);
            }
            if (ImGui::Button("Gas (Smoke/Fire)##TypeGas", ImVec2(button_width, 30))) {
                if (!is_gas) {
                    domain.type = RayTrophiSim::SimulationDomainType::Gas;
                    ui_ctx.start_render = true;
                    if (scene.active_particle_system_index >= 0 &&
                        scene.active_particle_system_index < static_cast<int>(scene.particle_systems.size())) {
                        auto& active_sys = scene.particle_systems[static_cast<size_t>(scene.active_particle_system_index)];
                        if (selected_domain_index >= 0 &&
                            selected_domain_index < static_cast<int>(active_sys.domain_last_fluid_render_mode.size())) {
                            active_sys.domain_last_fluid_render_mode[static_cast<size_t>(selected_domain_index)] = -1;
                        }
                    }
                }
            }
            ImGui::PopStyleColor(is_gas ? 3 : 1);
            
            ImGui::SameLine(0.0f, 10.0f);
            
            // --- Fluid Button ---
            if (is_fluid) {
                ImGui::PushStyleColor(ImGuiCol_Button, active_color);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.12f, 0.54f, 0.94f, 1.00f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.06f, 0.42f, 0.82f, 1.00f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, inactive_color);
            }
            if (ImGui::Button("Fluid (Liquid)##TypeFluid", ImVec2(button_width, 30))) {
                if (!is_fluid) {
                    domain.type = RayTrophiSim::SimulationDomainType::Fluid;
                    // Liquid wants a sealed box by default so it pools/settles;
                    // open walls would silently drain it. Only override the
                    // gas-default Open — leave an already-chosen Closed/Periodic.
                    if (domain.boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Open) {
                        domain.boundary_mode = RayTrophiSim::SimulationGridDomainBoundaryMode::Closed;
                    }
                    ui_ctx.start_render = true;
                    if (scene.active_particle_system_index >= 0 &&
                        scene.active_particle_system_index < static_cast<int>(scene.particle_systems.size())) {
                        auto& active_sys = scene.particle_systems[static_cast<size_t>(scene.active_particle_system_index)];
                        if (selected_domain_index >= 0 &&
                            selected_domain_index < static_cast<int>(active_sys.domain_last_fluid_render_mode.size())) {
                            active_sys.domain_last_fluid_render_mode[static_cast<size_t>(selected_domain_index)] = -1;
                        }
                    }
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Simulates realistic fluids like water, honey, or viscous liquids using APIC/FLIP algorithms.");
            }
            ImGui::PopStyleColor(is_fluid ? 3 : 1);
            ImGui::Spacing();
        }
        const bool is_fluid_domain = domain.type == RayTrophiSim::SimulationDomainType::Fluid;
        const bool is_gas_domain  = !is_fluid_domain;

        ImGui::Separator();
        // ── SUB-TABS FOR DOMAIN ──
        if (ImGui::BeginTabBar("DomainSubTabBar", ImGuiTabBarFlags_None)) {

            if (ImGui::BeginTabItem("Setup & Grid")) {
                ImGui::Spacing();

                // Group 1: Compute & Backend
                if (ImGui::CollapsingHeader("Compute Device & Backend", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Spacing();

                const char* backends[] = {
                    "CPU (Dense - Standard)",
                    "GPU (CUDA - High Speed)",
                    "CPU (Sparse OpenVDB - Low Memory)",
                    "GPU (Vulkan - EXPERIMENTAL / test only)"
                };
                int current_backend = static_cast<int>(domain.backend);
                extern bool g_hasCUDA;
                extern bool g_hasVulkanComputeSim;

                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::Combo("##DomainBackend", &current_backend, backends, 4)) {
                    domain.backend = static_cast<RayTrophiSim::SimulationDomainBackend>(current_backend);
                    extern bool g_gas_volumes_dirty;
                    g_gas_volumes_dirty = true;
                    ui_ctx.start_render = true;
                    if (ui_ctx.backend_ptr) {
                        ui_ctx.backend_ptr->resetAccumulation();
                    }
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Selects which hardware execution unit running the simulation solver:\n\n"
                                      "1. CPU (Dense): Stable standard processor solver. Ideal for small-scale tests.\n"
                                      "2. GPU (CUDA): Fully hardware-accelerated ultra-fast GPU compute mode. Recommended for large domains.\n"
                                      "3. CPU (Sparse OpenVDB): Sparse grid system skips empty air cells containing no gas/smoke, saving memory.\n"
                                      "4. GPU (Vulkan): EXPERIMENTAL / under development. The fluid (APIC) solve is\n"
                                      "   not yet correct on Vulkan (driver float-atomic accumulation issue) — for\n"
                                      "   testing only. Use CPU (Dense) or GPU (CUDA) for correct fluid results.");
                }

                ImGui::Spacing();
                if (domain.backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) {
                    if (g_hasVulkanComputeSim)
                        ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "  [GPU Status: Vulkan Compute Active]");
                    else
                        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "  [GPU Status: Vulkan Compute - SPIR-V shaders missing, CPU fallback]");
                    // Vulkan fluid solve is still under development — flag it so it
                    // isn't mistaken for a production-ready path. CPU/CUDA are correct.
                    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f),
                        "  WARNING: Vulkan fluid solve is EXPERIMENTAL (under development).");
                    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f),
                        "  Fluid does not flow correctly here yet — use CPU or CUDA for real runs.");
                } else if (g_hasCUDA) {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, 1.0f), "  [GPU Status: CUDA Acceleration Active]");
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "  [GPU Status: CUDA Capable GPU Not Found - CPU Fallback]");
                }
                }

                // Group 2: Grid Resolution & Scaling
                if (ImGui::CollapsingHeader("Grid Resolution & Scaling", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Spacing();

                int res[3] = { domain.resolution_x, domain.resolution_y, domain.resolution_z };
                if (ImGui::DragInt3("Grid Resolution (X, Y, Z)", res, 1.0f, 8, 512)) {
                    domain.resolution_x = std::clamp(res[0], 8, 512);
                    domain.resolution_y = std::clamp(res[1], 8, 512);
                    domain.resolution_z = std::clamp(res[2], 8, 512);
                    const Vec3 ext = Vec3::max(domain.bounds_min, domain.bounds_max) - Vec3::min(domain.bounds_min, domain.bounds_max);
                    const float me = std::max({ ext.x, ext.y, ext.z, 0.001f });
                    const int mr = std::max({ domain.resolution_x, domain.resolution_y, domain.resolution_z, 1 });
                    domain.voxel_size = me / static_cast<float>(mr);
                }
                seed_settled |= ImGui::IsItemDeactivatedAfterEdit();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Per-axis resolution of the 3D voxel simulation grid (8-512 each).\n\n"
                                      "WARNING: Cost scales with the product X*Y*Z (cubic for a cube)!\n"
                                      "Use 32-64 for fast interactive testing, 96-128+ for high-quality results.\n"
                                      "Prefer a non-cube box (e.g. 512x64x128) over a full 512^3 cube.");
                }
                // Live cell-count + rough grid memory so high resolutions are an
                // informed choice. The solver re-derives resolution each rebuild and
                // clamps every axis to Max Auto Resolution, so preview the clamped
                // values - otherwise the estimate lies whenever a requested axis
                // exceeds the ceiling. ~11 floats/cell estimates MAC velocity faces +
                // pressure/divergence/density/mask + PCG scratch vectors.
                {
                    const int eff_cap = std::clamp(domain.max_auto_resolution, 32, 512);
                    const int eff_x = std::clamp(domain.resolution_x, 8, eff_cap);
                    const int eff_y = std::clamp(domain.resolution_y, 8, eff_cap);
                    const int eff_z = std::clamp(domain.resolution_z, 8, eff_cap);
                    const std::size_t cell_preview =
                        static_cast<std::size_t>(eff_x) *
                        static_cast<std::size_t>(eff_y) *
                        static_cast<std::size_t>(eff_z);
                    const double grid_mb =
                        static_cast<double>(cell_preview) * 11.0 * sizeof(float) / (1024.0 * 1024.0);
                    ImVec4 col = (grid_mb > 3000.0) ? ImVec4(1.0f, 0.35f, 0.35f, 1.0f)   // >~3 GB: danger
                               : (grid_mb >  800.0) ? ImVec4(1.0f, 0.75f, 0.30f, 1.0f)   // >~0.8 GB: caution
                                                    : ImVec4(0.55f, 0.85f, 0.55f, 1.0f); // comfortable
                    ImGui::TextColored(col, "Effective: %dx%dx%d = %zu cells  (~%.0f MB grid est.)",
                                       eff_x, eff_y, eff_z, cell_preview, grid_mb);
                    if (eff_x < domain.resolution_x || eff_y < domain.resolution_y || eff_z < domain.resolution_z) {
                        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.30f, 1.0f),
                                           "  (clamped by Max Auto Resolution = %d - raise it below to go higher)", eff_cap);
                    }
                    if (grid_mb > 3000.0) {
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), " - OOM/freeze risk");
                    }
                }

                ImGui::Spacing();
                if (ImGui::Checkbox("Preserve Voxel Size", &domain.preserve_voxel_size_on_resize)) {
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("When the domain bounding box is resized, keeps the physical size of a single volume cell (voxel) constant.\n"
                                      "The grid resolution automatically increases/decreases as the domain expands/shrinks.");
                }

                if (ImGui::Checkbox("Sparse Grid System (Sparse Tiles)", &domain.use_sparse_tiles)) {
                    ui_ctx.start_render = true;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Reduces processing overhead by only allocating memory for cells containing active fluid/smoke.\n"
                                      "Yields huge performance and memory savings by ignoring empty air zones.");
                }

                if (ImGui::Checkbox("NanoVDB Volumetric Render (Ray Tracing)", &domain.render_to_nanovdb)) {
                    extern bool g_gas_volumes_dirty;
                    extern bool g_geometry_dirty;
                    extern bool g_vulkan_rebuild_pending;
                    extern bool g_optix_rebuild_pending;
                    scene.requestSimulationTimelineRenderResync();
                    g_gas_volumes_dirty = true;
                    g_geometry_dirty = true;
                    g_vulkan_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    ui_ctx.start_render = true;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Converts simulation grids into NanoVDB volumetric formats in real-time for Vulkan RT and OptiX ray-traced rendering.\n"
                                      "This option MUST be enabled to render realistic volumetric fog, clouds, or smoke.");
                }

                ImGui::Spacing();
                ImGui::SetNextItemWidth(150.0f);
                const int prev_max_auto_res = domain.max_auto_resolution;
                if (ImGui::DragInt("Max Auto Resolution", &domain.max_auto_resolution, 1.0f, 32, 512)) {
                    domain.max_auto_resolution = std::clamp(domain.max_auto_resolution, 32, 512);
                    // Changing the ceiling must keep the grid spanning the FULL domain, so
                    // pick the Preserve Voxel mode that covers it in each direction:
                    //  • LOWER  -> Preserve OFF: voxel is re-derived (coarser) so res*voxel
                    //    still spans the domain. With preserve ON, res clamps to the lower
                    //    ceiling and the grid's world coverage shrinks to a corner.
                    //  • RAISE  -> Preserve ON: keep the voxel and just add cells up to the
                    //    new cap (finer, still full coverage). With preserve OFF the budget
                    //    clamp + min-voxel recompute leaves the large axes partly uncovered.
                    if (domain.max_auto_resolution != prev_max_auto_res) {
                        domain.preserve_voxel_size_on_resize =
                            (domain.max_auto_resolution > prev_max_auto_res);
                        scene.requestSimulationTimelineRenderResync();
                        ui_ctx.start_render = true;
                    }
                }
                seed_settled |= ImGui::IsItemDeactivatedAfterEdit();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Hard per-axis ceiling for the simulation grid - the solver re-derives\n"
                                      "resolution from voxel size each rebuild and clamps every axis to this.\n"
                                      "It is BOTH the auto-scale safety limit and the cap the manual\n"
                                      "'Grid Resolution' above can actually reach. Raise to 512 for a full\n"
                                      "256^3+ run; watch the cell/memory estimate - 512^3 risks OOM/freeze.\n"
                                      "Changing this auto-sets Preserve Voxel Size (ON when raising, OFF when\n"
                                      "lowering) so the grid always re-covers the FULL domain, not a corner.");
                }
                domain.max_auto_resolution = std::clamp(domain.max_auto_resolution, 32, 512);

                ImGui::SameLine();
                ImGui::SetNextItemWidth(150.0f);
                ImGui::DragFloat("Boundary Padding", &domain.padding, 0.01f, 0.0f, 1000.0f, "%.3f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Buffer padding distance added outside the domain bounds.\n"
                                      "Prevents smoke from hitting the boundary wall abruptly.");
                }

                const Vec3 extent = Vec3::max(domain.bounds_min, domain.bounds_max) - Vec3::min(domain.bounds_min, domain.bounds_max);
                const float max_extent = std::max({ extent.x, extent.y, extent.z, 0.001f });
                const int max_res = std::max({ domain.resolution_x, domain.resolution_y, domain.resolution_z, 1 });
                const float preview_voxel_size = max_extent / static_cast<float>(max_res);
                if (!domain.preserve_voxel_size_on_resize) {
                    domain.voxel_size = preview_voxel_size;
                }
                const std::size_t cells =
                    static_cast<std::size_t>(domain.resolution_x) *
                    static_cast<std::size_t>(domain.resolution_y) *
                    static_cast<std::size_t>(domain.resolution_z);
                
                ImGui::Spacing();
                ImGui::Separator();
                }

                // Group 3: Bounds & Boundary Mode
                if (ImGui::CollapsingHeader("Domain Bounds & Behaviors", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Spacing();

                    // Source Mode Selector
                    const char* source_modes[] = { "Manual Box (Static)", "Object Bounds (Static)", "Adaptive Particles (Dynamic)" };
                    int current_source_mode = static_cast<int>(domain.source_mode);
                    if (ImGui::Combo("Domain Source Mode", &current_source_mode, source_modes, IM_ARRAYSIZE(source_modes))) {
                        domain.source_mode = static_cast<RayTrophiSim::SimulationGridDomainSourceMode>(current_source_mode);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Choose how the grid boundary limits are defined:\n\n"
                                          "1. Manual Box: Static, manually sized bounds.\n"
                                          "2. Object Bounds: Automatically sized to a static scene object's limits.\n"
                                          "3. Adaptive Particles: Dynamically sizes and snaps bounds to active particles every step!");
                    }

                    if (domain.source_mode == RayTrophiSim::SimulationGridDomainSourceMode::ObjectBounds) {
                        Vec3 resolved_min = domain.bounds_min;
                        Vec3 resolved_max = domain.bounds_max;
                        if (scene.resolveObjectBoundsForSimulation(domain.source_name, resolved_min, resolved_max)) {
                            domain.bounds_min = resolved_min;
                            domain.bounds_max = resolved_max;
                        }
                        domain.source_name.clear();
                        domain.source_mode = RayTrophiSim::SimulationGridDomainSourceMode::ManualBox;
                    }

                    ImGui::Spacing();

                    if (domain.source_mode == RayTrophiSim::SimulationGridDomainSourceMode::Adaptive) {
                        ImGui::TextColored(ImVec4(0.0f, 0.85f, 1.0f, 1.0f), "Adaptive Grid Domain Settings:");
                        ImGui::Spacing();

                        ImGui::Checkbox("Lock Ground Level (Y Min)", &domain.adaptive_lock_floor);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Ensures the bottom of the grid remains static (anchored to ground plane Y),\n"
                                              "while X, Z, and Y-max expand/shrink around liquid splashes. Recommended for basins!");
                        }

                        if (domain.adaptive_lock_floor) {
                            ImGui::DragFloat("Ground Level Y Height", &domain.adaptive_floor_y, 0.05f, -1000.0f, 1000.0f, "%.2f");
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("The fixed Y coordinate where the bottom plane of the grid will lock.");
                            }
                        }

                        ImGui::Spacing();
                        ImGui::TextDisabled("Dynamic Bounds Min: %.2f, %.2f, %.2f", domain.bounds_min.x, domain.bounds_min.y, domain.bounds_min.z);
                        ImGui::TextDisabled("Dynamic Bounds Max: %.2f, %.2f, %.2f", domain.bounds_max.x, domain.bounds_max.y, domain.bounds_max.z);
                    } else {
                        ImGui::DragFloat3("Domain Minimum Bounds", &domain.bounds_min.x, 0.05f, -10000.0f, 10000.0f, "%.2f");
                        seed_settled |= ImGui::IsItemDeactivatedAfterEdit();
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Minimum coordinates of the domain bounding box in world space (X, Y, Z).");
                        }
                        ImGui::DragFloat3("Domain Maximum Bounds", &domain.bounds_max.x, 0.05f, -10000.0f, 10000.0f, "%.2f");
                        seed_settled |= ImGui::IsItemDeactivatedAfterEdit();
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Maximum coordinates of the domain bounding box in world space (X, Y, Z).");
                        }
                    }

                    ImGui::Spacing();
                    const char* boundary_modes[] = { "Open (Outflow - Flows Out)", "Closed (Solid Wall - Collides)", "Periodic (Wrap-around - Re-enters)" };
                    int boundary_mode = static_cast<int>(domain.boundary_mode);
                    if (ImGui::Combo("Boundary Collision Mode", &boundary_mode, boundary_modes, IM_ARRAYSIZE(boundary_modes))) {
                        domain.boundary_mode = static_cast<RayTrophiSim::SimulationGridDomainBoundaryMode>(boundary_mode);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Physical behavior of fluid/smoke when touching domain borders:\n\n"
                                          "1. Open: Outflowing fluid/smoke vanishes at the boundary. Perfect for open scenes.\n"
                                          "2. Closed: Boundaries behave as solid, invisible walls. Ideal for indoor containers.\n"
                                          "3. Periodic: Outflowing fluid automatically re-enters from the opposite side.");
                    }
                }

                // Group 4: Statistics Summary
                if (ImGui::CollapsingHeader("Simulation & Collision Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Spacing();

                int intersect_n = 0;
                if (particles && !particles->colliders().empty()) {
                    const Vec3 dmn = Vec3::min(domain.bounds_min, domain.bounds_max);
                    const Vec3 dmx = Vec3::max(domain.bounds_min, domain.bounds_max);
                    for (const auto& c : particles->colliders()) {
                        if (!c.enabled) continue;
                        switch (c.source_mode) {
                            case RayTrophiSim::ParticleColliderSourceMode::PlaneY:
                                if (c.plane_y >= dmn.y - 1.0f && c.plane_y <= dmx.y + 1.0f) ++intersect_n;
                                break;
                            case RayTrophiSim::ParticleColliderSourceMode::Sphere: {
                                const Vec3 sc = c.sphere_center;
                                const float r = c.sphere_radius + c.thickness;
                                if (sc.x + r >= dmn.x && sc.x - r <= dmx.x &&
                                    sc.y + r >= dmn.y && sc.y - r <= dmx.y &&
                                    sc.z + r >= dmn.z && sc.z - r <= dmx.z) ++intersect_n;
                                break;
                            }
                            default:
                                ++intersect_n;
                                break;
                        }
                    }
                    ImGui::Text("Intersecting Colliders: %d / %zu", intersect_n, particles->colliders().size());
                    ImGui::TextDisabled("  (Manage colliders using the main 'Colliders' tab at the top)");
                } else {
                    ImGui::TextDisabled("No active colliders registered in the scene.");
                }

                const auto& domain_states = particles->gridDomainStates();
                if (selected_domain_index < static_cast<int>(domain_states.size())) {
                    const auto& state = domain_states[static_cast<std::size_t>(selected_domain_index)];
                    if (state.valid) {
                        ImGui::Spacing();
                        ImGui::Separator();
                        ImGui::Columns(2, "DomainStatsColumns", false);
                        if (domain.source_mode == RayTrophiSim::SimulationGridDomainSourceMode::Adaptive) {
                            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, 1.0f), "Dynamic Resolution:"); ImGui::NextColumn();
                            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, 1.0f), "%dx%dx%d", state.resolution_x, state.resolution_y, state.resolution_z); ImGui::NextColumn();
                        } else {
                            ImGui::TextDisabled("Active Resolution:"); ImGui::NextColumn();
                            ImGui::TextDisabled("%dx%dx%d", state.resolution_x, state.resolution_y, state.resolution_z); ImGui::NextColumn();
                        }
                        if (is_gas_domain) {
                            ImGui::TextDisabled("Active Dense Cells:"); ImGui::NextColumn();
                            ImGui::TextDisabled("%zu", state.active_density_cells); ImGui::NextColumn();
                            ImGui::TextDisabled("Max Smoke Density:"); ImGui::NextColumn();
                            ImGui::TextDisabled("%.3f", state.max_density); ImGui::NextColumn();
                        }
                        ImGui::Columns(1);
                    } else {
                        ImGui::TextDisabled("Simulation Status: Idle (Play the timeline to step simulation / bake)");
                    }
                }
                }

                ImGui::EndTabItem();
            }

            // =================================================================
            // TAB 2: Solver & Physics (Physical Parameters & Solvers)
            // =================================================================
            if (ImGui::BeginTabItem("Solver & Physics")) {
                ImGui::Spacing();

                if (is_gas_domain) {
                    // Gas Channel Flags
                    if (ImGui::CollapsingHeader("Simulation Solver Channels (Grids)", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Spacing();

                    bool channel_density = (domain.channels & static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Density)) != 0u;
                    bool channel_temperature = (domain.channels & static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Temperature)) != 0u;
                    bool channel_velocity = (domain.channels & static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Velocity)) != 0u;
                    bool channel_fuel = (domain.channels & static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Fuel)) != 0u;
                    bool channel_pressure = (domain.channels & static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Pressure)) != 0u;
                    bool channels_changed = false;
                    
                    channels_changed |= ImGui::Checkbox("Density Grid (Smoke Visualization)##DensityGrid", &channel_density);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Stores visual soot/smoke thickness. Must be ENABLED for smoke or dust simulations.");
                    }
                    channels_changed |= ImGui::Checkbox("Temperature Grid (Buoyant Heat Rise)##TempGrid", &channel_temperature);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Stores thermal distribution values. Controls dynamic buoyant upward expansion.");
                    }
                    channels_changed |= ImGui::Checkbox("Velocity Grid (Vector Flow Field)##VelocityGrid", &channel_velocity);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Stores 3D vector velocity flow. Required for the fluid/smoke to move.");
                    }
                    channels_changed |= ImGui::Checkbox("Fuel Grid (Combustion/Fire)##FuelGrid", &channel_fuel);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Stores flammable fuel concentration. Required for explosions, flame, and fire simulations.");
                    }
                    channels_changed |= ImGui::Checkbox("Pressure Grid (Volume Incompressibility)##PressGrid", &channel_pressure);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Stores internal compression pressure. Enforces grid incompressibility and forms realistic vortices.");
                    }
                    
                    if (channels_changed) {
                        domain.channels = 0u;
                        if (channel_density) domain.channels |= static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Density);
                        if (channel_temperature) domain.channels |= static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Temperature);
                        if (channel_velocity) domain.channels |= static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Velocity);
                        if (channel_fuel) domain.channels |= static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Fuel);
                        if (channel_pressure) domain.channels |= static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Pressure);
                    }
                    }

                    // Combustion / fire
                    if (ImGui::CollapsingHeader("Combustion & Fire Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Spacing();

                    if (ImGui::Checkbox("Enable Combustion Physics (Fire & Flames)##EnableFire", &domain.fire_enabled) && domain.fire_enabled) {
                        domain.channels |= static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Fuel);
                        domain.channels |= static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Temperature);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("When enabled, fuel in cells exceeding the ignition threshold will ignite, producing fire visuals and smoke.");
                    }

                    if (domain.fire_enabled) {
                        ImGui::DragFloat("Ignition Temperature", &domain.ignition_temperature, 0.01f, 0.0f, 10.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Minimum temperature required to ignite fuel.");
                        }
                        ImGui::DragFloat("Fuel Burn Rate", &domain.burn_rate, 0.05f, 0.0f, 20.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Controls how quickly fuel burns and converts into heat/flames.");
                        }
                        ImGui::DragFloat("Heat Release Rate", &domain.heat_release, 0.05f, 0.0f, 50.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Heat energy released to adjacent cells during burning. High values accelerate combustion spread.");
                        }
                        ImGui::DragFloat("Smoke Generation Rate", &domain.smoke_generation, 0.02f, 0.0f, 10.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Determines the amount of dark soot/smoke generated per unit of burned fuel.");
                        }
                        ImGui::DragFloat("Flame Dissipation Rate", &domain.flame_dissipation, 0.05f, 0.0f, 30.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Rate at which visual fire flames and thermal energy dissipate.");
                        }
                        ImGui::DragFloat("Maximum Temperature Limit", &domain.fire_max_temperature, 0.1f, 0.1f, 100.0f, "%.1f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Upper ceiling limit for thermal values inside combustion voxels.");
                        }
                        ImGui::DragFloat("Thermal Expansion (Blast)", &domain.fire_expansion, 0.02f, 0.0f, 20.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Hot gas dilates: the pressure solve targets an outward divergence\n"
                                              "proportional to (temperature - ambient). Gives fire its rolling\n"
                                              "billow, and a sudden fuel ignition becomes a real explosion blast.\n"
                                              "0 = incompressible smoke. Note: this domain runs on the CPU solver\n"
                                              "while expansion is > 0 (GPU grid path doesn't model expansion yet).");
                        }

                        ImGui::Spacing();
                        ImGui::TextDisabled("Physics Note: Remember to add a Flow Source emitting Fuel and Temperature.\n"
                                             "Set shader mode to 'Blackbody' in the Shading tab for realistic fire rendering.");
                    } else {
                        ImGui::TextDisabled("Combustion is disabled. Simulating smoke (Density) only.");
                    }
                    }

                    // Procedural turbulence (divergence-free curl-noise detail).
                    if (ImGui::CollapsingHeader("Turbulence (Procedural Detail)")) {
                        ImGui::Spacing();
                        ImGui::DragFloat("Turbulence Strength", &domain.turbulence_strength, 0.01f, 0.0f, 50.0f, "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Adds divergence-free swirling detail on top of the solved motion.\n"
                                              "0 = off. Modulated by local density/heat/edges so still air stays calm.\n"
                                              "Not applied on the Sparse VDB backend.");
                        }
                        if (domain.turbulence_strength > 0.0f) {
                            ImGui::DragFloat("Noise Scale", &domain.turbulence_scale, 0.02f, 0.05f, 20.0f, "%.2f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Base spatial frequency of the noise. Higher = finer, busier swirls.");
                            ImGui::DragInt("Octaves", &domain.turbulence_octaves, 1, 1, 8);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("FBM octaves. More octaves add finer layered detail at higher cost.");
                            ImGui::DragFloat("Lacunarity", &domain.turbulence_lacunarity, 0.02f, 1.0f, 4.0f, "%.2f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Frequency multiplier per octave (typical ~2.0).");
                            ImGui::DragFloat("Persistence", &domain.turbulence_persistence, 0.02f, 0.0f, 1.0f, "%.2f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Amplitude decay per octave (typical ~0.5).");
                            ImGui::DragFloat("Evolution Speed", &domain.turbulence_speed, 0.02f, 0.0f, 5.0f, "%.2f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("How fast the turbulence field animates over time.");
                        }
                    }
                } else {
                    auto& fp = domain.fluid_params;
                    // Fluid Seeding & Limits
                    if (ImGui::CollapsingHeader("Fluid Seeding & Capacity", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Spacing();

                    // Auto-reseed toggle (the accumulator + the actual reseed live
                    // at the top/bottom of this panel so shape params count too).
                    ImGui::Checkbox("Auto Reseed on Edit", &s_fluid_auto_reseed);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("When on, changing a seed OR grid-shape parameter (fill level, wall\n"
                                          "margin, seed box, particles/cell, max particles, seed mode, resolution,\n"
                                          "voxel size, domain bounds) automatically re-seeds the fluid and snaps\n"
                                          "the timeline to frame 0 when you release the control \xE2\x80\x94 no manual\n"
                                          "Reset + Seed Fluid Now. Solver params (viscosity, blends, ...) apply\n"
                                          "live and never reseed.");
                    }

                    using RayTrophiSim::FluidSeedMode;
                    const char* seed_mode_labels[] = { "Seed Box", "Fill Domain (resting tank)" };
                    int seed_mode_idx = static_cast<int>(domain.fluid_seed_mode);
                    ImGui::SetNextItemWidth(250.0f);
                    if (ImGui::Combo("Seed Mode", &seed_mode_idx, seed_mode_labels, IM_ARRAYSIZE(seed_mode_labels))) {
                        domain.fluid_seed_mode = static_cast<FluidSeedMode>(seed_mode_idx);
                        seed_settled = true;
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Seed Box: fill a user-positioned region (good for a localized blob you then drop/emit).\n"
                                          "Fill Domain: pre-fill the whole domain footprint from the floor up to the fill level "
                                          "as a resting tank \xE2\x80\x94 skips the long settling transient for standing water; "
                                          "colliders then carve waves on top.");
                    }

                    if (domain.fluid_seed_mode == FluidSeedMode::FillLevel) {
                        ImGui::SetNextItemWidth(250.0f);
                        ImGui::SliderFloat("Fill Level (target)", &domain.fluid_fill_level, 0.0f, 1.0f, "%.2f");
                        seed_settled |= ImGui::IsItemDeactivatedAfterEdit();
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("TARGET fraction of the domain height filled with liquid at rest.\n"
                                              "0.5 = half-full, 1.0 = brim-full. The ACTUAL level is capped by the\n"
                                              "particle budget below: ppc stays fixed for stability, so when the\n"
                                              "budget can't reach the target the level drops (complete layers from\n"
                                              "the floor up), never the density.");
                        }
                        ImGui::SetNextItemWidth(250.0f);
                        ImGui::DragFloat("Wall Margin", &domain.fluid_fill_wall_margin, 0.01f, 0.0f, 10000.0f, "%.3f");
                        seed_settled |= ImGui::IsItemDeactivatedAfterEdit();
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("World-unit inset from the side walls (X/Z). Leave 0 to fill wall-to-wall.");
                        }

                        // Budget readout. ppc is fixed for stability; the budget caps the
                        // ACTUAL fill HEIGHT (complete layers from the floor up), so show
                        // both the target and the effective level the budget reaches, plus
                        // a one-click "raise cap to hit target".
                        {
                            const int req_ppc = std::max(1, domain.fluid_seed_particles_per_cell);
                            // Effective fill at the current budget (replace-seed assumed).
                            Vec3 eff_lo, eff_hi;
                            const float eff_level = RayTrophiSim::computeFluidFillSeedAABB(
                                domain.bounds_min, domain.bounds_max, domain.voxel_size,
                                domain.fluid_fill_level, domain.fluid_fill_wall_margin,
                                req_ppc, domain.fluid_max_particles, eff_lo, eff_hi);
                            // Target need (no budget cap): cells in target region * ppc.
                            const float fl_m = std::max(0.0f, domain.fluid_fill_wall_margin);
                            const float fl_lvl = std::clamp(domain.fluid_fill_level, 0.0f, 1.0f);
                            const Vec3 tb_lo(domain.bounds_min.x + fl_m, domain.bounds_min.y, domain.bounds_min.z + fl_m);
                            const Vec3 tb_hi(domain.bounds_max.x - fl_m,
                                             domain.bounds_min.y + (domain.bounds_max.y - domain.bounds_min.y) * fl_lvl,
                                             domain.bounds_max.z - fl_m);
                            const std::size_t target_needed = RayTrophiSim::Fluid::estimateSeedBoxParticleCount(
                                domain.bounds_min, domain.resolution_x, domain.resolution_y, domain.resolution_z,
                                domain.voxel_size, tb_lo, tb_hi, req_ppc);

                            ImGui::TextDisabled("Target %.2f needs ~%zu particles @ %d ppc",
                                                fl_lvl, target_needed, req_ppc);
                            const bool budget_limited = eff_level < fl_lvl - 1e-3f;
                            if (budget_limited) {
                                ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.2f, 1.0f),
                                    "Budget reaches level ~%.2f (cap %zu). Liquid stays\n"
                                    "stable @ %d ppc, just shallower. Raise cap to hit target.",
                                    eff_level, domain.fluid_max_particles, req_ppc);
                                if (ImGui::Button("Set Max Particles to hit target##FitFill")) {
                                    domain.fluid_max_particles = target_needed + target_needed / 10u + 1000u; // +10%
                                    seed_settled = true;
                                }
                            } else {
                                ImGui::TextDisabled("Budget OK \xE2\x80\x94 reaches the target level.");
                            }
                        }
                    } else {
                    ImGui::DragFloat3("Fluid Seed Box Min", &domain.fluid_seed_min.x, 0.05f, -10000.0f, 10000.0f, "%.2f");
                    seed_settled |= ImGui::IsItemDeactivatedAfterEdit();
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Minimum coordinates of the initial volume region containing liquid at startup.");
                    }
                    ImGui::DragFloat3("Fluid Seed Box Max", &domain.fluid_seed_max.x, 0.05f, -10000.0f, 10000.0f, "%.2f");
                    seed_settled |= ImGui::IsItemDeactivatedAfterEdit();
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Maximum coordinates of the initial volume region containing liquid at startup.");
                    }
                    }

                    ImGui::SetNextItemWidth(250.0f);
                    ImGui::SliderInt("Particles Per Voxel", &domain.fluid_seed_particles_per_cell, 2, 16);
                    seed_settled |= ImGui::IsItemDeactivatedAfterEdit();
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Particles spawned per grid cell. This is a STABILITY constant, not a\n"
                                          "budget knob: at 1 ppc the cells can't build internal pressure and the\n"
                                          "liquid just collapses / settles slowly. Keep 4-8 for standard water.\n"
                                          "To fit a budget, change Voxel Size or Max Particles \xE2\x80\x94 not this.");
                    }

                    ImGui::SetNextItemWidth(250.0f);
                    int max_particles_ui = static_cast<int>(std::min<std::size_t>(domain.fluid_max_particles, 10000000u));
                    if (ImGui::DragInt("Max Particles Limit", &max_particles_ui, 1000.0f, 1000, 10000000)) {
                        domain.fluid_max_particles = static_cast<std::size_t>(std::max(1000, max_particles_ui));
                    }
                    seed_settled |= ImGui::IsItemDeactivatedAfterEdit();
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Maximum total active particles allowed to prevent VRAM or RAM overflow.");
                    }

                    ImGui::Checkbox("Clear Existing on Seed##ReplaceOnSeed", &domain.fluid_replace_on_seed);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("When enabled, clicking 'Seed Fluid' clears all pre-existing particles before seeding.");
                    }

                    if (ImGui::Button("Seed Fluid Now##SeedButton", ImVec2(-1, 30))) {
                        domain.fluid_pending_seed = true;
                        particles->synchronizeGridDomainsNow();
                        ui_ctx.start_render = true;
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Places liquid particles within the specified box coordinates.");
                    }
                    }

                    // APIC Solver Params
                    if (ImGui::CollapsingHeader("APIC / FLIP Liquid Solver Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Spacing();
                    ImGui::TextDisabled("Material Preset");
                    if (drawFluidPresetCombo("##GridFluidSolverPreset", fp)) {
                        ui_ctx.start_render = true;
                    }
                    // Manual edits to any preset-driven rheology field demote the
                    // dropdown to "Custom" so it stops claiming a stale material.
                    bool fp_edited = false;
                    ImGui::Spacing();
                    ImGui::DragFloat3("Gravity Force Vector", &fp.gravity.x, 0.05f, -100.0f, 100.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Gravitational acceleration applied to the fluid. Use (0, -9.81, 0) for Earth gravity.");
                    }

                    fp_edited |= ImGui::SliderFloat("APIC Momentum Blend", &fp.apic_blend, 0.0f, 1.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Preservation of angular momentum vs linear velocity.\n\n"
                                          "0.0 = Viscous, highly-damped flow (PIC).\n"
                                          "1.0 = Pure APIC. Values between 0.95 and 0.98 yield the most realistic swirls and splash turbulence for water.");
                    }
                    fp_edited |= ImGui::SliderFloat("FLIP Particle Blend", &fp.flip_blend, 0.0f, 1.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("The degree of dynamic particle splashing.\n\n"
                                          "0.0 = Damped, stable PIC movement.\n"
                                          "1.0 = Highly energetic, splashy FLIP motion. Around 0.97 prevents excessive chaotic noise.");
                    }

                    fp_edited |= ImGui::DragFloat("Velocity Damping", &fp.velocity_damping, 0.001f, 0.5f, 1.0f, "%.3f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Velocity damping factor applied per step. 1.0 = frictionless flow, <1.0 = viscous slowdown.");
                    }
                    fp_edited |= ImGui::DragFloat("Internal Viscous Friction", &fp.internal_friction, 0.01f, 0.0f, 10.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Viscous friction resistance between fluid particles. Higher values damp motion.");
                    }
                    fp_edited |= ImGui::DragFloat("Air Drag Resistance", &fp.air_drag, 0.01f, 0.0f, 10.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Aerodynamic friction drag applied to airborne splashing particles.");
                    }
                    fp_edited |= ImGui::DragFloat("Wall Friction Damping", &fp.wall_damping, 0.01f,  0.0f, 1.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Friction factor applied when liquid rubs boundaries or solid colliders.\n"
                                          "0 = Slippery walls (sliding), 1 = Sticky walls (no-slip).");
                    }

                    ImGui::SliderFloat("Domain Motion Coupling", &fp.domain_motion_coupling, 0.0f, 1.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Couples domain coordinate translation to fluid velocity. Allows creating sloshing liquids inside a moving cup.");
                    }

                    fp_edited |= ImGui::DragFloat("Viscosity Strength", &fp.viscosity, 0.01f,  0.0f, 30.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Fluid thickness and flow resistance (Laplacian velocity diffusion).\n"
                                          "0 = water, ~3 = oil, ~10 = mud, ~20 = honey, ~30 = lava.");
                    }
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(120.0f);
                    fp_edited |= ImGui::DragInt("Viscosity Iterations", &fp.viscosity_iterations, 1.0f, 1, 16);

                    fp_edited |= ImGui::DragFloat("Density Correction Strength", &fp.density_correction, 0.05f, 0.0f, 10.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Repulsive force preventing particles from clustering too close. Helps maintain fluid incompressibility. ~1.0 is recommended.");
                    }

                    ImGui::Checkbox("Free Surface Pressure Boundary", &fp.free_surface);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("ON: Sets pressure to zero at surface air boundaries, creating natural free-surface waves.\n"
                                          "OFF: Simulates enclosed pressurized fluid flow.");
                    }
                    ImGui::Checkbox("Ghost Fluid Method (GFM) Surface", &fp.ghost_fluid_surface);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Ghost Fluid Method (GFM) models sub-cell pressure extrapolation at the air-fluid boundary to eliminate staircasing/aliasing.");
                    }

                    ImGui::DragFloat("CFL Stability Factor", &fp.cfl,              0.01f,  0.05f, 1.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Courant stability factor limits timestep. Lower is more stable but requires more substeps.");
                    }
                    ImGui::DragInt("Max Solver Substeps", &fp.max_substeps,     1.0f,   1, 64);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Maximum number of substeps the solver takes to maintain stability within CFL safety.");
                    }
                    ImGui::DragInt("Poisson Pressure Iterations", &fp.pressure_iterations, 1.0f, 0, 200);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Iterations for solving incompressibility (Poisson equation). Higher values prevent compression.");
                    }
                    ImGui::DragFloat("Pressure Residual Target", &fp.pressure_relative_residual, 1.0e-6f, 1.0e-8f, 1.0e-2f, "%.1e");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Relative residual target for CPU PCG / GPU MGPCG pressure solve.\n"
                                          "1e-5 matches the current high-accuracy default; 1e-4 can reduce GPU dot-sync cost in heavy previews.");
                    }
                    ImGui::Checkbox("Pressure Layer B V-cycle", &fp.pressure_multigrid_preconditioner);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Experimental CUDA MGPCG multigrid preconditioner.\n"
                                          "Can cut iteration count on large grids, but adds extra dispatch work per iteration.");
                    }
                    if (fp_edited) {
                        fp.current_preset = RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Custom;
                    }
                    }

                    // Redistribution / Reseed settings
                    if (ImGui::CollapsingHeader("Dynamic Particle Reseeding (Reseed)", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Spacing();

                    ImGui::Checkbox("Enable Dynamic Reseeding##Reseed", &fp.reseed_enabled);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Dynamically generates and deletes particles to preserve density, maintain smooth level-set render boundaries, and prevent leaks.");
                    }

                    if (fp.reseed_enabled) {
                        ImGui::DragInt("Target Particles Per Cell", &fp.reseed_target_per_cell, 0.1f, 0, 64);
                        ImGui::DragInt("Minimum Threshold Per Cell", &fp.reseed_min_per_cell, 0.1f, 1, 32);
                        ImGui::DragInt("Maximum Threshold Per Cell", &fp.reseed_max_per_cell, 0.1f, 2, 64);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("If the particle count inside a cell drops below minimum, new particles spawn. If it exceeds maximum, excess is culled.");
                        }
                    } else {
                        ImGui::TextDisabled("Dynamic reseeding is disabled. Total particle count remains constant during simulation.");
                    }
                    }
                }

                ImGui::EndTabItem();
            }

            // =================================================================
            // TAB 3: Shading & Rendering (Visual Materials, Flow Sources & Baking)
            // =================================================================
            if (ImGui::BeginTabItem("Shading & Rendering")) {
                ImGui::Spacing();

                // Group 1: Flow Sources
                if (ImGui::CollapsingHeader("Flow Sources Registry", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Spacing();

                const bool can_add_object_flow =
                    ui_ctx.selection.selected.type == SelectableType::Object &&
                    ui_ctx.selection.selected.object != nullptr &&
                    !ui_ctx.selection.selected.object->getNodeName().empty();
                
                if (!can_add_object_flow) ImGui::BeginDisabled();
                if (ImGui::Button("Add Flow Source From Selection##DomainFlow", ImVec2(-1, 28))) {
                    // Mirror the Point-source path exactly: build the desc inline
                    // and add it straight to THIS panel's runtime with the
                    // currently-selected domain_index. The old scene-level helper
                    // hard-coded gas channels (density/temperature) and an upward
                    // velocity, so a source dropped on a FLUID domain carried gas
                    // parameters — it read as if it were emitting into the gas
                    // system too. Now the defaults follow the selected domain's
                    // type, and only one source is created, on that one domain.
                    RayTrophiSim::SimulationFlowSourceDesc desc;
                    const std::string node = ui_ctx.selection.selected.object->getNodeName();
                    desc.name = node.empty() ? "Object Flow Source" : node + " Flow";
                    desc.source_mode = RayTrophiSim::SimulationFlowSourceMode::ObjectBounds;
                    desc.source_name = node;
                    desc.domain_index = selected_domain_index;
                    if (is_fluid_domain) {
                        desc.velocity = Vec3(0.0f, -1.0f, 0.0f); // pour down
                        desc.density = 0.0f; desc.temperature = 0.0f; desc.fuel = 0.0f;
                    } else {
                        desc.velocity = Vec3(0.0f, 1.0f, 0.0f);  // plume up
                        desc.density = 2.0f; desc.temperature = 0.6f; desc.fuel = 0.0f;
                    }
                    Vec3 mn_b, mx_b;
                    if (scene.resolveObjectBoundsForSimulation(node, mn_b, mx_b)) {
                        const Vec3 lo = Vec3::min(mn_b, mx_b);
                        const Vec3 hi = Vec3::max(mn_b, mx_b);
                        desc.position = (lo + hi) * 0.5f;
                        desc.radius = std::max(0.05f, (hi - lo).length() * 0.25f);
                    }
                    particles->addFlowSource(desc);
                }
                if (ImGui::IsItemHovered() && can_add_object_flow) {
                    ImGui::SetTooltip("Injects a dynamic flow source emitting smoke or liquid utilizing the volume or surface shell of the selected 3D mesh.");
                }
                if (!can_add_object_flow) ImGui::EndDisabled();

                if (ImGui::Button("Add Point Flow Source##DomainFlow", ImVec2(-1, 28))) {
                    RayTrophiSim::SimulationFlowSourceDesc desc;
                    desc.name = "Point Flow Source";
                    desc.source_mode = RayTrophiSim::SimulationFlowSourceMode::Point;
                    desc.domain_index = selected_domain_index;
                    desc.position = (Vec3::min(domain.bounds_min, domain.bounds_max) +
                                     Vec3::max(domain.bounds_min, domain.bounds_max)) * 0.5f;
                    // A liquid source pours DOWN (gravity + emission agree); a gas
                    // source blasts UP (smoke/fire rises). The old shared (0,1,0)
                    // default made liquids shoot upward and bunch into a falling
                    // plate at the trajectory apex.
                    desc.velocity = is_fluid_domain ? Vec3(0.0f, -1.0f, 0.0f) : Vec3(0.0f, 1.0f, 0.0f);
                    particles->addFlowSource(desc);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Injects a spherical point flow emitter at the center of the domain bounds.");
                }

                ImGui::Spacing();
                ImGui::Separator();
                
                auto& flow_sources = particles->flowSources();
                int remove_flow_index = -1;
                int visible_flow_count = 0;
                
                // Draw flow sources as a flat list integrated into the collapsing section
                for (int flow_i = 0; flow_i < static_cast<int>(flow_sources.size()); ++flow_i) {
                    auto& source = flow_sources[static_cast<std::size_t>(flow_i)];
                    if (source.domain_index != selected_domain_index) continue;
                    
                    ++visible_flow_count;
                    ImGui::PushID(92000 + flow_i);
                    
                    ImGui::Spacing();
                    ImGui::Checkbox("Flow Source Enabled##FlowEnabled", &source.enabled);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Enables/disables active injection from this flow source.");
                    }
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "[%s]", source.name.c_str());

                    const char* mode_labels[] = { "Point (Sphere)", "Object Bounding Box", "Mesh Geometry Surface" };
                    int mode_idx = static_cast<int>(source.source_mode);
                    if (ImGui::Combo("Emitter Geometry Type##FlowMode", &mode_idx, mode_labels, IM_ARRAYSIZE(mode_labels))) {
                        source.source_mode = static_cast<RayTrophiSim::SimulationFlowSourceMode>(mode_idx);
                    }

                    ImGui::DragFloat("Source Radius", &source.radius, 0.01f, 0.001f, 1000.0f, "%.3f");
                    if (ImGui::IsItemHovered()) {
                        switch (source.source_mode) {
                            case RayTrophiSim::SimulationFlowSourceMode::Point:
                                ImGui::SetTooltip("Radius of the spherical spawn volume around the source position.");
                                break;
                            case RayTrophiSim::SimulationFlowSourceMode::MeshSurface:
                                ImGui::SetTooltip("Distance particles spawn off the mesh surface along its normal\n"
                                                  "(prevents embedding the spawn inside the geometry).");
                                break;
                            default:
                                ImGui::SetTooltip("Spawn-volume radius. For Object Bounding Box mode the box itself\n"
                                                  "defines the volume; radius only widens the gas injection falloff.");
                                break;
                        }
                    }
                    source.radius = std::max(0.001f, source.radius);

                    if (is_fluid_domain) {
                        ImGui::DragFloat("Injected Particles / Sec", &source.fluid_particles_per_second, 10.0f, 0.0f, 1000000.0f, "%.0f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Flow rate of liquid particles spawned per second.");
                        }
                        source.fluid_particles_per_second = std::max(0.0f, source.fluid_particles_per_second);
                    } else {
                        ImGui::DragFloat("Soot Density Rate",     &source.density,     0.05f, 0.0f, 1000.0f, "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Rate of visual smoke density injected per second.");
                        }
                        ImGui::DragFloat("Thermal Temperature Rate", &source.temperature, 0.05f, 0.0f, 1000.0f, "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Thermal energy injected per second, making gas rise faster due to buoyancy.");
                        }
                        ImGui::DragFloat("Combustion Fuel Rate",        &source.fuel,        0.05f, 0.0f, 1000.0f, "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Quantity of combustible fuel injected per second for fire rendering.");
                        }
                        ImGui::DragFloat("Radial Falloff Blend",     &source.falloff,     0.05f, 0.0f, 16.0f,    "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Radial falloff interpolation from emitter core to boundaries.");
                        }
                        source.falloff = std::max(0.0f, source.falloff);
                    }
                    ImGui::DragFloat3("Emission Velocity", &source.velocity.x, 0.05f, -10000.0f, 10000.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Initial velocity (m/s) given to each emitted particle / injected cell.\n"
                                          "This is a one-time launch velocity, NOT a continuous force — gravity and\n"
                                          "buoyancy take over afterwards. Liquid: point it down to pour; gas: up to plume.");
                    }

                    if (is_fluid_domain) {
                        ImGui::DragFloat("Velocity Spread", &source.fluid_velocity_spread, 0.01f, 0.0f, 2.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Per-particle random velocity jitter, as a fraction of the emission speed.\n"
                                              "0 = laminar (all particles share one velocity) — the stream stays a\n"
                                              "coherent sheet that falls as a slab and only breaks up on impact.\n"
                                              "0.1-0.3 disperses the stream at the source so it flows like water.");
                        }
                        source.fluid_velocity_spread = std::max(0.0f, source.fluid_velocity_spread);

                        if (source.source_mode == RayTrophiSim::SimulationFlowSourceMode::MeshSurface) {
                            ImGui::Checkbox("Emit Along Surface Normal", &source.fluid_emit_along_normal);
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("Spray particles outward along each surface point's normal\n"
                                                  "(speed = the Emission Velocity magnitude) instead of using the\n"
                                                  "single velocity vector. Makes liquid follow the mesh shape.");
                            }
                        }
                    }

                    if (source.source_mode == RayTrophiSim::SimulationFlowSourceMode::Point) {
                        ImGui::DragFloat3("World Coordinates Position", &source.position.x, 0.05f, -10000.0f, 10000.0f, "%.2f");
                    } else {
                        ImGui::TextDisabled("Linked Scene Mesh: %s", source.source_name.empty() ? "None" : source.source_name.c_str());
                    }

                    ImGui::Spacing();
                    if (ImGui::CollapsingHeader("Flow Control & Emission Limits##LimitsHeader")) {
                        ImGui::Indent();
                        
                        // Time Limits
                        ImGui::Checkbox("Use Time Range Limit##TimeLimit", &source.use_time_limit);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Restricts flow source injection to a specific timeline duration.");
                        }
                        if (source.use_time_limit) {
                            ImGui::DragFloat("Start Time (s)##StartTime", &source.start_time, 0.05f, 0.0f, 10000.0f, "%.2fs");
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("Time step at which emission starts.");
                            }
                            ImGui::DragFloat("End Time (s)##EndTime", &source.end_time, 0.05f, 0.0f, 10000.0f, "%.2fs");
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("Time step at which emission stops.");
                            }
                            if (source.start_time > source.end_time) source.end_time = source.start_time;
                        }
                        
                        // Particle Budget limits (only for fluid domains)
                        if (is_fluid_domain) {
                            ImGui::Spacing();
                            ImGui::Checkbox("Use Particle Budget Limit##ParticleLimit", &source.use_particle_limit);
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("Limits the total number of particles this flow source can inject.");
                            }
                            if (source.use_particle_limit) {
                                ImGui::DragInt("Max Particle Budget##MaxParticles", &source.max_emitted_particles, 100, 1, 10000000);
                                if (ImGui::IsItemHovered()) {
                                    ImGui::SetTooltip("Maximum particles allowed to spawn from this emitter.");
                                }
                                source.max_emitted_particles = std::max(1, source.max_emitted_particles);
                                
                                // Live Budget Progress Feedback HUD
                                float progress = 0.0f;
                                if (source.max_emitted_particles > 0) {
                                    progress = static_cast<float>(source.total_emitted_particles) / static_cast<float>(source.max_emitted_particles);
                                }
                                progress = std::min(1.0f, std::max(0.0f, progress));
                                char buf[64];
                                sprintf(buf, "Emitted: %d / %d", source.total_emitted_particles, source.max_emitted_particles);
                                ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), buf);
                                if (ImGui::IsItemHovered()) {
                                    ImGui::SetTooltip("Percentage of the particle emission budget currently utilized.");
                                }
                            } else {
                                ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Lifetime Emitted Particles: %d", source.total_emitted_particles);
                            }
                        }
                        
                        ImGui::Unindent();
                    }
                    ImGui::Spacing();
                    
                    if (ImGui::Button("Delete Flow Source##FlowRem", ImVec2(-1, 24))) {
                        remove_flow_index = flow_i;
                    }
                    ImGui::Separator();
                    ImGui::PopID();
                }
                if (visible_flow_count == 0) {
                    ImGui::TextDisabled("No active flow sources registered for this domain.");
                }
                // End flat flow sources listing
                
                if (remove_flow_index >= 0) {
                    particles->removeFlowSource(static_cast<std::size_t>(remove_flow_index));
                }
                }

                if (is_gas_domain) {
                    // Volume shader
                    if (!domain.shader) {
                        domain.shader = VolumeShader::createSmokePreset();
                    }
                    if (ImGui::CollapsingHeader("Unified Volume Shader Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Spacing();
                    
                    if (SceneUI::drawVolumeShaderUI(ui_ctx, domain.shader, nullptr, nullptr)) {
                        g_gas_volumes_dirty = true;
                        ui_ctx.start_render = true;
                    }
                    }
                } else {
                    // Fluid Render settings group
                    if (ImGui::CollapsingHeader("Liquid Visualization & Shading", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Spacing();

                    int current_mode_idx = 0; // default to Particles
                    if (domain.fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) {
                        current_mode_idx = 1;
                    } else if (domain.fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::Volume) {
                        domain.fluid_render_mode = RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
                        current_mode_idx = 1;
                    }
                    const char* fluid_render_modes[] = { "Splat Spheres (Fast Preview)", "Smooth Glassy Surface (Level Set SDF)" };
                    if (ImGui::Combo("Visualization Mode##DomainFluid", &current_mode_idx,
                                     fluid_render_modes, 2)) {
                        domain.fluid_render_mode = (current_mode_idx == 0)
                             ? RayTrophiSim::Fluid::FluidRenderMode::Particles
                             : RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
                        scene.requestSimulationTimelineRenderResync();
                        ui_ctx.start_render = true;
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Choose how the liquid particles are visualised:\n\n"
                                          "1. Splat Spheres: Renders individual particles as solid spheres (high performance).\n"
                                          "2. Smooth Surface: Reconstructs a glassy, refractive fluid mesh boundary.");
                    }

                    if (ImGui::Checkbox("Debug Particle Points Overlay##DomainFluid", &domain.fluid_debug_overlay)) {
                        ui_ctx.start_render = true;
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Draws raw simulation particle coordinates as lightweight blue viewport overlays.");
                    }

                    if (domain.fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) {
                        auto& mgr = MaterialManager::getInstance();
                        const auto& all_mats = mgr.getAllMaterials();
                        const char* current_label = "Auto (Color + Glow)";
                        if (domain.fluid_particle_material_id >= 0 &&
                            domain.fluid_particle_material_id != MaterialManager::INVALID_MATERIAL_ID &&
                            static_cast<std::size_t>(domain.fluid_particle_material_id) < all_mats.size()) {
                            current_label = all_mats[domain.fluid_particle_material_id]
                                                 ? all_mats[domain.fluid_particle_material_id]->materialName.c_str()
                                                 : "(missing)";
                        }
                        if (ImGui::BeginCombo("Refractive Material Override##DomainFluid", current_label)) {
                            const bool none_sel = (domain.fluid_particle_material_id < 0);
                            if (ImGui::Selectable("Auto (Color + Glow)", none_sel)) {
                                domain.fluid_particle_material_id = -1;
                                ui_ctx.start_render = true;
                            }
                            for (std::size_t mi = 0; mi < all_mats.size(); ++mi) {
                                if (!all_mats[mi]) continue;
                                const bool sel = (domain.fluid_particle_material_id == static_cast<int>(mi));
                                ImGui::PushID(static_cast<int>(mi));
                                if (ImGui::Selectable(all_mats[mi]->materialName.c_str(), sel)) {
                                    domain.fluid_particle_material_id = static_cast<int>(mi);
                                    ui_ctx.start_render = true;
                                }
                                ImGui::PopID();
                            }
                            ImGui::EndCombo();
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Overrides fluid rendering with a custom scene material. Set to 'Auto' to use raw color/emissive controls below.");
                        }

                        const bool auto_mat = domain.fluid_particle_material_id < 0;
                        if (auto_mat) {
                            ImGui::ColorEdit3("Raw Particle Color##DomainFluid", &domain.fluid_particle_color.x);
                            ImGui::Checkbox("Self-Emissive Glow##DomainFluid", &domain.fluid_particle_emissive);
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("Enables raw glowing luminance (useful for lava, glowing acid, or magical effects).");
                            }
                            if (domain.fluid_particle_emissive) {
                                ImGui::DragFloat("Emissive Glow Intensity##DomainFluid", &domain.fluid_particle_emission, 0.05f, 0.0f, 50.0f, "%.2f");
                            }
                        }
                        ImGui::DragFloat("Voxel Radius Factor##DomainFluid", &domain.fluid_particle_radius_factor, 0.01f, 0.05f, 1.5f, "%.2f");
                        ImGui::DragFloat("Visual Size Multiplier##DomainFluid", &domain.fluid_particle_size_multiplier, 0.01f, 0.05f, 8.0f, "%.2f");
                        ImGui::SliderInt("Sphere Subdivision Detail##DomainFluid", &domain.fluid_particle_subdivisions, 0, 3);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Geometric subdivision level of particle spheres. Higher values are smoother but reduce rendering performance.");
                        }
                    }

                    if (domain.fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) {
                        bool sdf_changed = false;
                        sdf_changed |= ImGui::DragFloat("Level Set Kernel Radius", &domain.fluid_level_set_params.kernel_radius_voxels, 0.05f, 0.5f, 6.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Splats radius determining how far apart particles fuse into a unified liquid body.\nLarger values produce a thicker, fuller appearance.");
                        }
                        sdf_changed |= ImGui::DragFloat("Particle Voxel Radius (vx)", &domain.fluid_level_set_params.particle_radius_voxels, 0.02f, 0.05f, 2.0f, "%.2f");
                        sdf_changed |= ImGui::DragFloat("SDF Narrow Band Width", &domain.fluid_level_set_params.narrow_band_voxels, 0.05f, 1.0f, 8.0f, "%.2f");
                        sdf_changed |= ImGui::DragFloat("SDF Surface Band Width", &domain.fluid_surface_band_voxels, 0.02f, 0.1f, 3.0f, "%.2f");
                        sdf_changed |= ImGui::SliderInt("Laplacian Surface Smoothing", &domain.fluid_level_set_params.smoothing_iterations, 0, 8);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Number of Laplacian smoothing passes applied to the surface level-set boundary. Prevents voxel stair-stepping.");
                        }
                        sdf_changed |= ImGui::SliderInt("Surface Detail (x sim grid)", &domain.fluid_level_set_params.surface_resolution_multiplier, 1, 4);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Reconstructs the render surface on a grid finer than the simulation\n"
                                              "(1 = same, 2 = half-voxel, ...). This is THE knob for detailed wavy/\n"
                                              "rocky coastlines: it adds sub-voxel surface detail WITHOUT raising\n"
                                              "the (cubic) simulation cost. SDF build + upload scale x^3, so keep it\n"
                                              "modest on large domains. The surface shape is unchanged - only its fineness.");
                        }
                        if (domain.fluid_level_set_params.surface_resolution_multiplier > 1) {
                            if (auto* sysp = scene.activeParticleSystemObject()) {
                                const std::size_t d = static_cast<std::size_t>(selected_domain_index);
                                if (d < sysp->domain_sdf_stats.size() && sysp->domain_sdf_stats[d].eff_nx > 0) {
                                    const auto& st = sysp->domain_sdf_stats[d];
                                    ImGui::TextDisabled("  Surface grid: %dx%dx%d (refined)", st.eff_nx, st.eff_ny, st.eff_nz);
                                }
                            }
                        }

                        ImGui::Spacing();
                        ImGui::TextColored(ImVec4(0.55f, 0.85f, 1.0f, 1.0f), "Metaball Blending (Anisotropic)");
                        sdf_changed |= ImGui::Checkbox("Anisotropic Kernel (clean merge)", &domain.fluid_level_set_params.anisotropy_enabled);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Yu-Turk 2013 anisotropic kernels. Orients/stretches each particle's\n"
                                              "splat by its neighbourhood shape: flat sheets stay flat, thin films\n"
                                              "and droplet 'necks' are cleaned, and close drops merge smoothly\n"
                                              "(metaball-like) instead of bumpy sphere unions. OFF = plain isotropic.");
                        }
                        if (domain.fluid_level_set_params.anisotropy_enabled) {
                            ImGui::Indent();
                            sdf_changed |= ImGui::DragFloat("Neighbour Radius (vx)##AnisoNeighbourRadius", &domain.fluid_level_set_params.anisotropy_radius_voxels, 0.05f, 1.0f, 6.0f, "%.2f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Radius (sim voxels) used to estimate each particle's local shape. ~2-3.");
                            sdf_changed |= ImGui::DragFloat("Max Stretch", &domain.fluid_level_set_params.anisotropy_max_stretch, 0.05f, 1.0f, 8.0f, "%.2f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max ellipsoid axis ratio. 1 = isotropic, 4 = strong sheet flattening.\nHigher widens the stencil (cost up).");
                            sdf_changed |= ImGui::DragFloat("Position Smoothing", &domain.fluid_level_set_params.position_smoothing, 0.01f, 0.0f, 1.0f, "%.2f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Pull particles toward their neighbour mean before surfacing.\n0 = raw (bumpy), 1 = fully smoothed.");
                            sdf_changed |= ImGui::SliderInt("Isolated Min Neighbours", &domain.fluid_level_set_params.anisotropy_neighbor_min, 1, 24);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Below this neighbour count a particle is treated as isolated spray\nand kept spherical (round droplets).");
                            ImGui::Unindent();
                        }

                        if (sdf_changed) {
                            scene.requestSimulationTimelineRenderResync();
                            ui_ctx.renderer.resetCPUAccumulation();
                            if (ui_ctx.backend_ptr) ui_ctx.backend_ptr->resetAccumulation();
                            ui_ctx.start_render = true;
                        }

                        bool mat_changed = false;
                        mat_changed |= ImGui::DragFloat("Index of Refraction (IOR)", &domain.fluid_surface_ior, 0.005f, 1.0f, 2.5f, "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Refractive index bending light passing through the liquid:\n1.333 = Water, 1.47 = Glycerin, 1.5 = Glass.");
                        }
                        mat_changed |= ImGui::DragFloat("Surface Roughness", &domain.fluid_surface_roughness, 0.005f, 0.0f, 1.0f, "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Microfacet roughness of the glassy liquid interface. 0.0 = perfectly mirror reflective, >0.0 = frosted reflection.");
                        }
                        mat_changed |= ImGui::DragFloat("Splash Foam Intensity", &domain.fluid_surface_foam, 0.005f, 0.0f, 1.0f, "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Luminance intensity of foam generated in high-velocity turbulent regions.");
                        }
                        if (mat_changed) {
                            scene.refreshFluidSurfaceMaterial();
                            ui_ctx.renderer.resetCPUAccumulation();
                            if (ui_ctx.backend_ptr) ui_ctx.backend_ptr->resetAccumulation();
                            ui_ctx.start_render = true;
                        }

                        if (auto* sys = scene.activeParticleSystemObject()) {
                            const std::size_t d = static_cast<std::size_t>(selected_domain_index);
                            if (d < sys->domain_sdf_stats.size()) {
                                const auto& st = sys->domain_sdf_stats[d];
                                ImGui::TextDisabled("Level-Set SDF Stats:\n  %zu active / %zu surface cells (SDF Build: %.2f ms)",
                                                     st.active_cells, st.surface_cells, st.build_ms);
                            }
                        }
                    }
                    }

                    // ── Whitewater (Foam / Spray / Bubbles) — Ihmsen 2012 ────
                    if (ImGui::CollapsingHeader("Whitewater (Foam / Spray / Bubbles)")) {
                        ImGui::Spacing();
                        auto& fo = domain.fluid_foam_params;
                        bool foam_changed = false;
                        foam_changed |= ImGui::Checkbox("Enable Whitewater", &fo.enabled);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                            "Physically-generated secondary particles (Ihmsen 2012):\n"
                            "spray from impacts, foam on the surface, bubbles below.\n"
                            "Render-only - never affects the liquid solve. Has a cost.");
                        if (fo.enabled) {
                            ImGui::Indent();
                            ImGui::TextDisabled("Generation");
                            foam_changed |= ImGui::DragFloat("Trapped-Air Rate", &fo.trapped_air_rate, 1.0f, 0.0f, 400.0f, "%.0f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Spray/foam from converging high-relative-velocity particles (impacts, splashes).");
                            foam_changed |= ImGui::DragFloat("Wave-Crest Rate", &fo.wave_crest_rate, 1.0f, 0.0f, 400.0f, "%.0f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Spray off convex surface crests moving outward (breaking waves, lips).");
                            foam_changed |= ImGui::DragFloat("Neighbour Radius (vx)##FoamNeighbourRadius", &fo.neighbor_radius_voxels, 0.05f, 1.0f, 4.0f, "%.2f");

                            ImGui::Spacing(); ImGui::TextDisabled("Dynamics");
                            foam_changed |= ImGui::DragFloat("Lifetime (s)", &fo.lifetime, 0.05f, 0.1f, 20.0f, "%.2f");
                            foam_changed |= ImGui::DragFloat("Bubble Buoyancy", &fo.buoyancy, 0.05f, 0.0f, 8.0f, "%.2f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("How fast submerged bubbles rise against gravity.");
                            foam_changed |= ImGui::DragFloat("Fluid Coupling", &fo.fluid_drag, 0.1f, 0.0f, 30.0f, "%.1f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("How strongly foam/bubbles follow the liquid velocity (1/s).");
                            foam_changed |= ImGui::DragFloat("Spray Air Drag", &fo.spray_drag, 0.01f, 0.0f, 5.0f, "%.2f");

                            ImGui::Spacing(); ImGui::TextDisabled("Classification (fluid neighbours)");
                            foam_changed |= ImGui::SliderInt("Spray when <= N", &fo.spray_max_neighbors, 0, 30);
                            foam_changed |= ImGui::SliderInt("Bubble when >= N", &fo.bubble_min_neighbors, 4, 60);

                            ImGui::Spacing(); ImGui::TextDisabled("Render");
                            // Foam render mode: Spheres (one instanced sphere per
                            // particle — granular, but O(N) TLAS instances so it
                            // crawls at high counts) vs Volume (foam splatted into the
                            // fluid SURFACE volume's temperature channel and marched as
                            // a white single-scatter medium — cost ~independent of
                            // particle count; the production whitewater approach).
                            {
                                int foam_mode_idx =
                                    (fo.render_mode == RayTrophiSim::Fluid::FoamRenderMode::Volume) ? 1 : 0;
                                const char* foam_render_modes[] = { "Spheres (granular, per-particle)",
                                                                    "Volume (whitewater medium, fast)" };
                                if (ImGui::Combo("Foam Render##FoamMode", &foam_mode_idx, foam_render_modes, 2)) {
                                    fo.render_mode = (foam_mode_idx == 1)
                                        ? RayTrophiSim::Fluid::FoamRenderMode::Volume
                                        : RayTrophiSim::Fluid::FoamRenderMode::Spheres;
                                    // Structural: flips the sphere instance group on/off
                                    // AND the volume foam splat on/off — force a re-sync
                                    // so the current (paused) frame rebuilds both paths.
                                    scene.requestSimulationTimelineRenderResync();
                                    foam_changed = true;
                                }
                                if (ImGui::IsItemHovered())
                                    ImGui::SetTooltip("Spheres: one instanced sphere per foam particle (granular close-up, "
                                                      "but O(N) instances).\nVolume: foam rides the fluid surface volume's "
                                                      "temperature channel as a white single-scatter medium — cheap at "
                                                      "millions of particles, the production approach.");
                                if (fo.render_mode == RayTrophiSim::Fluid::FoamRenderMode::Volume &&
                                    domain.fluid_render_mode != RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) {
                                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                                        "  Volume foam needs Fluid Render = Surface SDF (it rides that volume).");
                                }
                            }

                            const bool foam_is_volume =
                                (fo.render_mode == RayTrophiSim::Fluid::FoamRenderMode::Volume);

                            if (foam_is_volume) {
                                // ── Volume whitewater medium look ──
                                // Colour + Opacity are pure shader state → push them
                                // live (refreshFluidSurfaceMaterial) so the CURRENT
                                // frame updates without a re-splat. Density changes the
                                // temp grid each foam particle deposits → it needs a
                                // re-upload (requestSimulationTimelineRenderResync).
                                ImGui::Spacing(); ImGui::TextDisabled("Whitewater medium");
                                bool foam_look_changed  = false;
                                bool foam_resplat_changed = false;
                                foam_look_changed |= ImGui::ColorEdit3("Foam Color", &fo.volume_color.x);
                                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Scattering tint of the foam medium (cool white = sea foam).");
                                // Density drives the re-splat (SDF/temp re-upload) — gate
                                // it to the drag RELEASE so dragging doesn't rebuild the
                                // level set every frame (settle pattern, like auto-reseed).
                                ImGui::DragFloat("Foam Density", &fo.volume_density, 0.02f, 0.05f, 20.0f, "%.2f");
                                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Density each foam particle deposits into the volume. Higher = thicker / whiter foam.");
                                if (ImGui::IsItemDeactivatedAfterEdit()) foam_resplat_changed = true;
                                foam_look_changed |= ImGui::DragFloat("Foam Opacity", &fo.volume_opacity, 0.1f, 0.1f, 64.0f, "%.1f");
                                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Extinction multiplier — how strongly the foam medium occludes what is behind it.");
                                // Per-class whitewater contribution (Ihmsen spray/foam/bubble).
                                // These change the deposited temp grid → settle-gated re-splat
                                // like Foam Density. Surface foam is always full strength.
                                ImGui::DragFloat("Bubble Froth", &fo.volume_bubble_strength, 0.02f, 0.0f, 4.0f, "%.2f");
                                if (ImGui::IsItemHovered()) ImGui::SetTooltip("How much submerged BUBBLE particles deposit into the foam medium.\nThey sit deep in the water so the depth tint colours them cooler — subsurface froth.\n0 = no bubbles, 1 = same as surface foam, >1 = brighter silvery pop.");
                                if (ImGui::IsItemDeactivatedAfterEdit()) foam_resplat_changed = true;
                                ImGui::DragFloat("Spray in Volume", &fo.volume_spray_strength, 0.02f, 0.0f, 4.0f, "%.2f");
                                if (ImGui::IsItemHovered()) ImGui::SetTooltip("How much airborne SPRAY particles deposit into the foam medium.\nThe liquid SDF already surfaces splashes, so spray can be dialled down to avoid double froth.");
                                if (ImGui::IsItemDeactivatedAfterEdit()) foam_resplat_changed = true;
                                if (foam_look_changed) {
                                    scene.refreshFluidSurfaceMaterial();
                                    foam_changed = true;
                                }
                                if (foam_resplat_changed) {
                                    scene.requestSimulationTimelineRenderResync();
                                    foam_changed = true;
                                }
                            } else {
                            // ── Foam material pickers (with custom overrides) — Spheres mode ──
                            {
                                static bool show_material_overrides = false;
                                static int last_domain_id = -1;
                                if (last_domain_id != selected_domain_index) {
                                    show_material_overrides = (fo.spray_material_id >= 0 || fo.bubble_material_id >= 0);
                                    last_domain_id = selected_domain_index;
                                }

                                ImGui::Checkbox("Custom Material Overrides", &show_material_overrides);
                                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Assign distinct scene materials to Spray, Foam, and Bubble independently.");

                                auto drawMaterialCombo = [&](const char* label, int& mat_id_ref, const char* default_label) {
                                    auto& mm = MaterialManager::getInstance();
                                    const size_t mcount = mm.getMaterialCount();
                                    std::string cur_label = (mat_id_ref < 0)
                                        ? std::string(default_label)
                                        : mm.getMaterialName(static_cast<uint16_t>(mat_id_ref));
                                    if (cur_label.empty()) cur_label = default_label;
                                    
                                    bool changed = false;
                                    if (ImGui::BeginCombo(label, cur_label.c_str())) {
                                        if (ImGui::Selectable(default_label, mat_id_ref < 0)) {
                                            mat_id_ref = -1; changed = true;
                                        }
                                        for (size_t i = 0; i < mcount; ++i) {
                                            const std::string nm = mm.getMaterialName(static_cast<uint16_t>(i));
                                            const bool sel = (mat_id_ref == static_cast<int>(i));
                                            const std::string lbl = nm.empty() ? ("Material " + std::to_string(i)) : nm;
                                            if (ImGui::Selectable(lbl.c_str(), sel)) {
                                                mat_id_ref = static_cast<int>(i); changed = true;
                                            }
                                        }
                                        ImGui::EndCombo();
                                    }
                                    return changed;
                                };

                                // Inline full editor for an assigned foam material so it can be
                                // tuned (Bubble thin-shell toggle, emission, color, …) WITHOUT
                                // first assigning it to a scene object.
                                auto drawInlineMatEditor = [&](const char* title, int mat_id) {
                                    if (mat_id < 0) return;
                                    Material* m = MaterialManager::getInstance().getMaterial(static_cast<uint16_t>(mat_id));
                                    PrincipledBSDF* p = dynamic_cast<PrincipledBSDF*>(m);
                                    if (!p) return;
                                    ImGui::PushID(mat_id ^ 0x6F0A);
                                    if (ImGui::CollapsingHeader(title)) {
                                        ImGui::Indent();
                                        ui.drawPrincipledBSDFEditor(p, static_cast<uint16_t>(mat_id), ui_ctx);
                                        ImGui::Unindent();
                                    }
                                    ImGui::PopID();
                                };

                                if (show_material_overrides) {
                                    ImGui::Indent();
                                    if (drawMaterialCombo("Spray Material", fo.spray_material_id, "Default (water droplet)")) {
                                        foam_changed = true;
                                    }
                                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("PBR material for airborne water droplets (Spray).");
                                    drawInlineMatEditor("Edit Spray Material", fo.spray_material_id);

                                    if (drawMaterialCombo("Foam Material", fo.foam_material_id, "Default (white foam)")) {
                                        foam_changed = true;
                                    }
                                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("PBR material for surface foam clusters (Foam).");
                                    drawInlineMatEditor("Edit Foam Material", fo.foam_material_id);

                                    if (drawMaterialCombo("Bubble Material", fo.bubble_material_id, "Default (air bubble)")) {
                                        foam_changed = true;
                                    }
                                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("PBR material for submerged air bubbles (Bubble). Enable 'Bubble (thin shell)' in its editor below for the soap/champagne look.");
                                    drawInlineMatEditor("Edit Bubble Material", fo.bubble_material_id);
                                    ImGui::Unindent();
                                } else {
                                    if (drawMaterialCombo("Foam Material", fo.foam_material_id, "Default (white foam)")) {
                                        foam_changed = true;
                                        fo.spray_material_id = -1;
                                        fo.bubble_material_id = -1;
                                    }
                                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Scene material applied to all foam components. Enable Custom Material Overrides to assign them separately.");
                                }
                            }

                            foam_changed |= ImGui::DragFloat("Foam Sphere Radius (vx)", &fo.render_radius_voxels, 0.01f, 0.05f, 2.0f, "%.2f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Render size of each foam sphere (sim voxels). Small = fine spray/foam.");

                            foam_changed |= ImGui::SliderInt("Foam Subdivisions", &fo.foam_sphere_subdivisions, 0, 3);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Mesh subdivision level for rendering foam spheres. Higher = smoother close-up but slightly slower BVH rebuild.");
                            } // end Spheres-mode controls

                            int maxf = static_cast<int>(std::min<std::size_t>(fo.max_foam, 20000000u));
                            if (ImGui::DragInt("Max Foam Particles", &maxf, 1000.0f, 1000, 20000000)) {
                                fo.max_foam = static_cast<std::size_t>(std::max(1000, maxf));
                            }

                            // Live whitewater stats from the runtime state.
                            const auto& fstates = particles->gridDomainStates();
                            if (selected_domain_index < static_cast<int>(fstates.size())) {
                                const auto& fst = fstates[static_cast<std::size_t>(selected_domain_index)].foam_stats;
                                ImGui::Spacing();
                                ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f),
                                    "Foam: %zu alive  (spray %zu / foam %zu / bubble %zu)",
                                    fst.alive, fst.spray, fst.foam, fst.bubble);
                                ImGui::TextDisabled("  +%zu spawned/step   gen %.2f ms   advect %.2f ms",
                                    fst.spawned, fst.gen_ms, fst.advect_ms);
                            }
                            ImGui::Unindent();
                        }
                        if (foam_changed) ui_ctx.start_render = true;
                    }

                    // NanoVDB shader UI
                    const bool wants_volume_panel =
                        domain.fluid_render_mode != RayTrophiSim::Fluid::FluidRenderMode::Particles;
                    if (wants_volume_panel) {
                        if (!domain.shader) {
                            domain.shader = VolumeShader::createSmokePreset();
                            domain.shader->name = "Liquid NanoVDB Preview";
                            domain.shader->density.multiplier = 1.6f;
                            domain.shader->scattering.color = Vec3(0.62f, 0.78f, 0.92f);
                            domain.shader->scattering.coefficient = 1.1f;
                            domain.shader->absorption.coefficient = 0.04f;
                        }
                        if (ImGui::CollapsingHeader("Volumetric Absorption & Density", ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::Spacing();
                        if (SceneUI::drawVolumeShaderUI(ui_ctx, domain.shader, nullptr, nullptr)) {
                            scene.refreshFluidSurfaceMaterial();
                            ui_ctx.renderer.resetCPUAccumulation();
                            if (ui_ctx.backend_ptr) ui_ctx.backend_ptr->resetAccumulation();
                            ui_ctx.start_render = true;
                        }
                        }
                    }
                }

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        // Detailed live solver step stats drawn beautifully at the bottom
        {
            if (is_fluid_domain && ImGui::CollapsingHeader("Fluid Step Stats##DomainFluidStats", ImGuiTreeNodeFlags_DefaultOpen)) {
                const auto& states = particles->gridDomainStates();
                if (selected_domain_index < static_cast<int>(states.size())) {
                    const auto& st = states[static_cast<std::size_t>(selected_domain_index)];
                    const auto& fs = st.fluid_stats;
                    if (fs.gpu_requested) {
                        const ImVec4 color = fs.gpu_fallback
                            ? ImVec4(1.0f, 0.72f, 0.25f, 1.0f)
                            : ImVec4(0.45f, 0.95f, 0.55f, 1.0f);
                        ImGui::TextColored(color, "Compute: %s", fs.gpu_status.c_str());
                    } else {
                        ImGui::TextDisabled("Compute: CPU reference path");
                    }
                    ImGui::Columns(2, "FluidStatsColumns", false);
                    ImGui::TextDisabled("Particles");        ImGui::NextColumn();
                    ImGui::TextDisabled("%zu", fs.particle_count); ImGui::NextColumn();
                    ImGui::TextDisabled("Active fluid cells"); ImGui::NextColumn();
                    ImGui::TextDisabled("%zu", fs.active_fluid_cells); ImGui::NextColumn();
                    ImGui::TextDisabled("Step total");       ImGui::NextColumn();
                    ImGui::TextDisabled("%.2f ms", fs.total_ms); ImGui::NextColumn();
                    ImGui::TextDisabled("  P2G");            ImGui::NextColumn();
                    ImGui::TextDisabled("%.2f ms (%s)", fs.p2g_ms, fs.p2g_on_gpu ? "GPU" : "CPU"); ImGui::NextColumn();
                    ImGui::TextDisabled("  Pressure");       ImGui::NextColumn();
                    ImGui::TextDisabled("%.2f ms (%s)", fs.pressure_ms, fs.pressure_on_gpu ? "GPU" : "CPU"); ImGui::NextColumn();
                    if (fs.pressure_on_gpu && fs.pressure_cg_max_iterations > 0) {
                        ImGui::TextDisabled("    MGPCG precond"); ImGui::NextColumn();
                        ImGui::TextDisabled("%s", fs.pressure_cg_multigrid ? "Layer B V-cycle" : "Layer A Jacobi"); ImGui::NextColumn();
                        ImGui::TextDisabled("    MGPCG iters"); ImGui::NextColumn();
                        ImGui::TextDisabled("%d / %d", fs.pressure_cg_iterations, fs.pressure_cg_max_iterations); ImGui::NextColumn();
                        ImGui::TextDisabled("    MGPCG residual"); ImGui::NextColumn();
                        ImGui::TextDisabled("%.2e", fs.pressure_cg_final_relative_residual); ImGui::NextColumn();
                        ImGui::TextDisabled("    MGPCG dot sync"); ImGui::NextColumn();
                        ImGui::TextDisabled("%.2f ms (%d)", fs.pressure_cg_dot_ms, fs.pressure_cg_dot_count); ImGui::NextColumn();
                    }
                    ImGui::TextDisabled("  Viscosity");      ImGui::NextColumn();
                    ImGui::TextDisabled("%.2f ms", fs.viscosity_ms); ImGui::NextColumn();
                    ImGui::TextDisabled("  G2P");            ImGui::NextColumn();
                    ImGui::TextDisabled("%.2f ms (%s)", fs.g2p_ms, fs.g2p_on_gpu ? "GPU" : "CPU"); ImGui::NextColumn();
                    ImGui::TextDisabled("  Advect");         ImGui::NextColumn();
                    ImGui::TextDisabled("%.2f ms (%d substeps)", fs.advect_ms, fs.advect_substeps); ImGui::NextColumn();
                    ImGui::TextDisabled("  Density -> NanoVDB"); ImGui::NextColumn();
                    ImGui::TextDisabled("%.2f ms (%s)", fs.density_ms, fs.density_on_gpu ? "GPU" : "CPU"); ImGui::NextColumn();
                    ImGui::Columns(1);
                }
            }
        }

        {
            if (ImGui::CollapsingHeader("VDB Export##DomainVDBExport", ImGuiTreeNodeFlags_DefaultOpen)) {
                static char vdb_dir[2048] = "";
                static char vdb_base[128] = "vdb_export";
                static int vdb_start = 0;
                static int vdb_end = 100;
                static bool vdb_range_initialized = false;
                static std::string vdb_export_message;

                ImGui::Text("Export Directory:");
                float avail_w = ImGui::GetContentRegionAvail().x;
                float btn_w = ImGui::CalcTextSize("Browse").x + ImGui::GetStyle().FramePadding.x * 2.0f;
                float input_w = avail_w - btn_w - ImGui::GetStyle().ItemSpacing.x;
                ImGui::SetNextItemWidth(input_w);
                ImGui::InputText("##FluidDomainVDBDir", vdb_dir, sizeof(vdb_dir));
                ImGui::SameLine();
                if (ImGui::Button("Browse##FluidDomainVDBBrowse", ImVec2(btn_w, 0))) {
                    const std::string selected_dir = SceneUI::selectFolderDialogW(L"Select VDB Export Directory");
                    if (!selected_dir.empty()) {
                        std::snprintf(vdb_dir, sizeof(vdb_dir), "%s", selected_dir.c_str());
                    }
                }
                
                ImGui::SetNextItemWidth(avail_w * 0.6f);
                ImGui::InputText("Base Name##FluidDomainVDBBase", vdb_base, sizeof(vdb_base));

                if (timeline && !vdb_range_initialized) {
                    vdb_start = std::min(timeline->getStartFrame(), timeline->getEndFrame());
                    vdb_end = std::max(timeline->getStartFrame(), timeline->getEndFrame());
                    vdb_range_initialized = true;
                }
                ImGui::DragInt("Start Frame##FluidDomainVDBStart", &vdb_start, 1.0f, 0, 100000);
                ImGui::DragInt("End Frame##FluidDomainVDBEnd", &vdb_end, 1.0f, 0, 100000);
                if (vdb_end < vdb_start) vdb_end = vdb_start;

                const bool can_export = scene.active_particle_system_index >= 0 &&
                                        selected_domain_index >= 0 &&
                                        vdb_dir[0] != '\0' &&
                                        vdb_base[0] != '\0';
                if (!can_export) ImGui::BeginDisabled();
                if (ImGui::Button("Export Current Frame (.vdb)##FluidDomainVDBFrame", ImVec2(-1, 26))) {
                    std::error_code ec;
                    std::filesystem::create_directories(vdb_dir, ec);
                    const int current_frame = timeline ? timeline->getCurrentFrame() : 0;
                    const std::string filename = std::string(vdb_base) + "_" + std::to_string(current_frame) + ".vdb";
                    const std::string path = (std::filesystem::path(vdb_dir) / filename).string();
                    const bool ok = scene.exportDomainVDB(static_cast<std::size_t>(scene.active_particle_system_index),
                                                          static_cast<std::size_t>(selected_domain_index),
                                                          path);
                    vdb_export_message = ok ? ("Saved: " + path) : "Export failed";
                }
                if (ImGui::Button("Export Sequence (.vdb)##FluidDomainVDBSeq", ImVec2(-1, 26))) {
                    std::error_code ec;
                    std::filesystem::create_directories(vdb_dir, ec);
                    const float fps = static_cast<float>(std::max(1, ui_ctx.render_settings.animation_fps));
                    const int written = scene.exportDomainVDBSequence(
                        static_cast<std::size_t>(scene.active_particle_system_index),
                        static_cast<std::size_t>(selected_domain_index),
                        std::string(vdb_dir),
                        std::string(vdb_base),
                        vdb_start,
                        vdb_end,
                        fps);
                    vdb_export_message = "Wrote " + std::to_string(written) + " VDB frame(s)";
                }
                if (!can_export) ImGui::EndDisabled();
                if (!vdb_export_message.empty()) {
                    ImGui::TextDisabled("%s", vdb_export_message.c_str());
                }
            }
        }

        // Legacy standalone-FluidObject VDB cache UI. The grid-domain workflow
        // does NOT use FluidObjects (bake is the SimCache path below), so this is
        // muted: it never auto-creates a "Fluid 1" anymore and only shows if a
        // FluidObject already exists in the scene (old projects). Grid-domain-only
        // users never see it.
        if (false && is_fluid_domain && !scene.fluid_objects.empty()) {
            ImGui::Spacing();
            if (ImGui::CollapsingHeader("Fluid VDB Cache & Threaded Baking##FluidCacheBakeHeader", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (scene.active_fluid_object_index < 0 ||
                    scene.active_fluid_object_index >= static_cast<int>(scene.fluid_objects.size())) {
                    scene.active_fluid_object_index = 0;
                }
                auto* fluid = scene.activeFluidObject();
                if (fluid) {
                    ImGui::Spacing();
                    
                    // ── VDB Cache Mode ──
                    if (ImGui::Checkbox("Use Baked VDB Sequence##FluidCache", &fluid->use_vdb_cache)) {
                        ui_ctx.start_render = true;
                    }
                    if (fluid->use_vdb_cache) {
                        ImGui::Indent();
                        char cache_path_buf[256];
                        strncpy_s(cache_path_buf, fluid->vdb_cache_pattern.c_str(), sizeof(cache_path_buf) - 1);
                        if (ImGui::InputText("Cache Pattern##FluidCachePattern", cache_path_buf, sizeof(cache_path_buf))) {
                            fluid->vdb_cache_pattern = cache_path_buf;
                            ui_ctx.start_render = true;
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Browse##FluidCacheBrowse")) {
                            std::string path = SceneUI::openFileDialogW(L"VDB Files\0*.vdb;*.nvdb\0All Files\0*.*\0", "", "");
                            if (!path.empty()) {
                                std::filesystem::path fpath(path);
                                std::string stem = fpath.stem().string();
                                std::string directory = fpath.parent_path().string();
                                std::string ext = fpath.extension().string();
                                
                                size_t last_digit = std::string::npos;
                                size_t first_digit = std::string::npos;
                                for (size_t i = stem.length(); i > 0; --i) {
                                    if (isdigit(stem[i-1])) {
                                        if (last_digit == std::string::npos) last_digit = i-1;
                                        first_digit = i-1;
                                    } else if (last_digit != std::string::npos) {
                                        break; 
                                    }
                                }
                                if (last_digit != std::string::npos) {
                                    int num_len = (int)(last_digit - first_digit + 1);
                                    fluid->vdb_cache_digits = num_len;
                                    std::string prefix = stem.substr(0, first_digit);
                                    std::string suffix = stem.substr(last_digit + 1);
                                    std::string placeholder(num_len, '#');
                                    fluid->vdb_cache_pattern = (std::filesystem::path(directory) / (prefix + placeholder + suffix + ext)).string();
                                } else {
                                    fluid->vdb_cache_pattern = path;
                                }
                                ui_ctx.start_render = true;
                            }
                        }
                        ImGui::Unindent();
                    }
                    
                    ImGui::Spacing();
                    
                    // ── Export & Baking ──
                    if (UIWidgets::BeginSection("Export & Baking##FluidBake", ImVec4(1.0f, 0.5f, 0.2f, 1.0f), false)) {
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
                }
            }
        }

        ImGui::Separator();
        if (ImGui::Button("Reset Simulation (Free-run)##SimReset", ImVec2(-1, 0))) {
            extern bool g_viewport_raster_rebuild_pending;
            extern bool g_gas_volumes_dirty;
            extern bool g_geometry_dirty;
            extern std::atomic<uint64_t> g_scene_geometry_generation;

            drainSimulationMutationBackends();

            scene.resetSimulation();

            g_gas_volumes_dirty = true;
            g_geometry_dirty = true;
            g_viewport_raster_rebuild_pending = true;
            g_scene_geometry_generation.fetch_add(1, std::memory_order_release);

            ui_ctx.renderer.resetCPUAccumulation();
            if (ui_ctx.backend_ptr) {
                ui_ctx.backend_ptr->resetAccumulation();
            }
            ui_ctx.start_render = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Clear the bake cache and return to live free-run preview.\nPlay the timeline to bake each frame; scrub to replay cached frames.");
        }

        // ── Render-only point cache: persist the bake across reloads ──────────
        // Writes every particle system (fluid particles + foam + gas grids) for
        // the timeline range to "<project>.simcache" next to the project file.
        // Reloading restores the bake without re-simulating (SimCache).
        drawSimBakeControls();

        // ── Auto-reseed (toggle in the Fluid Seeding header) ──────────────────
        // A seed OR grid-shape parameter just settled this frame (released after
        // edit). Re-seed every fluid domain and snap the timeline to frame 0 so
        // the fresh initial state is what's shown — shape edits rebuild the grid,
        // which needs a clean restart. Reseeding ALL fluid domains keeps the
        // frame-0 rewind from leaving other tanks empty; the rewind clears the
        // particles first, so replace_on_seed is irrelevant (nothing to stack on).
        if (seed_settled && s_fluid_auto_reseed && particles &&
            domain.type == RayTrophiSim::SimulationDomainType::Fluid) {
            for (auto& sys : scene.particle_systems) {
                if (!sys.runtime) continue;
                for (auto& d : sys.runtime->gridDomains()) {
                    if (d.type == RayTrophiSim::SimulationDomainType::Fluid)
                        d.fluid_pending_seed = true;
                }
            }
            // Drain any in-flight GPU sim mutations before the reset clears state
            // (mirrors the "Reset Simulation" button's safe ordering).
            drainSimulationMutationBackends();
            // Rewind sim + clear the stale bake; skip capturing the empty pre-seed
            // state — we capture frame 0 AFTER seeding below.
            scene.resetSimulationToStart(/*clear_cache=*/true, /*capture_frame=*/false);
            for (auto& sys : scene.particle_systems) {
                if (sys.runtime) sys.runtime->synchronizeGridDomainsNow();
            }
            scene.captureSimFrame(0);
            if (timeline) timeline->setCurrentFrame(0);
            ui_ctx.start_render = true;
        }

        if (ImGui::Button("Remove Domain##SimulationPanel", ImVec2(-1, 0))) {
            particles->removeGridDomain(static_cast<std::size_t>(selected_domain_index));
            if (ui_ctx.selection.selected.type == SelectableType::SimulationDomain &&
                ui_ctx.selection.selected.particle_system_index == scene.active_particle_system_index &&
                ui_ctx.selection.selected.simulation_domain_index == selected_domain_index) {
                ui_ctx.selection.clearSelection();
            }
            selected_domain_index = std::min(selected_domain_index, static_cast<int>(particles->gridDomains().size()) - 1);
            ImGui::Columns(1);
            return;
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
        drawDomainControls();
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
