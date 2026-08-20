#include "scene_ui_forcefield.hpp"

#include "Api/RtApi.h"
#include "Fluid/FluidSplatMaterialAuthoring.h"

namespace ForceFieldUI {

void drawSimulationDomainControls(
    SceneUI& ui,
    UIContext& ui_ctx,
    SceneData& scene,
    TimelineWidget* timeline,
    int& selected_domain_index,
    const std::function<void()>& drainSimulationMutationBackends,
    const std::function<void()>& clearForceFieldSelection,
    const std::function<void()>& drawSimBakeControls) {
        auto particles = scene.getParticleSimulationSystem();
        const int domain_count = particles ? static_cast<int>(particles->gridDomains().size()) : 0;
        ImGui::Text("Domains: %d", domain_count);

        // ── Reset, AT THE TOP ────────────────────────────────────────────────
        // It used to live at the very bottom, past every domain section, every
        // seeding control and the whole Export & Baking block — so the one
        // action you reach for when a scene looks wrong was the hardest thing
        // in the panel to find, and reaching it meant scrolling past the
        // controls that had just confused you.
        //
        // ★ Editing a solver parameter no longer NEEDS this button: both
        // simulation signatures now hash the fluid domain's solver config, so a
        // physics edit drops the stale bake and rewinds by itself
        // (hashFluidDomainSolverConfig). This stays for the case a signature
        // cannot cover — going back to live free-run preview by hand.
        const auto resetSimulationNow = [&]() {
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
        };
        if (ImGui::Button("Reset Simulation (Free-run)##SimResetTop", ImVec2(-1, 0))) {
            resetSimulationNow();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Clear the bake cache and return to live free-run preview.\n"
                "Play the timeline to bake each frame; scrub to replay cached frames.\n\n"
                "Editing a PHYSICS parameter (viscosity, gravity, boundary, seed\n"
                "density, per-substance physics) already invalidates the bake on its\n"
                "own and rewinds to the start. Look parameters (material, drawn-as,\n"
                "porosity, IOR) repaint the current frame and never re-simulate.");
        }
        ImGui::Separator();

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
            // Use the fastest solver the machine actually has. The descriptor
            // default stays CPU for project-file compatibility; the choice is made
            // here, where the runtime capabilities are known.
            desc.backend = RayTrophiSim::defaultSimulationDomainBackend();
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

                // The CPU Sparse OpenVDB backend was removed from the UI: for
                // fluids it never differed from Dense, and for gas the sparse
                // path fell back to the dense step whenever a collider was
                // present. Legacy projects that saved it are coerced to Dense
                // (the `use_sparse_tiles` checkbox below still controls sparse
                // grid mode independently). The enum value stays for project
                // file compatibility.
                if (domain.backend == RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB) {
                    domain.backend = RayTrophiSim::SimulationDomainBackend::CPU_Dense;
                }
                const char* backends[] = {
                    "CPU (Dense - Standard)",
                    "GPU (CUDA - High Speed)",
                    "GPU (Vulkan Compute)"
                };
                const RayTrophiSim::SimulationDomainBackend backend_values[] = {
                    RayTrophiSim::SimulationDomainBackend::CPU_Dense,
                    RayTrophiSim::SimulationDomainBackend::GPU_Compute,
                    RayTrophiSim::SimulationDomainBackend::GPU_Vulkan
                };
                int current_backend = 0;
                for (int bi = 0; bi < 3; ++bi)
                    if (backend_values[bi] == domain.backend) current_backend = bi;

                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::Combo("##DomainBackend", &current_backend, backends, 3)) {
                    domain.backend = backend_values[current_backend];
                    scene.requestSimulationTimelineRenderResync();
                    g_gas_volumes_dirty = true;
                    ui_ctx.start_render = true;
                    if (ui_ctx.backend_ptr) {
                        ui_ctx.backend_ptr->resetAccumulation();
                    }
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Selects which hardware execution unit running the simulation solver:\n\n"
                                      "1. CPU (Dense): Stable standard processor solver. Ideal for small-scale tests.\n"
                                      "2. GPU (CUDA): NVIDIA-only compute path. Somewhat faster than Vulkan.\n"
                                      "3. GPU (Vulkan Compute): Cross-vendor GPU compute solver (APIC fluid + MGPCG\n"
                                      "   pressure + whitewater). The primary GPU path and the one that runs\n"
                                      "   everywhere; produces the same results as CPU/CUDA.\n\n"
                                      "New domains default to Vulkan when this machine supports it, CUDA when it\n"
                                      "does not, and CPU when neither is available.");
                }

                ImGui::Spacing();
                if (domain.backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) {
                    const bool actual_vulkan =
                        scene.simulation_world.compute().backendType() ==
                        RayTrophiSim::ComputeBackendType::VulkanCompute;
                    if (g_hasVulkanComputeSim && actual_vulkan)
                        ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f),
                                           "  [GPU Status: Vulkan Compute Active]");
                    else if (g_hasVulkanComputeSim)
                        ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.25f, 1.0f),
                                           "  [GPU Status: Vulkan requested; runtime backend not active]");
                    else
                        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "  [GPU Status: Vulkan Compute - SPIR-V shaders missing, CPU fallback]");
                } else if (g_hasCUDA) {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, 1.0f), "  [GPU Status: CUDA Acceleration Active]");
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "  [GPU Status: CUDA Capable GPU Not Found - CPU Fallback]");
                }
                }

                const char* quality_profiles[] = {
                    "Interactive (Fast)",
                    "Preview (Balanced)",
                    "Final (Production)",
                    "Cinema (Offline)",
                    "Custom"
                };
                int quality_profile = static_cast<int>(domain.quality_profile);
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::Combo("Quality Profile##DomainQuality", &quality_profile,
                                 quality_profiles, IM_ARRAYSIZE(quality_profiles))) {
                    domain.quality_profile =
                        static_cast<RayTrophiSim::SimulationDomainQualityProfile>(quality_profile);
                    domain.backend = RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
                    switch (domain.quality_profile) {
                        case RayTrophiSim::SimulationDomainQualityProfile::Interactive:
                            domain.max_auto_resolution = 96;
                            domain.resource_budget_mb = 512;
                            domain.force_disk_cache = false;
                            domain.turbulence_octaves = std::min(domain.turbulence_octaves, 2);
                            break;
                        case RayTrophiSim::SimulationDomainQualityProfile::Preview:
                            domain.max_auto_resolution = 192;
                            domain.resource_budget_mb = 1024;
                            domain.force_disk_cache = false;
                            domain.turbulence_octaves = std::clamp(domain.turbulence_octaves, 2, 4);
                            break;
                        case RayTrophiSim::SimulationDomainQualityProfile::Final: {
                            domain.max_auto_resolution = 512;
                            // Dynamic budget: ~25% of system RAM, min 4 GB, max 16 GB.
                            // Modern machines carry 32-128 GB; a fixed 4 GB wastes capacity.
                            const double total_ram_gb_f = ForceFieldUI::queryTotalPhysicalRamBytes() / (1024.0 * 1024.0 * 1024.0);
                            uint32_t dyn_budget_mb = 4096; // fallback
                            if (total_ram_gb_f >= 64.0)      dyn_budget_mb = 16384;
                            else if (total_ram_gb_f >= 32.0) dyn_budget_mb = 8192;
                            else if (total_ram_gb_f >= 16.0) dyn_budget_mb = 6144;
                            domain.resource_budget_mb = dyn_budget_mb;
                            domain.enforce_resource_budget = true;
                            domain.force_disk_cache = false;
                            domain.turbulence_octaves = std::max(domain.turbulence_octaves, 4);
                            break;
                        }
                        case RayTrophiSim::SimulationDomainQualityProfile::Cinema:
                            domain.max_auto_resolution = 1024;
                            domain.enforce_resource_budget = false; // RAM limit lifted
                            domain.force_disk_cache = true;         // disk bake mandatory
                            domain.use_sparse_tiles = true;         // required at cinema res
                            domain.turbulence_octaves = std::max(domain.turbulence_octaves, 6);
                            break;
                        case RayTrophiSim::SimulationDomainQualityProfile::Custom:
                            break;
                    }
                    // Ensure enforce_resource_budget for non-Cinema/Custom profiles
                    if (domain.quality_profile != RayTrophiSim::SimulationDomainQualityProfile::Cinema &&
                        domain.quality_profile != RayTrophiSim::SimulationDomainQualityProfile::Custom) {
                        domain.enforce_resource_budget = true;
                    }
                    if (domain.shader) {
                        const bool cinema_quality =
                            domain.quality_profile == RayTrophiSim::SimulationDomainQualityProfile::Cinema;
                        const bool final_quality =
                            domain.quality_profile == RayTrophiSim::SimulationDomainQualityProfile::Final;
                        const bool interactive_quality =
                            domain.quality_profile == RayTrophiSim::SimulationDomainQualityProfile::Interactive;
                        domain.shader->quality.max_steps = cinema_quality ? 1024 : (final_quality ? 512 : (interactive_quality ? 128 : 256));
                        domain.shader->quality.shadow_steps = cinema_quality ? 32 : (final_quality ? 16 : (interactive_quality ? 6 : 10));
                        domain.shader->quality.shadow_stride = cinema_quality ? 1 : (final_quality ? 1 : (interactive_quality ? 4 : 2));
                        domain.shader->quality.voxel_step_multiplier =
                            cinema_quality ? 0.25f : (final_quality ? 0.5f : (interactive_quality ? 1.5f : 0.85f));
                    }
                    scene.requestSimulationTimelineRenderResync();
                    ui_ctx.start_render = true;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Sets coherent simulation, memory and ray-march defaults.\n"
                        "Vulkan Compute is selected for every managed profile.\n"
                        "Manual edits remain available and switch the profile to Custom.\n\n"
                        "Cinema: removes RAM limit, forces disk bake, enables up to 1024 grid.");
                }

                // Group 2: Grid Resolution & Scaling
                if (ImGui::CollapsingHeader("Grid Resolution & Scaling", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Spacing();

                int res[3] = { domain.resolution_x, domain.resolution_y, domain.resolution_z };
                if (ImGui::DragInt3("Grid Resolution (X, Y, Z)", res, 1.0f, 8, 2048)) {
                    domain.resolution_x = std::clamp(res[0], 8, 2048);
                    domain.resolution_y = std::clamp(res[1], 8, 2048);
                    domain.resolution_z = std::clamp(res[2], 8, 2048);
                    const Vec3 ext = Vec3::max(domain.bounds_min, domain.bounds_max) - Vec3::min(domain.bounds_min, domain.bounds_max);
                    const float me = std::max({ ext.x, ext.y, ext.z, 0.001f });
                    const int mr = std::max({ domain.resolution_x, domain.resolution_y, domain.resolution_z, 1 });
                    domain.voxel_size = me / static_cast<float>(mr);
                }
                seed_settled |= ImGui::IsItemDeactivatedAfterEdit();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Per-axis resolution of the 3D voxel simulation grid (8-2048 each).\n\n"
                                      "WARNING: Cost scales with the product X*Y*Z (cubic for a cube)!\n"
                                      "Use 32-64 for fast interactive testing, 96-128+ for high-quality results.\n"
                                      "512+ requires sparse tiles and Cinema profile or disk bake for memory safety.\n"
                                      "Prefer a non-cube box (e.g. 512x64x128) over a full 512^3 cube.");
                }
                // Live cell-count + rough grid memory so high resolutions are an
                // informed choice. The solver re-derives resolution each rebuild and
                // clamps every axis to Max Auto Resolution, so preview the clamped
                // values - otherwise the estimate lies whenever a requested axis
                // exceeds the ceiling. ~11 floats/cell estimates MAC velocity faces +
                // pressure/divergence/density/mask + PCG scratch vectors.
                {
                    const int eff_cap = std::clamp(domain.max_auto_resolution, 32, 2048);
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
                if (ImGui::DragInt("Max Auto Resolution", &domain.max_auto_resolution, 1.0f, 32, 2048)) {
                    domain.quality_profile = RayTrophiSim::SimulationDomainQualityProfile::Custom;
                    domain.max_auto_resolution = std::clamp(domain.max_auto_resolution, 32, 2048);
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
                domain.max_auto_resolution = std::clamp(domain.max_auto_resolution, 32, 2048);
                // Force sparse tiles when any axis exceeds 512 — dense allocation
                // at these resolutions risks OOM without sparse tile culling.
                if (domain.max_auto_resolution > 512 || domain.resolution_x > 512 ||
                    domain.resolution_y > 512 || domain.resolution_z > 512) {
                    domain.use_sparse_tiles = true;
                }

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

                    if (ImGui::CollapsingHeader("Buoyancy & Gas Motion", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Spacing();
                        ImGui::DragFloat("Heat Lift", &domain.gas_buoyancy_heat,
                                         0.02f, -20.0f, 20.0f, "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Continuous upward/downward acceleration from temperature.\n"
                                              "This is domain-local and remains active in hybrid Spark + Gas effects.");
                        }
                        ImGui::DragFloat("Smoke Lift", &domain.gas_buoyancy_density,
                                         0.01f, -20.0f, 20.0f, "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Density-driven lift. Keep modest for stable smoke columns.");
                        }
                        ImGui::DragFloat("Solved Vorticity", &domain.gas_vorticity,
                                         0.01f, 0.0f, 50.0f, "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Grid-solver vorticity confinement. Separate from procedural turbulence.");
                        }
                        ImGui::Checkbox("MacCormack Advection", &domain.gas_maccormack_advection);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Limited second-order transport. Preserves wisps, sharp flame fronts\n"
                                              "and vortices that plain semi-Lagrangian smears away.\n"
                                              "Costs one extra advection pass per field.");
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
                        ImGui::Separator();
                        ImGui::Checkbox("Blast Damages Structures##StructCouple",
                                        &domain.structural_coupling_enabled);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Turn this fire's combustion into blast loading on breakable\n"
                                              "objects. Without it, burning still weakens material and lowers\n"
                                              "its fracture threshold, but nothing ever delivers the push -\n"
                                              "so a fire can char a structure and never bring it down.");
                        }
                        if (domain.structural_coupling_enabled) {
                            ImGui::DragFloat("Blast Pressure Scale", &domain.structural_pressure_scale,
                                             5.0f, 0.0f, 5000.0f, "%.0f kPa");
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("CALIBRATION, not a physical constant.\n"
                                                  "Fuel and temperature here are normalized units, so no honest\n"
                                                  "formula turns them into kilopascals. Raise this until a fire\n"
                                                  "of the size you author breaks what it should. Compare against\n"
                                                  "the Break Threshold you set on the fracture group.");
                            }
                            ImGui::DragFloat("Minimum Blast Intensity", &domain.structural_min_intensity,
                                             0.01f, 0.0f, 5.0f, "%.3f");
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("Below this mean burn rate the fire loads nothing.\n"
                                                  "Keeps a steady small flame from emitting an endless\n"
                                                  "drizzle of weak blast events.");
                            }
                            ImGui::DragFloat("Blast Interval", &domain.structural_event_interval,
                                             0.01f, 1.0f / 120.0f, 5.0f, "%.2f s");
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("Seconds between blast events from this domain, and the\n"
                                                  "duration each one claims. A sustained fire therefore\n"
                                                  "delivers repeated honest blows rather than one load\n"
                                                  "counted again every frame.");
                            }
                        }

                        ImGui::Spacing();
                        ImGui::TextDisabled("Physics Note: Remember to add a Flow Source emitting Fuel and Temperature.\n"
                                             "Set shader mode to 'Blackbody' in the Shading tab for realistic fire rendering.");
                    } else {
                        ImGui::TextDisabled("Combustion is disabled. Simulating smoke (Density) only.");
                    }
                    }

                    // ── Thermal boundary override ────────────────────────────
                    // The world defines ambient everywhere; a domain may override
                    // it inside its own bounds. Off by default, so a domain that
                    // says nothing simply inherits the world.
                    if (ImGui::CollapsingHeader("Thermal Override (Material State Field)")) {
                        ImGui::Spacing();
                        ImGui::Checkbox("Override World Ambient Inside This Domain##DomThermal",
                                        &domain.thermal_override_enabled);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Boundary conditions for object surface heating inside this\n"
                                "domain's box. Affects the Material State Field (heating,\n"
                                "ignition, char, glow) — NOT the gas solve itself.\n\n"
                                "Off: this domain inherits the world's ambient and oxygen.");
                        }
                        ImGui::BeginDisabled(!domain.thermal_override_enabled);
                        ImGui::DragFloat("Ambient (K)##DomThermalK",
                                         &domain.thermal_ambient_kelvin, 1.0f, 0.0f, 3000.0f, "%.0f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Ambient temperature a surface relaxes toward while it is\n"
                                "inside this box. Tested per surface ELEMENT, so an object\n"
                                "half in and half out is genuinely half-heated.");
                        }
                        ImGui::DragFloat("Oxygen##DomThermalO2",
                                         &domain.thermal_oxygen, 0.01f, 0.0f, 1.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "0..1. Throttles pyrolysis burn rate inside this domain.\n"
                                "0 smothers fire entirely. It can only slow burning down,\n"
                                "never start it.");
                        }
                        ImGui::EndDisabled();
                        // Said explicitly because its absence is a design decision,
                        // not an oversight, and someone WILL look for it here.
                        ImGui::TextDisabled("Kelvin-per-unit is global on purpose — the burn\n"
                                            "mask quantizes glow in absolute Kelvin, so a\n"
                                            "per-domain mapping would make the same object\n"
                                            "glow differently in different boxes.");
                    }

                    // Procedural turbulence (divergence-free curl-noise detail).
                    if (ImGui::CollapsingHeader("Turbulence (Procedural Detail)")) {
                        ImGui::Spacing();
                        ImGui::DragFloat("Turbulence Strength", &domain.turbulence_strength, 0.01f, 0.0f, 50.0f, "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Adds divergence-free swirling detail on top of the solved motion.\n"
                                              "0 = off. Modulated by local density/heat/edges so still air stays calm.");
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

                    ImGui::Checkbox("Recreate Seed on Reset##PersistentSeed",
                                    &domain.fluid_reseed_on_reset);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Stores this Seed Box as an initial-state recipe. When enabled, timeline\n"
                                          "rewind/reset recreates it. Disable for emitter-only or one-shot seeds.\n"
                                          "This is independent of clearing existing live particles.");
                    }

                    if (ImGui::Button("Seed Fluid Now##SeedButton", ImVec2(-1, 30))) {
                        if (rtapi::seedFluidParticles(
                                domain.name,
                                &domain.fluid_seed_min,
                                &domain.fluid_seed_max,
                                domain.fluid_seed_particles_per_cell,
                                domain.fluid_replace_on_seed,
                                domain.fluid_reseed_on_reset).ok) {
                            ui_ctx.start_render = true;
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Places particles once in the authoritative grid-domain state.\n"
                                          "Reset replay is controlled separately above.");
                    }

                    const float clear_width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
                    if (ImGui::Button("Clear Live##ClearFluidLive", ImVec2(clear_width, 0))) {
                        if (rtapi::clearFluidParticles(domain.name, false).ok) {
                            ui_ctx.start_render = true;
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Removes current particles but keeps the reset-time seed recipe armed.");
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Clear + Disarm Seed##ClearFluidRecipe", ImVec2(-1, 0))) {
                        if (rtapi::clearFluidParticles(domain.name, true).ok) {
                            ui_ctx.start_render = true;
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Removes current particles and disables Seed Box/Fill Level recreation.\n"
                                          "Emitters can keep adding particles without the original seed returning.");
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

                    // ★★ NAMED AS A GROUP so the four knobs below stop competing
                    // with the real one. Every name here contains "viscous",
                    // "friction" or "damping", and all four drag the WHOLE BODY —
                    // they slow a falling blob down instead of making it resist
                    // shear. Turning them up to fake thickness is what made honey
                    // reach a terminal fall speed and sand fall slower than honey.
                    ImGui::SeparatorText("Dissipation (slows motion - not thickness)");
                    fp_edited |= ImGui::DragFloat("Velocity Damping", &fp.velocity_damping, 0.001f, 0.5f, 1.0f, "%.3f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Velocity damping factor applied per step. 1.0 = frictionless flow, <1.0 = viscous slowdown.");
                    }
                    fp_edited |= ImGui::DragFloat("Internal Viscous Friction", &fp.internal_friction, 0.01f, 0.0f, 10.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Exponential decay on EVERY particle, in whatever direction it\n"
                                          "is moving: v *= exp(-rate * dt).\n\n"
                                          "This is NOT viscosity. It brakes the whole body, so it also\n"
                                          "brakes free FALL - a tall pour arrives slow and lands without\n"
                                          "a splash. Use Kinematic Viscosity below for thickness.\n\n"
                                          "0 = water and any real liquid. Non-zero is a stylised or\n"
                                          "deliberately dead liquid. 10+ = near-instant stop.");
                    }
                    fp_edited |= ImGui::DragFloat("Air Drag Resistance", &fp.air_drag, 0.01f, 0.0f, 10.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Quadratic drag on DETACHED droplets only (isolated spray), as\n"
                                          "v *= 1 / (1 + k|v|dt). Bulk liquid is untouched.\n\n"
                                          "k is in 1/m and follows from droplet size:\n"
                                          "  k = 3*rho_air*Cd / (4*rho_water*d)\n"
                                          "  ~0.15 for 3 mm drops | ~0.5 for sub-mm mist\n\n"
                                          "Raising it is the fastest way to kill a splash - the spray is\n"
                                          "exactly what it acts on.");
                    }
                    fp_edited |= ImGui::DragFloat("Wall Friction Damping", &fp.wall_damping, 0.01f,  0.0f, 1.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Friction factor applied when liquid rubs boundaries or solid colliders.\n"
                                          "0 = Slippery walls (sliding), 1 = Sticky walls (no-slip).");
                    }

                    ImGui::SeparatorText("Coupling");
                    ImGui::SliderFloat("Domain Motion Coupling", &fp.domain_motion_coupling, 0.0f, 1.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Couples domain coordinate translation to fluid velocity. Allows creating sloshing liquids inside a moving cup.");
                    }

                    // ★★★ THE ONE ROW THAT SETS THICKNESS, and it was the ninth
                    // row down in a block where four of the eight above it are
                    // also called viscous / friction / damping. "Internal Viscous
                    // Friction", "Velocity Damping", "Air Drag" and "Wall Friction
                    // Damping" all read like viscosity and none of them are one:
                    // they drag the whole body instead of resisting SHEAR. A user
                    // hunting for viscosity finds four plausible knobs before this
                    // one and reasonably concludes the real control is gone.
                    ImGui::SeparatorText("Rheology (how thick it is)");

                    // Logarithmic: the useful range spans six decades (water 1e-6
                    // to lava 1e2), so a linear drag bar would put every liquid
                    // anyone actually pours inside its first pixel.
                    fp_edited |= ImGui::DragFloat("Kinematic Viscosity (m^2/s)", &fp.kinematic_viscosity,
                                                  0.0001f, 0.0f, 100.0f, "%.6f",
                                                  ImGuiSliderFlags_Logarithmic);
                    // ★★★ AND IT DOES NOT APPLY TO EVERY PARTICLE. A substance
                    // binding that is not inheriting carries its OWN ν, captured
                    // when its "Inherit Domain Viscosity" box was unticked, and
                    // from that moment this row — and every material preset
                    // written through it — is a no-op for that liquid. The domain
                    // knob still moved, still read back, still changed the preset
                    // name, and nothing on screen got thicker. That reads exactly
                    // as "all the thick presets use one fixed high viscosity".
                    {
                        std::string pinned;
                        for (const auto& b : domain.fluid_substance_materials) {
                            if (b.kinematic_viscosity < 0.0f) continue;   // inheriting
                            if (!pinned.empty()) pinned += ", ";
                            pinned += b.substance.empty() ? std::string("(unnamed)") : b.substance;
                        }
                        if (!pinned.empty()) {
                            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.72f, 0.25f, 1.0f));
                            ImGui::TextWrapped(
                                "Not applied to: %s - these substances pin their own "
                                "viscosity. Tick \"Inherit Domain Viscosity\" on them "
                                "in Substance Overrides to let this row (and material "
                                "presets) reach them.", pinned.c_str());
                            ImGui::PopStyleColor();
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Physical kinematic viscosity in m^2/s, solved implicitly\n"
                                          "as nu*dt/h^2 - so the same value behaves the same at any\n"
                                          "voxel size.\n"
                                          "  water 1e-6 | olive oil 8e-5 | chocolate 4e-3\n"
                                          "  honey 7e-3 | molten plastic 0.3 | lava 0.5+\n"
                                          "0 skips the solve entirely.");
                    }
                    ImGui::SetNextItemWidth(120.0f);
                    fp_edited |= ImGui::DragInt("Viscosity Sweeps", &fp.viscosity_sweeps, 1.0f, 1, 64);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Red-black Gauss-Seidel sweeps for the implicit solve.\n"
                                          "Too few never explodes - it UNDER-applies the viscosity,\n"
                                          "so raise this if a thick liquid still flows too freely\n"
                                          "(especially after raising the resolution).");
                    }
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(140.0f);
                    fp_edited |= ImGui::SliderFloat("Wall Slip", &fp.viscosity_wall_slip, 0.0f, 1.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Tangential condition at colliders for the viscous solve.\n"
                                          "0 = no-slip: the liquid sticks to surfaces and is dragged\n"
                                          "    by moving ones. Honey, chocolate, mud, lava.\n"
                                          "1 = free-slip: slides freely. Water.");
                    }

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
                        ImGui::SetTooltip("Courant factor for the ADVECTION substeps below - how far a\n"
                                          "particle is allowed to travel per substep, in cells.\n\n"
                                          "It does not shorten the solver step: the transfer and the\n"
                                          "pressure solve still run once per frame.");
                    }
                    // ★★★ RENAMED, because the old name promised the whole solver
                    // and delivered advection only. P2G, the boundaries, the
                    // viscous solve, the pressure projection and G2P all run
                    // EXACTLY ONCE PER TIMELINE FRAME, at dt = 1/fps
                    // (scene_data.h: fixed_dt = 1/fps). Only particle positions
                    // are integrated in substeps.
                    //
                    // ★★ That single fact is the fixed, parameter-proof "high
                    // viscosity" a fast liquid runs into. At 24 fps a 6 m/s pour
                    // crosses 5-12 cells between two pressure solves; momentum is
                    // smeared over all of them and no crown, sheet or separate
                    // droplet can form. It is NUMERICAL viscosity — set by dt/h²,
                    // not by kinematic_viscosity, internal_friction or air_drag —
                    // which is why zeroing all three changes nothing, and why the
                    // thick presets look right (they are slow, so their CFL number
                    // is small AND their thickness is real).
                    //
                    // ★ A user reading "Max Solver Substeps" reasonably concludes
                    // the solver already sub-steps itself and looks elsewhere. The
                    // name was doing the hiding.
                    ImGui::DragInt("Max Advection Substeps", &fp.max_substeps,  1.0f,   1, 64);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Substeps for PARTICLE ADVECTION only.\n\n"
                                          "The transfer (P2G/G2P) and the pressure solve run ONCE per\n"
                                          "frame, at dt = 1/fps. Raising this does not shorten that\n"
                                          "step - it only stops fast particles from tunnelling.\n\n"
                                          "For a fast liquid (water falling more than a metre) the frame\n"
                                          "step itself is the limit: at 24 fps the liquid crosses several\n"
                                          "cells between two pressure solves, which smears momentum and\n"
                                          "flattens splashes no matter what the viscosity is set to.\n"
                                          "Raise the timeline FPS to shorten it.");
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

                    if (ImGui::CollapsingHeader("Granular Material", ImGuiTreeNodeFlags_DefaultOpen)) {
                        bool granular_edited = false;
                        granular_edited |= ImGui::Checkbox("Enable Granular MPM", &fp.granular_enabled);
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Switches this domain from incompressible liquid physics to compressible\n"
                                              "Drucker-Prager granular MPM on Vulkan. Liquid pressure projection,\n"
                                              "viscosity and particle reseeding are disabled. Use for sand, gravel,\n"
                                              "soil, powder, snow and other frictional bulk materials.");
                        ImGui::BeginDisabled(!fp.granular_enabled);
                        granular_edited |= ImGui::SliderFloat("Friction Angle (deg)", &fp.granular_friction_angle_degrees, 0.0f, 55.0f, "%.1f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Internal grain friction controlling the shear-yield surface and repose angle.\n"
                                              "Low values spread easily; high values form steeper, more stable piles.\n"
                                              "Typical guides: powder 15-25, dry sand 30-38, angular gravel 38-48 deg.");
                        granular_edited |= ImGui::DragFloat("Cohesion (Pa)", &fp.granular_cohesion, 1.0f, 0.0f, 100000.0f, "%.1f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Shear strength that remains even with no confining pressure.\n"
                                              "0 Pa gives dry, non-sticky grains. Raise it for damp sand, soil, clay\n"
                                              "or compacted snow. Excessive cohesion makes one rubber-like lump.");
                        granular_edited |= ImGui::SliderFloat("Dilatancy (deg)", &fp.granular_dilatancy_degrees, 0.0f, 30.0f, "%.1f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Volume expansion produced by plastic shear as grains climb over neighbours.\n"
                                              "0 keeps volume during shear; higher values make dense material swell\n"
                                              "and loosen while flowing. Usually below the friction angle; sand 0-12 deg.");
                        granular_edited |= ImGui::DragFloat("Young Modulus (Pa)", &fp.granular_young_modulus, 100.0f, 10.0f, 10000000.0f, "%.0f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Elastic stiffness before plastic yield. Higher values reduce the soft/rubber\n"
                                              "compression seen on impact but require more granular solver substeps.\n"
                                              "The runtime may cap the effective value for elastic CFL stability.");
                        granular_edited |= ImGui::DragInt("Max Granular Solver Substeps",
                                                          &fp.granular_max_solver_substeps,
                                                          1.0f, 1, 64);
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Maximum full P2G-grid-G2P elastic substeps per frame.\n"
                                              "Raise until Effective Young matches Requested Young. This is a\n"
                                              "real solver-quality cost: 15 substeps can cost about 15x one step.\n"
                                              "The density/render bridge still runs once after the final substep.");
                        granular_edited |= ImGui::SliderFloat("Poisson Ratio", &fp.granular_poisson_ratio, 0.0f, 0.49f, "%.3f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Couples axial compression to sideways expansion in the elastic response.\n"
                                              "0 is independently compressible; values near 0.5 resist volume change.\n"
                                              "Loose grains are commonly 0.15-0.30. Avoid 0.49 at coarse timesteps.");
                        granular_edited |= ImGui::DragFloat("Tensile Cutoff (Pa)", &fp.granular_tensile_cutoff, 1.0f, 0.0f, 100000.0f, "%.1f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Maximum tensile stress before material points detach.\n"
                                              "0 means dry grains cannot carry tension and separate immediately.\n"
                                              "Raise it for wet sand, clay, packed snow or weak bonded aggregates.");
                        granular_edited |= ImGui::DragFloat("Hardening", &fp.granular_hardening, 0.01f, 0.0f, 100.0f, "%.2f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Changes resistance after plastic deformation. 0 keeps constant strength;\n"
                                              "higher values make compressed/sheared material progressively harder.\n"
                                              "Useful for compacting soil and snow; keep near 0 for dry sand.");
                        ImGui::SeparatorText("Damage & Rebonding");
                        granular_edited |= ImGui::DragFloat("Fracture Strain", &fp.granular_fracture_strain, 0.001f, 0.001f, 1.0f, "%.3f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Maximum irreversible Rankine bond-opening strain where damage begins.\n"
                                              "It is not summed once per solver substep. Frictional compression/shear\n"
                                              "may still flow and harden without\n"
                                              "spending this fracture budget. Low values give brittle snowballs\n"
                                              "or soil clods; it is inactive when cohesion and tension are zero.");
                        granular_edited |= ImGui::DragFloat("Damage Rate", &fp.granular_damage_rate, 0.05f, 0.0f, 100.0f, "%.2f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Post-threshold softening slope per unit strain. Damage follows\n"
                                              "1-exp(-rate * excess strain), so it grows progressively instead\n"
                                              "of deleting every bond on the first yielded frame. 0 disables it.");
                        granular_edited |= ImGui::Checkbox("Allow Rebonding", &fp.granular_rebonding);
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Lets damaged grains rebuild bonds while compressed and below yield.\n"
                                              "Off for dry sand/gravel; on for wet sand, clay and compacting snow.");
                        ImGui::BeginDisabled(!fp.granular_rebonding);
                        granular_edited |= ImGui::DragFloat("Healing Rate", &fp.granular_healing_rate, 0.01f, 0.0f, 20.0f, "%.2f");
                        ImGui::SeparatorText("Thermal / Burn Softening");
                        granular_edited |= ImGui::DragFloat("Softening Temperature (K)",
                                                           &fp.granular_softening_temperature,
                                                           1.0f, 0.0f, 4000.0f, "%.0f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Temperature at which the granular skeleton has lost half"
                                              "its strength. 0 DISABLES softening entirely (sand does not"
                                              "melt). Bond strength falls faster than stiffness, so the body"
                                              "stops holding its shape before it goes soft."
                                              "Remaining mass_fraction multiplies this, so a charring body"
                                              "weakens as it burns off without a second dial.");
                        granular_edited |= ImGui::DragFloat("Softening Range (K)",
                                                           &fp.granular_softening_range,
                                                           1.0f, 1.0f, 2000.0f, "%.0f");
                        granular_edited |= ImGui::SliderFloat("Residual Strength",
                                                             &fp.granular_residual_strength,
                                                             0.0f, 1.0f, "%.3f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Fraction of strength kept once fully softened."
                                              "0 = a true melt; a small value leaves a molten residue.");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Fractional bond-damage recovery per second under compression.\n"
                                              "Higher values let compressed fragments clump and rebuild bonds faster.\n"
                                              "Airborne or freely separated fragments do not heal.");
                        ImGui::EndDisabled();
                        ImGui::EndDisabled();
                        if (granular_edited) {
                            fp.sanitizeGranularMaterial();
                            fp.current_preset = RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Custom;
                        }
                        // These parameters are in the fluid coupling signature,
                        // so committing an edit drops the bake and snaps the
                        // playhead to frame 0 by itself. Telling the user to
                        // Reset + Seed by hand described the old behaviour and
                        // would now just be a second, redundant round trip.
                        ImGui::TextDisabled(
                            "Material edits rewind to frame 0 and drop the bake automatically.");
                    }

                    if (ImGui::CollapsingHeader(
                            "Combustible Liquid / Gas Coupling",
                            ImGuiTreeNodeFlags_DefaultOpen)) {
                        using ChemistryPreset = RayTrophiSim::Fluid::FluidChemistryPreset;
                        static const char* chemistry_labels[] = {
                            "Inert", "Water", "Gasoline", "Alcohol", "Oil", "Custom", "Plastic", "Wax"
                        };
                        int chemistry_index = static_cast<int>(
                            domain.fluid_params.chemistry_preset);
                        chemistry_index = std::clamp(chemistry_index, 0, 7);
                        ImGui::SetNextItemWidth(180.0f);
                        if (ImGui::Combo("Chemistry Preset##FluidChemistry",
                                         &chemistry_index, chemistry_labels, 8)) {
                            const auto chosen = static_cast<ChemistryPreset>(chemistry_index);
                            domain.fluid_params.applyChemistryProfile(chosen);
                            const auto& chemistry = domain.fluid_params.fuel_profile;
                            domain.fluid_flammable = chemistry.flammable;
                            domain.fluid_extinguishing = chemistry.extinguishing;
                            domain.fluid_ignition_temperature = chemistry.flash_temperature;
                            domain.fluid_evaporation_rate = chemistry.vaporization_rate;
                            domain.fluid_cooling_power = chemistry.cooling_power;
                            domain.fluid_oxygen_dilution = chemistry.oxygen_dilution;
                            if (chemistry.extinguishing) {
                                domain.fluid_surface_cooling = std::max(
                                    domain.fluid_surface_cooling,
                                    chemistry.cooling_power);
                            }
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Chemical behavior is independent from the physical fluid preset.\n"
                                "Use Oil physics + Gasoline chemistry for a fast fuel jet,\n"
                                "or Water chemistry to cool and extinguish overlapping gas fire.");
                        }
                        ImGui::Checkbox(
                            "Enable Flammable Surface##FluidFire",
                            &domain.fluid_flammable);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                            "Exposes the liquid free surface to overlapping Vulkan Gas domains.\n"
                            "The liquid bulk remains incompressible; only the surface exchanges heat and vapor.");
                        if (domain.fluid_extinguishing) {
                            ImGui::TextDisabled("Extinguishing liquid active");
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Publishes only the exposed APIC free-surface band "
                                "into overlapping Vulkan Gas domains. The liquid "
                                "bulk remains incompressible and does not emit.");
                        }
                        if (domain.fluid_flammable) {
                            ImGui::Checkbox(
                                "Auto Ignite##FluidFire",
                                &domain.fluid_auto_ignite);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                                "Ignites the authored liquid vapor immediately when it reaches the\n"
                                "surface threshold. Disable this to require a pilot/flame contact.");
                            ImGui::DragFloat(
                                "Ignition Temperature##FluidFire",
                                &domain.fluid_ignition_temperature,
                                0.01f,0.0f,100.0f,"%.3f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                                "Normalized flash/ignition threshold of the liquid surface.\n"
                                "Lower values make gasoline/alcohol vapor ignite more readily.");
                            ImGui::DragFloat(
                                "Evaporation / Burn Rate##FluidFire",
                                &domain.fluid_evaporation_rate,
                                0.01f,0.0f,100.0f,"%.3f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                                "Rate at which exposed liquid becomes fuel vapor.\n"
                                "Higher values make gasoline/alcohol spread and ignite faster.");
                            ImGui::DragFloat(
                                "Surface Fuel Capacity##FluidFire",
                                &domain.fluid_surface_fuel_capacity,
                                0.05f,0.0f,1000.0f,"%.3f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                                "Maximum combustible material stored by an exposed surface cell.\n"
                                "This controls available fuel duration, not liquid viscosity.");
                            ImGui::DragFloat(
                                "Heat Release##FluidFire",
                                &domain.fluid_combustion_heat_release,
                                0.05f,0.0f,100.0f,"%.3f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                                "Heat added to the gas per unit of vapor consumed by combustion.");
                            ImGui::DragFloat(
                                "Smoke Yield##FluidFire",
                                &domain.fluid_combustion_smoke_yield,
                                0.01f,0.0f,100.0f,"%.3f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                                "Smoke/density generated by burning liquid vapor.");
                            ImGui::DragFloat(
                                "Surface Cooling##FluidFire",
                                &domain.fluid_surface_cooling,
                                0.01f,0.0f,100.0f,"%.3f");
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                                "Relaxes the liquid surface temperature toward ambient between hot contacts.");
                            ImGui::TextDisabled(
                                "Vulkan APIC mask -> Gas fuel/heat/smoke; "
                                "Gas temperature feeds ignition back.");
                        }
                    }

                    // Redistribution / Reseed settings
                    if (ImGui::CollapsingHeader("Dynamic Particle Reseeding (Reseed)", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Spacing();

                    if (fp.granular_enabled) ImGui::BeginDisabled();
                    ImGui::Checkbox("Enable Dynamic Reseeding##Reseed", &fp.reseed_enabled);
                    if (fp.granular_enabled) ImGui::EndDisabled();
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Redistributes sampling density without creating liquid mass.\n"
                                          "Only particles removed from crowded cells may be replaced in starved interior cells.\n"
                                          "Emitters and open boundaries remain independent count-changing paths.");
                    }

                    if (fp.granular_enabled) {
                        ImGui::TextDisabled("Disabled for granular MPM: reseeding would destroy plastic history and mass.");
                    } else if (fp.reseed_enabled) {
                        ImGui::DragInt("Target Particles Per Cell", &fp.reseed_target_per_cell, 0.1f, 0, 64);
                        ImGui::DragInt("Minimum Threshold Per Cell", &fp.reseed_min_per_cell, 0.1f, 1, 32);
                        ImGui::DragInt("Maximum Threshold Per Cell", &fp.reseed_max_per_cell, 0.1f, 2, 64);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Cells above maximum fund replacements in interior cells below minimum.\n"
                                              "A step can never add more particles than it removed.");
                        }
                    } else {
                        ImGui::TextDisabled("Reseeding will not alter particle count; emitters and open boundaries still can.");
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
                    // A freshly-created Gas preset owns one convenience point
                    // source named "Smoke Source". Replacing the authoring source
                    // with a selected object should not leave that hidden centre
                    // plume active. Remove only that exact preset placeholder;
                    // user-created point sources remain additive.
                    if (!is_fluid_domain) {
                        auto& existing_sources = particles->flowSources();
                        for (int source_i = static_cast<int>(existing_sources.size()) - 1;
                             source_i >= 0;
                             --source_i) {
                            const auto& existing =
                                existing_sources[static_cast<std::size_t>(source_i)];
                            if (existing.domain_index == selected_domain_index &&
                                existing.source_mode ==
                                    RayTrophiSim::SimulationFlowSourceMode::Point &&
                                existing.name == "Smoke Source") {
                                particles->removeFlowSource(
                                    static_cast<std::size_t>(source_i));
                            }
                        }
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

                if (!is_fluid_domain) {
                    bool key_authoring =
                        scene.simulationKeyAuthoringMode();
                    if (ImGui::Checkbox(
                            "Keyframe Edit Mode##FlowKeyAuthoring",
                            &key_authoring)) {
                        scene.setSimulationKeyAuthoringMode(key_authoring);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "ON: timeline scrub selects an authoring frame without running the gas solve.\n"
                            "Slider edits remain staged until you click a diamond; use this to create a\n"
                            "different second key. OFF: the solve runs normally.\n"
                            "The sliders always show the AUTHORED value; the interpolated value at the\n"
                            "playhead is shown read-only under each source.");
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled(
                        key_authoring ? "Staging values; solve paused"
                                      : "Playback/evaluation");
                }
                
                auto& flow_sources = particles->flowSources();
                int remove_flow_index = -1;
                int visible_flow_count = 0;
                
                // Draw flow sources as a flat list integrated into the collapsing section
                for (int flow_i = 0; flow_i < static_cast<int>(flow_sources.size()); ++flow_i) {
                    auto& source = flow_sources[static_cast<std::size_t>(flow_i)];
                    if (source.domain_index != selected_domain_index) continue;
                    
                    ++visible_flow_count;
                    ImGui::PushID(92000 + flow_i);

                    const int flow_key_frame =
                        timeline ? timeline->getCurrentFrame() : 0;
                    // ★★ NO PLAYHEAD WRITE-BACK. This used to mirror the evaluated
                    // key values into the DESCRIPTOR so the widgets below showed
                    // the animated value. Those same fields feed the simulation
                    // config signature, so the mirror silently re-authored the
                    // scene once per UI frame.
                    //
                    // For fluid that respawned frame 0 forever. For gas the cost
                    // was worse and took longer to see: while the playhead sits
                    // on a stretch where a key is actually INTERPOLATING (a
                    // temperature ramping down between two keys), the mirrored
                    // value differs every single frame, the signature changes,
                    // the sim is declared stale and a rewind-to-start is
                    // requested — playback snaps back to frame 0 mid-shot, with
                    // the frame number it happens at being wherever the ramp is.
                    // Constant stretches look fine, which is why this reads as an
                    // intermittent bug tied to having the panel open.
                    //
                    // The descriptor is the AUTHORED value and nothing may write
                    // it but the user. The keyed value is displayed read-only
                    // below, and the viewport gizmo resolves the real emission
                    // pose through resolveFlowSourceFrame (same as the solver).
                    if (!source.keyframes.empty()) {
                        const auto preview =
                            RayTrophiSim::evaluateSimulationFlowSource(
                                source, flow_key_frame);
                        ImGui::TextDisabled(
                            "Keyed @ frame %d: %s | temp %.2f | fuel %.2f | radius %.2f",
                            flow_key_frame, preview.enabled ? "on" : "off",
                            preview.temperature, preview.fuel, preview.radius);
                    }
                    auto flowKeyButton = [&](const char* id, bool keyed) {
                        ImGui::PushID(id);
                        const float s = ImGui::GetFrameHeight();
                        const ImVec2 pos = ImGui::GetCursorScreenPos();
                        const bool clicked =
                            ImGui::InvisibleButton("flow_key", ImVec2(s, s));
                        ImU32 fill = keyed ? IM_COL32(255, 200, 0, 255)
                                          : IM_COL32(40, 40, 40, 255);
                        ImU32 edge = ImGui::IsItemHovered()
                            ? IM_COL32(255, 255, 255, 255)
                            : IM_COL32(180, 180, 180, 255);
                        if (ImGui::IsItemHovered() && !keyed)
                            fill = IM_COL32(70, 70, 70, 255);
                        const float cx = pos.x + s * 0.5f;
                        const float cy = pos.y + s * 0.5f;
                        const float r = s * 0.22f;
                        const ImVec2 points[4] = {
                            {cx, cy-r}, {cx+r, cy}, {cx, cy+r}, {cx-r, cy}
                        };
                        ImGui::GetWindowDrawList()->AddConvexPolyFilled(
                            points, 4, fill);
                        ImGui::GetWindowDrawList()->AddPolyline(
                            points, 4, edge, ImDrawFlags_Closed, 1.0f);
                        ImGui::PopID();
                        return clicked;
                    };
                    auto mirrorFlowKey = [&](const auto& key) {
                        Keyframe marker(flow_key_frame);
                        marker.has_emitter = true;
                        marker.emitter.has_enabled = key.has_enabled;
                        marker.emitter.enabled = key.enabled;
                        marker.emitter.has_position = key.has_position;
                        marker.emitter.position = key.position;
                        marker.emitter.has_velocity = key.has_velocity;
                        marker.emitter.velocity = key.velocity;
                        marker.emitter.has_radius = key.has_radius;
                        marker.emitter.radius = key.radius;
                        marker.emitter.has_density_rate = key.has_density;
                        marker.emitter.density_rate = key.density;
                        marker.emitter.has_temperature = key.has_temperature;
                        marker.emitter.temperature = key.temperature;
                        marker.emitter.has_fuel_rate = key.has_fuel;
                        marker.emitter.fuel_rate = key.fuel;
                        scene.timeline.insertKeyframe(
                            "Simulation Flow " +
                                std::to_string(source.timeline_uid),
                            marker);
                    };
                    auto insertFlowPropertyKey = [&](auto set_property) {
                        auto& key = source.keyframes[flow_key_frame];
                        set_property(key);
                        mirrorFlowKey(key);
                        scene.clearSimFrameCache();
                        scene.requestSimulationTimelineRenderResync();
                    };
                    auto removeFlowPropertyKey = [&](auto clear_property) {
                        auto it = source.keyframes.find(flow_key_frame);
                        if (it == source.keyframes.end()) return;
                        clear_property(it->second);
                        const auto& k = it->second;
                        const bool has_any =
                            k.has_enabled || k.has_position ||
                            k.has_velocity || k.has_radius ||
                            k.has_density || k.has_temperature ||
                            k.has_fuel || k.has_falloff ||
                            k.has_velocity_coupling || k.has_flow_rate;
                        const std::string track_name =
                            "Simulation Flow " +
                            std::to_string(source.timeline_uid);
                        if (has_any) {
                            mirrorFlowKey(k);
                        } else {
                            source.keyframes.erase(it);
                            scene.timeline.removeKeyframe(
                                track_name, flow_key_frame);
                        }
                        scene.clearSimFrameCache();
                        scene.requestSimulationTimelineRenderResync();
                    };
                    auto keyedFlowProperty = [&](auto has_property) {
                        const auto it = source.keyframes.find(flow_key_frame);
                        return it != source.keyframes.end() &&
                               has_property(it->second);
                    };
                    auto updateCurrentFlowKey = [&](auto update_property) {
                        if (!ImGui::IsItemEdited()) return;
                        auto it = source.keyframes.find(flow_key_frame);
                        if (it != source.keyframes.end()) {
                            update_property(it->second);
                            mirrorFlowKey(it->second);
                            scene.clearSimFrameCache();
                            scene.requestSimulationTimelineRenderResync();
                        }
                    };

                    ImGui::Spacing();
                    if (!is_fluid_domain) {
                        const bool keyed = keyedFlowProperty(
                            [](const auto& k){ return k.has_enabled; });
                        if (flowKeyButton("enabled", keyed)) {
                            if (keyed) removeFlowPropertyKey(
                                [](auto& k){ k.has_enabled = false; });
                            else insertFlowPropertyKey([&](auto& k) {
                                k.has_enabled = true; k.enabled = source.enabled;
                            });
                        }
                        ImGui::SameLine();
                    }
                    ImGui::Checkbox("Flow Source Enabled##FlowEnabled", &source.enabled);
                    updateCurrentFlowKey([&](auto& k) {
                        if (k.has_enabled) k.enabled = source.enabled;
                    });
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

                    if (!is_fluid_domain) {
                        const bool keyed = keyedFlowProperty([](const auto& k){ return k.has_radius; });
                        if (flowKeyButton("radius", keyed)) {
                            if (keyed) removeFlowPropertyKey([](auto& k){ k.has_radius = false; });
                            else insertFlowPropertyKey([&](auto& k){ k.has_radius = true; k.radius = source.radius; });
                        }
                        ImGui::SameLine();
                    }
                    ImGui::DragFloat("Source Radius", &source.radius, 0.01f, 0.001f, 1000.0f, "%.3f");
                    updateCurrentFlowKey([&](auto& k) {
                        if (k.has_radius) k.radius = source.radius;
                    });
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
                                ImGui::SetTooltip("Object Bounding Box uses the linked object's complete AABB volume.\n"
                                                  "Radius is ignored in this mode.");
                                break;
                        }
                    }
                    source.radius = std::max(0.001f, source.radius);

                    if (is_fluid_domain) {
                        const bool rate_keyed = keyedFlowProperty([](const auto& k){ return k.has_flow_rate; });
                        if (flowKeyButton("flow_rate", rate_keyed)) {
                            if (rate_keyed) removeFlowPropertyKey([](auto& k){ k.has_flow_rate = false; });
                            else insertFlowPropertyKey([&](auto& k){
                                k.has_flow_rate = true;
                                k.flow_rate = source.fluid_particles_per_second;
                            });
                        }
                        ImGui::SameLine();
                        ImGui::DragFloat("Injected Particles / Sec", &source.fluid_particles_per_second, 10.0f, 0.0f, 1000000.0f, "%.0f");
                        updateCurrentFlowKey([&](auto& k) {
                            if (k.has_flow_rate) k.flow_rate = source.fluid_particles_per_second;
                        });
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Flow rate of liquid particles spawned per second.\n"
                                              "Key this to animate a valve — open at one frame, throttled at another.");
                        }
                        source.fluid_particles_per_second = std::max(0.0f, source.fluid_particles_per_second);
                    } else {
                        const bool density_keyed = keyedFlowProperty([](const auto& k){ return k.has_density; });
                        if (flowKeyButton("density", density_keyed)) {
                            if (density_keyed) removeFlowPropertyKey([](auto& k){ k.has_density = false; });
                            else insertFlowPropertyKey([&](auto& k){ k.has_density = true; k.density = source.density; });
                        }
                        ImGui::SameLine();
                        ImGui::DragFloat("Soot Density Rate",     &source.density,     0.05f, 0.0f, 1000.0f, "%.3f");
                        updateCurrentFlowKey([&](auto& k) {
                            if (k.has_density) k.density = source.density;
                        });
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Rate of visual smoke density injected per second.");
                        }
                        const bool temperature_keyed = keyedFlowProperty([](const auto& k){ return k.has_temperature; });
                        if (flowKeyButton("temperature", temperature_keyed)) {
                            if (temperature_keyed) removeFlowPropertyKey([](auto& k){ k.has_temperature = false; });
                            else insertFlowPropertyKey([&](auto& k){ k.has_temperature = true; k.temperature = source.temperature; });
                        }
                        ImGui::SameLine();
                        ImGui::DragFloat("Thermal Temperature Rate", &source.temperature, 0.05f, 0.0f, 1000.0f, "%.3f");
                        updateCurrentFlowKey([&](auto& k) {
                            if (k.has_temperature) k.temperature = source.temperature;
                        });
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Thermal energy injected per second, making gas rise faster due to buoyancy.");
                        }
                        const bool fuel_keyed = keyedFlowProperty([](const auto& k){ return k.has_fuel; });
                        if (flowKeyButton("fuel", fuel_keyed)) {
                            if (fuel_keyed) removeFlowPropertyKey([](auto& k){ k.has_fuel = false; });
                            else insertFlowPropertyKey([&](auto& k){ k.has_fuel = true; k.fuel = source.fuel; });
                        }
                        ImGui::SameLine();
                        ImGui::DragFloat("Combustion Fuel Rate",        &source.fuel,        0.05f, 0.0f, 1000.0f, "%.3f");
                        updateCurrentFlowKey([&](auto& k) {
                            if (k.has_fuel) k.fuel = source.fuel;
                        });
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Quantity of combustible fuel injected per second for fire rendering.");
                        }
                        const bool falloff_keyed = keyedFlowProperty([](const auto& k){ return k.has_falloff; });
                        if (flowKeyButton("falloff", falloff_keyed)) {
                            if (falloff_keyed) removeFlowPropertyKey([](auto& k){ k.has_falloff = false; });
                            else insertFlowPropertyKey([&](auto& k){ k.has_falloff = true; k.falloff = source.falloff; });
                        }
                        ImGui::SameLine();
                        ImGui::DragFloat("Radial Falloff Blend",     &source.falloff,     0.05f, 0.0f, 16.0f,    "%.2f");
                        updateCurrentFlowKey([&](auto& k) {
                            if (k.has_falloff) k.falloff = source.falloff;
                        });
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Radial falloff interpolation from emitter core to boundaries.");
                        }
                        source.falloff = std::max(0.0f, source.falloff);
                    }
                    if (!is_fluid_domain) {
                        const bool keyed = keyedFlowProperty([](const auto& k){ return k.has_velocity; });
                        if (flowKeyButton("velocity", keyed)) {
                            if (keyed) removeFlowPropertyKey([](auto& k){ k.has_velocity = false; });
                            else insertFlowPropertyKey([&](auto& k){ k.has_velocity = true; k.velocity = source.velocity; });
                        }
                        ImGui::SameLine();
                    }
                    ImGui::DragFloat3("Emission Velocity", &source.velocity.x, 0.05f, -10000.0f, 10000.0f, "%.2f");
                    updateCurrentFlowKey([&](auto& k) {
                        if (k.has_velocity) k.velocity = source.velocity;
                    });
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Initial velocity (m/s) given to each emitted particle / injected cell.\n"
                                          "This is a one-time launch velocity, NOT a continuous force — gravity and\n"
                                          "buoyancy take over afterwards. Liquid: point it down to pour; gas: up to plume.");
                    }

                    if (!is_fluid_domain) {
                        const bool keyed = keyedFlowProperty([](const auto& k){ return k.has_velocity_coupling; });
                        if (flowKeyButton("velocity_coupling", keyed)) {
                            if (keyed) removeFlowPropertyKey([](auto& k){ k.has_velocity_coupling = false; });
                            else insertFlowPropertyKey([&](auto& k){ k.has_velocity_coupling = true; k.velocity_coupling = source.velocity_coupling; });
                        }
                        ImGui::SameLine();
                        ImGui::DragFloat("Velocity Coupling / Sec",
                                         &source.velocity_coupling,
                                         0.1f, 0.0f, 100.0f, "%.2f");
                        updateCurrentFlowKey([&](auto& k) {
                            if (k.has_velocity_coupling)
                                k.velocity_coupling =
                                    source.velocity_coupling;
                        });
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("How quickly gas near the source approaches Emission Velocity.\n"
                                              "0 disables coupling; 4-12 is stable for continuous fire/smoke.");
                        }
                        source.velocity_coupling =
                            std::max(0.0f, source.velocity_coupling);
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

                        // ── Substance ────────────────────────────────────────
                        {
                            // Fixed buffer rather than an ImGui string callback:
                            // a substance name is an identifier, and 63 chars is
                            // past any sane one. Truncation is silent but the
                            // field shows exactly what was kept.
                            char subs_buf[64] = {0};
                            const std::size_t copy_n =
                                (std::min)(source.fluid_substance.size(), sizeof(subs_buf) - 1);
                            source.fluid_substance.copy(subs_buf, copy_n);
                            subs_buf[copy_n] = '\0';
                            if (ImGui::InputText("Substance", subs_buf, sizeof(subs_buf))) {
                                source.fluid_substance = subs_buf;
                            }
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip(
                                    "What this source POURS. Leave empty for untagged liquid, which\n"
                                    "takes the domain's single surface material \xE2\x80\x94 the behaviour every\n"
                                    "existing scene has.\n\n"
                                    "Two sources naming the SAME substance merge into one\n"
                                    "indistinguishable body. Two naming DIFFERENT substances mix, and\n"
                                    "the mixture is a real field carried by the liquid, not a blend\n"
                                    "applied at the surface.\n\n"
                                    "The name is the identity: it is hashed onto every particle this\n"
                                    "source emits and rides advection, compaction and reseeding.\n"
                                    "Renaming it makes NEW liquid a different substance \xE2\x80\x94 liquid\n"
                                    "already in the domain keeps what it was poured as.");
                            }
                        }
                    }

                    if (source.source_mode == RayTrophiSim::SimulationFlowSourceMode::Point) {
                        if (!is_fluid_domain) {
                            const bool keyed = keyedFlowProperty([](const auto& k){ return k.has_position; });
                            if (flowKeyButton("position", keyed)) {
                                if (keyed) removeFlowPropertyKey([](auto& k){ k.has_position = false; });
                                else insertFlowPropertyKey([&](auto& k){ k.has_position = true; k.position = source.position; });
                            }
                            ImGui::SameLine();
                        }
                        ImGui::DragFloat3(source.parent_object.empty()
                                              ? "World Coordinates Position"
                                              : "Local Offset From Parent",
                                          &source.position.x, 0.05f, -10000.0f, 10000.0f, "%.2f");
                        updateCurrentFlowKey([&](auto& k) {
                            if (k.has_position) k.position = source.position;
                        });
                        if (ImGui::IsItemHovered() && !source.parent_object.empty()) {
                            ImGui::SetTooltip("Offset in the parent object's local space.\n"
                                              "This is what puts a flame on a match's TIP rather than\n"
                                              "at the centre of its bounding box.");
                        }
                    } else {
                        ImGui::TextDisabled("Linked Scene Mesh: %s", source.source_name.empty() ? "None" : source.source_name.c_str());
                    }

                    // ── Object binding (parenting) ───────────────────────────
                    // Orthogonal to Emitter Geometry Type: the source rides the
                    // parent's transform whether it is keyframed or driven by
                    // rigid-body physics. ObjectBounds/MeshSurface resolve their
                    // centre from the linked mesh, so parenting only changes the
                    // emission point for Point sources — but the inherited
                    // velocity applies to every mode.
                    ImGui::Spacing();
                    if (source.parent_object.empty()) {
                        const bool can_parent =
                            ui_ctx.selection.selected.type == SelectableType::Object &&
                            ui_ctx.selection.selected.object != nullptr &&
                            !ui_ctx.selection.selected.object->getNodeName().empty();
                        if (!can_parent) ImGui::BeginDisabled();
                        if (ImGui::Button("Parent To Selected Object##FlowParent", ImVec2(-1, 24))) {
                            const std::string node =
                                ui_ctx.selection.selected.object->getNodeName();
                            Matrix4x4 parent_to_world;
                            if (scene.resolveObjectTransformForSimulation(node, parent_to_world)) {
                                // Convert the authored world position into the
                                // parent's space so the source does not jump the
                                // instant it is parented.
                                source.position =
                                    parent_to_world.inverse().transform_point(source.position);
                                if (source.velocity_space ==
                                    RayTrophiSim::SimulationEmissionVelocitySpace::Local) {
                                    source.velocity =
                                        parent_to_world.inverse().transform_vector(source.velocity);
                                }
                                source.parent_object = node;
                                source.parent_prev_position = Vec3(-1.0e10f, 0.0f, 0.0f);
                                source.parent_velocity = Vec3(0.0f);
                                scene.clearSimFrameCache();
                                scene.requestSimulationTimelineRenderResync();
                            }
                        }
                        if (!can_parent) ImGui::EndDisabled();
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Bind this source to the selected object. It then follows\n"
                                              "that object every frame — including motion produced by\n"
                                              "rigid-body physics, not just keyframes.");
                        }
                    } else {
                        Matrix4x4 parent_probe;
                        const bool parent_found =
                            scene.resolveObjectTransformForSimulation(
                                source.parent_object, parent_probe);
                        if (parent_found) {
                            ImGui::TextColored(ImVec4(0.5f, 0.85f, 1.0f, 1.0f),
                                               "Parented To: %s", source.parent_object.c_str());
                        } else {
                            // The solver skips an unresolvable parent, so say so
                            // here instead of leaving a silently dead source.
                            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f),
                                               "Parent NOT FOUND: %s (source inactive)",
                                               source.parent_object.c_str());
                        }
                        ImGui::SameLine();
                        if (ImGui::SmallButton("Unparent##FlowUnparent")) {
                            Matrix4x4 parent_to_world;
                            if (scene.resolveObjectTransformForSimulation(
                                    source.parent_object, parent_to_world)) {
                                // Bake the current world placement back so the
                                // source stays exactly where it is on release.
                                source.position = parent_to_world.transform_point(source.position);
                                if (source.velocity_space ==
                                    RayTrophiSim::SimulationEmissionVelocitySpace::Local) {
                                    source.velocity = parent_to_world.transform_vector(source.velocity);
                                }
                            }
                            source.parent_object.clear();
                            source.parent_prev_position = Vec3(-1.0e10f, 0.0f, 0.0f);
                            source.parent_velocity = Vec3(0.0f);
                            scene.clearSimFrameCache();
                            scene.requestSimulationTimelineRenderResync();
                        }

                        int space_idx = static_cast<int>(source.velocity_space);
                        const char* space_labels[] = { "Local (rotates with parent)", "World (fixed direction)" };
                        if (ImGui::Combo("Velocity Space##FlowVelSpace", &space_idx,
                                         space_labels, IM_ARRAYSIZE(space_labels))) {
                            source.velocity_space =
                                static_cast<RayTrophiSim::SimulationEmissionVelocitySpace>(space_idx);
                            scene.clearSimFrameCache();
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Local: the emission direction turns with the object, so a\n"
                                              "nozzle keeps spraying out of its own muzzle.\n"
                                              "World: the direction stays fixed no matter how it rotates.");
                        }

                        ImGui::DragFloat("Inherit Parent Velocity##FlowInherit",
                                         &source.inherit_velocity, 0.02f, 0.0f, 4.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("How much of the parent's own motion is carried into the\n"
                                              "emitted medium. 0 leaves the flame behind a waved match and\n"
                                              "makes a moving hose pour a vertical wall instead of an arc.");
                        }
                        if (ImGui::IsItemEdited()) scene.clearSimFrameCache();

                        const Vec3 pv = source.parent_velocity;
                        ImGui::TextDisabled("Parent speed: %.2f m/s  (%.2f, %.2f, %.2f)",
                                            pv.length(), pv.x, pv.y, pv.z);
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
                    if (ImGui::CollapsingHeader("Liquid Display", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Spacing();
                    ImGui::TextDisabled("Domain default; Substance Overrides can replace it per liquid type.");

                    int current_mode_idx = 0; // default to Particles
                    if (domain.fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) {
                        current_mode_idx = 1;
                    } else if (domain.fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::Volume) {
                        // Display only — do NOT write. Drawing a panel must never
                        // repair scene data: this assignment was the real reason a
                        // scripted liquid rendered only after its panel had been
                        // opened. syncSimulationRenderVolumes now normalises the
                        // invalid 'Volume' liquid mode where it is consumed.
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

                    // ★★★ WHAT IS ACTUALLY DRAWN, resolved the way the render
                    // bridge resolves it. Two knobs answer one question here — a
                    // domain default and a per-substance override — and the combo
                    // above can only ever show one of them. With several
                    // substances bound, the mode it displays may be true for
                    // NOTHING on screen. Reading the answer back from the same
                    // rule the bridge uses is the only line in this panel that
                    // cannot drift from the picture.
                    {
                        const bool domain_is_splat = (current_mode_idx == 0);
                        std::string as_spheres, as_surface;
                        for (const auto& b : domain.fluid_substance_materials) {
                            bool splat = domain_is_splat;
                            if (b.representation == RayTrophiSim::Fluid::SubstanceRepresentation::Splat)
                                splat = true;
                            else if (b.representation == RayTrophiSim::Fluid::SubstanceRepresentation::SurfaceSDF)
                                splat = false;
                            std::string& bucket = splat ? as_spheres : as_surface;
                            if (!bucket.empty()) bucket += ", ";
                            bucket += b.substance.empty() ? std::string("(unnamed)") : b.substance;
                        }
                        // Untagged particles never match a binding, so they always
                        // follow the domain default. They are the reason the
                        // default still matters once every substance overrides it.
                        {
                            std::string& bucket = domain_is_splat ? as_spheres : as_surface;
                            if (!bucket.empty()) bucket += ", ";
                            bucket += "untagged";
                        }
                        ImGui::TextDisabled("Now drawing:");
                        ImGui::Indent(8.0f);
                        if (!as_surface.empty())
                            ImGui::TextWrapped("Isosurface: %s", as_surface.c_str());
                        if (!as_spheres.empty())
                            ImGui::TextWrapped("Splat spheres: %s", as_spheres.c_str());
                        ImGui::Unindent(8.0f);
                    }

                    if (ImGui::Checkbox("Debug Particle Points Overlay##DomainFluid", &domain.fluid_debug_overlay)) {
                        ui_ctx.start_render = true;
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Draws raw simulation particle coordinates as lightweight blue viewport overlays.");
                    }

                    // Per-substance overrides are domain-level authoring, so keep
                    // them outside both representation branches. A mixed domain
                    // must remain editable regardless of its default mode.
                    {
                        std::vector<std::string> domain_substances;
                        for (const auto& source : particles->flowSources()) {
                            if (source.domain_index != selected_domain_index || source.fluid_substance.empty()) continue;
                            if (std::find(domain_substances.begin(), domain_substances.end(),
                                          source.fluid_substance) == domain_substances.end())
                                domain_substances.push_back(source.fluid_substance);
                        }
                        std::sort(domain_substances.begin(), domain_substances.end());

                        const std::string substance_header = "Substance Overrides (" +
                            std::to_string(domain.fluid_substance_materials.size()) + ")";
                        if (ImGui::CollapsingHeader(substance_header.c_str(),
                                domain.fluid_substance_materials.empty() ? 0 : ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::TextDisabled("Emitter identity -> representation + scene material");

                            ImGui::Spacing();
                            ImGui::TextDisabled("Shared APIC velocity/pressure field");
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip(
                                    "All substances share ONE velocity and pressure field, so they still\n"
                                    "cannot separate by density.\n\n"
                                    "What IS per-substance: kinematic viscosity (a real force, solved on\n"
                                    "the grid), miscibility (how wide the boundary between two substances\n"
                                    "is), and PHASE \xE2\x80\x94 a solid substance is stamped into the grid's solid\n"
                                    "mask each step, so the liquid flows around it instead of through it.\n"
                                    "All three are set per binding below.");
                            }
                            // ── Solid-phase coupling, DOMAIN-WIDE ────────────
                            // Above the per-substance list on purpose: it is the
                            // switch that answers "is the solid what is wrong
                            // here?" and it must be reachable without editing —
                            // and losing — the phase authoring being tested.
                            {
                                bool solid_on = domain.fluid_solid_phase_enabled;
                                if (ImGui::Checkbox("Solid Phase Blocks Flow", &solid_on)) {
                                    domain.fluid_solid_phase_enabled = solid_on;
                                    scene.requestSimulationTimelineRenderResync();
                                    ui_ctx.start_render = true;
                                }
                                if (ImGui::IsItemHovered()) {
                                    ImGui::SetTooltip(
                                        "Master switch for every substance marked Solid in this domain.\n\n"
                                        "Off: the phase stays authored (nothing is lost) and simply stops\n"
                                        "being stamped into the grid, so the solve behaves exactly as it\n"
                                        "did before phases existed. That makes it the fastest way to tell\n"
                                        "whether a solid is responsible for something you are seeing.\n\n"
                                        "This is physics: toggling it re-bakes the simulation.");
                                }
                                if (domain.fluid_solid_phase_enabled) {
                                    float fill = domain.fluid_solid_phase_fill;
                                    ImGui::SetNextItemWidth(-FLT_MIN);
                                    if (ImGui::SliderFloat("Solid Cell Fill", &fill, 0.05f, 1.5f, "%.2f x seed")) {
                                        domain.fluid_solid_phase_fill = std::clamp(fill, 0.01f, 4.0f);
                                        scene.requestSimulationTimelineRenderResync();
                                        ui_ctx.start_render = true;
                                    }
                                    if (ImGui::IsItemHovered()) {
                                        ImGui::SetTooltip(
                                            "How full of solid parcels a cell must be before it blocks flow,\n"
                                            "as a fraction of the seed density (Particles Per Cell).\n\n"
                                            "LOW: one stray parcel dams a channel with a full voxel of wall.\n"
                                            "HIGH: a thin chunk blocks nothing while the panel still says\n"
                                            "Solid.\n\n"
                                            "Set it by READING, not guessing: the Fluid Step block reports\n"
                                            "solid parcels and blocked cells separately, and parcels with\n"
                                            "zero blocked cells means the chunk is thinner than the voxel\n"
                                            "size can express.\n\n"
                                            "A cell holding more liquid than solid never blocks, whatever\n"
                                            "this is set to — that is what keeps the liquid from being\n"
                                            "ejected when a chunk arrives.");
                                    }
                                }
                            }
                            ImGui::Separator();
                            auto& smgr = MaterialManager::getInstance();
                            int remove_binding = -1;
                            for (std::size_t bi = 0; bi < domain.fluid_substance_materials.size(); ++bi) {
                                auto& binding = domain.fluid_substance_materials[bi];
                                const auto& mats = smgr.getAllMaterials();
                                ImGui::PushID(static_cast<int>(bi) + 61000);
                                // One framed block per substance. Two bindings
                                // used to run together as an undifferentiated
                                // column of widgets, so telling which slider
                                // belonged to which chocolate meant counting
                                // rows from the last coloured name.
                                ImGui::BeginGroup();
                                ImGui::TextColored(ImVec4(0.55f, 0.85f, 1.0f, 1.0f), "%s", binding.substance.c_str());
                                if (binding.phase == RayTrophiSim::Fluid::SubstancePhase::Solid) {
                                    // The badge says what the row IS at a
                                    // glance. Phase changes the flow, and a
                                    // state of matter hidden inside a combo two
                                    // rows down reads as a display option.
                                    ImGui::SameLine();
                                    ImGui::TextColored(ImVec4(0.75f, 0.85f, 1.0f, 1.0f), "[SOLID]");
                                }
                                // Right-aligned from the window's content edge,
                                // not from what is LEFT after the name: the
                                // remaining width shrinks with a longer
                                // substance name, so an avail-based offset walks
                                // the button around per row.
                                ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 62.0f);
                                if (ImGui::SmallButton("Remove")) remove_binding = static_cast<int>(bi);
                                ImGui::Indent(8.0f);

                                // ── PHYSICS first, LOOK second ───────────────
                                // The order is the answer to "why did my edit
                                // not show up": everything above the Look
                                // separator re-simulates, everything below
                                // repaints the frame you are already on. Mixing
                                // the two in one list is what made the panel
                                // feel arbitrary — a material pick and a
                                // viscosity edit look identical as widgets and
                                // cost three orders of magnitude apart.
                                ImGui::SeparatorText("Physics (re-bakes the sim)");

                                // Phase FIRST: it is the question about what the
                                // matter is, and representation below only says
                                // how to draw it. Putting the render routing
                                // first is what made the two read as one knob.
                                int phase_idx = static_cast<int>(binding.phase);
                                const char* phases[] = { "Liquid", "Solid (blocks flow)" };
                                ImGui::SetNextItemWidth(-FLT_MIN);
                                if (ImGui::Combo("Phase", &phase_idx, phases, 2)) {
                                    binding.phase =
                                        static_cast<RayTrophiSim::Fluid::SubstancePhase>(phase_idx);
                                    // Physics: the baked frames were solved with
                                    // the old phase and no longer describe this
                                    // scene. The signature drops them; this only
                                    // asks the view to catch up.
                                    scene.requestSimulationTimelineRenderResync();
                                    ui_ctx.start_render = true;
                                }
                                if (ImGui::IsItemHovered()) {
                                    ImGui::SetTooltip(
                                        "What this substance IS \xE2\x80\x94 not how it is drawn (that is\n"
                                        "\"Drawn as\", under Look below, and the two are independent).\n\n"
                                        "Solid: its parcels are rasterized into the grid's solid mask every\n"
                                        "step, so the liquid flows AROUND them and, with a no-slip wall\n"
                                        "setting, clings to them. A moving chunk pushes the liquid.\n\n"
                                        "This is an OBSTACLE, not a rigid body: parcels have no cohesion,\n"
                                        "so a pile spreads under load and a chunk does not rotate. Cohesive\n"
                                        "dragged clusters belong to the rigid-body solver.\n\n"
                                        "A cell only blocks once about a quarter of the seed density of\n"
                                        "solid parcels is in it \xE2\x80\x94 a chunk thinner than a voxel blocks\n"
                                        "nothing. The domain's stats line reports parcels and blocked\n"
                                        "cells separately so the two cases can be told apart.");
                                }

                                bool inherit_visc = binding.kinematic_viscosity < 0.0f;
                                if (ImGui::Checkbox("Inherit Domain Viscosity", &inherit_visc)) {
                                    binding.kinematic_viscosity = inherit_visc
                                        ? -1.0f
                                        : std::max(0.0f, domain.fluid_params.kinematic_viscosity);
                                    ui_ctx.start_render = true;
                                }
                                if (!inherit_visc) {
                                    float visc = binding.kinematic_viscosity;
                                    ImGui::SetNextItemWidth(-FLT_MIN);
                                    if (ImGui::DragFloat("Kinematic Viscosity (m^2/s)", &visc,
                                                         1.0e-4f, 0.0f, 100.0f, "%.6f")) {
                                        binding.kinematic_viscosity = std::max(0.0f, visc);
                                        ui_ctx.start_render = true;
                                    }
                                    if (ImGui::IsItemHovered()) {
                                        ImGui::SetTooltip(
                                            "This substance's own viscosity, in ABSOLUTE units, not a\n"
                                            "multiple of the domain's. Reference: water 1e-6, olive oil\n"
                                            "8e-5, molten chocolate ~4e-3, honey ~7e-3, lava 0.1 .. 100.\n\n"
                                            "Solved on the grid as a variable-coefficient diffusion, so two\n"
                                            "substances with different values really do lag and fold against\n"
                                            "each other rather than just looking different.\n\n"
                                            "Raise the domain's Viscosity Sweeps if a thick substance still\n"
                                            "flows too freely: under-converging under-applies viscosity, it\n"
                                            "never explodes.");
                                    }
                                }

                                float misc = binding.miscibility;
                                ImGui::SetNextItemWidth(-FLT_MIN);
                                if (ImGui::SliderFloat("Miscibility", &misc, 0.0f, 1.0f, "%.2f")) {
                                    binding.miscibility = std::clamp(misc, 0.0f, 1.0f);
                                    scene.requestSimulationTimelineRenderResync();
                                    ui_ctx.start_render = true;
                                }
                                if (ImGui::IsItemHovered()) {
                                    ImGui::SetTooltip(
                                        "How wide the boundary is where this substance meets a DIFFERENT\n"
                                        "one.\n\n"
                                        "1.0 = fully miscible: a soft gradient, milk into chocolate.\n"
                                        "0.0 = immiscible: a sharp front, oil in water.\n\n"
                                        "A PAIR takes the smaller of its two values \xE2\x80\x94 refusing to mix is\n"
                                        "unilateral.\n\n"
                                        "This narrows the composition FIELD, not the shader, so every\n"
                                        "consumer sees the same boundary. It does not stop the substances\n"
                                        "from flowing through each other: they still share one velocity\n"
                                        "field, so this changes how the mixture LOOKS, not whether the\n"
                                        "liquids interpenetrate.");
                                }

                                // ── LOOK ─────────────────────────────────────
                                // Nothing below this line re-simulates: it
                                // repaints the frame currently on screen.
                                ImGui::SeparatorText("Look (repaints this frame)");

                                int rep = static_cast<int>(binding.representation);
                                const char* reps[] = { "Inherit Domain Default", "Splat Spheres", "Surface SDF" };
                                ImGui::SetNextItemWidth(-FLT_MIN);
                                if (ImGui::Combo("Drawn as", &rep, reps, 3)) {
                                    binding.representation = static_cast<RayTrophiSim::Fluid::SubstanceRepresentation>(rep);
                                    scene.requestSimulationTimelineRenderResync();
                                    ui_ctx.start_render = true;
                                }
                                if (ImGui::IsItemHovered()) {
                                    ImGui::SetTooltip(
                                        "How this substance is DRAWN, and nothing else. A solid can be\n"
                                        "splat spheres or part of the isosurface; so can a liquid.\n\n"
                                        "Splat Spheres also removes the substance from the shared\n"
                                        "isosurface, so its material is read per sphere instead of\n"
                                        "blended into the mixture.");
                                }

                                // ★★ THE UNSET LABEL MUST NAME THE FALLBACK THAT WILL
                                // ACTUALLY BE USED, and the two routes do not share
                                // one. An unset SPLAT substance falls through to the
                                // domain's opaque sphere material; only the
                                // isosurface route reaches the built-in dielectric.
                                // Labelling both "Built-in Dielectric" told a user
                                // whose spheres rendered opaque that their material
                                // had not been applied — when in truth the panel had
                                // named a material the splat path never consults.
                                const bool binding_draws_splat =
                                    binding.representation == RayTrophiSim::Fluid::SubstanceRepresentation::Splat ||
                                    (binding.representation == RayTrophiSim::Fluid::SubstanceRepresentation::Inherit &&
                                     current_mode_idx == 0);
                                const char* unset_label = binding_draws_splat
                                    ? "Domain Splat Material" : "Built-in Dielectric";
                                const char* mat_label = unset_label;
                                if (binding.material_id >= 0 && static_cast<std::size_t>(binding.material_id) < mats.size() &&
                                    mats[static_cast<std::size_t>(binding.material_id)])
                                    mat_label = mats[static_cast<std::size_t>(binding.material_id)]->materialName.c_str();
                                ImGui::SetNextItemWidth(-FLT_MIN);
                                if (ImGui::BeginCombo("Material", mat_label)) {
                                    // ★★★ requestSimulationTimelineRenderResync
                                    // IS THE FIX FOR "picking a material does
                                    // nothing until I rewind". The per-substance
                                    // material reaches the shader only through
                                    // the COMPOSITION field (a per-cell material
                                    // index), and that field is rebuilt from a
                                    // signature over the particles. On a paused
                                    // timeline nothing in that signature moves,
                                    // so the pick was stored, reported back
                                    // correctly, and never drawn. Its neighbours
                                    // here (representation, miscibility) always
                                    // asked for the resync; this row did not,
                                    // which is why one panel answered two ways.
                                    // The signature now hashes material_id too,
                                    // so the rebuild is correct even when the
                                    // caller forgets — this call is what makes
                                    // it happen NOW, on the frame being viewed.
                                    if (ImGui::Selectable(unset_label, binding.material_id < 0)) {
                                        binding.material_id = -1;
                                        scene.refreshFluidSurfaceMaterial();
                                        scene.requestSimulationTimelineRenderResync();
                                        ui_ctx.start_render = true;
                                    }
                                    for (std::size_t mi = 0; mi < mats.size(); ++mi) {
                                        if (!mats[mi]) continue;
                                        if (ImGui::Selectable(mats[mi]->materialName.c_str(),
                                                              binding.material_id == static_cast<int>(mi))) {
                                            binding.material_id = static_cast<int>(mi);
                                            scene.refreshFluidSurfaceMaterial();
                                            scene.requestSimulationTimelineRenderResync();
                                            ui_ctx.start_render = true;
                                        }
                                    }
                                    ImGui::EndCombo();
                                }
                                if (ImGui::SmallButton("+ New Material")) {
                                    auto fresh = std::make_shared<PrincipledBSDF>();
                                    const uint16_t id = smgr.addUniqueMaterial(binding.substance + " Material", fresh);
                                    if (id != MaterialManager::INVALID_MATERIAL_ID) {
                                        binding.material_id = static_cast<int>(id);
                                        scene.refreshFluidSurfaceMaterial();
                                        scene.requestSimulationTimelineRenderResync();
                                        ui_ctx.start_render = true;
                                    }
                                }

                                ImGui::Unindent(8.0f);
                                ImGui::EndGroup();
                                ImGui::Spacing();
                                ImGui::Separator();
                                ImGui::Spacing();
                                ImGui::PopID();
                            }
                            if (remove_binding >= 0) {
                                domain.fluid_substance_materials.erase(
                                    domain.fluid_substance_materials.begin() + remove_binding);
                                scene.requestSimulationTimelineRenderResync();
                                ui_ctx.start_render = true;
                            }

                            std::vector<std::string> available;
                            for (const auto& name : domain_substances) {
                                const bool bound = std::any_of(domain.fluid_substance_materials.begin(),
                                    domain.fluid_substance_materials.end(),
                                    [&](const auto& b) { return b.substance == name; });
                                if (!bound) available.push_back(name);
                            }
                            static std::string pending_substance;
                            if (std::find(available.begin(), available.end(), pending_substance) == available.end())
                                pending_substance = available.empty() ? std::string() : available.front();
                            const char* pending_label = available.empty()
                                ? "No unbound emitter substances" : pending_substance.c_str();
                            ImGui::BeginDisabled(available.empty());
                            ImGui::SetNextItemWidth(-FLT_MIN);
                            if (ImGui::BeginCombo("Add From Domain Emitters", pending_label)) {
                                for (const auto& name : available) {
                                    if (ImGui::Selectable(name.c_str(), pending_substance == name)) pending_substance = name;
                                }
                                ImGui::EndCombo();
                            }
                            const bool at_limit = domain.fluid_substance_materials.size() >=
                                RayTrophiSim::Fluid::kMaxFluidSubstanceMaterials;
                            ImGui::BeginDisabled(at_limit);
                            if (ImGui::Button("Add Override")) {
                                RayTrophiSim::SimulationGridDomainDesc::SubstanceMaterial b;
                                b.substance = pending_substance;
                                domain.fluid_substance_materials.push_back(std::move(b));
                                pending_substance.clear();
                                ui_ctx.start_render = true;
                            }
                            ImGui::EndDisabled();
                            ImGui::EndDisabled();
                            if (domain_substances.empty())
                                ImGui::TextDisabled("Assign a Substance to an emitter in this domain first.");
                        }
                    }

                    // ★ Gate the parameter blocks on what the COMBO SHOWS
                    // (current_mode_idx), not on the raw enum. They used to read
                    // the enum, so a domain still holding the invalid 'Volume'
                    // mode displayed "Smooth Glassy Surface" as selected and
                    // then matched NEITHER block — the section opened with a
                    // mode chosen and nothing underneath it. That is the failure
                    // that gets reported as "Liquid Visualization does not
                    // open", and it looked seed-mode-dependent only because
                    // touching the seed controls runs the sim, which is where
                    // the mode gets normalised.
                    //
                    // This does NOT repair the scene data — the panel still
                    // writes nothing (see the note above); it only stops the
                    // combo from claiming a mode the rest of the panel ignores.
                    const bool has_splat_override = std::any_of(
                        domain.fluid_substance_materials.begin(), domain.fluid_substance_materials.end(),
                        [](const auto& b) { return b.representation == RayTrophiSim::Fluid::SubstanceRepresentation::Splat; });
                    if ((current_mode_idx == 0 || has_splat_override) &&
                        ImGui::CollapsingHeader("Splat Geometry & Preview",
                            current_mode_idx == 0 ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
                        const char* geometry_modes[] = { "Built-in Icosphere", "Scene Object / Mesh Group" };
                        int geometry_mode = domain.fluid_particle_geometry_mode == 1 ? 1 : 0;
                        if (ImGui::Combo("Geometry Source", &geometry_mode, geometry_modes, 2)) {
                            domain.fluid_particle_geometry_mode = geometry_mode;
                            scene.requestSimulationTimelineRenderResync();
                            ui_ctx.start_render = true;
                        }
                        if (geometry_mode == 1) {
                            std::vector<std::string> scene_nodes;
                            for (const auto& object : scene.world.objects) {
                                if (!object) continue;
                                const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
                                if (!mesh || !mesh->geometry) continue;
                                const std::string name = mesh->nodeName;
                                if (name.empty() || name.front() == '[') continue;
                                if (scene.isEditorPendingDeleteObjectName(name)) continue;
                                if (std::find(scene_nodes.begin(), scene_nodes.end(), name) == scene_nodes.end())
                                    scene_nodes.push_back(name);
                            }
                            std::sort(scene_nodes.begin(), scene_nodes.end());
                            const bool source_is_live =
                                std::binary_search(scene_nodes.begin(), scene_nodes.end(),
                                                   domain.fluid_particle_geometry_source);
                            const std::string missing_source_label =
                                domain.fluid_particle_geometry_source.empty()
                                    ? std::string("Select scene object")
                                    : std::string("Missing: ") + domain.fluid_particle_geometry_source;
                            const char* source_label = source_is_live
                                ? domain.fluid_particle_geometry_source.c_str()
                                : missing_source_label.c_str();
                            if (ImGui::BeginCombo("Scene Geometry", source_label)) {
                                for (const auto& name : scene_nodes) {
                                    if (ImGui::Selectable(name.c_str(),
                                            domain.fluid_particle_geometry_source == name)) {
                                        domain.fluid_particle_geometry_source = name;
                                        scene.requestSimulationTimelineRenderResync();
                                        ui_ctx.start_render = true;
                                    }
                                }
                                ImGui::EndCombo();
                            }
                            if (scene_nodes.empty()) ImGui::TextDisabled("No scene mesh groups available.");
                            if (!domain.fluid_particle_geometry_source.empty() && !source_is_live) {
                                ImGui::TextDisabled("Selected source is deleted or unavailable; using icosphere fallback.");
                            }
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("Instances the selected scene node at every splat position.\n"
                                                  "Its triangles are centered and normalized once, while original per-face materials are preserved.\n"
                                                  "Only an explicitly assigned domain or substance material overrides them.");
                            }
                        }
                        {
                            auto& material_mgr = MaterialManager::getInstance();
                            const auto& scene_materials = material_mgr.getAllMaterials();
                            const std::string authored_material =
                                RayTrophiSim::Fluid::fluidSplatMaterialName(domain);
                            const char* inherited_label = geometry_mode == 1
                                ? "Use Source Object Materials"
                                : "Scene Default Material";
                            const char* splat_material_label = authored_material.empty()
                                ? inherited_label : authored_material.c_str();
                            if (ImGui::BeginCombo("Splat Material##DomainFluid", splat_material_label)) {
                                if (ImGui::Selectable(inherited_label, authored_material.empty())) {
                                    const auto result = RayTrophiSim::Fluid::setFluidSplatMaterial(
                                        *particles, domain.name, std::string{});
                                    if (result.ok() && result.changed) {
                                        scene.requestSimulationTimelineRenderResync();
                                        ui_ctx.renderer.resetCPUAccumulation();
                                        if (ui_ctx.backend_ptr) ui_ctx.backend_ptr->resetAccumulation();
                                        ui_ctx.start_render = true;
                                    }
                                }
                                for (std::size_t mi = 0; mi < scene_materials.size(); ++mi) {
                                    if (!scene_materials[mi]) continue;
                                    const std::string& material_name = scene_materials[mi]->materialName;
                                    ImGui::PushID(static_cast<int>(mi) + 52000);
                                    if (ImGui::Selectable(material_name.c_str(),
                                                          authored_material == material_name)) {
                                        const auto result = RayTrophiSim::Fluid::setFluidSplatMaterial(
                                            *particles, domain.name, material_name);
                                        if (result.ok() && result.changed) {
                                            scene.requestSimulationTimelineRenderResync();
                                            ui_ctx.renderer.resetCPUAccumulation();
                                            if (ui_ctx.backend_ptr) ui_ctx.backend_ptr->resetAccumulation();
                                            ui_ctx.start_render = true;
                                        }
                                    }
                                    ImGui::PopID();
                                }
                                ImGui::EndCombo();
                            }
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip(
                                    "Material used by Built-in Icosphere splats.\n"
                                    "Scene Default reuses an existing scene material and creates no hidden fluid material.\n\n"
                                    "For Scene Object geometry, Use Source Object Materials preserves every face material.\n"
                                    "Choosing a material here deliberately overrides the whole splat mesh.");
                            }
                        }
                        ImGui::Separator();
                        ImGui::DragFloat("Voxel Radius Factor##DomainFluid", &domain.fluid_particle_radius_factor, 0.01f, 0.05f, 1.5f, "%.2f");
                        ImGui::DragFloat("Visual Size Multiplier##DomainFluid", &domain.fluid_particle_size_multiplier, 0.01f, 0.05f, 8.0f, "%.2f");
                        if (geometry_mode == 0)
                            ImGui::SliderInt("Sphere Subdivision Detail##DomainFluid", &domain.fluid_particle_subdivisions, 0, 3);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Geometric subdivision level of particle spheres. Higher values are smoother but reduce rendering performance.");
                        }
                    }

                    // ★★ MIRROR OF THE SPLAT GATE ABOVE, and it was missing.
                    // A domain defaulting to spheres with ONE substance overridden
                    // to Surface SDF really does render an isosurface — and every
                    // control that shapes it was hidden, because this gate asked
                    // only about the domain default. The surface was on screen and
                    // unreachable, which reads as "these settings do not exist"
                    // rather than as a missing gate. The splat half already got
                    // this right; the two halves must agree or the panel teaches
                    // that overrides are second-class.
                    const bool has_sdf_override = std::any_of(
                        domain.fluid_substance_materials.begin(), domain.fluid_substance_materials.end(),
                        [](const auto& b) { return b.representation == RayTrophiSim::Fluid::SubstanceRepresentation::SurfaceSDF; });
                    if ((current_mode_idx == 1 || has_sdf_override) &&
                        ImGui::CollapsingHeader("Surface SDF Settings",
                            current_mode_idx == 1 ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {   // see the note above: display, not enum
                        bool sdf_changed = false;
                        // ★ PURE SHADER STATE, as opposed to sdf_changed. Nothing
                        // is re-simulated or re-uploaded from the level set; only
                        // values the shader reads out of the volume table change.
                        //
                        // It still needs its OWN consumer, and that is the thing
                        // that was missing: `ui_ctx.start_render = true` alone
                        // asks for another frame WITHOUT clearing the accumulator,
                        // so the new samples average into a converged image made
                        // with the old value. On a settled render the change is
                        // then invisible until something else happens to reset —
                        // which reads as "the control does nothing", the exact
                        // failure the greyed-out IOR slider two screens down
                        // exists to avoid.
                        bool look_changed = false;
                        sdf_changed |= ImGui::DragFloat("Level Set Kernel Radius", &domain.fluid_level_set_params.kernel_radius_voxels, 0.05f, 0.5f, 6.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Splats radius determining how far apart particles fuse into a unified liquid body.\nLarger values produce a thicker, fuller appearance.");
                        }
                        sdf_changed |= ImGui::DragFloat("Particle Voxel Radius (vx)", &domain.fluid_level_set_params.particle_radius_voxels, 0.02f, 0.05f, 2.0f, "%.2f");
                        sdf_changed |= ImGui::DragFloat("Surface Fullness (vx)", &domain.fluid_level_set_params.surface_offset_voxels, 0.02f, -0.75f, 1.25f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Moves the reconstructed SDF surface in simulation-voxel units.\n"
                                "Positive values make the liquid body fuller; negative values shrink it.\n"
                                "This is geometric and resolution-independent: it does not change fog/\n"
                                "absorption density, particle physics, grid resolution, or kernel cost.\n"
                                "For fracture work keep this modest so narrow cracks remain visible.");
                        }
                        sdf_changed |= ImGui::DragFloat("SDF Narrow Band Width", &domain.fluid_level_set_params.narrow_band_voxels, 0.05f, 1.0f, 8.0f, "%.2f");
                        sdf_changed |= ImGui::DragFloat("SDF Surface Band Width", &domain.fluid_surface_band_voxels, 0.02f, 0.1f, 3.0f, "%.2f");
                        sdf_changed |= ImGui::SliderInt("Laplacian Surface Smoothing", &domain.fluid_level_set_params.smoothing_iterations, 0, 8);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Number of Laplacian smoothing passes applied to the surface level-set boundary. Prevents voxel stair-stepping.");
                        }
                        // ── Coordinate space ─────────────────────────────────
                        // NOT an sdf_changed knob: the level set is untouched.
                        // This only chooses which coordinate the SHADER
                        // addresses its patterns in, so it repaints live.
                        ImGui::Separator();
                        ImGui::TextDisabled("Surface Coordinate Space");
                        {
                            // Order matches the shader's COORD_* constants; the
                            // combo index IS the value, so inserting an entry
                            // here silently re-labels every saved project.
                            // Append only.
                            const char* kCoordSpaces[] = { "Material (flows with liquid)",
                                                           "Domain (fixed to container)",
                                                           "World (fixed to scene)" };
                            int space = domain.fluid_surface_coord_space;
                            if (space < 0) space = 0;
                            if (space > 2) space = 2;
                            if (ImGui::Combo("Coordinate Space", &space, kCoordSpaces, 3)) {
                                domain.fluid_surface_coord_space = space;
                                look_changed = true;
                            }
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip(
                                    "Which coordinate ALL isosurface patterns are addressed in \xE2\x80\x94\n"
                                    "textures, the resin interior, porosity and the opacity mask.\n\n"
                                    "Material : each parcel of liquid carries its own coordinate, so a\n"
                                    "           pour takes its texture WITH it. Identical to World for\n"
                                    "           anything that has not moved yet.\n"
                                    "Domain   : fixed to the container. A carried vessel takes the\n"
                                    "           pattern along, but liquid flows THROUGH it.\n"
                                    "World    : nailed to the scene, like a slide projector.\n\n"
                                    "Material stretches with the flow. Two coordinate generations\n"
                                    "are blended to bound that stretch \xE2\x80\x94 see Refresh Period below.");
                            }

                            // Only Material mode uses the coordinate field, so
                            // the refresh schedule is meaningless in the other
                            // two. Shown greyed rather than hidden: a control
                            // that vanishes reads as a missing feature, while a
                            // disabled one says "not for this mode".
                            const bool material_mode = (space == 0);
                            ImGui::BeginDisabled(!material_mode);
                            int period = domain.fluid_params.uvw_refresh_period;
                            if (ImGui::DragInt("Coord Refresh Period", &period, 1.0f, 30, 2000,
                                               "%d steps")) {
                                domain.fluid_params.uvw_refresh_period =
                                    period < 2 ? 2 : period;
                                look_changed = true;
                            }
                            ImGui::EndDisabled();
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip(
                                    "Solver steps before a coordinate generation is reset to the\n"
                                    "identity. Two generations run half a period apart and are\n"
                                    "blended, so the texture is never carried by a map older than\n"
                                    "one period.\n\n"
                                    "Higher : the pattern follows the material further before it is\n"
                                    "         refreshed \xE2\x80\x94 more faithful advection, more smearing\n"
                                    "         near the end of each generation's life.\n"
                                    "Lower  : a crisper map, but the crossfade between the two\n"
                                    "         generations becomes visible as a soft pulsing.\n\n"
                                    "Counted in STEPS, not frames: the map degrades with deformation,\n"
                                    "and steps are what deform it.");
                            }
                        }

                        // ── Procedural porosity ──────────────────────────────
                        // NOT an sdf_changed knob: nothing is rebuilt. The pore
                        // field is evaluated in the shader against the SAME
                        // level set, so these repaint the current frame.
                        ImGui::Separator();
                        ImGui::TextDisabled("Porosity (crumb / aeration)");
                        if (ImGui::SliderFloat("Pore Amount", &domain.fluid_surface_pore_amount, 0.0f, 0.5f, "%.3f")) {
                            look_changed = true;
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Carves bubbles OUT OF THE FIELD before the surface is found, so the\n"
                                              "pores are real geometry: their rims get correct normals, refraction\n"
                                              "and self-shadowing. (An alpha cutout would punch rimless holes.)\n\n"
                                              "0 = off, identical to before. Past ~0.5 the body is eaten faster than\n"
                                              "it can close and disintegrates instead of becoming porous.");
                        }
                        if (domain.fluid_surface_pore_amount > 1e-4f) {
                            ImGui::Indent();
                            if (ImGui::DragFloat("Bubble Size (m)", &domain.fluid_surface_pore_scale, 0.002f, 0.001f, 2.0f, "%.3f")) {
                                look_changed = true;
                            }
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("Typical bubble diameter in WORLD units, not voxels — changing the\n"
                                                  "domain resolution re-renders the same crumb instead of resizing it.");
                            }
                            if (ImGui::SliderFloat("Size Variation", &domain.fluid_surface_pore_detail, 0.0f, 1.0f, "%.2f")) {
                                look_changed = true;
                            }
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("Mixes a finer bubble size into the coarse one.\n"
                                                  "0 = one uniform size (packing foam, aerated batter).\n"
                                                  "High = mixed sizes (bread crumb, fermented dough).");
                            }
                            ImGui::TextDisabled("Also clips gas handoff \xE2\x80\x94 by design.");
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("The pores are cut into the field itself, so the gas/liquid handoff\n"
                                                  "arbiter sees the same holes the shader draws. That is deliberate:\n"
                                                  "if the two used different fields, smoke would be clipped against a\n"
                                                  "surface that is not rendered, with nothing reporting it.");
                            }
                            ImGui::Unindent();
                        }
                        // Consume the pure-shader-state edits. refreshFluidSurfaceMaterial
                        // pushes the values onto the render volume AND sets
                        // g_gas_volumes_dirty, so they reach the SSBO this frame;
                        // both accumulators are then cleared so the change is
                        // actually visible instead of averaging into what is
                        // already on screen.
                        //
                        // ★ Deliberately NOT requestSimulationTimelineRenderResync
                        // (which sdf_changed uses): nothing about the level set
                        // moved, so forcing a rebuild + NanoVDB re-upload would
                        // make a colour knob cost a full surface reconstruction.
                        if (look_changed) {
                            scene.refreshFluidSurfaceMaterial();
                            ui_ctx.renderer.resetCPUAccumulation();
                            if (ui_ctx.backend_ptr) ui_ctx.backend_ptr->resetAccumulation();
                            ui_ctx.start_render = true;
                        }
                        ImGui::Separator();

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
                        {
                            auto& smgr = MaterialManager::getInstance();
                            const auto& surf_mats = smgr.getAllMaterials();
                            const char* surf_label = "Built-in Dielectric (IOR below)";
                            if (domain.fluid_surface_material_id >= 0 &&
                                domain.fluid_surface_material_id != MaterialManager::INVALID_MATERIAL_ID &&
                                static_cast<std::size_t>(domain.fluid_surface_material_id) < surf_mats.size()) {
                                surf_label = surf_mats[domain.fluid_surface_material_id]
                                                 ? surf_mats[domain.fluid_surface_material_id]->materialName.c_str()
                                                 : "(missing)";
                            }
                            if (ImGui::BeginCombo("Surface Material##DomainFluidSDF", surf_label)) {
                                const bool none_sel = (domain.fluid_surface_material_id < 0);
                                if (ImGui::Selectable("Built-in Dielectric (IOR below)", none_sel)) {
                                    domain.fluid_surface_material_id = -1;
                                    mat_changed = true;
                                }
                                for (std::size_t mi = 0; mi < surf_mats.size(); ++mi) {
                                    if (!surf_mats[mi]) continue;
                                    const bool sel = (domain.fluid_surface_material_id == static_cast<int>(mi));
                                    ImGui::PushID(static_cast<int>(mi) + 40000);
                                    if (ImGui::Selectable(surf_mats[mi]->materialName.c_str(), sel)) {
                                        domain.fluid_surface_material_id = static_cast<int>(mi);
                                        mat_changed = true;
                                    }
                                    ImGui::PopID();
                                }
                                ImGui::EndCombo();
                            }
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip(
                                    "Shades the reconstructed liquid surface with a full scene material —\n"
                                    "the same Principled BSDF a mesh gets: metallic, clearcoat,\n"
                                    "transmission and subsurface scattering. Use it for molten glass or\n"
                                    "metal, lava, mud, chocolate — anything the built-in water/glass\n"
                                    "dielectric cannot express.\n\n"
                                    "With a material bound, the material owns the look: its own\n"
                                    "roughness, IOR and transmission apply, and the two sliders below\n"
                                    "grey out because they drive the built-in dielectric only.\n\n"
                                    "TEXTURES DO APPLY. An isosurface has no UVs, so albedo,\n"
                                    "roughness, metallic and emission maps are projected tri-planarly\n"
                                    "and the material's UV Scale becomes WORLD UNITS PER TILE. The\n"
                                    "Coordinate Space above decides whether that projection travels\n"
                                    "with the liquid, with the container, or stays fixed to the scene.\n\n"
                                    "An OPACITY texture cuts REAL HOLES: it is subtracted from the\n"
                                    "field before the surface is found, so the rims get correct\n"
                                    "normals and refraction. It is a hard mask, not a dither — the\n"
                                    "gas handoff reads the same field and has to agree with it sample\n"
                                    "for sample. Scalar Opacity below 1 becomes transmission instead,\n"
                                    "exactly as it does on a mesh.\n\n"
                                    "NORMAL MAPS apply too, tri-planar with a whiteout blend, and\n"
                                    "they ride the same Coordinate Space â so bump detail travels\n"
                                    "with the liquid instead of the body flowing through it.");
                            }

                        }
                        // IOR and roughness drive the built-in dielectric only. With a
                        // material bound the material owns both, so the sliders are
                        // disabled rather than left live and ignored — a control that
                        // moves and changes nothing is read as a bug, and reported as one.
                        const bool builtin_surface = (domain.fluid_surface_material_id < 0);
                        ImGui::BeginDisabled(!builtin_surface);
                        mat_changed |= ImGui::DragFloat("Index of Refraction (IOR)", &domain.fluid_surface_ior, 0.005f, 1.0f, 2.5f, "%.3f");
                        // AllowWhenDisabled: without it a greyed slider shows no tooltip at
                        // all, which is precisely the "why is this dead?" state being avoided.
                        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                            ImGui::SetTooltip(builtin_surface
                                ? "Refractive index bending light passing through the liquid:\n1.333 = Water, 1.47 = Glycerin, 1.5 = Glass."
                                : "Driven by the bound Surface Material's IOR.\nClear the material to use this slider.");
                        }
                        mat_changed |= ImGui::DragFloat("Surface Roughness", &domain.fluid_surface_roughness, 0.005f, 0.0f, 1.0f, "%.3f");
                        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                            ImGui::SetTooltip(builtin_surface
                                ? "Microfacet roughness of the glassy liquid interface. 0.0 = perfectly mirror reflective, >0.0 = frosted reflection."
                                : "Driven by the bound Surface Material's roughness.\nClear the material to use this slider.");
                        }
                        ImGui::EndDisabled();
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
                                // Warn only when the mode actually SHOWN above is
                                // Particles. Testing `!= SurfaceSDF` also caught the
                                // invalid 'Volume' mode, which displays as — and
                                // normalises to — Surface SDF: the panel told you to
                                // pick a setting the combo already showed as picked.
                                if (fo.render_mode == RayTrophiSim::Fluid::FoamRenderMode::Volume &&
                                    domain.fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) {
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
                                // ★bin and classify were previously inside NEITHER
                                // published timer, so the old line understated the
                                // real per-step cost. foam_total is the honest one
                                // and drives the bars, so the phase a GPU foam
                                // solver must replace first is visible at a glance.
                                const float foam_total =
                                    fst.bin_ms + fst.advect_ms + fst.gen_ms +
                                    fst.classify_ms + fst.crit_gpu_ms + fst.neigh_gpu_ms +
                                    fst.gpu_exec_ms;
                                const bool foam_any_gpu = fst.crit_on_gpu || fst.neigh_on_gpu;
                                UIWidgets::PerfBlock fb("Foam", foam_total);
                                fb.Value("Alive", "%zu", fst.alive);
                                fb.Value("  spray / foam / bubble", "%zu / %zu / %zu",
                                         fst.spray, fst.foam, fst.bubble);
                                fb.Value("Spawned per step", "%zu", fst.spawned);
                                fb.Total(foam_any_gpu ? "Total (CPU+GPU)" : "CPU total",
                                         foam_total);
                                if (fst.crit_on_gpu) {
                                    fb.Time("crit dispatch+readback", fst.crit_gpu_ms, "GPU", 1);
                                }
                                if (fst.neigh_on_gpu) {
                                    fb.Time("neighbour readback", fst.neigh_gpu_ms, "GPU", 1);
                                }
                                if (fst.gpu_exec_ms > 0.0f) {
                                    fb.Time("kernel exec (bins+crit+neigh)",
                                            fst.gpu_exec_ms, "GPU", 1);
                                }
                                fb.Time("bin (CSR, serial)", fst.bin_ms, nullptr, 1);
                                fb.Time("advect", fst.advect_ms, nullptr, 1);
                                fb.Time("  neighbour gather", fst.advect_neighbour_ms,
                                        fst.neigh_on_gpu ? "GPU" : nullptr, 2);
                                fb.Time("gen", fst.gen_ms, nullptr, 1);
                                fb.Time("  crit", fst.crit_ms,
                                        fst.crit_on_gpu ? "GPU" : "CPU", 2);
                                fb.Time("  emit (serial RNG)", fst.emit_ms, nullptr, 2);
                                fb.Time("classify", fst.classify_ms, nullptr, 1);
                                fb.End();
                                UIWidgets::HelpMarker(
                                    "bin: CSR binning of the FLUID particles (fully serial)\n"
                                    "crit: Ihmsen spawn potentials, parallel over fluid particles\n"
                                    "emit: serial stochastic spawn (RNG)\n"
                                    "classify: per-type tally over every live foam particle\n\n"
                                    "neighbour gather: fluid-neighbour count per foam particle;\n"
                                    "  it picks each particle's type (spray/foam/bubble).\n\n"
                                    "gen = crit + emit. bin and classify are counted in CPU total\n"
                                    "but were in neither of the previously reported timers.\n"
                                    "crit scales with the FLUID particle count, not the foam count,\n"
                                    "so lowering Max Foam does not reduce it. The gather scales with\n"
                                    "the FOAM count, so Max Foam does bound that one.\n\n"
                                    "Both run on GPU only while the domain solves on Vulkan; the CUDA\n"
                                    "backend has no foam kernels, so foam stays on the host there.\n"
                                    "When bin reads 0.00 the device produced both, so the host has\n"
                                    "nothing left to look up.\n\n"
                                    "kernel exec: the submit+fence that actually runs the four foam\n"
                                    "kernels. It is the real GPU cost of whitewater — before it was\n"
                                    "measured, that time was billed to Density -> NanoVDB.");
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
                            const bool has_surface_sdf =
                                domain.fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF ||
                                std::any_of(domain.fluid_substance_materials.begin(),
                                            domain.fluid_substance_materials.end(),
                                            [](const auto& b) {
                                                return b.representation ==
                                                    RayTrophiSim::Fluid::SubstanceRepresentation::SurfaceSDF;
                                            });
                            if (has_surface_sdf) {
                                ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.30f, 1.0f),
                                                   "Surface SDF uses a geometric density proxy");
                                ImGui::TextWrapped(
                                    "Density, Remap and Edge Cutoff are fog controls and do not thicken "
                                    "the SDF. Shape comes from Surface SDF Settings. Scattering/Absorption "
                                    "shade the built-in dielectric; a bound Principled BSDF uses its own "
                                    "Transmission and Interior controls.");
                                ImGui::Separator();
                            }
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
            if (!is_fluid_domain && ImGui::CollapsingHeader(
                    "Gas Step Stats##DomainGasComputeStats",
                    ImGuiTreeNodeFlags_DefaultOpen)) {
                const auto& states = particles->gridDomainStates();
                if (selected_domain_index >= 0 &&
                    selected_domain_index < static_cast<int>(states.size())) {
                    const auto& st =
                        states[static_cast<std::size_t>(selected_domain_index)];
                    const ImVec4 color = !st.gas_gpu_requested
                        ? ImVec4(0.65f, 0.65f, 0.65f, 1.0f)
                        : (st.gas_gpu_active && !st.gas_gpu_partial
                            ? ImVec4(0.45f, 0.95f, 0.55f, 1.0f)
                            : ImVec4(1.0f, 0.72f, 0.25f, 1.0f));
                    ImGui::TextColored(color, "Compute: %s",
                                       st.gas_compute_status.c_str());

                    const auto& gs = st.gas_stats;
                    if (!st.valid || !gs.stepped) {
                        ImGui::TextDisabled(
                            "No step ran for this domain in the last frame, so "
                            "there is nothing to time.");
                    } else {
                    const auto& cpu = gs.cpu;
                    // Same guard as the fluid block: the measured total can be
                    // smaller than the rows it is supposed to contain (a stage
                    // whose cost lands in a submit the host never waited on), and
                    // dividing by it would print nonsense shares. Fall back to
                    // the sum of the rows whenever it fails to account for them.
                    const float gpu_sum =
                        gs.gpu_collider_source_ms + gs.gpu_source_upload_ms +
                        gs.gpu_fluid_combustion_ms + gs.gpu_velocity_advect_ms +
                        gs.gpu_scalar_advect_ms + gs.gpu_combustion_ms +
                        gs.gpu_body_forces_ms + gs.gpu_dissipation_ms +
                        gs.gpu_pressure_ms + gs.gpu_publish_ms + gs.gpu_majorant_ms;
                    const float phase_sum =
                        gs.voxelize_ms + gs.analysis_ms + gpu_sum + cpu.total_ms;
                    const float step_total = std::max(gs.total_ms, phase_sum);
                    const bool any_gpu = gpu_sum > 0.0f;

                    UIWidgets::PerfBlock blk("Gas Step", step_total);
                    blk.Value("Resolution", "%dx%dx%d  (%zu cells)",
                              st.resolution_x, st.resolution_y, st.resolution_z,
                              gs.cell_count);
                    const double occupancy = gs.cell_count > 0
                        ? 100.0 * static_cast<double>(st.active_density_cells) /
                          static_cast<double>(gs.cell_count)
                        : 0.0;
                    blk.Value("Active smoke cells", "%zu  (%.1f%% of grid)",
                              st.active_density_cells, occupancy);
                    blk.Value("Max / total density", "%.3f / %.1f",
                              st.max_density, gs.total_density);
                    blk.Value("Max temperature", "%.3f", gs.max_temperature);
                    if (gs.active_fuel_cells > 0 || gs.burning_cells > 0) {
                        blk.Value("Fuel cells / total fuel", "%zu / %.2f",
                                  gs.active_fuel_cells, gs.total_fuel);
                        blk.Value("Burning cells", "%zu", gs.burning_cells);
                    }
                    if (gs.solid_cells > 0) {
                        blk.Value("Collider cells", "%zu  (%.1f%% blocked)",
                                  gs.solid_cells,
                                  gs.cell_count > 0
                                      ? 100.0 * static_cast<double>(gs.solid_cells) /
                                        static_cast<double>(gs.cell_count)
                                      : 0.0);
                    }
                    blk.Value("Max speed", "%.2f u/s", gs.max_speed);
                    // Above 1 the semi-Lagrangian trace crosses more than one
                    // cell per step. It stays stable (unconditionally so), but
                    // detail smears — and that reads as "the solver is soft",
                    // not as "the timestep is too coarse", unless it is stated.
                    blk.Value("CFL (max)", gs.cfl > 1.0f ? "%.2f  (> 1: detail smears)" : "%.2f",
                              gs.cfl);
                    blk.Value("Dense grid memory", "%.1f MB",
                              static_cast<double>(gs.grid_memory_bytes) / (1024.0 * 1024.0));

                    blk.Total(any_gpu ? "Step total (CPU+GPU)" : "Step total",
                              step_total);
                    if (gpu_sum > 0.0f) {
                        blk.Section("Device stages");
                        if (gs.gpu_collider_source_ms > 0.0f)
                            blk.Time("collider gas source", gs.gpu_collider_source_ms, "GPU", 1);
                        if (gs.gpu_msf_ms > 0.0f)
                            blk.Time("material state field (thermal)", gs.gpu_msf_ms, "GPU", 1);
                        if (gs.gpu_source_upload_ms > 0.0f)
                            blk.Time("source upload (host -> device)", gs.gpu_source_upload_ms, "GPU", 1);
                        if (gs.fluid_combustion_on_gpu || gs.gpu_fluid_combustion_ms > 0.0f)
                            blk.Time("liquid surface combustion", gs.gpu_fluid_combustion_ms, "GPU", 1);
                        blk.Time("velocity advection", gs.gpu_velocity_advect_ms,
                                 gs.velocity_advect_on_gpu ? "GPU" : "CPU", 1);
                        blk.Time("scalar advection", gs.gpu_scalar_advect_ms,
                                 gs.scalar_advect_on_gpu ? "GPU" : "CPU", 1);
                        blk.Time("combustion", gs.gpu_combustion_ms,
                                 gs.combustion_on_gpu ? "GPU" : "CPU", 1);
                        // The chain is all-or-nothing: if any link fails the
                        // solver clears every flag and the host redoes all four
                        // forces from the same post-advection checkpoint. A CPU
                        // tag with a non-zero time above therefore means device
                        // work was done and then discarded — worth seeing.
                        const bool forces_all_gpu =
                            gs.buoyancy_on_gpu && gs.force_fields_on_gpu &&
                            gs.vorticity_on_gpu && gs.turbulence_on_gpu;
                        blk.Time("body forces (buoyancy+fields+vort+turb)",
                                 gs.gpu_body_forces_ms,
                                 forces_all_gpu ? "GPU" : "CPU redo", 1);
                        blk.Time("velocity dissipation + clamp", gs.gpu_dissipation_ms,
                                 gs.dissipation_on_gpu ? "GPU" : "CPU", 1);
                        blk.Time("pressure projection", gs.gpu_pressure_ms,
                                 gs.pressure_on_gpu ? "GPU" : "CPU", 1);
                        blk.Time("field publication (RT bridge)", gs.gpu_publish_ms, "GPU", 1);
                        blk.Time("majorant (RT empty-space skip)", gs.gpu_majorant_ms, "GPU", 1);
                    }
                    blk.Section(gs.cpu.sparse_vdb ? "Host solver (sparse VDB)"
                                                  : "Host solver (dense)");
                    blk.Time("GridFluid::step", cpu.total_ms, "CPU", 1);
                    blk.Time("velocity advection", cpu.advect_velocity_ms, nullptr, 2);
                    blk.Time("scalar advection", cpu.advect_scalar_ms, nullptr, 2);
                    blk.Time("boundaries + solids", cpu.boundary_ms, nullptr, 2);
                    blk.Time("combustion", cpu.combustion_ms, nullptr, 2);
                    blk.Time("buoyancy", cpu.buoyancy_ms, nullptr, 2);
                    blk.Time("force fields", cpu.force_fields_ms, nullptr, 2);
                    blk.Time("vorticity", cpu.vorticity_ms, nullptr, 2);
                    blk.Time("turbulence", cpu.turbulence_ms, nullptr, 2);
                    blk.Time("dissipation", cpu.dissipation_ms, nullptr, 2);
                    if (cpu.pressure_iterations > 0) {
                        char sor_tag[32];
                        std::snprintf(sor_tag, sizeof(sor_tag), "%d sweeps",
                                      cpu.pressure_iterations);
                        blk.Time("pressure (SOR)", cpu.pressure_ms, sor_tag, 2);
                    } else {
                        blk.Time("pressure (SOR)", cpu.pressure_ms, nullptr, 2);
                    }
                    blk.Section("Host overhead");
                    blk.Time("collider voxelize + face weights", gs.voxelize_ms, "CPU", 1);
                    blk.Time("field analysis scan", gs.analysis_ms, "CPU", 1);
                    blk.End();
                    UIWidgets::HelpMarker(
                        "Device rows are measured around the dispatch, so they include "
                        "whatever submit/fence the host waited on — the cost the frame "
                        "really pays, not isolated kernel time.\n\n"
                        "A stage runs in EITHER column. A 0.00 ms host row under a GPU "
                        "tag above means the device covered it; 0.00 in both columns "
                        "means the stage never ran (channel off, or the feature is "
                        "disabled — vorticity/turbulence early-out at 0 strength).\n\n"
                        "body forces is one row because buoyancy, force fields, "
                        "vorticity and turbulence share a single velocity upload and "
                        "readback; splitting them would count that transfer four times.\n\n"
                        "field analysis is one pass over the cells producing the counters "
                        "above, plus the three face passes for max speed. It also covers "
                        "the density/bounds scan the renderer needs anyway, so it is not "
                        "all telemetry overhead — but it runs every step and is billed "
                        "here rather than hidden inside the total.\n\n"
                        "CFL = max face speed * dt / voxel size. Over 1 the advection "
                        "trace jumps more than a cell per step: stable, but smeared. "
                        "Lower the timestep or raise the resolution.");
                    }
                }
            }
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
                    // Bars are relative to the measured step total, so the
                    // dominant phase is visible without reading every number.
                    // fs.total_ms only covers the host-side Fluid::step call, so on
                    // the GPU path it collapses to ~0 while the phases below still
                    // report milliseconds — dividing by it printed shares in the
                    // hundred-thousands of percent. Fall back to the sum of the
                    // phases whenever it fails to account for them.
                    const float phase_sum =
                        fs.p2g_ms + fs.pressure_ms + fs.viscosity_ms +
                        fs.g2p_ms + fs.advect_ms + fs.density_ms;
                    const float step_total = std::max(fs.total_ms, phase_sum);
                    UIWidgets::PerfBlock blk("Fluid Step", step_total);
                    blk.Value("Particles", "%zu", fs.particle_count);
                    if (fs.reseed_added_particles > 0 || fs.reseed_removed_particles > 0) {
                        const long long reseed_net =
                            static_cast<long long>(fs.reseed_added_particles) -
                            static_cast<long long>(fs.reseed_removed_particles);
                        blk.Value("Dynamic reseed", "+%zu / -%zu (net %+lld)",
                                  fs.reseed_added_particles,
                                  fs.reseed_removed_particles,
                                  reseed_net);
                    }
                    blk.Value("Active fluid cells", "%zu", fs.active_fluid_cells);
                    // ★★★ THE RESOLUTION READING, and the row whose absence cost
                    // days. A fluid cell that touches air is held near p = 0 by
                    // the free-surface condition; only INTERIOR cells (liquid on
                    // all six sides) carry a pressure field. A stream one or two
                    // cells across is ALL surface: there is no pressure to build,
                    // nothing to eject a droplet with on impact, and it falls as
                    // loose particles and lands in a heap.
                    //
                    // ★★ On screen that is indistinguishable from a viscous
                    // liquid — which is why viscosity, damping, air drag and the
                    // surface settings were all tried first, and none of them
                    // could have worked. The cure is mass per second or voxel
                    // size, and this percentage is the only thing that says so.
                    if (fs.sealed_pockets_measured && fs.active_fluid_cells > 0) {
                        const double pct = 100.0 * static_cast<double>(fs.interior_fluid_cells) /
                                           static_cast<double>(fs.active_fluid_cells);
                        blk.Value("  of which interior", "%zu (%.0f%%)",
                                  fs.interior_fluid_cells, pct);
                        if (pct < 15.0) {
                            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.72f, 0.25f, 1.0f));
                            ImGui::TextWrapped(
                                "Almost every cell touches air, so this liquid has no "
                                "pressure field to speak of - it will fall without a "
                                "splash and pile up, whatever the viscosity says. "
                                "Give it more mass per second, or a smaller voxel "
                                "size, until the stream is several cells across.");
                            ImGui::PopStyleColor();
                        }
                    }
                    blk.Value("Recovered from solids", "%zu", fs.recovered_solid_particles);
                    // Both numbers, and only when a solid substance is actually
                    // present. Parcels with ZERO blocked cells is the reading
                    // that says the chunk is thinner than the voxel size can
                    // express — reporting only the cells would make that
                    // indistinguishable from "the binding never took", and the
                    // picture cannot tell them apart either.
                    // Shown only when it fires. A row reading 0 every frame is a
                    // row the eye stops seeing, and this one has to be noticed the
                    // one frame it turns non-zero.
                    if (domain.fluid_params.granular_enabled) {
                        blk.Value("Sealed pockets", "%s", "not applicable (granular MPM)");
                    } else if (!fs.sealed_pockets_measured) {
                        blk.Value("Sealed pockets", "%s", "not measured (GPU pressure)");
                    } else if (fs.sealed_pockets > 0) {
                        blk.Value("Sealed pockets", "%zu", fs.sealed_pockets);
                        blk.Value("  cells (no p reference)", "%zu", fs.sealed_pocket_cells);
                    }
                    if (fs.solid_phase_particles > 0 || fs.solid_phase_cells > 0) {
                        blk.Value("Solid-phase parcels", "%zu", fs.solid_phase_particles);
                        blk.Value("  blocking cells", "%zu", fs.solid_phase_cells);
                    }
                    blk.Total("Step total", step_total);
                    blk.Time("P2G", fs.p2g_ms, fs.p2g_on_gpu ? "GPU" : "CPU", 1);
                    blk.Time("Pressure", fs.pressure_ms,
                             domain.fluid_params.granular_enabled &&
                             fs.p2g_on_gpu && fs.g2p_on_gpu
                                 ? "off (granular MPM)"
                                 : (fs.pressure_on_gpu ? "GPU" : "CPU"), 1);
                    if (fs.pressure_on_gpu && fs.pressure_cg_max_iterations > 0) {
                        blk.Value("    MGPCG precond", "%s",
                                  fs.pressure_cg_multigrid ? "Layer B V-cycle" : "Layer A Jacobi");
                        blk.Value("    MGPCG iters", "%d / %d",
                                  fs.pressure_cg_iterations, fs.pressure_cg_max_iterations);
                        blk.Value("    MGPCG residual", "%.2e",
                                  fs.pressure_cg_final_relative_residual);
                        char dot_tag[32];
                        std::snprintf(dot_tag, sizeof(dot_tag), "%d", fs.pressure_cg_dot_count);
                        blk.Time("    MGPCG dot sync", fs.pressure_cg_dot_ms, dot_tag, 2);
                    }
                    // Tag it with the sweep count, not just a time. "0.00 ms" is
                    // the same reading for "the solve is fast" and "nu is 0 so
                    // nothing ran" — and the whole reason the old viscosity dial
                    // stayed broken for so long is that its silence looked normal.
                    {
                        char visc_tag[48];
                        if (fs.viscosity_sweeps_run > 0) {
                            std::snprintf(visc_tag, sizeof(visc_tag), "%s, %d sweeps",
                                          fs.viscosity_on_gpu ? "GPU" : "CPU",
                                          fs.viscosity_sweeps_run);
                        } else {
                            std::snprintf(visc_tag, sizeof(visc_tag), "off (nu=0)");
                        }
                        blk.Time("Viscosity", fs.viscosity_ms, visc_tag, 1);
                    }
                    blk.Time("G2P", fs.g2p_ms, fs.g2p_on_gpu ? "GPU" : "CPU", 1);
                    if (domain.fluid_params.granular_enabled) {
                        blk.Value("Granular yielded", "%zu", fs.granular_yielded_particles);
                        blk.Value("Granular detached", "%zu", fs.granular_detached_particles);
                        blk.Value("Granular invalid", "%zu", fs.granular_invalid_particles);
                        blk.Value("Granular sleeping", "%zu", fs.granular_sleeping_particles);
                        blk.Value("Granular damaged (>0.01%)", "%zu", fs.granular_damaged_particles);
                        blk.Value("  damage >= 10%", "%zu", fs.granular_damage_over_10_particles);
                        blk.Value("  damage >= 50%", "%zu", fs.granular_damage_over_50_particles);
                        blk.Value("  damage >= 90%", "%zu", fs.granular_damage_over_90_particles);
                        blk.Value("Granular max yield", "%.3e", fs.granular_max_yield_value);
                        blk.Value("Granular max plastic increment", "%.3e", fs.granular_max_plastic_increment);
                        blk.Value("Granular max plastic accumulated", "%.3e", fs.granular_max_accumulated_plastic);
                        blk.Value("Granular mean plastic accumulated", "%.3e", fs.granular_mean_accumulated_plastic);
                        blk.Value("Granular max bond opening", "%.3e", fs.granular_max_fracture_history);
                        blk.Value("Granular mean bond opening", "%.3e", fs.granular_mean_fracture_history);
                        blk.Value("Granular max damage", "%.3f", fs.granular_max_damage);
                        blk.Value("Granular mean damage", "%.3f", fs.granular_mean_damage);
                        blk.Value("Granular Young requested", "%.0f Pa", fs.granular_requested_young_modulus);
                        blk.Value("Granular Young effective", "%.0f Pa%s",
                                  fs.granular_effective_young_modulus,
                                  fs.granular_stiffness_capped ? " (CFL capped)" : "");
                        blk.Value("Granular elastic substeps needed", "%d",
                                  fs.granular_required_substeps);
                        blk.Value("  from wave CFL", "%d", fs.granular_wave_substeps);
                        blk.Value("  from strain rate", "%d / |C| = %.1f 1/s",
                                  fs.granular_strain_substeps, fs.granular_strain_rate);
                        blk.Value("Granular solver substeps run", "%d",
                                  fs.granular_solver_substeps);
                        // ★ THE ROW THAT MATTERS WHEN A PILE EXPLODES. Non-zero
                        // means the subcycle could not cover the motion and the
                        // stress kernel clamped dt*C to stay finite: the step
                        // survived, it was not correct. Raise
                        // granular_max_solver_substeps until this reads 0.
                        if (fs.granular_strain_limited_particles > 0) {
                            blk.Value("Granular strain-rate CLAMPED", "%zu particles",
                                      fs.granular_strain_limited_particles);
                        }
                        // Not an error row. Soft material compacts permanently,
                        // and this is that compaction being recorded instead of
                        // being dumped through a det(F) reset. A steady count on
                        // a soft preset is expected; a count on a stiff one means
                        // the material is carrying more load than it can hold.
                        if (fs.granular_compaction_capped_particles > 0) {
                            blk.Value("Granular plastic compaction", "%zu particles",
                                      fs.granular_compaction_capped_particles);
                        }
                        // Stability and validity are different questions. This
                        // one says the material is too soft to hold its own
                        // weight inside the small-strain model, which no amount
                        // of substepping repairs.
                        // Melting IS the material crossing the load gate, so the
                        // two rows belong together: how far it has softened, and
                        // whether it can still hold itself up.
                        if (fs.granular_softened_particles > 0) {
                            blk.Value("Granular softened", "%zu particles, min %.3f",
                                      fs.granular_softened_particles,
                                      fs.granular_min_softening);
                        }
                        if (fs.granular_stiffness_below_load) {
                            blk.Value("Granular TOO SOFT FOR LOAD",
                                      "%.0f Pa needed, overburden %.0f Pa",
                                      fs.granular_young_modulus_for_load,
                                      fs.granular_overburden_pressure);
                        }
                    }
                    char adv_tag[32];
                    std::snprintf(adv_tag, sizeof(adv_tag), "%d substeps", fs.advect_substeps);
                    blk.Time("Advect", fs.advect_ms, adv_tag, 1);
                    blk.Time("Density -> NanoVDB", fs.density_ms,
                             fs.density_on_gpu ? "GPU" : "CPU", 1);
                    blk.End();
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

        // Reset lives at the TOP of this panel now (see resetSimulationNow).
        // Deliberately NOT duplicated here: two buttons doing the same thing in
        // one panel is how a user ends up unsure whether they differ.
        ImGui::Separator();

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
                    // Only domains that OWN an authored initial state get re-seeded:
                    // a resting tank (FillLevel) or one the user actually seeded
                    // (fluid_reseed_on_reset, armed by Seed Fluid Now). Same
                    // predicate the reset path uses in SceneData.
                    //
                    // Re-arming every fluid domain instead handed an emitter-fed
                    // domain the DEFAULT seed AABB — the one synchronizeGridDomains
                    // drops into the upper half of the bounds for the gizmo to have
                    // something to show — so changing grid resolution while a hose
                    // was running dumped a block of water into the tank.
                    if (d.type == RayTrophiSim::SimulationDomainType::Fluid &&
                        (d.fluid_seed_mode == RayTrophiSim::FluidSeedMode::FillLevel ||
                         d.fluid_reseed_on_reset)) {
                        d.fluid_pending_seed = true;
                    }
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
            drainSimulationMutationBackends();
            scene.removeSimulationGridDomain(
                static_cast<std::size_t>(scene.active_particle_system_index),
                static_cast<std::size_t>(selected_domain_index));
            if (ui_ctx.selection.selected.type == SelectableType::SimulationDomain &&
                ui_ctx.selection.selected.particle_system_index == scene.active_particle_system_index &&
                ui_ctx.selection.selected.simulation_domain_index == selected_domain_index) {
                ui_ctx.selection.clearSelection();
            }
            selected_domain_index = std::min(selected_domain_index, static_cast<int>(particles->gridDomains().size()) - 1);
            ImGui::Columns(1);
            return;
        }
}

} // namespace ForceFieldUI
