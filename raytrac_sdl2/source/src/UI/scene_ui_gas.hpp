/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_gas.hpp
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
/**
 * @file scene_ui_gas.hpp
 * @brief UI Panel for Gas/Smoke Simulation
 * 
 * Provides controls for:
 * - Grid resolution and voxel size
 * - Simulation parameters (timestep, dissipation)
 * - Forces (buoyancy, vorticity, wind)
 * - Emitter management
 * - Playback controls
 * - Baking and export
 */

#pragma once

#include "scene_ui.h"
#include "GasVolume.h"
#include <memory>
#include <string>

namespace GasUI {

// Currently selected gas volume for UI (Shared within Volumetric panel)
static std::shared_ptr<GasVolume> selected_gas_volume = nullptr;

/**
 * @brief Draw only the properties of a Gas Volume (no list)
 */
inline void drawGasSimulationProperties(UIContext& ui_ctx, std::shared_ptr<GasVolume> gas) {
    if (!gas) return;
    auto& scene = ui_ctx.scene;
    auto& settings = gas->getSettings();
    
    // Name
    char name_buf[128];
    strncpy_s(name_buf, gas->name.c_str(), sizeof(name_buf) - 1);
    if (ImGui::InputText("Name", name_buf, sizeof(name_buf))) {
        gas->name = name_buf;
    }
    
    // Visibility
    ImGui::Checkbox("Visible", &gas->visible);
    
    ImGui::Spacing();
    
    // ─────────────────────────────────────────────────────────────────────────
    // PLAYBACK CONTROLS
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Playback & Timeline", ImGuiTreeNodeFlags_DefaultOpen)) {
        // ... (rest of playback section)
        // Timeline Integration
        bool linked = gas->isLinkedToTimeline();
        if (ImGui::Checkbox("Link to Timeline", &linked)) {
            gas->setLinkedToTimeline(linked);
        }
        UIWidgets::HelpMarker("When enabled, simulation playback is driven by the main timeline.");
        
        if (linked) {
            int offset = gas->getFrameOffset();
            if (ImGui::DragInt("Frame Offset", &offset)) {
                gas->setFrameOffset(offset);
            }
        }
        
        ImGui::Separator();
        
        // Buttons
        bool is_playing = gas->isPlaying();
        bool is_baking = gas->isBaking();
        
        if (!is_baking) {
            if (is_playing) {
                if (ImGui::Button("Pause", ImVec2(60, 0))) {
                    gas->pause();
                }
            } else {
                if (ImGui::Button("Play", ImVec2(60, 0))) {
                    if (!gas->isInitialized()) {
                        gas->initialize();
                    }
                    gas->play();
                }
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Stop", ImVec2(60, 0))) {
                gas->stop();
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Reset", ImVec2(60, 0))) {
                gas->reset();
                if (ui_ctx.optix_gpu_ptr) ui_ctx.optix_gpu_ptr->resetAccumulation();
                ui_ctx.renderer.resetCPUAccumulation();
            }
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Controls locked during bake...");
        }
        
        // Stats
        ImGui::Text("Frame: %d", gas->getCurrentFrame());
        ImGui::Text("Active Voxels: %d", gas->getActiveVoxelCount());
        ImGui::Text("Max Density: %.2f", gas->getMaxDensity());
        ImGui::Text("Step Time: %.2f ms", gas->getLastStepTime());
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // GRID & DOMAIN SETTINGS
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Domain / Grid", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool needs_reinit = false;
        
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Physical Dimensions (Bounding Box)");
        float grid_size[3] = { settings.grid_size.x, settings.grid_size.y, settings.grid_size.z };
        if (ImGui::DragFloat3("Grid Size", grid_size, 0.1f, 0.1f, 100.0f, "%.1f m")) {
            settings.grid_size = Vec3(grid_size[0], grid_size[1], grid_size[2]);
            needs_reinit = true;
        }
        
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Resolution (Detail)");
        
        int res[3] = { settings.resolution_x, settings.resolution_y, settings.resolution_z };
        if (ImGui::DragInt3("Resolution", res, 1, 8, 1024)) {
            settings.resolution_x = res[0];
            settings.resolution_y = res[1];
            settings.resolution_z = res[2];
            needs_reinit = true;
        }
        
        static bool link_res = true;
        ImGui::Checkbox("Link Resolution Proportions", &link_res);
        if (link_res && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Recommended: Adjusts resolution axes to match Grid Size proportions for square voxels.");
        }

        if (link_res && (ImGui::IsItemDeactivatedAfterEdit() || needs_reinit)) {
            // Adjust resolution axes to match Grid Size proportions for square voxels.
            float max_dim = std::max({settings.grid_size.x, settings.grid_size.y, settings.grid_size.z});
            float master_res = (float)settings.resolution_x; // Use X as master
            
            // Re-calculate based on master resolution and physical proportions
            float ratio_y = settings.grid_size.y / settings.grid_size.x;
            float ratio_z = settings.grid_size.z / settings.grid_size.x;
            settings.resolution_y = std::max(8, (int)(master_res * ratio_y));
            settings.resolution_z = std::max(8, (int)(master_res * ratio_z));
        }

        ImGui::Spacing();
        
        // Detailed Stats
        ImGui::BeginChild("GridStats", ImVec2(0, 65), true);
        ImGui::Columns(2, "gridcols", false);
        ImGui::Text("Voxel Size:"); ImGui::NextColumn();
        float calc_voxel = settings.grid_size.x / (float)settings.resolution_x;
        ImGui::Text("%.4f m (%s)", calc_voxel, (calc_voxel < 0.05f) ? "High Detail" : "Low Detail"); ImGui::NextColumn();
        
        size_t cells = (size_t)settings.resolution_x * settings.resolution_y * settings.resolution_z;
        ImGui::Text("Total Cells:"); ImGui::NextColumn();
        ImGui::Text("%.2f Million", cells / 1000000.0f); ImGui::NextColumn();
        
        float memory_mb = (cells * sizeof(float) * 9) / (1024.0f * 1024.0f); // ~9 buffers
        ImGui::Text("Est. VRAM:"); ImGui::NextColumn();
        ImGui::TextColored(memory_mb > 2000.0f ? ImVec4(1,0.5f,0,1) : ImVec4(1,1,1,1), "%.1f MB", memory_mb);
        ImGui::EndColumns();
        ImGui::EndChild();
        
        if (needs_reinit) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.9f, 0.1f, 0.1f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            
            if (ImGui::Button("!!! RESTART SIMULATION TO APPLY NEW RESOLUTION !!!", ImVec2(-1, 40))) {
                gas->stop();
                gas->initialize();
                gas->play();
                SceneUI::syncVDBVolumesToGPU(ui_ctx); // Ensure OptiX gets new pointers/params
                if (ui_ctx.optix_gpu_ptr) ui_ctx.optix_gpu_ptr->resetAccumulation();
            }
            ImGui::PopStyleVar();
            ImGui::PopStyleColor(2);
            
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Changes to Grid Size or Resolution require re-allocating memory and restarting the solver.");
            
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Warning: Restarting will clear all current simulation data.");
        }
        
        // Sync Logic: If using CUDA and this tab is open/stats are visible, download data for UI feedback
        // Otherwise counters will be 0 because grid.density is empty on CPU.
        if (settings.backend == FluidSim::SolverBackend::CUDA && gas->isInitialized()) {
             // Only do this expensive download if we really want to see the numbers
             // Or maybe just show "GPU Mode" instead of 0? 
             // Let's force download for now so user trusts it works.
             // But do it only once per few frames to save performance? 
             // For now, raw download.
             gas->getSimulator().downloadFromGPU();
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // SIMULATION PARAMETERS
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Simulation")) {
        ImGui::DragFloat("Timestep", &settings.timestep, 0.001f, 0.001f, 0.1f, "%.4f");
        ImGui::DragInt("Substeps", &settings.substeps, 1, 1, 10);
        ImGui::DragInt("Pressure Iterations", &settings.pressure_iterations, 1, 10, 100);
        
        ImGui::Spacing();
        ImGui::Text("Dissipation");
        ImGui::DragFloat("Density", &settings.density_dissipation, 0.001f, 0.9f, 1.0f, "%.4f");
        ImGui::DragFloat("Velocity", &settings.velocity_dissipation, 0.001f, 0.9f, 1.0f, "%.4f");
        ImGui::DragFloat("Temperature", &settings.temperature_dissipation, 0.001f, 0.9f, 1.0f, "%.4f");
        ImGui::DragFloat("Fuel", &settings.fuel_dissipation, 0.001f, 0.9f, 1.0f, "%.4f");
        
        ImGui::Spacing();
        
        // Backend selection
        const char* backends[] = { "CPU", "CUDA" };
        int current_backend = static_cast<int>(settings.backend);
        if (ImGui::Combo("Backend", &current_backend, backends, 2)) {
            settings.backend = static_cast<FluidSim::SolverBackend>(current_backend);
        }
        
        // Mode selection
        const char* modes[] = { "Real-time", "Baked" };
        int current_mode = static_cast<int>(settings.mode);
        if (ImGui::Combo("Mode", &current_mode, modes, 2)) {
            settings.mode = static_cast<FluidSim::SimulationMode>(current_mode);
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // COMBUSTION (Fire, Flame)
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Combustion (Fire)")) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Flame / Fire Parameters");
        
        ImGui::DragFloat("Ignition Temperature", &settings.ignition_temperature, 10.0f, 200.0f, 1000.0f, "%.0f K");
        UIWidgets::HelpMarker("Fuel will start burning when temperature exceeds this value.");
        
        ImGui::DragFloat("Burn Rate", &settings.burn_rate, 0.1f, 0.0f, 10.0f);
        UIWidgets::HelpMarker("How fast fuel is consumed when burning.");
        
        ImGui::DragFloat("Heat Release", &settings.heat_release, 10.0f, 0.0f, 2000.0f);
        UIWidgets::HelpMarker("Temperature increase per unit of burned fuel.");
        
        ImGui::DragFloat("Smoke Generation", &settings.smoke_generation, 0.1f, 0.0f, 5.0f);
        UIWidgets::HelpMarker("Density (smoke) generated per unit of burned fuel.");
        
        ImGui::DragFloat("Expansion", &settings.expansion_strength, 1.0f, 0.0f, 50.0f);
        UIWidgets::HelpMarker("Outward pressure force from combustion (explosion effect).");
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // PRESETS
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.6f, 1.0f), "Quick Presets");
        
        if (ImGui::Button("Fire", ImVec2(80, 0))) {
            gas->getSimulator().applyPreset("Fire");
        }
        ImGui::SameLine();
        if (ImGui::Button("Smoke", ImVec2(80, 0))) {
            gas->getSimulator().applyPreset("Smoke");
        }
        ImGui::SameLine();
        if (ImGui::Button("Explosion", ImVec2(80, 0))) {
            gas->getSimulator().applyPreset("Explosion");
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // FORCES
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Forces", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Internal Physics");
        ImGui::DragFloat("Buoyancy Density", &settings.buoyancy_density, 0.01f, -2.0f, 2.0f);
        ImGui::DragFloat("Heat Rise (Buoyancy)", &settings.buoyancy_temperature, 0.01f, 0.0f, 5.0f);
        ImGui::DragFloat("Ambient Temp (K)", &settings.ambient_temperature, 1.0f, 200.0f, 400.0f);
        
        ImGui::Spacing();
        ImGui::DragFloat("Turbulence (Vorticity)", &settings.vorticity_strength, 0.01f, 0.0f, 2.0f);
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "External Forces");
        ImGui::TextWrapped("Gas simulation is now automatically affected by global Force Fields (Wind, Vortex, Noise, etc.).");
        
        if (ImGui::Button("Manage Force Fields", ImVec2(-1, 0))) {
           // ui_ctx.tab_to_focus = "Force Fields";
        }
        
        // Hide legacy local wind to prevent confusion, but keep it in data for compat
        if (ImGui::TreeNode("Legacy Local Wind (Deprecated)")) {
            float wind[3] = { settings.wind.x, settings.wind.y, settings.wind.z };
            if (ImGui::DragFloat3("Local Dir", wind, 0.1f, -10.0f, 10.0f)) {
                settings.wind = Vec3(wind[0], wind[1], wind[2]);
            }
            ImGui::TreePop();
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // EMITTERS
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Emitters", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto& emitters = gas->getEmitters();
        
        if (ImGui::Button("+ Add Emitter")) {
            FluidSim::Emitter e;
            e.name = "Emitter " + std::to_string(emitters.size() + 1);
            e.position = Vec3(
                settings.resolution_x * settings.voxel_size * 0.5f,
                settings.voxel_size * 2.0f,
                settings.resolution_z * settings.voxel_size * 0.5f
            );
            gas->addEmitter(e);
        }
        
        static int selected_emitter = 0;
        
        // Keyframe Helper for Emitters
        auto KeyframeButton = [&](const char* id, bool keyed) -> bool {
            ImGui::PushID(id);
            float s = ImGui::GetFrameHeight();
            ImVec2 pos = ImGui::GetCursorScreenPos();
            bool clicked = ImGui::InvisibleButton("kbtn", ImVec2(s, s));

            ImU32 bg = keyed ? IM_COL32(255, 200, 0, 255) : IM_COL32(40, 40, 40, 255);
            ImU32 border = IM_COL32(180, 180, 180, 255);

            if (ImGui::IsItemHovered()) {
                border = IM_COL32(255, 255, 255, 255);
                bg = keyed ? IM_COL32(255, 220, 50, 255) : IM_COL32(70, 70, 70, 255);
            }

            ImDrawList* dl = ImGui::GetWindowDrawList();
            float cx = pos.x + s * 0.5f;
            float cy = pos.y + s * 0.5f;
            float r = s * 0.22f;

            ImVec2 p[4] = {
                ImVec2(cx, cy - r),
                ImVec2(cx + r, cy),
                ImVec2(cx, cy + r),
                ImVec2(cx - r, cy)
            };

            dl->AddQuadFilled(p[0], p[1], p[2], p[3], bg);
            dl->AddQuad(p[0], p[1], p[2], p[3], border, 1.0f);

            ImGui::PopID();
            return clicked;
        };

        auto insertEmitterKey = [&](const std::string& track_name, FluidSim::Emitter& e,
                                  bool key_fuel, bool key_density, bool key_temp, bool key_pos, bool key_vel, bool key_enabled, bool key_size, bool key_radius) {
            int current_frame = ui_ctx.render_settings.animation_current_frame;
            auto& track = ui_ctx.scene.timeline.tracks[track_name];

            // Toggle Behavior
            for (auto it = track.keyframes.begin(); it != track.keyframes.end(); ++it) {
                if (it->frame == current_frame && it->has_emitter) {
                    bool removed = false;
                    if (key_fuel && it->emitter.has_fuel_rate) { it->emitter.has_fuel_rate = false; removed = true; }
                    if (key_density && it->emitter.has_density_rate) { it->emitter.has_density_rate = false; removed = true; }
                    if (key_temp && it->emitter.has_temperature) { it->emitter.has_temperature = false; removed = true; }
                    if (key_pos && it->emitter.has_position) { it->emitter.has_position = false; removed = true; }
                    if (key_vel && it->emitter.has_velocity) { it->emitter.has_velocity = false; removed = true; }
                    if (key_enabled && it->emitter.has_enabled) { it->emitter.has_enabled = false; removed = true; }
                    if (key_size && it->emitter.has_size) { it->emitter.has_size = false; removed = true; }
                    if (key_radius && it->emitter.has_radius) { it->emitter.has_radius = false; removed = true; }

                    if (removed) {
                        bool hasAny = it->emitter.has_fuel_rate || it->emitter.has_density_rate || it->emitter.has_temperature ||
                                     it->emitter.has_position || it->emitter.has_velocity || it->emitter.has_enabled ||
                                     it->emitter.has_size || it->emitter.has_radius;
                        if (!hasAny) {
                            it->has_emitter = false;
                            if (!it->has_transform && !it->has_light && !it->has_camera && !it->has_world) {
                                track.keyframes.erase(it);
                            }
                        }
                        return;
                    }
                }
            }

            // Add New Keyframe
            Keyframe kf(current_frame);
            kf.has_emitter = true;
            kf.emitter.has_fuel_rate = key_fuel;
            kf.emitter.has_density_rate = key_density;
            kf.emitter.has_temperature = key_temp;
            kf.emitter.has_position = key_pos;
            kf.emitter.has_velocity = key_vel;
            kf.emitter.has_enabled = key_enabled;
            kf.emitter.has_size = key_size;
            kf.emitter.has_radius = key_radius;

            kf.emitter.fuel_rate = e.fuel_rate;
            kf.emitter.density_rate = e.density_rate;
            kf.emitter.temperature = e.temperature;
            kf.emitter.position = e.position;
            kf.emitter.velocity = e.velocity;
            kf.emitter.enabled = e.enabled;
            kf.emitter.size = e.size;
            kf.emitter.radius = e.radius;

            track.addKeyframe(kf);
        };

        auto isEmitterPropertyKeyed = [&](const std::string& track_name, bool f, bool d, bool t, bool p, bool v, bool e, bool s, bool r) {
            auto& tracks = ui_ctx.scene.timeline.tracks;
            if (tracks.find(track_name) == tracks.end()) return false;
            int cf = ui_ctx.render_settings.animation_current_frame;
            for (auto& kf : tracks[track_name].keyframes) {
                if (kf.frame == cf && kf.has_emitter) {
                    if (f && kf.emitter.has_fuel_rate) return true;
                    if (d && kf.emitter.has_density_rate) return true;
                    if (t && kf.emitter.has_temperature) return true;
                    if (p && kf.emitter.has_position) return true;
                    if (v && kf.emitter.has_velocity) return true;
                    if (e && kf.emitter.has_enabled) return true;
                    if (s && kf.emitter.has_size) return true;
                    if (r && kf.emitter.has_radius) return true;
                }
            }
            return false;
        };

        for (int i = 0; i < static_cast<int>(emitters.size()); ++i) {
            auto& e = emitters[i];
            // Use UID for stable track naming that persists across index shifts
            std::string track_name = gas->getName() + "_" + e.name + "_" + std::to_string(e.uid);
            
            ImGui::PushID(i);
            
            bool open = ImGui::TreeNode((e.name + "##emitter").c_str());
            
            // Delete button
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 20);
            if (ImGui::SmallButton("X")) {
                // CLEAN UP TRACK BEFORE REMOVAL
                if (ui_ctx.scene.timeline.tracks.count(track_name)) {
                    ui_ctx.scene.timeline.tracks.erase(track_name);
                }
                gas->removeEmitter(i);
                ImGui::PopID();
                if (open) ImGui::TreePop();
                break;
            }
            
            if (open) {
                // Name
                char ename[64];
                strncpy_s(ename, e.name.c_str(), sizeof(ename) - 1);
                if (ImGui::InputText("Name##e", ename, sizeof(ename))) {
                    e.name = ename;
                }
                
                // Enabled with Key
                bool endKeyed = isEmitterPropertyKeyed(track_name, false, false, false, false, false, true, false, false);
                if (KeyframeButton("##KEn", endKeyed)) { insertEmitterKey(track_name, e, false, false, false, false, false, true, false, false); }
                ImGui::SameLine();
                ImGui::Checkbox("Enabled", &e.enabled);
                
                // Shape
                const char* shapes[] = { "Sphere", "Box", "Point" };
                int shape = static_cast<int>(e.shape);
                if (ImGui::Combo("Shape", &shape, shapes, 3)) {
                    e.shape = static_cast<FluidSim::EmitterShape>(shape);
                }
                
                // Position with Key
                bool posKeyed = isEmitterPropertyKeyed(track_name, false, false, false, true, false, false, false, false);
                if (KeyframeButton("##KPos", posKeyed)) { insertEmitterKey(track_name, e, false, false, false, true, false, false, false, false); }
                ImGui::SameLine();
                float pos[3] = { e.position.x, e.position.y, e.position.z };
                if (ImGui::DragFloat3("Position", pos, 0.1f)) {
                    e.position = Vec3(pos[0], pos[1], pos[2]);
                }
                
                // Size with Keyframe support
                if (e.shape == FluidSim::EmitterShape::Sphere) {
                    bool radKeyed = isEmitterPropertyKeyed(track_name, false, false, false, false, false, false, false, true);
                    if (KeyframeButton("##KRad", radKeyed)) { insertEmitterKey(track_name, e, false, false, false, false, false, false, false, true); }
                    ImGui::SameLine();
                    ImGui::DragFloat("Radius", &e.radius, 0.05f, 0.1f, 10.0f);
                } else if (e.shape == FluidSim::EmitterShape::Box) {
                    bool sizKeyed = isEmitterPropertyKeyed(track_name, false, false, false, false, false, false, true, false);
                    if (KeyframeButton("##KSiz", sizKeyed)) { insertEmitterKey(track_name, e, false, false, false, false, false, false, true, false); }
                    ImGui::SameLine();
                    float size[3] = { e.size.x, e.size.y, e.size.z };
                    if (ImGui::DragFloat3("Size", size, 0.1f, 0.1f, 10.0f)) {
                        e.size = Vec3(size[0], size[1], size[2]);
                    }
                }
                
                // Emission: Density
                bool denKeyed = isEmitterPropertyKeyed(track_name, false, true, false, false, false, false, false, false);
                if (KeyframeButton("##KDen", denKeyed)) { insertEmitterKey(track_name, e, false, true, false, false, false, false, false, false); }
                ImGui::SameLine();
                ImGui::DragFloat("Density Rate", &e.density_rate, 1.0f, 0.0f, 1000.0f);
                
                // Emission: Fuel
                bool fueKeyed = isEmitterPropertyKeyed(track_name, true, false, false, false, false, false, false, false);
                if (KeyframeButton("##KFue", fueKeyed)) { insertEmitterKey(track_name, e, true, false, false, false, false, false, false, false); }
                ImGui::SameLine();
                ImGui::DragFloat("Fuel Rate", &e.fuel_rate, 1.0f, 0.0f, 1000.0f);
                UIWidgets::HelpMarker("Fuel injection rate. Fuel burns when hot enough, producing flames.");
                
                // Emission: Temperature
                bool tmpKeyed = isEmitterPropertyKeyed(track_name, false, false, true, false, false, false, false, false);
                if (KeyframeButton("##KTmp", tmpKeyed)) { insertEmitterKey(track_name, e, false, false, true, false, false, false, false, false); }
                ImGui::SameLine();
                ImGui::DragFloat("Temperature (K)", &e.temperature, 10.0f, 300.0f, 5000.0f);
                
                // Initial velocity with Key
                bool velKeyed = isEmitterPropertyKeyed(track_name, false, false, false, false, true, false, false, false);
                if (KeyframeButton("##KVel", velKeyed)) { insertEmitterKey(track_name, e, false, false, false, false, true, false, false, false); }
                ImGui::SameLine();
                float v[3] = { e.velocity.x, e.velocity.y, e.velocity.z };
                if (ImGui::DragFloat3("Velocity", v, 0.1f, -100.0f, 100.0f)) {
                    e.velocity = Vec3(v[0], v[1], v[2]);
                }
                
                ImGui::TreePop();
            }
            
            ImGui::PopID();
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // EXPORT & BAKING
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Export & Baking")) {
        static char export_dir[256] = "";
        static bool export_success = false;
        static bool export_error = false;
        static std::string export_message;
        
        ImGui::Text("Bake / Output Directory:");
        ImGui::InputText("##dir", export_dir, sizeof(export_dir));
        ImGui::SameLine();
        if (ImGui::Button("Browse Folder")) {
            std::string path = SceneUI::selectFolderDialogW(L"Select Export/Bake Directory");
            if (!path.empty()) {
                strncpy_s(export_dir, path.c_str(), sizeof(export_dir) - 1);
            }
        }
        
        ImGui::Separator();

        // Single Frame Export
        if (ImGui::Button("Export Current Frame (.vdb)", ImVec2(-1, 0))) {
            if (strlen(export_dir) == 0) {
                export_error = true;
                export_message = "Please specify a directory first";
            } else {
                std::string full_path = std::string(export_dir) + "/frame_" + std::to_string(gas->getCurrentFrame()) + ".vdb";
                bool result = gas->exportToVDB(full_path);
                export_success = result;
                export_error = !result;
                export_message = result ? ("Saved: " + std::string(export_dir)) : "Export failed!";
            }
        }

        ImGui::Spacing();
        
        // Baking
        UIWidgets::ColoredHeader("Sequence Baking", ImVec4(1.0f, 0.6f, 0.4f, 1.0f));
        static int bake_start = 0, bake_end = 100;
        ImGui::DragInt("Start Frame", &bake_start, 1, 0, 1000);
        ImGui::DragInt("End Frame", &bake_end, 1, 1, 1000);
        
        if (gas->isBaking()) {
            float progress = gas->getBakeProgress();
            int current_bake_frame = gas->getSimulator().getBakingFrame();
            
            std::string progress_text = "Baking Frame: " + std::to_string(current_bake_frame) + " (" + std::to_string((int)(progress * 100)) + "%)";
            ImGui::ProgressBar(progress, ImVec2(-1, 0), progress_text.c_str());
            
            if (ImGui::Button("Cancel Bake", ImVec2(-1, 0))) {
                gas->cancelBake();
            }
        } else {
            if (ImGui::Button("Start Bake Sequence", ImVec2(-1, 0))) {
                if (strlen(export_dir) == 0) {
                    export_error = true;
                    export_message = "Specify directory first!";
                } else {
                    // SYNC KEYFRAMES FROM TIMELINE TO EMITTERS BEFORE BAKE
                    for (auto& e : gas->getEmitters()) {
                        e.keyframes.clear();
                        std::string track_name = gas->getName() + "_" + e.name + "_" + std::to_string(e.uid);
                        if (ui_ctx.scene.timeline.tracks.count(track_name)) {
                            auto& track = ui_ctx.scene.timeline.tracks[track_name];
                            for (const auto& kf_timeline : track.keyframes) {
                                if (kf_timeline.has_emitter) {
                                    // Copy the whole emitter keyframe data
                                    e.keyframes[kf_timeline.frame] = kf_timeline.emitter;
                                    
                                    // Ensure flags are preserved
                                    e.keyframes[kf_timeline.frame].has_enabled = kf_timeline.emitter.has_enabled;
                                    e.keyframes[kf_timeline.frame].has_density_rate = kf_timeline.emitter.has_density_rate;
                                    e.keyframes[kf_timeline.frame].has_radius = kf_timeline.emitter.has_radius;
                                    e.keyframes[kf_timeline.frame].has_position = kf_timeline.emitter.has_position;
                                }
                            }
                        }
                    }

                    // Start baking with the specified directory
                    gas->exportSequence(export_dir, bake_start, bake_end);
                }
            }
        }// Feedback Text
        if (export_success) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "%s", export_message.c_str());
        } else if (export_error) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", export_message.c_str());
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // SHADER / RENDERING (UNIFIED VOLUMETRIC SHADER)
    // ─────────────────────────────────────────────────────────────────────────
    // Raymarch Quality is very relevant for Gas Simulation when using the Unified VDB path!
    auto shader = gas->getOrCreateShader();
    if (SceneUI::drawVolumeShaderUI(ui_ctx, shader, nullptr, gas.get())) {
        if (gas->render_path == GasVolume::VolumeRenderPath::VDBUnified) {
            SceneUI::syncVDBVolumesToGPU(ui_ctx); // Unified path sync
        } else {
            ui_ctx.renderer.updateOptiXGasVolumes(scene, ui_ctx.optix_gpu_ptr); // Legacy path sync
        }
        if (ui_ctx.optix_gpu_ptr) ui_ctx.optix_gpu_ptr->resetAccumulation();
        ui_ctx.renderer.resetCPUAccumulation();
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // TRANSFORM
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Transform")) {
        Vec3 pos = gas->getPosition();
        float p[3] = { pos.x, pos.y, pos.z };
        if (ImGui::DragFloat3("Position##transform", p, 0.1f)) {
            gas->setPosition(Vec3(p[0], p[1], p[2]));
            if (gas->render_path == GasVolume::VolumeRenderPath::VDBUnified) SceneUI::syncVDBVolumesToGPU(ui_ctx);
        }
        
        Vec3 rot = gas->getRotation();
        float r[3] = { rot.x, rot.y, rot.z };
        if (ImGui::DragFloat3("Rotation", r, 1.0f, -180.0f, 180.0f)) {
            gas->setRotation(Vec3(r[0], r[1], r[2]));
            if (gas->render_path == GasVolume::VolumeRenderPath::VDBUnified) SceneUI::syncVDBVolumesToGPU(ui_ctx);
        }
        
        Vec3 sc = gas->getScale();
        float s[3] = { sc.x, sc.y, sc.z };
        if (ImGui::DragFloat3("Scale", s, 0.1f, 0.1f, 10.0f)) {
            gas->setScale(Vec3(s[0], s[1], s[2]));
            if (gas->render_path == GasVolume::VolumeRenderPath::VDBUnified) SceneUI::syncVDBVolumesToGPU(ui_ctx);
        }
    }
}

} // namespace GasUI

