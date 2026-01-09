// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - RIVER PANEL
// ═══════════════════════════════════════════════════════════════════════════════
// UI for creating and editing river splines
// ═══════════════════════════════════════════════════════════════════════════════
#ifndef SCENE_UI_RIVER_HPP
#define SCENE_UI_RIVER_HPP

#include "scene_ui.h"
#include "imgui.h"
#include "RiverSpline.h"
#include "TerrainManager.h"
#include "ProjectManager.h"
#include "WaterSystem.h"

// ═══════════════════════════════════════════════════════════════════════════════
// RIVER PANEL IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════
inline void SceneUI::drawRiverPanel(UIContext& ctx) {
    auto& riverMgr = RiverManager::getInstance();
    
    // Track if river panel is currently visible/active
    static bool s_riverPanelActive = false;
    s_riverPanelActive = true;
    riverMgr.lastActiveFrame = ImGui::GetFrameCount();
    
    // ════════════════════════════════════════════════════════════════════════════════
    // PROCEDURAL OCEAN GENERATOR (High-Quality Terrain-Like Water)
    // ════════════════════════════════════════════════════════════════════════════════

    if (ImGui::CollapsingHeader("Procedural Ocean Generator", ImGuiTreeNodeFlags_DefaultOpen)) {
        static float new_ocean_size = 500.0f;
        static int new_ocean_res = 256;      // Explicit Resolution (Grid subdivisions)
        static float new_ocean_y = 0.0f;
        
        // Generative Params
        static int gen_noise_type = 2;       // Default: Ridge
        static float gen_wave_height = 5.0f;
        static float gen_wave_scale = 80.0f;
        
        ImGui::Text("Grid & Resolution");
        ImGui::InputFloat("Size (m)", &new_ocean_size);
        ImGui::InputFloat("Height (Y)", &new_ocean_y);
        
        // Resolution Slider limited to sensible values (32..2048)
        // 1024x1024 = 1M quads = 2M tris. Beware performance.
        ImGui::SliderInt("Resolution (Subdivs)", &new_ocean_res, 32, 1024);
        
        int total_tris = new_ocean_res * new_ocean_res * 2;
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Total Triangles: %d", total_tris);
        
        ImGui::Separator();
        ImGui::Text("Initial Geometry Pattern");
        
        const char* noiseTypeNames[] = { "Perlin", "FBM", "Ridge", "Voronoi (Fake)", "Billow" };
        ImGui::Combo("Pattern Type", &gen_noise_type, noiseTypeNames, IM_ARRAYSIZE(noiseTypeNames));
        
        if (SceneUI::DrawSmartFloat("wamp", "Wave Amplitude", &gen_wave_height, 0.1f, 50.0f, "%.1f", false, nullptr, 16)) {}
        if (SceneUI::DrawSmartFloat("wscl", "Wave Scale", &gen_wave_scale, 1.0f, 500.0f, "%.1f", false, nullptr, 16)) {}
        
        if (ImGui::Button("Generate Sculpted Ocean Mesh", ImVec2(-1, 0))) {
            // Calculate density: density = segments / size
            if (new_ocean_size < 1.0f) new_ocean_size = 1.0f;
            float density = (float)new_ocean_res / new_ocean_size;
            
            WaterSurface* surf = WaterManager::getInstance().createWaterPlane(ctx.scene, Vec3(0, new_ocean_y, 0), new_ocean_size, density);
            
            if (surf) {
                 // Apply Procedural Noise Settings
                 surf->params.use_geometric_waves = true;
                 surf->params.geo_wave_height = gen_wave_height;
                 surf->params.geo_wave_scale = gen_wave_scale;
                 surf->params.geo_noise_type = (WaterWaveParams::NoiseType)gen_noise_type;
                 
                 // Default detailing depending on type
                 surf->params.geo_octaves = 5;
                 surf->params.geo_lacunarity = 2.0f;
                 surf->params.geo_persistence = 0.5f;

                 // Enable FFT for micro-details as well (Best of both worlds)
                 surf->params.use_fft_ocean = true;
                 surf->params.fft_ocean_size = gen_wave_scale * 0.5f; 
                 
                 // Deform the flat mesh
                 WaterManager::getInstance().updateWaterMesh(surf);
                 
                 ctx.renderer.resetCPUAccumulation();
                 extern bool g_bvh_rebuild_pending;
                 extern bool g_optix_rebuild_pending;
                 g_bvh_rebuild_pending = true;
                 g_optix_rebuild_pending = true; 
            }
        }
        ImGui::Separator();
        
        // List Existing Oceans
        auto& surfaces = WaterManager::getInstance().getWaterSurfaces();
        static int selected_surf_id = -1;
        
        if (surfaces.empty()) {
            ImGui::TextDisabled("No ocean surfaces created.");
        } else {
            ImGui::Text("Active Surfaces:");
            for (auto& surf : surfaces) {
                if (surf.type == WaterSurface::Type::River) continue; // Skip rivers (handled below)
                
                std::string label = surf.name + "##" + std::to_string(surf.id);
                bool is_selected = (surf.id == selected_surf_id);
                if (ImGui::Selectable(label.c_str(), is_selected)) {
                    selected_surf_id = surf.id;
                }
            }
        }
        
        // Selected Surface Editor
        if (selected_surf_id != -1) {
            WaterSurface* selSurf = WaterManager::getInstance().getWaterSurface(selected_surf_id);
            if (selSurf) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Properties: %s", selSurf->name.c_str());
                
                // Push unique ID for this surface to avoid widget ID conflicts
                ImGui::PushID(selSurf->id);
                
                // Auto-select water track in timeline for keyframe manipulation
                std::string waterTrackName = "Water_" + std::to_string(selSurf->id);
                timeline.selected_track = waterTrackName;
                
                bool changed = false;
                WaterWaveParams& wp = selSurf->params;
                
                // ═══════════════════════════════════════════════════════════
                // KEYFRAME SYSTEM FOR WATER
                // ═══════════════════════════════════════════════════════════
                
                // Keyframe diamond button
                auto KeyframeButton = [&](const char* id, bool keyed, const char* prop_name = nullptr) -> bool {
                    ImGui::PushID(id);
                    float s = ImGui::GetFrameHeight();
                    ImVec2 pos = ImGui::GetCursorScreenPos();
                    bool clicked = ImGui::InvisibleButton("kbtn", ImVec2(s, s));
                    
                    ImU32 bg = keyed ? IM_COL32(100, 180, 255, 255) : IM_COL32(40, 40, 40, 255);
                    ImU32 border = IM_COL32(180, 180, 180, 255);
                    
                    bool hovered = ImGui::IsItemHovered();
                    if (hovered) {
                        border = IM_COL32(255, 255, 255, 255);
                        bg = keyed ? IM_COL32(120, 200, 255, 255) : IM_COL32(70, 70, 70, 255);
                        
                        if (prop_name) {
                            ImGui::SetTooltip(keyed ? "%s: Click to REMOVE keyframe" : "%s: Click to ADD keyframe", prop_name);
                        } else {
                            ImGui::SetTooltip(keyed ? "Click to REMOVE keyframe" : "Click to ADD keyframe");
                        }
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
                
                // Water property enum
                enum class WaterProp {
                    WaveHeight, WaveScale, WindDirection, Choppiness,
                    Alignment, Damping, SwellAmplitude, Sharpening, DetailStrength
                };
                
                // Check if water property is keyed at current frame
                auto isWaterKeyed = [&](WaterProp prop) -> bool {
                    auto it = ctx.scene.timeline.tracks.find(waterTrackName);
                    if (it == ctx.scene.timeline.tracks.end()) return false;
                    int cf = ctx.render_settings.animation_current_frame;
                    for (auto& kf : it->second.keyframes) {
                        if (kf.frame == cf && kf.has_water) {
                            switch(prop) {
                                case WaterProp::WaveHeight: return kf.water.has_wave_height;
                                case WaterProp::WaveScale: return kf.water.has_wave_scale;
                                case WaterProp::WindDirection: return kf.water.has_wind_direction;
                                case WaterProp::Choppiness: return kf.water.has_choppiness;
                                case WaterProp::Alignment: return kf.water.has_alignment;
                                case WaterProp::Damping: return kf.water.has_damping;
                                case WaterProp::SwellAmplitude: return kf.water.has_swell_amplitude;
                                case WaterProp::Sharpening: return kf.water.has_sharpening;
                                case WaterProp::DetailStrength: return kf.water.has_detail_strength;
                            }
                        }
                    }
                    return false;
                };
                
                // Insert/Toggle water keyframe
                auto insertWaterKey = [&](const std::string& label, WaterProp prop) {
                    int cf = ctx.render_settings.animation_current_frame;
                    auto& track = ctx.scene.timeline.tracks[waterTrackName];
                    
                    // Helper to check if prop is keyed
                    auto isPropKeyed = [](const Keyframe& kf, WaterProp p) -> bool {
                        if (!kf.has_water) return false;
                        switch(p) {
                            case WaterProp::WaveHeight: return kf.water.has_wave_height;
                            case WaterProp::WaveScale: return kf.water.has_wave_scale;
                            case WaterProp::WindDirection: return kf.water.has_wind_direction;
                            case WaterProp::Choppiness: return kf.water.has_choppiness;
                            case WaterProp::Alignment: return kf.water.has_alignment;
                            case WaterProp::Damping: return kf.water.has_damping;
                            case WaterProp::SwellAmplitude: return kf.water.has_swell_amplitude;
                            case WaterProp::Sharpening: return kf.water.has_sharpening;
                            case WaterProp::DetailStrength: return kf.water.has_detail_strength;
                        }
                        return false;
                    };
                    
                    // Helper to clear prop
                    auto clearProp = [](Keyframe& kf, WaterProp p) {
                        switch(p) {
                            case WaterProp::WaveHeight: kf.water.has_wave_height = false; break;
                            case WaterProp::WaveScale: kf.water.has_wave_scale = false; break;
                            case WaterProp::WindDirection: kf.water.has_wind_direction = false; break;
                            case WaterProp::Choppiness: kf.water.has_choppiness = false; break;
                            case WaterProp::Alignment: kf.water.has_alignment = false; break;
                            case WaterProp::Damping: kf.water.has_damping = false; break;
                            case WaterProp::SwellAmplitude: kf.water.has_swell_amplitude = false; break;
                            case WaterProp::Sharpening: kf.water.has_sharpening = false; break;
                            case WaterProp::DetailStrength: kf.water.has_detail_strength = false; break;
                        }
                    };
                    
                    // Find existing keyframe
                    for (auto it = track.keyframes.begin(); it != track.keyframes.end(); ++it) {
                        if (it->frame == cf) {
                            // Toggle: if keyed, remove
                            if (isPropKeyed(*it, prop)) {
                                clearProp(*it, prop);
                                
                                // Check if keyframe is empty
                                WaterKeyframe& wk = it->water;
                                bool hasAnyProp = wk.has_wave_height || wk.has_wave_scale || 
                                                  wk.has_wind_direction || wk.has_choppiness || 
                                                  wk.has_alignment || wk.has_damping ||
                                                  wk.has_swell_amplitude || wk.has_sharpening || 
                                                  wk.has_detail_strength;
                                
                                if (!hasAnyProp) {
                                    it->has_water = false;
                                    if (!it->has_transform && !it->has_camera && !it->has_light && 
                                        !it->has_material && !it->has_world && !it->has_terrain) {
                                        track.keyframes.erase(it);
                                    }
                                }
                                return;
                            }
                            
                            // Add prop to existing keyframe
                            it->has_water = true;
                            it->water.water_surface_id = selSurf->id;
                            
                            switch(prop) {
                                case WaterProp::WaveHeight:
                                    it->water.has_wave_height = true;
                                    it->water.wave_height = wp.geo_wave_height;
                                    break;
                                case WaterProp::WaveScale:
                                    it->water.has_wave_scale = true;
                                    it->water.wave_scale = wp.geo_wave_scale;
                                    break;
                                case WaterProp::WindDirection:
                                    it->water.has_wind_direction = true;
                                    it->water.wind_direction = wp.geo_swell_direction;
                                    break;
                                case WaterProp::Choppiness:
                                    it->water.has_choppiness = true;
                                    it->water.choppiness = wp.geo_wave_choppiness;
                                    break;
                                case WaterProp::Alignment:
                                    it->water.has_alignment = true;
                                    it->water.alignment = wp.geo_alignment;
                                    break;
                                case WaterProp::Damping:
                                    it->water.has_damping = true;
                                    it->water.damping = wp.geo_damping;
                                    break;
                                case WaterProp::SwellAmplitude:
                                    it->water.has_swell_amplitude = true;
                                    it->water.swell_amplitude = wp.geo_swell_amplitude;
                                    break;
                                case WaterProp::Sharpening:
                                    it->water.has_sharpening = true;
                                    it->water.sharpening = wp.geo_sharpening;
                                    break;
                                case WaterProp::DetailStrength:
                                    it->water.has_detail_strength = true;
                                    it->water.detail_strength = wp.geo_detail_strength;
                                    break;
                            }
                            return;
                        }
                    }
                    
                    // Create new keyframe
                    Keyframe kf(cf);
                    kf.has_water = true;
                    kf.water.water_surface_id = selSurf->id;
                    
                    switch(prop) {
                        case WaterProp::WaveHeight:
                            kf.water.has_wave_height = true;
                            kf.water.wave_height = wp.geo_wave_height;
                            break;
                        case WaterProp::WaveScale:
                            kf.water.has_wave_scale = true;
                            kf.water.wave_scale = wp.geo_wave_scale;
                            break;
                        case WaterProp::WindDirection:
                            kf.water.has_wind_direction = true;
                            kf.water.wind_direction = wp.geo_swell_direction;
                            break;
                        case WaterProp::Choppiness:
                            kf.water.has_choppiness = true;
                            kf.water.choppiness = wp.geo_wave_choppiness;
                            break;
                        case WaterProp::Alignment:
                            kf.water.has_alignment = true;
                            kf.water.alignment = wp.geo_alignment;
                            break;
                        case WaterProp::Damping:
                            kf.water.has_damping = true;
                            kf.water.damping = wp.geo_damping;
                            break;
                        case WaterProp::SwellAmplitude:
                            kf.water.has_swell_amplitude = true;
                            kf.water.swell_amplitude = wp.geo_swell_amplitude;
                            break;
                        case WaterProp::Sharpening:
                            kf.water.has_sharpening = true;
                            kf.water.sharpening = wp.geo_sharpening;
                            break;
                        case WaterProp::DetailStrength:
                            kf.water.has_detail_strength = true;
                            kf.water.detail_strength = wp.geo_detail_strength;
                            break;
                    }
                    
                ctx.scene.timeline.insertKeyframe(waterTrackName, kf);
                };
                
                // Helper to apply preset with optional mesh rebuild
                auto applyPreset = [&](float height, float scale, float dir, float chop, float align, float damp, float swell, float sharp, float detail, WaterWaveParams::NoiseType type) {
                    wp.geo_wave_height = height;
                    wp.geo_wave_scale = scale;
                    wp.geo_swell_direction = dir;
                    wp.geo_wave_choppiness = chop;
                    wp.geo_alignment = align;
                    wp.geo_damping = damp;
                    wp.geo_swell_amplitude = swell;
                    wp.geo_sharpening = sharp;
                    wp.geo_detail_strength = detail;
                    wp.geo_noise_type = type;
                    wp.use_geometric_waves = true;
                    WaterManager::getInstance().updateWaterMesh(selSurf);
                    ctx.renderer.resetCPUAccumulation();
                    extern bool g_bvh_rebuild_pending;
                    extern bool g_optix_rebuild_pending;
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                };
                
                bool rebuild_mesh = false;
                
                // ═══════════════════════════════════════════════════════════
                // 🌊 WAVE GENERATION (Main Section - Always Visible)
                // ═══════════════════════════════════════════════════════════
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Wave Generation");
                ImGui::Separator();
                
                // Wave Type Selection
                const char* waveTypeNames[] = { "Perlin Noise", "FBM Noise", "Ridge Noise", "Voronoi", "Billow", "Gerstner Waves", "Ocean Spectrum" };
                int currentType = (int)wp.geo_noise_type;
                if (ImGui::Combo("Wave Type", &currentType, waveTypeNames, IM_ARRAYSIZE(waveTypeNames))) {
                    wp.geo_noise_type = (WaterWaveParams::NoiseType)currentType;
                    rebuild_mesh = true;
                }
                
                ImGui::Spacing();
                
                // Main Wave Parameters with Global LCD Sliders
                bool heightKeyed = isWaterKeyed(WaterProp::WaveHeight);
                if (SceneUI::DrawSmartFloat("height", "Height", &wp.geo_wave_height, 0.1f, 30.0f, "%.1f", heightKeyed,
                    [&](){ insertWaterKey("Height", WaterProp::WaveHeight); }, 16)) { rebuild_mesh = true; }
                
                bool scaleKeyed = isWaterKeyed(WaterProp::WaveScale);
                if (SceneUI::DrawSmartFloat("scale", "Scale", &wp.geo_wave_scale, 10.0f, 500.0f, "%.0f", scaleKeyed,
                    [&](){ insertWaterKey("Scale", WaterProp::WaveScale); }, 16)) { rebuild_mesh = true; }
                
                bool chopKeyed = isWaterKeyed(WaterProp::Choppiness);
                if (SceneUI::DrawSmartFloat("chop", "Chop", &wp.geo_wave_choppiness, 0.0f, 3.0f, "%.2f", chopKeyed,
                    [&](){ insertWaterKey("Choppiness", WaterProp::Choppiness); }, 16)) { rebuild_mesh = true; }
                
                bool wdirKeyed = isWaterKeyed(WaterProp::WindDirection);
                if (SceneUI::DrawSmartFloat("wind", "Wind", &wp.geo_swell_direction, 0.0f, 360.0f, "%.0f", wdirKeyed,
                    [&](){ insertWaterKey("Wind Dir", WaterProp::WindDirection); }, 16)) { rebuild_mesh = true; }
                
                // ═══════════════════════════════════════════════════════════
                // 🎨 WATER APPEARANCE (Collapsible)
                // ═══════════════════════════════════════════════════════════
                ImGui::Spacing();
                if (ImGui::CollapsingHeader("Water Appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Indent(10);
                    
                    ImGui::ColorEdit3("Deep Color", &wp.deep_color.x);
                    ImGui::ColorEdit3("Surface Color", &wp.shallow_color.x);
                    
                    if (SceneUI::DrawSmartFloat("clty", "Clear", &wp.clarity, 0.0f, 1.0f, "%.2f", false, nullptr, 12)) {
                        // Display as percentage
                    }
                    
                    if (SceneUI::DrawSmartFloat("wior", "IOR", &wp.ior, 1.0f, 1.5f, "%.3f", false, nullptr, 12)) {
                        // Water IOR is typically 1.33
                    }
                    ImGui::SameLine(); ImGui::TextDisabled("(?)");
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Real water: 1.333");
                    
                    ImGui::Unindent(10);
                }
                
                // ═══════════════════════════════════════════════════════════
                // ⚙️ ADVANCED SETTINGS (Collapsible, closed by default)
                // ═══════════════════════════════════════════════════════════
                if (ImGui::CollapsingHeader("Advanced Settings")) {
                    ImGui::Indent(10);
                    
                    // Wind alignment
                    bool alignKeyed = isWaterKeyed(WaterProp::Alignment);
                    if (SceneUI::DrawSmartFloat("align", "Align", &wp.geo_alignment, 0.0f, 1.0f, "%.2f", alignKeyed,
                         [&](){ insertWaterKey("Alignment", WaterProp::Alignment); }, 12)) { rebuild_mesh = true; }
                    
                    // Damping
                    bool dampKeyed = isWaterKeyed(WaterProp::Damping);
                    if (SceneUI::DrawSmartFloat("damp", "Damp", &wp.geo_damping, 0.0f, 1.0f, "%.2f", dampKeyed,
                         [&](){ insertWaterKey("Damping", WaterProp::Damping); }, 12)) { rebuild_mesh = true; }
                    
                    // Swell
                    bool swellKeyed = isWaterKeyed(WaterProp::SwellAmplitude);
                    if (SceneUI::DrawSmartFloat("swell", "Swell", &wp.geo_swell_amplitude, 0.0f, 1.0f, "%.2f", swellKeyed,
                         [&](){ insertWaterKey("Swell", WaterProp::SwellAmplitude); }, 12)) { rebuild_mesh = true; }
                    
                    // Sharpening
                    bool sharpKeyed = isWaterKeyed(WaterProp::Sharpening);
                    if (SceneUI::DrawSmartFloat("sharp", "Sharp", &wp.geo_sharpening, 0.0f, 1.0f, "%.2f", sharpKeyed,
                         [&](){ insertWaterKey("Sharpening", WaterProp::Sharpening); }, 12)) { rebuild_mesh = true; }
                    
                    ImGui::Separator();
                    ImGui::TextDisabled("Detail Noise:");
                    
                    // Detail
                    bool detailKeyed = isWaterKeyed(WaterProp::DetailStrength);
                    if (SceneUI::DrawSmartFloat("detail", "Detail", &wp.geo_detail_strength, 0.0f, 0.5f, "%.3f", detailKeyed,
                         [&](){ insertWaterKey("Detail", WaterProp::DetailStrength); }, 12)) { rebuild_mesh = true; }
                    
                    if (SceneUI::DrawSmartFloat("dscale", "DScale", &wp.geo_detail_scale, 1.0f, 10.0f, "%.2f", false, nullptr, 12)) { rebuild_mesh = true; }
                    
                    ImGui::Separator();
                    ImGui::TextDisabled("Mesh Quality:");
                    
                    if (ImGui::Checkbox("Smooth Normals", &wp.geo_smooth_normals)) { rebuild_mesh = true; }
                    ImGui::SameLine(); ImGui::TextDisabled("(?)");
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Calculate per-vertex smooth normals\nSlightly slower but better lighting");
                    
                    // Noise octaves (only show for noise-based types)
                    if (wp.geo_noise_type != WaterWaveParams::NoiseType::Gerstner && 
                        wp.geo_noise_type != WaterWaveParams::NoiseType::TessendorfSimple) {
                        ImGui::Separator();
                        ImGui::TextDisabled("Noise Parameters:");
                        if (ImGui::SliderInt("Octaves", &wp.geo_octaves, 1, 6)) { rebuild_mesh = true; }
                        if (SceneUI::DrawSmartFloat("lac", "Lacunarity", &wp.geo_lacunarity, 1.5f, 3.0f, "%.2f", false, nullptr, 16)) { rebuild_mesh = true; }
                        if (SceneUI::DrawSmartFloat("pers", "Persistence", &wp.geo_persistence, 0.3f, 0.7f, "%.2f", false, nullptr, 16)) { rebuild_mesh = true; }
                    }
                    
                    ImGui::Unindent(10);
                }
                
                // ═══════════════════════════════════════════════════════════
                // 🎯 PRESETS (Quick Setup)
                // ═══════════════════════════════════════════════════════════
                if (ImGui::CollapsingHeader("Quick Presets")) {
                    ImGui::Indent(10);
                    
                    float btnWidth = (ImGui::GetContentRegionAvail().x - 20) / 3.0f;
                    
                    if (ImGui::Button("Calm Lake", ImVec2(btnWidth, 0))) {
                        applyPreset(0.3f, 30.0f, 0.0f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.05f, WaterWaveParams::NoiseType::Perlin);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Ocean Swell", ImVec2(btnWidth, 0))) {
                        applyPreset(3.0f, 100.0f, 270.0f, 1.0f, 0.7f, 0.3f, 0.5f, 0.3f, 0.1f, WaterWaveParams::NoiseType::TessendorfSimple);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Stormy Sea", ImVec2(btnWidth, 0))) {
                        applyPreset(8.0f, 60.0f, 315.0f, 2.5f, 0.9f, 0.5f, 0.8f, 0.7f, 0.2f, WaterWaveParams::NoiseType::Gerstner);
                    }
                    
                    if (ImGui::Button("Pool/Tank", ImVec2(btnWidth, 0))) {
                        applyPreset(0.1f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.02f, WaterWaveParams::NoiseType::FBM);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("River", ImVec2(btnWidth, 0))) {
                        applyPreset(0.5f, 20.0f, 90.0f, 0.5f, 1.0f, 0.8f, 0.2f, 0.1f, 0.08f, WaterWaveParams::NoiseType::Ridge);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Tropical", ImVec2(btnWidth, 0))) {
                        applyPreset(1.5f, 80.0f, 200.0f, 0.8f, 0.5f, 0.2f, 0.3f, 0.2f, 0.05f, WaterWaveParams::NoiseType::Gerstner);
                    }
                    
                    ImGui::Unindent(10);
                }
                
                // ═══════════════════════════════════════════════════════════
                // MESH REBUILD
                // ═══════════════════════════════════════════════════════════
                if (rebuild_mesh && wp.use_geometric_waves) {
                    WaterManager::getInstance().updateWaterMesh(selSurf);
                    ctx.renderer.resetCPUAccumulation();
                    extern bool g_bvh_rebuild_pending;
                    extern bool g_optix_rebuild_pending;
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                }
                
                // Enable geometric waves if any wave param is modified but not enabled
                if (rebuild_mesh && !wp.use_geometric_waves) {
                    wp.use_geometric_waves = true;
                    WaterManager::getInstance().updateWaterMesh(selSurf);
                    ctx.renderer.resetCPUAccumulation();
                    extern bool g_bvh_rebuild_pending;
                    extern bool g_optix_rebuild_pending;
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                }
                
                // ═══════════════════════════════════════════════════════════
                // DELETE BUTTON
                // ═══════════════════════════════════════════════════════════
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.2f, 0.2f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
                if (ImGui::Button("Delete This Ocean", ImVec2(-1, 0))) {
                    WaterManager::getInstance().removeWaterSurface(ctx.scene, selected_surf_id);
                    selected_surf_id = -1;
                    ctx.renderer.resetCPUAccumulation();
                    extern bool g_bvh_rebuild_pending;
                    extern bool g_optix_rebuild_pending;
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                }
                ImGui::PopStyleColor(2);
                
                ImGui::PopID(); // End surface ID scope
            }
        }
        ImGui::Spacing();
    }
    
    // ════════════════════════════════════════════════════════════════════════════════
    // RIVER MANAGER
    // ════════════════════════════════════════════════════════════════════════════════
    if (ImGui::CollapsingHeader("Manage Rivers", ImGuiTreeNodeFlags_DefaultOpen)) {
        
        // Option to show gizmos even when panel not active
        ImGui::Checkbox("Always Show Gizmos", &riverMgr.showGizmosWhenInactive);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Show river splines in viewport\neven when River panel is not focused");
        }
        
        // Create new river button
        if (ImGui::Button("+ New River", ImVec2(-1, 0))) {
            RiverSpline* newRiver = riverMgr.createRiver("River");
            riverMgr.editingRiverId = newRiver->id;
            riverMgr.isEditing = true;
            riverMgr.selectedControlPoint = -1;
            ProjectManager::getInstance().markModified();
        }
        
        ImGui::Separator();
        
        // List existing rivers
        auto& rivers = riverMgr.getRivers();
        for (auto& river : rivers) {
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_SpanAvailWidth;
            if (riverMgr.editingRiverId == river.id) {
                flags |= ImGuiTreeNodeFlags_Selected;
            }
            
            bool open = ImGui::TreeNodeEx(
                (void*)(intptr_t)river.id,
                flags,
                "%s (%d pts)", river.name.c_str(), (int)river.controlPointCount()
            );
            
            if (ImGui::IsItemClicked()) {
                riverMgr.editingRiverId = river.id;
                riverMgr.isEditing = true;
                riverMgr.selectedControlPoint = -1;
            }
            
            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Rename")) {
                    // TODO: Rename dialog
                }
                if (ImGui::MenuItem("Delete")) {
                    int deleteId = river.id;
                    
                    // Reset editing state BEFORE deletion
                    if (riverMgr.editingRiverId == deleteId) {
                        riverMgr.editingRiverId = -1;
                        riverMgr.isEditing = false;
                        riverMgr.selectedControlPoint = -1;
                    }
                    
                    riverMgr.removeRiver(ctx.scene, deleteId);
                    
                    // Trigger rebuild
                    extern bool g_bvh_rebuild_pending;
                    extern bool g_optix_rebuild_pending;
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    
                    ProjectManager::getInstance().markModified();
                    ImGui::EndPopup();
                    if (open) ImGui::TreePop();
                    break; // Exit loop after deletion
                }
                ImGui::EndPopup();
            }
            
            if (open) ImGui::TreePop();
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // SELECTED RIVER PROPERTIES
    // ─────────────────────────────────────────────────────────────────────────
    RiverSpline* selectedRiver = riverMgr.getRiver(riverMgr.editingRiverId);
    
    if (selectedRiver) {
        ImGui::Separator();
        
        if (ImGui::CollapsingHeader("River Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Name
            char nameBuf[128];
            strncpy(nameBuf, selectedRiver->name.c_str(), sizeof(nameBuf) - 1);
            nameBuf[sizeof(nameBuf) - 1] = 0;
            if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf))) {
                selectedRiver->name = nameBuf;
                ProjectManager::getInstance().markModified();
            }
            
            ImGui::Separator();
            ImGui::Text("Mesh Generation");
            
            if (ImGui::SliderInt("Length Subdivs", &selectedRiver->lengthSubdivisions, 4, 128)) {
                selectedRiver->needsRebuild = true;
            }
            if (ImGui::SliderInt("Width Segments", &selectedRiver->widthSegments, 1, 16)) {
                selectedRiver->needsRebuild = true;
            }
            if (SceneUI::DrawSmartFloat("rbnk", "Bank Height", &selectedRiver->bankHeight, -0.5f, 1.0f, "%.3f", false, nullptr, 16)) {
                selectedRiver->needsRebuild = true;
            }
            if (ImGui::Checkbox("Follow Terrain", &selectedRiver->followTerrain)) {
                selectedRiver->needsRebuild = true;
            }
            
            ImGui::Separator();
            ImGui::Text("Default Values (for new points)");
            SceneUI::DrawSmartFloat("rdw", "Default Width", &riverMgr.defaultWidth, 0.5f, 20.0f, "%.1f", false, nullptr, 16);
            SceneUI::DrawSmartFloat("rdd", "Default Depth", &riverMgr.defaultDepth, 0.1f, 5.0f, "%.1f", false, nullptr, 16);
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // FLOW PHYSICS & DYNAMICS
        // ─────────────────────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Flow Physics & Dynamics")) {
            RiverSpline::PhysicsParams& pp = selectedRiver->physics;
            bool changed = false;
            
            ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Geometric Deformation");
            
            if (ImGui::Checkbox("Enable Rapids/Turbulence", &pp.enableTurbulence)) changed = true;
            if (pp.enableTurbulence) {
                ImGui::Indent();
                if (SceneUI::DrawSmartFloat("rts", "Turbulence Strength", &pp.turbulenceStrength, 0.0f, 5.0f, "%.2f", false, nullptr, 16)) changed = true;
                if (SceneUI::DrawSmartFloat("rrt", "Rapids Threshold", &pp.turbulenceThreshold, 0.01f, 0.2f, "%.2f", false, nullptr, 16)) changed = true;
                if (SceneUI::DrawSmartFloat("rns", "Noise Scale", &pp.noiseScale, 0.1f, 5.0f, "%.2f", false, nullptr, 16)) changed = true;
                ImGui::Unindent();
            }
            
            if (ImGui::Checkbox("Enable Banking (Curves)", &pp.enableBanking)) changed = true;
            if (pp.enableBanking) {
                ImGui::Indent();
                if (SceneUI::DrawSmartFloat("rbs", "Banking Strength", &pp.bankingStrength, 0.0f, 3.0f, "%.2f", false, nullptr, 16)) changed = true;
                ImGui::Unindent();
            }
            
            if (ImGui::Checkbox("Enable Flow Bulge", &pp.enableFlowBulge)) changed = true;
            if (pp.enableFlowBulge) {
                ImGui::Indent();
                if (SceneUI::DrawSmartFloat("rfb", "Flow Bulge", &pp.flowBulgeStrength, 0.0f, 2.0f, "%.2f", false, nullptr, 16)) changed = true;
                ImGui::Unindent();
            }
            
            if (changed) {
                selectedRiver->needsRebuild = true;
                // Auto-rebuild for quick feedback
                if (!selectedRiver->meshTriangles.empty()) {
                    riverMgr.generateMesh(selectedRiver, ctx.scene);
                    
                    extern bool g_bvh_rebuild_pending;
                    extern bool g_optix_rebuild_pending;
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    ctx.renderer.resetCPUAccumulation();
                }
                ProjectManager::getInstance().markModified();
            }
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // CONTROL POINTS
        // ─────────────────────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Control Points", ImGuiTreeNodeFlags_DefaultOpen)) {
            
            // Edit mode toggle
            ImVec4 editColor = riverMgr.isEditing ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) : ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Button, editColor);
            if (ImGui::Button(riverMgr.isEditing ? "Stop Editing (Click terrain to add points)" : "Start Editing", ImVec2(-1, 0))) {
                riverMgr.isEditing = !riverMgr.isEditing;
            }
            ImGui::PopStyleColor();
            
            if (riverMgr.isEditing) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Click on terrain to add control points");
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Right-click point to delete");
            }
            
            ImGui::Separator();
            
            // List control points
            for (int i = 0; i < (int)selectedRiver->controlPointCount(); ++i) {
                BezierControlPoint* pt = selectedRiver->getControlPoint(i);
                if (!pt) continue;
                
                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf;
                if (riverMgr.selectedControlPoint == i) {
                    flags |= ImGuiTreeNodeFlags_Selected;
                }
                
                char label[64];
                snprintf(label, sizeof(label), "Point %d (W: %.1f)", i, pt->userData1);
                
                bool open = ImGui::TreeNodeEx((void*)(intptr_t)(i + 1000), flags, "%s", label);
                
                if (ImGui::IsItemClicked()) {
                    riverMgr.selectedControlPoint = i;
                }
                
                if (open) ImGui::TreePop();
            }
            
            // Selected point properties
            if (riverMgr.selectedControlPoint >= 0 && 
                riverMgr.selectedControlPoint < (int)selectedRiver->controlPointCount()) {
                
                BezierControlPoint* pt = selectedRiver->getControlPoint(riverMgr.selectedControlPoint);
                if (pt) {
                    ImGui::Separator();
                    
                    // Show if last point (for extrude hint)
                    bool isLastPoint = (riverMgr.selectedControlPoint == (int)selectedRiver->controlPointCount() - 1);
                    if (isLastPoint) {
                        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.8f, 1.0f), 
                            "Point %d (Last - Click terrain to extend)", riverMgr.selectedControlPoint);
                    } else {
                        ImGui::Text("Selected Point %d", riverMgr.selectedControlPoint);
                    }
                    
                    bool changed = false;
                    
                    float pos[3] = {pt->position.x, pt->position.y, pt->position.z};
                    if (ImGui::DragFloat3("Position", pos, 0.1f)) {
                        pt->position = Vec3(pos[0], pos[1], pos[2]);
                        changed = true;
                    }
                    
                    if (SceneUI::DrawSmartFloat("cpw", "Width", &pt->userData1, 0.1f, 50.0f, "%.1f", false, nullptr, 16)) {
                        changed = true;
                    }
                    
                    if (SceneUI::DrawSmartFloat("cpd", "Depth", &pt->userData2, 0.0f, 10.0f, "%.2f", false, nullptr, 16)) {
                        changed = true;
                    }
                    
                    if (ImGui::Checkbox("Auto Tangent", &pt->autoTangent)) {
                        if (pt->autoTangent) {
                            selectedRiver->spline.calculateAutoTangents();
                        }
                        changed = true;
                    }
                    
                    if (!pt->autoTangent) {
                        float tin[3] = {pt->tangentIn.x, pt->tangentIn.y, pt->tangentIn.z};
                        float tout[3] = {pt->tangentOut.x, pt->tangentOut.y, pt->tangentOut.z};
                        
                        if (ImGui::DragFloat3("Tangent In", tin, 0.1f)) {
                            pt->tangentIn = Vec3(tin[0], tin[1], tin[2]);
                            changed = true;
                        }
                        if (ImGui::DragFloat3("Tangent Out", tout, 0.1f)) {
                            pt->tangentOut = Vec3(tout[0], tout[1], tout[2]);
                            changed = true;
                        }
                    }
                    
                    // Delete button + DEL key hint
                    if (ImGui::Button("Delete Point [DEL]")) {
                        selectedRiver->removeControlPoint(riverMgr.selectedControlPoint);
                        riverMgr.selectedControlPoint = -1;
                        changed = true;
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("(Press DEL key)");
                    
                    // Handle DEL key press
                    if (ImGui::IsKeyPressed(ImGuiKey_Delete) && !ImGui::GetIO().WantTextInput) {
                        selectedRiver->removeControlPoint(riverMgr.selectedControlPoint);
                        // Select previous point or -1 if no points left
                        if (selectedRiver->controlPointCount() > 0) {
                            riverMgr.selectedControlPoint = (std::min)(
                                riverMgr.selectedControlPoint, 
                                (int)selectedRiver->controlPointCount() - 1);
                        } else {
                            riverMgr.selectedControlPoint = -1;
                        }
                        changed = true;
                    }
                    
                    // Auto-rebuild on changes (quick feedback)
                    if (changed) {
                        selectedRiver->needsRebuild = true;
                        
                        // Auto-rebuild if mesh exists (for quick iteration)
                        if (!selectedRiver->meshTriangles.empty()) {
                            riverMgr.generateMesh(selectedRiver, ctx.scene);
                            
                            extern bool g_bvh_rebuild_pending;
                            extern bool g_optix_rebuild_pending;
                            g_bvh_rebuild_pending = true;
                            g_optix_rebuild_pending = true;
                            ctx.renderer.resetCPUAccumulation();
                        }
                        
                        ProjectManager::getInstance().markModified();
                    }
                }
            }
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // WATER MATERIAL INFO (Parameters are managed via Water panel)
        // ─────────────────────────────────────────────────────────────────────
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Water Material");
        if (selectedRiver->waterSurfaceId >= 0) {
            ImGui::BulletText("Registered as WaterSurface (ID: %d)", selectedRiver->waterSurfaceId);
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), 
                "Edit waves, colors, and effects in the Water panel");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), 
                "Build mesh to create water surface");
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // ACTIONS
        // ─────────────────────────────────────────────────────────────────────
        ImGui::Separator();
        
        if (selectedRiver->needsRebuild) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Mesh needs rebuild!");
        }
        
        if (ImGui::Button("Rebuild Mesh", ImVec2(-1, 0))) {
            riverMgr.generateMesh(selectedRiver, ctx.scene);
            
            // Trigger scene rebuild
            extern bool g_bvh_rebuild_pending;
            extern bool g_optix_rebuild_pending;
            g_bvh_rebuild_pending = true;
            g_optix_rebuild_pending = true;
            ctx.renderer.resetCPUAccumulation();
        }
        
        // Carve River Bed into terrain
        if (TerrainManager::getInstance().hasActiveTerrain()) {
            if (ImGui::CollapsingHeader("Carve Settings")) {
                SceneUI::DrawSmartFloat("cdm", "Depth Multiplier", &riverMgr.carveDepthMult, 0.1f, 3.0f, "%.1f", false, nullptr, 16);
                SceneUI::DrawSmartFloat("csm", "Smoothness", &riverMgr.carveSmoothness, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                ImGui::Checkbox("Apply Post-Erosion (Smooth Edges)", &riverMgr.carveAutoPostErosion);
                if (riverMgr.carveAutoPostErosion) {
                    ImGui::SliderInt("Erosion Iterations", &riverMgr.carveErosionIterations, 5, 30);
                }
                
                ImGui::Separator();
                
                // ═══════════════════════════════════════════════════════════════
                // NATURAL CAVE SETTINGS (Doğal Nehir Yatağı)
                // ═══════════════════════════════════════════════════════════════
                ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.7f, 1.0f), "Natural Riverbed Features");
                
                // Noise-based edge irregularity
                ImGui::Checkbox("Edge Noise", &riverMgr.carveEnableNoise);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Add natural irregularity to river edges");
                }
                if (riverMgr.carveEnableNoise) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("cns", "Noise Scale", &riverMgr.carveNoiseScale, 0.05f, 0.5f, "%.2f", false, nullptr, 16);
                    SceneUI::DrawSmartFloat("cnst", "Noise Strength", &riverMgr.carveNoiseStrength, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                    ImGui::Unindent();
                }
                
                // Deep Pools
                ImGui::Checkbox("Deep Pools", &riverMgr.carveEnableDeepPools);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Add random deep pools along the river");
                }
                if (riverMgr.carveEnableDeepPools) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("cpf", "Pool Frequency", &riverMgr.carvePoolFrequency, 0.0f, 0.5f, "%.2f", false, nullptr, 16);
                    SceneUI::DrawSmartFloat("cpdm", "Pool Depth Mult", &riverMgr.carvePoolDepthMult, 1.0f, 3.0f, "%.1f", false, nullptr, 16);
                    ImGui::Unindent();
                }
                
                // Riffles (shallow zones)
                ImGui::Checkbox("Riffles (Shallow Zones)", &riverMgr.carveEnableRiffles);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Add shallow riffle zones (rapids-like areas)");
                }
                if (riverMgr.carveEnableRiffles) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("crf", "Riffle Frequency", &riverMgr.carveRiffleFrequency, 0.0f, 0.5f, "%.2f", false, nullptr, 16);
                    SceneUI::DrawSmartFloat("crdm", "Riffle Depth Mult", &riverMgr.carveRiffleDepthMult, 0.1f, 0.8f, "%.2f", false, nullptr, 16);
                    ImGui::Unindent();
                }
                
                // Asymmetric Banks (meander physics)
                ImGui::Checkbox("Asymmetric Banks", &riverMgr.carveEnableAsymmetry);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Make outer bends deeper, inner bends shallower (realistic meander physics)");
                }
                if (riverMgr.carveEnableAsymmetry) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("cas", "Asymmetry Strength", &riverMgr.carveAsymmetryStrength, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                    ImGui::Unindent();
                }
                
                // Point Bars
                ImGui::Checkbox("Point Bars", &riverMgr.carveEnablePointBars);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Add sediment deposits on inner bends (Point Bar formation)");
                }
                if (riverMgr.carveEnablePointBars) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("cpbs", "Point Bar Strength", &riverMgr.carvePointBarStrength, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                    ImGui::Unindent();
                }
                
                ImGui::Separator();
                
                // Auto-Carve Option
                ImGui::Checkbox("Auto-Carve on Move", &riverMgr.autoCarveOnMove);
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Automatically carves terrain when points are moved.\nWARNING: Destructive (leaves old channel).");
                }
                
                ImGui::Separator();
                
                // ═══════════════════════════════════════════════════════════════
                // CARVE BUTTONS
                // ═══════════════════════════════════════════════════════════════
                
                // Standard Carve Button
                if (ImGui::Button("Carve River Bed (Simple)", ImVec2(-1, 0))) {
                    if (selectedRiver->spline.pointCount() >= 2) {
                        // Sample many points along spline
                        std::vector<Vec3> samplePoints;
                        std::vector<float> sampleWidths;
                        std::vector<float> sampleDepths;
                        
                        int numSamples = selectedRiver->lengthSubdivisions * 3;
                        for (int i = 0; i <= numSamples; ++i) {
                            float t = (float)i / (float)numSamples;
                            samplePoints.push_back(selectedRiver->spline.samplePosition(t));
                            sampleWidths.push_back(selectedRiver->spline.sampleUserData1(t));
                            sampleDepths.push_back(selectedRiver->spline.sampleUserData2(t) * riverMgr.carveDepthMult);
                        }
                        
                        // Carve the river bed
                        TerrainManager::getInstance().carveRiverBed(
                            -1,
                            samplePoints,
                            sampleWidths,
                            sampleDepths,
                            riverMgr.carveSmoothness,
                            ctx.scene
                        );
                        
                        // Apply post-erosion
                        if (riverMgr.carveAutoPostErosion) {
                            auto& terrains = TerrainManager::getInstance().getTerrains();
                            if (!terrains.empty()) {
                                ThermalErosionParams ep;
                                ep.iterations = riverMgr.carveErosionIterations;
                                ep.talusAngle = 0.3f;
                                ep.erosionAmount = 0.4f;
                                TerrainManager::getInstance().thermalErosion(&terrains[0], ep);
                            }
                        }
                        
                        // Rebuild river mesh
                        selectedRiver->needsRebuild = true;
                        riverMgr.generateMesh(selectedRiver, ctx.scene);
                        
                        extern bool g_bvh_rebuild_pending;
                        extern bool g_optix_rebuild_pending;
                        g_bvh_rebuild_pending = true;
                        g_optix_rebuild_pending = true;
                        ctx.renderer.resetCPUAccumulation();
                        
                        ProjectManager::getInstance().markModified();
                    }
                }
                
                // NATURAL Carve Button (Main Feature)
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.4f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.5f, 1.0f));
                if (ImGui::Button("Carve Natural Riverbed", ImVec2(-1, 0))) {
                    if (selectedRiver->spline.pointCount() >= 2) {
                        // Sample many points along spline
                        std::vector<Vec3> samplePoints;
                        std::vector<float> sampleWidths;
                        std::vector<float> sampleDepths;
                        
                        int numSamples = selectedRiver->lengthSubdivisions * 3;
                        for (int i = 0; i <= numSamples; ++i) {
                            float t = (float)i / (float)numSamples;
                            samplePoints.push_back(selectedRiver->spline.samplePosition(t));
                            sampleWidths.push_back(selectedRiver->spline.sampleUserData1(t));
                            sampleDepths.push_back(selectedRiver->spline.sampleUserData2(t) * riverMgr.carveDepthMult);
                        }
                        
                        // Build NaturalCarveParams from UI settings
                        TerrainManager::NaturalCarveParams np;
                        np.enableNoise = riverMgr.carveEnableNoise;
                        np.noiseScale = riverMgr.carveNoiseScale;
                        np.noiseStrength = riverMgr.carveNoiseStrength;
                        np.enableDeepPools = riverMgr.carveEnableDeepPools;
                        np.poolFrequency = riverMgr.carvePoolFrequency;
                        np.poolDepthMult = riverMgr.carvePoolDepthMult;
                        np.enableRiffles = riverMgr.carveEnableRiffles;
                        np.riffleFrequency = riverMgr.carveRiffleFrequency;
                        np.riffleDepthMult = riverMgr.carveRiffleDepthMult;
                        np.enableAsymmetry = riverMgr.carveEnableAsymmetry;
                        np.asymmetryStrength = riverMgr.carveAsymmetryStrength;
                        np.enablePointBars = riverMgr.carveEnablePointBars;
                        np.pointBarStrength = riverMgr.carvePointBarStrength;
                        
                        // Carve the NATURAL river bed
                        TerrainManager::getInstance().carveRiverBedNatural(
                            -1,
                            samplePoints,
                            sampleWidths,
                            sampleDepths,
                            riverMgr.carveSmoothness,
                            np,
                            ctx.scene
                        );
                        
                        // Apply post-erosion
                        if (riverMgr.carveAutoPostErosion) {
                            auto& terrains = TerrainManager::getInstance().getTerrains();
                            if (!terrains.empty()) {
                                ThermalErosionParams ep;
                                ep.iterations = riverMgr.carveErosionIterations;
                                ep.talusAngle = 0.3f;
                                ep.erosionAmount = 0.4f;
                                TerrainManager::getInstance().thermalErosion(&terrains[0], ep);
                            }
                        }
                        
                        // Rebuild river mesh
                        selectedRiver->needsRebuild = true;
                        riverMgr.generateMesh(selectedRiver, ctx.scene);
                        
                        extern bool g_bvh_rebuild_pending;
                        extern bool g_optix_rebuild_pending;
                        g_bvh_rebuild_pending = true;
                        g_optix_rebuild_pending = true;
                        ctx.renderer.resetCPUAccumulation();
                        
                        ProjectManager::getInstance().markModified();
                    }
                }
                ImGui::PopStyleColor(2);
                
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
                    "Natural: Pools, Riffles, Asymmetry, Point Bars");
            }
        }
        
        if (ImGui::Button("Clear All Points", ImVec2(-1, 0))) {
            // Remove existing mesh from scene
            for (auto& tri : selectedRiver->meshTriangles) {
                auto it = std::find(ctx.scene.world.objects.begin(), ctx.scene.world.objects.end(),
                                   std::static_pointer_cast<Hittable>(tri));
                if (it != ctx.scene.world.objects.end()) {
                    ctx.scene.world.objects.erase(it);
                }
            }
            selectedRiver->meshTriangles.clear();
            selectedRiver->spline.clear();
            selectedRiver->needsRebuild = true;
            riverMgr.selectedControlPoint = -1;
            
            extern bool g_bvh_rebuild_pending;
            extern bool g_optix_rebuild_pending;
            g_bvh_rebuild_pending = true;
            g_optix_rebuild_pending = true;
            
            ProjectManager::getInstance().markModified();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RIVER SPLINE VISUALIZATION (Call from draw loop)
// ═══════════════════════════════════════════════════════════════════════════════
inline void SceneUI::drawRiverGizmos(UIContext& ctx, bool& gizmo_hit) {
    auto& riverMgr = RiverManager::getInstance();
    
    // Check if panel is active (drawn this frame or last frame)
    bool isPanelActive = (riverMgr.lastActiveFrame >= ImGui::GetFrameCount() - 1);
    
    // Hide gizmos if not editing active, panel is not focused, and "Always Show" is off
    if (!riverMgr.showGizmosWhenInactive && !isPanelActive) {
        return;
    }
    
    if (!ctx.scene.camera) return;
    
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImGuiIO& io = ImGui::GetIO();
    Camera& cam = *ctx.scene.camera;
    
    Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    
    float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    float tan_half_fov = tanf(fov_rad * 0.5f);
    float aspect = io.DisplaySize.x / io.DisplaySize.y;
    
    auto Project = [&](const Vec3& p) -> ImVec2 {
        Vec3 to_point = p - cam.lookfrom;
        float depth = to_point.dot(cam_forward);
        if (depth <= 0.1f) return ImVec2(-10000, -10000);
        
        float local_x = to_point.dot(cam_right);
        float local_y = to_point.dot(cam_up);
        
        float half_h = depth * tan_half_fov;
        float half_w = half_h * aspect;
        
        return ImVec2(
            ((local_x / half_w) * 0.5f + 0.5f) * io.DisplaySize.x,
            (0.5f - (local_y / half_h) * 0.5f) * io.DisplaySize.y
        );
    };
    
    auto IsOnScreen = [](const ImVec2& v) { return v.x > -5000; };
    
    // Draw all rivers
    for (auto& river : riverMgr.getRivers()) {
        bool isSelected = (riverMgr.editingRiverId == river.id);
        
        if (!river.showSpline) continue;
        if (river.spline.pointCount() < 2) {
            // Just draw single point if only one exists
            if (river.spline.pointCount() == 1) {
                ImVec2 pt = Project(river.spline.points[0].position);
                if (IsOnScreen(pt)) {
                    draw_list->AddCircleFilled(pt, 8.0f, IM_COL32(100, 150, 255, 200));
                    draw_list->AddCircle(pt, 8.0f, IM_COL32(255, 255, 255, 255), 0, 2.0f);
                }
            }
            continue;
        }
        
        // Draw spline curve
        ImU32 splineColor = isSelected ? IM_COL32(100, 200, 255, 200) : IM_COL32(50, 100, 200, 150);
        
        Vec3 prevPos = river.spline.samplePosition(0);
        for (int i = 1; i <= 50; ++i) {
            float t = (float)i / 50.0f;
            Vec3 pos = river.spline.samplePosition(t);
            
            ImVec2 p1 = Project(prevPos);
            ImVec2 p2 = Project(pos);
            
            if (IsOnScreen(p1) && IsOnScreen(p2)) {
                draw_list->AddLine(p1, p2, splineColor, 2.0f);
            }
            
            prevPos = pos;
        }
        
        // Draw width visualization (dashed lines on sides)
        if (isSelected) {
            for (int i = 0; i <= 20; ++i) {
                float t = (float)i / 20.0f;
                Vec3 center = river.spline.samplePosition(t);
                Vec3 right = river.spline.sampleRight(t);
                float width = river.spline.sampleUserData1(t);
                
                Vec3 left3d = center - right * (width * 0.5f);
                Vec3 right3d = center + right * (width * 0.5f);
                
                ImVec2 leftPt = Project(left3d);
                ImVec2 rightPt = Project(right3d);
                
                if (i % 2 == 0) { // Dashed effect
                    if (IsOnScreen(leftPt)) {
                        draw_list->AddCircleFilled(leftPt, 2.0f, IM_COL32(100, 200, 255, 100));
                    }
                    if (IsOnScreen(rightPt)) {
                        draw_list->AddCircleFilled(rightPt, 2.0f, IM_COL32(100, 200, 255, 100));
                    }
                }
            }
        }
        
        // Draw control points
        if (river.showControlPoints || isSelected) {
            for (int i = 0; i < (int)river.spline.pointCount(); ++i) {
                auto& pt = river.spline.points[i];
                ImVec2 screenPt = Project(pt.position);
                
                if (!IsOnScreen(screenPt)) continue;
                
                bool isPointSelected = (isSelected && riverMgr.selectedControlPoint == i);
                
                // Point appearance
                float radius = isPointSelected ? 10.0f : 7.0f;
                ImU32 fillColor = isPointSelected ? IM_COL32(255, 200, 50, 255) : IM_COL32(100, 150, 255, 200);
                ImU32 outlineColor = IM_COL32(255, 255, 255, 255);
                
                draw_list->AddCircleFilled(screenPt, radius, fillColor);
                draw_list->AddCircle(screenPt, radius, outlineColor, 0, 2.0f);
                
                // Point index label
                char label[16];
                snprintf(label, sizeof(label), "%d", i);
                draw_list->AddText(ImVec2(screenPt.x + 12, screenPt.y - 6), IM_COL32(255, 255, 255, 200), label);
                
                // Draw tangent handles for selected point
                if (isPointSelected && !pt.autoTangent) {
                    Vec3 tangentInWorld = pt.position + pt.tangentIn;
                    Vec3 tangentOutWorld = pt.position + pt.tangentOut;
                    
                    ImVec2 tin = Project(tangentInWorld);
                    ImVec2 tout = Project(tangentOutWorld);
                    
                    if (IsOnScreen(tin)) {
                        draw_list->AddLine(screenPt, tin, IM_COL32(255, 100, 100, 200), 1.5f);
                        draw_list->AddCircleFilled(tin, 5.0f, IM_COL32(255, 100, 100, 255));
                    }
                    if (IsOnScreen(tout)) {
                        draw_list->AddLine(screenPt, tout, IM_COL32(100, 255, 100, 200), 1.5f);
                        draw_list->AddCircleFilled(tout, 5.0f, IM_COL32(100, 255, 100, 255));
                    }
                }
                
                // Click detection for point selection
                if (isSelected && !ImGuizmo::IsOver()) {
                    float dx = io.MousePos.x - screenPt.x;
                    float dy = io.MousePos.y - screenPt.y;
                    float dist = sqrtf(dx * dx + dy * dy);
                    
                    if (dist < 15.0f) {
                        // Left click - select
                        if (ImGui::IsMouseClicked(0)) {
                            riverMgr.selectedControlPoint = i;
                            gizmo_hit = true;
                        }
                        // Right click - delete
                        else if (ImGui::IsMouseClicked(1)) {
                            river.removeControlPoint(i);
                            river.needsRebuild = true;
                            riverMgr.selectedControlPoint = -1;
                            ProjectManager::getInstance().markModified();
                            gizmo_hit = true;
                            break;
                        }
                    }
                }
                
                // ─────────────────────────────────────────────────────────────
                // DRAG TO MOVE selected control point
                // ─────────────────────────────────────────────────────────────
                if (isPointSelected && ImGui::IsMouseDragging(0) && !ImGui::GetIO().WantCaptureMouse) {
                    // Project mouse delta to world movement
                    Vec3 forward = cam_forward;
                    Vec3 right = cam_right;
                    Vec3 up = cam_up;
                    
                    // Calculate depth of point from camera
                    Vec3 toPoint = pt.position - cam.lookfrom;
                    float depth = toPoint.dot(forward);
                    
                    if (depth > 0.1f) {
                        // Convert pixel delta to world delta
                        float half_h = depth * tan_half_fov;
                        float pixelsPerUnit = io.DisplaySize.y / (2.0f * half_h);
                        
                        ImVec2 delta = io.MouseDelta;
                        float worldDeltaX = delta.x / pixelsPerUnit;
                        float worldDeltaY = -delta.y / pixelsPerUnit;
                        
                        // Move in camera plane (right + up)
                        Vec3 movement = right * worldDeltaX + up * worldDeltaY;
                        pt.position = pt.position + movement;
                        
                        // Optionally snap Y to terrain
                        if (river.followTerrain && TerrainManager::getInstance().hasActiveTerrain()) {
                            pt.position.y = TerrainManager::getInstance().sampleHeight(pt.position.x, pt.position.z);
                        }
                        
                        // Recalculate tangents if auto
                        if (pt.autoTangent) {
                            river.spline.calculateAutoTangents();
                        }
                        
                        river.needsRebuild = true;
                        riverMgr.isDraggingPoint = true;  // Track drag state
                        gizmo_hit = true;
                    }
                }
                
                // Rebuild mesh when drag ends
                if (isPointSelected && riverMgr.isDraggingPoint && ImGui::IsMouseReleased(0)) {
                    riverMgr.isDraggingPoint = false;
                    
                    // AUTO-CARVE ON MOVE
                    if (riverMgr.autoCarveOnMove && TerrainManager::getInstance().hasActiveTerrain()) {
                        // Sample many points along spline
                        std::vector<Vec3> samplePoints;
                        std::vector<float> sampleWidths;
                        std::vector<float> sampleDepths;
                        
                        int numSamples = river.lengthSubdivisions * 3;
                        for (int k = 0; k <= numSamples; ++k) {
                            float t = (float)k / (float)numSamples;
                            samplePoints.push_back(river.spline.samplePosition(t));
                            sampleWidths.push_back(river.spline.sampleUserData1(t));
                            sampleDepths.push_back(river.spline.sampleUserData2(t) * riverMgr.carveDepthMult);
                        }
                        
                        // Carve
                        TerrainManager::getInstance().carveRiverBed(
                            -1,
                            samplePoints,
                            sampleWidths,
                            sampleDepths,
                            riverMgr.carveSmoothness,
                            ctx.scene
                        );
                        
                        // Post-Erosion
                        if (riverMgr.carveAutoPostErosion) {
                            auto& terrains = TerrainManager::getInstance().getTerrains();
                            if (!terrains.empty()) {
                                ThermalErosionParams ep;
                                ep.iterations = riverMgr.carveErosionIterations;
                                ep.talusAngle = 0.3f;
                                ep.erosionAmount = 0.4f;
                                TerrainManager::getInstance().thermalErosion(&terrains[0], ep);
                            }
                        }
                    }
                    
                    // Rebuild the mesh after dragging

                    if (river.needsRebuild && !river.meshTriangles.empty()) {
                        riverMgr.generateMesh(&river, ctx.scene);
                        
                        extern bool g_bvh_rebuild_pending;
                        extern bool g_optix_rebuild_pending;
                        g_bvh_rebuild_pending = true;
                        g_optix_rebuild_pending = true;
                        ctx.renderer.resetCPUAccumulation();
                    }
                    
                    ProjectManager::getInstance().markModified();
                }
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // ADD NEW POINT ON TERRAIN CLICK (when editing)
    // ─────────────────────────────────────────────────────────────────────────
    RiverSpline* editingRiver = riverMgr.getRiver(riverMgr.editingRiverId);
    
    if (editingRiver && riverMgr.isEditing && !gizmo_hit && !ImGuizmo::IsOver()) {
        if (ImGui::IsMouseClicked(0) && !ImGui::GetIO().WantCaptureMouse) {
            // Raycast to terrain
            Vec3 rayOrigin, rayDir;
            float mx = io.MousePos.x;
            float my = io.MousePos.y;
            
            // Calculate ray from camera
            float ndc_x = (mx / io.DisplaySize.x) * 2.0f - 1.0f;
            float ndc_y = 1.0f - (my / io.DisplaySize.y) * 2.0f;
            
            Vec3 forward = (cam.lookat - cam.lookfrom).normalize();
            Vec3 right = forward.cross(cam.vup).normalize();
            Vec3 up = right.cross(forward).normalize();
            
            float half_h = tan_half_fov;
            float half_w = half_h * aspect;
            
            rayDir = (forward + right * (ndc_x * half_w) + up * (ndc_y * half_h)).normalize();
            rayOrigin = cam.lookfrom;
            
            // Simple ground plane intersection for now
            // TODO: Proper terrain raycast
            float groundY = 0.0f;
            if (TerrainManager::getInstance().hasActiveTerrain()) {
                // Intersect with approximate ground plane
                if (rayDir.y < -0.01f) {
                    float t = (groundY - rayOrigin.y) / rayDir.y;
                    if (t > 0) {
                        Vec3 hitPoint = rayOrigin + rayDir * t;
                        // Get actual terrain height at hit point
                        hitPoint.y = TerrainManager::getInstance().sampleHeight(hitPoint.x, hitPoint.z);
                        
                        // Add control point
                        editingRiver->addControlPoint(hitPoint, riverMgr.defaultWidth, riverMgr.defaultDepth);
                        riverMgr.selectedControlPoint = (int)editingRiver->controlPointCount() - 1;
                        ProjectManager::getInstance().markModified();
                        gizmo_hit = true;
                    }
                }
            } else {
                // No terrain - intersect with Y=0 plane
                if (fabsf(rayDir.y) > 0.01f) {
                    float t = -rayOrigin.y / rayDir.y;
                    if (t > 0) {
                        Vec3 hitPoint = rayOrigin + rayDir * t;
                        editingRiver->addControlPoint(hitPoint, riverMgr.defaultWidth, riverMgr.defaultDepth);
                        riverMgr.selectedControlPoint = (int)editingRiver->controlPointCount() - 1;
                        ProjectManager::getInstance().markModified();
                        gizmo_hit = true;
                    }
                }
            }
        }
    }
}

#endif // SCENE_UI_RIVER_HPP
