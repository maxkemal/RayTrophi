// scene_ui_world.cpp
// World/Environment settings UI implementation
// Part of SceneUI - extracted for maintainability

#include "scene_ui.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "ui_modern.h"
#include "imgui.h"
#include "scene_data.h"

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>

// Forward declaration for file dialog (defined in scene_ui.cpp)
extern std::string openFileDialogW(const wchar_t* filter, const std::string& initialDir = "", const std::string& defaultFilename = "");
#endif

void SceneUI::drawWorldContent(UIContext& ctx) {
    World& world = ctx.renderer.world;
    WorldMode current_mode = world.getMode();
    
    // Auto-select "World" track in timeline when World panel is active
    // This allows keyframe manipulation (move, delete) for world keyframes
    timeline.selected_track = "World";
    
    UIWidgets::ColoredHeader("Environment Settings", ImVec4(0.3f, 0.7f, 1.0f, 1.0f));
    UIWidgets::Divider();
    
    // Set uniform field width for all inputs (compact layout)
    const float FIELD_WIDTH = 140.0f;
    ImGui::PushItemWidth(FIELD_WIDTH);
    
    bool changed = false;

    // ═══════════════════════════════════════════════════════════
    // ANIMATION HELPER LAMBDAS
    // ═══════════════════════════════════════════════════════════
    auto KeyframeButton = [&](const char* id, bool keyed, const char* prop_name = nullptr) -> bool {
        ImGui::PushID(id);
        float s = ImGui::GetFrameHeight();
        ImVec2 pos = ImGui::GetCursorScreenPos();
        bool clicked = ImGui::InvisibleButton("kbtn", ImVec2(s, s));
        
        ImU32 bg = keyed ? IM_COL32(255, 200, 0, 255) : IM_COL32(40, 40, 40, 255);
        ImU32 border = IM_COL32(180, 180, 180, 255);
        
        bool hovered = ImGui::IsItemHovered();
        if (hovered) {
            border = IM_COL32(255, 255, 255, 255);
            bg = keyed ? IM_COL32(255, 220, 50, 255) : IM_COL32(70, 70, 70, 255);
            
            // Tooltip - reflects toggle behavior
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

    // Enum for world property spec ification
    enum class WorldProp {
        BackgroundColor, BackgroundStrength, HDRIRotation,
        SunElevation, SunAzimuth, SunIntensity, SunSize,
        AirDensity, DustDensity, OzoneDensity, Altitude, MieAnisotropy,
        CloudDensity, CloudCoverage, CloudScale, CloudOffset
    };
    
    auto isWorldKeyed = [&](WorldProp prop) -> bool {
        auto it = ctx.scene.timeline.tracks.find("World");
        if (it == ctx.scene.timeline.tracks.end()) return false;
        int cf = ctx.render_settings.animation_current_frame;
        for (auto& kf : it->second.keyframes) {
            if (kf.frame == cf && kf.has_world) {
                switch(prop) {
                    case WorldProp::BackgroundColor: return kf.world.has_background_color;
                    case WorldProp::BackgroundStrength: return kf.world.has_background_strength;
                    case WorldProp::HDRIRotation: return kf.world.has_hdri_rotation;
                    case WorldProp::SunElevation: return kf.world.has_sun_elevation;
                    case WorldProp::SunAzimuth: return kf.world.has_sun_azimuth;
                    case WorldProp::SunIntensity: return kf.world.has_sun_intensity;
                    case WorldProp::SunSize: return kf.world.has_sun_size;
                    case WorldProp::AirDensity: return kf.world.has_air_density;
                    case WorldProp::DustDensity: return kf.world.has_dust_density;
                    case WorldProp::OzoneDensity: return kf.world.has_ozone_density;
                    case WorldProp::Altitude: return kf.world.has_altitude;
                    case WorldProp::MieAnisotropy: return kf.world.has_mie_anisotropy;
                    case WorldProp::CloudDensity: return kf.world.has_cloud_density;
                    case WorldProp::CloudCoverage: return kf.world.has_cloud_coverage;
                    case WorldProp::CloudScale: return kf.world.has_cloud_scale;
                    case WorldProp::CloudOffset: return kf.world.has_cloud_offset;
                }
            }
        }
        return false;
    };

    auto insertWorldKey = [&](const std::string& label, WorldProp prop) {
        int cf = ctx.render_settings.animation_current_frame;
        World& w = ctx.renderer.world;
        NishitaSkyParams np = w.getNishitaParams();
        
        auto& track = ctx.scene.timeline.tracks["World"];
        bool found = false;
        
        // Helper lambda to check if specific property is keyed
        auto isPropKeyed = [](const Keyframe& kf, WorldProp p) -> bool {
            if (!kf.has_world) return false;
            switch(p) {
                case WorldProp::BackgroundColor: return kf.world.has_background_color;
                case WorldProp::BackgroundStrength: return kf.world.has_background_strength;
                case WorldProp::HDRIRotation: return kf.world.has_hdri_rotation;
                case WorldProp::SunElevation: return kf.world.has_sun_elevation;
                case WorldProp::SunAzimuth: return kf.world.has_sun_azimuth;
                case WorldProp::SunIntensity: return kf.world.has_sun_intensity;
                case WorldProp::SunSize: return kf.world.has_sun_size;
                case WorldProp::AirDensity: return kf.world.has_air_density;
                case WorldProp::DustDensity: return kf.world.has_dust_density;
                case WorldProp::OzoneDensity: return kf.world.has_ozone_density;
                case WorldProp::Altitude: return kf.world.has_altitude;
                case WorldProp::MieAnisotropy: return kf.world.has_mie_anisotropy;
                case WorldProp::CloudDensity: return kf.world.has_cloud_density;
                case WorldProp::CloudCoverage: return kf.world.has_cloud_coverage;
                case WorldProp::CloudScale: return kf.world.has_cloud_scale;
                case WorldProp::CloudOffset: return kf.world.has_cloud_offset;
            }
            return false;
        };
        
        // Helper lambda to clear specific property
        auto clearProp = [](Keyframe& kf, WorldProp p) {
            switch(p) {
                case WorldProp::BackgroundColor: kf.world.has_background_color = false; break;
                case WorldProp::BackgroundStrength: kf.world.has_background_strength = false; break;
                case WorldProp::HDRIRotation: kf.world.has_hdri_rotation = false; break;
                case WorldProp::SunElevation: kf.world.has_sun_elevation = false; break;
                case WorldProp::SunAzimuth: kf.world.has_sun_azimuth = false; break;
                case WorldProp::SunIntensity: kf.world.has_sun_intensity = false; break;
                case WorldProp::SunSize: kf.world.has_sun_size = false; break;
                case WorldProp::AirDensity: kf.world.has_air_density = false; break;
                case WorldProp::DustDensity: kf.world.has_dust_density = false; break;
                case WorldProp::OzoneDensity: kf.world.has_ozone_density = false; break;
                case WorldProp::Altitude: kf.world.has_altitude = false; break;
                case WorldProp::MieAnisotropy: kf.world.has_mie_anisotropy = false; break;
                case WorldProp::CloudDensity: kf.world.has_cloud_density = false; break;
                case WorldProp::CloudCoverage: kf.world.has_cloud_coverage = false; break;
                case WorldProp::CloudScale: kf.world.has_cloud_scale = false; break;
                case WorldProp::CloudOffset: kf.world.has_cloud_offset = false; break;
            }
        };
        
        // Try to find existing keyframe on current frame
        for (auto it = track.keyframes.begin(); it != track.keyframes.end(); ++it) {
            if (it->frame == cf) {
                // TOGGLE BEHAVIOR: If property is already keyed, remove it
                if (isPropKeyed(*it, prop)) {
                    clearProp(*it, prop);
                    
                    // Check if keyframe is now empty (no world properties left)
                    WorldKeyframe& wk = it->world;
                    bool hasAnyProp = wk.has_background_color || wk.has_background_strength || 
                                      wk.has_hdri_rotation || wk.has_sun_elevation || 
                                      wk.has_sun_azimuth || wk.has_sun_intensity || wk.has_sun_size ||
                                      wk.has_air_density || wk.has_dust_density || wk.has_ozone_density ||
                                      wk.has_altitude || wk.has_mie_anisotropy || wk.has_cloud_density ||
                                      wk.has_cloud_coverage || wk.has_cloud_scale || wk.has_cloud_offset;
                    
                    if (!hasAnyProp) {
                        it->has_world = false;
                        // If no other data in keyframe, remove it entirely
                        if (!it->has_transform && !it->has_camera && !it->has_light && !it->has_material) {
                            track.keyframes.erase(it);
                        }
                    }
                    SCENE_LOG_INFO("Removed " + label + " keyframe at frame " + std::to_string(cf));
                    return;
                }
                
                // Property not keyed - add it (existing merge logic)
                it->has_world = true;
                
                switch(prop) {
                    case WorldProp::BackgroundColor:
                        it->world.has_background_color = true;
                        it->world.background_color = ctx.scene.background_color;
                        break;
                    case WorldProp::BackgroundStrength:
                        it->world.has_background_strength = true;
                        it->world.background_strength = w.getColorIntensity();
                        break;
                    case WorldProp::HDRIRotation:
                        it->world.has_hdri_rotation = true;
                        it->world.hdri_rotation = w.getHDRIRotation();
                        break;
                    case WorldProp::SunElevation:
                        it->world.has_sun_elevation = true;
                        it->world.sun_elevation = np.sun_elevation;
                        break;
                    case WorldProp::SunAzimuth:
                        it->world.has_sun_azimuth = true;
                        it->world.sun_azimuth = np.sun_azimuth;
                        break;
                    case WorldProp::SunIntensity:
                        it->world.has_sun_intensity = true;
                        it->world.sun_intensity = np.sun_intensity;
                        break;
                    case WorldProp::SunSize:
                        it->world.has_sun_size = true;
                        it->world.sun_size = np.sun_size;
                        break;
                    case WorldProp::AirDensity:
                        it->world.has_air_density = true;
                        it->world.air_density = np.air_density;
                        break;
                    case WorldProp::DustDensity:
                        it->world.has_dust_density = true;
                        it->world.dust_density = np.dust_density;
                        break;
                    case WorldProp::OzoneDensity:
                        it->world.has_ozone_density = true;
                        it->world.ozone_density = np.ozone_density;
                        break;
                    case WorldProp::Altitude:
                        it->world.has_altitude = true;
                        it->world.altitude = np.altitude;
                        break;
                    case WorldProp::MieAnisotropy:
                        it->world.has_mie_anisotropy = true;
                        it->world.mie_anisotropy = np.mie_anisotropy;
                        break;
                    case WorldProp::CloudDensity:
                        it->world.has_cloud_density = true;
                        it->world.cloud_density = np.cloud_density;
                        break;
                    case WorldProp::CloudCoverage:
                        it->world.has_cloud_coverage = true;
                        it->world.cloud_coverage = np.cloud_coverage;
                        break;
                    case WorldProp::CloudScale:
                        it->world.has_cloud_scale = true;
                        it->world.cloud_scale = np.cloud_scale;
                        break;
                    case WorldProp::CloudOffset:
                        it->world.has_cloud_offset = true;
                        it->world.cloud_offset_x = np.cloud_offset_x;
                        it->world.cloud_offset_z = np.cloud_offset_z;
                        break;
                }
                
                found = true;
                break;
            }
        }
        
        if (!found) {
            // Create new keyframe with ONLY the specified property
            Keyframe kf(cf);
            kf.has_world = true;
            WorldKeyframe& wk = kf.world;
            
            switch(prop) {
                case WorldProp::BackgroundColor:
                    wk.has_background_color = true;
                    wk.background_color = ctx.scene.background_color;
                    break;
                case WorldProp::BackgroundStrength:
                    wk.has_background_strength = true;
                    wk.background_strength = w.getColorIntensity();
                    break;
                case WorldProp::HDRIRotation:
                    wk.has_hdri_rotation = true;
                    wk.hdri_rotation = w.getHDRIRotation();
                    break;
                case WorldProp::SunElevation:
                    wk.has_sun_elevation = true;
                    wk.sun_elevation = np.sun_elevation;
                    break;
                case WorldProp::SunAzimuth:
                    wk.has_sun_azimuth = true;
                    wk.sun_azimuth = np.sun_azimuth;
                    break;
                case WorldProp::SunIntensity:
                    wk.has_sun_intensity = true;
                    wk.sun_intensity = np.sun_intensity;
                    break;
                case WorldProp::SunSize:
                    wk.has_sun_size = true;
                    wk.sun_size = np.sun_size;
                    break;
               case WorldProp::AirDensity:
                    wk.has_air_density = true;
                    wk.air_density = np.air_density;
                    break;
                case WorldProp::DustDensity:
                    wk.has_dust_density = true;
                    wk.dust_density = np.dust_density;
                    break;
                case WorldProp::OzoneDensity:
                    wk.has_ozone_density = true;
                    wk.ozone_density = np.ozone_density;
                    break;
                case WorldProp::Altitude:
                    wk.has_altitude = true;
                    wk.altitude = np.altitude;
                    break;
                case WorldProp::MieAnisotropy:
                    wk.has_mie_anisotropy = true;
                    wk.mie_anisotropy = np.mie_anisotropy;
                    break;
                case WorldProp::CloudDensity:
                    wk.has_cloud_density = true;
                    wk.cloud_density = np.cloud_density;
                    break;
                case WorldProp::CloudCoverage:
                    wk.has_cloud_coverage = true;
                    wk.cloud_coverage = np.cloud_coverage;
                    break;
                case WorldProp::CloudScale:
                    wk.has_cloud_scale = true;
                    wk.cloud_scale = np.cloud_scale;
                    break;
                case WorldProp::CloudOffset:
                    wk.has_cloud_offset = true;
                    wk.cloud_offset_x = np.cloud_offset_x;
                    wk.cloud_offset_z = np.cloud_offset_z;
                    break;
            }
            
            ctx.scene.timeline.insertKeyframe("World", kf);
        }
    };
    
    // ═══════════════════════════════════════════════════════════
    // Sky Model Selection
    // ═══════════════════════════════════════════════════════════
    if (UIWidgets::BeginSection("Sky Model", ImVec4(0.4f, 0.6f, 0.9f, 1.0f))) {
        const char* modes[] = { "Solid Color", "HDRI Environment", "Raytrophi Spectral Sky" };
        int mode_idx = static_cast<int>(current_mode);
        
        ImGui::PushItemWidth(-1);
        if (ImGui::Combo("##SkyModel", &mode_idx, modes, IM_ARRAYSIZE(modes))) {
            world.setMode(static_cast<WorldMode>(mode_idx));
            changed = true;
        }
        ImGui::PopItemWidth();
        UIWidgets::EndSection();
    }
    
    ImGui::Spacing();
    
    // ═══════════════════════════════════════════════════════════
    // Solid Color Mode
    // ═══════════════════════════════════════════════════════════
    if (current_mode == WORLD_MODE_COLOR) {
        if (UIWidgets::BeginSection("Background", ImVec4(0.6f, 0.4f, 0.8f, 1.0f))) {
            // Color
            Vec3 color = world.getColor();
            bool bgKeyed = isWorldKeyed(WorldProp::BackgroundColor);
            if (KeyframeButton("##WBgCol", bgKeyed, "Color")) { insertWorldKey("BG Color", WorldProp::BackgroundColor); }
            ImGui::SameLine();
            if (ImGui::ColorEdit3("Color", &color.x)) {
                world.setColor(color);
                ctx.scene.background_color = color;
                changed = true;
            }
            
            // Intensity
            float intensity = world.getColorIntensity();
            if (ImGui::SliderFloat("Intensity", &intensity, 0.0f, 10.0f)) {
                world.setColorIntensity(intensity);
                changed = true;
            }
            UIWidgets::EndSection();
        }
    }
    // ═══════════════════════════════════════════════════════════
    // HDRI Environment Mode
    // ═══════════════════════════════════════════════════════════
    else if (current_mode == WORLD_MODE_HDRI) {
        if (UIWidgets::BeginSection("HDRI Map", ImVec4(0.2f, 0.7f, 0.5f, 1.0f))) {
            // Load Button
            if (UIWidgets::PrimaryButton("Load Environment", ImVec2(-1, 0))) {
#ifdef _WIN32
                std::string file = openFileDialogW(L"Environment Maps\0*.hdr;*.exr;*.jpg;*.jpeg;*.png\0HDR/EXR\0*.hdr;*.exr\0All Files\0*.*\0");
                if (!file.empty()) {
                    world.setHDRI(file);
                    changed = true;
                    addViewportMessage("HDRI Loaded: " + file.substr(file.find_last_of("/\\") + 1), 3.0f);
                }
#endif
            }
            
            // Current file display
            std::string path = world.getHDRIPath();
            if (!path.empty()) {
                size_t lastSlash = path.find_last_of("/\\");
                std::string filename = (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "%s", filename.c_str());
            } else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "(No HDRI loaded)");
            }
            UIWidgets::EndSection();
        }
        
        if (UIWidgets::BeginSection("Transform", ImVec4(0.5f, 0.6f, 0.8f, 1.0f))) {
            // Rotation
            float rotation = world.getHDRIRotation();
            bool rotKeyed = isWorldKeyed(WorldProp::HDRIRotation);
            if (KeyframeButton("##WRot", rotKeyed, "Rotation")) { insertWorldKey("HDRI Rot", WorldProp::HDRIRotation); }
            ImGui::SameLine();
            if (ImGui::SliderFloat("Rotation", &rotation, 0.0f, 360.0f, "%.1f deg")) {
                world.setHDRIRotation(rotation);
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Y-axis rotation (0-360)");
            
            // Intensity
            float intensity = world.getHDRIIntensity();
            if (ImGui::SliderFloat("Intensity", &intensity, 0.0f, 10.0f, "%.2f")) {
                world.setHDRIIntensity(intensity);
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Brightness multiplier");
            UIWidgets::EndSection();
        }
    }
    // ═══════════════════════════════════════════════════════════
    // “Raytrophi Spectral Sky Mode
    // ═══════════════════════════════════════════════════════════
    else if (current_mode == WORLD_MODE_NISHITA) {
        NishitaSkyParams params = world.getNishitaParams();
        // static bool syncWithDirectionalLight replaced by member
        
        // Sync with Directional Light Section
        if (UIWidgets::BeginSection("Light Sync", ImVec4(0.6f, 0.8f, 1.0f, 1.0f))) {
            if (ImGui::Checkbox("Sync with Scene Light", &sync_sun_with_light)) {
                // Checkbox toggled
            }
            UIWidgets::HelpMarker("Automatically sync sun direction with the first directional light in the scene");
            

            
            if (sync_sun_with_light) {
                // Find first directional light
                bool foundDirLight = false;
                for (const auto& light : ctx.scene.lights) {
                    if (light && light->type() == LightType::Directional) {
                        // Convert light direction to sun direction
                        // Note: light->direction is the direction light TRAVELS (sun to ground)
                        // We need direction TO the sun, so negate it
                        Vec3 dir = -(light->direction.normalize());
                        
                        // Elevation: angle from horizon (Y component)
                        float elevation = asinf(dir.y) * 180.0f / M_PI;
                        
                        // Azimuth: horizontal angle (XZ plane)
                        float azimuth = atan2f(dir.x, dir.z) * 180.0f / M_PI;
                        if (azimuth < 0) azimuth += 360.0f;
                        
                        // Update params if different
                        if (fabsf(params.sun_elevation - elevation) > 0.1f || 
                            fabsf(params.sun_azimuth - azimuth) > 0.1f) {
                            params.sun_elevation = elevation;
                            params.sun_azimuth = azimuth;
                            changed = true;
                        }
                        
                        foundDirLight = true;
                        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Synced with Directional Light");
                        break;
                    }
                }
                
                if (!foundDirLight) {
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "No directional light found");
                }
            }
            UIWidgets::EndSection();
        }
        
        // Sun Position Section (controls both sun and directional light when synced)
        const char* sunPosTitle = sync_sun_with_light ? "Sun/Light Position" : "Sun Position";
        if (UIWidgets::BeginSection(sunPosTitle, ImVec4(1.0f, 0.7f, 0.3f, 1.0f))) {
            // Elevation
            bool sunKeyed = isWorldKeyed(WorldProp::SunElevation);
            if (KeyframeButton("##WSunEl", sunKeyed, "Elevation")) { insertWorldKey("Sun Elev", WorldProp::SunElevation); }
            ImGui::SameLine();
            if (ImGui::SliderFloat("Elevation", &params.sun_elevation, -10.0f, 90.0f, "%.1f deg")) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker(sync_sun_with_light ? 
                "Sun/Light height above horizon - controls both sky sun and directional light" :
                "Sun height above horizon (0 = horizon, 90 = zenith)");
            
            // Azimuth
            bool azKeyed = isWorldKeyed(WorldProp::SunAzimuth);
            if (KeyframeButton("##WSunAz", azKeyed, "Azimuth")) { insertWorldKey("Sun Azimuth", WorldProp::SunAzimuth); }
            ImGui::SameLine();
            if (ImGui::SliderFloat("Azimuth", &params.sun_azimuth, 0.0f, 360.0f, "%.1f deg")) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker(sync_sun_with_light ?
                "Sun/Light horizontal rotation - controls both sky sun and directional light" :
                "Sun horizontal rotation (compass direction)");
            UIWidgets::EndSection();
        }
        
        // Sun Appearance Section
        if (UIWidgets::BeginSection("Sun", ImVec4(1.0f, 0.5f, 0.3f, 1.0f))) {
            // Intensity
            bool intKeyed = isWorldKeyed(WorldProp::SunIntensity);
            if (KeyframeButton("##WSunInt", intKeyed, "Intensity")) { insertWorldKey("Sun Intensity", WorldProp::SunIntensity); }
            ImGui::SameLine();
            if (ImGui::SliderFloat("Intensity", &params.sun_intensity, 0.0f, 100.0f, "%.1f")) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Brightness of the sun");
            
            // Calculate automatic sun size based on elevation (atmospheric magnification)
            float elevationFactor = 1.0f;
            if (params.sun_elevation < 15.0f) {
                elevationFactor = 1.0f + (15.0f - fmaxf(params.sun_elevation, -10.0f)) * 0.04f;
            }
            float displaySize = params.sun_size * elevationFactor;
            
            // Size
            bool sizeKeyed = isWorldKeyed(WorldProp::SunSize);
            if (KeyframeButton("##WSunSz", sizeKeyed, "Size")) { insertWorldKey("Sun Size", WorldProp::SunSize); }
            ImGui::SameLine();
            if (ImGui::SliderFloat("Size", &params.sun_size, 0.1f, 5.0f, "%.3f deg")) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Angular diameter of the sun disc (real sun = 0.545 deg)");
            if (elevationFactor > 1.01f) {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.5f, 1.0f), "Effective: %.3f deg (horizon boost)", displaySize);
            }
            UIWidgets::EndSection();
        }
        
        // Atmosphere Section (Blender-style parameters)
        if (UIWidgets::BeginSection("Atmosphere", ImVec4(0.4f, 0.7f, 1.0f, 1.0f))) {
            // Air
            bool airKeyed = isWorldKeyed(WorldProp::AirDensity);
            if (KeyframeButton("##WAir", airKeyed, "Air")) { insertWorldKey("Air", WorldProp::AirDensity); }
            ImGui::SameLine();
            if (ImGui::SliderFloat("Air", &params.air_density, 0.0f, 10.0f, "%.2f")) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Rayleigh scattering density");
            
            // Dust
            bool dustKeyed = isWorldKeyed(WorldProp::DustDensity);
            if (KeyframeButton("##WDust", dustKeyed, "Dust")) { insertWorldKey("Dust", WorldProp::DustDensity); }
            ImGui::SameLine();
            if (ImGui::SliderFloat("Dust", &params.dust_density, 0.0f, 10.0f, "%.2f")) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Mie scattering (haze/sun glow)");
            
            // Ozone
            bool ozoKeyed = isWorldKeyed(WorldProp::OzoneDensity);
            if (KeyframeButton("##WOzn", ozoKeyed, "Ozone")) { insertWorldKey("Ozone", WorldProp::OzoneDensity); }
            ImGui::SameLine();
            if (ImGui::SliderFloat("Ozone", &params.ozone_density, 0.0f, 10.0f, "%.2f")) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Ozone layer density");
            
            // Altitude
            float altitudeKm = params.altitude / 1000.0f;
            bool altKeyed = isWorldKeyed(WorldProp::Altitude);
            if (KeyframeButton("##WAlt", altKeyed, "Altitude")) { insertWorldKey("Altitude", WorldProp::Altitude); }
            ImGui::SameLine();
            if (ImGui::SliderFloat("Altitude", &altitudeKm, 0.0f, 60.0f, "%.1f km")) {
                params.altitude = altitudeKm * 1000.0f;
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Camera altitude above sea level");
            UIWidgets::EndSection();
        }
        
        // Advanced Physics (Collapsible)
        if (ImGui::CollapsingHeader("Advanced Physics")) {
            ImGui::Indent();
            
            // Mie Anisotropy
            bool mieKeyed = isWorldKeyed(WorldProp::MieAnisotropy);
            if (KeyframeButton("##WMie", mieKeyed, "Mie")) { insertWorldKey("Mie", WorldProp::MieAnisotropy); }
            ImGui::SameLine();
            if (ImGui::SliderFloat("Mie Anisotropy", &params.mie_anisotropy, 0.0f, 0.99f, "%.2f")) {
                changed = true;
            }
            UIWidgets::HelpMarker("Sun glow directionality (0 = uniform, 0.8+ = strong forward scatter)");
            
            ImGui::Unindent();
        }
        
        // ═══════════════════════════════════════════════════════════
        // ENVIRONMENT TEXTURE OVERLAY (HDR/EXR blending with procedural)
        // ═══════════════════════════════════════════════════════════
        if (UIWidgets::BeginSection("Environment Overlay", ImVec4(0.5f, 0.6f, 0.9f, 1.0f))) {
            bool envEnabled = params.env_overlay_enabled != 0;
            if (ImGui::Checkbox("Enable Overlay", &envEnabled)) {
                params.env_overlay_enabled = envEnabled ? 1 : 0;
                changed = true;
            }
            UIWidgets::HelpMarker("Blend an HDR/EXR environment texture with the procedural Nishita sky");
            
            if (params.env_overlay_enabled) {
                ImGui::Separator();
                
                // Load Environment Texture Button
                if (UIWidgets::PrimaryButton("Load Environment Map", ImVec2(-1, 0))) {
#ifdef _WIN32
                    std::string file = openFileDialogW(L"Environment Maps\0*.hdr;*.exr;*.jpg;*.jpeg;*.png\0HDR/EXR\0*.hdr;*.exr\0All Files\0*.*\0");
                    if (!file.empty()) {
                        // Load texture and set env_overlay_tex
                        world.setNishitaEnvOverlay(file);
                        changed = true;
                        addViewportMessage("Atmosphere Overlay Loaded", 3.0f);
                    }
#endif
                }
                
                // Current overlay file display
                std::string overlayPath = world.getNishitaEnvOverlayPath();
                if (!overlayPath.empty()) {
                    size_t lastSlash = overlayPath.find_last_of("/\\");
                    std::string filename = (lastSlash != std::string::npos) ? overlayPath.substr(lastSlash + 1) : overlayPath;
                    ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "%s", filename.c_str());
                } else {
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "(No overlay texture)");
                }
                
                ImGui::Spacing();
                
                // Blend Mode
                const char* blendModes[] = { "Mix", "Multiply", "Screen", "Replace" };
                if (ImGui::Combo("Blend Mode", &params.env_overlay_blend_mode, blendModes, IM_ARRAYSIZE(blendModes))) {
                    changed = true;
                }
                UIWidgets::HelpMarker(
                    "Mix: Blend between Nishita and texture\n"
                    "Multiply: Use texture for color grading\n"
                    "Screen: Brightens without washing out\n"
                    "Replace: Use ONLY the texture");
                
                // Intensity
                if (ImGui::SliderFloat("Overlay Intensity", &params.env_overlay_intensity, 0.0f, 3.0f, "%.2f")) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Strength of the overlay texture");
                
                // Rotation
                if (ImGui::SliderFloat("Overlay Rotation", &params.env_overlay_rotation, 0.0f, 360.0f, "%.1f deg")) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Rotate the overlay texture around Y axis");
            }
            UIWidgets::EndSection();
        }
        // ═══════════════════════════════════════════════════════════
        // ATMOSPHERIC EFFECTS (Fog, God Rays, Multi-Scattering)
        // ═══════════════════════════════════════════════════════════
        if (UIWidgets::BeginSection("Atmospheric Effects", ImVec4(0.6f, 0.7f, 0.9f, 1.0f))) {
            
            // --- ATMOSPHERIC FOG ---
            ImGui::TextColored(ImVec4(0.7f, 0.8f, 1.0f, 1.0f), "Atmospheric Fog:");
            
            bool fogEnabled = params.fog_enabled != 0;
            if (ImGui::Checkbox("Enable Fog", &fogEnabled)) {
                params.fog_enabled = fogEnabled ? 1 : 0;
                changed = true;
            }
            UIWidgets::HelpMarker("Height-based exponential fog with sun scattering");
            
            if (params.fog_enabled) {
                ImGui::Indent();
                
                if (ImGui::SliderFloat("Fog Density", &params.fog_density, 0.001f, 0.1f, "%.4f")) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Base fog density (lower = lighter fog)");
                
                if (ImGui::SliderFloat("Fog Height", &params.fog_height, 10.0f, 2000.0f, "%.0f m")) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Fog is concentrated below this height");
                
                if (ImGui::SliderFloat("Fog Falloff", &params.fog_falloff, 0.0005f, 0.02f, "%.4f")) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Rate at which fog decreases with height");
                
                float fogDistKm = params.fog_distance / 1000.0f;
                if (ImGui::SliderFloat("Fog Distance", &fogDistKm, 0.1f, 50.0f, "%.1f km")) {
                    params.fog_distance = fogDistKm * 1000.0f;
                    changed = true;
                }
                UIWidgets::HelpMarker("Maximum distance for fog effect");
                
                float fogColor[3] = { params.fog_color.x, params.fog_color.y, params.fog_color.z };
                if (ImGui::ColorEdit3("Fog Color", fogColor)) {
                    params.fog_color = make_float3(fogColor[0], fogColor[1], fogColor[2]);
                    changed = true;
                }
                
                if (ImGui::SliderFloat("Sun Scatter", &params.fog_sun_scatter, 0.0f, 2.0f, "%.2f")) {
                    changed = true;
                }
                UIWidgets::HelpMarker("How much fog glows when looking towards sun");
                
                ImGui::Unindent();
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // --- GOD RAYS ---
            ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.5f, 1.0f), "Volumetric Light Rays:");
            
            bool godraysEnabled = params.godrays_enabled != 0;
            if (ImGui::Checkbox("Enable God Rays", &godraysEnabled)) {
                params.godrays_enabled = godraysEnabled ? 1 : 0;
                changed = true;
            }
            UIWidgets::HelpMarker("Volumetric light shafts near the sun");
            
            if (params.godrays_enabled) {
                ImGui::Indent();
                
                if (ImGui::SliderFloat("Ray Intensity", &params.godrays_intensity, 0.0f, 3.0f, "%.2f")) {
                    changed = true;
                }
                
                if (ImGui::SliderFloat("Ray Density", &params.godrays_density, 0.01f, 1.0f, "%.2f")) {
                    changed = true;
                }
                
                if (ImGui::SliderFloat("Ray Decay", &params.godrays_decay, 0.8f, 1.0f, "%.2f")) {
                    changed = true;
                }
                UIWidgets::HelpMarker("How quickly rays fade with distance from sun");
                
                ImGui::Unindent();
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // --- MULTI-SCATTERING ---
            ImGui::TextColored(ImVec4(0.5f, 0.9f, 1.0f, 1.0f), "Multi-Scattering:");
            
            bool msEnabled = params.multi_scatter_enabled != 0;
            if (ImGui::Checkbox("Enable Multi-Scattering", &msEnabled)) {
                params.multi_scatter_enabled = msEnabled ? 1 : 0;
                changed = true;
            }
            UIWidgets::HelpMarker("Simulates multiple light bounces in atmosphere.\nBrightens horizon and makes sky more uniform.");
            
            if (params.multi_scatter_enabled) {
                ImGui::Indent();
                
                if (ImGui::SliderFloat("MS Intensity", &params.multi_scatter_factor, 0.0f, 1.0f, "%.2f")) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Strength of multi-scattering contribution");
                
                ImGui::Unindent();
            }
            
            UIWidgets::EndSection();
        }
        
        // Update sun direction from elevation/azimuth
        if (changed) {
            float elevationRad = params.sun_elevation * M_PI / 180.0f;
            float azimuthRad = params.sun_azimuth * M_PI / 180.0f;
            params.sun_direction = make_float3(
                cosf(elevationRad) * sinf(azimuthRad),
                sinf(elevationRad),
                cosf(elevationRad) * cosf(azimuthRad)
            );
            world.setNishitaParams(params);
            
            // If sync is enabled, also update the directional light
            if (sync_sun_with_light) {
                for (auto& light : ctx.scene.lights) {
                    if (light && light->type() == LightType::Directional) {
                        // Light direction is opposite of sun direction (sun to ground)
                        light->direction = Vec3(
                            -params.sun_direction.x,
                            -params.sun_direction.y,
                            -params.sun_direction.z
                        );
                        
                        // CRITICAL: Mark lights dirty so GPU gets updated light direction
                        extern bool g_lights_dirty;
                        g_lights_dirty = true;
                        break;
                    }
                }
            }
        }

    }
    
    // ═══════════════════════════════════════════════════════════
    // Global Atmosphere (Clouds)
    // ═══════════════════════════════════════════════════════════
    ImGui::Spacing();
    if (UIWidgets::BeginSection("Global Atmosphere (Clouds)", ImVec4(0.5f, 0.6f, 0.7f, 1.0f))) {
        NishitaSkyParams cloudParams = world.getNishitaParams();
        bool cloudParamsChanged = false;

        bool cloudsEnabled = cloudParams.clouds_enabled != 0;
        if (ImGui::Checkbox("Enable Volumetric Clouds", &cloudsEnabled)) {
            cloudParams.clouds_enabled = cloudsEnabled ? 1 : 0;
            cloudParamsChanged = true;
        }
        UIWidgets::HelpMarker("Render volumetric clouds over any sky model.\nNote: High Performance Cost!");

        if (cloudParams.clouds_enabled) {
            ImGui::Spacing();
            
            // === CLOUD QUALITY (Performance) ===
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Quality:");
            static int quality_preset = 0; // Normal
            const char* quality_presets[] = { "Fast Preview", "Low", "Normal", "High", "Ultra" };
            float quality_values[] = { 0.25f, 0.5f, 1.0f, 1.5f, 2.5f };
            
            ImGui::PushItemWidth(120);
            if (ImGui::Combo("##CloudQuality", &quality_preset, quality_presets, IM_ARRAYSIZE(quality_presets))) {
                cloudParams.cloud_quality = quality_values[quality_preset];
                cloudParamsChanged = true;
            }
            ImGui::PopItemWidth();
            ImGui::SameLine();
            ImGui::TextDisabled("(%.0f-%.0f steps)", 
                48.0f * cloudParams.cloud_quality, 
                96.0f * cloudParams.cloud_quality);
            UIWidgets::HelpMarker("Lower quality = faster render\nHigher quality = better details");
            
            ImGui::Spacing();
            
            // === WEATHER PRESETS (Combo Box) ===
            static int weather_preset_index = 7; // Default to Custom
            const char* weather_presets[] = { 
                "Clear Sky", 
                "Cirrus (Wispy)",
                "Cumulus (Puffy)",
                "Stratocumulus",
                "Overcast", 
                "Cumulonimbus (Storm)",
                "Fog/Low Clouds",
                "Custom" 
            };
            
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Weather Type:");
            ImGui::PushItemWidth(-1);
            if (ImGui::Combo("##WeatherType", &weather_preset_index, weather_presets, IM_ARRAYSIZE(weather_presets))) {
                switch (weather_preset_index) {
                    case 0: // Clear Sky
                        cloudParams.clouds_enabled = 0;
                        cloudParamsChanged = true;
                        break;
                    case 1: // Cirrus (Wispy high clouds) - VERY THIN layer
                        cloudParams.cloud_height_min = 8000.0f;   // 8km - high altitude
                        cloudParams.cloud_height_max = 8500.0f;   // Only 500m thick!
                        cloudParams.cloud_scale = 15.0f;          // Very stretched/wispy
                        cloudParams.cloud_coverage = 0.35f;       // Sparse
                        cloudParams.cloud_density = 0.15f;        // Very transparent
                        cloudParamsChanged = true;
                        break;
                    case 2: // Cumulus (Classic puffy clouds)
                        cloudParams.cloud_height_min = 1000.0f;
                        cloudParams.cloud_height_max = 3000.0f;
                        cloudParams.cloud_scale = 3.0f;
                        cloudParams.cloud_coverage = 0.4f;
                        cloudParams.cloud_density = 1.5f;
                        cloudParamsChanged = true;
                        break;
                    case 3: // Stratocumulus (Layered puffy)
                        cloudParams.cloud_height_min = 500.0f;
                        cloudParams.cloud_height_max = 2000.0f;
                        cloudParams.cloud_scale = 2.0f;
                        cloudParams.cloud_coverage = 0.88f;
                        cloudParams.cloud_density = 1.2f;
                        cloudParamsChanged = true;
                        break;
                    case 4: // Overcast (Stratus)
                        cloudParams.cloud_height_min = 200.0f;
                        cloudParams.cloud_height_max = 600.0f;
                        cloudParams.cloud_scale = 0.5f;
                        cloudParams.cloud_coverage = 0.9f;
                        cloudParams.cloud_density = 2.5f;
                        cloudParamsChanged = true;
                        break;
                    case 5: // Cumulonimbus (Storm/Rain clouds)
                        cloudParams.cloud_height_min = 500.0f;
                        cloudParams.cloud_height_max = 10000.0f;
                        cloudParams.cloud_scale = 5.0f;
                        cloudParams.cloud_coverage = 0.9f;
                        cloudParams.cloud_density = 4.0f;
                        cloudParamsChanged = true;
                        break;
                    case 6: // Fog/Low Clouds
                        cloudParams.cloud_height_min = 0.0f;
                        cloudParams.cloud_height_max = 200.0f;
                        cloudParams.cloud_scale = 0.3f;
                        cloudParams.cloud_coverage = 0.9f;
                        cloudParams.cloud_density = 3.0f;
                        cloudParamsChanged = true;
                        break;
                    case 7: // Custom
                        break;
                }
            }
            ImGui::PopItemWidth();
            UIWidgets::HelpMarker("Cloud type presets:\n"
                "- Cirrus: High wispy clouds\n"
                "- Cumulus: Classic puffy clouds\n"
                "- Stratocumulus: Layered clouds\n"
                "- Overcast: Gray flat sky\n"
                "- Cumulonimbus: Towering storm clouds\n"
                "- Fog: Ground-level clouds");
            
            ImGui::Spacing();
            ImGui::Separator();
            
            // Show detailed controls in a collapsible section or when Custom is selected
            bool show_details = (weather_preset_index == 7); // Custom mode
            
            if (show_details || ImGui::CollapsingHeader("Cloud Details")) {
                // Coverage
                bool covKeyed = isWorldKeyed(WorldProp::CloudCoverage);
                if (KeyframeButton("##WCov", covKeyed, "Coverage")) { 
                    insertWorldKey("Cloud Cov", WorldProp::CloudCoverage); 
                }
                ImGui::SameLine();
                if (ImGui::SliderFloat("Coverage", &cloudParams.cloud_coverage, 0.0f, 1.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    weather_preset_index = 7;
                }
                ImGui::SameLine(); UIWidgets::HelpMarker("0 = clear, 1 = overcast");

                // Density
                bool denKeyed = isWorldKeyed(WorldProp::CloudDensity);
                if (KeyframeButton("##WCldDen", denKeyed, "Density")) { 
                    insertWorldKey("Cloud Den", WorldProp::CloudDensity); 
                }
                ImGui::SameLine();
                if (ImGui::DragFloat("Density", &cloudParams.cloud_density, 0.05f, 0.0f, 10.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    weather_preset_index = 7;
                }
                ImGui::SameLine(); UIWidgets::HelpMarker("Cloud opacity");
                
                // Scale
                bool sclKeyed = isWorldKeyed(WorldProp::CloudScale);
                if (KeyframeButton("##WScl", sclKeyed, "Scale")) { 
                    insertWorldKey("Cloud Scale", WorldProp::CloudScale); 
                }
                ImGui::SameLine();
                if (ImGui::DragFloat("Scale", &cloudParams.cloud_scale, 0.1f, 0.1f, 100.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    weather_preset_index = 7;
                }
                ImGui::SameLine(); UIWidgets::HelpMarker("Cloud size");

                // Wind / Seed Offsets
                bool offKeyed = isWorldKeyed(WorldProp::CloudOffset);
                if (KeyframeButton("##WOffX", offKeyed, "Offset X")) { 
                    insertWorldKey("Cloud OffX", WorldProp::CloudOffset); 
                }
                ImGui::SameLine();
                if (ImGui::DragFloat("Offset X", &cloudParams.cloud_offset_x, 10.0f, -100000.0f, 100000.0f, "%.0f m")) {
                    cloudParamsChanged = true;
                }
                
                if (KeyframeButton("##WOffZ", offKeyed, "Offset Z")) { 
                    insertWorldKey("Cloud OffZ", WorldProp::CloudOffset); 
                }
                ImGui::SameLine();
                if (ImGui::DragFloat("Offset Z", &cloudParams.cloud_offset_z, 10.0f, -100000.0f, 100000.0f, "%.0f m")) {
                    cloudParamsChanged = true;
                }
                ImGui::SameLine(); UIWidgets::HelpMarker("Wind animation");
            }
            
            // ═══════════════════════════════════════════════════════════
            // CLOUD LIGHTING SETTINGS
            // ═══════════════════════════════════════════════════════════
            if (ImGui::CollapsingHeader("Cloud Lighting")) {
                ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.5f, 1.0f), "Self-Shadowing:");
                
                // Light Steps (0 = disabled for performance)
                if (ImGui::SliderInt("Light Steps", &cloudParams.cloud_light_steps, 0, 12)) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Number of light marching steps.\n0 = Disabled (fast)\n4-6 = Normal\n8-12 = High quality");
                
                // Shadow Strength
                if (ImGui::SliderFloat("Shadow Strength", &cloudParams.cloud_shadow_strength, 0.0f, 2.0f, "%.2f")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Cloud self-shadowing intensity.\n0 = No shadows\n1 = Normal\n2 = Dark, dramatic shadows");
                
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.5f, 1.0f), "Lighting Effects:");
                
                // Silver Lining
                if (ImGui::SliderFloat("Silver Lining", &cloudParams.cloud_silver_intensity, 0.0f, 3.0f, "%.2f")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Bright rim effect when backlit by sun.\n0 = Off\n1 = Normal\n2+ = Strong glow");
                
                // Ambient Strength
                if (ImGui::SliderFloat("Ambient Light", &cloudParams.cloud_ambient_strength, 0.0f, 2.0f, "%.2f")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Sky ambient light contribution.\nLower = Darker shadows\nHigher = Softer look");
                
                // Absorption
                if (ImGui::SliderFloat("Absorption", &cloudParams.cloud_absorption, 0.2f, 3.0f, "%.2f")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Light absorption rate.\n0.5 = Thin, transparent\n1.0 = Normal\n2.0 = Thick, opaque");
            }

            // === LAYER 1 ALTITUDE ===
            if (ImGui::TreeNode("Layer 1 Altitude")) {
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "Layer 1: %.0f - %.0f m (%.0f m thick)", 
                    cloudParams.cloud_height_min, cloudParams.cloud_height_max,
                    cloudParams.cloud_height_max - cloudParams.cloud_height_min);
                
                if (ImGui::DragFloat("Min Height##L1", &cloudParams.cloud_height_min, 10.0f, 0.0f, 20000.0f, "%.0f m")) {
                    cloudParamsChanged = true;
                }
                if (ImGui::DragFloat("Max Height##L1", &cloudParams.cloud_height_max, 10.0f, 0.0f, 20000.0f, "%.0f m")) {
                    cloudParamsChanged = true;
                }
                ImGui::TreePop();
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            
            // ==================================================================
            // CLOUD LAYER 2 (Secondary/High Altitude)
            // ==================================================================
            bool layer2Enabled = cloudParams.cloud_layer2_enabled != 0;
            if (ImGui::Checkbox("Enable Cloud Layer 2", &layer2Enabled)) {
                cloudParams.cloud_layer2_enabled = layer2Enabled ? 1 : 0;
                cloudParamsChanged = true;
            }
            UIWidgets::HelpMarker("Enable a second cloud layer.\nUseful for high cirrus above low cumulus.");
            
            if (cloudParams.cloud_layer2_enabled) {
                ImGui::Indent();
                ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1.0f), "Layer 2 Settings:");
                
                // Layer 2 presets
                static int layer2_preset = 0;
                const char* layer2_presets[] = { "Custom", "High Cirrus", "Mid Stratus", "Low Fog" };
                if (ImGui::Combo("Layer 2 Preset", &layer2_preset, layer2_presets, IM_ARRAYSIZE(layer2_presets))) {
                    switch (layer2_preset) {
                        case 1: // High Cirrus
                            cloudParams.cloud2_height_min = 8000.0f;
                            cloudParams.cloud2_height_max = 9000.0f;
                            cloudParams.cloud2_scale = 12.0f;
                            cloudParams.cloud2_coverage = 0.25f;
                            cloudParams.cloud2_density = 0.2f;
                            cloudParamsChanged = true;
                            break;
                        case 2: // Mid Stratus
                            cloudParams.cloud2_height_min = 2500.0f;
                            cloudParams.cloud2_height_max = 3500.0f;
                            cloudParams.cloud2_scale = 3.0f;
                            cloudParams.cloud2_coverage = 0.5f;
                            cloudParams.cloud2_density = 0.8f;
                            cloudParamsChanged = true;
                            break;
                        case 3: // Low Fog
                            cloudParams.cloud2_height_min = 0.0f;
                            cloudParams.cloud2_height_max = 150.0f;
                            cloudParams.cloud2_scale = 0.5f;
                            cloudParams.cloud2_coverage = 0.7f;
                            cloudParams.cloud2_density = 2.0f;
                            cloudParamsChanged = true;
                            break;
                    }
                }
                
                if (ImGui::SliderFloat("Coverage##L2", &cloudParams.cloud2_coverage, 0.0f, 1.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                if (ImGui::DragFloat("Density##L2", &cloudParams.cloud2_density, 0.05f, 0.0f, 5.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                if (ImGui::DragFloat("Scale##L2", &cloudParams.cloud2_scale, 0.1f, 0.1f, 50.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                
                ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1.0f), "Layer 2: %.0f - %.0f m", 
                    cloudParams.cloud2_height_min, cloudParams.cloud2_height_max);
                if (ImGui::DragFloat("Min Height##L2", &cloudParams.cloud2_height_min, 50.0f, 0.0f, 20000.0f, "%.0f m")) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                if (ImGui::DragFloat("Max Height##L2", &cloudParams.cloud2_height_max, 50.0f, 0.0f, 20000.0f, "%.0f m")) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                
                ImGui::Unindent();
            }
            
            ImGui::Spacing();
            
            // Camera altitude info
            if (ImGui::TreeNode("Camera Altitude Info")) {
                ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.5f, 1.0f), "Camera Y: %.0f m", cloudParams.altitude);
                
                if (cloudParams.altitude >= cloudParams.cloud_height_min && 
                    cloudParams.altitude <= cloudParams.cloud_height_max) {
                    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Camera is INSIDE Layer 1!");
                }
              
                
                ImGui::TreePop();
            }
        }
        UIWidgets::EndSection();

        if (cloudParamsChanged) {
            world.setNishitaParams(cloudParams);
            changed = true;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Apply Changes
    // ═══════════════════════════════════════════════════════════
    if (changed) {
        world_params_changed_this_frame = true;
        ctx.renderer.resetCPUAccumulation();
        
        // OPTIMIZATION: Only update OptiX when OptiX rendering is enabled
        if (ctx.render_settings.use_optix && ctx.optix_gpu_ptr) {
            ctx.optix_gpu_ptr->setWorld(world.getGPUData());
            ctx.optix_gpu_ptr->resetAccumulation();
        }
        
        // Mark world buffer dirty for main loop (in case direct update is skipped)
        extern bool g_world_dirty;
        g_world_dirty = true;
    }
    
    // Pop the global item width set at function start
    ImGui::PopItemWidth();
}

// Global Sun Synchronization Logic
void SceneUI::processSunSync(UIContext& ctx) {
    if (!sync_sun_with_light) return;
    if (world_params_changed_this_frame) return; // User is modifying sliders, skip sync to avoid fighting

    World& world = ctx.renderer.world;
    if (world.getMode() != WORLD_MODE_NISHITA) return;

    // Reverse Sync: Directional Light -> Nishita Params
    for (const auto& light : ctx.scene.lights) {
        if (light && light->type() == LightType::Directional) {
             Vec3 lightDir = light->direction.normalize();
             Vec3 sunDir = -lightDir; 
             
             float elevRad = asinf(sunDir.y);
             float elevDeg = elevRad * 180.0f / 3.14159265f;
             
             float azimRad = atan2f(sunDir.x, sunDir.z); 
             float azimDeg = azimRad * 180.0f / 3.14159265f;
             if (azimDeg < 0.0f) azimDeg += 360.0f;
             
             NishitaSkyParams params = world.getNishitaParams();
             
             // Update only if significantly different
             if (fabsf(params.sun_elevation - elevDeg) > 0.1f || fabsf(params.sun_azimuth - azimDeg) > 0.1f) {
                 params.sun_elevation = elevDeg;
                 params.sun_azimuth = azimDeg;
                 params.sun_direction = make_float3(sunDir.x, sunDir.y, sunDir.z);
                 world.setNishitaParams(params);
                 
                 // Mark world dirty so GPU gets updated sky direction
                 extern bool g_world_dirty;
                 g_world_dirty = true;
             }
             break; // Sync with first directional light
        }
    }
}
