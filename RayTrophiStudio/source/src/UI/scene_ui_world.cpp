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
#include "Api/RtApi.h"
#include <algorithm>
#include <cmath>

namespace {
bool timelineHasAnimatedWorldSun(const TimelineManager& timeline) {
    auto it = timeline.tracks.find("World");
    if (it == timeline.tracks.end()) return false;

    for (const auto& kf : it->second.keyframes) {
        if (!kf.has_world) continue;
        const WorldKeyframe& world = kf.world;
        if (world.has_sun_elevation ||
            world.has_sun_azimuth ||
            world.has_sun_intensity ||
            world.has_sun_size) {
            return true;
        }
    }

    return false;
}
}

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>

// Forward declaration for file dialog (defined in scene_ui.cpp)
extern std::string openFileDialogW(const wchar_t* filter, const std::string& initialDir = "", const std::string& defaultFilename = "");
#endif

void SceneUI::drawWorldContent(UIContext& ctx) {
    World& world = ctx.renderer.world;
    WorldMode current_mode = world.getMode();
    
    // Auto-select the "World" track ONLY while the World panel is focused, so the
    // user can move/delete/add world keyframes from here. Doing this every frame the
    // panel is merely drawn (docked/visible but not focused) permanently hijacked
    // selected_track to "World" — which silently blocked keying objects for the rest
    // of the session, because handleSelectionSync only restores it on a selection
    // CHANGE. On focus release we force a selection re-sync to hand the timeline back.
    {
        const bool world_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
        static bool prev_world_focused = false;
        if (world_focused) {
            timeline.selected_track = "World";
        } else if (prev_world_focused) {
            timeline.invalidateSelectionSync();  // released: restore live selection's track
        }
        prev_world_focused = world_focused;
    }
    
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
        Climate, OzoneStrength,
        FogParams, GodRaysParams,
        Clouds,
        MultiScatterFactor, WeatherParams
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
                    case WorldProp::Clouds: return kf.world.has_clouds;
                    case WorldProp::Climate: return kf.world.has_climate;
                    case WorldProp::OzoneStrength: return kf.world.has_ozone_absorption_scale;
                    case WorldProp::FogParams: return kf.world.has_fog_params;
                    case WorldProp::GodRaysParams: return kf.world.has_godrays_params;
                    case WorldProp::MultiScatterFactor: return kf.world.has_multi_scatter;
                    case WorldProp::WeatherParams: return kf.world.has_weather_params;
                }
            }
        }
        return false;
    };

    auto insertWorldKey = [&](const std::string& label, WorldProp prop) {
        int cf = ctx.render_settings.animation_current_frame;
        World& w = ctx.renderer.world;
        NishitaSkyParams np = w.getNishitaParams();
        AtmosphereAdvanced adv = w.getAdvancedParams();
        WeatherParams weather = w.getWeatherParams();
        // Keys read the climate AUTHORITY, never the derived render packet.
        const atmosphere::ClimateState clim = w.getClimate();
        auto writeClimateKey = [&clim](WorldKeyframe& k) {
            k.has_climate = true;
            k.climate_temperature_k = clim.surface_temperature_k;
            k.climate_lapse_rate_k_per_m = clim.lapse_rate_k_per_m;
            k.climate_relative_humidity = clim.surface_relative_humidity;
            k.climate_pressure_pa = clim.surface_pressure_pa;
            k.climate_wind_direction = clim.wind_direction;
            k.climate_wind_speed_mps = clim.wind_speed_mps;
        };

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
                case WorldProp::Clouds: return kf.world.has_clouds;
                case WorldProp::Climate: return kf.world.has_climate;
                case WorldProp::OzoneStrength: return kf.world.has_ozone_absorption_scale;
                case WorldProp::FogParams: return kf.world.has_fog_params;
                case WorldProp::GodRaysParams: return kf.world.has_godrays_params;
                case WorldProp::MultiScatterFactor: return kf.world.has_multi_scatter;
                case WorldProp::WeatherParams: return kf.world.has_weather_params;
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
                case WorldProp::Clouds: kf.world.has_clouds = false; break;
                case WorldProp::Climate: kf.world.has_climate = false; break;
                case WorldProp::OzoneStrength: kf.world.has_ozone_absorption_scale = false; break;
                case WorldProp::FogParams: kf.world.has_fog_params = false; break;
                case WorldProp::GodRaysParams: kf.world.has_godrays_params = false; break;
                case WorldProp::MultiScatterFactor: kf.world.has_multi_scatter = false; break;
                case WorldProp::WeatherParams: kf.world.has_weather_params = false; break;
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
                                      wk.has_altitude || wk.has_mie_anisotropy || wk.has_clouds ||
                                      wk.has_climate || wk.has_ozone_absorption_scale ||
                                      wk.has_fog_params || wk.has_godrays_params ||
                                      wk.has_multi_scatter ||
                                      wk.has_weather_params;
                    
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
                    case WorldProp::Clouds:
                        it->world.has_clouds = true;
                        it->world.clouds = w.getClouds();
                        break;
                    case WorldProp::Climate:
                        writeClimateKey(it->world);
                        break;
                    case WorldProp::OzoneStrength:
                        it->world.has_ozone_absorption_scale = true;
                        it->world.ozone_absorption_scale = np.ozone_absorption_scale;
                        break;
                    case WorldProp::FogParams:
                        it->world.has_fog_params = true;
                        it->world.fog_density = np.fog_density;
                        it->world.fog_height = np.fog_height;
                        it->world.fog_falloff = np.fog_falloff;
                        it->world.fog_distance = np.fog_distance;
                        it->world.fog_albedo = Vec3(np.fog_albedo.x, np.fog_albedo.y, np.fog_albedo.z);
                        it->world.fog_anisotropy = np.fog_anisotropy;
                        break;
                    case WorldProp::GodRaysParams:
                        it->world.has_godrays_params = true;
                        it->world.godrays_intensity = np.godrays_intensity;
                        it->world.godrays_density = np.godrays_density;
                       
                        break;
                    case WorldProp::MultiScatterFactor:
                        it->world.has_multi_scatter = true;
                        it->world.multi_scatter_factor = adv.multi_scatter_factor;
                        break;
                    case WorldProp::WeatherParams:
                        it->world.has_weather_params = true;
                        it->world.weather_enabled = weather.enabled;
                        it->world.weather_type = weather.type;
                        it->world.weather_intensity = weather.intensity;
                        it->world.weather_density = weather.density;
                        it->world.weather_precipitation_scale = weather.precipitation_scale;
                        it->world.weather_visibility = weather.visibility;
                        it->world.weather_surface_wetness = weather.surface_wetness_output;
                        it->world.weather_surface_accumulation = weather.surface_accumulation_output;
                        it->world.weather_surface_settling = weather.surface_settling_output;
                        it->world.weather_surface_height = weather.surface_height_output;
                        it->world.weather_visual_mode = weather.visual_mode;
                        it->world.weather_surface_response_enabled = weather.surface_response_enabled;
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
                case WorldProp::Clouds:
                    wk.has_clouds = true;
                    wk.clouds = w.getClouds();
                    break;
                case WorldProp::Climate:
                    writeClimateKey(wk);
                    break;
                case WorldProp::OzoneStrength:
                    wk.has_ozone_absorption_scale = true;
                    wk.ozone_absorption_scale = np.ozone_absorption_scale;
                    break;
                case WorldProp::FogParams:
                    wk.has_fog_params = true;
                    wk.fog_density = np.fog_density;
                    wk.fog_height = np.fog_height;
                    wk.fog_falloff = np.fog_falloff;
                    wk.fog_distance = np.fog_distance;
                    wk.fog_albedo = Vec3(np.fog_albedo.x, np.fog_albedo.y, np.fog_albedo.z);
                    wk.fog_anisotropy = np.fog_anisotropy;
                    break;
                case WorldProp::GodRaysParams:
                    wk.has_godrays_params = true;
                    wk.godrays_intensity = np.godrays_intensity;
                    wk.godrays_density = np.godrays_density;
                   
                    break;
                case WorldProp::MultiScatterFactor:
                    wk.has_multi_scatter = true;
                    wk.multi_scatter_factor = adv.multi_scatter_factor;
                    break;
                case WorldProp::WeatherParams:
                    wk.has_weather_params = true;
                    wk.weather_enabled = weather.enabled;
                    wk.weather_type = weather.type;
                    wk.weather_intensity = weather.intensity;
                    wk.weather_density = weather.density;
                    wk.weather_precipitation_scale = weather.precipitation_scale;
                    wk.weather_visibility = weather.visibility;
                    wk.weather_surface_wetness = weather.surface_wetness_output;
                    wk.weather_surface_accumulation = weather.surface_accumulation_output;
                    wk.weather_surface_settling = weather.surface_settling_output;
                    wk.weather_surface_height = weather.surface_height_output;
                    wk.weather_visual_mode = weather.visual_mode;
                    wk.weather_surface_response_enabled = weather.surface_response_enabled;
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
            // Same path as world.set_mode: entering nishita adds a "Sun"
            // directional when the scene has none (the sky sun never lights
            // surfaces by itself).
            if (!rtapi::setWorldMode(mode_idx == WORLD_MODE_NISHITA ? "nishita"
                                     : mode_idx == WORLD_MODE_HDRI ? "hdri" : "solid"))
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
            if (SceneUI::DrawSmartFloat("sint", "Intensity", &intensity, 0.0f, 10.0f, "%.2f", isWorldKeyed(WorldProp::BackgroundStrength), [&]{ insertWorldKey("BG Intensity", WorldProp::BackgroundStrength); }, 16)) {
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
            if (UIWidgets::PrimaryButton("Load Environment", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
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
            if (SceneUI::DrawSmartFloat("hrot", "Rotation", &rotation, 0.0f, 360.0f, "%.1f deg", rotKeyed, [&]{ insertWorldKey("HDRI Rot", WorldProp::HDRIRotation); }, 16)) {
                world.setHDRIRotation(rotation);
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Y-axis rotation (0-360)");
            
            // Intensity
            float intensity = world.getHDRIIntensity();
            if (SceneUI::DrawSmartFloat("hint", "Intensity", &intensity, 0.0f, 10.0f, "%.2f", false, nullptr, 16)) {
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
        AtmosphereAdvanced adv = world.getAdvancedParams();
        const bool worldSunDrivenByTimeline = timelineHasAnimatedWorldSun(ctx.scene.timeline);
        // static bool syncWithDirectionalLight replaced by member
        
        // Sync with Directional Light Section
        if (UIWidgets::BeginSection("Light Sync", ImVec4(0.6f, 0.8f, 1.0f, 1.0f))) {
            if (ImGui::Checkbox("Sync with Scene Light", &sync_sun_with_light)) {
                // Checkbox toggled
            }
            UIWidgets::HelpMarker("Automatically sync sun direction with the first directional light in the scene");
            

            
            if (sync_sun_with_light && !worldSunDrivenByTimeline) {
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

                        // Also sync light color and intensity into world params
                        // Use top-level world color as sun tint (used by Vulkan miss shader)
                        Vec3 lightColor = light->color;
                        world.setColor(lightColor);
                        // Sun intensity stored in Nishita params
                        params.sun_intensity = light->intensity;
                        world.setSunIntensity(light->intensity);
                        
                        foundDirLight = true;
                        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Synced with Directional Light");
                        break;
                    }
                }
                
                if (!foundDirLight) {
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "No directional light found");
                }
            } else if (sync_sun_with_light && worldSunDrivenByTimeline) {
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Timeline controls sun");
            }
            // The sky sun never lights surfaces by itself (both Vulkan backends):
            // without a directional the scene gets sky light only. Switching into
            // this mode adds one; this covers projects opened already in it.
            // Same call as world.ensure_sun_light.
            bool hasDirectional = false;
            for (const auto& light : ctx.scene.lights)
                if (light && light->type() == LightType::Directional) { hasDirectional = true; break; }
            if (!hasDirectional) {
                if (ImGui::Button("Add Sun Light")) {
                    bool created = false;
                    std::string name;
                    rtapi::ensureWorldSunLight(created, name);
                }
                UIWidgets::HelpMarker("The sky sun is drawn but does not light the scene.\nAdds a directional 'Sun' light aligned with it.");
            }
            UIWidgets::EndSection();
        }
        
        // Sun Position Section (controls both sun and directional light when synced)
        const char* sunPosTitle = sync_sun_with_light ? "Sun/Light Position" : "Sun Position";
        if (UIWidgets::BeginSection(sunPosTitle, ImVec4(1.0f, 0.7f, 0.3f, 1.0f))) {
            // Elevation
            bool sunKeyed = isWorldKeyed(WorldProp::SunElevation);
            if (SceneUI::DrawSmartFloat("ssel", "Elevation", &params.sun_elevation, -10.0f, 90.0f, "%.1f deg", sunKeyed, [&]{ insertWorldKey("Sun Elev", WorldProp::SunElevation); }, 16)) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker(sync_sun_with_light ? 
                "Sun/Light height above horizon - controls both sky sun and directional light" :
                "Sun height above horizon (0 = horizon, 90 = zenith)");
            
            // Azimuth
            bool azKeyed = isWorldKeyed(WorldProp::SunAzimuth);
            if (SceneUI::DrawSmartFloat("ssaz", "Azimuth", &params.sun_azimuth, 0.0f, 360.0f, "%.1f deg", azKeyed, [&]{ insertWorldKey("Sun Azimuth", WorldProp::SunAzimuth); }, 16)) {
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
            if (SceneUI::DrawSmartFloat("ssin", "Intensity", &params.sun_intensity, 0.0f, 100.0f, "%.1f", intKeyed, [&]{ insertWorldKey("Sun Intensity", WorldProp::SunIntensity); }, 16)) {
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
            if (SceneUI::DrawSmartFloat("sssz", "Size", &params.sun_size, 0.1f, 5.0f, "%.3f deg", sizeKeyed, [&]{ insertWorldKey("Sun Size", WorldProp::SunSize); }, 16)) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Angular diameter of the sun disc (real sun = 0.545 deg)");
            if (elevationFactor > 1.01f) {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.5f, 1.0f), "Effective: %.3f deg (horizon boost)", displaySize);
            }
            UIWidgets::EndSection();
        }
        
        // Atmosphere Section (Realistic parameters)
        if (UIWidgets::BeginSection("Atmosphere", ImVec4(0.4f, 0.7f, 1.0f, 1.0f))) {
            // Atmosphere Intensity (independent of sun)
            if (SceneUI::DrawSmartFloat("satm", "Intensity", &params.atmosphere_intensity, 0.0f, 100.0f, "%.1f", false, nullptr, 16)) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Atmospheric scattering brightness (sky color, halo, ambient). Independent of sun intensity.");
            
            // Air
            bool airKeyed = isWorldKeyed(WorldProp::AirDensity);
            if (SceneUI::DrawSmartFloat("sair", "Air", &params.air_density, 0.0f, 10.0f, "%.2f", airKeyed, [&]{ insertWorldKey("Air", WorldProp::AirDensity); }, 16)) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Rayleigh scattering density");
            
            // Dust/Haze (Mie)
            bool dustKeyed = isWorldKeyed(WorldProp::DustDensity);
            if (SceneUI::DrawSmartFloat("sdst", "Dust (Haze)", &params.dust_density, 0.0f, 10.0f, "%.2f", dustKeyed, [&]{ insertWorldKey("Dust", WorldProp::DustDensity); }, 16)) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Mie scattering (base dust density)");
            
            // Humidity and temperature moved to the Climate section: they are
            // climate state, not sky parameters, and the sky only reads them.

            // Ozone
            bool ozoKeyed = isWorldKeyed(WorldProp::OzoneDensity);
            if (SceneUI::DrawSmartFloat("sozn", "Ozone Density", &params.ozone_density, 0.0f, 10.0f, "%.2f", ozoKeyed, [&]{ insertWorldKey("Ozone", WorldProp::OzoneDensity); }, 16)) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Ozone layer presence");
            
            bool ozsKeyed = isWorldKeyed(WorldProp::OzoneStrength);
            if (SceneUI::DrawSmartFloat("sozs", "Ozone Strength", &params.ozone_absorption_scale, 0.0f, 10.0f, "%.2f", ozsKeyed, [&]{ insertWorldKey("Ozone Str", WorldProp::OzoneStrength); }, 16)) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Scales ozone absorption (Blue Hour intensity)");
            
            // Altitude
            float altitudeKm = params.altitude / 1000.0f;
            bool altKeyed = isWorldKeyed(WorldProp::Altitude);
            if (SceneUI::DrawSmartFloat("salt", "Altitude", &altitudeKm, 0.0f, 60.0f, "%.1f km", altKeyed, [&]{ insertWorldKey("Altitude", WorldProp::Altitude); }, 16)) {
                params.altitude = altitudeKm * 1000.0f;
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Camera altitude above sea level");
            UIWidgets::EndSection();
        }
        
        // Advanced Physics (Collapsible)
        if (UIWidgets::CollapsingHeader("Advanced Physics")) {
            ImGui::Indent();
            
            // Mie Anisotropy
            bool mieKeyed = isWorldKeyed(WorldProp::MieAnisotropy);
            if (SceneUI::DrawSmartFloat("smie", "Mie Anisotropy", &params.mie_anisotropy, 0.0f, 0.99f, "%.2f", mieKeyed, [&]{ insertWorldKey("Mie", WorldProp::MieAnisotropy); }, 16)) {
                changed = true;
            }
            UIWidgets::HelpMarker("Sun glow directionality (0 = uniform, 0.8+ = strong forward scatter)");
            
            ImGui::Unindent();
        }
        
        // ═══════════════════════════════════════════════════════════
        // ENVIRONMENT TEXTURE OVERLAY (HDR/EXR blending with procedural)
        // ═══════════════════════════════════════════════════════════
        if (UIWidgets::BeginSection("Environment Overlay", ImVec4(0.5f, 0.6f, 0.9f, 1.0f))) {
            bool envEnabled = adv.env_overlay_enabled != 0;
            if (ImGui::Checkbox("Enable Overlay", &envEnabled)) {
                adv.env_overlay_enabled = envEnabled ? 1 : 0;
                changed = true;
            }
            UIWidgets::HelpMarker("Blend an HDR/EXR environment texture with the procedural Nishita sky");
            
            if (adv.env_overlay_enabled) {
                ImGui::Separator();
                
                // Load Environment Texture Button
                if (UIWidgets::PrimaryButton("Load Environment Map", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
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
                if (ImGui::Combo("Blend Mode", &adv.env_overlay_blend_mode, blendModes, IM_ARRAYSIZE(blendModes))) {
                    changed = true;
                }
                UIWidgets::HelpMarker(
                    "Mix: Blend between Nishita and texture\n"
                    "Multiply: Use texture for color grading\n"
                    "Screen: Brightens without washing out\n"
                    "Replace: Use ONLY the texture");
                
                // Intensity
                if (SceneUI::DrawSmartFloat("eoint", "Overlay Intensity", &adv.env_overlay_intensity, 0.0f, 3.0f, "%.2f", false, nullptr, 16)) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Strength of the overlay texture");
                
                // Rotation
                if (SceneUI::DrawSmartFloat("eorot", "Overlay Rotation", &adv.env_overlay_rotation, 0.0f, 360.0f, "%.1f deg", false, nullptr, 16)) {
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
                bool fogKeyed = isWorldKeyed(WorldProp::FogParams);
                
                if (SceneUI::DrawSmartFloat("fogd", "Fog Density", &params.fog_density, 0.00001f, 0.01f, "%.5f", fogKeyed, [&]{ insertWorldKey("Fog", WorldProp::FogParams); }, 16)) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Extinction per metre. 0.0001 is light atmospheric fog; 0.001 becomes dense over kilometre-scale views.");
                
                if (SceneUI::DrawSmartFloat("fogh", "Fog Height", &params.fog_height, 10.0f, 2000.0f, "%.0f m", fogKeyed, [&]{ insertWorldKey("Fog", WorldProp::FogParams); }, 16)) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Top of the uniform fog layer (scene Y, metres).\nBelow it the density is constant; above it the fog thins out by Fog Falloff.");
                
                if (SceneUI::DrawSmartFloat("fogf", "Fog Falloff", &params.fog_falloff, 0.0005f, 0.02f, "%.4f", fogKeyed, [&]{ insertWorldKey("Fog", WorldProp::FogParams); }, 16)) {
                    changed = true;
                }
                UIWidgets::HelpMarker("How fast the fog thins above Fog Height (per metre).");
                
                float fogDistKm = params.fog_distance / 1000.0f;
                if (SceneUI::DrawSmartFloat("fogD", "Fog Distance", &fogDistKm, 0.1f, 50.0f, "%.1f km", fogKeyed, [&]{ insertWorldKey("Fog", WorldProp::FogParams); }, 16)) {
                    params.fog_distance = fogDistKm * 1000.0f;
                    changed = true;
                }
                UIWidgets::HelpMarker("Maximum distance for fog effect");
                
                float fogAlbedo[3] = { params.fog_albedo.x, params.fog_albedo.y, params.fog_albedo.z };
                if (ImGui::ColorEdit3("Fog Albedo", fogAlbedo)) {
                    params.fog_albedo = make_float3(fogAlbedo[0], fogAlbedo[1], fogAlbedo[2]);
                    changed = true;
                    if (fogKeyed) insertWorldKey("Fog", WorldProp::FogParams);
                }
                UIWidgets::HelpMarker("Fraction of the light the fog scatters instead of absorbing.\nWith the physical sky the fog is lit by the sun and sky, so its\ncolour follows the time of day. Other world modes use this as the fog colour.");
                
                if (SceneUI::DrawSmartFloat("fogg", "Anisotropy", &params.fog_anisotropy, 0.0f, 0.95f, "%.2f", fogKeyed, [&]{ insertWorldKey("Fog", WorldProp::FogParams); }, 16)) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Forward scattering of the fog droplets (Henyey-Greenstein g).\nHigher = brighter glow around the sun, darker away from it.");
                
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
                bool grKeyed = isWorldKeyed(WorldProp::GodRaysParams);
                
                if (SceneUI::DrawSmartFloat("grint", "Ray Intensity", &params.godrays_intensity, 0.0f, 3.0f, "%.2f", grKeyed, [&]{ insertWorldKey("God Rays", WorldProp::GodRaysParams); }, 16)) {
                    changed = true;
                }
                
                if (SceneUI::DrawSmartFloat("grden", "Ray Density", &params.godrays_density, 0.01f, 1.0f, "%.2f", grKeyed, [&]{ insertWorldKey("God Rays", WorldProp::GodRaysParams); }, 16)) {
                    changed = true;
                }
                
                if (ImGui::SliderInt("Ray Samples", &params.godrays_samples, 8, 48)) {
                    changed = true;
                    if (grKeyed) insertWorldKey("God Rays", WorldProp::GodRaysParams);
                }
                UIWidgets::HelpMarker("Number of samples for volumetric rays.\nHigh values = Better quality but slower.");
                
                ImGui::Unindent();
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // --- MULTI-SCATTERING ---
            ImGui::TextColored(ImVec4(0.5f, 0.9f, 1.0f, 1.0f), "Multi-Scattering:");
            
            bool msEnabled = adv.multi_scatter_enabled != 0;
            if (ImGui::Checkbox("Enable Multi-Scattering", &msEnabled)) {
                adv.multi_scatter_enabled = msEnabled ? 1 : 0;
                changed = true;
            }
            UIWidgets::HelpMarker("Simulates multiple light bounces in atmosphere.\nBrightens horizon and makes sky more uniform.");
            
            if (adv.multi_scatter_enabled) {
                ImGui::Indent();
                
            bool msKeyed = isWorldKeyed(WorldProp::MultiScatterFactor);
            if (SceneUI::DrawSmartFloat("msint", "MS Intensity", &adv.multi_scatter_factor, 0.0f, 1.0f, "%.2f", msKeyed, [&]{ insertWorldKey("MultiScatter", WorldProp::MultiScatterFactor); }, 16)) {
                changed = true;
            }
                UIWidgets::HelpMarker("Strength of multi-scattering contribution");
                
                ImGui::Unindent();
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // --- VOLUME ATMOSPHERE AMBIENT (OptiX parity gate) ---
            {
                bool volAtmo = world.getVolumeAtmosphereAmbient();
                if (ImGui::Checkbox("Atmosphere Lights Volumes", &volAtmo)) {
                    world.setVolumeAtmosphereAmbient(volAtmo);
                    changed = true;
                }
                UIWidgets::HelpMarker("Let the Nishita sky ambient light VDB/fluid volumes (OptiX).\nOFF by default: the raw sky over-lights volumes vs the Vulkan\nLUT ambient, breaking backend parity. Turn on once the\nbase-radiance parity is sorted.");
            }

            // --- AERIAL PERSPECTIVE ---
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Aerial Perspective:");

            bool apEnabled = adv.aerial_perspective != 0;
            if (ImGui::Checkbox("Enable Aerial Perspective", &apEnabled)) {
                adv.aerial_perspective = apEnabled ? 1 : 0;
                changed = true;
            }
            UIWidgets::HelpMarker("Air scattering between the camera and distant surfaces.\nThe amount comes from the atmosphere itself (air and dust density,\nclimate humidity) -- the same medium the sky is drawn from.");
            
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
            world.setAdvancedParams(adv);
            
            // If sync is enabled, also update the directional light
            if (sync_sun_with_light) {
                for (auto& light : ctx.scene.lights) {
                    if (light && light->type() == LightType::Directional) {
                        // Light direction is opposite of sun direction (sun to ground)
                        const Vec3 newDirection(
                            -params.sun_direction.x,
                            -params.sun_direction.y,
                            -params.sun_direction.z
                        );
                        const bool lightChanged =
                            fabsf(light->direction.x - newDirection.x) > 1e-5f ||
                            fabsf(light->direction.y - newDirection.y) > 1e-5f ||
                            fabsf(light->direction.z - newDirection.z) > 1e-5f ||
                            fabsf(light->intensity - params.sun_intensity) > 1e-5f;

                        // Air/dust/fog/overlay edits must not rebuild the light
                        // buffer and the complete realtime shadow atlas. Only an
                        // actual sun direction/intensity change owns that cost.
                        if (lightChanged) {
                            light->direction = newDirection;
                            light->intensity = params.sun_intensity;
                            extern bool g_lights_dirty;
                            g_lights_dirty = true;
                        }
                        break;
                    }
                }
            }
        }

    }
    
    // ═══════════════════════════════════════════════════════════
    // Climate (atmosphere::ClimateState) — the world's single owner of ambient
    // temperature, humidity, pressure and wind. Shown in every world mode: the
    // climate is not a sky parameter, the sky merely reads it.
    // Edits go through World::setClimate (rejects, never clamps silently), so
    // what this panel shows is exactly what the renderers receive.
    // ═══════════════════════════════════════════════════════════
    ImGui::Spacing();
    if (UIWidgets::BeginSection("Climate", ImVec4(0.45f, 0.65f, 0.85f, 1.0f))) {
        const atmosphere::ClimateState current = world.getClimate();
        atmosphere::ClimateState edit = current;
        bool climateChanged = false;
        const bool climateKeyed = isWorldKeyed(WorldProp::Climate);
        auto keyClimate = [&]{ insertWorldKey("Climate", WorldProp::Climate); };

        float temperatureC = edit.surface_temperature_k - atmosphere::kKelvinOffset;
        if (SceneUI::DrawSmartFloat("ctmp", "Temperature", &temperatureC, -50.0f, 50.0f, "%.1f C", climateKeyed, keyClimate, 16)) {
            edit.surface_temperature_k = temperatureC + atmosphere::kKelvinOffset;
            climateChanged = true;
        }
        ImGui::SameLine(); UIWidgets::HelpMarker("Surface air temperature. Stored in Kelvin; scales the sky's scale heights.");

        float humidityPct = edit.surface_relative_humidity * 100.0f;
        if (SceneUI::DrawSmartFloat("chum", "Rel. Humidity", &humidityPct, 0.0f, 100.0f, "%.0f %%", climateKeyed, keyClimate, 16)) {
            edit.surface_relative_humidity = humidityPct / 100.0f;
            climateChanged = true;
        }
        ImGui::SameLine(); UIWidgets::HelpMarker("Relative humidity. Wet aerosol swells and scatters more: haze grows as (1-RH)^-0.5.");

        float lapseKPerKm = edit.lapse_rate_k_per_m * 1000.0f;
        if (SceneUI::DrawSmartFloat("clps", "Lapse Rate", &lapseKPerKm, -5.0f, 10.0f, "%.2f K/km", climateKeyed, keyClimate, 16)) {
            edit.lapse_rate_k_per_m = lapseKPerKm / 1000.0f;
            climateChanged = true;
        }
        ImGui::SameLine(); UIWidgets::HelpMarker("Temperature drop per km of altitude (ISA 6.5). Negative = inversion.");

        float pressureHpa = edit.surface_pressure_pa / 100.0f;
        if (SceneUI::DrawSmartFloat("cprs", "Pressure", &pressureHpa, 500.0f, 1085.0f, "%.0f hPa", climateKeyed, keyClimate, 16)) {
            edit.surface_pressure_pa = pressureHpa * 100.0f;
            climateChanged = true;
        }
        ImGui::SameLine(); UIWidgets::HelpMarker("Surface pressure (sea level 1013 hPa).");

        if (SceneUI::DrawSmartFloat("cwsp", "Wind Speed", &edit.wind_speed_mps, 0.0f, 60.0f, "%.1f m/s", climateKeyed, keyClimate, 16)) {
            climateChanged = true;
        }
        // Azimuth uses the sun's convention: direction = (sin az, 0, cos az).
        float windAzimuthDeg = std::atan2(edit.wind_direction.x, edit.wind_direction.z) * 180.0f / 3.14159265f;
        if (windAzimuthDeg < 0.0f) windAzimuthDeg += 360.0f;
        if (SceneUI::DrawSmartFloat("cwaz", "Wind Toward", &windAzimuthDeg, 0.0f, 360.0f, "%.0f deg", climateKeyed, keyClimate, 16)) {
            const float az = windAzimuthDeg * 3.14159265f / 180.0f;
            edit.wind_direction = Vec3(std::sin(az), 0.0f, std::cos(az));
            climateChanged = true;
        }
        ImGui::SameLine(); UIWidgets::HelpMarker("Direction the wind blows TOWARD, same azimuth convention as the sun.");
        if (ImGui::SliderFloat("Instability", &edit.instability, 0.0f, 1.0f, "%.2f")) {
            climateChanged = true;
        }
        ImGui::SameLine(); UIWidgets::HelpMarker("Convective potential (0 stable -> stratiform sky,\n1 violently unstable -> cumulonimbus).\nWith 'Derive from climate' on, the low cloud layer follows it.");

        if (climateChanged) {
            std::string climateError;
            if (world.setClimate(edit, &climateError)) {
                changed = true;
            } else {
                SCENE_LOG_WARN("[World] Climate edit rejected: " + climateError);
            }
        }
        UIWidgets::EndSection();
    }

    // ═══════════════════════════════════════════════════════════
    // Weather Controls (shared payload for CPU/OptiX/Vulkan)
    // ═══════════════════════════════════════════════════════════
    ImGui::Spacing();
    if (UIWidgets::BeginSection("Weather", ImVec4(0.45f, 0.65f, 0.85f, 1.0f))) {
        WeatherParams weather = world.getWeatherParams();
        bool weatherChanged = false;
        bool weatherKeyed = isWorldKeyed(WorldProp::WeatherParams);

        if (KeyframeButton("##WeatherKey", weatherKeyed, "Weather")) {
            insertWorldKey("Weather", WorldProp::WeatherParams);
        }
        ImGui::SameLine();
        bool weatherEnabled = weather.enabled != 0;
        if (ImGui::Checkbox("Enable Weather", &weatherEnabled)) {
            weather.enabled = weatherEnabled ? 1 : 0;
            weatherChanged = true;
        }

        const char* weatherTypes[] = { "None", "Rain", "Snow", "Dust", "Mist" };
        int weatherType = std::clamp(weather.type, 0, 4);
        if (ImGui::Combo("Type", &weatherType, weatherTypes, IM_ARRAYSIZE(weatherTypes))) {
            weather.type = weatherType;
            weatherChanged = true;
        }

        if (SceneUI::DrawSmartFloat("wint", "Intensity", &weather.intensity, 0.0f, 1.0f, "%.2f", weatherKeyed, [&]{ insertWorldKey("Weather", WorldProp::WeatherParams); }, 16)) {
            weatherChanged = true;
        }
        if (SceneUI::DrawSmartFloat("wden", "Density", &weather.density, 0.0f, 1.0f, "%.2f", weatherKeyed, [&]{ insertWorldKey("Weather", WorldProp::WeatherParams); }, 16)) {
            weatherChanged = true;
        }
        // Wind is climate state now (Climate section); weather reads it.

        if (SceneUI::DrawSmartFloat("wpsc", "Precip Scale", &weather.precipitation_scale, 0.1f, 10.0f, "%.2f", weatherKeyed, [&]{ insertWorldKey("Weather", WorldProp::WeatherParams); }, 16)) {
            weatherChanged = true;
        }
        if (SceneUI::DrawSmartFloat("wvis", "Visibility", &weather.visibility, 0.0f, 1.0f, "%.2f", weatherKeyed, [&]{ insertWorldKey("Weather", WorldProp::WeatherParams); }, 16)) {
            weatherChanged = true;
        }

        const char* weatherVisualModes[] = { "Overlay", "Surface Only" };
        int weatherVisualMode = std::clamp(weather.visual_mode, 0, 1);
        if (ImGui::Combo("Visual Mode", &weatherVisualMode, weatherVisualModes, IM_ARRAYSIZE(weatherVisualModes))) {
            weather.visual_mode = weatherVisualMode;
            weatherChanged = true;
        }

        bool surfaceResponseEnabled = weather.surface_response_enabled != 0;
        if (ImGui::Checkbox("Surface Response", &surfaceResponseEnabled)) {
            weather.surface_response_enabled = surfaceResponseEnabled ? 1 : 0;
            weatherChanged = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Apply wetness or accumulation to surface shading even if the visual precipitation layer is disabled.");
        }

        if (weather.type == WEATHER_RAIN) {
            if (SceneUI::DrawSmartFloat("wswt", "Surface Wetness", &weather.surface_wetness_output, 0.0f, 1.0f, "%.2f", weatherKeyed, [&]{ insertWorldKey("Weather", WorldProp::WeatherParams); }, 16)) {
                weatherChanged = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Surface wetness amount. This can stay high even when the atmospheric rain overlay is low or disabled.");
            }
        } else if (weather.type == WEATHER_SNOW || weather.type == WEATHER_DUST) {
            if (SceneUI::DrawSmartFloat("wsac", "Surface Accum", &weather.surface_accumulation_output, 0.0f, 1.0f, "%.2f", weatherKeyed, [&]{ insertWorldKey("Weather", WorldProp::WeatherParams); }, 16)) {
                weatherChanged = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Surface buildup amount. Snow or dust can stay on the scene even after the atmospheric effect is reduced or disabled.");
            }
            if (SceneUI::DrawSmartFloat("wsst", "Surface Settling", &weather.surface_settling_output, 0.0f, 1.0f, "%.2f", weatherKeyed, [&]{ insertWorldKey("Weather", WorldProp::WeatherParams); }, 16)) {
                weatherChanged = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Extra settling into cavities, sheltered pockets, and slope bases.");
            }
            if (SceneUI::DrawSmartFloat("wshg", "Surface Height", &weather.surface_height_output, 0.0f, 1.0f, "%.2f", weatherKeyed, [&]{ insertWorldKey("Weather", WorldProp::WeatherParams); }, 16)) {
                weatherChanged = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Boost deposited thickness in the shading normal so snow and dust read more volumetric.");
            }
        }

        bool realtimeWeatherPreview = ctx.render_settings.realtime_weather_preview;
        if (ImGui::Checkbox("Realtime Weather Preview", &realtimeWeatherPreview)) {
            ctx.render_settings.realtime_weather_preview = realtimeWeatherPreview;
            changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Continuously resets interactive accumulation so rain, snow, and dust keep moving even without keyframes or camera motion.");
        }

        if (weatherChanged) {
            weather.intensity = std::clamp(weather.intensity, 0.0f, 1.0f);
            weather.density = std::clamp(weather.density, 0.0f, 1.0f);
            weather.visibility = std::clamp(weather.visibility, 0.0f, 1.0f);
            weather.surface_wetness_output = std::clamp(weather.surface_wetness_output, 0.0f, 1.0f);
            weather.surface_accumulation_output = std::clamp(weather.surface_accumulation_output, 0.0f, 1.0f);
            weather.surface_settling_output = std::clamp(weather.surface_settling_output, 0.0f, 1.0f);
            weather.surface_height_output = std::clamp(weather.surface_height_output, 0.0f, 1.0f);
            weather.visual_mode = std::clamp(weather.visual_mode, static_cast<int>(WEATHER_VISUAL_OVERLAY), static_cast<int>(WEATHER_VISUAL_SURFACE_ONLY));
            world.setWeatherParams(weather);
            changed = true;
        }

        UIWidgets::EndSection();
    }

    // ═══════════════════════════════════════════════════════════
    // Global Atmosphere (Clouds)
    // ═══════════════════════════════════════════════════════════
    ImGui::Spacing();
    if (UIWidgets::BeginSection("Clouds", ImVec4(0.5f, 0.6f, 0.7f, 1.0f))) {
        // Edits a copy of the AUTHORITY (atmosphere::CloudState) and commits it
        // through setClouds, which validates. The legacy nishita.cloud_* packet
        // is derived from it and never edited here.
        atmosphere::CloudState clouds = world.getClouds();
        bool cloudsChanged = false;
        static std::string cloudError;

        const bool cloudsKeyed = isWorldKeyed(WorldProp::Clouds);
        if (KeyframeButton("##WClouds", cloudsKeyed, "Clouds")) { insertWorldKey("Clouds", WorldProp::Clouds); }
        ImGui::SameLine();
        ImGui::TextDisabled("Key all cloud settings as one block");

        // Presets replace the whole state; edit afterwards.
        static int presetIndex = 0;
        const std::vector<std::string> presets = atmosphere::cloudPresetNames();
        std::vector<const char*> presetLabels;
        for (const auto& s : presets) presetLabels.push_back(s.c_str());
        ImGui::PushItemWidth(180);
        ImGui::Combo("##CloudPreset", &presetIndex, presetLabels.data(), static_cast<int>(presetLabels.size()));
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("Apply Preset")) {
            atmosphere::CloudState preset;
            if (atmosphere::cloudPreset(presets[static_cast<size_t>(presetIndex)], preset)) {
                preset.quality = clouds.quality;   // a look preset does not change quality
                clouds = preset;
                cloudsChanged = true;
            }
        }

        // Weather derived from the climate (ATMOSPHERE_WEATHER.md §2).
        cloudsChanged |= ImGui::Checkbox("Derive from climate", &clouds.derive_from_climate);
        ImGui::SameLine(); UIWidgets::HelpMarker("Layer 1 base (condensation level), type, coverage and\ndepth follow temperature, humidity and instability.\nApplying a preset turns this off.");
        {
            const atmosphere::DerivedWeather dw = world.derivedWeather();
            ImGui::TextDisabled("Base %.0f m  Type %.2f  Cover %.2f  Depth %.0f m",
                                dw.cloud_base_m, dw.cloud_type, dw.cloud_coverage, dw.cloud_thickness_m);
            ImGui::TextDisabled("Precip %.1f mm/h (%s)  Freezing %.0f m  Lightning %.1f/min",
                                dw.precipitation_mm_h, dw.snow ? "snow" : "rain",
                                dw.freezing_level_m, dw.lightning_per_minute);
        }

        static const char* kLayerNames[atmosphere::kMaxCloudLayers] = { "Layer 1 (low)", "Layer 2 (mid)", "Layer 3" };
        for (int i = 0; i < atmosphere::kMaxCloudLayers; ++i) {
            atmosphere::CloudLayer& l = clouds.layers[i];
            ImGui::PushID(i);
            if (ImGui::TreeNodeEx(kLayerNames[i], l.enabled ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
                // Layer 1's shape fields are the climate's while deriving: shown,
                // not editable (an edit would be overwritten on the next sync).
                const bool derived = (i == 0) && clouds.derive_from_climate;
                ImGui::BeginDisabled(derived);
                cloudsChanged |= ImGui::Checkbox("Enabled", &l.enabled);
                cloudsChanged |= ImGui::DragFloat("Base (m)", &l.base_altitude_m, 10.0f, -500.0f, 20000.0f, "%.0f");
                cloudsChanged |= ImGui::DragFloat("Thickness (m)", &l.thickness_m, 10.0f, 10.0f, 15000.0f, "%.0f");
                cloudsChanged |= ImGui::SliderFloat("Coverage", &l.coverage, 0.0f, 1.0f, "%.2f");
                UIWidgets::HelpMarker("Fraction of the sky this layer covers.");
                cloudsChanged |= ImGui::SliderFloat("Type", &l.type, 0.0f, 1.0f, "%.2f");
                UIWidgets::HelpMarker("0 = stratus (flat sheet)\n0.5 = cumulus (heaped)\n1 = cumulonimbus (towering)");
                ImGui::EndDisabled();
                cloudsChanged |= ImGui::DragFloat("Extinction (1/m)", &l.extinction_per_m, 0.001f, 0.0f, 2.0f, "%.3f");
                UIWidgets::HelpMarker("Physical density. Cumulus ~0.05: a 1 km column is opaque.\nStratus ~0.03-0.04, storm cores ~0.1+.");
                cloudsChanged |= ImGui::SliderFloat("Droplet (um)", &l.droplet_diameter_um, 5.0f, 50.0f, "%.1f");
                UIWidgets::HelpMarker("Droplet diameter. Drives the phase function:\nlarger drops -> brighter silver lining toward the sun.");
                cloudsChanged |= ImGui::SliderFloat("Erosion", &l.erosion, 0.0f, 1.0f, "%.2f");
                cloudsChanged |= ImGui::DragFloat("Cell size (m)", &l.cell_size_m, 10.0f, 50.0f, 50000.0f, "%.0f");
                cloudsChanged |= ImGui::DragFloat("Detail size (m)", &l.detail_size_m, 1.0f, 5.0f, 5000.0f, "%.0f");
                ImGui::TreePop();
            }
            ImGui::PopID();
        }

        // Precipitation shafts under layer 1 (climate-owned while deriving).
        if (ImGui::TreeNodeEx("Precipitation", clouds.precipitation.rate_mm_h > 0.0f ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
            ImGui::BeginDisabled(clouds.derive_from_climate);
            cloudsChanged |= ImGui::DragFloat("Rate (mm/h)##precip", &clouds.precipitation.rate_mm_h, 0.1f, 0.0f, 200.0f, "%.1f");
            cloudsChanged |= ImGui::Checkbox("Snow##precip", &clouds.precipitation.snow);
            cloudsChanged |= ImGui::SliderFloat("Reaches ground##precip", &clouds.precipitation.ground_fraction, 0.0f, 1.0f, "%.2f");
            UIWidgets::HelpMarker("Share of the base-to-ground column the shaft survives;\nbelow 1 it evaporates on the way down (virga).");
            ImGui::EndDisabled();
            ImGui::TreePop();
        }
        if (ImGui::TreeNodeEx("Cirrus (high)", clouds.cirrus.enabled ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
            cloudsChanged |= ImGui::Checkbox("Enabled##cirrus", &clouds.cirrus.enabled);
            cloudsChanged |= ImGui::DragFloat("Altitude (m)##cirrus", &clouds.cirrus.altitude_m, 10.0f, 3000.0f, 20000.0f, "%.0f");
            cloudsChanged |= ImGui::SliderFloat("Coverage##cirrus", &clouds.cirrus.coverage, 0.0f, 1.0f, "%.2f");
            cloudsChanged |= ImGui::SliderFloat("Optical depth##cirrus", &clouds.cirrus.opacity, 0.0f, 5.0f, "%.2f");
            cloudsChanged |= ImGui::DragFloat("Streak scale (m)##cirrus", &clouds.cirrus.scale_m, 50.0f, 100.0f, 100000.0f, "%.0f");
            ImGui::TextDisabled("Rendered from Faz 3c.");
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Weather map")) {
            int seed = static_cast<int>(clouds.weather.seed);
            if (ImGui::DragInt("Seed", &seed, 1.0f, 0, 1000000)) {
                clouds.weather.seed = static_cast<uint32_t>(std::max(0, seed));
                cloudsChanged = true;
            }
            cloudsChanged |= ImGui::DragFloat("Extent (m)", &clouds.weather.extent_m, 1000.0f, 5000.0f, 1000000.0f, "%.0f");
            cloudsChanged |= ImGui::DragFloat("Cluster size (m)", &clouds.weather.feature_size_m, 100.0f, 500.0f, 200000.0f, "%.0f");
            cloudsChanged |= ImGui::SliderFloat("Type variation", &clouds.weather.type_variation, 0.0f, 1.0f, "%.2f");
            cloudsChanged |= ImGui::DragFloat("Evolution speed", &clouds.weather.evolution_speed, 0.0001f, 0.0f, 1.0f, "%.4f");
            const Vec3 drift = world.cloudWindOffset();
            ImGui::TextDisabled("Drift from climate wind: (%.0f, %.0f) m at t=%.1f s",
                                drift.x, drift.z, world.getCloudTime());
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Quality")) {
            cloudsChanged |= ImGui::DragInt("RT march steps", &clouds.quality.rt_steps, 1.0f, 16, 1024);
            cloudsChanged |= ImGui::Checkbox("RT reference path tracer (slow)", &clouds.quality.rt_reference_path_trace);
            cloudsChanged |= ImGui::DragInt("RT max bounces", &clouds.quality.rt_max_bounces, 1.0f, 1, 1024);
            cloudsChanged |= ImGui::DragInt("Realtime steps", &clouds.quality.realtime_steps, 1.0f, 8, 512);
            cloudsChanged |= ImGui::SliderInt("Realtime res divisor", &clouds.quality.realtime_resolution_divisor, 1, 8);
            cloudsChanged |= ImGui::Checkbox("RT secondary rays: full march", &clouds.quality.rt_secondary_full_march);
            UIWidgets::HelpMarker("Off (default): reflected/bounce rays see the cloud-aware sky\npanorama (fast, measured bias). On: full march (slow, unbiased).");
            ImGui::TreePop();
        }

        // Stated, not hidden: until Faz 3b the Vulkan RT image still comes
        // from the legacy cloud volume, driven by the packet derived above.
        ImGui::TextDisabled("Vulkan RT: path traced. OptiX: legacy volume. Realtime: Faz 3c.");
        if (!cloudError.empty()) ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.3f, 1.0f), "%s", cloudError.c_str());

        UIWidgets::EndSection();

        if (cloudsChanged) {
            if (world.setClouds(clouds, &cloudError)) {
                cloudError.clear();
                changed = true;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Apply Changes
    // ═══════════════════════════════════════════════════════════
    if (changed) {
        world_params_changed_this_frame = true;
        ctx.renderer.resetCPUAccumulation();
        
        // DEFERRED: Don't call setWorld()/resetAccumulation() here!
        // Let Main loop handle it once per frame after flushLUT() to avoid
        // double GPU transfer and ensure LUT is fresh before upload.
        extern void markWorldDirty();
        markWorldDirty();
        // World/atmosphere parameters live in the world buffer. Re-uploading
        // all VDB/gas volume payloads here made every sky slider scale with the
        // scene's volume data and could disturb the active volume descriptor.
        
        // Reset GPU accumulation so change is visible on next render pass
        if (ctx.backend_ptr) {
            ctx.backend_ptr->resetAccumulation();
        }
    }
    
    // Pop the global item width set at function start
    ImGui::PopItemWidth();
}

// Global Sun Synchronization Logic
void SceneUI::processSunSync(UIContext& ctx) {
    if (!sync_sun_with_light) return;
    if (world_params_changed_this_frame) return; // User is modifying sliders, skip sync to avoid fighting

    // [RENDER-LOCK RACE FIX] During an active sequence render the worker thread
    // owns scene.lights and ctx.renderer.world — Renderer::updateAnimationState
    // writes light->direction/intensity for animated directional lights and
    // calls world.setSunDirection/setSunIntensity each frame, then uploads the
    // result via m_backend->setWorldData(world.getGPUData()).
    // This function runs every UI frame on the main thread and mutates the
    // SAME state in both directions (Forward: writes light->direction,
    // light->intensity; Reverse: reads light->direction, writes
    // world.setNishitaParams). Concurrent host-side struct mutation produces
    // a torn world payload that the worker then uploads to the GPU. The
    // resulting NaN / garbage sun direction makes shaders take infinite /
    // divergent paths, and Vulkan setLights() waitIdle never returns →
    // application freeze. OptiX is unaffected because CUDA stream
    // serialization happens to mask the host-side race. The worker keeps
    // light + world fully in sync on its own during render, so we skip the
    // UI-thread sync entirely while it is active.
    if (ctx.render_settings.animation_render_locked && rendering_in_progress.load()) {
        return;
    }

    World& world = ctx.renderer.world;
    if (world.getMode() != WORLD_MODE_NISHITA) return;

    if (timelineHasAnimatedWorldSun(ctx.scene.timeline)) {
        const NishitaSkyParams params = world.getNishitaParams();
        const Vec3 lightDirection(
            -params.sun_direction.x,
            -params.sun_direction.y,
            -params.sun_direction.z);

        for (auto& light : ctx.scene.lights) {
            if (light && light->type() == LightType::Directional) {
                const bool directionChanged =
                    fabsf(light->direction.x - lightDirection.x) > 0.001f ||
                    fabsf(light->direction.y - lightDirection.y) > 0.001f ||
                    fabsf(light->direction.z - lightDirection.z) > 0.001f;
                const bool intensityChanged = fabsf(light->intensity - params.sun_intensity) > 0.001f;

                if (directionChanged || intensityChanged) {
                    light->direction = lightDirection;
                    light->intensity = params.sun_intensity;

                    extern bool g_lights_dirty;
                    g_lights_dirty = true;
                }
                break;
            }
        }
        return;
    }

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
                 extern void markWorldDirty();
                 markWorldDirty();
             }
             break; // Sync with first directional light
        }
    }
}
