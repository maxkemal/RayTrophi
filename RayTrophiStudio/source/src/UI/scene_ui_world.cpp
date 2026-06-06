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
        Humidity, Temperature, OzoneStrength,
        FogParams, GodRaysParams,
        CloudDensity, CloudCoverage, CloudScale, CloudOffset,
        CloudLighting, CloudLayer2, CloudLayer2Params,
        AerialPerspectiveParams, MultiScatterFactor, WeatherParams
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
                    case WorldProp::Humidity: return kf.world.has_humidity;
                    case WorldProp::Temperature: return kf.world.has_temperature;
                    case WorldProp::OzoneStrength: return kf.world.has_ozone_absorption_scale;
                    case WorldProp::FogParams: return kf.world.has_fog_params;
                    case WorldProp::GodRaysParams: return kf.world.has_godrays_params;
                    case WorldProp::CloudLighting: return kf.world.has_cloud_lighting;
                    case WorldProp::CloudLayer2Params: return kf.world.has_cloud_layer2_params;
                    case WorldProp::AerialPerspectiveParams: return kf.world.has_aerial_params;
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
                case WorldProp::Humidity: return kf.world.has_humidity;
                case WorldProp::Temperature: return kf.world.has_temperature;
                case WorldProp::OzoneStrength: return kf.world.has_ozone_absorption_scale;
                case WorldProp::FogParams: return kf.world.has_fog_params;
                case WorldProp::GodRaysParams: return kf.world.has_godrays_params;
                case WorldProp::CloudLighting: return kf.world.has_cloud_lighting;
                case WorldProp::CloudLayer2Params: return kf.world.has_cloud_layer2_params;
                case WorldProp::AerialPerspectiveParams: return kf.world.has_aerial_params;
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
                case WorldProp::CloudDensity: kf.world.has_cloud_density = false; break;
                case WorldProp::CloudCoverage: kf.world.has_cloud_coverage = false; break;
                case WorldProp::CloudScale: kf.world.has_cloud_scale = false; break;
                case WorldProp::CloudOffset: kf.world.has_cloud_offset = false; break;
                case WorldProp::Humidity: kf.world.has_humidity = false; break;
                case WorldProp::Temperature: kf.world.has_temperature = false; break;
                case WorldProp::OzoneStrength: kf.world.has_ozone_absorption_scale = false; break;
                case WorldProp::FogParams: kf.world.has_fog_params = false; break;
                case WorldProp::GodRaysParams: kf.world.has_godrays_params = false; break;
                case WorldProp::CloudLighting: kf.world.has_cloud_lighting = false; break;
                case WorldProp::CloudLayer2Params: kf.world.has_cloud_layer2_params = false; break;
                case WorldProp::AerialPerspectiveParams: kf.world.has_aerial_params = false; break;
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
                                      wk.has_altitude || wk.has_mie_anisotropy || wk.has_cloud_density ||
                                      wk.has_cloud_coverage || wk.has_cloud_scale || wk.has_cloud_offset ||
                                      wk.has_humidity || wk.has_temperature || wk.has_ozone_absorption_scale ||
                                      wk.has_fog_params || wk.has_godrays_params || wk.has_cloud_lighting ||
                                      wk.has_cloud_layer2_params || wk.has_aerial_params || wk.has_multi_scatter ||
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
                    case WorldProp::Humidity:
                        it->world.has_humidity = true;
                        it->world.humidity = np.humidity;
                        break;
                    case WorldProp::Temperature:
                        it->world.has_temperature = true;
                        it->world.temperature = np.temperature;
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
                        it->world.fog_color = Vec3(np.fog_color.x, np.fog_color.y, np.fog_color.z);
                        it->world.fog_sun_scatter = np.fog_sun_scatter;
                        break;
                    case WorldProp::GodRaysParams:
                        it->world.has_godrays_params = true;
                        it->world.godrays_intensity = np.godrays_intensity;
                        it->world.godrays_density = np.godrays_density;
                       
                        break;
                    case WorldProp::CloudLighting:
                        it->world.has_cloud_lighting = true;
                        it->world.cloud_shadow_strength = np.cloud_shadow_strength;
                        it->world.cloud_ambient_strength = np.cloud_ambient_strength;
                        it->world.cloud_silver_intensity = np.cloud_silver_intensity;
                        it->world.cloud_absorption = np.cloud_absorption;
                        break;
                    case WorldProp::CloudLayer2Params:
                        it->world.has_cloud_layer2_params = true;
                        it->world.cloud2_coverage = np.cloud2_coverage;
                        it->world.cloud2_density = np.cloud2_density;
                        it->world.cloud2_scale = np.cloud2_scale;
                        break;
                    case WorldProp::AerialPerspectiveParams:
                        it->world.has_aerial_params = true;
                        it->world.aerial_density = adv.aerial_density;
                        it->world.aerial_min_distance = adv.aerial_min_distance;
                        it->world.aerial_max_distance = adv.aerial_max_distance;
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
                        it->world.weather_wind_direction = Vec3(weather.wind_direction.x, weather.wind_direction.y, weather.wind_direction.z);
                        it->world.weather_wind_speed = weather.wind_speed;
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
                case WorldProp::Humidity:
                    wk.has_humidity = true;
                    wk.humidity = np.humidity;
                    break;
                case WorldProp::Temperature:
                    wk.has_temperature = true;
                    wk.temperature = np.temperature;
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
                    wk.fog_color = Vec3(np.fog_color.x, np.fog_color.y, np.fog_color.z);
                    wk.fog_sun_scatter = np.fog_sun_scatter;
                    break;
                case WorldProp::GodRaysParams:
                    wk.has_godrays_params = true;
                    wk.godrays_intensity = np.godrays_intensity;
                    wk.godrays_density = np.godrays_density;
                   
                    break;
                case WorldProp::CloudLighting:
                    wk.has_cloud_lighting = true;
                    wk.cloud_shadow_strength = np.cloud_shadow_strength;
                    wk.cloud_ambient_strength = np.cloud_ambient_strength;
                    wk.cloud_silver_intensity = np.cloud_silver_intensity;
                    wk.cloud_absorption = np.cloud_absorption;
                    break;
                case WorldProp::CloudLayer2Params:
                    wk.has_cloud_layer2_params = true;
                    wk.cloud2_coverage = np.cloud2_coverage;
                    wk.cloud2_density = np.cloud2_density;
                    wk.cloud2_scale = np.cloud2_scale;
                    break;
                case WorldProp::AerialPerspectiveParams:
                    wk.has_aerial_params = true;
                    wk.aerial_density = adv.aerial_density;
                    wk.aerial_min_distance = adv.aerial_min_distance;
                    wk.aerial_max_distance = adv.aerial_max_distance;
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
                    wk.weather_wind_direction = Vec3(weather.wind_direction.x, weather.wind_direction.y, weather.wind_direction.z);
                    wk.weather_wind_speed = weather.wind_speed;
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
            
            // Humidity (Advanced Mie)
            bool humKeyed = isWorldKeyed(WorldProp::Humidity);
            if (SceneUI::DrawSmartFloat("shum", "Humidity", &params.humidity, 0.0f, 1.0f, "%.2f", humKeyed, [&]{ insertWorldKey("Humidity", WorldProp::Humidity); }, 16)) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Atmospheric moisture - increases haze and sunlight glow");
            
            // Temperature
            bool tempKeyed = isWorldKeyed(WorldProp::Temperature);
            if (SceneUI::DrawSmartFloat("stmp", "Temperature", &params.temperature, -50.0f, 50.0f, "%.1f C", tempKeyed, [&]{ insertWorldKey("Temperature", WorldProp::Temperature); }, 16)) {
                changed = true;
            }
            ImGui::SameLine(); UIWidgets::HelpMarker("Ambient temperature - affects Rayleigh scattering purity");
            
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
        if (ImGui::CollapsingHeader("Advanced Physics")) {
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
                
                if (SceneUI::DrawSmartFloat("fogd", "Fog Density", &params.fog_density, 0.001f, 1.0f, "%.4f", fogKeyed, [&]{ insertWorldKey("Fog", WorldProp::FogParams); }, 16)) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Base fog density (lower = lighter fog)");
                
                if (SceneUI::DrawSmartFloat("fogh", "Fog Height", &params.fog_height, 10.0f, 2000.0f, "%.0f m", fogKeyed, [&]{ insertWorldKey("Fog", WorldProp::FogParams); }, 16)) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Fog is concentrated below this height");
                
                if (SceneUI::DrawSmartFloat("fogf", "Fog Falloff", &params.fog_falloff, 0.0005f, 0.02f, "%.4f", fogKeyed, [&]{ insertWorldKey("Fog", WorldProp::FogParams); }, 16)) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Rate at which fog decreases with height");
                
                float fogDistKm = params.fog_distance / 1000.0f;
                if (SceneUI::DrawSmartFloat("fogD", "Fog Distance", &fogDistKm, 0.1f, 50.0f, "%.1f km", fogKeyed, [&]{ insertWorldKey("Fog", WorldProp::FogParams); }, 16)) {
                    params.fog_distance = fogDistKm * 1000.0f;
                    changed = true;
                }
                UIWidgets::HelpMarker("Maximum distance for fog effect");
                
                float fogColor[3] = { params.fog_color.x, params.fog_color.y, params.fog_color.z };
                if (ImGui::ColorEdit3("Fog Color", fogColor)) {
                    params.fog_color = make_float3(fogColor[0], fogColor[1], fogColor[2]);
                    changed = true;
                    if (fogKeyed) insertWorldKey("Fog", WorldProp::FogParams);
                }
                
                if (SceneUI::DrawSmartFloat("ssss", "Sun Scatter", &params.fog_sun_scatter, 0.0f, 2.0f, "%.2f", fogKeyed, [&]{ insertWorldKey("Fog", WorldProp::FogParams); }, 16)) {
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
            UIWidgets::HelpMarker("Adds atmospheric haze to distant objects.\nObjects fade into sky color with distance.");
            
            if (adv.aerial_perspective) {
                ImGui::Indent();
                
                bool apKeyed = isWorldKeyed(WorldProp::AerialPerspectiveParams);
                if (SceneUI::DrawSmartFloat("apden", "Density", &adv.aerial_density, 0.0f, 10.0f, "%.2f", apKeyed, [&]{ insertWorldKey("Aerial Persp", WorldProp::AerialPerspectiveParams); }, 16)) {
                    adv.aerial_density = std::clamp(adv.aerial_density, 0.0f, 10.0f);
                    changed = true;
                }
                UIWidgets::HelpMarker("Independent aerial haze density.\nDistance controls where haze ramps; density controls how strongly it accumulates.");

                // Min Distance (km) - Flexible range for different scene scales
                float minDistKm = adv.aerial_min_distance / 1000.0f;
                if (SceneUI::DrawSmartFloat("apmin", "Min Distance", &minDistKm, 0.0f, 5.0f, "%.2f km", apKeyed, [&]{ insertWorldKey("Aerial Persp", WorldProp::AerialPerspectiveParams); }, 16)) {
                    adv.aerial_min_distance = minDistKm * 1000.0f;
                    changed = true;
                }
                UIWidgets::HelpMarker("Objects closer than this distance have NO haze.\nCtrl+Click for manual input. 0 = haze everywhere.");
                
                // Max Distance (km)
                float maxDistKm = adv.aerial_max_distance / 1000.0f;
                if (SceneUI::DrawSmartFloat("apmax", "Full Haze Distance", &maxDistKm, 0.1f, 50.0f, "%.1f km", apKeyed, [&]{ insertWorldKey("Aerial Persp", WorldProp::AerialPerspectiveParams); }, 16)) {
                    adv.aerial_max_distance = maxDistKm * 1000.0f;
                    changed = true;
                }
                UIWidgets::HelpMarker("Full aerial perspective effect at this distance.\nObjects beyond this are fully hazed.");
                
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
            world.setAdvancedParams(adv);
            
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
                        
                        // Keep light intensity in sync with the Nishita params when user edits the sun
                        light->intensity = params.sun_intensity;
                        
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
        if (SceneUI::DrawSmartFloat("wwsp", "Wind Speed", &weather.wind_speed, 0.0f, 100.0f, "%.1f m/s", weatherKeyed, [&]{ insertWorldKey("Weather", WorldProp::WeatherParams); }, 16)) {
            weatherChanged = true;
        }

        Vec3 wind(weather.wind_direction.x, weather.wind_direction.y, weather.wind_direction.z);
        if (SceneUI::DrawSmartFloat("wwdx", "Wind X", &wind.x, -1.0f, 1.0f, "%.2f", weatherKeyed, [&]{ insertWorldKey("Weather", WorldProp::WeatherParams); }, 16)) weatherChanged = true;
        if (SceneUI::DrawSmartFloat("wwdy", "Wind Y", &wind.y, -1.0f, 1.0f, "%.2f", weatherKeyed, [&]{ insertWorldKey("Weather", WorldProp::WeatherParams); }, 16)) weatherChanged = true;
        if (SceneUI::DrawSmartFloat("wwdz", "Wind Z", &wind.z, -1.0f, 1.0f, "%.2f", weatherKeyed, [&]{ insertWorldKey("Weather", WorldProp::WeatherParams); }, 16)) weatherChanged = true;

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
            float len = std::sqrt(wind.x * wind.x + wind.y * wind.y + wind.z * wind.z);
            if (len > 1e-5f) {
                wind = Vec3(wind.x / len, wind.y / len, wind.z / len);
            } else {
                wind = Vec3(1.0f, 0.0f, 0.0f);
            }
            weather.wind_direction = make_float3(wind.x, wind.y, wind.z);
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
                    case 1: // Cirrus (Wispy high clouds)
                        cloudParams.clouds_enabled = 1;
                        cloudParams.cloud_height_min = 8000.0f;
                        cloudParams.cloud_height_max = 8800.0f;
                        cloudParams.cloud_scale = 0.25f;
                        cloudParams.cloud_coverage = 0.16f;
                        cloudParams.cloud_density = 0.22f;
                        cloudParams.cloud_detail = 0.75f;
                        cloudParams.cloud_shadow_strength = 0.1f;
                        cloudParamsChanged = true;
                        break;
                    case 2: // Cumulus (Classic puffy clouds)
                        cloudParams.clouds_enabled = 1;
                        cloudParams.cloud_height_min = 1000.0f;
                        cloudParams.cloud_height_max = 3600.0f;
                        cloudParams.cloud_scale = 0.48f;
                        cloudParams.cloud_coverage = 0.28f;
                        cloudParams.cloud_density = 0.82f;
                        cloudParams.cloud_detail = 0.42f;
                        cloudParams.cloud_shadow_strength = 0.28f;
                        cloudParams.cloud_absorption = 0.75f;
                        cloudParams.cloud_ambient_strength = 1.2f;
                        cloudParams.cloud_silver_intensity = 0.35f;
                        cloudParamsChanged = true;
                        break;
                    case 3: // Stratocumulus (Layered puffy)
                        cloudParams.clouds_enabled = 1;
                        cloudParams.cloud_height_min = 800.0f;
                        cloudParams.cloud_height_max = 1800.0f;
                        cloudParams.cloud_scale = 0.42f;
                        cloudParams.cloud_coverage = 0.42f;
                        cloudParams.cloud_density = 0.85f;
                        cloudParams.cloud_detail = 0.45f;
                        cloudParams.cloud_shadow_strength = 0.32f;
                        cloudParams.cloud_absorption = 0.85f;
                        cloudParams.cloud_ambient_strength = 1.15f;
                        cloudParamsChanged = true;
                        break;
                    case 4: // Overcast (Stratus)
                        cloudParams.clouds_enabled = 1;
                        cloudParams.cloud_height_min = 400.0f;
                        cloudParams.cloud_height_max = 1000.0f;
                        cloudParams.cloud_scale = 0.28f;
                        cloudParams.cloud_coverage = 0.58f;
                        cloudParams.cloud_density = 0.92f;
                        cloudParams.cloud_detail = 0.35f;
                        cloudParams.cloud_shadow_strength = 0.38f;
                        cloudParams.cloud_absorption = 0.9f;
                        cloudParams.cloud_ambient_strength = 1.1f;
                        cloudParamsChanged = true;
                        break;
                    case 5: // Cumulonimbus (Storm/Rain clouds)
                        cloudParams.clouds_enabled = 1;
                        cloudParams.cloud_height_min = 600.0f;
                        cloudParams.cloud_height_max = 12000.0f;
                        cloudParams.cloud_scale = 0.65f;
                        cloudParams.cloud_coverage = 0.52f;
                        cloudParams.cloud_density = 1.2f;
                        cloudParams.cloud_detail = 0.72f;
                        cloudParams.cloud_shadow_strength = 0.55f;
                        cloudParams.cloud_absorption = 1.2f;
                        cloudParams.cloud_ambient_strength = 0.95f;
                        cloudParams.cloud_silver_intensity = 0.45f;
                        cloudParamsChanged = true;
                        break;
                    case 6: // Fog/Low Clouds
                        cloudParams.clouds_enabled = 1;
                        cloudParams.cloud_height_min = 0.0f;
                        cloudParams.cloud_height_max = 350.0f;
                        cloudParams.cloud_scale = 0.18f;
                        cloudParams.cloud_coverage = 0.55f;
                        cloudParams.cloud_density = 0.9f;
                        cloudParams.cloud_detail = 0.25f;
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
                if (SceneUI::DrawSmartFloat("clcov", "Coverage", &cloudParams.cloud_coverage, 0.0f, 0.6f, "%.2f", covKeyed, [&]{ insertWorldKey("Cloud Cov", WorldProp::CloudCoverage); }, 16)) {
                    cloudParamsChanged = true;
                    weather_preset_index = 7;
                }
                ImGui::SameLine(); UIWidgets::HelpMarker("0 = clear, 0.3 = broken cumulus, 0.6 = near overcast");

                // Density
                bool denKeyed = isWorldKeyed(WorldProp::CloudDensity);
                if (SceneUI::DrawSmartFloat("clden", "Density", &cloudParams.cloud_density, 0.0f, 1.5f, "%.2f", denKeyed, [&]{ insertWorldKey("Cloud Den", WorldProp::CloudDensity); }, 16)) {
                    cloudParamsChanged = true;
                    weather_preset_index = 7;
                }
                ImGui::SameLine(); UIWidgets::HelpMarker("Cloud opacity");
                
                // Scale
                bool sclKeyed = isWorldKeyed(WorldProp::CloudScale);
                if (SceneUI::DrawSmartFloat("clscl", "Scale", &cloudParams.cloud_scale, 0.1f, 1.0f, "%.2f", sclKeyed, [&]{ insertWorldKey("Cloud Scale", WorldProp::CloudScale); }, 16)) {
                    cloudParamsChanged = true;
                    weather_preset_index = 7;
                }
                ImGui::SameLine(); UIWidgets::HelpMarker("Cloud feature size. Lower = broader layers, higher = smaller puffs.");

                // Detail
                if (ImGui::SliderFloat("Noise Detail", &cloudParams.cloud_detail, 0.0f, 1.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    weather_preset_index = 7;
                }
                UIWidgets::HelpMarker("Richness of high-frequency noise.");

                // Base Steps
                if (ImGui::SliderInt("Base Steps", &cloudParams.cloud_base_steps, 4, 96)) {
                    cloudParamsChanged = true;
                    weather_preset_index = 7;
                }
                UIWidgets::HelpMarker("Base number of ray-marching steps.\nLower = Faster preview\nHigher = Cinematic quality (128+ recommended for horizon)");

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

                int cloudSeed = cloudParams.cloud_seed;
                if (ImGui::DragInt("Seed", &cloudSeed, 1.0f, 0, 1000000)) {
                    cloudParams.cloud_seed = cloudSeed;
                    cloudParamsChanged = true;
                    weather_preset_index = 7;
                }
                UIWidgets::HelpMarker("Changes the procedural cloud pattern while keeping coverage, scale, and wind offsets intact.");
            }
            
            // ═══════════════════════════════════════════════════════════
            // CLOUD LIGHTING SETTINGS
            // ═══════════════════════════════════════════════════════════
            if (ImGui::CollapsingHeader("Cloud Lighting")) {
                bool lightKeyed = isWorldKeyed(WorldProp::CloudLighting);
                ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.5f, 1.0f), "Self-Shadowing:");
                
                // Light Steps (0 = disabled for performance)
                if (ImGui::SliderInt("Light Steps", &cloudParams.cloud_light_steps, 0, 24)) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Number of light marching steps.\n0 = Disabled (fast)\n4-6 = Normal\n8-12 = High quality");
                
                // Shadow Strength
                if (SceneUI::DrawSmartFloat("cshd", "Shadow Strength", &cloudParams.cloud_shadow_strength, 0.0f, 1.0f, "%.2f", lightKeyed, [&]{ insertWorldKey("Cloud Light", WorldProp::CloudLighting); }, 16)) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Cloud self-shadowing intensity.\n0 = No shadows\n1 = Normal\n2 = Dark, dramatic shadows");
                
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.5f, 1.0f), "Lighting Effects:");
                
                // Silver Lining
                if (SceneUI::DrawSmartFloat("csil", "Silver Lining", &cloudParams.cloud_silver_intensity, 0.0f, 1.0f, "%.2f", lightKeyed, [&]{ insertWorldKey("Cloud Light", WorldProp::CloudLighting); }, 16)) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Bright rim effect when backlit by sun.\n0 = Off\n1 = Normal\n2+ = Strong glow");
                
                // Ambient Strength
                if (SceneUI::DrawSmartFloat("camb", "Ambient Light", &cloudParams.cloud_ambient_strength, 0.0f, 2.0f, "%.2f", lightKeyed, [&]{ insertWorldKey("Cloud Light", WorldProp::CloudLighting); }, 16)) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Sky ambient light contribution.\nLower = Darker shadows\nHigher = Softer look");
                
                // Absorption
                if (SceneUI::DrawSmartFloat("cabs", "Absorption", &cloudParams.cloud_absorption, 0.1f, 3.0f, "%.2f", lightKeyed, [&]{ insertWorldKey("Cloud Light", WorldProp::CloudLighting); }, 16)) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Light absorption rate.\n0.5 = Thin, transparent\n1.0 = Normal\n2.0 = Thick, opaque");

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.5f, 1.0f), "Advanced Scattering (Anisotropy):");
                
                // Scattering Anisotropy (Forward)
                if (ImGui::SliderFloat("Anisotropy (Forward)", &cloudParams.cloud_anisotropy, 0.0f, 0.85f, "%.2f")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Forward scattering intensity (g-factor).\n0.0 = Uniform light\n0.9 = Strong forward glow (silver lining)");

                // Scattering Anisotropy (Back)
                if (ImGui::SliderFloat("Anisotropy (Back)", &cloudParams.cloud_anisotropy_back, -0.6f, 0.0f, "%.2f")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Back scattering intensity.\nControls how much light 'bounces back' toward the sun.");

                // Lobe Mix
                if (ImGui::SliderFloat("Lobe Mix", &cloudParams.cloud_lobe_mix, 0.0f, 1.0f, "%.2f")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Blend between forward and back scattering.\n0.5 = Balanced volumetric look");

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.5f, 1.0f), "Cloud Emission (Glow):");
                
                // Emissive Intensity
                if (ImGui::SliderFloat("Emission Intensity", &cloudParams.cloud_emissive_intensity, 0.0f, 1.0f, "%.2f")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Internal cloud glow brightness.");

                // Emissive Color
                float emissiveColor[3] = { cloudParams.cloud_emissive_color.x, cloudParams.cloud_emissive_color.y, cloudParams.cloud_emissive_color.z };
                if (ImGui::ColorEdit3("Emission Color", emissiveColor)) {
                    cloudParams.cloud_emissive_color = make_float3(emissiveColor[0], emissiveColor[1], emissiveColor[2]);
                    cloudParamsChanged = true;
                }
            }

            // === LAYER 1 ALTITUDE ===
            if (ImGui::TreeNode("Layer 1 Altitude")) {
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "Layer 1: %.0f - %.0f m (%.0f m thick)", 
                    cloudParams.cloud_height_min, cloudParams.cloud_height_max,
                    cloudParams.cloud_height_max - cloudParams.cloud_height_min);
                
                if (SceneUI::DrawSmartFloat("cmnh", "Min Height##L1", &cloudParams.cloud_height_min, 500.0f, 12000.0f, "%.0f m", false, nullptr, 16)) {
                    cloudParamsChanged = true;
                }
                if (SceneUI::DrawSmartFloat("cmxh", "Max Height##L1", &cloudParams.cloud_height_max, 1000.0f, 16000.0f, "%.0f m", false, nullptr, 16)) {
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
                            cloudParams.cloud2_scale = 0.25f;
                            cloudParams.cloud2_coverage = 0.14f;
                            cloudParams.cloud2_density = 0.18f;
                            cloudParamsChanged = true;
                            break;
                        case 2: // Mid Stratus
                            cloudParams.cloud2_height_min = 2500.0f;
                            cloudParams.cloud2_height_max = 3500.0f;
                            cloudParams.cloud2_scale = 0.35f;
                            cloudParams.cloud2_coverage = 0.35f;
                            cloudParams.cloud2_density = 0.65f;
                            cloudParamsChanged = true;
                            break;
                        case 3: // Low Fog
                            cloudParams.cloud2_height_min = 0.0f;
                            cloudParams.cloud2_height_max = 150.0f;
                            cloudParams.cloud2_scale = 0.15f;
                            cloudParams.cloud2_coverage = 0.5f;
                            cloudParams.cloud2_density = 0.8f;
                            cloudParamsChanged = true;
                            break;
                    }
                }
                
                bool l2pKeyed = isWorldKeyed(WorldProp::CloudLayer2Params);
                if (SceneUI::DrawSmartFloat("ccov2", "Coverage##L2", &cloudParams.cloud2_coverage, 0.0f, 0.6f, "%.2f", l2pKeyed, [&]{ insertWorldKey("Cloud L2", WorldProp::CloudLayer2Params); }, 16)) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                if (SceneUI::DrawSmartFloat("cden2", "Density##L2", &cloudParams.cloud2_density, 0.0f, 1.5f, "%.2f", l2pKeyed, [&]{ insertWorldKey("Cloud L2", WorldProp::CloudLayer2Params); }, 16)) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                if (SceneUI::DrawSmartFloat("cscl2", "Scale##L2", &cloudParams.cloud2_scale, 0.1f, 1.0f, "%.2f", l2pKeyed, [&]{ insertWorldKey("Cloud L2", WorldProp::CloudLayer2Params); }, 16)) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                
                ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1.0f), "Layer 2: %.0f - %.0f m", 
                    cloudParams.cloud2_height_min, cloudParams.cloud2_height_max);
                if (SceneUI::DrawSmartFloat("cmnh2", "Min Height##L2", &cloudParams.cloud2_height_min, 500.0f, 12000.0f, "%.0f m", l2pKeyed, [&]{ insertWorldKey("Cloud L2", WorldProp::CloudLayer2Params); }, 16)) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                if (SceneUI::DrawSmartFloat("cmxh2", "Max Height##L2", &cloudParams.cloud2_height_max, 1000.0f, 16000.0f, "%.0f m", l2pKeyed, [&]{ insertWorldKey("Cloud L2", WorldProp::CloudLayer2Params); }, 16)) {
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
            cloudParams.cloud_coverage = (std::max)(0.0f, (std::min)(0.6f, cloudParams.cloud_coverage));
            cloudParams.cloud_density = (std::max)(0.0f, (std::min)(1.5f, cloudParams.cloud_density));
            cloudParams.cloud_scale = (std::max)(0.1f, (std::min)(1.0f, cloudParams.cloud_scale));
            cloudParams.cloud_seed = (std::max)(0, (std::min)(1000000, cloudParams.cloud_seed));
            cloudParams.cloud_detail = (std::max)(0.0f, (std::min)(1.0f, cloudParams.cloud_detail));
            cloudParams.cloud_base_steps = (std::max)(4, (std::min)(96, cloudParams.cloud_base_steps));
            cloudParams.cloud2_coverage = (std::max)(0.0f, (std::min)(0.6f, cloudParams.cloud2_coverage));
            cloudParams.cloud2_density = (std::max)(0.0f, (std::min)(1.5f, cloudParams.cloud2_density));
            cloudParams.cloud2_scale = (std::max)(0.1f, (std::min)(1.0f, cloudParams.cloud2_scale));
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
        
        // DEFERRED: Don't call setWorld()/resetAccumulation() here!
        // Let Main loop handle it once per frame after flushLUT() to avoid
        // double GPU transfer and ensure LUT is fresh before upload.
        extern bool g_world_dirty;
        extern bool g_gas_volumes_dirty;
        g_world_dirty = true;
        g_gas_volumes_dirty = true;
        
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
                 extern bool g_world_dirty;
                 g_world_dirty = true;
             }
             break; // Sync with first directional light
        }
    }
}
