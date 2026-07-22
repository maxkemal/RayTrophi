/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          WaterSystem.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include "Vec3.h"
#include "Material.h"
#include "Triangle.h"
#include "WaterShaderCommon.h"
#include "json.hpp"
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// ═══════════════════════════════════════════════════════════════════════════════
// WATER SYSTEM DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

enum class WaterBodyType : uint8_t {
    River = 0,
    Lake,
    Reservoir,
    Wetland
};

// Shared terrain-to-water contract. Terrain hydrology owns the analytical
// description; WaterManager consumes it later to build render meshes without
// having to reinterpret masks or solve lake levels a second time. Coordinates
// are terrain-local and metric except for grid coordinates.
struct WaterBodyData {
    uint64_t id = 0;
    int sourceNodeId = -1;
    int terrainId = -1;
    int waterSurfaceId = -1;
    WaterBodyType type = WaterBodyType::Lake;
    std::string name;

    float surfaceElevation = 0.0f;
    float maximumDepth = 0.0f;
    float area = 0.0f;
    float volume = 0.0f;
    int cellCount = 0;

    Vec3 centroid = Vec3(0.0f);
    Vec3 boundsMin = Vec3(0.0f);
    Vec3 boundsMax = Vec3(0.0f);
    Vec3 spillPoint = Vec3(0.0f);
    int spillGridX = -1;
    int spillGridY = -1;
    int outletGridX = -1;
    int outletGridY = -1;
    bool closedBasin = true;
    float fieldValue = 0.0f;
};

struct WaterWaveParams {
    // Wave dynamics
    float wave_speed = 1.0f;
    float wave_strength = 0.5f;
    float wave_frequency = 1.0f;
    
    // Base colors
    Vec3 deep_color = Vec3(0.01f, 0.02f, 0.05f);     // Very dark blue for deep water
    Vec3 shallow_color = Vec3(2.0f/255.0f, 3.0f/255.0f, 3.0f/255.0f); // Dark blue (2, 3, 3) for physical look
    
    // Physics
    float clarity = 0.8f;      // 0.0 = murky, 1.0 = crystal clear
    float foam_level = 0.01f;  // Lower default foam
    float ior = 1.333f;        // Index of Refraction (water = 1.333)
    float roughness = 0.02f;   // Surface micro-roughness
    
    // === ADVANCED: Depth-based rendering ===
    float depth_max = 15.0f;                         // Depth for full deep_color (meters)
    Vec3 absorption_color = Vec3(0.3f, 0.6f, 0.7f); // Absorption tint (what color is absorbed)
    float absorption_density = 0.5f;                 // How quickly light is absorbed
    
    // === ADVANCED: Shore foam ===
    float shore_foam_distance = 1.5f;   // Distance from shore for foam effect
    float shore_foam_intensity = 0.6f;  // Shore foam strength (0-1)
    
    // === ADVANCED: Caustics ===
    float caustic_intensity = 0.4f;     // Caustic brightness
    float caustic_scale = 2.0f;         // Caustic pattern size
    float caustic_speed = 1.0f;         // Caustic animation speed
    
    // === ADVANCED: Sub-surface scattering ===
    float sss_intensity = 0.15f;        // Light scattering inside water
    Vec3 sss_color = Vec3(0.1f, 0.4f, 0.5f);  // SSS tint color
    
    // === ADVANCED: FFT Ocean (Tessendorf) ===
    bool use_fft_ocean = false;         // Enable FFT ocean simulation
    int fft_resolution = 256;           // FFT grid size (64, 128, 256, 512)
    float fft_ocean_size = 100.0f;      // World space coverage (meters)
    bool auto_domain_from_mesh = false; // Deprecated: keep explicit FFT domain stable across mesh scale
    float domain_size_multiplier = 1.0f;// Deprecated: ignored to keep one stable water domain
    float fft_wind_speed = 10.0f;       // Wind speed (m/s) - affects wave size
    float fft_wind_direction = 0.0f;    // Wind direction (degrees)
    float fft_choppiness = 1.0f;        // Horizontal displacement strength
    float fft_amplitude = 0.001f;       // Phillips spectrum amplitude (higher = bigger waves)
    float fft_time_scale = 1.0f;        // Animation speed

    // === ADVANCED: Realistic Details ===
    float micro_detail_strength = 0.05f;// Strength of high-freq noise (ripples) - Default small
    float micro_detail_scale = 20.0f;   // Scale of noise (higher = smaller ripples)
    float micro_anim_speed = 0.1f;      // Animation speed multiplier for micro details
    float micro_morph_speed = 1.0f;     // Morph/shape-change speed for micro details
    float foam_noise_scale = 4.0f;      // Scale of foam breakup noise
    float foam_threshold = 0.4f;        // Offset for foam appearance

    // === GEOMETRIC DISPLACEMENT (Physical Mesh) ===
    enum class NoiseType { Perlin, FBM, Ridge, Voronoi, Billow, Gerstner, TessendorfSimple };
    NoiseType geo_noise_type = NoiseType::Ridge;

    bool use_geometric_waves = false;
    float geo_wave_height = 2.0f;       // Amplitude
    float geo_wave_scale = 5.0f;        // Global Scale (world units; keep <= mesh size for visible waves)
    float geo_wave_choppiness = 1.0f;   // Ridge Offset / Sharpness
    float geo_wave_speed = 0.5f;        // Animation Speed (Phase Shift)
    
    // Detailed Noise Params
    int geo_octaves = 4;
    float geo_persistence = 0.5f;
    float geo_lacunarity = 2.0f;
    float geo_ridge_offset = 1.0f;
    
    // ===  OCEAN PARAMS ===
    float geo_damping = 0.0f;           // Damping for wind perpendicular waves (0-1)
    float geo_alignment = 0.5f;         // Wave alignment to wind direction (0=omni, 1=aligned)
    float geo_depth = 200.0f;           // Ocean depth in meters (affects shallow water behavior)
    float geo_swell_direction = 0.0f;   // Swell direction offset (degrees)
    float geo_swell_amplitude = 0.2f;   // Swell (long-distance waves) contribution
    float geo_sharpening = 0.0f;        // Post-process sharpening (0=smooth, 1=peaked waves)
    float geo_detail_scale = 3.0f;      // Secondary detail noise scale multiplier
    float geo_detail_strength = 0.15f;  // Secondary detail noise strength
    
    // Smooth Normals
    bool geo_smooth_normals = true;     // Enable smooth shading (vertex normal averaging)
    
    // === FFT-DRIVEN MESH DISPLACEMENT (Best Quality - Combines FFT + Mesh) ===
    bool use_fft_mesh_displacement = false;  // Use FFT data to displace mesh vertices
    float fft_mesh_height_scale = 50.0f;     // Amplification for FFT height (raw FFT values are small)
    float fft_mesh_choppiness = 1.0f;        // Scale factor for FFT horizontal displacement
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // WATER PRESETS - Quick setup for common water types
    // ═══════════════════════════════════════════════════════════════════════════════
    enum class WaterPreset { 
        Custom,         // User-defined settings
        CalmOcean,      // Calm open ocean - gentle swells
        StormyOcean,    // Stormy sea - high waves, lots of foam
        TropicalOcean,  // Crystal clear tropical water
        Lake,           // Still lake - minimal waves
        River,          // Flowing river with current
        Pool,           // Swimming pool - very calm
        Pond            // Small pond with subtle ripples
    };
    WaterPreset current_preset = WaterPreset::Custom;

    // Complete, reusable material contract. WaterSurface and RiverSpline both
    // persist this payload so rebuilding a linked river cannot replace a saved
    // preset/custom material with RiverSpline's defaults.
    nlohmann::json serializeParams() const;
    void deserializeParams(const nlohmann::json& j);

    WaterShader::SurfaceParams toShaderParams(float time_seconds = 0.0f, float resolved_domain_size = -1.0f) const {
        WaterShader::SurfaceParams out;
        if (current_preset == WaterPreset::River) {
            out.profile = WaterShader::SurfaceProfile::River;
        } else if (current_preset == WaterPreset::Lake ||
                   current_preset == WaterPreset::Pond ||
                   current_preset == WaterPreset::Pool) {
            out.profile = WaterShader::SurfaceProfile::Lake;
        }
        out.wave_speed = wave_speed;
        out.wave_strength = wave_strength;
        out.wave_frequency = wave_frequency;
        out.shallow_color = shallow_color;
        out.deep_color = deep_color;
        out.absorption_color = absorption_color;
        out.depth_max = depth_max;
        out.absorption_density = absorption_density;
        out.clarity = clarity;
        out.ior = ior;
        out.roughness = roughness;
        out.foam_level = foam_level;
        out.shore_foam_distance = shore_foam_distance;
        out.shore_foam_intensity = shore_foam_intensity;
        out.caustic_intensity = caustic_intensity;
        out.caustic_scale = caustic_scale;
        out.caustic_speed = caustic_speed;
        out.sss_intensity = sss_intensity;
        out.sss_color = sss_color;
        out.use_fft_ocean = use_fft_ocean;
        out.fft_ocean_size = resolved_domain_size > 0.0f ? resolved_domain_size : fft_ocean_size;
        out.fft_choppiness = fft_choppiness;
        out.fft_amplitude = fft_amplitude;
        out.animation_speed = fft_time_scale;
        out.micro_detail_strength = micro_detail_strength;
        out.micro_detail_scale = micro_detail_scale;
        out.micro_anim_speed = micro_anim_speed;
        out.micro_morph_speed = micro_morph_speed;
        out.foam_noise_scale = foam_noise_scale;
        out.foam_threshold = foam_threshold;
        out.wind_direction = fft_wind_direction;
        out.wind_speed = fft_wind_speed;
        out.time = time_seconds;
        return out;
    }
    
    // Apply preset values
    void applyPreset(WaterPreset preset) {
        current_preset = preset;
        
        switch (preset) {
            case WaterPreset::CalmOcean:
                wave_speed = 0.55f;
                wave_strength = 0.18f;
                wave_frequency = 0.65f;
                // FFT settings for calm ocean
                use_fft_ocean = true;
                fft_resolution = 256;
                fft_ocean_size = 100.0f;      // 100m tile
                fft_wind_speed = 10.0f;       // Moderate wind
                fft_wind_direction = 0.0f;
                fft_choppiness = 1.0f;
                fft_amplitude = 0.001f;       // Higher for visible waves
                fft_time_scale = 1.0f;
                // Appearance
                deep_color = Vec3(0.01f, 0.03f, 0.08f);
                shallow_color = Vec3(0.05f, 0.15f, 0.2f);
                roughness = 0.02f;
                foam_level = 0.05f;
                // Micro details
                micro_detail_strength = 0.03f;
                micro_detail_scale = 15.0f;
                micro_anim_speed = 0.05f;
                micro_morph_speed = 0.3f;
                // FFT Mesh - Direct height control
                use_fft_mesh_displacement = true;
                fft_mesh_height_scale = 50.0f;   // Amplify FFT output significantly
                fft_mesh_choppiness = 1.0f;
                break;
                
            case WaterPreset::StormyOcean:
                wave_speed = 1.50f;
                wave_strength = 0.80f;
                wave_frequency = 1.20f;
                use_fft_ocean = true;
                fft_resolution = 256;
                fft_ocean_size = 150.0f;
                fft_wind_speed = 25.0f;       // Strong wind
                fft_wind_direction = 0.0f;
                fft_choppiness = 2.0f;
                fft_amplitude = 0.003f;       // High amplitude for storms
                fft_time_scale = 1.2f;
                // Dark stormy colors
                deep_color = Vec3(0.02f, 0.04f, 0.06f);
                shallow_color = Vec3(0.1f, 0.15f, 0.18f);
                roughness = 0.05f;
                foam_level = 0.4f;
                foam_threshold = 0.25f;
                // Aggressive micro details
                micro_detail_strength = 0.08f;
                micro_detail_scale = 25.0f;
                micro_anim_speed = 0.15f;
                micro_morph_speed = 0.8f;
                // FFT Mesh
                use_fft_mesh_displacement = true;
                fft_mesh_height_scale = 80.0f;   // High waves
                fft_mesh_choppiness = 2.5f;
                break;
                
            case WaterPreset::TropicalOcean:
                wave_speed = 0.45f;
                wave_strength = 0.12f;
                wave_frequency = 0.55f;
                use_fft_ocean = true;
                fft_resolution = 256;
                fft_ocean_size = 80.0f;
                fft_wind_speed = 5.0f;
                fft_wind_direction = 0.0f;
                fft_choppiness = 0.5f;
                fft_amplitude = 0.0005f;
                fft_time_scale = 0.6f;
                // Crystal clear tropical
                deep_color = Vec3(0.0f, 0.05f, 0.1f);
                shallow_color = Vec3(0.1f, 0.4f, 0.5f);
                clarity = 0.95f;
                roughness = 0.01f;
                foam_level = 0.02f;
                absorption_density = 0.2f;
                // Very subtle micro details
                micro_detail_strength = 0.02f;
                micro_detail_scale = 10.0f;
                micro_anim_speed = 0.03f;
                micro_morph_speed = 0.2f;
                // FFT Mesh
                use_fft_mesh_displacement = true;
                fft_mesh_height_scale = 30.0f;
                fft_mesh_choppiness = 0.5f;
                break;
                
            case WaterPreset::Lake:
                wave_speed = 0.25f;
                wave_strength = 0.035f;
                wave_frequency = 0.45f;
                use_fft_ocean = true;
                fft_resolution = 128;
                fft_ocean_size = 50.0f;
                fft_wind_speed = 3.0f;
                fft_wind_direction = 0.0f;
                fft_choppiness = 0.3f;
                fft_amplitude = 0.0003f;
                fft_time_scale = 0.4f;
                // Lake colors
                deep_color = Vec3(0.02f, 0.05f, 0.08f);
                shallow_color = Vec3(0.08f, 0.2f, 0.25f);
                clarity = 0.7f;
                roughness = 0.01f;
                foam_level = 0.0f;
                // Almost no micro details - glassy
                micro_detail_strength = 0.01f;
                micro_detail_scale = 8.0f;
                micro_anim_speed = 0.02f;
                micro_morph_speed = 0.1f;
                // Gentle mesh displacement
                use_fft_mesh_displacement = true;
                fft_mesh_height_scale = 15.0f;
                fft_mesh_choppiness = 0.3f;
                break;
                
            case WaterPreset::River:
                wave_speed = 1.50f;
                wave_strength = 0.06f;
                wave_frequency = 1.10f;
                // Rivers are spline/UV-flow surfaces, not cropped ocean tiles.
                // Vulkan RT animates their normals and foam in the ribbon frame.
                use_fft_ocean = false;
                fft_resolution = 128;
                fft_ocean_size = 30.0f;  // Smaller for river
                fft_wind_speed = 4.0f;
                fft_wind_direction = 0.0f;  // Fixed flow direction
                fft_choppiness = 0.5f;
                fft_amplitude = 0.0004f;
                fft_time_scale = 1.5f;    // Faster animation
                // River colors
                deep_color = Vec3(0.03f, 0.06f, 0.05f);
                shallow_color = Vec3(0.1f, 0.18f, 0.15f);
                clarity = 0.5f;
                roughness = 0.03f;
                foam_level = 0.1f;
                // Directional micro details (flow feeling)
                micro_detail_strength = 0.04f;
                micro_detail_scale = 12.0f;
                micro_anim_speed = 0.2f;  // Faster - flowing
                micro_morph_speed = 0.5f;
                // River mesh displacement
                use_fft_mesh_displacement = false;
                fft_mesh_height_scale = 20.0f;
                fft_mesh_choppiness = 0.5f;
                break;
                
            case WaterPreset::Pool:
                wave_speed = 0.12f;
                wave_strength = 0.01f;
                wave_frequency = 0.35f;
                use_fft_ocean = false;  // No FFT - very calm
                use_fft_mesh_displacement = false;
                // Pool colors
                deep_color = Vec3(0.0f, 0.1f, 0.2f);
                shallow_color = Vec3(0.2f, 0.5f, 0.6f);
                clarity = 1.0f;
                roughness = 0.005f;
                foam_level = 0.0f;
                // Minimal ripples
                micro_detail_strength = 0.005f;
                micro_detail_scale = 5.0f;
                micro_anim_speed = 0.01f;
                micro_morph_speed = 0.05f;
                break;
                
            case WaterPreset::Pond:
                wave_speed = 0.18f;
                wave_strength = 0.02f;
                wave_frequency = 0.40f;
                use_fft_ocean = true;
                fft_resolution = 64;
                fft_ocean_size = 20.0f;
                fft_wind_speed = 2.0f;
                fft_wind_direction = 0.0f;
                fft_choppiness = 0.2f;
                fft_amplitude = 0.0001f;
                fft_time_scale = 0.3f;
                // Murky pond
                deep_color = Vec3(0.02f, 0.04f, 0.03f);
                shallow_color = Vec3(0.08f, 0.12f, 0.08f);
                clarity = 0.4f;
                roughness = 0.02f;
                foam_level = 0.0f;
                // Very subtle
                micro_detail_strength = 0.01f;
                micro_detail_scale = 6.0f;
                micro_anim_speed = 0.015f;
                micro_morph_speed = 0.1f;
                // Light displacement
                use_fft_mesh_displacement = true;
                fft_mesh_height_scale = 10.0f;
                fft_mesh_choppiness = 0.2f;
                break;
                
            case WaterPreset::Custom:
            default:
                // Keep current values
                break;
        }

        // Vulkan RT is the authoritative water path. The previous ocean mode
        // ran CUDA FFT, downloaded it to the CPU, then uploaded it again to
        // Vulkan every refresh. Keep those legacy fields only for the dormant
        // CPU/OptiX fallback until the native Vulkan compute spectrum lands.
        if (preset != WaterPreset::Custom) {
            use_fft_ocean = false;
            use_fft_mesh_displacement = false;
            use_geometric_waves = false;
        }
    }
};

inline nlohmann::json WaterWaveParams::serializeParams() const {
    return {
        {"wave_speed", wave_speed}, {"wave_strength", wave_strength}, {"wave_frequency", wave_frequency},
        {"deep_color", {deep_color.x, deep_color.y, deep_color.z}},
        {"shallow_color", {shallow_color.x, shallow_color.y, shallow_color.z}},
        {"clarity", clarity}, {"foam_level", foam_level}, {"ior", ior}, {"roughness", roughness},
        {"depth_max", depth_max},
        {"absorption_color", {absorption_color.x, absorption_color.y, absorption_color.z}},
        {"absorption_density", absorption_density},
        {"shore_foam_distance", shore_foam_distance}, {"shore_foam_intensity", shore_foam_intensity},
        {"caustic_intensity", caustic_intensity}, {"caustic_scale", caustic_scale},
        {"caustic_speed", caustic_speed}, {"sss_intensity", sss_intensity},
        {"sss_color", {sss_color.x, sss_color.y, sss_color.z}},
        {"use_fft_ocean", use_fft_ocean}, {"fft_resolution", fft_resolution},
        {"fft_ocean_size", fft_ocean_size}, {"auto_domain_from_mesh", auto_domain_from_mesh},
        {"domain_size_multiplier", domain_size_multiplier}, {"fft_wind_speed", fft_wind_speed},
        {"fft_wind_direction", fft_wind_direction}, {"fft_choppiness", fft_choppiness},
        {"fft_amplitude", fft_amplitude}, {"fft_time_scale", fft_time_scale},
        {"micro_detail_strength", micro_detail_strength}, {"micro_detail_scale", micro_detail_scale},
        {"micro_anim_speed", micro_anim_speed}, {"micro_morph_speed", micro_morph_speed},
        {"foam_noise_scale", foam_noise_scale}, {"foam_threshold", foam_threshold},
        {"geo_noise_type", static_cast<int>(geo_noise_type)},
        {"use_geometric_waves", use_geometric_waves}, {"geo_wave_height", geo_wave_height},
        {"geo_wave_scale", geo_wave_scale}, {"geo_wave_choppiness", geo_wave_choppiness},
        {"geo_wave_speed", geo_wave_speed}, {"geo_octaves", geo_octaves},
        {"geo_persistence", geo_persistence}, {"geo_lacunarity", geo_lacunarity},
        {"geo_ridge_offset", geo_ridge_offset}, {"geo_damping", geo_damping},
        {"geo_alignment", geo_alignment}, {"geo_depth", geo_depth},
        {"geo_swell_direction", geo_swell_direction}, {"geo_swell_amplitude", geo_swell_amplitude},
        {"geo_sharpening", geo_sharpening}, {"geo_detail_scale", geo_detail_scale},
        {"geo_detail_strength", geo_detail_strength}, {"geo_smooth_normals", geo_smooth_normals},
        {"use_fft_mesh_displacement", use_fft_mesh_displacement},
        {"fft_mesh_height_scale", fft_mesh_height_scale},
        {"fft_mesh_choppiness", fft_mesh_choppiness},
        {"current_preset", static_cast<int>(current_preset)}
    };
}

inline void WaterWaveParams::deserializeParams(const nlohmann::json& j) {
    wave_speed = j.value("wave_speed", wave_speed);
    wave_strength = j.value("wave_strength", wave_strength);
    wave_frequency = j.value("wave_frequency", wave_frequency);
    const auto readColor = [&](const char* key, Vec3& color) {
        if (j.contains(key) && j[key].is_array() && j[key].size() >= 3) {
            color = Vec3(j[key][0].get<float>(), j[key][1].get<float>(), j[key][2].get<float>());
        }
    };
    readColor("deep_color", deep_color);
    readColor("shallow_color", shallow_color);
    clarity = j.value("clarity", clarity); foam_level = j.value("foam_level", foam_level);
    ior = j.value("ior", ior); roughness = j.value("roughness", roughness);
    depth_max = j.value("depth_max", depth_max); readColor("absorption_color", absorption_color);
    absorption_density = j.value("absorption_density", absorption_density);
    shore_foam_distance = j.value("shore_foam_distance", shore_foam_distance);
    shore_foam_intensity = j.value("shore_foam_intensity", shore_foam_intensity);
    caustic_intensity = j.value("caustic_intensity", caustic_intensity);
    caustic_scale = j.value("caustic_scale", caustic_scale);
    caustic_speed = j.value("caustic_speed", caustic_speed);
    sss_intensity = j.value("sss_intensity", sss_intensity); readColor("sss_color", sss_color);
    use_fft_ocean = j.value("use_fft_ocean", use_fft_ocean);
    fft_resolution = j.value("fft_resolution", fft_resolution);
    fft_ocean_size = j.value("fft_ocean_size", fft_ocean_size);
    auto_domain_from_mesh = j.value("auto_domain_from_mesh", auto_domain_from_mesh);
    domain_size_multiplier = j.value("domain_size_multiplier", domain_size_multiplier);
    fft_wind_speed = j.value("fft_wind_speed", fft_wind_speed);
    fft_wind_direction = j.value("fft_wind_direction", fft_wind_direction);
    fft_choppiness = j.value("fft_choppiness", fft_choppiness);
    fft_amplitude = j.value("fft_amplitude", fft_amplitude);
    fft_time_scale = j.value("fft_time_scale", fft_time_scale);
    micro_detail_strength = j.value("micro_detail_strength", micro_detail_strength);
    micro_detail_scale = j.value("micro_detail_scale", micro_detail_scale);
    micro_anim_speed = j.value("micro_anim_speed", micro_anim_speed);
    micro_morph_speed = j.value("micro_morph_speed", micro_morph_speed);
    foam_noise_scale = j.value("foam_noise_scale", foam_noise_scale);
    foam_threshold = j.value("foam_threshold", foam_threshold);
    geo_noise_type = static_cast<NoiseType>(j.value("geo_noise_type", static_cast<int>(geo_noise_type)));
    use_geometric_waves = j.value("use_geometric_waves", use_geometric_waves);
    geo_wave_height = j.value("geo_wave_height", geo_wave_height);
    geo_wave_scale = j.value("geo_wave_scale", geo_wave_scale);
    geo_wave_choppiness = j.value("geo_wave_choppiness", geo_wave_choppiness);
    geo_wave_speed = j.value("geo_wave_speed", geo_wave_speed);
    geo_octaves = j.value("geo_octaves", geo_octaves);
    geo_persistence = j.value("geo_persistence", geo_persistence);
    geo_lacunarity = j.value("geo_lacunarity", geo_lacunarity);
    geo_ridge_offset = j.value("geo_ridge_offset", geo_ridge_offset);
    geo_damping = j.value("geo_damping", geo_damping);
    geo_alignment = j.value("geo_alignment", geo_alignment);
    geo_depth = j.value("geo_depth", geo_depth);
    geo_swell_direction = j.value("geo_swell_direction", geo_swell_direction);
    geo_swell_amplitude = j.value("geo_swell_amplitude", geo_swell_amplitude);
    geo_sharpening = j.value("geo_sharpening", geo_sharpening);
    geo_detail_scale = j.value("geo_detail_scale", geo_detail_scale);
    geo_detail_strength = j.value("geo_detail_strength", geo_detail_strength);
    geo_smooth_normals = j.value("geo_smooth_normals", geo_smooth_normals);
    use_fft_mesh_displacement = j.value("use_fft_mesh_displacement", use_fft_mesh_displacement);
    fft_mesh_height_scale = j.value("fft_mesh_height_scale", fft_mesh_height_scale);
    fft_mesh_choppiness = j.value("fft_mesh_choppiness", fft_mesh_choppiness);
    current_preset = static_cast<WaterPreset>(j.value("current_preset", static_cast<int>(current_preset)));
}

class TriangleMesh;

struct WaterSurface {
    // Keep new values appended for serialized enum stability.
    enum class Type { Plane, River, Custom, Lake };
    Type type = Type::Plane;
    
    int id = -1;
    std::string name;
    WaterWaveParams params;

    // Persistent identity for surfaces rebuilt by a terrain graph. Runtime
    // water/river IDs and display names may change after Evaluate, so authored
    // material settings must be associated with the producer and geometry.
    bool has_generated_identity = false;
    int generated_terrain_id = -1;
    uint32_t generated_node_id = 0;
    uint64_t generated_feature_id = 0;
    Vec3 generated_anchor = Vec3(0.0f); // terrain-local for lakes, world-space for rivers
    float generated_extent = 0.0f;      // lake area or river length
    
    // The physics mesh (flat TriangleMesh structure)
    std::shared_ptr<TriangleMesh> flatMesh;
    // Generated terrain/ocean water owns its scene object. A Water modifier only
    // binds an existing evaluated mesh and must never delete that object.
    bool owns_scene_mesh = true;
    
    // Original vertex positions (for animation - keeps base grid positions)
    std::vector<Vec3> original_positions;
    std::vector<Vec3> original_normals;
    size_t bound_index_count = 0; // Runtime topology guard for modifier rebinding.
    
    // Material ID used for this water
    uint16_t material_id = 0;
    
    // Runtime FFT State (opaque handle to FFTOceanState)
    void* fft_state = nullptr;
    
    // Runtime GPU Geometric Wave State (opaque handle to GPUGeoWaveState)
    void* gpu_geo_state = nullptr;
    
    // Animation state
    float animation_time = 0.0f;
    bool animate_mesh = false;      // Enable mesh animation
    bool use_gpu_animation = true;  // Use GPU for geometric waves (faster)
    int64_t vulkan_fft_height_texture = 0;
    int64_t vulkan_fft_normal_texture = 0;
};

enum class WaterPreviewTimeMode {
    Realtime,
    Timeline,
    Static
};

struct WaterUpdateResult {
    bool time_changed = false;
    bool material_changed = false;
    bool mesh_changed = false;

    bool requiresAccumulationReset() const {
        return time_changed || material_changed || mesh_changed;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// WATER MANAGER CLASS
// ═══════════════════════════════════════════════════════════════════════════════

struct SceneData;
class Renderer;
class OptixWrapper;
namespace Backend { class IBackend; }

class WaterManager {
public:
    static WaterManager& getInstance() {
        static WaterManager instance;
        return instance;
    }
    
    // Create a new water plane at given position
    WaterSurface* createWaterPlane(SceneData& scene, const Vec3& pos, float size, float density);
    
    // Create water from existing mesh triangles (e.g. from Terrain)
    WaterSurface* createWaterFromMesh(SceneData& scene, const std::string& name, const std::vector<std::shared_ptr<Triangle>>& triangles);

    // Shared indexed-mesh entry used by terrain lakes and future reservoirs.
    WaterSurface* createWaterFromIndexedMesh(
        SceneData& scene,
        const std::string& name,
        const std::vector<Vec3>& positions,
        const std::vector<Vec2>& uvs,
        const std::vector<uint32_t>& indices,
        const std::shared_ptr<Transform>& transform,
        WaterSurface::Type type,
        const WaterWaveParams& params,
        const std::vector<float>& waterDepth = {},
        const std::vector<float>& shoreFactor = {});

    // Non-destructive modifier entry: register or rebind an existing evaluated
    // TriangleMesh without copying it or adding a duplicate scene object.
    WaterSurface* bindExistingWaterMesh(
        const std::shared_ptr<TriangleMesh>& mesh,
        WaterSurface::Type type = WaterSurface::Type::Plane);
    WaterSurface* getWaterSurfaceByNodeName(const std::string& nodeName);
    
    // Update mesh geometry based on physics parameters (static, called once)
    void updateWaterMesh(WaterSurface* surf);
    
    // Update mesh with animation - CPU path (called each frame for animated surfaces)
    bool updateAnimatedWaterMesh(WaterSurface* surf, float time);
    bool restoreAnimatedWaterMesh(WaterSurface* surf);
    
    // Update mesh with animation - GPU path (much faster for large meshes)
    bool updateGPUAnimatedWaterMesh(WaterSurface* surf, float time);
    
    // Update mesh using FFT ocean data - highest quality (GPU accelerated)
    bool updateFFTDrivenMesh(WaterSurface* surf, float time);
    
    // Store original positions for animation (call after initial mesh creation)
    void cacheOriginalPositions(WaterSurface* surf);
    void invalidateGeometricAnimationState(WaterSurface* surf);
    
    // Get all water surfaces
    std::vector<WaterSurface>& getWaterSurfaces() { return water_surfaces; }
    
    // Get water surface by ID
    WaterSurface* getWaterSurface(int id);
    
    // Updates all water surfaces for an absolute water time (seconds).
    WaterUpdateResult update(float waterTime);

    // Synchronize the water surface params into its bound PrincipledBSDF/GpuMaterial.
    void syncSurfaceMaterial(WaterSurface* surf);
    bool syncVulkanFFTTexturesForMaterial(uint16_t material_id, Backend::IBackend* backend, int64_t& outHeightTexture, int64_t& outNormalTexture);
    float getSurfaceWorldExtent(const WaterSurface* surf) const;
    float resolveWaveDomainSize(const WaterSurface* surf) const;
    float getLegacyDomainReferenceSize() const;
    float resolveSharedAnimationSpeed(const WaterSurface* surf) const;

    void setPreviewTimeMode(WaterPreviewTimeMode mode);
    WaterPreviewTimeMode getPreviewTimeMode() const { return preview_time_mode; }
    float resolvePreviewWaterTime(float realtimeSeconds, int timelineFrame, float fps);
    
    // Returns the height map texture of the first active FFT water surface (or 0)
    cudaTextureObject_t getFirstFFTHeightMap();
    
    // Remove water surface
    void removeWaterSurface(SceneData& scene, int id);
    
    // Clear all water surfaces (for new/load project)
    void clear();

    // Release backend-dependent CUDA simulation state without deleting water
    // surfaces or their authored parameters. Recreated lazily when needed.
    void releaseRuntimeGPUResources();
    
    // Apply keyframe values to water surface and rebuild mesh
    void applyKeyframe(WaterSurface* surf, const struct WaterKeyframe& keyframe);
    
    // Capture current state to keyframe track (for recording)
    void captureKeyframeToTrack(WaterSurface* surf, struct ObjectAnimationTrack& track, int frame);
    
    // Apply interpolated keyframe from track (for playback)
    void updateFromTrack(WaterSurface* surf, const struct ObjectAnimationTrack& track, int currentFrame);
    
    // Serialization
    nlohmann::json serialize() const;
    void deserialize(const nlohmann::json& j, SceneData& scene);

private:
    WaterManager() = default;
    
    std::vector<WaterSurface> water_surfaces;
    int next_id = 1;
    WaterPreviewTimeMode preview_time_mode = WaterPreviewTimeMode::Static;
    float last_resolved_preview_time = 0.0f;
    float static_preview_time = 0.0f;
    float last_simulation_time = 0.0f;
    bool has_last_resolved_preview_time = false;
    bool has_last_simulation_time = false;
};

