/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          VolumeShader.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
/**
 * @file VolumeShader.h
 * @brief Industry-standard Volume Shader for VDB rendering
 * 
 * Designed for future node-graph integration. Compatible with Houdini/Blender VDB exports.
 * 
 * @note Grid naming conventions (Houdini standard):
 *   - "density" : Smoke/cloud opacity
 *   - "temperature" : Fire heat (for blackbody emission)
 *   - "vel" : Velocity vector field
 *   - "emission" : Direct light emission
 */

#ifndef VOLUME_SHADER_H
#define VOLUME_SHADER_H

#include "Vec3.h"
#include <string>
#include <memory>
#include <vector>
// Add json forward declaration or include if lightweight, but header-only json library is usually included in cpp.
// For header, we can use forward declaration if possible or just include json.hpp if it is used elsewhere.
// Project uses "json.hpp".
#include "json.hpp"
using json = nlohmann::json;

// Forward declarations for future node system
class ShaderNode;
class ShaderGraph;

/**
 * @brief GPU-exportable volume shader data
 * 
 * This structure is uploaded to GPU for ray marching.
 * Aligned for CUDA compatibility.
 */
struct GpuVolumeShaderData {
    // ─────────────────────────────────────────────────────────────────────────
    // DENSITY
    // ─────────────────────────────────────────────────────────────────────────
    float density_multiplier = 1.0f;
    float density_remap_low = 0.0f;
    float density_remap_high = 1.0f;
    float density_pad = 0.0f;
    
    // ─────────────────────────────────────────────────────────────────────────
    // SCATTERING
    // ─────────────────────────────────────────────────────────────────────────
    float scatter_color_r = 1.0f;
    float scatter_color_g = 1.0f;
    float scatter_color_b = 1.0f;
    float scatter_coefficient = 1.0f;
    
    float scatter_anisotropy = 0.0f;       // G forward (-1 to 1)
    float scatter_anisotropy_back = -0.3f; // G backward
    float scatter_lobe_mix = 0.7f;         // Forward/back blend
    float scatter_multi = 0.3f;            // Multi-scatter approximation
    
    // ─────────────────────────────────────────────────────────────────────────
    // ABSORPTION
    // ─────────────────────────────────────────────────────────────────────────
    float absorption_color_r = 0.0f;
    float absorption_color_g = 0.0f;
    float absorption_color_b = 0.0f;
    float absorption_coefficient = 0.1f;
    
    // ─────────────────────────────────────────────────────────────────────────
    // EMISSION
    // ─────────────────────────────────────────────────────────────────────────
    int emission_mode = 0;  // 0=None, 1=Constant, 2=Blackbody, 3=Channel
    float emission_color_r = 1.0f;
    float emission_color_g = 0.5f;
    float emission_color_b = 0.1f;
    
    float emission_intensity = 0.0f;
    float temperature_scale = 1.0f;
    float blackbody_intensity = 10.0f;
    float emission_pad = 0.0f;
    
    // ─────────────────────────────────────────────────────────────────────────
    // RAY MARCHING QUALITY
    // ─────────────────────────────────────────────────────────────────────────
    float step_size = 0.25f;
    int max_steps = 32;
    int shadow_steps = 6;
    float shadow_strength = 0.8f;
    
    // ─────────────────────────────────────────────────────────────────────────
    // COLOR RAMP
    // ─────────────────────────────────────────────────────────────────────────
    int color_ramp_enabled = 0;
    int ramp_stop_count = 0;
    float ramp_positions[8];
    float ramp_colors_r[8];
    float ramp_colors_g[8];
    float ramp_colors_b[8];
    
    // ─────────────────────────────────────────────────────────────────────────
    // MOTION BLUR
    // ─────────────────────────────────────────────────────────────────────────
    int motion_blur_enabled = 0;
    float velocity_scale = 1.0f;
    float motion_pad1 = 0.0f;
    float motion_pad2 = 0.0f;
};

/**
 * @brief Emission mode for volume shaders
 */
enum class VolumeEmissionMode {
    None = 0,           ///< No emission
    Constant = 1,       ///< Fixed color + intensity
    Blackbody = 2,      ///< Temperature-driven (fire/explosions)
    ChannelDriven = 3   ///< Custom channel → color ramp
};

/**
 * @brief Color Ramp for mapping scalar values to colors
 * 
 * Used for volume emission/color based on density or temperature
 */
struct ColorRampStop {
    float position = 0.0f;  ///< Position on ramp (0-1)
    Vec3 color = Vec3(1.0f);
    float alpha = 1.0f;
};

struct ColorRamp {
    std::vector<ColorRampStop> stops;
    bool enabled = false;
    
    ColorRamp() {
        // Default fire ramp
        stops.push_back({0.0f, Vec3(0.0f, 0.0f, 0.0f), 0.0f});      // Transparent
        stops.push_back({0.2f, Vec3(0.1f, 0.0f, 0.0f), 0.3f});      // Dark red
        stops.push_back({0.4f, Vec3(0.8f, 0.2f, 0.0f), 0.7f});      // Orange
        stops.push_back({0.7f, Vec3(1.0f, 0.6f, 0.1f), 0.9f});      // Yellow-orange
        stops.push_back({1.0f, Vec3(1.0f, 1.0f, 0.8f), 1.0f});      // White-hot
    }
    
    Vec3 sample(float t) const {
        if (stops.empty()) return Vec3(1.0f);
        if (t <= stops[0].position) return stops[0].color;
        if (t >= stops.back().position) return stops.back().color;
        
        for (size_t i = 1; i < stops.size(); ++i) {
            if (t <= stops[i].position) {
                float blend = (t - stops[i-1].position) / 
                              (stops[i].position - stops[i-1].position);
                return stops[i-1].color * (1.0f - blend) + stops[i].color * blend;
            }
        }
        return stops.back().color;
    }
    
    float sampleAlpha(float t) const {
        if (stops.empty()) return 1.0f;
        if (t <= stops[0].position) return stops[0].alpha;
        if (t >= stops.back().position) return stops.back().alpha;
        
        for (size_t i = 1; i < stops.size(); ++i) {
            if (t <= stops[i].position) {
                float blend = (t - stops[i-1].position) / 
                              (stops[i].position - stops[i-1].position);
                return stops[i-1].alpha * (1.0f - blend) + stops[i].alpha * blend;
            }
        }
        return stops.back().alpha;
    }
};


/**
 * @brief Volume Shader - Defines how a volume is rendered
 * 
 * This class is designed to be node-system compatible:
 * - Each property can be a constant OR driven by a node graph (future)
 * - Supports Houdini/Blender-style channel mapping
 * - Emission can be temperature-driven (blackbody) or custom
 * 
 * @example
 * ```cpp
 * auto shader = std::make_shared<VolumeShader>();
 * shader->name = "Fire Shader";
 * shader->emission.mode = VolumeEmissionMode::Blackbody;
 * shader->emission.temperature_channel = "temperature";
 * shader->emission.blackbody_intensity = 15.0f;
 * 
 * vdb_volume->setShader(shader);
 * ```
 */
class VolumeShader {
public:
    std::string name = "Untitled Volume Shader";
    
// ... (previous content)

    // ═══════════════════════════════════════════════════════════════════════════
    // DENSITY PROPERTIES (Controls opacity/scattering)
    // ═══════════════════════════════════════════════════════════════════════════
    struct DensitySettings {
        float multiplier = 1.0f;          ///< Overall density scale
        std::string channel = "density";   ///< VDB grid to sample
        float remap_low = 0.0f;           ///< Remap input range (low)
        float remap_high = 1.0f;          ///< Remap input range (high)
        
        // Edge handling - prevents dark bounding box
        float cutoff_threshold = 0.01f;   ///< Density values below this are ignored (fixes edge artifacts)
        float edge_falloff = 0.0f;        ///< Distance from edge for smooth fade (0 = disabled)
        
        // Serialization
        json toJson() const {
            json j;
            j["multiplier"] = multiplier;
            j["channel"] = channel;
            j["remap_low"] = remap_low;
            j["remap_high"] = remap_high;
            j["cutoff_threshold"] = cutoff_threshold;
            j["edge_falloff"] = edge_falloff;
            return j;
        }

        void fromJson(const json& j) {
            if (j.contains("multiplier")) multiplier = j["multiplier"];
            if (j.contains("channel")) channel = j["channel"];
            if (j.contains("remap_low")) remap_low = j["remap_low"];
            if (j.contains("remap_high")) remap_high = j["remap_high"];
            if (j.contains("cutoff_threshold")) cutoff_threshold = j["cutoff_threshold"];
            if (j.contains("edge_falloff")) edge_falloff = j["edge_falloff"];
        }
    } density;

    
    // ═══════════════════════════════════════════════════════════════════════════
    // SCATTERING PROPERTIES (Light interaction)
    // ═══════════════════════════════════════════════════════════════════════════
    struct ScatteringSettings {
        Vec3 color = Vec3(1.0f);           ///< Scatter albedo
        float coefficient = 1.0f;           ///< Scatter strength (sigma_s)
        float anisotropy = 0.0f;           ///< Phase function G (-1 to 1)
        float anisotropy_back = -0.3f;     ///< Backward scatter G (silver lining)
        float lobe_mix = 0.7f;             ///< Forward/back blend (1=all forward)
        float multi_scatter = 0.3f;        ///< Multi-scatter approximation (0-1)

        // Serialization
        json toJson() const {
            json j;
            j["color"] = {color.x, color.y, color.z};
            j["coefficient"] = coefficient;
            j["anisotropy"] = anisotropy;
            j["anisotropy_back"] = anisotropy_back;
            j["lobe_mix"] = lobe_mix;
            j["multi_scatter"] = multi_scatter;
            return j;
        }

        void fromJson(const json& j) {
            if (j.contains("color")) { auto c = j["color"]; color = Vec3(c[0], c[1], c[2]); }
            if (j.contains("coefficient")) coefficient = j["coefficient"];
            if (j.contains("anisotropy")) anisotropy = j["anisotropy"];
            if (j.contains("anisotropy_back")) anisotropy_back = j["anisotropy_back"];
            if (j.contains("lobe_mix")) lobe_mix = j["lobe_mix"];
            if (j.contains("multi_scatter")) multi_scatter = j["multi_scatter"];
        }
    } scattering;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ABSORPTION PROPERTIES (Light removal)
    // ═══════════════════════════════════════════════════════════════════════════
    struct AbsorptionSettings {
        Vec3 color = Vec3(0.0f);           ///< Absorption color tint
        float coefficient = 0.1f;           ///< Absorption strength (sigma_a)

        // Serialization
        json toJson() const {
            json j;
            j["color"] = {color.x, color.y, color.z};
            j["coefficient"] = coefficient;
            return j;
        }

        void fromJson(const json& j) {
            if (j.contains("color")) { auto c = j["color"]; color = Vec3(c[0], c[1], c[2]); }
            if (j.contains("coefficient")) coefficient = j["coefficient"];
        }
    } absorption;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // EMISSION PROPERTIES (Self-illumination / Fire)
    // ═══════════════════════════════════════════════════════════════════════════
    struct EmissionSettings {
        VolumeEmissionMode mode = VolumeEmissionMode::None;
        
        // Constant mode
        Vec3 color = Vec3(1.0f, 0.5f, 0.1f);  ///< Emission color
        float intensity = 0.0f;               ///< Emission strength
        
        // Blackbody mode (Fire/Explosions)
        std::string temperature_channel = "temperature";  ///< VDB grid for temp
        float temperature_scale = 1.0f;       ///< Temperature multiplier
        float blackbody_intensity = 10.0f;    ///< Blackbody emission strength
        
        // Channel-driven mode
        std::string emission_channel = "emission";  ///< Custom emission grid
        
        // ColorRamp for density/temperature to color mapping
        ColorRamp color_ramp;

         // Serialization
        json toJson() const {
            json j;
            j["mode"] = static_cast<int>(mode);
            j["color"] = {color.x, color.y, color.z};
            j["intensity"] = intensity;
            j["temperature_channel"] = temperature_channel;
            j["temperature_scale"] = temperature_scale;
            j["blackbody_intensity"] = blackbody_intensity;
            j["emission_channel"] = emission_channel;

            // Manual ramp serialization (ColorRamp is outside struct scope, annoying. 
            // Wait, ColorRamp IS defined above. We should add toJson to ColorRamp struct first? 
            // But let's just do it inline here or add to ColorRamp via separate edit.
            // Let's add toJson to ColorRamp struct itself first in separate chunk? 
            // Or just serialize it here.)
            
            json stops_json = json::array();
            for (const auto& s : color_ramp.stops) {
                stops_json.push_back({
                    {"position", s.position},
                    {"color", {s.color.x, s.color.y, s.color.z}},
                    {"alpha", s.alpha}
                });
            }
            j["color_ramp"] = {
                {"enabled", color_ramp.enabled},
                {"stops", stops_json}
            };
            return j;
        }

        void fromJson(const json& j) {
            if (j.contains("mode")) mode = static_cast<VolumeEmissionMode>(j["mode"]);
            if (j.contains("color")) { auto c = j["color"]; color = Vec3(c[0], c[1], c[2]); }
            if (j.contains("intensity")) intensity = j["intensity"];
            if (j.contains("temperature_channel")) temperature_channel = j["temperature_channel"];
            if (j.contains("temperature_scale")) temperature_scale = j["temperature_scale"];
            if (j.contains("blackbody_intensity")) blackbody_intensity = j["blackbody_intensity"];
            if (j.contains("emission_channel")) emission_channel = j["emission_channel"];
            
            if (j.contains("color_ramp")) {
                auto r = j["color_ramp"];
                if (r.contains("enabled")) color_ramp.enabled = r["enabled"];
                if (r.contains("stops")) {
                    color_ramp.stops.clear();
                    for (const auto& s : r["stops"]) {
                        ColorRampStop stop;
                        stop.position = s["position"];
                        auto c = s["color"];
                        stop.color = Vec3(c[0], c[1], c[2]);
                        stop.alpha = s["alpha"];
                        color_ramp.stops.push_back(stop);
                    }
                }
            }
        }

    } emission;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // RAY MARCHING QUALITY
    // ═══════════════════════════════════════════════════════════════════════════
    struct QualitySettings {
        float step_size = 0.15f;            ///< Base step size (auto-adjusted)
        int max_steps = 64;               ///< Maximum steps per ray
        int shadow_steps = 8;              ///< Steps for self-shadowing
        float shadow_strength = 0.8f;      ///< Self-shadow intensity (0-1)
        bool adaptive_stepping = true;     ///< Adjust step based on density

        // Serialization
        json toJson() const {
            json j;
            j["step_size"] = step_size;
            j["max_steps"] = max_steps;
            j["shadow_steps"] = shadow_steps;
            j["shadow_strength"] = shadow_strength;
            j["adaptive_stepping"] = adaptive_stepping;
            return j;
        }

        void fromJson(const json& j) {
            if (j.contains("step_size")) step_size = j["step_size"];
            if (j.contains("max_steps")) max_steps = j["max_steps"];
            if (j.contains("shadow_steps")) shadow_steps = j["shadow_steps"];
            if (j.contains("shadow_strength")) shadow_strength = j["shadow_strength"];
            if (j.contains("adaptive_stepping")) adaptive_stepping = j["adaptive_stepping"];
        }
    } quality;

    // ... (rest of file)

// ...

    static std::shared_ptr<VolumeShader> createExplosionPreset() {
        auto shader = std::make_shared<VolumeShader>();
        shader->name = "Explosion";
        
        shader->density.multiplier = 10.0f;
        shader->scattering.color = Vec3(0.5f, 0.45f, 0.4f);  // Thick grey smoke
        shader->scattering.coefficient = 4.0f;
        shader->scattering.anisotropy = 0.65f;
        shader->scattering.multi_scatter = 0.6f;
        shader->absorption.coefficient = 1.5f;
        shader->absorption.color = Vec3(0.15f, 0.12f, 0.1f);
        
        shader->emission.mode = VolumeEmissionMode::Blackbody;
        shader->emission.temperature_channel = "temperature";
        shader->emission.temperature_scale = 1.2f;
        shader->emission.blackbody_intensity = 35.0f;
        
        // High quality rendering for explosions
        shader->quality.step_size = 0.05f; 
        shader->quality.max_steps = 512;
        shader->quality.shadow_steps = 12;
        shader->quality.shadow_strength = 0.9f;
        
        return shader;
    }

    static std::shared_ptr<VolumeShader> createCloudPreset() {
        auto shader = std::make_shared<VolumeShader>();
        shader->name = "Cloud";
        
        shader->density.multiplier = 0.5f;
        shader->scattering.color = Vec3(1.0f, 1.0f, 1.0f);
        shader->scattering.coefficient = 3.0f;
        shader->scattering.anisotropy = 0.7f;  // Strong forward scatter
        shader->scattering.anisotropy_back = -0.3f;
        shader->scattering.lobe_mix = 0.8f;
        shader->scattering.multi_scatter = 0.6f;
        shader->absorption.coefficient = 0.01f;
        shader->emission.mode = VolumeEmissionMode::None;
        
        shader->quality.step_size = 0.2f;
        shader->quality.max_steps = 512;
        shader->quality.shadow_steps = 8;
        shader->quality.shadow_strength = 0.9f;
        
        return shader;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // MOTION BLUR (Velocity-based)
    // ═══════════════════════════════════════════════════════════════════════════
    struct MotionBlurSettings {
        bool enabled = false;
        std::string velocity_channel = "vel";
        float scale = 1.0f;                ///< Velocity multiplier
    } motion_blur;
    
    // Serialization
    json toJson() const {
        json j;
        j["name"] = name;
        j["density"] = density.toJson();
        j["scattering"] = scattering.toJson();
        j["absorption"] = absorption.toJson();
        j["emission"] = emission.toJson();
        j["quality"] = quality.toJson();
        j["motion_blur"] = {
            {"enabled", motion_blur.enabled},
            {"velocity_channel", motion_blur.velocity_channel},
            {"scale", motion_blur.scale}
        };
        return j;
    }

    void fromJson(const json& j) {
        if (j.contains("name")) name = j["name"];
        if (j.contains("density")) density.fromJson(j["density"]);
        if (j.contains("scattering")) scattering.fromJson(j["scattering"]);
        if (j.contains("absorption")) absorption.fromJson(j["absorption"]);
        if (j.contains("emission")) emission.fromJson(j["emission"]);
        if (j.contains("quality")) quality.fromJson(j["quality"]);
        if (j.contains("motion_blur")) {
            auto mb = j["motion_blur"];
            motion_blur.enabled = mb.value("enabled", false);
            motion_blur.velocity_channel = mb.value("velocity_channel", "vel");
            motion_blur.scale = mb.value("scale", 1.0f);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // GPU EXPORT
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Export shader data for GPU upload
     * @return GPU-compatible shader structure
     */
    GpuVolumeShaderData toGPU() const {
        GpuVolumeShaderData gpu;
        
        // Density
        gpu.density_multiplier = density.multiplier;
        gpu.density_remap_low = density.remap_low;
        gpu.density_remap_high = density.remap_high;
        
        // Scattering
        gpu.scatter_color_r = static_cast<float>(scattering.color.x);
        gpu.scatter_color_g = static_cast<float>(scattering.color.y);
        gpu.scatter_color_b = static_cast<float>(scattering.color.z);
        gpu.scatter_coefficient = scattering.coefficient;
        gpu.scatter_anisotropy = scattering.anisotropy;
        gpu.scatter_anisotropy_back = scattering.anisotropy_back;
        gpu.scatter_lobe_mix = scattering.lobe_mix;
        gpu.scatter_multi = scattering.multi_scatter;
        
        // Absorption
        gpu.absorption_color_r = static_cast<float>(absorption.color.x);
        gpu.absorption_color_g = static_cast<float>(absorption.color.y);
        gpu.absorption_color_b = static_cast<float>(absorption.color.z);
        gpu.absorption_coefficient = absorption.coefficient;
        
        // Emission
        gpu.emission_mode = static_cast<int>(emission.mode);
        gpu.emission_color_r = static_cast<float>(emission.color.x);
        gpu.emission_color_g = static_cast<float>(emission.color.y);
        gpu.emission_color_b = static_cast<float>(emission.color.z);
        gpu.emission_intensity = emission.intensity;
        gpu.temperature_scale = emission.temperature_scale;
        gpu.blackbody_intensity = emission.blackbody_intensity;
        
        // Quality
        gpu.step_size = quality.step_size;
        gpu.max_steps = quality.max_steps;
        gpu.shadow_steps = quality.shadow_steps;
        gpu.shadow_strength = quality.shadow_strength;
        
        // Color Ramp
        gpu.color_ramp_enabled = emission.color_ramp.enabled ? 1 : 0;
        gpu.ramp_stop_count = static_cast<int>(std::min(emission.color_ramp.stops.size(), static_cast<size_t>(8)));
        for (int i = 0; i < gpu.ramp_stop_count; ++i) {
            gpu.ramp_positions[i] = emission.color_ramp.stops[i].position;
            gpu.ramp_colors_r[i] = static_cast<float>(emission.color_ramp.stops[i].color.x);
            gpu.ramp_colors_g[i] = static_cast<float>(emission.color_ramp.stops[i].color.y);
            gpu.ramp_colors_b[i] = static_cast<float>(emission.color_ramp.stops[i].color.z);
        }

        // Motion blur
        gpu.motion_blur_enabled = motion_blur.enabled ? 1 : 0;
        gpu.velocity_scale = motion_blur.scale;
        
        return gpu;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // PRESET FACTORY METHODS
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Create a smoke shader preset
     */
    static std::shared_ptr<VolumeShader> createSmokePreset() {
        auto shader = std::make_shared<VolumeShader>();
        shader->name = "Smoke";
        
        shader->density.multiplier = 2.0f;
        shader->scattering.color = Vec3(0.9f, 0.9f, 0.92f);
        shader->scattering.coefficient = 1.5f;
        shader->scattering.anisotropy = 0.3f;
        shader->absorption.coefficient = 0.05f;
        shader->emission.mode = VolumeEmissionMode::None;
        
        return shader;
    }
    
    /**
     * @brief Create a fire shader preset
     */
    static std::shared_ptr<VolumeShader> createFirePreset() {
        auto shader = std::make_shared<VolumeShader>();
        shader->name = "Fire";
        
        shader->density.multiplier = 1.0f;
        shader->scattering.color = Vec3(1.0f, 0.8f, 0.4f);
        shader->scattering.coefficient = 0.5f;
        shader->scattering.anisotropy = -0.2f;  // Back-scatter for glow
        shader->absorption.coefficient = 0.3f;
        
        shader->emission.mode = VolumeEmissionMode::Blackbody;
        shader->emission.temperature_channel = "temperature";
        shader->emission.temperature_scale = 1.0f;
        shader->emission.blackbody_intensity = 15.0f;
        
        return shader;
    }

};

#endif // VOLUME_SHADER_H

