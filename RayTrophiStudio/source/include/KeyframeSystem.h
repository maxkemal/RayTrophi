/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          KeyframeSystem.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include "Vec3.h"
#include "material_gpu.h"  // For GpuMaterial
#include "json.hpp"
#include "world.h"
using json = nlohmann::json;

// Vec3 Serialization
inline void to_json(json& j, const Vec3& v) {
    j = json::array({v.x, v.y, v.z});
}

inline void from_json(const json& j, Vec3& v) {
    if(j.is_array() && j.size() >= 3) {
        v.x = j[0]; v.y = j[1]; v.z = j[2];
    } else {
        v = Vec3(0,0,0);
    }
}

// Interpolation mode of the segment that STARTS at this key (this key -> next key).
enum class KeyInterp : uint8_t {
    Constant = 0,   // hold this key's value until the next key
    Linear   = 1,   // straight lerp (legacy behavior)
    Bezier   = 2    // cubic 2D Bezier driven by in/out handles
};

// Per-channel curve metadata stored on each transform key.
// Handles are RELATIVE offsets from the key: dx in frames, dy in channel units.
// in_* is the left handle (dx <= 0), out_* the right handle (dx >= 0).
// New keys default to Bezier+auto (smooth motion); legacy project files
// deserialize as Linear so old scenes evaluate bit-identical to before.
struct ChannelKeyMeta {
    KeyInterp interp = KeyInterp::Bezier;
    bool auto_tangent = true;   // handles recomputed from neighbors (auto-clamped Catmull-Rom)
    float in_dx = 0.0f,  in_dy = 0.0f;
    float out_dx = 0.0f, out_dy = 0.0f;

    // Legacy-equivalent state: what an old project file (no curve data) loads as.
    // Used to keep serialization sparse for untouched keys.
    bool isLegacyLinear() const {
        return interp == KeyInterp::Linear && auto_tangent &&
               in_dx == 0.0f && in_dy == 0.0f && out_dx == 0.0f && out_dy == 0.0f;
    }
    // Factory state of a freshly authored key (no user customization yet).
    bool isPristineAuto() const {
        return interp == KeyInterp::Bezier && auto_tangent &&
               in_dx == 0.0f && in_dy == 0.0f && out_dx == 0.0f && out_dy == 0.0f;
    }
    void resetToLegacyLinear() {
        interp = KeyInterp::Linear; auto_tangent = true;
        in_dx = in_dy = out_dx = out_dy = 0.0f;
    }
};

// ============================================================================
// KEYFRAME SYSTEM - Object-based animation with transform + material props
// ============================================================================
// Channel counts and indices
enum : int {
    CURVE_LIGHT_POS_X = 0, CURVE_LIGHT_POS_Y, CURVE_LIGHT_POS_Z,
    CURVE_LIGHT_COLOR_R, CURVE_LIGHT_COLOR_G, CURVE_LIGHT_COLOR_B,
    CURVE_LIGHT_INTENSITY,
    CURVE_LIGHT_DIR_X, CURVE_LIGHT_DIR_Y, CURVE_LIGHT_DIR_Z,
    CURVE_LIGHT_CHANNEL_COUNT
};

enum : int {
    CURVE_CAM_POS_X = 0, CURVE_CAM_POS_Y, CURVE_CAM_POS_Z,
    CURVE_CAM_TGT_X, CURVE_CAM_TGT_Y, CURVE_CAM_TGT_Z,
    CURVE_CAM_FOV,
    CURVE_CAM_FOCUS_DIST,
    CURVE_CAM_LENS_RAD,
    CURVE_CAM_CHANNEL_COUNT
};

enum : int {
    CURVE_MAT_ALBEDO_R = 0, CURVE_MAT_ALBEDO_G, CURVE_MAT_ALBEDO_B,
    CURVE_MAT_OPACITY,
    CURVE_MAT_ROUGHNESS,
    CURVE_MAT_METALLIC,
    CURVE_MAT_CLEARCOAT,
    CURVE_MAT_TRANSMISSION,
    CURVE_MAT_IOR,
    CURVE_MAT_EMISSION_R, CURVE_MAT_EMISSION_G, CURVE_MAT_EMISSION_B,
    CURVE_MAT_NORMAL_STRENGTH,
    CURVE_MAT_EMISSION_STRENGTH,
    CURVE_MAT_CHANNEL_COUNT
};

// Material property keyframe - ALIGNED WITH GpuMaterial for GPU/CPU compatibility
struct MaterialKeyframe {
    // Material identification
    uint16_t material_id = 0;
    
    // Per-Property Flags (compound)
    bool has_albedo = false;
    bool has_opacity = false; // NEW: Separated from albedo
    bool has_roughness = false;
    bool has_metallic = false;
    bool has_emission = false;
    bool has_transmission = false;
    bool has_ior = false;
    bool has_clearcoat = false;
    bool has_subsurface = false;
    bool has_sheen = false;
    bool has_anisotropic = false;
    bool has_specular = false;
    bool has_normal = false;

    // Detailed per-axis flags
    bool has_alb_r = true; bool has_alb_g = true; bool has_alb_b = true;
    bool has_opac = true;
    bool has_rough = true;
    bool has_metal = true;
    bool has_clear = true;
    bool has_transm = true;
    bool has_ior_val = true;
    bool has_emis_r = true; bool has_emis_g = true; bool has_emis_b = true;
    bool has_norm_str = true;
    bool has_emis_str = true;
    
    // Block 1: Albedo + opacity
    Vec3 albedo = Vec3(0.8, 0.8, 0.8);
    float opacity = 1.0f;
    
    // Block 2: PBR core properties
    float roughness = 0.5f;
    float metallic = 0.0f;
    float clearcoat = 0.0f;
    float transmission = 0.0f;
    
    // Block 3: Emission + IOR
    Vec3 emission = Vec3(0, 0, 0);
    float ior = 1.45f;
    
    // Block 4: Subsurface
    Vec3 subsurface_color = Vec3(0, 0, 0);
    float subsurface = 0.0f;
    
    // Block 5: Additional properties
    float anisotropic = 0.0f;
    float sheen = 0.0f;
    float sheen_tint = 0.0f;
    
    // Extra CPU-only properties
    float specular = 0.5f;
    float specular_tint = 0.0f;
    float clearcoat_roughness = 0.0f;
    float normal_strength = 1.0f;
    float emission_strength = 1.0f;

    // Per-channel curve metadata (graph editor)
    ChannelKeyMeta curve[CURVE_MAT_CHANNEL_COUNT];
    
    MaterialKeyframe() = default;
    
    MaterialKeyframe(const GpuMaterial& gpu) {
        albedo = Vec3(gpu.albedo.x, gpu.albedo.y, gpu.albedo.z);
        opacity = gpu.opacity;
        roughness = gpu.roughness;
        metallic = gpu.metallic;
        clearcoat = gpu.clearcoat;
        transmission = gpu.transmission;
        emission = Vec3(gpu.emission.x, gpu.emission.y, gpu.emission.z);
        ior = gpu.ior;
        subsurface_color = Vec3(gpu.subsurface_color.x, gpu.subsurface_color.y, gpu.subsurface_color.z);
        subsurface = gpu.subsurface;
        anisotropic = gpu.anisotropic;
        sheen = gpu.sheen;
        sheen_tint = gpu.sheen_tint;
        
        specular = gpu.specular;
        specular_tint = 0.0f;
        clearcoat_roughness = 0.1f;
        normal_strength = gpu.normal_strength;
        emission_strength = 1.0f;
    }
    
    void applyTo(GpuMaterial& gpu) const {
        gpu.albedo = make_float3(albedo.x, albedo.y, albedo.z);
        gpu.opacity = opacity;
        gpu.roughness = roughness;
        gpu.metallic = metallic;
        gpu.specular = specular;
        gpu.clearcoat = clearcoat;
        gpu.transmission = transmission;
        gpu.emission = make_float3(emission.x, emission.y, emission.z);
        gpu.ior = ior;
        gpu.subsurface_color = make_float3(subsurface_color.x, subsurface_color.y, subsurface_color.z);
        gpu.subsurface = subsurface;
        gpu.anisotropic = anisotropic;
        gpu.sheen = sheen;
        gpu.sheen_tint = sheen_tint;
        gpu.normal_strength = normal_strength;
    }

    // ----- Generic channel access -----
    float channelValue(int ch) const {
        switch (ch) {
        case CURVE_MAT_ALBEDO_R: return albedo.x;
        case CURVE_MAT_ALBEDO_G: return albedo.y;
        case CURVE_MAT_ALBEDO_B: return albedo.z;
        case CURVE_MAT_OPACITY: return opacity;
        case CURVE_MAT_ROUGHNESS: return roughness;
        case CURVE_MAT_METALLIC: return metallic;
        case CURVE_MAT_CLEARCOAT: return clearcoat;
        case CURVE_MAT_TRANSMISSION: return transmission;
        case CURVE_MAT_IOR: return ior;
        case CURVE_MAT_EMISSION_R: return emission.x;
        case CURVE_MAT_EMISSION_G: return emission.y;
        case CURVE_MAT_EMISSION_B: return emission.z;
        case CURVE_MAT_NORMAL_STRENGTH: return normal_strength;
        case CURVE_MAT_EMISSION_STRENGTH: return emission_strength;
        }
        return 0.0f;
    }
    void setChannelValue(int ch, float v) {
        switch (ch) {
        case CURVE_MAT_ALBEDO_R: albedo.x = v; break;
        case CURVE_MAT_ALBEDO_G: albedo.y = v; break;
        case CURVE_MAT_ALBEDO_B: albedo.z = v; break;
        case CURVE_MAT_OPACITY: opacity = v; break;
        case CURVE_MAT_ROUGHNESS: roughness = v; break;
        case CURVE_MAT_METALLIC: metallic = v; break;
        case CURVE_MAT_CLEARCOAT: clearcoat = v; break;
        case CURVE_MAT_TRANSMISSION: transmission = v; break;
        case CURVE_MAT_IOR: ior = v; break;
        case CURVE_MAT_EMISSION_R: emission.x = v; break;
        case CURVE_MAT_EMISSION_G: emission.y = v; break;
        case CURVE_MAT_EMISSION_B: emission.z = v; break;
        case CURVE_MAT_NORMAL_STRENGTH: normal_strength = v; break;
        case CURVE_MAT_EMISSION_STRENGTH: emission_strength = v; break;
        }
    }
    bool channelKeyed(int ch) const {
        switch (ch) {
        case CURVE_MAT_ALBEDO_R: return has_alb_r;
        case CURVE_MAT_ALBEDO_G: return has_alb_g;
        case CURVE_MAT_ALBEDO_B: return has_alb_b;
        case CURVE_MAT_OPACITY: return has_opac;
        case CURVE_MAT_ROUGHNESS: return has_rough;
        case CURVE_MAT_METALLIC: return has_metal;
        case CURVE_MAT_CLEARCOAT: return has_clear;
        case CURVE_MAT_TRANSMISSION: return has_transm;
        case CURVE_MAT_IOR: return has_ior_val;
        case CURVE_MAT_EMISSION_R: return has_emis_r;
        case CURVE_MAT_EMISSION_G: return has_emis_g;
        case CURVE_MAT_EMISSION_B: return has_emis_b;
        case CURVE_MAT_NORMAL_STRENGTH: return has_norm_str;
        case CURVE_MAT_EMISSION_STRENGTH: return has_emis_str;
        }
        return false;
    }
    void setChannelKeyed(int ch, bool on) {
        switch (ch) {
        case CURVE_MAT_ALBEDO_R: has_alb_r = on; break;
        case CURVE_MAT_ALBEDO_G: has_alb_g = on; break;
        case CURVE_MAT_ALBEDO_B: has_alb_b = on; break;
        case CURVE_MAT_OPACITY: has_opac = on; break;
        case CURVE_MAT_ROUGHNESS: has_rough = on; break;
        case CURVE_MAT_METALLIC: has_metal = on; break;
        case CURVE_MAT_CLEARCOAT: has_clear = on; break;
        case CURVE_MAT_TRANSMISSION: has_transm = on; break;
        case CURVE_MAT_IOR: has_ior_val = on; break;
        case CURVE_MAT_EMISSION_R: has_emis_r = on; break;
        case CURVE_MAT_EMISSION_G: has_emis_g = on; break;
        case CURVE_MAT_EMISSION_B: has_emis_b = on; break;
        case CURVE_MAT_NORMAL_STRENGTH: has_norm_str = on; break;
        case CURVE_MAT_EMISSION_STRENGTH: has_emis_str = on; break;
        }
    }
    void refreshCompoundFlags() {
        has_albedo = has_alb_r || has_alb_g || has_alb_b;
        has_opacity = has_opac;
        has_roughness = has_rough;
        has_metallic = has_metal;
        has_clearcoat = has_clear;
        has_transmission = has_transm;
        has_ior = has_ior_val;
        has_emission = has_emis_r || has_emis_g || has_emis_b || has_emis_str;
        has_normal = has_norm_str;
    }
    void clearAllChannels() {
        has_albedo = has_opacity = has_roughness = has_metallic = has_clearcoat = has_transmission = has_ior = has_emission = has_normal = false;
        has_alb_r = has_alb_g = has_alb_b = false;
        has_opac = false;
        has_rough = false;
        has_metal = false;
        has_clear = false;
        has_transm = false;
        has_ior_val = false;
        has_emis_r = has_emis_g = has_emis_b = false;
        has_norm_str = false;
        has_emis_str = false;
    }
    static const char* channelName(int ch) {
        static const char* names[CURVE_MAT_CHANNEL_COUNT] = {
            "Albedo R", "Albedo G", "Albedo B",
            "Opacity",
            "Roughness",
            "Metallic",
            "Clearcoat",
            "Transmission",
            "IOR",
            "Emission R", "Emission G", "Emission B",
            "Normal Strength",
            "Emission Strength"
        };
        return (ch >= 0 && ch < CURVE_MAT_CHANNEL_COUNT) ? names[ch] : "?";
    }
    
    static MaterialKeyframe lerp(const MaterialKeyframe& a, const MaterialKeyframe& b, float t) {
        MaterialKeyframe result;
        result.material_id = (t < 0.5f) ? a.material_id : b.material_id;
        
        // Albedo
        if (a.has_alb_r && b.has_alb_r) result.albedo.x = a.albedo.x + (b.albedo.x - a.albedo.x) * t;
        else if (a.has_alb_r) result.albedo.x = a.albedo.x;
        else if (b.has_alb_r) result.albedo.x = b.albedo.x;
        result.has_alb_r = a.has_alb_r || b.has_alb_r;

        if (a.has_alb_g && b.has_alb_g) result.albedo.y = a.albedo.y + (b.albedo.y - a.albedo.y) * t;
        else if (a.has_alb_g) result.albedo.y = a.albedo.y;
        else if (b.has_alb_g) result.albedo.y = b.albedo.y;
        result.has_alb_g = a.has_alb_g || b.has_alb_g;

        if (a.has_alb_b && b.has_alb_b) result.albedo.z = a.albedo.z + (b.albedo.z - a.albedo.z) * t;
        else if (a.has_alb_b) result.albedo.z = a.albedo.z;
        else if (b.has_alb_b) result.albedo.z = b.albedo.z;
        result.has_alb_b = a.has_alb_b || b.has_alb_b;

        // Opacity
        if (a.has_opac && b.has_opac) result.opacity = a.opacity + (b.opacity - a.opacity) * t;
        else if (a.has_opac) result.opacity = a.opacity;
        else if (b.has_opac) result.opacity = b.opacity;
        result.has_opac = a.has_opac || b.has_opac;

        // Roughness
        if (a.has_rough && b.has_rough) result.roughness = a.roughness + (b.roughness - a.roughness) * t;
        else if (a.has_rough) result.roughness = a.roughness;
        else if (b.has_rough) result.roughness = b.roughness;
        result.has_rough = a.has_rough || b.has_rough;

        // Metallic
        if (a.has_metal && b.has_metal) result.metallic = a.metallic + (b.metallic - a.metallic) * t;
        else if (a.has_metal) result.metallic = a.metallic;
        else if (b.has_metal) result.metallic = b.metallic;
        result.has_metal = a.has_metal || b.has_metal;

        // Emission
        if (a.has_emis_r && b.has_emis_r) result.emission.x = a.emission.x + (b.emission.x - a.emission.x) * t;
        else if (a.has_emis_r) result.emission.x = a.emission.x;
        else if (b.has_emis_r) result.emission.x = b.emission.x;
        result.has_emis_r = a.has_emis_r || b.has_emis_r;

        if (a.has_emis_g && b.has_emis_g) result.emission.y = a.emission.y + (b.emission.y - a.emission.y) * t;
        else if (a.has_emis_g) result.emission.y = a.emission.y;
        else if (b.has_emis_g) result.emission.y = b.emission.y;
        result.has_emis_g = a.has_emis_g || b.has_emis_g;

        if (a.has_emis_b && b.has_emis_b) result.emission.z = a.emission.z + (b.emission.z - a.emission.z) * t;
        else if (a.has_emis_b) result.emission.z = a.emission.z;
        else if (b.has_emis_b) result.emission.z = b.emission.z;
        result.has_emis_b = a.has_emis_b || b.has_emis_b;

        // Emission Strength
        if (a.has_emis_str && b.has_emis_str) result.emission_strength = a.emission_strength + (b.emission_strength - a.emission_strength) * t;
        else if (a.has_emis_str) result.emission_strength = a.emission_strength;
        else if (b.has_emis_str) result.emission_strength = b.emission_strength;
        result.has_emis_str = a.has_emis_str || b.has_emis_str;

        // Transmission
        if (a.has_transm && b.has_transm) result.transmission = a.transmission + (b.transmission - a.transmission) * t;
        else if (a.has_transm) result.transmission = a.transmission;
        else if (b.has_transm) result.transmission = b.transmission;
        result.has_transm = a.has_transm || b.has_transm;

        // IOR
        if (a.has_ior_val && b.has_ior_val) result.ior = a.ior + (b.ior - a.ior) * t;
        else if (a.has_ior_val) result.ior = a.ior;
        else if (b.has_ior_val) result.ior = b.ior;
        result.has_ior_val = a.has_ior_val || b.has_ior_val;

        // Clearcoat
        if (a.has_clear && b.has_clear) {
             result.clearcoat = a.clearcoat + (b.clearcoat - a.clearcoat) * t;
             result.clearcoat_roughness = a.clearcoat_roughness + (b.clearcoat_roughness - a.clearcoat_roughness) * t;
        } else if (a.has_clear) {
             result.clearcoat = a.clearcoat; result.clearcoat_roughness = a.clearcoat_roughness;
        } else if (b.has_clear) {
             result.clearcoat = b.clearcoat; result.clearcoat_roughness = b.clearcoat_roughness;
        }
        result.has_clear = a.has_clear || b.has_clear;

        // Subsurface
        result.has_subsurface = a.has_subsurface || b.has_subsurface;
        if (a.has_subsurface && b.has_subsurface) {
            result.subsurface = a.subsurface + (b.subsurface - a.subsurface) * t;
            result.subsurface_color = a.subsurface_color + (b.subsurface_color - a.subsurface_color) * t;
        } else if (a.has_subsurface) {
            result.subsurface = a.subsurface; result.subsurface_color = a.subsurface_color;
        } else if (b.has_subsurface) {
            result.subsurface = b.subsurface; result.subsurface_color = b.subsurface_color;
        }
        
        // Specular
        result.has_specular = a.has_specular || b.has_specular;
        if (a.has_specular && b.has_specular) {
            result.specular = a.specular + (b.specular - a.specular) * t;
            result.specular_tint = a.specular_tint + (b.specular_tint - a.specular_tint) * t;
        } else if (a.has_specular) {
            result.specular = a.specular; result.specular_tint = a.specular_tint;
        } else if (b.has_specular) {
            result.specular = b.specular; result.specular_tint = b.specular_tint;
        }
        
        // Anisotropic
        result.has_anisotropic = a.has_anisotropic || b.has_anisotropic;
        if (a.has_anisotropic && b.has_anisotropic) result.anisotropic = a.anisotropic + (b.anisotropic - a.anisotropic) * t;
        else if (a.has_anisotropic) result.anisotropic = a.anisotropic;
        else if (b.has_anisotropic) result.anisotropic = b.anisotropic;
        
        // Sheen
        result.has_sheen = a.has_sheen || b.has_sheen;
        if (a.has_sheen && b.has_sheen) {
            result.sheen = a.sheen + (b.sheen - a.sheen) * t;
            result.sheen_tint = a.sheen_tint + (b.sheen_tint - a.sheen_tint) * t;
        } else if (a.has_sheen) {
            result.sheen = a.sheen; result.sheen_tint = a.sheen_tint;
        } else if (b.has_sheen) {
            result.sheen = b.sheen; result.sheen_tint = b.sheen_tint;
        }
        
        // Normal Strength
        if (a.has_norm_str && b.has_norm_str) result.normal_strength = a.normal_strength + (b.normal_strength - a.normal_strength) * t;
        else if (a.has_norm_str) result.normal_strength = a.normal_strength;
        else if (b.has_norm_str) result.normal_strength = b.normal_strength;
        result.has_norm_str = a.has_norm_str || b.has_norm_str;

        result.refreshCompoundFlags();
        return result;
    }
};

// ============================================================================
// LIGHT KEYFRAME - Position, color, intensity, direction
// ============================================================================
struct LightKeyframe {
    // Per-property flags - which properties are keyed
    bool has_position = true;
    bool has_color = true;
    bool has_intensity = true;
    bool has_direction = true;
    
    // Detailed per-axis flags
    bool has_pos_x = true; bool has_pos_y = true; bool has_pos_z = true;
    bool has_col_r = true; bool has_col_g = true; bool has_col_b = true;
    bool has_int = true;
    bool has_dir_x = true; bool has_dir_y = true; bool has_dir_z = true;

    // Property values
    Vec3 position = Vec3(0, 0, 0);
    Vec3 color = Vec3(1, 1, 1);
    float intensity = 1.0f;
    Vec3 direction = Vec3(0, -1, 0);  // For directional/spot lights
    
    // Per-channel curve metadata (graph editor)
    ChannelKeyMeta curve[CURVE_LIGHT_CHANNEL_COUNT];

    LightKeyframe() = default;
    
    // ----- Generic channel access -----
    float channelValue(int ch) const {
        switch (ch) {
        case CURVE_LIGHT_POS_X: return position.x;
        case CURVE_LIGHT_POS_Y: return position.y;
        case CURVE_LIGHT_POS_Z: return position.z;
        case CURVE_LIGHT_COLOR_R: return color.x;
        case CURVE_LIGHT_COLOR_G: return color.y;
        case CURVE_LIGHT_COLOR_B: return color.z;
        case CURVE_LIGHT_INTENSITY: return intensity;
        case CURVE_LIGHT_DIR_X: return direction.x;
        case CURVE_LIGHT_DIR_Y: return direction.y;
        case CURVE_LIGHT_DIR_Z: return direction.z;
        }
        return 0.0f;
    }
    void setChannelValue(int ch, float v) {
        switch (ch) {
        case CURVE_LIGHT_POS_X: position.x = v; break;
        case CURVE_LIGHT_POS_Y: position.y = v; break;
        case CURVE_LIGHT_POS_Z: position.z = v; break;
        case CURVE_LIGHT_COLOR_R: color.x = v; break;
        case CURVE_LIGHT_COLOR_G: color.y = v; break;
        case CURVE_LIGHT_COLOR_B: color.z = v; break;
        case CURVE_LIGHT_INTENSITY: intensity = v; break;
        case CURVE_LIGHT_DIR_X: direction.x = v; break;
        case CURVE_LIGHT_DIR_Y: direction.y = v; break;
        case CURVE_LIGHT_DIR_Z: direction.z = v; break;
        }
    }
    bool channelKeyed(int ch) const {
        switch (ch) {
        case CURVE_LIGHT_POS_X: return has_pos_x;
        case CURVE_LIGHT_POS_Y: return has_pos_y;
        case CURVE_LIGHT_POS_Z: return has_pos_z;
        case CURVE_LIGHT_COLOR_R: return has_col_r;
        case CURVE_LIGHT_COLOR_G: return has_col_g;
        case CURVE_LIGHT_COLOR_B: return has_col_b;
        case CURVE_LIGHT_INTENSITY: return has_int;
        case CURVE_LIGHT_DIR_X: return has_dir_x;
        case CURVE_LIGHT_DIR_Y: return has_dir_y;
        case CURVE_LIGHT_DIR_Z: return has_dir_z;
        }
        return false;
    }
    void setChannelKeyed(int ch, bool on) {
        switch (ch) {
        case CURVE_LIGHT_POS_X: has_pos_x = on; break;
        case CURVE_LIGHT_POS_Y: has_pos_y = on; break;
        case CURVE_LIGHT_POS_Z: has_pos_z = on; break;
        case CURVE_LIGHT_COLOR_R: has_col_r = on; break;
        case CURVE_LIGHT_COLOR_G: has_col_g = on; break;
        case CURVE_LIGHT_COLOR_B: has_col_b = on; break;
        case CURVE_LIGHT_INTENSITY: has_int = on; break;
        case CURVE_LIGHT_DIR_X: has_dir_x = on; break;
        case CURVE_LIGHT_DIR_Y: has_dir_y = on; break;
        case CURVE_LIGHT_DIR_Z: has_dir_z = on; break;
        }
    }
    void refreshCompoundFlags() {
        has_position = has_pos_x || has_pos_y || has_pos_z;
        has_color = has_col_r || has_col_g || has_col_b;
        has_intensity = has_int;
        has_direction = has_dir_x || has_dir_y || has_dir_z;
    }
    void clearAllChannels() {
        has_position = has_color = has_intensity = has_direction = false;
        has_pos_x = has_pos_y = has_pos_z = false;
        has_col_r = has_col_g = has_col_b = false;
        has_int = false;
        has_dir_x = has_dir_y = has_dir_z = false;
    }
    static const char* channelName(int ch) {
        static const char* names[CURVE_LIGHT_CHANNEL_COUNT] = {
            "Pos X", "Pos Y", "Pos Z",
            "Color R", "Color G", "Color B",
            "Intensity",
            "Dir X", "Dir Y", "Dir Z"
        };
        return (ch >= 0 && ch < CURVE_LIGHT_CHANNEL_COUNT) ? names[ch] : "?";
    }

    // Lerp only interpolates properties that are keyed in BOTH keyframes
    static LightKeyframe lerp(const LightKeyframe& a, const LightKeyframe& b, float t) {
        LightKeyframe result;
        
        // Position
        if (a.has_pos_x && b.has_pos_x) result.position.x = a.position.x + (b.position.x - a.position.x) * t;
        else if (a.has_pos_x) result.position.x = a.position.x;
        else if (b.has_pos_x) result.position.x = b.position.x;
        result.has_pos_x = a.has_pos_x || b.has_pos_x;

        if (a.has_pos_y && b.has_pos_y) result.position.y = a.position.y + (b.position.y - a.position.y) * t;
        else if (a.has_pos_y) result.position.y = a.position.y;
        else if (b.has_pos_y) result.position.y = b.position.y;
        result.has_pos_y = a.has_pos_y || b.has_pos_y;

        if (a.has_pos_z && b.has_pos_z) result.position.z = a.position.z + (b.position.z - a.position.z) * t;
        else if (a.has_pos_z) result.position.z = a.position.z;
        else if (b.has_pos_z) result.position.z = b.position.z;
        result.has_pos_z = a.has_pos_z || b.has_pos_z;
        
        // Color
        if (a.has_col_r && b.has_col_r) result.color.x = a.color.x + (b.color.x - a.color.x) * t;
        else if (a.has_col_r) result.color.x = a.color.x;
        else if (b.has_col_r) result.color.x = b.color.x;
        result.has_col_r = a.has_col_r || b.has_col_r;

        if (a.has_col_g && b.has_col_g) result.color.y = a.color.y + (b.color.y - a.color.y) * t;
        else if (a.has_col_g) result.color.y = a.color.y;
        else if (b.has_col_g) result.color.y = b.color.y;
        result.has_col_g = a.has_col_g || b.has_col_g;

        if (a.has_col_b && b.has_col_b) result.color.z = a.color.z + (b.color.z - a.color.z) * t;
        else if (a.has_col_b) result.color.z = a.color.z;
        else if (b.has_col_b) result.color.z = b.color.z;
        result.has_col_b = a.has_col_b || b.has_col_b;
        
        // Intensity
        if (a.has_int && b.has_int) result.intensity = a.intensity + (b.intensity - a.intensity) * t;
        else if (a.has_int) result.intensity = a.intensity;
        else if (b.has_int) result.intensity = b.intensity;
        result.has_int = a.has_int || b.has_int;
        
        // Direction
        if (a.has_dir_x && b.has_dir_x) result.direction.x = a.direction.x + (b.direction.x - a.direction.x) * t;
        else if (a.has_dir_x) result.direction.x = a.direction.x;
        else if (b.has_dir_x) result.direction.x = b.direction.x;
        result.has_dir_x = a.has_dir_x || b.has_dir_x;

        if (a.has_dir_y && b.has_dir_y) result.direction.y = a.direction.y + (b.direction.y - a.direction.y) * t;
        else if (a.has_dir_y) result.direction.y = a.direction.y;
        else if (b.has_dir_y) result.direction.y = b.direction.y;
        result.has_dir_y = a.has_dir_y || b.has_dir_y;

        if (a.has_dir_z && b.has_dir_z) result.direction.z = a.direction.z + (b.direction.z - a.direction.z) * t;
        else if (a.has_dir_z) result.direction.z = a.direction.z;
        else if (b.has_dir_z) result.direction.z = b.direction.z;
        result.has_dir_z = a.has_dir_z || b.has_dir_z;
        
        result.refreshCompoundFlags();
        return result;
    }
};

// ============================================================================
// CAMERA KEYFRAME - Position, target, FOV, DOF
// ============================================================================
struct CameraKeyframe {
    // Per-property flags - which properties are keyed
    bool has_position = true;
    bool has_target = true;
    bool has_fov = true;
    bool has_focus = true;
    bool has_aperture = true;
    
    // Detailed per-axis flags
    bool has_pos_x = true; bool has_pos_y = true; bool has_pos_z = true;
    bool has_tgt_x = true; bool has_tgt_y = true; bool has_tgt_z = true;
    bool has_fv = true;
    bool has_foc_dist = true;
    bool has_lens_rad = true;

    // Property values
    Vec3 position = Vec3(0, 0, 0);
    Vec3 target = Vec3(0, 0, -1);
    float fov = 40.0f;
    float focus_distance = 10.0f;
    float lens_radius = 0.0f;
    
    // Per-channel curve metadata (graph editor)
    ChannelKeyMeta curve[CURVE_CAM_CHANNEL_COUNT];

    CameraKeyframe() = default;
    
    // ----- Generic channel access -----
    float channelValue(int ch) const {
        switch (ch) {
        case CURVE_CAM_POS_X: return position.x;
        case CURVE_CAM_POS_Y: return position.y;
        case CURVE_CAM_POS_Z: return position.z;
        case CURVE_CAM_TGT_X: return target.x;
        case CURVE_CAM_TGT_Y: return target.y;
        case CURVE_CAM_TGT_Z: return target.z;
        case CURVE_CAM_FOV: return fov;
        case CURVE_CAM_FOCUS_DIST: return focus_distance;
        case CURVE_CAM_LENS_RAD: return lens_radius;
        }
        return 0.0f;
    }
    void setChannelValue(int ch, float v) {
        switch (ch) {
        case CURVE_CAM_POS_X: position.x = v; break;
        case CURVE_CAM_POS_Y: position.y = v; break;
        case CURVE_CAM_POS_Z: position.z = v; break;
        case CURVE_CAM_TGT_X: target.x = v; break;
        case CURVE_CAM_TGT_Y: target.y = v; break;
        case CURVE_CAM_TGT_Z: target.z = v; break;
        case CURVE_CAM_FOV: fov = v; break;
        case CURVE_CAM_FOCUS_DIST: focus_distance = v; break;
        case CURVE_CAM_LENS_RAD: lens_radius = v; break;
        }
    }
    bool channelKeyed(int ch) const {
        switch (ch) {
        case CURVE_CAM_POS_X: return has_pos_x;
        case CURVE_CAM_POS_Y: return has_pos_y;
        case CURVE_CAM_POS_Z: return has_pos_z;
        case CURVE_CAM_TGT_X: return has_tgt_x;
        case CURVE_CAM_TGT_Y: return has_tgt_y;
        case CURVE_CAM_TGT_Z: return has_tgt_z;
        case CURVE_CAM_FOV: return has_fv;
        case CURVE_CAM_FOCUS_DIST: return has_foc_dist;
        case CURVE_CAM_LENS_RAD: return has_lens_rad;
        }
        return false;
    }
    void setChannelKeyed(int ch, bool on) {
        switch (ch) {
        case CURVE_CAM_POS_X: has_pos_x = on; break;
        case CURVE_CAM_POS_Y: has_pos_y = on; break;
        case CURVE_CAM_POS_Z: has_pos_z = on; break;
        case CURVE_CAM_TGT_X: has_tgt_x = on; break;
        case CURVE_CAM_TGT_Y: has_tgt_y = on; break;
        case CURVE_CAM_TGT_Z: has_tgt_z = on; break;
        case CURVE_CAM_FOV: has_fv = on; break;
        case CURVE_CAM_FOCUS_DIST: has_foc_dist = on; break;
        case CURVE_CAM_LENS_RAD: has_lens_rad = on; break;
        }
    }
    void refreshCompoundFlags() {
        has_position = has_pos_x || has_pos_y || has_pos_z;
        has_target = has_tgt_x || has_tgt_y || has_tgt_z;
        has_fov = has_fv;
        has_focus = has_foc_dist;
        has_aperture = has_lens_rad;
    }
    void clearAllChannels() {
        has_position = has_target = has_fov = has_focus = has_aperture = false;
        has_pos_x = has_pos_y = has_pos_z = false;
        has_tgt_x = has_tgt_y = has_tgt_z = false;
        has_fv = false;
        has_foc_dist = false;
        has_lens_rad = false;
    }
    static const char* channelName(int ch) {
        static const char* names[CURVE_CAM_CHANNEL_COUNT] = {
            "Pos X", "Pos Y", "Pos Z",
            "Target X", "Target Y", "Target Z",
            "FOV",
            "Focus Dist",
            "Lens Rad"
        };
        return (ch >= 0 && ch < CURVE_CAM_CHANNEL_COUNT) ? names[ch] : "?";
    }

    // Lerp only interpolates properties that are keyed in BOTH keyframes
    static CameraKeyframe lerp(const CameraKeyframe& a, const CameraKeyframe& b, float t) {
        CameraKeyframe result;
        
        // Position
        if (a.has_pos_x && b.has_pos_x) result.position.x = a.position.x + (b.position.x - a.position.x) * t;
        else if (a.has_pos_x) result.position.x = a.position.x;
        else if (b.has_pos_x) result.position.x = b.position.x;
        result.has_pos_x = a.has_pos_x || b.has_pos_x;

        if (a.has_pos_y && b.has_pos_y) result.position.y = a.position.y + (b.position.y - a.position.y) * t;
        else if (a.has_pos_y) result.position.y = a.position.y;
        else if (b.has_pos_y) result.position.y = b.position.y;
        result.has_pos_y = a.has_pos_y || b.has_pos_y;

        if (a.has_pos_z && b.has_pos_z) result.position.z = a.position.z + (b.position.z - a.position.z) * t;
        else if (a.has_pos_z) result.position.z = a.position.z;
        else if (b.has_pos_z) result.position.z = b.position.z;
        result.has_pos_z = a.has_pos_z || b.has_pos_z;
        
        // Target
        if (a.has_tgt_x && b.has_tgt_x) result.target.x = a.target.x + (b.target.x - a.target.x) * t;
        else if (a.has_tgt_x) result.target.x = a.target.x;
        else if (b.has_tgt_x) result.target.x = b.target.x;
        result.has_tgt_x = a.has_tgt_x || b.has_tgt_x;

        if (a.has_tgt_y && b.has_tgt_y) result.target.y = a.target.y + (b.target.y - a.target.y) * t;
        else if (a.has_tgt_y) result.target.y = a.target.y;
        else if (b.has_tgt_y) result.target.y = b.target.y;
        result.has_tgt_y = a.has_tgt_y || b.has_tgt_y;

        if (a.has_tgt_z && b.has_tgt_z) result.target.z = a.target.z + (b.target.z - a.target.z) * t;
        else if (a.has_tgt_z) result.target.z = a.target.z;
        else if (b.has_tgt_z) result.target.z = b.target.z;
        result.has_tgt_z = a.has_tgt_z || b.has_tgt_z;
        
        // FOV
        if (a.has_fv && b.has_fv) result.fov = a.fov + (b.fov - a.fov) * t;
        else if (a.has_fv) result.fov = a.fov;
        else if (b.has_fv) result.fov = b.fov;
        result.has_fv = a.has_fv || b.has_fv;
        
        // Focus Distance
        if (a.has_foc_dist && b.has_foc_dist) result.focus_distance = a.focus_distance + (b.focus_distance - a.focus_distance) * t;
        else if (a.has_foc_dist) result.focus_distance = a.focus_distance;
        else if (b.has_foc_dist) result.focus_distance = b.focus_distance;
        result.has_foc_dist = a.has_foc_dist || b.has_foc_dist;
        
        // Lens Radius
        if (a.has_lens_rad && b.has_lens_rad) result.lens_radius = a.lens_radius + (b.lens_radius - a.lens_radius) * t;
        else if (a.has_lens_rad) result.lens_radius = a.lens_radius;
        else if (b.has_lens_rad) result.lens_radius = b.lens_radius;
        result.has_lens_rad = a.has_lens_rad || b.has_lens_rad;
        
        result.refreshCompoundFlags();
        return result;
    }
};

// ============================================================================
// WORLD KEYFRAME - Background, Nishita sky, volumetric settings
// ============================================================================
struct WorldKeyframe {
    // GRANULAR PER-PROPERTY FLAGS - Every property can be keyed independently
    
    // Background properties
    bool has_background_color = false;
    bool has_background_strength = false;
    bool has_hdri_rotation = false;
    
    // Sun properties
    bool has_sun_elevation = false;
    bool has_sun_azimuth = false;
    bool has_sun_intensity = false;
    bool has_sun_size = false;
    
    // Atmosphere properties
    bool has_atmosphere_intensity = false;
    bool has_air_density = false;
    bool has_dust_density = false;
    bool has_ozone_density = false;
    bool has_humidity = false;
    bool has_temperature = false;
    bool has_ozone_absorption_scale = false;
    bool has_altitude = false;
    bool has_mie_anisotropy = false;
    
    // Cloud properties (Layer 1)
    bool has_cloud_density = false;
    bool has_cloud_coverage = false;
    bool has_cloud_scale = false;
    bool has_cloud_offset = false;
    bool has_cloud_quality = false;
    bool has_cloud_detail = false;

    // Cloud properties (Layer 2)
    bool has_cloud_layer2 = false;
    bool has_cloud_layer2_params = false; // Coverage, Density, Scale
    bool has_cloud_layer2_heights = false;

    // Cloud Lighting
    bool has_cloud_lighting = false; // Steps, Shadow, Ambient, Silver, Absorption
    
    // Atmospheric Effects
    bool has_fog = false;
    bool has_fog_params = false; // Density, Height, Falloff, Distance, Color, Scatter
    bool has_godrays = false;
    bool has_godrays_params = false; // Intensity, Density, Samples

    // Advanced Environment
    bool has_multi_scatter = false;
    bool has_aerial_perspective = false;
    bool has_aerial_params = false; // Min/Max distance
    bool has_overlay = false;
    bool has_overlay_params = false; // Intensity, Rotation, Mode

    // Weather
    bool has_weather_params = false;
    
    // Property values
    Vec3 background_color = Vec3(0.5, 0.7, 1.0);
    float background_strength = 1.0f;
    float hdri_rotation = 0.0f;
    float hdri_intensity = 1.0f;
    
    float sun_elevation = 15.0f;
    float sun_azimuth = 0.0f;
    float sun_intensity = 1.0f;
    float sun_size = 0.545f;
    
    float air_density = 1.0f;
    float atmosphere_intensity = 10.0f;
    float dust_density = 1.0f;
    float ozone_density = 1.0f;
    float humidity = 0.1f;
    float temperature = 15.0f;
    float ozone_absorption_scale = 1.0f;
    float altitude = 0.0f;
    float mie_anisotropy = 0.76f;
    
    // Cloud L1
    float cloud_density = 0.5f;
    float cloud_coverage = 0.5f;
    float cloud_scale = 1.0f;
    float cloud_offset_x = 0.0f;
    float cloud_offset_z = 0.0f;
    float cloud_quality = 1.0f;
    float cloud_detail = 1.0f;
    int cloud_base_steps = 8;
    float cloud_height_min = 500.0f;
    float cloud_height_max = 2000.0f;

    // Cloud L2
    int cloud_layer2_enabled = 0;
    float cloud2_coverage = 0.3f;
    float cloud2_density = 0.3f;
    float cloud2_scale = 8.0f;
    float cloud2_height_min = 6000.0f;
    float cloud2_height_max = 7000.0f;

    // Cloud Lighting
    int cloud_light_steps = 0;
    float cloud_shadow_strength = 1.0f;
    float cloud_ambient_strength = 1.0f;
    float cloud_silver_intensity = 1.0f;
    float cloud_absorption = 1.0f;
    float cloud_anisotropy = 0.85f;
    float cloud_anisotropy_back = -0.3f;
    float cloud_lobe_mix = 0.5f;
    float cloud_emissive_intensity = 0.0f;
    Vec3 cloud_emissive_color = Vec3(1.0, 1.0, 1.0);

    // Fog
    int fog_enabled = 0;
    float fog_density = 0.01f;
    float fog_height = 500.0f;
    float fog_falloff = 0.003f;
    float fog_distance = 10000.0f;
    Vec3 fog_color = Vec3(0.7, 0.8, 0.9);
    float fog_sun_scatter = 0.5f;

    // God Rays
    int godrays_enabled = 0;
    float godrays_intensity = 0.5f;
    float godrays_density = 0.1f;
    int godrays_samples = 16;
   

    // Advanced
    int multi_scatter_enabled = 1;
    float multi_scatter_factor = 0.3f;
    int aerial_perspective = 1;
    float aerial_density = 1.0f;
    float aerial_min_distance = 10.0f;
    float aerial_max_distance = 5000.0f;
    int env_overlay_enabled = 0;
    float env_overlay_intensity = 1.0f;
    float env_overlay_rotation = 0.0f;
    int env_overlay_blend_mode = 0;

    int weather_enabled = 0;
    int weather_type = 0;
    float weather_intensity = 0.0f;
    float weather_density = 0.0f;
    Vec3 weather_wind_direction = Vec3(1.0f, 0.0f, 0.0f);
    float weather_wind_speed = 0.0f;
    float weather_precipitation_scale = 1.0f;
    float weather_visibility = 1.0f;
    float weather_surface_wetness = 0.0f;
    float weather_surface_accumulation = 0.0f;
    float weather_surface_settling = 0.0f;
    float weather_surface_height = 0.0f;
    int weather_visual_mode = 0;
    int weather_surface_response_enabled = 1;
    
    WorldKeyframe() = default;
    
    static WorldKeyframe lerp(const WorldKeyframe& a, const WorldKeyframe& b, float t) {
        WorldKeyframe result;
        
        // ===== BACKGROUND PROPERTIES =====
        // Background Color
        result.has_background_color = a.has_background_color || b.has_background_color;
        if (a.has_background_color && b.has_background_color) {
            result.background_color = a.background_color + (b.background_color - a.background_color) * t;
        } else if (a.has_background_color) {
            result.background_color = a.background_color;
        } else if (b.has_background_color) {
            result.background_color = b.background_color;
        }
        
        // Background Strength
        result.has_background_strength = a.has_background_strength || b.has_background_strength;
        if (a.has_background_strength && b.has_background_strength) {
            result.background_strength = a.background_strength + (b.background_strength - a.background_strength) * t;
        } else if (a.has_background_strength) {
            result.background_strength = a.background_strength;
        } else if (b.has_background_strength) {
            result.background_strength = b.background_strength;
        }
        
        // HDRI Rotation
        result.has_hdri_rotation = a.has_hdri_rotation || b.has_hdri_rotation;
        if (a.has_hdri_rotation && b.has_hdri_rotation) {
            result.hdri_rotation = a.hdri_rotation + (b.hdri_rotation - a.hdri_rotation) * t;
            result.hdri_intensity = a.hdri_intensity + (b.hdri_intensity - a.hdri_intensity) * t;
        } else if (a.has_hdri_rotation) {
            result.hdri_rotation = a.hdri_rotation; result.hdri_intensity = a.hdri_intensity;
        } else if (b.has_hdri_rotation) {
            result.hdri_rotation = b.hdri_rotation; result.hdri_intensity = b.hdri_intensity;
        }

        // ===== SUN PROPERTIES =====
        // Sun Elevation
        result.has_sun_elevation = a.has_sun_elevation || b.has_sun_elevation;
        if (a.has_sun_elevation && b.has_sun_elevation) {
            result.sun_elevation = a.sun_elevation + (b.sun_elevation - a.sun_elevation) * t;
        } else if (a.has_sun_elevation) {
            result.sun_elevation = a.sun_elevation;
        } else if (b.has_sun_elevation) {
            result.sun_elevation = b.sun_elevation;
        }
        
        // Sun Azimuth
        result.has_sun_azimuth = a.has_sun_azimuth || b.has_sun_azimuth;
        if (a.has_sun_azimuth && b.has_sun_azimuth) {
            result.sun_azimuth = a.sun_azimuth + (b.sun_azimuth - a.sun_azimuth) * t;
        } else if (a.has_sun_azimuth) {
            result.sun_azimuth = a.sun_azimuth;
        } else if (b.has_sun_azimuth) {
            result.sun_azimuth = b.sun_azimuth;
        }
        
        // Sun Intensity
        result.has_sun_intensity = a.has_sun_intensity || b.has_sun_intensity;
        if (a.has_sun_intensity && b.has_sun_intensity) {
            result.sun_intensity = a.sun_intensity + (b.sun_intensity - a.sun_intensity) * t;
        } else if (a.has_sun_intensity) {
            result.sun_intensity = a.sun_intensity;
        } else if (b.has_sun_intensity) {
            result.sun_intensity = b.sun_intensity;
        }
        
        // Sun Size
        result.has_sun_size = a.has_sun_size || b.has_sun_size;
        if (a.has_sun_size && b.has_sun_size) {
            result.sun_size = a.sun_size + (b.sun_size - a.sun_size) * t;
        } else if (a.has_sun_size) {
            result.sun_size = a.sun_size;
        } else if (b.has_sun_size) {
            result.sun_size = b.sun_size;
        }
        
        // ===== ATMOSPHERE PROPERTIES =====
        // Atmosphere Intensity
        result.has_atmosphere_intensity = a.has_atmosphere_intensity || b.has_atmosphere_intensity;
        if (a.has_atmosphere_intensity && b.has_atmosphere_intensity) {
            result.atmosphere_intensity = a.atmosphere_intensity + (b.atmosphere_intensity - a.atmosphere_intensity) * t;
        } else if (a.has_atmosphere_intensity) {
            result.atmosphere_intensity = a.atmosphere_intensity;
        } else if (b.has_atmosphere_intensity) {
            result.atmosphere_intensity = b.atmosphere_intensity;
        }
        
        // Air Density
        result.has_air_density = a.has_air_density || b.has_air_density;
        if (a.has_air_density && b.has_air_density) {
            result.air_density = a.air_density + (b.air_density - a.air_density) * t;
        } else if (a.has_air_density) {
            result.air_density = a.air_density;
        } else if (b.has_air_density) {
            result.air_density = b.air_density;
        }
        
        // Dust Density
        result.has_dust_density = a.has_dust_density || b.has_dust_density;
        if (a.has_dust_density && b.has_dust_density) {
            result.dust_density = a.dust_density + (b.dust_density - a.dust_density) * t;
        } else if (a.has_dust_density) {
            result.dust_density = a.dust_density;
        } else if (b.has_dust_density) {
            result.dust_density = b.dust_density;
        }

        // Humidity
        result.has_humidity = a.has_humidity || b.has_humidity;
        if (a.has_humidity && b.has_humidity) {
            result.humidity = a.humidity + (b.humidity - a.humidity) * t;
        } else if (a.has_humidity) {
            result.humidity = a.humidity;
        } else if (b.has_humidity) {
            result.humidity = b.humidity;
        }

        // Temperature
        result.has_temperature = a.has_temperature || b.has_temperature;
        if (a.has_temperature && b.has_temperature) {
            result.temperature = a.temperature + (b.temperature - a.temperature) * t;
        } else if (a.has_temperature) {
            result.temperature = a.temperature;
        } else if (b.has_temperature) {
            result.temperature = b.temperature;
        }
        
        // Ozone Density & Strength
        result.has_ozone_density = a.has_ozone_density || b.has_ozone_density;
        if (a.has_ozone_density && b.has_ozone_density) {
            result.ozone_density = a.ozone_density + (b.ozone_density - a.ozone_density) * t;
            result.ozone_absorption_scale = a.ozone_absorption_scale + (b.ozone_absorption_scale - a.ozone_absorption_scale) * t;
        } else if (a.has_ozone_density) {
            result.ozone_density = a.ozone_density; result.ozone_absorption_scale = a.ozone_absorption_scale;
        } else if (b.has_ozone_density) {
            result.ozone_density = b.ozone_density; result.ozone_absorption_scale = b.ozone_absorption_scale;
        }
        
        // Altitude
        result.has_altitude = a.has_altitude || b.has_altitude;
        if (a.has_altitude && b.has_altitude) {
            result.altitude = a.altitude + (b.altitude - a.altitude) * t;
        } else if (a.has_altitude) {
            result.altitude = a.altitude;
        } else if (b.has_altitude) {
            result.altitude = b.altitude;
        }
        
        // Mie Anisotropy
        result.has_mie_anisotropy = a.has_mie_anisotropy || b.has_mie_anisotropy;
        if (a.has_mie_anisotropy && b.has_mie_anisotropy) {
            result.mie_anisotropy = a.mie_anisotropy + (b.mie_anisotropy - a.mie_anisotropy) * t;
        } else if (a.has_mie_anisotropy) {
            result.mie_anisotropy = a.mie_anisotropy;
        } else if (b.has_mie_anisotropy) {
            result.mie_anisotropy = b.mie_anisotropy;
        }
        
        // ===== CLOUD PROPERTIES (L1) =====
        // Cloud Density
        result.has_cloud_density = a.has_cloud_density || b.has_cloud_density;
        if (a.has_cloud_density && b.has_cloud_density) {
            result.cloud_density = a.cloud_density + (b.cloud_density - a.cloud_density) * t;
        } else if (a.has_cloud_density) {
            result.cloud_density = a.cloud_density;
        } else if (b.has_cloud_density) {
            result.cloud_density = b.cloud_density;
        }
        
        // Cloud Coverage
        result.has_cloud_coverage = a.has_cloud_coverage || b.has_cloud_coverage;
        if (a.has_cloud_coverage && b.has_cloud_coverage) {
            result.cloud_coverage = a.cloud_coverage + (b.cloud_coverage - a.cloud_coverage) * t;
        } else if (a.has_cloud_coverage) {
            result.cloud_coverage = a.cloud_coverage;
        } else if (b.has_cloud_coverage) {
            result.cloud_coverage = b.cloud_coverage;
        }
        
        // Cloud Scale
        result.has_cloud_scale = a.has_cloud_scale || b.has_cloud_scale;
        if (a.has_cloud_scale && b.has_cloud_scale) {
            result.cloud_scale = a.cloud_scale + (b.cloud_scale - a.cloud_scale) * t;
        } else if (a.has_cloud_scale) {
            result.cloud_scale = a.cloud_scale;
        } else if (b.has_cloud_scale) {
            result.cloud_scale = b.cloud_scale;
        }
        
        // Cloud Offset (X and Z together)
        result.has_cloud_offset = a.has_cloud_offset || b.has_cloud_offset;
        if (a.has_cloud_offset && b.has_cloud_offset) {
            result.cloud_offset_x = a.cloud_offset_x + (b.cloud_offset_x - a.cloud_offset_x) * t;
            result.cloud_offset_z = a.cloud_offset_z + (b.cloud_offset_z - a.cloud_offset_z) * t;
        } else if (a.has_cloud_offset) {
            result.cloud_offset_x = a.cloud_offset_x;
            result.cloud_offset_z = a.cloud_offset_z;
        } else if (b.has_cloud_offset) {
            result.cloud_offset_x = b.cloud_offset_x;
            result.cloud_offset_z = b.cloud_offset_z;
        }
        
        // Quality & Detail
        result.has_cloud_quality = a.has_cloud_quality || b.has_cloud_quality;
        if (a.has_cloud_quality && b.has_cloud_quality) {
            result.cloud_quality = a.cloud_quality + (b.cloud_quality - a.cloud_quality) * t;
            result.cloud_detail = a.cloud_detail + (b.cloud_detail - a.cloud_detail) * t;
            result.cloud_base_steps = (t < 0.5f) ? a.cloud_base_steps : b.cloud_base_steps;
        } else if (a.has_cloud_quality) {
            result.cloud_quality = a.cloud_quality; result.cloud_detail = a.cloud_detail; result.cloud_base_steps = a.cloud_base_steps;
        } else if (b.has_cloud_quality) {
            result.cloud_quality = b.cloud_quality; result.cloud_detail = b.cloud_detail; result.cloud_base_steps = b.cloud_base_steps;
        }

        // ===== CLOUD LAYER 2 =====
        result.has_cloud_layer2 = a.has_cloud_layer2 || b.has_cloud_layer2;
        if (result.has_cloud_layer2) {
            result.cloud_layer2_enabled = (t < 0.5f) ? a.cloud_layer2_enabled : b.cloud_layer2_enabled;
        }
        result.has_cloud_layer2_params = a.has_cloud_layer2_params || b.has_cloud_layer2_params;
        if (a.has_cloud_layer2_params && b.has_cloud_layer2_params) {
            result.cloud2_coverage = a.cloud2_coverage + (b.cloud2_coverage - a.cloud2_coverage) * t;
            result.cloud2_density = a.cloud2_density + (b.cloud2_density - a.cloud2_density) * t;
            result.cloud2_scale = a.cloud2_scale + (b.cloud2_scale - a.cloud2_scale) * t;
        } else if (a.has_cloud_layer2_params) {
            result.cloud2_coverage = a.cloud2_coverage; result.cloud2_density = a.cloud2_density; result.cloud2_scale = a.cloud2_scale;
        } else if (b.has_cloud_layer2_params) {
            result.cloud2_coverage = b.cloud2_coverage; result.cloud2_density = b.cloud2_density; result.cloud2_scale = b.cloud2_scale;
        }

        // ===== CLOUD LIGHTING =====
        result.has_cloud_lighting = a.has_cloud_lighting || b.has_cloud_lighting;
        if (a.has_cloud_lighting && b.has_cloud_lighting) {
            result.cloud_light_steps = (t < 0.5f) ? a.cloud_light_steps : b.cloud_light_steps;
            result.cloud_shadow_strength = a.cloud_shadow_strength + (b.cloud_shadow_strength - a.cloud_shadow_strength) * t;
            result.cloud_ambient_strength = a.cloud_ambient_strength + (b.cloud_ambient_strength - a.cloud_ambient_strength) * t;
            result.cloud_silver_intensity = a.cloud_silver_intensity + (b.cloud_silver_intensity - a.cloud_silver_intensity) * t;
            result.cloud_absorption = a.cloud_absorption + (b.cloud_absorption - a.cloud_absorption) * t;
            result.cloud_anisotropy = a.cloud_anisotropy + (b.cloud_anisotropy - a.cloud_anisotropy) * t;
            result.cloud_anisotropy_back = a.cloud_anisotropy_back + (b.cloud_anisotropy_back - a.cloud_anisotropy_back) * t;
            result.cloud_lobe_mix = a.cloud_lobe_mix + (b.cloud_lobe_mix - a.cloud_lobe_mix) * t;
            result.cloud_emissive_intensity = a.cloud_emissive_intensity + (b.cloud_emissive_intensity - a.cloud_emissive_intensity) * t;
            result.cloud_emissive_color = a.cloud_emissive_color + (b.cloud_emissive_color - a.cloud_emissive_color) * t;
        } else if (a.has_cloud_lighting) {
            result.cloud_light_steps = a.cloud_light_steps; result.cloud_shadow_strength = a.cloud_shadow_strength;
            result.cloud_ambient_strength = a.cloud_ambient_strength; result.cloud_silver_intensity = a.cloud_silver_intensity;
            result.cloud_absorption = a.cloud_absorption; result.cloud_anisotropy = a.cloud_anisotropy;
            result.cloud_anisotropy_back = a.cloud_anisotropy_back; result.cloud_lobe_mix = a.cloud_lobe_mix;
            result.cloud_emissive_intensity = a.cloud_emissive_intensity; result.cloud_emissive_color = a.cloud_emissive_color;
        } else if (b.has_cloud_lighting) {
            result.cloud_light_steps = b.cloud_light_steps; result.cloud_shadow_strength = b.cloud_shadow_strength;
            result.cloud_ambient_strength = b.cloud_ambient_strength; result.cloud_silver_intensity = b.cloud_silver_intensity;
            result.cloud_absorption = b.cloud_absorption; result.cloud_anisotropy = b.cloud_anisotropy;
            result.cloud_anisotropy_back = b.cloud_anisotropy_back; result.cloud_lobe_mix = b.cloud_lobe_mix;
            result.cloud_emissive_intensity = b.cloud_emissive_intensity; result.cloud_emissive_color = b.cloud_emissive_color;
        }

        // ===== FOG =====
        result.has_fog = a.has_fog || b.has_fog;
        if (result.has_fog) {
            result.fog_enabled = (t < 0.5f) ? a.fog_enabled : b.fog_enabled;
        }
        result.has_fog_params = a.has_fog_params || b.has_fog_params;
        if (a.has_fog_params && b.has_fog_params) {
            result.fog_density = a.fog_density + (b.fog_density - a.fog_density) * t;
            result.fog_height = a.fog_height + (b.fog_height - a.fog_height) * t;
            result.fog_falloff = a.fog_falloff + (b.fog_falloff - a.fog_falloff) * t;
            result.fog_distance = a.fog_distance + (b.fog_distance - a.fog_distance) * t;
            result.fog_color = a.fog_color + (b.fog_color - a.fog_color) * t;
            result.fog_sun_scatter = a.fog_sun_scatter + (b.fog_sun_scatter - a.fog_sun_scatter) * t;
        } else if (a.has_fog_params) {
            result.fog_density = a.fog_density; result.fog_height = a.fog_height; result.fog_falloff = a.fog_falloff;
            result.fog_distance = a.fog_distance; result.fog_color = a.fog_color; result.fog_sun_scatter = a.fog_sun_scatter;
        } else if (b.has_fog_params) {
            result.fog_density = b.fog_density; result.fog_height = b.fog_height; result.fog_falloff = b.fog_falloff;
            result.fog_distance = b.fog_distance; result.fog_color = b.fog_color; result.fog_sun_scatter = b.fog_sun_scatter;
        }

        // ===== GOD RAYS =====
        result.has_godrays = a.has_godrays || b.has_godrays;
        if (result.has_godrays) {
            result.godrays_enabled = (t < 0.5f) ? a.godrays_enabled : b.godrays_enabled;
        }
        result.has_godrays_params = a.has_godrays_params || b.has_godrays_params;
        if (a.has_godrays_params && b.has_godrays_params) {
            result.godrays_intensity = a.godrays_intensity + (b.godrays_intensity - a.godrays_intensity) * t;
            result.godrays_density = a.godrays_density + (b.godrays_density - a.godrays_density) * t;
            result.godrays_samples = (t < 0.5f) ? a.godrays_samples : b.godrays_samples;
        } else if (a.has_godrays_params) {
            result.godrays_intensity = a.godrays_intensity; result.godrays_density = a.godrays_density;
            result.godrays_samples = a.godrays_samples;
        } else if (b.has_godrays_params) {
            result.godrays_intensity = b.godrays_intensity; result.godrays_density = b.godrays_density;
            result.godrays_samples = b.godrays_samples;
        }

        // ===== ADVANCED =====
        result.has_aerial_perspective = a.has_aerial_perspective || b.has_aerial_perspective;
        if (result.has_aerial_perspective) {
            result.aerial_perspective = (t < 0.5f) ? a.aerial_perspective : b.aerial_perspective;
        }
        result.has_aerial_params = a.has_aerial_params || b.has_aerial_params;
        if (a.has_aerial_params && b.has_aerial_params) {
            result.aerial_density = a.aerial_density + (b.aerial_density - a.aerial_density) * t;
            result.aerial_min_distance = a.aerial_min_distance + (b.aerial_min_distance - a.aerial_min_distance) * t;
            result.aerial_max_distance = a.aerial_max_distance + (b.aerial_max_distance - a.aerial_max_distance) * t;
        } else if (a.has_aerial_params) {
            result.aerial_density = a.aerial_density; result.aerial_min_distance = a.aerial_min_distance; result.aerial_max_distance = a.aerial_max_distance;
        } else if (b.has_aerial_params) {
            result.aerial_density = b.aerial_density; result.aerial_min_distance = b.aerial_min_distance; result.aerial_max_distance = b.aerial_max_distance;
        }

        result.has_multi_scatter = a.has_multi_scatter || b.has_multi_scatter;
        if (a.has_multi_scatter && b.has_multi_scatter) {
            result.multi_scatter_factor = a.multi_scatter_factor + (b.multi_scatter_factor - a.multi_scatter_factor) * t;
        } else if (a.has_multi_scatter) {
            result.multi_scatter_factor = a.multi_scatter_factor;
        } else if (b.has_multi_scatter) {
            result.multi_scatter_factor = b.multi_scatter_factor;
        }

        result.has_overlay = a.has_overlay || b.has_overlay;
        if (result.has_overlay) {
            result.env_overlay_enabled = (t < 0.5f) ? a.env_overlay_enabled : b.env_overlay_enabled;
        }
        result.has_overlay_params = a.has_overlay_params || b.has_overlay_params;
        if (a.has_overlay_params && b.has_overlay_params) {
            result.env_overlay_intensity = a.env_overlay_intensity + (b.env_overlay_intensity - a.env_overlay_intensity) * t;
            result.env_overlay_rotation = a.env_overlay_rotation + (b.env_overlay_rotation - a.env_overlay_rotation) * t;
        } else if (a.has_overlay_params) {
            result.env_overlay_intensity = a.env_overlay_intensity; result.env_overlay_rotation = a.env_overlay_rotation;
        } else if (b.has_overlay_params) {
            result.env_overlay_intensity = b.env_overlay_intensity; result.env_overlay_rotation = b.env_overlay_rotation;
        }

        result.has_weather_params = a.has_weather_params || b.has_weather_params;
        if (a.has_weather_params && b.has_weather_params) {
            result.weather_enabled = (t < 0.5f) ? a.weather_enabled : b.weather_enabled;
            result.weather_type = (t < 0.5f) ? a.weather_type : b.weather_type;
            result.weather_intensity = a.weather_intensity + (b.weather_intensity - a.weather_intensity) * t;
            result.weather_density = a.weather_density + (b.weather_density - a.weather_density) * t;
            result.weather_wind_direction = a.weather_wind_direction + (b.weather_wind_direction - a.weather_wind_direction) * t;
            result.weather_wind_speed = a.weather_wind_speed + (b.weather_wind_speed - a.weather_wind_speed) * t;
            result.weather_precipitation_scale = a.weather_precipitation_scale + (b.weather_precipitation_scale - a.weather_precipitation_scale) * t;
            result.weather_visibility = a.weather_visibility + (b.weather_visibility - a.weather_visibility) * t;
            result.weather_surface_wetness = a.weather_surface_wetness + (b.weather_surface_wetness - a.weather_surface_wetness) * t;
            result.weather_surface_accumulation = a.weather_surface_accumulation + (b.weather_surface_accumulation - a.weather_surface_accumulation) * t;
            result.weather_surface_settling = a.weather_surface_settling + (b.weather_surface_settling - a.weather_surface_settling) * t;
            result.weather_surface_height = a.weather_surface_height + (b.weather_surface_height - a.weather_surface_height) * t;
            result.weather_visual_mode = (t < 0.5f) ? a.weather_visual_mode : b.weather_visual_mode;
            result.weather_surface_response_enabled = (t < 0.5f) ? a.weather_surface_response_enabled : b.weather_surface_response_enabled;
        } else if (a.has_weather_params) {
            result.weather_enabled = a.weather_enabled; result.weather_type = a.weather_type;
            result.weather_intensity = a.weather_intensity; result.weather_density = a.weather_density;
            result.weather_wind_direction = a.weather_wind_direction; result.weather_wind_speed = a.weather_wind_speed;
            result.weather_precipitation_scale = a.weather_precipitation_scale; result.weather_visibility = a.weather_visibility;
            result.weather_surface_wetness = a.weather_surface_wetness; result.weather_surface_accumulation = a.weather_surface_accumulation;
            result.weather_surface_settling = a.weather_surface_settling;
            result.weather_surface_height = a.weather_surface_height;
            result.weather_visual_mode = a.weather_visual_mode;
            result.weather_surface_response_enabled = a.weather_surface_response_enabled;
        } else if (b.has_weather_params) {
            result.weather_enabled = b.weather_enabled; result.weather_type = b.weather_type;
            result.weather_intensity = b.weather_intensity; result.weather_density = b.weather_density;
            result.weather_wind_direction = b.weather_wind_direction; result.weather_wind_speed = b.weather_wind_speed;
            result.weather_precipitation_scale = b.weather_precipitation_scale; result.weather_visibility = b.weather_visibility;
            result.weather_surface_wetness = b.weather_surface_wetness; result.weather_surface_accumulation = b.weather_surface_accumulation;
            result.weather_surface_settling = b.weather_surface_settling;
            result.weather_surface_height = b.weather_surface_height;
            result.weather_visual_mode = b.weather_visual_mode;
            result.weather_surface_response_enabled = b.weather_surface_response_enabled;
        }

        
        return result;
    }
};

// ============================================================================
// TERRAIN KEYFRAME - Heightmap and Hardness map snapshots
// ============================================================================
struct TerrainKeyframe {
    std::vector<float> heightData;
    std::vector<float> hardnessData;
    std::vector<unsigned char> splatData;  // RGBA splat map (4 bytes per pixel)
    int width = 0;
    int height = 0;
    int splat_width = 0;   // Splat map dimensions (may differ from height map)
    int splat_height = 0;
    
    // Flags
    bool has_data = false;
    bool has_splat = false;
    
    TerrainKeyframe() = default;
    
    // Linear Interpolation for Terrain
    // This could be expensive for large terrains, but necessary for morphing
    static TerrainKeyframe lerp(const TerrainKeyframe& a, const TerrainKeyframe& b, float t) {
        TerrainKeyframe result;
        
        if (!a.has_data && !b.has_data) return result;
        
        // If dimensions mismatch or one is missing, just return the close one (snap)
        // Or handle simple replacement
        if (!a.has_data) return b;
        if (!b.has_data) return a;
        if (a.width != b.width || a.height != b.height) return (t < 0.5f) ? a : b;
        
        result.width = a.width;
        result.height = a.height;
        result.has_data = true;
        
        size_t size = a.heightData.size();
        result.heightData.resize(size);
        bool hasHardness = !a.hardnessData.empty() && !b.hardnessData.empty();
        if (hasHardness) result.hardnessData.resize(size);
        
        // Fast Lerp for height data
        for (size_t i = 0; i < size; i++) {
            result.heightData[i] = a.heightData[i] + (b.heightData[i] - a.heightData[i]) * t;
            if (hasHardness) {
                result.hardnessData[i] = a.hardnessData[i] + (b.hardnessData[i] - a.hardnessData[i]) * t;
            }
        }
        
        // Splat map interpolation (if both have splat data)
        bool hasSplat = a.has_splat && b.has_splat && 
                        a.splat_width == b.splat_width && 
                        a.splat_height == b.splat_height &&
                        !a.splatData.empty() && !b.splatData.empty();
        if (hasSplat) {
            result.has_splat = true;
            result.splat_width = a.splat_width;
            result.splat_height = a.splat_height;
            size_t splat_size = a.splatData.size();
            result.splatData.resize(splat_size);
            
            for (size_t i = 0; i < splat_size; i++) {
                float va = (float)a.splatData[i];
                float vb = (float)b.splatData[i];
                result.splatData[i] = (unsigned char)(va + (vb - va) * t);
            }
        } else if (a.has_splat) {
            result.has_splat = true;
            result.splat_width = a.splat_width;
            result.splat_height = a.splat_height;
            result.splatData = a.splatData;
        } else if (b.has_splat) {
            result.has_splat = true;
            result.splat_width = b.splat_width;
            result.splat_height = b.splat_height;
            result.splatData = b.splatData;
        }
        
        return result;
    }
};

// ============================================================================
// EMITTER KEYFRAME - Gas/Smoke emitter animation (Fuel, Density, Temperature)
// ============================================================================
struct EmitterKeyframe {
    // Per-property flags - which properties are keyed
    bool has_fuel_rate = false;
    bool has_density_rate = false;
    bool has_temperature = false;
    bool has_velocity = false;
    bool has_position = false;
    bool has_enabled = false;
    bool has_size = false;          // NEW
    bool has_radius = false;        // NEW
    
    // Property values
    float fuel_rate = 0.0f;          // Fuel injection rate
    float density_rate = 10.0f;       // Density injection rate
    float temperature = 500.0f;       // Temperature in Kelvin
    Vec3 velocity = Vec3(0, 2, 0);   // Emission velocity
    Vec3 position = Vec3(0, 0, 0);   // Emitter position
    Vec3 size = Vec3(1, 1, 1);       // Emitter size/extents (NEW)
    float radius = 1.0f;             // Emitter radius (NEW)
    bool enabled = true;              // On/Off state
    
    EmitterKeyframe() = default;
    
    // Lerp only interpolates properties that are keyed in BOTH keyframes
    static EmitterKeyframe lerp(const EmitterKeyframe& a, const EmitterKeyframe& b, float t) {
        EmitterKeyframe result;
        
        // Fuel Rate
        result.has_fuel_rate = a.has_fuel_rate || b.has_fuel_rate;
        if (a.has_fuel_rate && b.has_fuel_rate) {
            result.fuel_rate = a.fuel_rate + (b.fuel_rate - a.fuel_rate) * t;
        } else if (a.has_fuel_rate) {
            result.fuel_rate = a.fuel_rate;
        } else if (b.has_fuel_rate) {
            result.fuel_rate = b.fuel_rate;
        }
        
        // Density Rate
        result.has_density_rate = a.has_density_rate || b.has_density_rate;
        if (a.has_density_rate && b.has_density_rate) {
            result.density_rate = a.density_rate + (b.density_rate - a.density_rate) * t;
        } else if (a.has_density_rate) {
            result.density_rate = a.density_rate;
        } else if (b.has_density_rate) {
            result.density_rate = b.density_rate;
        }
        
        // Temperature
        result.has_temperature = a.has_temperature || b.has_temperature;
        if (a.has_temperature && b.has_temperature) {
            result.temperature = a.temperature + (b.temperature - a.temperature) * t;
        } else if (a.has_temperature) {
            result.temperature = a.temperature;
        } else if (b.has_temperature) {
            result.temperature = b.temperature;
        }
        
        // Velocity
        result.has_velocity = a.has_velocity || b.has_velocity;
        if (a.has_velocity && b.has_velocity) {
            result.velocity = a.velocity + (b.velocity - a.velocity) * t;
        } else if (a.has_velocity) {
            result.velocity = a.velocity;
        } else if (b.has_velocity) {
            result.velocity = b.velocity;
        }
        
        // Position
        result.has_position = a.has_position || b.has_position;
        if (a.has_position && b.has_position) {
            result.position = a.position + (b.position - a.position) * t;
        } else if (a.has_position) {
            result.position = a.position;
        } else if (b.has_position) {
            result.position = b.position;
        }
        
        // Enabled (boolean - snap at t=0.5)
        result.has_enabled = a.has_enabled || b.has_enabled;
        if (a.has_enabled && b.has_enabled) {
            result.enabled = (t < 0.5f) ? a.enabled : b.enabled;
        } else if (a.has_enabled) {
            result.enabled = a.enabled;
        } else if (b.has_enabled) {
            result.enabled = b.enabled;
        }
        
        // Size
        result.has_size = a.has_size || b.has_size;
        if (a.has_size && b.has_size) {
            result.size = a.size + (b.size - a.size) * t;
        } else if (a.has_size) {
            result.size = a.size;
        } else if (b.has_size) {
            result.size = b.size;
        }

        // Radius
        result.has_radius = a.has_radius || b.has_radius;
        if (a.has_radius && b.has_radius) {
            result.radius = a.radius + (b.radius - a.radius) * t;
        } else if (a.has_radius) {
            result.radius = a.radius;
        } else if (b.has_radius) {
            result.radius = b.radius;
        }
        
        return result;
    }
};

// ============================================================================
// ANIM GRAPH KEYFRAME - Character animation parameters and state overrides
// ============================================================================
struct AnimGraphKeyframe {
    std::unordered_map<std::string, float> float_params;
    std::unordered_map<std::string, bool> bool_params;
    std::unordered_map<std::string, int> int_params;
    std::unordered_map<uint32_t, std::string> clip_overrides;
    std::unordered_map<uint32_t, float> clip_speed_overrides;
    std::vector<std::string> triggers;
    std::string force_state;

    static AnimGraphKeyframe blend(const AnimGraphKeyframe& a, const AnimGraphKeyframe& b, float t) {
        AnimGraphKeyframe result;

        for (const auto& [name, value] : a.float_params) {
            auto it = b.float_params.find(name);
            result.float_params[name] = (it != b.float_params.end())
                ? value + (it->second - value) * t
                : value;
        }
        for (const auto& [name, value] : b.float_params) {
            if (result.float_params.find(name) == result.float_params.end()) {
                result.float_params[name] = value;
            }
        }

        for (const auto& [name, value] : a.bool_params) {
            auto it = b.bool_params.find(name);
            result.bool_params[name] = (it != b.bool_params.end()) ? ((t < 0.5f) ? value : it->second) : value;
        }
        for (const auto& [name, value] : b.bool_params) {
            if (result.bool_params.find(name) == result.bool_params.end()) {
                result.bool_params[name] = value;
            }
        }

        for (const auto& [name, value] : a.int_params) {
            auto it = b.int_params.find(name);
            result.int_params[name] = (it != b.int_params.end()) ? ((t < 0.5f) ? value : it->second) : value;
        }
        for (const auto& [name, value] : b.int_params) {
            if (result.int_params.find(name) == result.int_params.end()) {
                result.int_params[name] = value;
            }
        }

        result.clip_overrides = (t < 0.5f) ? a.clip_overrides : b.clip_overrides;
        result.clip_speed_overrides = (t < 0.5f) ? a.clip_speed_overrides : b.clip_speed_overrides;

        result.force_state = (t < 0.5f) ? a.force_state : b.force_state;
        return result;
    }
};

// ============================================================================
// CURVE INTERPOLATION - per-channel keyframe interpolation (graph editor)
// ============================================================================

// ChannelKeyMeta and KeyInterp moved to top of file

// Transform channel indices shared by curve storage, evaluation and the graph editor UI.
enum : int {
    CURVE_POS_X = 0, CURVE_POS_Y, CURVE_POS_Z,
    CURVE_ROT_X, CURVE_ROT_Y, CURVE_ROT_Z,
    CURVE_SCL_X, CURVE_SCL_Y, CURVE_SCL_Z,
    CURVE_CHANNEL_COUNT
};

// Evaluate one curve segment between (f0,v0) and (f1,v1) at `frame`.
// The segment's mode comes from the LEFT key's meta (m0); handles use m0.out / m1.in.
// Bezier control X is clamped inside [f0,f1] so the curve stays a function of time;
// the parametric root is found by bisection (robust even for degenerate handles).
// NOTE: a Bezier key with zero-length handles reduces exactly to Linear (the x and
// y polynomials share the same basis weights), so un-refreshed keys degrade safely.
inline float evalCurveSegment(float f0, float v0, const ChannelKeyMeta& m0,
                              float f1, float v1, const ChannelKeyMeta& m1,
                              float frame)
{
    if (f1 <= f0)   return v0;
    if (frame <= f0) return v0;
    if (frame >= f1) return v1;

    switch (m0.interp) {
    case KeyInterp::Constant:
        return v0;
    case KeyInterp::Linear:
        return v0 + (v1 - v0) * ((frame - f0) / (f1 - f0));
    case KeyInterp::Bezier:
    default: {
        const float seg = f1 - f0;
        const float x1 = f0 + std::clamp(m0.out_dx, 0.0f, seg);
        const float y1 = v0 + m0.out_dy;
        const float x2 = f1 + std::clamp(m1.in_dx, -seg, 0.0f);
        const float y2 = v1 + m1.in_dy;

        // Solve cubic-bezier x(t) = frame by bisection (x(0)=f0 < frame < f1=x(1)).
        float lo = 0.0f, hi = 1.0f, t = 0.5f;
        for (int i = 0; i < 26; ++i) {
            t = 0.5f * (lo + hi);
            const float u = 1.0f - t;
            const float x = u*u*u*f0 + 3.0f*u*u*t*x1 + 3.0f*u*t*t*x2 + t*t*t*f1;
            if (x < frame) lo = t; else hi = t;
        }
        const float u = 1.0f - t;
        return u*u*u*v0 + 3.0f*u*u*t*y1 + 3.0f*u*t*t*y2 + t*t*t*v1;
    }
    }
}

// ============================================================================
// TRANSFORM KEYFRAME - Position, rotation (Euler), scale
// ============================================================================

// Transform keyframe - stores position, rotation (euler), scale
// Each component can be independently keyed
struct TransformKeyframe {
    Vec3 position = Vec3(0, 0, 0);
    Vec3 rotation = Vec3(0, 0, 0);  // Euler angles in degrees
    Vec3 scale = Vec3(1, 1, 1);
    
    // Per-channel keyed flags (compound)
    bool has_position = true;   // Location keyed
    bool has_rotation = true;   // Rotation keyed
    bool has_scale = true;      // Scale keyed

    // Detailed per-axis flags
    bool has_pos_x = true; bool has_pos_y = true; bool has_pos_z = true;
    bool has_rot_x = true; bool has_rot_y = true; bool has_rot_z = true;
    bool has_scl_x = true; bool has_scl_y = true; bool has_scl_z = true;

    // Per-channel curve metadata (graph editor): indexed by CURVE_* channel ids.
    ChannelKeyMeta curve[CURVE_CHANNEL_COUNT];

    TransformKeyframe() = default;

    TransformKeyframe(const Vec3& pos, const Vec3& rot, const Vec3& scl)
        : position(pos), rotation(rot), scale(scl) {
    };

    // ----- Generic channel access (single authority for axis <-> value mapping) -----
    float channelValue(int ch) const {
        switch (ch) {
        case CURVE_POS_X: return position.x;
        case CURVE_POS_Y: return position.y;
        case CURVE_POS_Z: return position.z;
        case CURVE_ROT_X: return rotation.x;
        case CURVE_ROT_Y: return rotation.y;
        case CURVE_ROT_Z: return rotation.z;
        case CURVE_SCL_X: return scale.x;
        case CURVE_SCL_Y: return scale.y;
        case CURVE_SCL_Z: return scale.z;
        }
        return 0.0f;
    }
    void setChannelValue(int ch, float v) {
        switch (ch) {
        case CURVE_POS_X: position.x = v; break;
        case CURVE_POS_Y: position.y = v; break;
        case CURVE_POS_Z: position.z = v; break;
        case CURVE_ROT_X: rotation.x = v; break;
        case CURVE_ROT_Y: rotation.y = v; break;
        case CURVE_ROT_Z: rotation.z = v; break;
        case CURVE_SCL_X: scale.x = v; break;
        case CURVE_SCL_Y: scale.y = v; break;
        case CURVE_SCL_Z: scale.z = v; break;
        }
    }
    bool channelKeyed(int ch) const {
        switch (ch) {
        case CURVE_POS_X: return has_pos_x;
        case CURVE_POS_Y: return has_pos_y;
        case CURVE_POS_Z: return has_pos_z;
        case CURVE_ROT_X: return has_rot_x;
        case CURVE_ROT_Y: return has_rot_y;
        case CURVE_ROT_Z: return has_rot_z;
        case CURVE_SCL_X: return has_scl_x;
        case CURVE_SCL_Y: return has_scl_y;
        case CURVE_SCL_Z: return has_scl_z;
        }
        return false;
    }
    void setChannelKeyed(int ch, bool on) {
        switch (ch) {
        case CURVE_POS_X: has_pos_x = on; break;
        case CURVE_POS_Y: has_pos_y = on; break;
        case CURVE_POS_Z: has_pos_z = on; break;
        case CURVE_ROT_X: has_rot_x = on; break;
        case CURVE_ROT_Y: has_rot_y = on; break;
        case CURVE_ROT_Z: has_rot_z = on; break;
        case CURVE_SCL_X: has_scl_x = on; break;
        case CURVE_SCL_Y: has_scl_y = on; break;
        case CURVE_SCL_Z: has_scl_z = on; break;
        }
    }
    // Rebuild compound L/R/S flags from the per-axis flags.
    void refreshCompoundFlags() {
        has_position = has_pos_x || has_pos_y || has_pos_z;
        has_rotation = has_rot_x || has_rot_y || has_rot_z;
        has_scale    = has_scl_x || has_scl_y || has_scl_z;
    }
    // Clear every keyed flag (defaults are all-true for authoring convenience,
    // so evaluation/result keyframes must start from an unkeyed state).
    void clearAllChannels() {
        has_position = has_rotation = has_scale = false;
        has_pos_x = has_pos_y = has_pos_z = false;
        has_rot_x = has_rot_y = has_rot_z = false;
        has_scl_x = has_scl_y = has_scl_z = false;
    }
    static const char* channelName(int ch) {
        static const char* names[CURVE_CHANNEL_COUNT] = {
            "Pos X", "Pos Y", "Pos Z",
            "Rot X", "Rot Y", "Rot Z",
            "Scale X", "Scale Y", "Scale Z"
        };
        return (ch >= 0 && ch < CURVE_CHANNEL_COUNT) ? names[ch] : "?";
    }

    // Linear interpolation - respects per-axis flags
    static TransformKeyframe lerp(const TransformKeyframe& a, const TransformKeyframe& b, float t) {
        TransformKeyframe result;
        
        // --- POSITION ---
        // X
        if (a.has_pos_x && b.has_pos_x) result.position.x = a.position.x + (b.position.x - a.position.x) * t;
        else if (a.has_pos_x) result.position.x = a.position.x;
        else if (b.has_pos_x) result.position.x = b.position.x;
        result.has_pos_x = a.has_pos_x || b.has_pos_x;

        // Y
        if (a.has_pos_y && b.has_pos_y) result.position.y = a.position.y + (b.position.y - a.position.y) * t;
        else if (a.has_pos_y) result.position.y = a.position.y;
        else if (b.has_pos_y) result.position.y = b.position.y;
        result.has_pos_y = a.has_pos_y || b.has_pos_y;

        // Z
        if (a.has_pos_z && b.has_pos_z) result.position.z = a.position.z + (b.position.z - a.position.z) * t;
        else if (a.has_pos_z) result.position.z = a.position.z;
        else if (b.has_pos_z) result.position.z = b.position.z;
        result.has_pos_z = a.has_pos_z || b.has_pos_z;

        // Compound Flag Update
        result.has_position = result.has_pos_x || result.has_pos_y || result.has_pos_z;
        
        // --- ROTATION ---
        // X
        if (a.has_rot_x && b.has_rot_x) result.rotation.x = a.rotation.x + (b.rotation.x - a.rotation.x) * t;
        else if (a.has_rot_x) result.rotation.x = a.rotation.x;
        else if (b.has_rot_x) result.rotation.x = b.rotation.x;
        result.has_rot_x = a.has_rot_x || b.has_rot_x;

        // Y
        if (a.has_rot_y && b.has_rot_y) result.rotation.y = a.rotation.y + (b.rotation.y - a.rotation.y) * t;
        else if (a.has_rot_y) result.rotation.y = a.rotation.y;
        else if (b.has_rot_y) result.rotation.y = b.rotation.y;
        result.has_rot_y = a.has_rot_y || b.has_rot_y;

        // Z
        if (a.has_rot_z && b.has_rot_z) result.rotation.z = a.rotation.z + (b.rotation.z - a.rotation.z) * t;
        else if (a.has_rot_z) result.rotation.z = a.rotation.z;
        else if (b.has_rot_z) result.rotation.z = b.rotation.z;
        result.has_rot_z = a.has_rot_z || b.has_rot_z;

        result.has_rotation = result.has_rot_x || result.has_rot_y || result.has_rot_z;
        
        // --- SCALE ---
        // X
        if (a.has_scl_x && b.has_scl_x) result.scale.x = a.scale.x + (b.scale.x - a.scale.x) * t;
        else if (a.has_scl_x) result.scale.x = a.scale.x;
        else if (b.has_scl_x) result.scale.x = b.scale.x;
        result.has_scl_x = a.has_scl_x || b.has_scl_x;

        // Y
        if (a.has_scl_y && b.has_scl_y) result.scale.y = a.scale.y + (b.scale.y - a.scale.y) * t;
        else if (a.has_scl_y) result.scale.y = a.scale.y;
        else if (b.has_scl_y) result.scale.y = b.scale.y;
        result.has_scl_y = a.has_scl_y || b.has_scl_y;

        // Z
        if (a.has_scl_z && b.has_scl_z) result.scale.z = a.scale.z + (b.scale.z - a.scale.z) * t;
        else if (a.has_scl_z) result.scale.z = a.scale.z;
        else if (b.has_scl_z) result.scale.z = b.scale.z;
        result.has_scl_z = a.has_scl_z || b.has_scl_z;
        
        result.has_scale = result.has_scl_x || result.has_scl_y || result.has_scl_z;
        
        return result;
    }
};


// ============================================================================
// WATER KEYFRAME - Ocean/water surface animation parameters
// ============================================================================
struct WaterKeyframe {
    // Water surface identification
    int water_surface_id = -1;
    
    // Per-property flags - Geometric Waves
    bool has_wave_height = false;
    bool has_wave_scale = false;
    bool has_wind_direction = false;
    bool has_choppiness = false;
    bool has_geo_speed = false;
    bool has_alignment = false;
    bool has_damping = false;
    bool has_swell_amplitude = false;
    bool has_sharpening = false;
    bool has_detail_strength = false;
    
    // Per-property flags - FFT Ocean
    bool has_fft_wind_speed = false;
    bool has_fft_wind_direction = false;
    bool has_fft_amplitude = false;
    bool has_fft_choppiness = false;
    bool has_fft_time_scale = false;
    bool has_fft_ocean_size = false;
    
    // Property values - Geometric Waves (matches WaterWaveParams)
    float wave_height = 2.0f;
    float wave_scale = 50.0f;
    float wind_direction = 0.0f;      // degrees
    float choppiness = 1.0f;
    float geo_speed = 1.0f;           // Animation speed
    float alignment = 0.5f;
    float damping = 0.0f;
    float swell_amplitude = 0.2f;
    float sharpening = 0.0f;
    float detail_strength = 0.15f;
    
    // Property values - FFT Ocean
    float fft_wind_speed = 10.0f;       // m/s - Storm intensity
    float fft_wind_direction = 0.0f;    // degrees - Wave travel direction
    float fft_amplitude = 0.0002f;      // Phillips A - Overall wave scale
    float fft_choppiness = 1.0f;        // Horizontal displacement
    float fft_time_scale = 1.0f;        // Animation speed
    float fft_ocean_size = 100.0f;      // Tile size
    
    WaterKeyframe() = default;
    
    static WaterKeyframe lerp(const WaterKeyframe& a, const WaterKeyframe& b, float t) {
        WaterKeyframe result;
        result.water_surface_id = a.water_surface_id;
        
        // Wave Height
        result.has_wave_height = a.has_wave_height || b.has_wave_height;
        if (a.has_wave_height && b.has_wave_height) {
            result.wave_height = a.wave_height + (b.wave_height - a.wave_height) * t;
        } else if (a.has_wave_height) {
            result.wave_height = a.wave_height;
        } else if (b.has_wave_height) {
            result.wave_height = b.wave_height;
        }
        
        // Wave Scale
        result.has_wave_scale = a.has_wave_scale || b.has_wave_scale;
        if (a.has_wave_scale && b.has_wave_scale) {
            result.wave_scale = a.wave_scale + (b.wave_scale - a.wave_scale) * t;
        } else if (a.has_wave_scale) {
            result.wave_scale = a.wave_scale;
        } else if (b.has_wave_scale) {
            result.wave_scale = b.wave_scale;
        }
        
        // Wind Direction (with angle wrapping)
        result.has_wind_direction = a.has_wind_direction || b.has_wind_direction;
        if (a.has_wind_direction && b.has_wind_direction) {
            // Handle angle wrapping for smooth interpolation
            float diff = b.wind_direction - a.wind_direction;
            if (diff > 180.0f) diff -= 360.0f;
            if (diff < -180.0f) diff += 360.0f;
            result.wind_direction = a.wind_direction + diff * t;
            if (result.wind_direction < 0) result.wind_direction += 360.0f;
            if (result.wind_direction >= 360.0f) result.wind_direction -= 360.0f;
        } else if (a.has_wind_direction) {
            result.wind_direction = a.wind_direction;
        } else if (b.has_wind_direction) {
            result.wind_direction = b.wind_direction;
        }
        
        // Choppiness
        result.has_choppiness = a.has_choppiness || b.has_choppiness;
        if (a.has_choppiness && b.has_choppiness) {
            result.choppiness = a.choppiness + (b.choppiness - a.choppiness) * t;
        } else if (a.has_choppiness) {
            result.choppiness = a.choppiness;
        } else if (b.has_choppiness) {
            result.choppiness = b.choppiness;
        }
        
        // Geo Speed (animation speed for geometric waves)
        result.has_geo_speed = a.has_geo_speed || b.has_geo_speed;
        if (a.has_geo_speed && b.has_geo_speed) {
            result.geo_speed = a.geo_speed + (b.geo_speed - a.geo_speed) * t;
        } else if (a.has_geo_speed) {
            result.geo_speed = a.geo_speed;
        } else if (b.has_geo_speed) {
            result.geo_speed = b.geo_speed;
        }
        
        // Alignment
        result.has_alignment = a.has_alignment || b.has_alignment;
        if (a.has_alignment && b.has_alignment) {
            result.alignment = a.alignment + (b.alignment - a.alignment) * t;
        } else if (a.has_alignment) {
            result.alignment = a.alignment;
        } else if (b.has_alignment) {
            result.alignment = b.alignment;
        }
        
        // Damping
        result.has_damping = a.has_damping || b.has_damping;
        if (a.has_damping && b.has_damping) {
            result.damping = a.damping + (b.damping - a.damping) * t;
        } else if (a.has_damping) {
            result.damping = a.damping;
        } else if (b.has_damping) {
            result.damping = b.damping;
        }
        
        // Swell Amplitude
        result.has_swell_amplitude = a.has_swell_amplitude || b.has_swell_amplitude;
        if (a.has_swell_amplitude && b.has_swell_amplitude) {
            result.swell_amplitude = a.swell_amplitude + (b.swell_amplitude - a.swell_amplitude) * t;
        } else if (a.has_swell_amplitude) {
            result.swell_amplitude = a.swell_amplitude;
        } else if (b.has_swell_amplitude) {
            result.swell_amplitude = b.swell_amplitude;
        }
        
        // Sharpening
        result.has_sharpening = a.has_sharpening || b.has_sharpening;
        if (a.has_sharpening && b.has_sharpening) {
            result.sharpening = a.sharpening + (b.sharpening - a.sharpening) * t;
        } else if (a.has_sharpening) {
            result.sharpening = a.sharpening;
        } else if (b.has_sharpening) {
            result.sharpening = b.sharpening;
        }
        
        // Detail Strength
        result.has_detail_strength = a.has_detail_strength || b.has_detail_strength;
        if (a.has_detail_strength && b.has_detail_strength) {
            result.detail_strength = a.detail_strength + (b.detail_strength - a.detail_strength) * t;
        } else if (a.has_detail_strength) {
            result.detail_strength = a.detail_strength;
        } else if (b.has_detail_strength) {
            result.detail_strength = b.detail_strength;
        }
        
        // ═══════════════════════════════════════════════════════════════════════
        // FFT OCEAN PARAMETERS
        // ═══════════════════════════════════════════════════════════════════════
        
        // FFT Wind Speed (Storm intensity transition)
        result.has_fft_wind_speed = a.has_fft_wind_speed || b.has_fft_wind_speed;
        if (a.has_fft_wind_speed && b.has_fft_wind_speed) {
            result.fft_wind_speed = a.fft_wind_speed + (b.fft_wind_speed - a.fft_wind_speed) * t;
        } else if (a.has_fft_wind_speed) {
            result.fft_wind_speed = a.fft_wind_speed;
        } else if (b.has_fft_wind_speed) {
            result.fft_wind_speed = b.fft_wind_speed;
        }
        
        // FFT Wind Direction (with angle wrapping)
        result.has_fft_wind_direction = a.has_fft_wind_direction || b.has_fft_wind_direction;
        if (a.has_fft_wind_direction && b.has_fft_wind_direction) {
            float diff = b.fft_wind_direction - a.fft_wind_direction;
            if (diff > 180.0f) diff -= 360.0f;
            if (diff < -180.0f) diff += 360.0f;
            result.fft_wind_direction = a.fft_wind_direction + diff * t;
            if (result.fft_wind_direction < 0) result.fft_wind_direction += 360.0f;
            if (result.fft_wind_direction >= 360.0f) result.fft_wind_direction -= 360.0f;
        } else if (a.has_fft_wind_direction) {
            result.fft_wind_direction = a.fft_wind_direction;
        } else if (b.has_fft_wind_direction) {
            result.fft_wind_direction = b.fft_wind_direction;
        }
        
        // FFT Amplitude
        result.has_fft_amplitude = a.has_fft_amplitude || b.has_fft_amplitude;
        if (a.has_fft_amplitude && b.has_fft_amplitude) {
            result.fft_amplitude = a.fft_amplitude + (b.fft_amplitude - a.fft_amplitude) * t;
        } else if (a.has_fft_amplitude) {
            result.fft_amplitude = a.fft_amplitude;
        } else if (b.has_fft_amplitude) {
            result.fft_amplitude = b.fft_amplitude;
        }
        
        // FFT Choppiness
        result.has_fft_choppiness = a.has_fft_choppiness || b.has_fft_choppiness;
        if (a.has_fft_choppiness && b.has_fft_choppiness) {
            result.fft_choppiness = a.fft_choppiness + (b.fft_choppiness - a.fft_choppiness) * t;
        } else if (a.has_fft_choppiness) {
            result.fft_choppiness = a.fft_choppiness;
        } else if (b.has_fft_choppiness) {
            result.fft_choppiness = b.fft_choppiness;
        }
        
        // FFT Time Scale (Animation speed)
        result.has_fft_time_scale = a.has_fft_time_scale || b.has_fft_time_scale;
        if (a.has_fft_time_scale && b.has_fft_time_scale) {
            result.fft_time_scale = a.fft_time_scale + (b.fft_time_scale - a.fft_time_scale) * t;
        } else if (a.has_fft_time_scale) {
            result.fft_time_scale = a.fft_time_scale;
        } else if (b.has_fft_time_scale) {
            result.fft_time_scale = b.fft_time_scale;
        }
        
        // FFT Ocean Size
        result.has_fft_ocean_size = a.has_fft_ocean_size || b.has_fft_ocean_size;
        if (a.has_fft_ocean_size && b.has_fft_ocean_size) {
            result.fft_ocean_size = a.fft_ocean_size + (b.fft_ocean_size - a.fft_ocean_size) * t;
        } else if (a.has_fft_ocean_size) {
            result.fft_ocean_size = a.fft_ocean_size;
        } else if (b.has_fft_ocean_size) {
            result.fft_ocean_size = b.fft_ocean_size;
        }
        
        return result;
    }
};


// Complete keyframe - combines all types of animation data
struct Keyframe {
    int frame = 0;
    TransformKeyframe transform;
    MaterialKeyframe material;
    LightKeyframe light;
    CameraKeyframe camera;
    WorldKeyframe world;
    
    // Flags to track what's keyframed
    bool has_transform = false;
    bool has_material = false;
    bool has_light = false;
    bool has_camera = false;   
    bool has_world = false;
    bool has_terrain = false;
    bool has_water = false;     // NEW: Water animation support
    bool has_anim_graph = false;
    
    TerrainKeyframe terrain;
    WaterKeyframe water;        // NEW
    bool has_emitter = false;   // NEW: Gas Emitter support
    EmitterKeyframe emitter;    // NEW
    AnimGraphKeyframe anim_graph;
    
    Keyframe() = default;
    Keyframe(int f) : frame(f) {}
};

// Object animation track - stores all keyframes for a single object
struct ObjectAnimationTrack {
    std::string object_name;          // Name/ID of the object
    int object_index = -1;            // Index in scene.world.objects
    std::vector<Keyframe> keyframes;  // Sorted by frame number
    
    // Add a keyframe (maintains sorted order)
    void addKeyframe(const Keyframe& kf) {
        // Find insertion point
        auto it = std::lower_bound(keyframes.begin(), keyframes.end(), kf,
            [](const Keyframe& a, const Keyframe& b) { return a.frame < b.frame; });
        
        // If keyframe already exists at this frame, MERGE it (don't overwrite)
        if (it != keyframes.end() && it->frame == kf.frame) {
            if (kf.has_transform) {
                // Preserve curve metadata the user customised on this key: a re-key
                // (auto-key / Insert Keyframe) carries factory-default meta, which
                // must not wipe manual interpolation modes or hand-edited tangents.
                TransformKeyframe merged = kf.transform;
                for (int ch = 0; ch < CURVE_CHANNEL_COUNT; ++ch) {
                    if (kf.transform.curve[ch].isPristineAuto())
                        merged.curve[ch] = it->transform.curve[ch];
                }
                it->transform = merged;
                it->has_transform = true;
            }
            if (kf.has_material) {
                MaterialKeyframe merged = kf.material;
                for (int ch = 0; ch < CURVE_MAT_CHANNEL_COUNT; ++ch) {
                    if (kf.material.curve[ch].isPristineAuto())
                        merged.curve[ch] = it->material.curve[ch];
                }
                it->material = merged;
                it->has_material = true;
            }
            if (kf.has_light) {
                LightKeyframe merged = kf.light;
                for (int ch = 0; ch < CURVE_LIGHT_CHANNEL_COUNT; ++ch) {
                    if (kf.light.curve[ch].isPristineAuto())
                        merged.curve[ch] = it->light.curve[ch];
                }
                it->light = merged;
                it->has_light = true;
            }
            if (kf.has_camera) {
                CameraKeyframe merged = kf.camera;
                for (int ch = 0; ch < CURVE_CAM_CHANNEL_COUNT; ++ch) {
                    if (kf.camera.curve[ch].isPristineAuto())
                        merged.curve[ch] = it->camera.curve[ch];
                }
                it->camera = merged;
                it->has_camera = true;
            }
            if (kf.has_world) {
                it->world = kf.world;
                it->has_world = true;
            }
            if (kf.has_terrain) {
                it->terrain = kf.terrain;
                it->has_terrain = true;
            }
            if (kf.has_water) {
                it->water = kf.water;
                it->has_water = true;
            }
            if (kf.has_anim_graph) {
                it->anim_graph = kf.anim_graph;
                it->has_anim_graph = true;
            }
            if (kf.has_emitter) {
                it->emitter = kf.emitter;
                it->has_emitter = true;
            }
        } else {
            keyframes.insert(it, kf);
        }
    }
    
    // Remove keyframe at frame
    void removeKeyframe(int frame) {
        keyframes.erase(
            std::remove_if(keyframes.begin(), keyframes.end(),
                [frame](const Keyframe& kf) { return kf.frame == frame; }),
            keyframes.end()
        );
    }

    // Recompute auto handles (auto-clamped Catmull-Rom) for transform keys whose
    // channel meta has auto_tangent set. Handles only matter where a Bezier
    // segment touches the key, so Linear-only keys are left untouched (keeps
    // legacy serialization sparse). Call after any key add/move/value edit.
    void refreshAutoTangents() {
        std::vector<Keyframe*> keys;
        
        // Transform
        for (int ch = 0; ch < CURVE_CHANNEL_COUNT; ++ch) {
            keys.clear();
            for (auto& kf : keyframes)
                if (kf.has_transform && kf.transform.channelKeyed(ch)) keys.push_back(&kf);
            const int count = (int)keys.size();
            for (int i = 0; i < count; ++i) {
                ChannelKeyMeta& m = keys[i]->transform.curve[ch];
                if (!m.auto_tangent) continue;
                // The in-handle participates in the segment coming FROM the previous
                // key, so refresh also when only the previous key is Bezier.
                const bool touches_bezier =
                    m.interp == KeyInterp::Bezier ||
                    (i > 0 && keys[i - 1]->transform.curve[ch].interp == KeyInterp::Bezier);
                if (!touches_bezier) continue;

                const float f  = (float)keys[i]->frame;
                const float v  = keys[i]->transform.channelValue(ch);
                const float fp = (i > 0) ? (float)keys[i - 1]->frame : f;
                const float vp = (i > 0) ? keys[i - 1]->transform.channelValue(ch) : v;
                const float fn = (i + 1 < count) ? (float)keys[i + 1]->frame : f;
                const float vn = (i + 1 < count) ? keys[i + 1]->transform.channelValue(ch) : v;

                // Catmull-Rom slope, flattened at local extremes so the curve never
                // overshoots its neighbouring keys (Blender "Auto Clamped").
                // Endpoints stay flat: ease in from rest / settle at the end.
                float slope = 0.0f;
                if (i > 0 && i + 1 < count && fn > fp) {
                    const bool is_max = v >= vp && v >= vn;
                    const bool is_min = v <= vp && v <= vn;
                    if (!is_max && !is_min) slope = (vn - vp) / (fn - fp);
                }

                const float in_len  = (i > 0) ? (f - fp) / 3.0f : 0.0f;
                const float out_len = (i + 1 < count) ? (fn - f) / 3.0f : 0.0f;
                m.in_dx  = -in_len;  m.in_dy  = -slope * in_len;
                m.out_dx =  out_len; m.out_dy =  slope * out_len;

                // Clamp handle endpoints into the neighbour value range (no overshoot).
                if (i > 0) {
                    const float lo = std::min(vp, v), hi = std::max(vp, v);
                    m.in_dy = std::clamp(v + m.in_dy, lo, hi) - v;
                }
                if (i + 1 < count) {
                    const float lo = std::min(v, vn), hi = std::max(v, vn);
                    m.out_dy = std::clamp(v + m.out_dy, lo, hi) - v;
                }
            }
        }

        // Light
        for (int ch = 0; ch < CURVE_LIGHT_CHANNEL_COUNT; ++ch) {
            keys.clear();
            for (auto& kf : keyframes)
                if (kf.has_light && kf.light.channelKeyed(ch)) keys.push_back(&kf);
            const int count = (int)keys.size();
            for (int i = 0; i < count; ++i) {
                ChannelKeyMeta& m = keys[i]->light.curve[ch];
                if (!m.auto_tangent) continue;
                const bool touches_bezier =
                    m.interp == KeyInterp::Bezier ||
                    (i > 0 && keys[i - 1]->light.curve[ch].interp == KeyInterp::Bezier);
                if (!touches_bezier) continue;

                const float f  = (float)keys[i]->frame;
                const float v  = keys[i]->light.channelValue(ch);
                const float fp = (i > 0) ? (float)keys[i - 1]->frame : f;
                const float vp = (i > 0) ? keys[i - 1]->light.channelValue(ch) : v;
                const float fn = (i + 1 < count) ? (float)keys[i + 1]->frame : f;
                const float vn = (i + 1 < count) ? keys[i + 1]->light.channelValue(ch) : v;

                float slope = 0.0f;
                if (i > 0 && i + 1 < count && fn > fp) {
                    const bool is_max = v >= vp && v >= vn;
                    const bool is_min = v <= vp && v <= vn;
                    if (!is_max && !is_min) slope = (vn - vp) / (fn - fp);
                }

                const float in_len  = (i > 0) ? (f - fp) / 3.0f : 0.0f;
                const float out_len = (i + 1 < count) ? (fn - f) / 3.0f : 0.0f;
                m.in_dx  = -in_len;  m.in_dy  = -slope * in_len;
                m.out_dx =  out_len; m.out_dy =  slope * out_len;

                if (i > 0) {
                    const float lo = std::min(vp, v), hi = std::max(vp, v);
                    m.in_dy = std::clamp(v + m.in_dy, lo, hi) - v;
                }
                if (i + 1 < count) {
                    const float lo = std::min(v, vn), hi = std::max(v, vn);
                    m.out_dy = std::clamp(v + m.out_dy, lo, hi) - v;
                }
            }
        }

        // Camera
        for (int ch = 0; ch < CURVE_CAM_CHANNEL_COUNT; ++ch) {
            keys.clear();
            for (auto& kf : keyframes)
                if (kf.has_camera && kf.camera.channelKeyed(ch)) keys.push_back(&kf);
            const int count = (int)keys.size();
            for (int i = 0; i < count; ++i) {
                ChannelKeyMeta& m = keys[i]->camera.curve[ch];
                if (!m.auto_tangent) continue;
                const bool touches_bezier =
                    m.interp == KeyInterp::Bezier ||
                    (i > 0 && keys[i - 1]->camera.curve[ch].interp == KeyInterp::Bezier);
                if (!touches_bezier) continue;

                const float f  = (float)keys[i]->frame;
                const float v  = keys[i]->camera.channelValue(ch);
                const float fp = (i > 0) ? (float)keys[i - 1]->frame : f;
                const float vp = (i > 0) ? keys[i - 1]->camera.channelValue(ch) : v;
                const float fn = (i + 1 < count) ? (float)keys[i + 1]->frame : f;
                const float vn = (i + 1 < count) ? keys[i + 1]->camera.channelValue(ch) : v;

                float slope = 0.0f;
                if (i > 0 && i + 1 < count && fn > fp) {
                    const bool is_max = v >= vp && v >= vn;
                    const bool is_min = v <= vp && v <= vn;
                    if (!is_max && !is_min) slope = (vn - vp) / (fn - fp);
                }

                const float in_len  = (i > 0) ? (f - fp) / 3.0f : 0.0f;
                const float out_len = (i + 1 < count) ? (fn - f) / 3.0f : 0.0f;
                m.in_dx  = -in_len;  m.in_dy  = -slope * in_len;
                m.out_dx =  out_len; m.out_dy =  slope * out_len;

                if (i > 0) {
                    const float lo = std::min(vp, v), hi = std::max(vp, v);
                    m.in_dy = std::clamp(v + m.in_dy, lo, hi) - v;
                }
                if (i + 1 < count) {
                    const float lo = std::min(v, vn), hi = std::max(v, vn);
                    m.out_dy = std::clamp(v + m.out_dy, lo, hi) - v;
                }
            }
        }

        // Material
        for (int ch = 0; ch < CURVE_MAT_CHANNEL_COUNT; ++ch) {
            keys.clear();
            for (auto& kf : keyframes)
                if (kf.has_material && kf.material.channelKeyed(ch)) keys.push_back(&kf);
            const int count = (int)keys.size();
            for (int i = 0; i < count; ++i) {
                ChannelKeyMeta& m = keys[i]->material.curve[ch];
                if (!m.auto_tangent) continue;
                const bool touches_bezier =
                    m.interp == KeyInterp::Bezier ||
                    (i > 0 && keys[i - 1]->material.curve[ch].interp == KeyInterp::Bezier);
                if (!touches_bezier) continue;

                const float f  = (float)keys[i]->frame;
                const float v  = keys[i]->material.channelValue(ch);
                const float fp = (i > 0) ? (float)keys[i - 1]->frame : f;
                const float vp = (i > 0) ? keys[i - 1]->material.channelValue(ch) : v;
                const float fn = (i + 1 < count) ? (float)keys[i + 1]->frame : f;
                const float vn = (i + 1 < count) ? keys[i + 1]->material.channelValue(ch) : v;

                float slope = 0.0f;
                if (i > 0 && i + 1 < count && fn > fp) {
                    const bool is_max = v >= vp && v >= vn;
                    const bool is_min = v <= vp && v <= vn;
                    if (!is_max && !is_min) slope = (vn - vp) / (fn - fp);
                }

                const float in_len  = (i > 0) ? (f - fp) / 3.0f : 0.0f;
                const float out_len = (i + 1 < count) ? (fn - f) / 3.0f : 0.0f;
                m.in_dx  = -in_len;  m.in_dy  = -slope * in_len;
                m.out_dx =  out_len; m.out_dy =  slope * out_len;

                if (i > 0) {
                    const float lo = std::min(vp, v), hi = std::max(vp, v);
                    m.in_dy = std::clamp(v + m.in_dy, lo, hi) - v;
                }
                if (i + 1 < count) {
                    const float lo = std::min(v, vn), hi = std::max(v, vn);
                    m.out_dy = std::clamp(v + m.out_dy, lo, hi) - v;
                }
            }
        }
    }
    
    // Get keyframe at exact frame (returns nullptr if not found)
    Keyframe* getKeyframeAt(int frame) {
        auto it = std::find_if(keyframes.begin(), keyframes.end(),
            [frame](const Keyframe& kf) { return kf.frame == frame; });
        return (it != keyframes.end()) ? &(*it) : nullptr;
    }

    const Keyframe* getKeyframeAt(int frame) const {
        auto it = std::find_if(keyframes.begin(), keyframes.end(),
            [frame](const Keyframe& kf) { return kf.frame == frame; });
        return (it != keyframes.end()) ? &(*it) : nullptr;
    }
    
    // Evaluate animation at given frame (with INDEPENDENT CHANNEL INTERPOLATION)
    Keyframe evaluate(int current_frame) const {
        Keyframe result(current_frame);
        if (keyframes.empty()) return result;

        // Start from an unkeyed transform state. TransformKeyframe defaults to all
        // channels enabled for authoring convenience, but evaluation must rebuild
        // keyed channels explicitly; otherwise material-only keys look like identity
        // transform keys and raster playback resets object orientation.
        result.transform.clearAllChannels();

        // Lambda to find previous valid keyframe matching a predicate
        auto findPrev = [&](auto predicate) -> const Keyframe* {
            for (auto it = keyframes.rbegin(); it != keyframes.rend(); ++it) {
                if (it->frame <= current_frame) {
                    if (predicate(*it)) return &(*it);
                }
            }
            return nullptr;
        };

        // Lambda to find next valid keyframe matching a predicate 
        auto findNext = [&](auto predicate) -> const Keyframe* {
             for (auto it = keyframes.begin(); it != keyframes.end(); ++it) {
                if (it->frame >= current_frame) {
                    if (predicate(*it)) return &(*it);
                }
            }
            return nullptr;
        };
        
        // --- TRANSFORM CHANNELS (per-axis, curve-aware) ---
        // Each axis interpolates independently between its own neighbouring keys.
        // The segment shape (Constant/Linear/Bezier + handles) lives on the keys'
        // ChannelKeyMeta; a Linear meta reproduces the old lerp exactly.
        for (int ch = 0; ch < CURVE_CHANNEL_COUNT; ++ch) {
            auto has = [ch](const Keyframe& k) { return k.has_transform && k.transform.channelKeyed(ch); };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                float v;
                if (p == n) {
                    v = p->transform.channelValue(ch);
                } else {
                    v = evalCurveSegment(
                        (float)p->frame, p->transform.channelValue(ch), p->transform.curve[ch],
                        (float)n->frame, n->transform.channelValue(ch), n->transform.curve[ch],
                        (float)current_frame);
                }
                result.transform.setChannelValue(ch, v);
                result.transform.setChannelKeyed(ch, true);
            } else if (p) {
                result.transform.setChannelValue(ch, p->transform.channelValue(ch));
                result.transform.setChannelKeyed(ch, true);
            } else if (n) {
                result.transform.setChannelValue(ch, n->transform.channelValue(ch));
                result.transform.setChannelKeyed(ch, true);
            }
        }
        result.transform.refreshCompoundFlags();
        result.has_transform = result.transform.has_position || result.transform.has_rotation || result.transform.has_scale;
        
        // --- MATERIAL CHANNELS (per-axis, curve-aware) ---
        result.material.clearAllChannels();
        result.material.material_id = 0;
        bool has_any_mat = false;
        for (int ch = 0; ch < CURVE_MAT_CHANNEL_COUNT; ++ch) {
            auto has = [ch](const Keyframe& k) { return k.has_material && k.material.channelKeyed(ch); };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                float v;
                if (p == n) {
                    v = p->material.channelValue(ch);
                } else {
                    v = evalCurveSegment(
                        (float)p->frame, p->material.channelValue(ch), p->material.curve[ch],
                        (float)n->frame, n->material.channelValue(ch), n->material.curve[ch],
                        (float)current_frame);
                }
                result.material.setChannelValue(ch, v);
                result.material.setChannelKeyed(ch, true);
                result.material.material_id = p->material.material_id;
                has_any_mat = true;
            } else if (p) {
                result.material.setChannelValue(ch, p->material.channelValue(ch));
                result.material.setChannelKeyed(ch, true);
                result.material.material_id = p->material.material_id;
                has_any_mat = true;
            } else if (n) {
                result.material.setChannelValue(ch, n->material.channelValue(ch));
                result.material.setChannelKeyed(ch, true);
                result.material.material_id = n->material.material_id;
                has_any_mat = true;
            }
        }
        result.material.refreshCompoundFlags();
        result.has_material = has_any_mat;

        // --- LIGHT CHANNELS (per-axis, curve-aware) ---
        result.light.clearAllChannels();
        bool has_any_light = false;
        for (int ch = 0; ch < CURVE_LIGHT_CHANNEL_COUNT; ++ch) {
            auto has = [ch](const Keyframe& k) { return k.has_light && k.light.channelKeyed(ch); };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                float v;
                if (p == n) {
                    v = p->light.channelValue(ch);
                } else {
                    v = evalCurveSegment(
                        (float)p->frame, p->light.channelValue(ch), p->light.curve[ch],
                        (float)n->frame, n->light.channelValue(ch), n->light.curve[ch],
                        (float)current_frame);
                }
                result.light.setChannelValue(ch, v);
                result.light.setChannelKeyed(ch, true);
                has_any_light = true;
            } else if (p) {
                result.light.setChannelValue(ch, p->light.channelValue(ch));
                result.light.setChannelKeyed(ch, true);
                has_any_light = true;
            } else if (n) {
                result.light.setChannelValue(ch, n->light.channelValue(ch));
                result.light.setChannelKeyed(ch, true);
                has_any_light = true;
            }
        }
        result.light.refreshCompoundFlags();
        result.has_light = has_any_light;

        // --- CAMERA CHANNELS (per-axis, curve-aware) ---
        result.camera.clearAllChannels();
        bool has_any_cam = false;
        for (int ch = 0; ch < CURVE_CAM_CHANNEL_COUNT; ++ch) {
            auto has = [ch](const Keyframe& k) { return k.has_camera && k.camera.channelKeyed(ch); };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                float v;
                if (p == n) {
                    v = p->camera.channelValue(ch);
                } else {
                    v = evalCurveSegment(
                        (float)p->frame, p->camera.channelValue(ch), p->camera.curve[ch],
                        (float)n->frame, n->camera.channelValue(ch), n->camera.curve[ch],
                        (float)current_frame);
                }
                result.camera.setChannelValue(ch, v);
                result.camera.setChannelKeyed(ch, true);
                has_any_cam = true;
            } else if (p) {
                result.camera.setChannelValue(ch, p->camera.channelValue(ch));
                result.camera.setChannelKeyed(ch, true);
                has_any_cam = true;
            } else if (n) {
                result.camera.setChannelValue(ch, n->camera.channelValue(ch));
                result.camera.setChannelKeyed(ch, true);
                has_any_cam = true;
            }
        }
        result.camera.refreshCompoundFlags();
        result.has_camera = has_any_cam;

        // --- WORLD CHANNEL ---
        {
            auto interpolateT = [&](const Keyframe* p, const Keyframe* n) {
                if (!p || !n || p == n) return 0.0f;
                const float range = float(n->frame - p->frame);
                return (range > 0.0f) ? float(current_frame - p->frame) / range : 0.0f;
            };

            auto evalWorldFloat = [&](bool WorldKeyframe::*flag, float WorldKeyframe::*value) {
                auto has = [&](const Keyframe& k) { return k.has_world && (k.world.*flag); };
                const Keyframe* p = findPrev(has);
                const Keyframe* n = findNext(has);
                if (p && n) {
                    const float t = interpolateT(p, n);
                    result.world.*value = (p->world.*value) + ((n->world.*value) - (p->world.*value)) * t;
                    result.world.*flag = true;
                    result.has_world = true;
                } else if (p) {
                    result.world.*value = p->world.*value;
                    result.world.*flag = true;
                    result.has_world = true;
                } else if (n && n->frame == current_frame) {
                    result.world.*value = n->world.*value;
                    result.world.*flag = true;
                    result.has_world = true;
                }
            };

            auto evalWorldInt = [&](bool WorldKeyframe::*flag, int WorldKeyframe::*value) {
                auto has = [&](const Keyframe& k) { return k.has_world && (k.world.*flag); };
                const Keyframe* p = findPrev(has);
                const Keyframe* n = findNext(has);
                if (p && n) {
                    const float t = interpolateT(p, n);
                    result.world.*value = (t < 0.5f) ? (p->world.*value) : (n->world.*value);
                    result.world.*flag = true;
                    result.has_world = true;
                } else if (p) {
                    result.world.*value = p->world.*value;
                    result.world.*flag = true;
                    result.has_world = true;
                } else if (n && n->frame == current_frame) {
                    result.world.*value = n->world.*value;
                    result.world.*flag = true;
                    result.has_world = true;
                }
            };

            auto evalWorldVec3 = [&](bool WorldKeyframe::*flag, Vec3 WorldKeyframe::*value) {
                auto has = [&](const Keyframe& k) { return k.has_world && (k.world.*flag); };
                const Keyframe* p = findPrev(has);
                const Keyframe* n = findNext(has);
                if (p && n) {
                    const float t = interpolateT(p, n);
                    result.world.*value = (p->world.*value) + ((n->world.*value) - (p->world.*value)) * t;
                    result.world.*flag = true;
                    result.has_world = true;
                } else if (p) {
                    result.world.*value = p->world.*value;
                    result.world.*flag = true;
                    result.has_world = true;
                } else if (n && n->frame == current_frame) {
                    result.world.*value = n->world.*value;
                    result.world.*flag = true;
                    result.has_world = true;
                }
            };

            evalWorldVec3(&WorldKeyframe::has_background_color, &WorldKeyframe::background_color);
            evalWorldFloat(&WorldKeyframe::has_background_strength, &WorldKeyframe::background_strength);
            evalWorldFloat(&WorldKeyframe::has_hdri_rotation, &WorldKeyframe::hdri_rotation);

            evalWorldFloat(&WorldKeyframe::has_sun_elevation, &WorldKeyframe::sun_elevation);
            evalWorldFloat(&WorldKeyframe::has_sun_azimuth, &WorldKeyframe::sun_azimuth);
            evalWorldFloat(&WorldKeyframe::has_sun_intensity, &WorldKeyframe::sun_intensity);
            evalWorldFloat(&WorldKeyframe::has_sun_size, &WorldKeyframe::sun_size);

            evalWorldFloat(&WorldKeyframe::has_atmosphere_intensity, &WorldKeyframe::atmosphere_intensity);
            evalWorldFloat(&WorldKeyframe::has_air_density, &WorldKeyframe::air_density);
            evalWorldFloat(&WorldKeyframe::has_dust_density, &WorldKeyframe::dust_density);
            evalWorldFloat(&WorldKeyframe::has_ozone_density, &WorldKeyframe::ozone_density);
            evalWorldFloat(&WorldKeyframe::has_humidity, &WorldKeyframe::humidity);
            evalWorldFloat(&WorldKeyframe::has_temperature, &WorldKeyframe::temperature);
            evalWorldFloat(&WorldKeyframe::has_ozone_absorption_scale, &WorldKeyframe::ozone_absorption_scale);
            evalWorldFloat(&WorldKeyframe::has_altitude, &WorldKeyframe::altitude);
            evalWorldFloat(&WorldKeyframe::has_mie_anisotropy, &WorldKeyframe::mie_anisotropy);

            evalWorldFloat(&WorldKeyframe::has_cloud_density, &WorldKeyframe::cloud_density);
            evalWorldFloat(&WorldKeyframe::has_cloud_coverage, &WorldKeyframe::cloud_coverage);
            evalWorldFloat(&WorldKeyframe::has_cloud_scale, &WorldKeyframe::cloud_scale);

            {
                auto has = [](const Keyframe& k) { return k.has_world && k.world.has_cloud_offset; };
                const Keyframe* p = findPrev(has);
                const Keyframe* n = findNext(has);
                if (p && n) {
                    const float t = interpolateT(p, n);
                    result.world.cloud_offset_x = p->world.cloud_offset_x + (n->world.cloud_offset_x - p->world.cloud_offset_x) * t;
                    result.world.cloud_offset_z = p->world.cloud_offset_z + (n->world.cloud_offset_z - p->world.cloud_offset_z) * t;
                    result.world.has_cloud_offset = true;
                    result.has_world = true;
                } else if (p) {
                    result.world.cloud_offset_x = p->world.cloud_offset_x;
                    result.world.cloud_offset_z = p->world.cloud_offset_z;
                    result.world.has_cloud_offset = true;
                    result.has_world = true;
                } else if (n && n->frame == current_frame) {
                    result.world.cloud_offset_x = n->world.cloud_offset_x;
                    result.world.cloud_offset_z = n->world.cloud_offset_z;
                    result.world.has_cloud_offset = true;
                    result.has_world = true;
                }
            }

            evalWorldFloat(&WorldKeyframe::has_cloud_quality, &WorldKeyframe::cloud_quality);
            evalWorldInt(&WorldKeyframe::has_cloud_quality, &WorldKeyframe::cloud_base_steps);
            evalWorldFloat(&WorldKeyframe::has_cloud_detail, &WorldKeyframe::cloud_detail);
            evalWorldInt(&WorldKeyframe::has_cloud_layer2, &WorldKeyframe::cloud_layer2_enabled);

            {
                auto has = [](const Keyframe& k) { return k.has_world && k.world.has_cloud_lighting; };
                const Keyframe* p = findPrev(has);
                const Keyframe* n = findNext(has);
                if (p && n) {
                    const float t = interpolateT(p, n);
                    result.world.cloud_light_steps = (t < 0.5f) ? p->world.cloud_light_steps : n->world.cloud_light_steps;
                    result.world.cloud_shadow_strength = p->world.cloud_shadow_strength + (n->world.cloud_shadow_strength - p->world.cloud_shadow_strength) * t;
                    result.world.cloud_ambient_strength = p->world.cloud_ambient_strength + (n->world.cloud_ambient_strength - p->world.cloud_ambient_strength) * t;
                    result.world.cloud_silver_intensity = p->world.cloud_silver_intensity + (n->world.cloud_silver_intensity - p->world.cloud_silver_intensity) * t;
                    result.world.cloud_absorption = p->world.cloud_absorption + (n->world.cloud_absorption - p->world.cloud_absorption) * t;
                    result.world.cloud_anisotropy = p->world.cloud_anisotropy + (n->world.cloud_anisotropy - p->world.cloud_anisotropy) * t;
                    result.world.cloud_anisotropy_back = p->world.cloud_anisotropy_back + (n->world.cloud_anisotropy_back - p->world.cloud_anisotropy_back) * t;
                    result.world.cloud_lobe_mix = p->world.cloud_lobe_mix + (n->world.cloud_lobe_mix - p->world.cloud_lobe_mix) * t;
                    result.world.cloud_emissive_intensity = p->world.cloud_emissive_intensity + (n->world.cloud_emissive_intensity - p->world.cloud_emissive_intensity) * t;
                    result.world.cloud_emissive_color = p->world.cloud_emissive_color + (n->world.cloud_emissive_color - p->world.cloud_emissive_color) * t;
                    result.world.has_cloud_lighting = true;
                    result.has_world = true;
                } else if (p) {
                    result.world.cloud_light_steps = p->world.cloud_light_steps;
                    result.world.cloud_shadow_strength = p->world.cloud_shadow_strength;
                    result.world.cloud_ambient_strength = p->world.cloud_ambient_strength;
                    result.world.cloud_silver_intensity = p->world.cloud_silver_intensity;
                    result.world.cloud_absorption = p->world.cloud_absorption;
                    result.world.cloud_anisotropy = p->world.cloud_anisotropy;
                    result.world.cloud_anisotropy_back = p->world.cloud_anisotropy_back;
                    result.world.cloud_lobe_mix = p->world.cloud_lobe_mix;
                    result.world.cloud_emissive_intensity = p->world.cloud_emissive_intensity;
                    result.world.cloud_emissive_color = p->world.cloud_emissive_color;
                    result.world.has_cloud_lighting = true;
                    result.has_world = true;
                } else if (n && n->frame == current_frame) {
                    result.world.cloud_light_steps = n->world.cloud_light_steps;
                    result.world.cloud_shadow_strength = n->world.cloud_shadow_strength;
                    result.world.cloud_ambient_strength = n->world.cloud_ambient_strength;
                    result.world.cloud_silver_intensity = n->world.cloud_silver_intensity;
                    result.world.cloud_absorption = n->world.cloud_absorption;
                    result.world.cloud_anisotropy = n->world.cloud_anisotropy;
                    result.world.cloud_anisotropy_back = n->world.cloud_anisotropy_back;
                    result.world.cloud_lobe_mix = n->world.cloud_lobe_mix;
                    result.world.cloud_emissive_intensity = n->world.cloud_emissive_intensity;
                    result.world.cloud_emissive_color = n->world.cloud_emissive_color;
                    result.world.has_cloud_lighting = true;
                    result.has_world = true;
                }
            }

            {
                auto has = [](const Keyframe& k) { return k.has_world && k.world.has_cloud_layer2_params; };
                const Keyframe* p = findPrev(has);
                const Keyframe* n = findNext(has);
                if (p && n) {
                    const float t = interpolateT(p, n);
                    result.world.cloud2_coverage = p->world.cloud2_coverage + (n->world.cloud2_coverage - p->world.cloud2_coverage) * t;
                    result.world.cloud2_density = p->world.cloud2_density + (n->world.cloud2_density - p->world.cloud2_density) * t;
                    result.world.cloud2_scale = p->world.cloud2_scale + (n->world.cloud2_scale - p->world.cloud2_scale) * t;
                    result.world.has_cloud_layer2_params = true;
                    result.has_world = true;
                } else if (p) {
                    result.world.cloud2_coverage = p->world.cloud2_coverage;
                    result.world.cloud2_density = p->world.cloud2_density;
                    result.world.cloud2_scale = p->world.cloud2_scale;
                    result.world.has_cloud_layer2_params = true;
                    result.has_world = true;
                } else if (n && n->frame == current_frame) {
                    result.world.cloud2_coverage = n->world.cloud2_coverage;
                    result.world.cloud2_density = n->world.cloud2_density;
                    result.world.cloud2_scale = n->world.cloud2_scale;
                    result.world.has_cloud_layer2_params = true;
                    result.has_world = true;
                }
            }

            {
                auto has = [](const Keyframe& k) { return k.has_world && k.world.has_cloud_layer2_heights; };
                const Keyframe* p = findPrev(has);
                const Keyframe* n = findNext(has);
                if (p && n) {
                    const float t = interpolateT(p, n);
                    result.world.cloud2_height_min = p->world.cloud2_height_min + (n->world.cloud2_height_min - p->world.cloud2_height_min) * t;
                    result.world.cloud2_height_max = p->world.cloud2_height_max + (n->world.cloud2_height_max - p->world.cloud2_height_max) * t;
                    result.world.has_cloud_layer2_heights = true;
                    result.has_world = true;
                } else if (p) {
                    result.world.cloud2_height_min = p->world.cloud2_height_min;
                    result.world.cloud2_height_max = p->world.cloud2_height_max;
                    result.world.has_cloud_layer2_heights = true;
                    result.has_world = true;
                } else if (n && n->frame == current_frame) {
                    result.world.cloud2_height_min = n->world.cloud2_height_min;
                    result.world.cloud2_height_max = n->world.cloud2_height_max;
                    result.world.has_cloud_layer2_heights = true;
                    result.has_world = true;
                }
            }

            evalWorldInt(&WorldKeyframe::has_fog, &WorldKeyframe::fog_enabled);
            {
                auto has = [](const Keyframe& k) { return k.has_world && k.world.has_fog_params; };
                const Keyframe* p = findPrev(has);
                const Keyframe* n = findNext(has);
                if (p && n) {
                    const float t = interpolateT(p, n);
                    result.world.fog_density = p->world.fog_density + (n->world.fog_density - p->world.fog_density) * t;
                    result.world.fog_height = p->world.fog_height + (n->world.fog_height - p->world.fog_height) * t;
                    result.world.fog_falloff = p->world.fog_falloff + (n->world.fog_falloff - p->world.fog_falloff) * t;
                    result.world.fog_distance = p->world.fog_distance + (n->world.fog_distance - p->world.fog_distance) * t;
                    result.world.fog_color = p->world.fog_color + (n->world.fog_color - p->world.fog_color) * t;
                    result.world.fog_sun_scatter = p->world.fog_sun_scatter + (n->world.fog_sun_scatter - p->world.fog_sun_scatter) * t;
                    result.world.has_fog_params = true;
                    result.has_world = true;
                } else if (p) {
                    result.world.fog_density = p->world.fog_density;
                    result.world.fog_height = p->world.fog_height;
                    result.world.fog_falloff = p->world.fog_falloff;
                    result.world.fog_distance = p->world.fog_distance;
                    result.world.fog_color = p->world.fog_color;
                    result.world.fog_sun_scatter = p->world.fog_sun_scatter;
                    result.world.has_fog_params = true;
                    result.has_world = true;
                } else if (n && n->frame == current_frame) {
                    result.world.fog_density = n->world.fog_density;
                    result.world.fog_height = n->world.fog_height;
                    result.world.fog_falloff = n->world.fog_falloff;
                    result.world.fog_distance = n->world.fog_distance;
                    result.world.fog_color = n->world.fog_color;
                    result.world.fog_sun_scatter = n->world.fog_sun_scatter;
                    result.world.has_fog_params = true;
                    result.has_world = true;
                }
            }

            evalWorldInt(&WorldKeyframe::has_godrays, &WorldKeyframe::godrays_enabled);
            evalWorldFloat(&WorldKeyframe::has_godrays_params, &WorldKeyframe::godrays_intensity);
            evalWorldFloat(&WorldKeyframe::has_godrays_params, &WorldKeyframe::godrays_density);
            evalWorldInt(&WorldKeyframe::has_godrays_params, &WorldKeyframe::godrays_samples);

            evalWorldInt(&WorldKeyframe::has_multi_scatter, &WorldKeyframe::multi_scatter_enabled);
            evalWorldFloat(&WorldKeyframe::has_multi_scatter, &WorldKeyframe::multi_scatter_factor);
            evalWorldInt(&WorldKeyframe::has_aerial_perspective, &WorldKeyframe::aerial_perspective);
            evalWorldFloat(&WorldKeyframe::has_aerial_params, &WorldKeyframe::aerial_density);
            evalWorldFloat(&WorldKeyframe::has_aerial_params, &WorldKeyframe::aerial_min_distance);
            evalWorldFloat(&WorldKeyframe::has_aerial_params, &WorldKeyframe::aerial_max_distance);
            evalWorldInt(&WorldKeyframe::has_overlay, &WorldKeyframe::env_overlay_enabled);
            evalWorldFloat(&WorldKeyframe::has_overlay_params, &WorldKeyframe::env_overlay_intensity);
            evalWorldFloat(&WorldKeyframe::has_overlay_params, &WorldKeyframe::env_overlay_rotation);
            evalWorldInt(&WorldKeyframe::has_overlay_params, &WorldKeyframe::env_overlay_blend_mode);

            {
                auto has = [](const Keyframe& k) { return k.has_world && k.world.has_weather_params; };
                const Keyframe* p = findPrev(has);
                const Keyframe* n = findNext(has);
                if (p && n) {
                    const float t = interpolateT(p, n);
                    result.world.weather_enabled = (t < 0.5f) ? p->world.weather_enabled : n->world.weather_enabled;
                    result.world.weather_type = (t < 0.5f) ? p->world.weather_type : n->world.weather_type;
                    result.world.weather_intensity = p->world.weather_intensity + (n->world.weather_intensity - p->world.weather_intensity) * t;
                    result.world.weather_density = p->world.weather_density + (n->world.weather_density - p->world.weather_density) * t;
                    result.world.weather_wind_direction = p->world.weather_wind_direction + (n->world.weather_wind_direction - p->world.weather_wind_direction) * t;
                    result.world.weather_wind_speed = p->world.weather_wind_speed + (n->world.weather_wind_speed - p->world.weather_wind_speed) * t;
                    result.world.weather_precipitation_scale = p->world.weather_precipitation_scale + (n->world.weather_precipitation_scale - p->world.weather_precipitation_scale) * t;
                    result.world.weather_visibility = p->world.weather_visibility + (n->world.weather_visibility - p->world.weather_visibility) * t;
                    result.world.weather_surface_wetness = p->world.weather_surface_wetness + (n->world.weather_surface_wetness - p->world.weather_surface_wetness) * t;
                    result.world.weather_surface_accumulation = p->world.weather_surface_accumulation + (n->world.weather_surface_accumulation - p->world.weather_surface_accumulation) * t;
                    result.world.weather_surface_settling = p->world.weather_surface_settling + (n->world.weather_surface_settling - p->world.weather_surface_settling) * t;
                    result.world.weather_surface_height = p->world.weather_surface_height + (n->world.weather_surface_height - p->world.weather_surface_height) * t;
                    result.world.weather_visual_mode = (t < 0.5f) ? p->world.weather_visual_mode : n->world.weather_visual_mode;
                    result.world.weather_surface_response_enabled = (t < 0.5f) ? p->world.weather_surface_response_enabled : n->world.weather_surface_response_enabled;
                    result.world.has_weather_params = true;
                    result.has_world = true;
                } else if (p) {
                    result.world.weather_enabled = p->world.weather_enabled;
                    result.world.weather_type = p->world.weather_type;
                    result.world.weather_intensity = p->world.weather_intensity;
                    result.world.weather_density = p->world.weather_density;
                    result.world.weather_wind_direction = p->world.weather_wind_direction;
                    result.world.weather_wind_speed = p->world.weather_wind_speed;
                    result.world.weather_precipitation_scale = p->world.weather_precipitation_scale;
                    result.world.weather_visibility = p->world.weather_visibility;
                    result.world.weather_surface_wetness = p->world.weather_surface_wetness;
                    result.world.weather_surface_accumulation = p->world.weather_surface_accumulation;
                    result.world.weather_surface_settling = p->world.weather_surface_settling;
                    result.world.weather_surface_height = p->world.weather_surface_height;
                    result.world.weather_visual_mode = p->world.weather_visual_mode;
                    result.world.weather_surface_response_enabled = p->world.weather_surface_response_enabled;
                    result.world.has_weather_params = true;
                    result.has_world = true;
                } else if (n && n->frame == current_frame) {
                    result.world.weather_enabled = n->world.weather_enabled;
                    result.world.weather_type = n->world.weather_type;
                    result.world.weather_intensity = n->world.weather_intensity;
                    result.world.weather_density = n->world.weather_density;
                    result.world.weather_wind_direction = n->world.weather_wind_direction;
                    result.world.weather_wind_speed = n->world.weather_wind_speed;
                    result.world.weather_precipitation_scale = n->world.weather_precipitation_scale;
                    result.world.weather_visibility = n->world.weather_visibility;
                    result.world.weather_surface_wetness = n->world.weather_surface_wetness;
                    result.world.weather_surface_accumulation = n->world.weather_surface_accumulation;
                    result.world.weather_surface_settling = n->world.weather_surface_settling;
                    result.world.weather_surface_height = n->world.weather_surface_height;
                    result.world.weather_visual_mode = n->world.weather_visual_mode;
                    result.world.weather_surface_response_enabled = n->world.weather_surface_response_enabled;
                    result.world.has_weather_params = true;
                    result.has_world = true;
                }
            }
        }

        // --- TERRAIN ---
        const Keyframe* p_terrain = findPrev([](const auto& k){ return k.has_terrain; });
        const Keyframe* n_terrain = findNext([](const auto& k){ return k.has_terrain; });
        
        if (p_terrain && n_terrain) {
            float range = (float)(n_terrain->frame - p_terrain->frame);
            float t = (range > 0) ? (float)(current_frame - p_terrain->frame) / range : 0.0f;
            result.terrain = TerrainKeyframe::lerp(p_terrain->terrain, n_terrain->terrain, t);
            result.has_terrain = true;
        } else if (p_terrain) {
            result.terrain = p_terrain->terrain;
            result.has_terrain = true;
        } else if (n_terrain) {
            result.terrain = n_terrain->terrain;
            result.has_terrain = true;
        }
        
        // --- WATER ---
        const Keyframe* p_water = findPrev([](const auto& k){ return k.has_water; });
        const Keyframe* n_water = findNext([](const auto& k){ return k.has_water; });
        
        if (p_water && n_water) {
            float range = (float)(n_water->frame - p_water->frame);
            float t = (range > 0) ? (float)(current_frame - p_water->frame) / range : 0.0f;
            result.water = WaterKeyframe::lerp(p_water->water, n_water->water, t);
            result.has_water = true;
        } else if (p_water) {
            result.water = p_water->water;
            result.has_water = true;
        } else if (n_water) {
            result.water = n_water->water;
            result.has_water = true;
        }

        // --- ANIM GRAPH ---
        const Keyframe* exact_anim = getKeyframeAt(current_frame);
        if (exact_anim && exact_anim->has_anim_graph && !exact_anim->anim_graph.triggers.empty()) {
            result.anim_graph.triggers = exact_anim->anim_graph.triggers;
        }

        const Keyframe* p_anim = findPrev([](const auto& k){ return k.has_anim_graph; });
        const Keyframe* n_anim = findNext([](const auto& k){ return k.has_anim_graph; });
        if (p_anim && n_anim) {
            float range = (float)(n_anim->frame - p_anim->frame);
            float t = (range > 0) ? (float)(current_frame - p_anim->frame) / range : 0.0f;
            result.anim_graph = AnimGraphKeyframe::blend(p_anim->anim_graph, n_anim->anim_graph, t);
            if (exact_anim && exact_anim->has_anim_graph && !exact_anim->anim_graph.triggers.empty()) {
                result.anim_graph.triggers = exact_anim->anim_graph.triggers;
            }
            result.has_anim_graph = true;
        } else if (p_anim) {
            result.anim_graph = p_anim->anim_graph;
            if (exact_anim && exact_anim->has_anim_graph && !exact_anim->anim_graph.triggers.empty()) {
                result.anim_graph.triggers = exact_anim->anim_graph.triggers;
            } else {
                result.anim_graph.triggers.clear();
            }
            result.has_anim_graph = true;
        } else if (n_anim) {
            result.anim_graph = n_anim->anim_graph;
            if (exact_anim && exact_anim->has_anim_graph && !exact_anim->anim_graph.triggers.empty()) {
                result.anim_graph.triggers = exact_anim->anim_graph.triggers;
            } else {
                result.anim_graph.triggers.clear();
            }
            result.has_anim_graph = true;
        }
        
        // --- EMITTER ---
        const Keyframe* p_emitter = findPrev([](const auto& k){ return k.has_emitter; });
        const Keyframe* n_emitter = findNext([](const auto& k){ return k.has_emitter; });
        
        if (p_emitter && n_emitter) {
            float range = (float)(n_emitter->frame - p_emitter->frame);
            float t = (range > 0) ? (float)(current_frame - p_emitter->frame) / range : 0.0f;
            result.emitter = EmitterKeyframe::lerp(p_emitter->emitter, n_emitter->emitter, t);
            result.has_emitter = true;
        } else if (p_emitter) {
            result.emitter = p_emitter->emitter;
            result.has_emitter = true;
        } else if (n_emitter) {
            result.emitter = n_emitter->emitter;
            result.has_emitter = true;
        }
        
        return result;
    }
};

// Timeline Manager - manages all object animation tracks
struct TimelineManager {
    std::map<std::string, ObjectAnimationTrack> tracks;  // object_name -> track
    int current_frame = 0;
    
    // Get or create track for object
    ObjectAnimationTrack& getTrack(const std::string& object_name) {
        return tracks[object_name];
    }
    
    // Insert keyframe for object
    void insertKeyframe(const std::string& object_name, const Keyframe& kf) {
        auto& track = tracks[object_name];
        track.addKeyframe(kf);
        // Single-key authoring path: keep neighbouring auto handles in sync.
        // (Bulk imports go through ObjectAnimationTrack::addKeyframe directly and
        // skip this; their zero-length handles evaluate as linear, which is correct
        // for densely baked data.)
        track.refreshAutoTangents();
    }

    // Remove keyframe for object at frame
    void removeKeyframe(const std::string& object_name, int frame) {
        auto it = tracks.find(object_name);
        if (it != tracks.end()) {
            it->second.removeKeyframe(frame);
            it->second.refreshAutoTangents();
        }
    }
    
    // Get all keyframes at a specific frame (across all objects)
    std::vector<std::pair<std::string, Keyframe*>> getKeyframesAtFrame(int frame) {
        std::vector<std::pair<std::string, Keyframe*>> result;
        for (auto& [name, track] : tracks) {
            if (Keyframe* kf = track.getKeyframeAt(frame)) {
                result.push_back({name, kf});
            }
        }
        return result;
    }
    
    // Clear all keyframes
    void clear() {
        tracks.clear();
        current_frame = 0;
    }
    
    // Serialization
    void serialize(json& j) const;
    void deserialize(const json& j);
};

// ============================================================================
// JSON SERIALIZATION IMPLEMENTATION
// ============================================================================

// MaterialKeyframe
inline void to_json(json& j, const MaterialKeyframe& m) {
    j = json{
        {"id", m.material_id},
        {"falb", m.has_albedo}, {"fopa", m.has_opacity}, {"frgh", m.has_roughness},
        {"fmet", m.has_metallic}, {"fems", m.has_emission}, {"ftrn", m.has_transmission},
        {"ior", m.has_ior}, {"fclr", m.has_clearcoat}, {"fsub", m.has_subsurface},
        {"fshe", m.has_sheen}, {"fani", m.has_anisotropic}, {"fspc", m.has_specular},
        {"fnrm", m.has_normal},
        {"far", m.has_alb_r}, {"fag", m.has_alb_g}, {"fab", m.has_alb_b},
        {"fop", m.has_opac},
        {"fro", m.has_rough},
        {"fme", m.has_metal},
        {"fcl", m.has_clear},
        {"ftr", m.has_transm},
        {"fio", m.has_ior_val},
        {"fer", m.has_emis_r}, {"feg", m.has_emis_g}, {"feb", m.has_emis_b},
        {"fns", m.has_norm_str},
        {"fes", m.has_emis_str},
        {"alb", m.albedo}, {"opa", m.opacity}, {"rgh", m.roughness},
        {"met", m.metallic}, {"clr", m.clearcoat}, {"trn", m.transmission},
        {"ems", m.emission}, {"v_ior", m.ior}, 
        {"sub_c", m.subsurface_color}, {"sub", m.subsurface},
        {"ani", m.anisotropic}, {"she", m.sheen}, {"she_t", m.sheen_tint},
        {"spc", m.specular}, {"spc_t", m.specular_tint},
        {"clr_r", m.clearcoat_roughness}, {"nrm_s", m.normal_strength},
        {"ems_s", m.emission_strength}
    };
    json crv = json::array();
    for (int ch = 0; ch < CURVE_MAT_CHANNEL_COUNT; ++ch) {
        const ChannelKeyMeta& meta = m.curve[ch];
        if (meta.isLegacyLinear()) continue;
        crv.push_back(json::array({ ch, (int)meta.interp, meta.auto_tangent ? 1 : 0,
                                    meta.in_dx, meta.in_dy, meta.out_dx, meta.out_dy }));
    }
    if (!crv.empty()) j["crv"] = crv;
}

inline void from_json(const json& j, MaterialKeyframe& m) {
    m.material_id = j.value("id", (uint16_t)0);
    m.has_albedo = j.value("falb", false); m.has_opacity = j.value("fopa", false);
    m.has_roughness = j.value("frgh", false); m.has_metallic = j.value("fmet", false);
    m.has_emission = j.value("fems", false); m.has_transmission = j.value("ftrn", false);
    m.has_ior = j.value("ior", false); m.has_clearcoat = j.value("fclr", false);
    m.has_subsurface = j.value("fsub", false); m.has_sheen = j.value("fshe", false);
    m.has_anisotropic = j.value("fani", false); m.has_specular = j.value("fspc", false);
    m.has_normal = j.value("fnrm", false);

    m.has_alb_r = j.value("far", m.has_albedo);
    m.has_alb_g = j.value("fag", m.has_albedo);
    m.has_alb_b = j.value("fab", m.has_albedo);
    m.has_opac = j.value("fop", m.has_opacity);
    m.has_rough = j.value("fro", m.has_roughness);
    m.has_metal = j.value("fme", m.has_metallic);
    m.has_clear = j.value("fcl", m.has_clearcoat);
    m.has_transm = j.value("ftr", m.has_transmission);
    m.has_ior_val = j.value("fio", m.has_ior);
    m.has_emis_r = j.value("fer", m.has_emission);
    m.has_emis_g = j.value("feg", m.has_emission);
    m.has_emis_b = j.value("feb", m.has_emission);
    m.has_norm_str = j.value("fns", m.has_normal);
    m.has_emis_str = j.value("fes", m.has_emission);
    
    if(j.contains("alb")) j.at("alb").get_to(m.albedo);
    m.opacity = j.value("opa", 1.0f);
    m.roughness = j.value("rgh", 0.5f);
    m.metallic = j.value("met", 0.0f);
    m.clearcoat = j.value("clr", 0.0f);
    m.transmission = j.value("trn", 0.0f);
    if(j.contains("ems")) j.at("ems").get_to(m.emission);
    m.ior = j.value("v_ior", 1.45f);
    if(j.contains("sub_c")) j.at("sub_c").get_to(m.subsurface_color);
    m.subsurface = j.value("sub", 0.0f);
    m.anisotropic = j.value("ani", 0.0f);
    m.sheen = j.value("she", 0.0f);
    m.sheen_tint = j.value("she_t", 0.0f);
    m.specular = j.value("spc", 0.5f);
    m.specular_tint = j.value("spc_t", 0.0f);
    m.clearcoat_roughness = j.value("clr_r", 0.0f);
    m.normal_strength = j.value("nrm_s", 1.0f);
    m.emission_strength = j.value("ems_s", 1.0f);

    for (auto& meta : m.curve) meta.resetToLegacyLinear();
    if (j.contains("crv") && j["crv"].is_array()) {
        for (const auto& e : j["crv"]) {
            if (!e.is_array() || e.size() < 7) continue;
            const int ch = e[0].get<int>();
            if (ch < 0 || ch >= CURVE_MAT_CHANNEL_COUNT) continue;
            ChannelKeyMeta& meta = m.curve[ch];
            const int interp = e[1].get<int>();
            meta.interp = (interp == (int)KeyInterp::Constant) ? KeyInterp::Constant
                        : (interp == (int)KeyInterp::Bezier)   ? KeyInterp::Bezier
                                                               : KeyInterp::Linear;
            meta.auto_tangent = e[2].get<int>() != 0;
            meta.in_dx = e[3].get<float>();  meta.in_dy = e[4].get<float>();
            meta.out_dx = e[5].get<float>(); meta.out_dy = e[6].get<float>();
        }
    }
}

// LightKeyframe
inline void to_json(json& j, const LightKeyframe& l) {
    j = json{
        {"fpos", l.has_position}, {"fcol", l.has_color},
        {"fint", l.has_intensity}, {"fdir", l.has_direction},
        {"fpx", l.has_pos_x}, {"fpy", l.has_pos_y}, {"fpz", l.has_pos_z},
        {"fcr", l.has_col_r}, {"fcg", l.has_col_g}, {"fcb", l.has_col_b},
        {"fi", l.has_int},
        {"fdx", l.has_dir_x}, {"fdy", l.has_dir_y}, {"fdz", l.has_dir_z},
        {"int", l.intensity}
    };
    // Manual Vec3 serialization to ensure stability
    j["pos"] = {l.position.x, l.position.y, l.position.z};
    j["col"] = {l.color.x, l.color.y, l.color.z};
    j["dir"] = {l.direction.x, l.direction.y, l.direction.z};

    json crv = json::array();
    for (int ch = 0; ch < CURVE_LIGHT_CHANNEL_COUNT; ++ch) {
        const ChannelKeyMeta& meta = l.curve[ch];
        if (meta.isLegacyLinear()) continue;
        crv.push_back(json::array({ ch, (int)meta.interp, meta.auto_tangent ? 1 : 0,
                                    meta.in_dx, meta.in_dy, meta.out_dx, meta.out_dy }));
    }
    if (!crv.empty()) j["crv"] = crv;
}

inline void from_json(const json& j, LightKeyframe& l) {
    // Robust loading: Check explicit flag key first, fallback to data key existence
    l.has_position = j.value("fpos", j.contains("pos")); 
    l.has_color = j.value("fcol", j.contains("col"));
    l.has_intensity = j.value("fint", j.contains("int")); 
    l.has_direction = j.value("fdir", j.contains("dir"));

    l.has_pos_x = j.value("fpx", l.has_position);
    l.has_pos_y = j.value("fpy", l.has_position);
    l.has_pos_z = j.value("fpz", l.has_position);
    l.has_col_r = j.value("fcr", l.has_color);
    l.has_col_g = j.value("fcg", l.has_color);
    l.has_col_b = j.value("fcb", l.has_color);
    l.has_int   = j.value("fi",  l.has_intensity);
    l.has_dir_x = j.value("fdx", l.has_direction);
    l.has_dir_y = j.value("fdy", l.has_direction);
    l.has_dir_z = j.value("fdz", l.has_direction);

    l.intensity = j.value("int", 1.0f);

    // Manual Vec3 extraction
    if(j.contains("pos") && j["pos"].is_array() && j["pos"].size() >= 3) {
        l.position.x = j["pos"][0]; l.position.y = j["pos"][1]; l.position.z = j["pos"][2];
    }
    if(j.contains("col") && j["col"].is_array() && j["col"].size() >= 3) {
        l.color.x = j["col"][0]; l.color.y = j["col"][1]; l.color.z = j["col"][2];
    }
    if(j.contains("dir") && j["dir"].is_array() && j["dir"].size() >= 3) {
        l.direction.x = j["dir"][0]; l.direction.y = j["dir"][1]; l.direction.z = j["dir"][2];
    }

    for (auto& meta : l.curve) meta.resetToLegacyLinear();
    if (j.contains("crv") && j["crv"].is_array()) {
        for (const auto& e : j["crv"]) {
            if (!e.is_array() || e.size() < 7) continue;
            const int ch = e[0].get<int>();
            if (ch < 0 || ch >= CURVE_LIGHT_CHANNEL_COUNT) continue;
            ChannelKeyMeta& meta = l.curve[ch];
            const int interp = e[1].get<int>();
            meta.interp = (interp == (int)KeyInterp::Constant) ? KeyInterp::Constant
                        : (interp == (int)KeyInterp::Bezier)   ? KeyInterp::Bezier
                                                               : KeyInterp::Linear;
            meta.auto_tangent = e[2].get<int>() != 0;
            meta.in_dx = e[3].get<float>();  meta.in_dy = e[4].get<float>();
            meta.out_dx = e[5].get<float>(); meta.out_dy = e[6].get<float>();
        }
    }
}

// EmitterKeyframe
inline void to_json(json& j, const EmitterKeyframe& e) {
    j = json{
        {"ffr", e.has_fuel_rate}, {"fdr", e.has_density_rate}, {"ftmp", e.has_temperature},
        {"fvel", e.has_velocity}, {"fpos", e.has_position}, {"fen", e.has_enabled},
        {"fsz", e.has_size}, {"frad", e.has_radius},
        {"fr", e.fuel_rate}, {"dr", e.density_rate}, {"tp", e.temperature},
        {"vel", e.velocity}, {"pos", e.position}, {"sz", e.size}, {"rad", e.radius}, {"en", e.enabled}
    };
}

inline void from_json(const json& j, EmitterKeyframe& e) {
    e.has_fuel_rate = j.value("ffr", false); e.has_density_rate = j.value("fdr", false);
    e.has_temperature = j.value("ftmp", false); e.has_velocity = j.value("fvel", false);
    e.has_position = j.value("fpos", false); e.has_enabled = j.value("fen", false);
    e.has_size = j.value("fsz", false); e.has_radius = j.value("frad", false);
    
    e.fuel_rate = j.value("fr", 0.0f); e.density_rate = j.value("dr", 10.0f);
    e.temperature = j.value("tp", 500.0f); e.enabled = j.value("en", true);
    e.radius = j.value("rad", 1.0f);
    if(j.contains("vel")) j.at("vel").get_to(e.velocity);
    if(j.contains("pos")) j.at("pos").get_to(e.position);
    if(j.contains("sz")) j.at("sz").get_to(e.size);
}

// CameraKeyframe
inline void to_json(json& j, const CameraKeyframe& c) {
    j = json{
        {"fpos", c.has_position}, {"ftgt", c.has_target},
        {"ffov", c.has_fov}, {"ffoc", c.has_focus}, {"fapt", c.has_aperture},
        {"fpx", c.has_pos_x}, {"fpy", c.has_pos_y}, {"fpz", c.has_pos_z},
        {"ftx", c.has_tgt_x}, {"fty", c.has_tgt_y}, {"ftz", c.has_tgt_z},
        {"ffv", c.has_fv}, {"ffd", c.has_foc_dist}, {"flr", c.has_lens_rad},
        {"pos", c.position}, {"tgt", c.target},
        {"fov", c.fov}, {"foc", c.focus_distance}, {"apt", c.lens_radius}
    };
    json crv = json::array();
    for (int ch = 0; ch < CURVE_CAM_CHANNEL_COUNT; ++ch) {
        const ChannelKeyMeta& meta = c.curve[ch];
        if (meta.isLegacyLinear()) continue;
        crv.push_back(json::array({ ch, (int)meta.interp, meta.auto_tangent ? 1 : 0,
                                    meta.in_dx, meta.in_dy, meta.out_dx, meta.out_dy }));
    }
    if (!crv.empty()) j["crv"] = crv;
}

inline void from_json(const json& j, CameraKeyframe& c) {
    // Robust loading
    c.has_position = j.value("fpos", j.contains("pos")); 
    c.has_target = j.value("ftgt", j.contains("tgt"));
    c.has_fov = j.value("ffov", j.contains("fov")); 
    c.has_focus = j.value("ffoc", j.contains("foc"));
    c.has_aperture = j.value("fapt", j.contains("apt"));

    c.has_pos_x = j.value("fpx", c.has_position);
    c.has_pos_y = j.value("fpy", c.has_position);
    c.has_pos_z = j.value("fpz", c.has_position);
    c.has_tgt_x = j.value("ftx", c.has_target);
    c.has_tgt_y = j.value("fty", c.has_target);
    c.has_tgt_z = j.value("ftz", c.has_target);
    c.has_fv = j.value("ffv", c.has_fov);
    c.has_foc_dist = j.value("ffd", c.has_focus);
    c.has_lens_rad = j.value("flr", c.has_aperture);

    // Safe extraction
    if(j.contains("pos")) j.at("pos").get_to(c.position);
    if(j.contains("tgt")) j.at("tgt").get_to(c.target);
    c.fov = j.value("fov", 40.0f);
    c.focus_distance = j.value("foc", 10.0f);
    c.lens_radius = j.value("apt", 0.0f);

    for (auto& meta : c.curve) meta.resetToLegacyLinear();
    if (j.contains("crv") && j["crv"].is_array()) {
        for (const auto& e : j["crv"]) {
            if (!e.is_array() || e.size() < 7) continue;
            const int ch = e[0].get<int>();
            if (ch < 0 || ch >= CURVE_CAM_CHANNEL_COUNT) continue;
            ChannelKeyMeta& meta = c.curve[ch];
            const int interp = e[1].get<int>();
            meta.interp = (interp == (int)KeyInterp::Constant) ? KeyInterp::Constant
                        : (interp == (int)KeyInterp::Bezier)   ? KeyInterp::Bezier
                                                               : KeyInterp::Linear;
            meta.auto_tangent = e[2].get<int>() != 0;
            meta.in_dx = e[3].get<float>();  meta.in_dy = e[4].get<float>();
            meta.out_dx = e[5].get<float>(); meta.out_dy = e[6].get<float>();
        }
    }
}

// WorldKeyframe
inline void to_json(json& j, const WorldKeyframe& w) {
    j = json{
        {"fbgc", w.has_background_color}, {"fbgs", w.has_background_strength}, {"fhr", w.has_hdri_rotation},
        {"fse", w.has_sun_elevation}, {"fsa", w.has_sun_azimuth}, {"fsi", w.has_sun_intensity}, {"fss", w.has_sun_size},
        {"fad", w.has_air_density}, {"fdd", w.has_dust_density}, {"fod", w.has_ozone_density},
        {"fhum", w.has_humidity}, {"ftmp", w.has_temperature}, {"fozs", w.has_ozone_absorption_scale},
        {"falt", w.has_altitude}, {"fma", w.has_mie_anisotropy},
        {"fcd", w.has_cloud_density}, {"fcc", w.has_cloud_coverage}, {"fcs", w.has_cloud_scale}, {"fco", w.has_cloud_offset},
        {"fcq", w.has_cloud_quality}, {"fcdt", w.has_cloud_detail},
        {"fcl2", w.has_cloud_layer2}, {"fcl2p", w.has_cloud_layer2_params}, {"fcl2h", w.has_cloud_layer2_heights},
        {"fcll", w.has_cloud_lighting}, {"ffog", w.has_fog}, {"ffogp", w.has_fog_params},
        {"fgr", w.has_godrays}, {"fgrp", w.has_godrays_params},
        {"fms", w.has_multi_scatter}, {"fap", w.has_aerial_perspective}, {"fapp", w.has_aerial_params},
        {"fovr", w.has_overlay}, {"fovrp", w.has_overlay_params}, {"fwth", w.has_weather_params},

        {"bgc", w.background_color}, {"bgs", w.background_strength}, {"hr", w.hdri_rotation}, {"hint", w.hdri_intensity},
        {"se", w.sun_elevation}, {"sa", w.sun_azimuth}, {"si", w.sun_intensity}, {"ss", w.sun_size},
        {"ad", w.air_density}, {"dd", w.dust_density}, {"od", w.ozone_density},
        {"hum", w.humidity}, {"tmp", w.temperature}, {"ozs", w.ozone_absorption_scale},
        {"alt", w.altitude}, {"ma", w.mie_anisotropy},
        {"cd", w.cloud_density}, {"cc", w.cloud_coverage}, {"cs", w.cloud_scale},
        {"cox", w.cloud_offset_x}, {"coz", w.cloud_offset_z},
        {"cq", w.cloud_quality}, {"cdt", w.cloud_detail}, {"cbs", w.cloud_base_steps},
        {"cmn", w.cloud_height_min}, {"cmx", w.cloud_height_max},

        {"cl2e", w.cloud_layer2_enabled}, {"ccov2", w.cloud2_coverage}, {"cden2", w.cloud2_density}, {"cscl2", w.cloud2_scale},
        {"cmnh2", w.cloud2_height_min}, {"cmxh2", w.cloud2_height_max},

        {"cstl", w.cloud_light_steps}, {"cshd", w.cloud_shadow_strength}, {"camb", w.cloud_ambient_strength}, {"csil", w.cloud_silver_intensity}, {"cabs", w.cloud_absorption},
        {"cani", w.cloud_anisotropy}, {"canib", w.cloud_anisotropy_back}, {"clmx", w.cloud_lobe_mix}, {"cemi", w.cloud_emissive_intensity}, {"cemc", w.cloud_emissive_color},

        {"fge", w.fog_enabled}, {"fgd", w.fog_density}, {"fgh", w.fog_height}, {"fgf", w.fog_falloff}, {"fgds", w.fog_distance}, {"fgc", w.fog_color}, {"fgs", w.fog_sun_scatter},

        {"gre", w.godrays_enabled}, {"gri", w.godrays_intensity}, {"grd", w.godrays_density}, {"grs", w.godrays_samples},

        {"mse", w.multi_scatter_enabled}, {"msf", w.multi_scatter_factor}, {"ape", w.aerial_perspective}, {"apden", w.aerial_density}, {"apmin", w.aerial_min_distance}, {"apmax", w.aerial_max_distance},
        {"ove", w.env_overlay_enabled}, {"ovi", w.env_overlay_intensity}, {"ovr", w.env_overlay_rotation}, {"ovm", w.env_overlay_blend_mode},
        {"wte", w.weather_enabled}, {"wtt", w.weather_type}, {"wti", w.weather_intensity}, {"wtd", w.weather_density},
        {"wtw", w.weather_wind_direction}, {"wtws", w.weather_wind_speed}, {"wtps", w.weather_precipitation_scale},
        {"wtv", w.weather_visibility}, {"wtwet", w.weather_surface_wetness}, {"wtacc", w.weather_surface_accumulation},
        {"wtset", w.weather_surface_settling}, {"wthgt", w.weather_surface_height},
        {"wtvm", w.weather_visual_mode}, {"wtsr", w.weather_surface_response_enabled}
    };
}

inline void from_json(const json& j, WorldKeyframe& w) {
    w.has_background_color = j.value("fbgc", false); w.has_background_strength = j.value("fbgs", false);
    w.has_hdri_rotation = j.value("fhr", false);
    w.has_sun_elevation = j.value("fse", false); w.has_sun_azimuth = j.value("fsa", false);
    w.has_sun_intensity = j.value("fsi", false); w.has_sun_size = j.value("fss", false);
    w.has_air_density = j.value("fad", false); w.has_dust_density = j.value("fdd", false);
    w.has_ozone_density = j.value("fod", false); 
    w.has_humidity = j.value("fhum", false); w.has_temperature = j.value("ftmp", false);
    w.has_ozone_absorption_scale = j.value("fozs", false);
    w.has_altitude = j.value("falt", false); w.has_mie_anisotropy = j.value("fma", false);
    w.has_cloud_density = j.value("fcd", false); w.has_cloud_coverage = j.value("fcc", false);
    w.has_cloud_scale = j.value("fcs", false); w.has_cloud_offset = j.value("fco", false);
    w.has_cloud_quality = j.value("fcq", false); w.has_cloud_detail = j.value("fcdt", false);
    w.has_cloud_layer2 = j.value("fcl2", false); w.has_cloud_layer2_params = j.value("fcl2p", false);
    w.has_cloud_layer2_heights = j.value("fcl2h", false);
    w.has_cloud_lighting = j.value("fcll", false);
    w.has_fog = j.value("ffog", false); w.has_fog_params = j.value("ffogp", false);
    w.has_godrays = j.value("fgr", false); w.has_godrays_params = j.value("fgrp", false);
    w.has_multi_scatter = j.value("fms", false); w.has_aerial_perspective = j.value("fap", false);
    w.has_aerial_params = j.value("fapp", false);
    w.has_overlay = j.value("fovr", false); w.has_overlay_params = j.value("fovrp", false);
    w.has_weather_params = j.value("fwth", false);

    if(j.contains("bgc")) j.at("bgc").get_to(w.background_color);
    w.background_strength = j.value("bgs", 1.0f); 
    w.hdri_rotation = j.value("hr", 0.0f); w.hdri_intensity = j.value("hint", 1.0f);
    w.sun_elevation = j.value("se", 15.0f); w.sun_azimuth = j.value("sa", 0.0f);
    w.sun_intensity = j.value("si", 1.0f); w.sun_size = j.value("ss", 0.545f);
    w.air_density = j.value("ad", 1.0f); w.dust_density = j.value("dd", 1.0f);
    w.ozone_density = j.value("od", 1.0f); 
    w.humidity = j.value("hum", 0.1f); w.temperature = j.value("tmp", 15.0f);
    w.ozone_absorption_scale = j.value("ozs", 1.0f);
    w.altitude = j.value("alt", 0.0f);
    w.mie_anisotropy = j.value("ma", 0.76f);
    w.cloud_density = j.value("cd", 0.5f); w.cloud_coverage = j.value("cc", 0.5f);
    w.cloud_scale = j.value("cs", 1.0f);
    w.cloud_offset_x = j.value("cox", 0.0f); w.cloud_offset_z = j.value("coz", 0.0f);
    w.cloud_quality = j.value("cq", 1.0f); w.cloud_detail = j.value("cdt", 1.0f);
    w.cloud_base_steps = j.value("cbs", 8);
    w.cloud_height_min = j.value("cmn", 500.0f); w.cloud_height_max = j.value("cmx", 2000.0f);

    w.cloud_layer2_enabled = j.value("cl2e", 0);
    w.cloud2_coverage = j.value("ccov2", 0.3f); w.cloud2_density = j.value("cden2", 0.3f);
    w.cloud2_scale = j.value("cscl2", 8.0f);
    w.cloud2_height_min = j.value("cmnh2", 6000.0f); w.cloud2_height_max = j.value("cmxh2", 7000.0f);

    w.cloud_light_steps = j.value("cstl", 0);
    w.cloud_shadow_strength = j.value("cshd", 1.0f); w.cloud_ambient_strength = j.value("camb", 1.0f);
    w.cloud_silver_intensity = j.value("csil", 1.0f); w.cloud_absorption = j.value("cabs", 1.0f);
    w.cloud_anisotropy = j.value("cani", 0.85f); w.cloud_anisotropy_back = j.value("canib", -0.3f);
    w.cloud_lobe_mix = j.value("clmx", 0.5f); w.cloud_emissive_intensity = j.value("cemi", 0.0f);
    if(j.contains("cemc")) j.at("cemc").get_to(w.cloud_emissive_color);

    w.fog_enabled = j.value("fge", 0);
    w.fog_density = j.value("fgd", 0.01f); w.fog_height = j.value("fgh", 500.0f);
    w.fog_falloff = j.value("fgf", 0.003f); w.fog_distance = j.value("fgds", 10000.0f);
    if(j.contains("fgc")) j.at("fgc").get_to(w.fog_color);
    w.fog_sun_scatter = j.value("fgs", 0.5f);

    w.godrays_enabled = j.value("gre", 0);
    w.godrays_intensity = j.value("gri", 0.5f); w.godrays_density = j.value("grd", 0.1f);
    w.godrays_samples = j.value("grs", 16);

    w.multi_scatter_enabled = j.value("mse", 1); w.multi_scatter_factor = j.value("msf", 0.3f);
    w.aerial_perspective = j.value("ape", 1);
    w.aerial_density = j.value("apden", 1.0f);
    w.aerial_min_distance = j.value("apmin", 10.0f); w.aerial_max_distance = j.value("apmax", 5000.0f);

    w.env_overlay_enabled = j.value("ove", 0);
    w.env_overlay_intensity = j.value("ovi", 1.0f); w.env_overlay_rotation = j.value("ovr", 0.0f);
    w.env_overlay_blend_mode = j.value("ovm", 0);

    w.weather_enabled = j.value("wte", 0);
    w.weather_type = j.value("wtt", 0);
    w.weather_intensity = j.value("wti", 0.0f);
    w.weather_density = j.value("wtd", 0.0f);
    if(j.contains("wtw")) j.at("wtw").get_to(w.weather_wind_direction);
    w.weather_wind_speed = j.value("wtws", 0.0f);
    w.weather_precipitation_scale = j.value("wtps", 1.0f);
    w.weather_visibility = j.value("wtv", 1.0f);
    w.weather_surface_wetness = j.value("wtwet", 0.0f);
    w.weather_surface_accumulation = j.value("wtacc", 0.0f);
    w.weather_surface_settling = j.value("wtset", 0.0f);
    w.weather_surface_height = j.value("wthgt", 0.0f);
    w.weather_visual_mode = j.value("wtvm", WEATHER_VISUAL_OVERLAY);
    w.weather_surface_response_enabled = j.value("wtsr", 1);
}

// TerrainKeyframe
inline void to_json(json& j, const TerrainKeyframe& t) {
    j = json{
        {"has", t.has_data}, {"hsia", t.has_splat},
        {"w", t.width}, {"h", t.height},
        {"sw", t.splat_width}, {"sh", t.splat_height},
        {"hdat", t.heightData}, {"hard", t.hardnessData},
        {"sdat", t.splatData}
    };
}

inline void from_json(const json& j, TerrainKeyframe& t) {
    t.has_data = j.value("has", false); t.has_splat = j.value("hsia", false);
    t.width = j.value("w", 0); t.height = j.value("h", 0);
    t.splat_width = j.value("sw", 0); t.splat_height = j.value("sh", 0);
    if(j.contains("hdat")) j.at("hdat").get_to(t.heightData);
    if(j.contains("hard")) j.at("hard").get_to(t.hardnessData);
    if(j.contains("sdat")) j.at("sdat").get_to(t.splatData);
}

// TransformKeyframe
inline void to_json(json& j, const TransformKeyframe& tk) {
    j = json{
        {"pos", tk.position}, {"rot", tk.rotation}, {"scl", tk.scale},
        {"fpos", tk.has_position}, {"frot", tk.has_rotation}, {"fscl", tk.has_scale},
        {"fpx", tk.has_pos_x}, {"fpy", tk.has_pos_y}, {"fpz", tk.has_pos_z},
        {"frx", tk.has_rot_x}, {"fry", tk.has_rot_y}, {"frz", tk.has_rot_z},
        {"fsx", tk.has_scl_x}, {"fsy", tk.has_scl_y}, {"fsz", tk.has_scl_z}
    };
    // Curve metadata (graph editor). Sparse: channels in the legacy-linear state
    // are omitted entirely, so files from before this feature round-trip unchanged.
    // Entry layout: [channel, interp, auto, in_dx, in_dy, out_dx, out_dy]
    json crv = json::array();
    for (int ch = 0; ch < CURVE_CHANNEL_COUNT; ++ch) {
        const ChannelKeyMeta& m = tk.curve[ch];
        if (m.isLegacyLinear()) continue;
        crv.push_back(json::array({ ch, (int)m.interp, m.auto_tangent ? 1 : 0,
                                    m.in_dx, m.in_dy, m.out_dx, m.out_dy }));
    }
    if (!crv.empty()) j["crv"] = crv;
}

inline void from_json(const json& j, TransformKeyframe& tk) {
    if(j.contains("pos")) j.at("pos").get_to(tk.position);
    if(j.contains("rot")) j.at("rot").get_to(tk.rotation);
    if(j.contains("scl")) j.at("scl").get_to(tk.scale);

    tk.has_position = j.value("fpos", true);
    tk.has_rotation = j.value("frot", true);
    tk.has_scale = j.value("fscl", true);

    tk.has_pos_x = j.value("fpx", true); tk.has_pos_y = j.value("fpy", true); tk.has_pos_z = j.value("fpz", true);
    tk.has_rot_x = j.value("frx", true); tk.has_rot_y = j.value("fry", true); tk.has_rot_z = j.value("frz", true);
    tk.has_scl_x = j.value("fsx", true); tk.has_scl_y = j.value("fsy", true); tk.has_scl_z = j.value("fsz", true);

    // BACKWARD COMPAT: keys with no stored curve data must evaluate exactly like
    // the old system, i.e. pure linear — NOT the Bezier default of new keys.
    for (auto& m : tk.curve) m.resetToLegacyLinear();
    if (j.contains("crv") && j["crv"].is_array()) {
        for (const auto& e : j["crv"]) {
            if (!e.is_array() || e.size() < 7) continue;
            const int ch = e[0].get<int>();
            if (ch < 0 || ch >= CURVE_CHANNEL_COUNT) continue;
            ChannelKeyMeta& m = tk.curve[ch];
            const int interp = e[1].get<int>();
            m.interp = (interp == (int)KeyInterp::Constant) ? KeyInterp::Constant
                     : (interp == (int)KeyInterp::Bezier)   ? KeyInterp::Bezier
                                                            : KeyInterp::Linear;
            m.auto_tangent = e[2].get<int>() != 0;
            m.in_dx = e[3].get<float>();  m.in_dy = e[4].get<float>();
            m.out_dx = e[5].get<float>(); m.out_dy = e[6].get<float>();
        }
    }
}

// Keyframe
inline void to_json(json& j, const AnimGraphKeyframe& k) {
    j = json{
        {"fp", k.float_params},
        {"bp", k.bool_params},
        {"ip", k.int_params},
        {"co", k.clip_overrides},
        {"cso", k.clip_speed_overrides},
        {"trg", k.triggers},
        {"state", k.force_state}
    };
}

inline void from_json(const json& j, AnimGraphKeyframe& k) {
    k.float_params = j.value("fp", decltype(k.float_params){});
    k.bool_params = j.value("bp", decltype(k.bool_params){});
    k.int_params = j.value("ip", decltype(k.int_params){});
    k.clip_overrides = j.value("co", decltype(k.clip_overrides){});
    k.clip_speed_overrides = j.value("cso", decltype(k.clip_speed_overrides){});
    k.triggers = j.value("trg", decltype(k.triggers){});
    k.force_state = j.value("state", std::string{});
}

// Keyframe
inline void to_json(json& j, const Keyframe& k) {
    j = json{
        {"fr", k.frame},
        {"ftr", k.has_transform}, {"fmat", k.has_material},
        {"fli", k.has_light}, {"fcam", k.has_camera},
        {"fwor", k.has_world}, {"fter", k.has_terrain}, {"femi", k.has_emitter},
        {"fagr", k.has_anim_graph}
    };
    if(k.has_transform) j["tr"] = k.transform;
    if(k.has_material) j["mat"] = k.material;
    if(k.has_light) j["li"] = k.light;
    if(k.has_camera) j["cam"] = k.camera;
    if(k.has_world) j["wor"] = k.world;
    if(k.has_terrain) j["ter"] = k.terrain;
    if(k.has_emitter) j["emi"] = k.emitter;
    if(k.has_anim_graph) j["agr"] = k.anim_graph;
}

inline void from_json(const json& j, Keyframe& k) {
    k.frame = j.value("fr", 0);
    
    // Robust loading: Check explicit flags, fallback to data block existence
    k.has_transform = j.value("ftr", j.contains("tr"));
    k.has_material = j.value("fmat", j.contains("mat"));
    k.has_light = j.value("fli", j.contains("li"));
    k.has_camera = j.value("fcam", j.contains("cam"));
    k.has_world = j.value("fwor", j.contains("wor"));
    k.has_terrain = j.value("fter", j.contains("ter"));
    k.has_emitter = j.value("femi", j.contains("emi"));
    k.has_anim_graph = j.value("fagr", j.contains("agr"));
    
    if(k.has_transform && j.contains("tr")) j.at("tr").get_to(k.transform);
    if(k.has_material && j.contains("mat")) j.at("mat").get_to(k.material);
    if(k.has_light && j.contains("li")) j.at("li").get_to(k.light);
    if(k.has_camera && j.contains("cam")) j.at("cam").get_to(k.camera);
    if(k.has_world && j.contains("wor")) j.at("wor").get_to(k.world);
    if(k.has_terrain && j.contains("ter")) j.at("ter").get_to(k.terrain);
    if(k.has_emitter && j.contains("emi")) j.at("emi").get_to(k.emitter);
    if(k.has_anim_graph && j.contains("agr")) j.at("agr").get_to(k.anim_graph);
}

// ObjectAnimationTrack
inline void to_json(json& j, const ObjectAnimationTrack& t) {
    j = json{
        {"name", t.object_name},
        {"idx", t.object_index},
        {"kfs", t.keyframes}
    };
}

inline void from_json(const json& j, ObjectAnimationTrack& t) {
    t.object_name = j.value("name", "");
    t.object_index = j.value("idx", -1);
    if(j.contains("kfs")) {
        j.at("kfs").get_to(t.keyframes);
        // CRITICAL: Ensure keyframes are sorted, otherwise binary search in evaluate() fails
        std::sort(t.keyframes.begin(), t.keyframes.end(),
            [](const Keyframe& a, const Keyframe& b){ return a.frame < b.frame; });
        // Stored auto handles may predate a key that was added later; recompute once.
        t.refreshAutoTangents();
    }
}

// TimelineManager Member Implementations
inline void TimelineManager::serialize(json& j) const {
    j = json{
        {"fr", current_frame},
        {"tracks", tracks}
    };
}

inline void TimelineManager::deserialize(const json& j) {
    current_frame = j.value("fr", 0);
    if(j.contains("tracks")) j.at("tracks").get_to(tracks);
}


