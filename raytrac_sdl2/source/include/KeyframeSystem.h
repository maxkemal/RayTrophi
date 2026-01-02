#pragma once
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "Vec3.h"
#include "material_gpu.h"  // For GpuMaterial
#include "json.hpp"
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

// ============================================================================
// KEYFRAME SYSTEM - Object-based animation with transform + material props
// ============================================================================

// Material property keyframe - ALIGNED WITH GpuMaterial for GPU/CPU compatibility
struct MaterialKeyframe {
    // Material identification
    uint16_t material_id = 0;
    
    // Per-Property Flags
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
        
        specular = 0.5f;
        specular_tint = 0.0f;
        clearcoat_roughness = 0.1f;
        normal_strength = 1.0f;
        emission_strength = 1.0f;
    }
    
    void applyTo(GpuMaterial& gpu) const {
        gpu.albedo = make_float3(albedo.x, albedo.y, albedo.z);
        gpu.opacity = opacity;
        gpu.roughness = roughness;
        gpu.metallic = metallic;
        gpu.clearcoat = clearcoat;
        gpu.transmission = transmission;
        gpu.emission = make_float3(emission.x, emission.y, emission.z);
        gpu.ior = ior;
        gpu.subsurface_color = make_float3(subsurface_color.x, subsurface_color.y, subsurface_color.z);
        gpu.subsurface = subsurface;
        gpu.anisotropic = anisotropic;
        gpu.sheen = sheen;
        gpu.sheen_tint = sheen_tint;
    }
    
    static MaterialKeyframe lerp(const MaterialKeyframe& a, const MaterialKeyframe& b, float t) {
        MaterialKeyframe result;
        result.material_id = (t < 0.5f) ? a.material_id : b.material_id;
        
        // Albedo
        result.has_albedo = a.has_albedo || b.has_albedo;
        if (a.has_albedo && b.has_albedo) {
            result.albedo = a.albedo + (b.albedo - a.albedo) * t;
        } else if (a.has_albedo) {
            result.albedo = a.albedo;
        } else if (b.has_albedo) {
            result.albedo = b.albedo;
        }

        // Opacity
        result.has_opacity = a.has_opacity || b.has_opacity;
        if (a.has_opacity && b.has_opacity) {
            result.opacity = a.opacity + (b.opacity - a.opacity) * t;
        } else if (a.has_opacity) {
            result.opacity = a.opacity;
        } else if (b.has_opacity) {
            result.opacity = b.opacity;
        }

        // Roughness
        result.has_roughness = a.has_roughness || b.has_roughness;
        if (a.has_roughness && b.has_roughness) result.roughness = a.roughness + (b.roughness - a.roughness) * t;
        else if (a.has_roughness) result.roughness = a.roughness;
        else if (b.has_roughness) result.roughness = b.roughness;

        // Metallic
        result.has_metallic = a.has_metallic || b.has_metallic;
        if (a.has_metallic && b.has_metallic) result.metallic = a.metallic + (b.metallic - a.metallic) * t;
        else if (a.has_metallic) result.metallic = a.metallic;
        else if (b.has_metallic) result.metallic = b.metallic;

        // Emission
        result.has_emission = a.has_emission || b.has_emission;
        if (a.has_emission && b.has_emission) {
            result.emission = a.emission + (b.emission - a.emission) * t;
            result.emission_strength = a.emission_strength + (b.emission_strength - a.emission_strength) * t;
        } else if (a.has_emission) {
            result.emission = a.emission; result.emission_strength = a.emission_strength;
        } else if (b.has_emission) {
            result.emission = b.emission; result.emission_strength = b.emission_strength;
        }

        // Transmission
        result.has_transmission = a.has_transmission || b.has_transmission;
        if (a.has_transmission && b.has_transmission) result.transmission = a.transmission + (b.transmission - a.transmission) * t;
        else if (a.has_transmission) result.transmission = a.transmission;
        else if (b.has_transmission) result.transmission = b.transmission;

        // IOR
        result.has_ior = a.has_ior || b.has_ior;
        if (a.has_ior && b.has_ior) result.ior = a.ior + (b.ior - a.ior) * t;
        else if (a.has_ior) result.ior = a.ior;
        else if (b.has_ior) result.ior = b.ior;

        // Clearcoat
        result.has_clearcoat = a.has_clearcoat || b.has_clearcoat;
        if (a.has_clearcoat && b.has_clearcoat) {
             result.clearcoat = a.clearcoat + (b.clearcoat - a.clearcoat) * t;
             result.clearcoat_roughness = a.clearcoat_roughness + (b.clearcoat_roughness - a.clearcoat_roughness) * t;
        } else if (a.has_clearcoat) {
             result.clearcoat = a.clearcoat; result.clearcoat_roughness = a.clearcoat_roughness;
        } else if (b.has_clearcoat) {
             result.clearcoat = b.clearcoat; result.clearcoat_roughness = b.clearcoat_roughness;
        }

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
        result.has_normal = a.has_normal || b.has_normal;
        if (a.has_normal && b.has_normal) result.normal_strength = a.normal_strength + (b.normal_strength - a.normal_strength) * t;
        else if (a.has_normal) result.normal_strength = a.normal_strength;
        else if (b.has_normal) result.normal_strength = b.normal_strength;

        return result;
    }
};

// ============================================================================
// LIGHT KEYFRAME - Position, color, intensity, direction
// ============================================================================
struct LightKeyframe {
    // Per-property flags - which properties are keyed
    bool has_position = false;
    bool has_color = false;
    bool has_intensity = false;
    bool has_direction = false;
    
    // Property values
    Vec3 position = Vec3(0, 0, 0);
    Vec3 color = Vec3(1, 1, 1);
    float intensity = 1.0f;
    Vec3 direction = Vec3(0, -1, 0);  // For directional/spot lights
    
    LightKeyframe() = default;
    
    // Lerp only interpolates properties that are keyed in BOTH keyframes
    static LightKeyframe lerp(const LightKeyframe& a, const LightKeyframe& b, float t) {
        LightKeyframe result;
        
        // Position - interpolate if both keyed, otherwise use 'a' value
        result.has_position = a.has_position || b.has_position;
        if (a.has_position && b.has_position) {
            result.position = a.position + (b.position - a.position) * t;
        } else if (a.has_position) {
            result.position = a.position;
        } else if (b.has_position) {
            result.position = b.position;
        }
        
        // Color
        result.has_color = a.has_color || b.has_color;
        if (a.has_color && b.has_color) {
            result.color = a.color + (b.color - a.color) * t;
        } else if (a.has_color) {
            result.color = a.color;
        } else if (b.has_color) {
            result.color = b.color;
        }
        
        // Intensity
        result.has_intensity = a.has_intensity || b.has_intensity;
        if (a.has_intensity && b.has_intensity) {
            result.intensity = a.intensity + (b.intensity - a.intensity) * t;
        } else if (a.has_intensity) {
            result.intensity = a.intensity;
        } else if (b.has_intensity) {
            result.intensity = b.intensity;
        }
        
        // Direction
        result.has_direction = a.has_direction || b.has_direction;
        if (a.has_direction && b.has_direction) {
            result.direction = a.direction + (b.direction - a.direction) * t;
        } else if (a.has_direction) {
            result.direction = a.direction;
        } else if (b.has_direction) {
            result.direction = b.direction;
        }
        
        return result;
    }
};

// ============================================================================
// CAMERA KEYFRAME - Position, target, FOV, DOF
// ============================================================================
struct CameraKeyframe {
    // Per-property flags - which properties are keyed
    bool has_position = false;
    bool has_target = false;
    bool has_fov = false;
    bool has_focus = false;
    bool has_aperture = false;
    
    // Property values
    Vec3 position = Vec3(0, 0, 0);
    Vec3 target = Vec3(0, 0, -1);
    float fov = 40.0f;
    float focus_distance = 10.0f;
    float lens_radius = 0.0f;
    
    CameraKeyframe() = default;
    
    // Lerp only interpolates properties that are keyed in BOTH keyframes
    static CameraKeyframe lerp(const CameraKeyframe& a, const CameraKeyframe& b, float t) {
        CameraKeyframe result;
        
        // Position
        result.has_position = a.has_position || b.has_position;
        if (a.has_position && b.has_position) {
            result.position = a.position + (b.position - a.position) * t;
        } else if (a.has_position) {
            result.position = a.position;
        } else if (b.has_position) {
            result.position = b.position;
        }
        
        // Target
        result.has_target = a.has_target || b.has_target;
        if (a.has_target && b.has_target) {
            result.target = a.target + (b.target - a.target) * t;
        } else if (a.has_target) {
            result.target = a.target;
        } else if (b.has_target) {
            result.target = b.target;
        }
        
        // FOV
        result.has_fov = a.has_fov || b.has_fov;
        if (a.has_fov && b.has_fov) {
            result.fov = a.fov + (b.fov - a.fov) * t;
        } else if (a.has_fov) {
            result.fov = a.fov;
        } else if (b.has_fov) {
            result.fov = b.fov;
        }
        
        // Focus Distance
        result.has_focus = a.has_focus || b.has_focus;
        if (a.has_focus && b.has_focus) {
            result.focus_distance = a.focus_distance + (b.focus_distance - a.focus_distance) * t;
        } else if (a.has_focus) {
            result.focus_distance = a.focus_distance;
        } else if (b.has_focus) {
            result.focus_distance = b.focus_distance;
        }
        
        // Lens Radius (Aperture)
        result.has_aperture = a.has_aperture || b.has_aperture;
        if (a.has_aperture && b.has_aperture) {
            result.lens_radius = a.lens_radius + (b.lens_radius - a.lens_radius) * t;
        } else if (a.has_aperture) {
            result.lens_radius = a.lens_radius;
        } else if (b.has_aperture) {
            result.lens_radius = b.lens_radius;
        }
        
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
    bool has_air_density = false;
    bool has_dust_density = false;
    bool has_ozone_density = false;
    bool has_altitude = false;
    bool has_mie_anisotropy = false;
    
    // Cloud properties
    bool has_cloud_density = false;
    bool has_cloud_coverage = false;
    bool has_cloud_scale = false;
    bool has_cloud_offset = false;  // For offset_x and offset_z together
    
    // Other groups (kept as groups for now, can be expanded if needed)
    bool has_fog = false;
    bool has_overlay = false;
    
    // Property values
    Vec3 background_color = Vec3(0.5, 0.7, 1.0);
    float background_strength = 1.0f;
    float hdri_rotation = 0.0f;
    
    float sun_elevation = 15.0f;
    float sun_azimuth = 0.0f;
    float sun_intensity = 1.0f;
    float sun_size = 0.545f;
    
    float air_density = 1.0f;
    float dust_density = 1.0f;
    float ozone_density = 1.0f;
    float altitude = 0.0f;
    float mie_anisotropy = 0.76f;
    
    float cloud_density = 0.5f;
    float cloud_coverage = 0.5f;
    float cloud_scale = 1.0f;
    float cloud_offset_x = 0.0f;
    float cloud_offset_z = 0.0f;
    
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
        } else if (a.has_hdri_rotation) {
            result.hdri_rotation = a.hdri_rotation;
        } else if (b.has_hdri_rotation) {
            result.hdri_rotation = b.hdri_rotation;
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
        
        // Ozone Density
        result.has_ozone_density = a.has_ozone_density || b.has_ozone_density;
        if (a.has_ozone_density && b.has_ozone_density) {
            result.ozone_density = a.ozone_density + (b.ozone_density - a.ozone_density) * t;
        } else if (a.has_ozone_density) {
            result.ozone_density = a.ozone_density;
        } else if (b.has_ozone_density) {
            result.ozone_density = b.ozone_density;
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
        
        // ===== CLOUD PROPERTIES =====
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
    
    TransformKeyframe() = default;
    
    TransformKeyframe(const Vec3& pos, const Vec3& rot, const Vec3& scl)
        : position(pos), rotation(rot), scale(scl) {
    };
    
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


// Complete keyframe - combines all types of animation data
struct Keyframe {
    int frame = 0;
    TransformKeyframe transform;
    MaterialKeyframe material;
    LightKeyframe light;
    CameraKeyframe camera;
    WorldKeyframe world;        // NEW
    
    // Flags to track what's keyframed
    bool has_transform = false;
    bool has_material = false;
    bool has_light = false;
    bool has_camera = false;   
    bool has_world = false;
    bool has_terrain = false;   // NEW: Terrain morphing support
    
    TerrainKeyframe terrain;    // NEW
    
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
                it->transform = kf.transform;
                it->has_transform = true;
            }
            if (kf.has_material) {
                it->material = kf.material;
                it->has_material = true;
            }
            if (kf.has_light) {
                it->light = kf.light;
                it->has_light = true;
            }
            if (kf.has_camera) {
                it->camera = kf.camera;
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
    
    // Get keyframe at exact frame (returns nullptr if not found)
    Keyframe* getKeyframeAt(int frame) {
        auto it = std::find_if(keyframes.begin(), keyframes.end(),
            [frame](const Keyframe& kf) { return kf.frame == frame; });
        return (it != keyframes.end()) ? &(*it) : nullptr;
    }
    
    // Evaluate animation at given frame (with INDEPENDENT CHANNEL INTERPOLATION)
    Keyframe evaluate(int current_frame) const {
        Keyframe result(current_frame);
        if (keyframes.empty()) return result;

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
        
        // Helper specifically for Transform values (float scalar lerp)
        auto interpolateScalar = [&](float& result_val, const Keyframe* p, const Keyframe* n, float (TransformKeyframe::*val_ptr), bool (TransformKeyframe::*flag_ptr)) {
            if (p && n) {
                if (p == n) result_val = (p->transform.*val_ptr); // Cast if needed for vector components? using member pointers is tricky for Vec3 struct
                else {
                    float t = float(current_frame - p->frame) / float(n->frame - p->frame);
                    // Manually lerp scalars? no, we can't easily use member pointers for Vec3.x
                    // Let's just do it manually for each axis to be safe and clear.
                }
            }
        };

        // --- POSITION X ---
        {
            auto has = [](const Keyframe& k) { return k.has_transform && k.transform.has_pos_x; };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                 float t = (p == n) ? 0.0f : float(current_frame - p->frame) / float(n->frame - p->frame);
                 result.transform.position.x = p->transform.position.x + (n->transform.position.x - p->transform.position.x) * t;
                 result.transform.has_pos_x = true;
            } else if (p) { result.transform.position.x = p->transform.position.x; result.transform.has_pos_x = true; }
            else if (n) { result.transform.position.x = n->transform.position.x; result.transform.has_pos_x = true; }
        }
        // --- POSITION Y ---
        {
            auto has = [](const Keyframe& k) { return k.has_transform && k.transform.has_pos_y; };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                 float t = (p == n) ? 0.0f : float(current_frame - p->frame) / float(n->frame - p->frame);
                 result.transform.position.y = p->transform.position.y + (n->transform.position.y - p->transform.position.y) * t;
                 result.transform.has_pos_y = true;
            } else if (p) { result.transform.position.y = p->transform.position.y; result.transform.has_pos_y = true; }
            else if (n) { result.transform.position.y = n->transform.position.y; result.transform.has_pos_y = true; }
        }
        // --- POSITION Z ---
        {
            auto has = [](const Keyframe& k) { return k.has_transform && k.transform.has_pos_z; };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                 float t = (p == n) ? 0.0f : float(current_frame - p->frame) / float(n->frame - p->frame);
                 result.transform.position.z = p->transform.position.z + (n->transform.position.z - p->transform.position.z) * t;
                 result.transform.has_pos_z = true;
            } else if (p) { result.transform.position.z = p->transform.position.z; result.transform.has_pos_z = true; }
            else if (n) { result.transform.position.z = n->transform.position.z; result.transform.has_pos_z = true; }
        }
        result.transform.has_position = result.transform.has_pos_x || result.transform.has_pos_y || result.transform.has_pos_z;

        // --- ROTATION X ---
        {
            auto has = [](const Keyframe& k) { return k.has_transform && k.transform.has_rot_x; };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                 float t = (p == n) ? 0.0f : float(current_frame - p->frame) / float(n->frame - p->frame);
                 result.transform.rotation.x = p->transform.rotation.x + (n->transform.rotation.x - p->transform.rotation.x) * t;
                 result.transform.has_rot_x = true;
            } else if (p) { result.transform.rotation.x = p->transform.rotation.x; result.transform.has_rot_x = true; }
            else if (n) { result.transform.rotation.x = n->transform.rotation.x; result.transform.has_rot_x = true; }
        }
        // --- ROTATION Y ---
        {
            auto has = [](const Keyframe& k) { return k.has_transform && k.transform.has_rot_y; };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                 float t = (p == n) ? 0.0f : float(current_frame - p->frame) / float(n->frame - p->frame);
                 result.transform.rotation.y = p->transform.rotation.y + (n->transform.rotation.y - p->transform.rotation.y) * t;
                 result.transform.has_rot_y = true;
            } else if (p) { result.transform.rotation.y = p->transform.rotation.y; result.transform.has_rot_y = true; }
            else if (n) { result.transform.rotation.y = n->transform.rotation.y; result.transform.has_rot_y = true; }
        }
         // --- ROTATION Z ---
        {
            auto has = [](const Keyframe& k) { return k.has_transform && k.transform.has_rot_z; };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                 float t = (p == n) ? 0.0f : float(current_frame - p->frame) / float(n->frame - p->frame);
                 result.transform.rotation.z = p->transform.rotation.z + (n->transform.rotation.z - p->transform.rotation.z) * t;
                 result.transform.has_rot_z = true;
            } else if (p) { result.transform.rotation.z = p->transform.rotation.z; result.transform.has_rot_z = true; }
            else if (n) { result.transform.rotation.z = n->transform.rotation.z; result.transform.has_rot_z = true; }
        }
        result.transform.has_rotation = result.transform.has_rot_x || result.transform.has_rot_y || result.transform.has_rot_z;

        // --- SCALE X ---
        {
            auto has = [](const Keyframe& k) { return k.has_transform && k.transform.has_scl_x; };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                 float t = (p == n) ? 0.0f : float(current_frame - p->frame) / float(n->frame - p->frame);
                 result.transform.scale.x = p->transform.scale.x + (n->transform.scale.x - p->transform.scale.x) * t;
                 result.transform.has_scl_x = true;
            } else if (p) { result.transform.scale.x = p->transform.scale.x; result.transform.has_scl_x = true; }
            else if (n) { result.transform.scale.x = n->transform.scale.x; result.transform.has_scl_x = true; }
        }
        // --- SCALE Y ---
        {
            auto has = [](const Keyframe& k) { return k.has_transform && k.transform.has_scl_y; };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                 float t = (p == n) ? 0.0f : float(current_frame - p->frame) / float(n->frame - p->frame);
                 result.transform.scale.y = p->transform.scale.y + (n->transform.scale.y - p->transform.scale.y) * t;
                 result.transform.has_scl_y = true;
            } else if (p) { result.transform.scale.y = p->transform.scale.y; result.transform.has_scl_y = true; }
            else if (n) { result.transform.scale.y = n->transform.scale.y; result.transform.has_scl_y = true; }
        }
        // --- SCALE Z ---
        {
            auto has = [](const Keyframe& k) { return k.has_transform && k.transform.has_scl_z; };
            const Keyframe* p = findPrev(has);
            const Keyframe* n = findNext(has);
            if (p && n) {
                 float t = (p == n) ? 0.0f : float(current_frame - p->frame) / float(n->frame - p->frame);
                 result.transform.scale.z = p->transform.scale.z + (n->transform.scale.z - p->transform.scale.z) * t;
                 result.transform.has_scl_z = true;
            } else if (p) { result.transform.scale.z = p->transform.scale.z; result.transform.has_scl_z = true; }
            else if (n) { result.transform.scale.z = n->transform.scale.z; result.transform.has_scl_z = true; }
        }
        result.transform.has_scale = result.transform.has_scl_x || result.transform.has_scl_y || result.transform.has_scl_z;
        result.has_transform = result.transform.has_position || result.transform.has_rotation || result.transform.has_scale;
        
        // --- MATERIAL CHANNEL (Simplified block handling for now, can be updated later) ---
        {
             auto has = [](const Keyframe& k) { return k.has_material; };
             const Keyframe* prev_mat = findPrev(has);
             const Keyframe* next_mat = findNext(has);
             if (prev_mat && next_mat) {
                 if (prev_mat == next_mat) result.material = prev_mat->material;
                 else {
                     float t = float(current_frame - prev_mat->frame) / float(next_mat->frame - prev_mat->frame);
                     result.material = MaterialKeyframe::lerp(prev_mat->material, next_mat->material, t);
                 }
                 result.has_material = true;
             } else if (prev_mat) { result.material = prev_mat->material; result.has_material = true; }
             else if (next_mat) { result.material = next_mat->material; result.has_material = true; }
        }

        // --- LIGHT CHANNEL ---
        {
            auto has = [](const Keyframe& k) { return k.has_light; };
            const Keyframe* prev_light = findPrev(has);
            const Keyframe* next_light = findNext(has);
            if (prev_light && next_light) {
                if (prev_light == next_light) result.light = prev_light->light;
                else {
                    float t = float(current_frame - prev_light->frame) / float(next_light->frame - prev_light->frame);
                    result.light = LightKeyframe::lerp(prev_light->light, next_light->light, t);
                }
                result.has_light = true;
            } else if (prev_light) { result.light = prev_light->light; result.has_light = true; }
            else if (next_light) { result.light = next_light->light; result.has_light = true; }
        }

        // --- CAMERA CHANNEL ---
        {
            auto has = [](const Keyframe& k) { return k.has_camera; };
            const Keyframe* prev_cam = findPrev(has);
            const Keyframe* next_cam = findNext(has);
            if (prev_cam && next_cam) {
                if (prev_cam == next_cam) result.camera = prev_cam->camera;
                else {
                    float t = float(current_frame - prev_cam->frame) / float(next_cam->frame - prev_cam->frame);
                    result.camera = CameraKeyframe::lerp(prev_cam->camera, next_cam->camera, t);
                }
                result.has_camera = true;
            } else if (prev_cam) { result.camera = prev_cam->camera; result.has_camera = true; }
            else if (next_cam) { result.camera = next_cam->camera; result.has_camera = true; }
        }

        // --- WORLD CHANNEL ---
        {
            auto has = [](const Keyframe& k) { return k.has_world; };
            const Keyframe* p_world = findPrev(has);
            const Keyframe* n_world = findNext(has);
            if (p_world && n_world) {
                float range = (float)(n_world->frame - p_world->frame);
                float t = (range > 0) ? (float)(current_frame - p_world->frame) / range : 0.0f;
                result.world = WorldKeyframe::lerp(p_world->world, n_world->world, t);
                result.has_world = true;
            } else if (p_world) {
                result.world = p_world->world; result.has_world = true;
            } else if (n_world) {
                result.world = n_world->world; result.has_world = true;
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
        tracks[object_name].addKeyframe(kf);
    }
    
    // Remove keyframe for object at frame
    void removeKeyframe(const std::string& object_name, int frame) {
        auto it = tracks.find(object_name);
        if (it != tracks.end()) {
            it->second.removeKeyframe(frame);
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
        {"alb", m.albedo}, {"opa", m.opacity}, {"rgh", m.roughness},
        {"met", m.metallic}, {"clr", m.clearcoat}, {"trn", m.transmission},
        {"ems", m.emission}, {"v_ior", m.ior}, 
        {"sub_c", m.subsurface_color}, {"sub", m.subsurface},
        {"ani", m.anisotropic}, {"she", m.sheen}, {"she_t", m.sheen_tint},
        {"spc", m.specular}, {"spc_t", m.specular_tint},
        {"clr_r", m.clearcoat_roughness}, {"nrm_s", m.normal_strength},
        {"ems_s", m.emission_strength}
    };
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
}

// LightKeyframe
// LightKeyframe
inline void to_json(json& j, const LightKeyframe& l) {
    j = json{
        {"fpos", l.has_position}, {"fcol", l.has_color},
        {"fint", l.has_intensity}, {"fdir", l.has_direction},
        {"int", l.intensity}
    };
    // Manual Vec3 serialization to ensure stability
    j["pos"] = {l.position.x, l.position.y, l.position.z};
    j["col"] = {l.color.x, l.color.y, l.color.z};
    j["dir"] = {l.direction.x, l.direction.y, l.direction.z};
}

inline void from_json(const json& j, LightKeyframe& l) {
    // Robust loading: Check explicit flag key first, fallback to data key existence
    l.has_position = j.value("fpos", j.contains("pos")); 
    l.has_color = j.value("fcol", j.contains("col"));
    l.has_intensity = j.value("fint", j.contains("int")); 
    l.has_direction = j.value("fdir", j.contains("dir"));

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
}

// CameraKeyframe
inline void to_json(json& j, const CameraKeyframe& c) {
    j = json{
        {"fpos", c.has_position}, {"ftgt", c.has_target},
        {"ffov", c.has_fov}, {"ffoc", c.has_focus}, {"fapt", c.has_aperture},
        {"pos", c.position}, {"tgt", c.target},
        {"fov", c.fov}, {"foc", c.focus_distance}, {"apt", c.lens_radius}
    };
}

inline void from_json(const json& j, CameraKeyframe& c) {
    // Robust loading
    c.has_position = j.value("fpos", j.contains("pos")); 
    c.has_target = j.value("ftgt", j.contains("tgt"));
    c.has_fov = j.value("ffov", j.contains("fov")); 
    c.has_focus = j.value("ffoc", j.contains("foc"));
    c.has_aperture = j.value("fapt", j.contains("apt"));

    // Safe extraction
    if(j.contains("pos")) j.at("pos").get_to(c.position);
    if(j.contains("tgt")) j.at("tgt").get_to(c.target);
    c.fov = j.value("fov", 40.0f);
    c.focus_distance = j.value("foc", 10.0f);
    c.lens_radius = j.value("apt", 0.0f);
}

// WorldKeyframe
inline void to_json(json& j, const WorldKeyframe& w) {
    j = json{
        {"fbgc", w.has_background_color}, {"fbgs", w.has_background_strength}, {"fhr", w.has_hdri_rotation},
        {"fse", w.has_sun_elevation}, {"fsa", w.has_sun_azimuth}, {"fsi", w.has_sun_intensity}, {"fss", w.has_sun_size},
        {"fad", w.has_air_density}, {"fdd", w.has_dust_density}, {"fod", w.has_ozone_density},
        {"falt", w.has_altitude}, {"fma", w.has_mie_anisotropy},
        {"fcd", w.has_cloud_density}, {"fcc", w.has_cloud_coverage}, {"fcs", w.has_cloud_scale}, {"fco", w.has_cloud_offset},
        {"bgc", w.background_color}, {"bgs", w.background_strength}, {"hr", w.hdri_rotation},
        {"se", w.sun_elevation}, {"sa", w.sun_azimuth}, {"si", w.sun_intensity}, {"ss", w.sun_size},
        {"ad", w.air_density}, {"dd", w.dust_density}, {"od", w.ozone_density},
        {"alt", w.altitude}, {"ma", w.mie_anisotropy},
        {"cd", w.cloud_density}, {"cc", w.cloud_coverage}, {"cs", w.cloud_scale},
        {"cox", w.cloud_offset_x}, {"coz", w.cloud_offset_z}
    };
}

inline void from_json(const json& j, WorldKeyframe& w) {
    w.has_background_color = j.value("fbgc", false); w.has_background_strength = j.value("fbgs", false);
    w.has_hdri_rotation = j.value("fhr", false);
    w.has_sun_elevation = j.value("fse", false); w.has_sun_azimuth = j.value("fsa", false);
    w.has_sun_intensity = j.value("fsi", false); w.has_sun_size = j.value("fss", false);
    w.has_air_density = j.value("fad", false); w.has_dust_density = j.value("fdd", false);
    w.has_ozone_density = j.value("fod", false); w.has_altitude = j.value("falt", false);
    w.has_mie_anisotropy = j.value("fma", false);
    w.has_cloud_density = j.value("fcd", false); w.has_cloud_coverage = j.value("fcc", false);
    w.has_cloud_scale = j.value("fcs", false); w.has_cloud_offset = j.value("fco", false);

    if(j.contains("bgc")) j.at("bgc").get_to(w.background_color);
    w.background_strength = j.value("bgs", 1.0f); w.hdri_rotation = j.value("hr", 0.0f);
    w.sun_elevation = j.value("se", 15.0f); w.sun_azimuth = j.value("sa", 0.0f);
    w.sun_intensity = j.value("si", 1.0f); w.sun_size = j.value("ss", 0.545f);
    w.air_density = j.value("ad", 1.0f); w.dust_density = j.value("dd", 1.0f);
    w.ozone_density = j.value("od", 1.0f); w.altitude = j.value("alt", 0.0f);
    w.mie_anisotropy = j.value("ma", 0.76f);
    w.cloud_density = j.value("cd", 0.5f); w.cloud_coverage = j.value("cc", 0.5f);
    w.cloud_scale = j.value("cs", 1.0f);
    w.cloud_offset_x = j.value("cox", 0.0f); w.cloud_offset_z = j.value("coz", 0.0f);
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
}

// Keyframe
inline void to_json(json& j, const Keyframe& k) {
    j = json{
        {"fr", k.frame},
        {"ftr", k.has_transform}, {"fmat", k.has_material},
        {"fli", k.has_light}, {"fcam", k.has_camera},
        {"fwor", k.has_world}, {"fter", k.has_terrain}
    };
    if(k.has_transform) j["tr"] = k.transform;
    if(k.has_material) j["mat"] = k.material;
    if(k.has_light) j["li"] = k.light;
    if(k.has_camera) j["cam"] = k.camera;
    if(k.has_world) j["wor"] = k.world;
    if(k.has_terrain) j["ter"] = k.terrain;
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
    
    if(k.has_transform && j.contains("tr")) j.at("tr").get_to(k.transform);
    if(k.has_material && j.contains("mat")) j.at("mat").get_to(k.material);
    if(k.has_light && j.contains("li")) j.at("li").get_to(k.light);
    if(k.has_camera && j.contains("cam")) j.at("cam").get_to(k.camera);
    if(k.has_world && j.contains("wor")) j.at("wor").get_to(k.world);
    if(k.has_terrain && j.contains("ter")) j.at("ter").get_to(k.terrain);
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

