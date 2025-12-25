#pragma once
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "Vec3.h"
#include "material_gpu.h"  // For GpuMaterial

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
// TRANSFORM KEYFRAME - Position, rotation (Euler), scale
// ============================================================================

// Transform keyframe - stores position, rotation (euler), scale
// Each component can be independently keyed
struct TransformKeyframe {
    Vec3 position = Vec3(0, 0, 0);
    Vec3 rotation = Vec3(0, 0, 0);  // Euler angles in degrees
    Vec3 scale = Vec3(1, 1, 1);
    
    // Per-channel keyed flags (for independent L/R/S keyframing)
    bool has_position = true;   // Location keyed
    bool has_rotation = true;   // Rotation keyed
    bool has_scale = true;      // Scale keyed
    
    TransformKeyframe() = default;
    
    TransformKeyframe(const Vec3& pos, const Vec3& rot, const Vec3& scl)
        : position(pos), rotation(rot), scale(scl) {}
    
    // Linear interpolation - respects per-channel flags
    static TransformKeyframe lerp(const TransformKeyframe& a, const TransformKeyframe& b, float t) {
        TransformKeyframe result;
        
        // Only interpolate channels that are keyed in both keyframes
        if (a.has_position && b.has_position) {
            result.position = a.position + (b.position - a.position) * t;
            result.has_position = true;
        } else if (a.has_position) {
            result.position = a.position;
            result.has_position = true;
        } else if (b.has_position) {
            result.position = b.position;
            result.has_position = true;
        } else {
            result.has_position = false;
        }
        
        if (a.has_rotation && b.has_rotation) {
            result.rotation = a.rotation + (b.rotation - a.rotation) * t;
            result.has_rotation = true;
        } else if (a.has_rotation) {
            result.rotation = a.rotation;
            result.has_rotation = true;
        } else if (b.has_rotation) {
            result.rotation = b.rotation;
            result.has_rotation = true;
        } else {
            result.has_rotation = false;
        }
        
        if (a.has_scale && b.has_scale) {
            result.scale = a.scale + (b.scale - a.scale) * t;
            result.has_scale = true;
        } else if (a.has_scale) {
            result.scale = a.scale;
            result.has_scale = true;
        } else if (b.has_scale) {
            result.scale = b.scale;
            result.has_scale = true;
        } else {
            result.has_scale = false;
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
    WorldKeyframe world;        // NEW
    
    // Flags to track what's keyframed
    bool has_transform = false;
    bool has_material = false;
    bool has_light = false;
    bool has_camera = false;
    bool has_world = false;     // NEW
    
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

        // Lambda to find previous valid keyframe for a channel
        auto findPrev = [&](bool (Keyframe::*has_flag)) -> const Keyframe* {
            // Start from the keyframe occurring at or before current_frame
            // lower_bound gives first element >= value.
            // So we want the one before that, or that one if equal?
            // Let's iterate backwards.
            for (auto it = keyframes.rbegin(); it != keyframes.rend(); ++it) {
                if (it->frame <= current_frame) {
                    if ((*it).*has_flag) return &(*it);
                }
            }
            return nullptr;
        };

        // Lambda to find next valid keyframe for a channel
        auto findNext = [&](bool (Keyframe::*has_flag)) -> const Keyframe* {
             for (auto it = keyframes.begin(); it != keyframes.end(); ++it) {
                if (it->frame >= current_frame) {
                    if ((*it).*has_flag) return &(*it);
                }
            }
            return nullptr;
        };
        
        // --- TRANSFORM CHANNEL ---
        const Keyframe* prev_trans = findPrev(&Keyframe::has_transform);
        const Keyframe* next_trans = findNext(&Keyframe::has_transform);
        
        if (prev_trans && next_trans) {
             if (prev_trans == next_trans) {
                 result.transform = prev_trans->transform;
             } else {
                 float t = float(current_frame - prev_trans->frame) / float(next_trans->frame - prev_trans->frame);
                 result.transform = TransformKeyframe::lerp(prev_trans->transform, next_trans->transform, t);
             }
             result.has_transform = true;
        } else if (prev_trans) {
            result.transform = prev_trans->transform;
            result.has_transform = true;
        } else if (next_trans) {
            result.transform = next_trans->transform;
            result.has_transform = true;
        }

        // --- MATERIAL CHANNEL ---
        const Keyframe* prev_mat = findPrev(&Keyframe::has_material);
        const Keyframe* next_mat = findNext(&Keyframe::has_material);
        
        if (prev_mat && next_mat) {
             if (prev_mat == next_mat) {
                 result.material = prev_mat->material;
             } else {
                 float t = float(current_frame - prev_mat->frame) / float(next_mat->frame - prev_mat->frame);
                 result.material = MaterialKeyframe::lerp(prev_mat->material, next_mat->material, t);
             }
             result.has_material = true;
        } else if (prev_mat) {
            result.material = prev_mat->material;
            result.has_material = true;
        } else if (next_mat) {
            result.material = next_mat->material;
            result.has_material = true;
        }

        // --- LIGHT CHANNEL ---
        const Keyframe* prev_light = findPrev(&Keyframe::has_light);
        const Keyframe* next_light = findNext(&Keyframe::has_light);
        if (prev_light && next_light) {
            if (prev_light == next_light) result.light = prev_light->light;
            else {
                float t = float(current_frame - prev_light->frame) / float(next_light->frame - prev_light->frame);
                result.light = LightKeyframe::lerp(prev_light->light, next_light->light, t);
            }
            result.has_light = true;
        } else if (prev_light) { result.light = prev_light->light; result.has_light = true; }
        else if (next_light) { result.light = next_light->light; result.has_light = true; }

        // --- CAMERA CHANNEL ---
        const Keyframe* prev_cam = findPrev(&Keyframe::has_camera);
        const Keyframe* next_cam = findNext(&Keyframe::has_camera);
        if (prev_cam && next_cam) {
            if (prev_cam == next_cam) result.camera = prev_cam->camera;
            else {
                float t = float(current_frame - prev_cam->frame) / float(next_cam->frame - prev_cam->frame);
                result.camera = CameraKeyframe::lerp(prev_cam->camera, next_cam->camera, t);
            }
            result.has_camera = true;
        } else if (prev_cam) { result.camera = prev_cam->camera; result.has_camera = true; }
        else if (next_cam) { result.camera = next_cam->camera; result.has_camera = true; }

        // --- WORLD CHANNEL ---
        const Keyframe* prev_world = findPrev(&Keyframe::has_world);
        const Keyframe* next_world = findNext(&Keyframe::has_world);
        if (prev_world && next_world) {
            if (prev_world == next_world) result.world = prev_world->world;
            else {
                float t = float(current_frame - prev_world->frame) / float(next_world->frame - prev_world->frame);
                result.world = WorldKeyframe::lerp(prev_world->world, next_world->world, t);
            }
            result.has_world = true;
        } else if (prev_world) { result.world = prev_world->world; result.has_world = true; }
        else if (next_world) { result.world = next_world->world; result.has_world = true; }

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
};
