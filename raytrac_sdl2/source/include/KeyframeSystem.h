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
    // Block 1: Albedo + opacity (matches GpuMaterial)
    Vec3 albedo = Vec3(0.8, 0.8, 0.8);    // base color
    float opacity = 1.0f;                   // alpha
    
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
    
    // Extra CPU-only properties (not in GpuMaterial)
    float specular = 0.5f;
    float specular_tint = 0.0f;
    float clearcoat_roughness = 0.0f;
    float normal_strength = 1.0f;
    float emission_strength = 1.0f;
    
    // Constructor from PrincipledBSDF
    // Constructor from GpuMaterial (GPU/CPU compatible)
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
        
        // CPU-only extras - not in GpuMaterial, use defaults
        specular = 0.5f;
        specular_tint = 0.0f;
        clearcoat_roughness = 0.1f;
        normal_strength = 1.0f;
        emission_strength = 1.0f;
    }
    
    // Apply to GpuMaterial
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
    
    // Linear interpolation between two material keyframes
    static MaterialKeyframe lerp(const MaterialKeyframe& a, const MaterialKeyframe& b, float t) {
        MaterialKeyframe result;
        // Manual Vec3 interpolation
        result.albedo = a.albedo + (b.albedo - a.albedo) * t;
        result.opacity = a.opacity + (b.opacity - a.opacity) * t;
        result.roughness = a.roughness + (b.roughness - a.roughness) * t;
        result.metallic = a.metallic + (b.metallic - a.metallic) * t;
        result.clearcoat = a.clearcoat + (b.clearcoat - a.clearcoat) * t;
        result.transmission = a.transmission + (b.transmission - a.transmission) * t;
        result.emission = a.emission + (b.emission - a.emission) * t;
        result.ior = a.ior + (b.ior - a.ior) * t;
        result.subsurface_color = a.subsurface_color + (b.subsurface_color - a.subsurface_color) * t;
        result.subsurface = a.subsurface + (b.subsurface - a.subsurface) * t;
        result.anisotropic = a.anisotropic + (b.anisotropic - a.anisotropic) * t;
        result.sheen = a.sheen + (b.sheen - a.sheen) * t;
        result.sheen_tint = a.sheen_tint + (b.sheen_tint - a.sheen_tint) * t;
        
        // CPU-only extras
        result.specular = a.specular + (b.specular - a.specular) * t;
        result.specular_tint = a.specular_tint + (b.specular_tint - a.specular_tint) * t;
        result.clearcoat_roughness = a.clearcoat_roughness + (b.clearcoat_roughness - a.clearcoat_roughness) * t;
        result.normal_strength = a.normal_strength + (b.normal_strength - a.normal_strength) * t;
        result.emission_strength = a.emission_strength + (b.emission_strength - a.emission_strength) * t;
        return result;
    }
};

// ============================================================================
// LIGHT KEYFRAME - Position, color, intensity, direction
// ============================================================================
struct LightKeyframe {
    Vec3 position = Vec3(0, 0, 0);
    Vec3 color = Vec3(1, 1, 1);
   float intensity = 1.0f;
    Vec3 direction = Vec3(0, -1, 0);  // For directional/spot lights
    
    LightKeyframe() = default;
    
    static LightKeyframe lerp(const LightKeyframe& a, const LightKeyframe& b, float t) {
        LightKeyframe result;
        result.position = a.position + (b.position - a.position) * t;
        result.color = a.color + (b.color - a.color) * t;
        result.intensity = a.intensity + (b.intensity - a.intensity) * t;
        result.direction = a.direction + (b.direction - a.direction) * t;
        return result;
    }
};

// ============================================================================
// CAMERA KEYFRAME - Position, target, FOV, DOF
// ============================================================================
struct CameraKeyframe {
    Vec3 position = Vec3(0, 0, 0);
    Vec3 target = Vec3(0, 0, -1);
    float fov = 40.0f;
    float focus_distance = 10.0f;
    float lens_radius = 0.0f;
    
    CameraKeyframe() = default;
    
    static CameraKeyframe lerp(const CameraKeyframe& a, const CameraKeyframe& b, float t) {
        CameraKeyframe result;
        result.position = a.position + (b.position - a.position) * t;
        result.target = a.target + (b.target - a.target) * t;
        result.fov = a.fov + (b.fov - a.fov) * t;
        result.focus_distance = a.focus_distance + (b.focus_distance - a.focus_distance) * t;
        result.lens_radius = a.lens_radius + (b.lens_radius - a.lens_radius) * t;
        return result;
    }
};

// ============================================================================
// WORLD KEYFRAME - Background, Nishita sky, volumetric settings
// ============================================================================
struct WorldKeyframe {
    Vec3 background_color = Vec3(0.5, 0.7, 1.0);
    float background_strength = 1.0f;
    
    // Nishita sky parameters
    float sun_elevation = 15.0f;
    float sun_rotation = 0.0f;
    float sun_intensity = 1.0f;
    float air_density = 1.0f;
    float dust_density = 1.0f;
    
    // Volumetric settings
    float cloud_density = 0.5f;
    float cloud_coverage = 0.5f;
    
    WorldKeyframe() = default;
    
    static WorldKeyframe lerp(const WorldKeyframe& a, const WorldKeyframe& b, float t) {
        WorldKeyframe result;
        result.background_color = a.background_color + (b.background_color - a.background_color) * t;
        result.background_strength = a.background_strength + (b.background_strength - a.background_strength) * t;
        result.sun_elevation = a.sun_elevation + (b.sun_elevation - a.sun_elevation) * t;
        result.sun_rotation = a.sun_rotation + (b.sun_rotation - a.sun_rotation) * t;
        result.sun_intensity = a.sun_intensity + (b.sun_intensity - a.sun_intensity) * t;
        result.air_density = a.air_density + (b.air_density - a.air_density) * t;
        result.dust_density = a.dust_density + (b.dust_density - a.dust_density) * t;
        result.cloud_density = a.cloud_density + (b.cloud_density - a.cloud_density) * t;
        result.cloud_coverage = a.cloud_coverage + (b.cloud_coverage - a.cloud_coverage) * t;
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
        
        // If keyframe already exists at this frame, replace it
        if (it != keyframes.end() && it->frame == kf.frame) {
            *it = kf;
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
    
    // Evaluate animation at given frame (with interpolation)
    Keyframe evaluate(int current_frame) const {
        if (keyframes.empty()) return Keyframe(current_frame);
        
        // Before first keyframe - use first keyframe
        if (current_frame <= keyframes.front().frame) {
            return keyframes.front();
        }
        
        // After last keyframe - use last keyframe
        if (current_frame >= keyframes.back().frame) {
            return keyframes.back();
        }
        
        // Find surrounding keyframes
        for (size_t i = 0; i < keyframes.size() - 1; ++i) {
            if (current_frame >= keyframes[i].frame && current_frame <= keyframes[i + 1].frame) {
                const Keyframe& kf1 = keyframes[i];
                const Keyframe& kf2 = keyframes[i + 1];
                
                // Calculate interpolation factor
                float t = float(current_frame - kf1.frame) / float(kf2.frame - kf1.frame);
                
                // Interpolate
                Keyframe result;
                result.frame = current_frame;
                
                if (kf1.has_transform && kf2.has_transform) {
                    result.transform = TransformKeyframe::lerp(kf1.transform, kf2.transform, t);
                    result.has_transform = true;
                }
                
                if (kf1.has_material && kf2.has_material) {
                    result.material = MaterialKeyframe::lerp(kf1.material, kf2.material, t);
                    result.has_material = true;
                }
                
                if (kf1.has_light && kf2.has_light) {
                    result.light = LightKeyframe::lerp(kf1.light, kf2.light, t);
                    result.has_light = true;
                }
                
                if (kf1.has_camera && kf2.has_camera) {
                    result.camera = CameraKeyframe::lerp(kf1.camera, kf2.camera, t);
                    result.has_camera = true;
                }
                
                if (kf1.has_world && kf2.has_world) {
                    result.world = WorldKeyframe::lerp(kf1.world, kf2.world, t);
                    result.has_world = true;
                }
                
                return result;
            }
        }
        
        return keyframes.back();
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
