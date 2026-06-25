/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          InstanceGroup.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include "Vec3.h"
#include "Matrix4x4.h"
#include "AABB.h"
#include <vector>
#include <string>
#include <memory>

class Triangle;
class Hittable; // Forward declaration

// ═══════════════════════════════════════════════════════════════════════════════
// INSTANCE GROUP - Represents a group of instanced objects (e.g., grass, trees)
// ═══════════════════════════════════════════════════════════════════════════════

struct InstanceTransform {
    Vec3 position;
    Vec3 rotation;  // Euler angles (degrees)
    Vec3 scale;
    
    InstanceTransform() : position(0), rotation(0), scale(1) {}
    InstanceTransform(const Vec3& pos, const Vec3& rot, const Vec3& scl)
        : position(pos), rotation(rot), scale(scl) {}
    
    // Convert to 4x4 matrix
    Matrix4x4 toMatrix() const;
    
    // Multi-Source Support
    int source_index = 0;
};

// ═══════════════════════════════════════════════════════════════════════════════
// SCATTER SOURCE - A single mesh that can be scattered with a weight
// ═══════════════════════════════════════════════════════════════════════════════

struct ScatterSource {
    std::string name;                               // Source name (mesh node name)
    std::vector<std::shared_ptr<Triangle>> triangles;  // Source triangles
    Vec3 mesh_center;                               // Pre-computed center
    float weight = 1.0f;                            // Selection probability weight (0.0 - 1.0)
    
    // Runtime Instancing Data
    std::shared_ptr<Hittable> bvh; 
    std::shared_ptr<std::vector<std::shared_ptr<Triangle>>> centered_triangles_ptr;
    AABB local_bbox;
    bool has_local_bbox = false;
    
    // Per-Source Appearance Settings
    struct SourceSettings {
        float scale_min = 0.8f;
        float scale_max = 1.2f;
        float rotation_random_y = 360.0f;
        float rotation_random_xz = 5.0f;
        float y_offset_min = 0.0f;
        float y_offset_max = 0.0f;
        bool align_to_normal = true;
        float normal_influence = 0.8f;
        float wind_strength_scale = 1.0f;
        float wind_speed_scale = 1.0f;
        float wind_turbulence_scale = 1.0f;
        float wind_bend_limit_scale = 1.0f;
        float wind_phase_offset = 0.0f;
    };
    SourceSettings settings;

    ScatterSource() = default;
    ScatterSource(const std::string& n, const std::vector<std::shared_ptr<Triangle>>& tris);

    void computeCenter();                           // Calculate mesh center from triangles
};

// ═══════════════════════════════════════════════════════════════════════════════
// MESH SURFACE SAMPLER - Area-weighted uniform sampling on arbitrary mesh
// ═══════════════════════════════════════════════════════════════════════════════
// Usage:
//   MeshSurfaceSampler mss;
//   mss.build(triangles);
//   auto s = mss.sample(rng);   // s.position, s.normal
// ─────────────────────────────────────────────────────────────────────────────
// normal_influence in BrushSettings controls orientation:
//   0.0 = always world-up  (buildings, large rocks)
//   1.0 = full face normal (foliage, grass, ivy)
//   0.x = blend            (medium rocks, props)
// ═══════════════════════════════════════════════════════════════════════════════

struct MeshSurfaceSampler {
    struct Sample {
        Vec3 position;
        Vec3 normal;    // Face normal, flipped to face upward hemisphere
    };

    // Build area-weighted CDF from triangle list. Call once before sampling.
    void build(const std::vector<std::shared_ptr<Triangle>>& triangles);

    // Draw a uniformly-distributed random sample from the surface.
    Sample sample(std::mt19937& rng) const;

    bool  isValid()    const { return !cdf.empty() && source_tris != nullptr; }
    float totalArea()  const { return total_area; }

private:
    std::vector<float>                                  cdf;         // Normalised cumulative area [0..1]
    float                                               total_area = 0.f;
    const std::vector<std::shared_ptr<Triangle>>*       source_tris = nullptr;
};

// ═══════════════════════════════════════════════════════════════════════════════
// EXTENDED INSTANCE TRANSFORM - Includes source index for multi-source
// ═══════════════════════════════════════════════════════════════════════════════

struct InstanceTransformEx : public InstanceTransform {
    int source_index = 0;                           // Which source mesh this instance uses
    
    InstanceTransformEx() = default;
    InstanceTransformEx(const Vec3& pos, const Vec3& rot, const Vec3& scl, int src = 0)
        : InstanceTransform(pos, rot, scl), source_index(src) {}
};

struct InstanceGroup {
    // Identity
    std::string name;                               // Group name (e.g., "Forest_01")
    int id = -1;                                    // Unique group ID

    // Transient groups are runtime-only render bridges (e.g. particle systems
    // mirroring their live SoA into instances each step). They are consumed by
    // the RT backends exactly like foliage groups, but are NEVER serialized and
    // are hidden from the foliage UI / brush tools. The owning system rebuilds
    // their instances every frame, so persisting them would be meaningless.
    bool transient = false;
    
    // Multi-Source Meshes (NEW)
    std::vector<ScatterSource> sources;             // Multiple source meshes with weights
    
    // Legacy single-source (backward compatibility)
    std::string source_node_name;                   // Primary source node name
    std::vector<std::shared_ptr<Triangle>> source_triangles;  // Primary source triangles (Keep for now)
    
    // Instancing Support
    std::shared_ptr<Hittable> source_bvh; // CPU BVH for raytracing source
    std::shared_ptr<std::vector<std::shared_ptr<Triangle>>> source_triangles_ptr; // Shared ptr for passing to HittableInstance
    
    // Instances
    std::vector<InstanceTransform> instances;   // All instance transforms

    // Point-sphere render mode (foam / fluid splat). When true, the OptiX
    // backend does NOT emit one TLAS instance per InstanceTransform; instead it
    // partitions the instances by source_index (foam type) and builds ONE
    // analytic sphere GAS per source — the whole set is 3 TLAS instances, not N.
    // instances[i].position = sphere centre (world), scale.x*0.5 = radius,
    // source_index = which source/material (sphere GAS) it belongs to. Other
    // backends (Vulkan / CPU) ignore this flag and keep the InstanceTransform
    // path, so foam still renders there via the icosphere primitive.
    bool point_sphere_mode = false;

    // Targeting: allow scattering on TERRAIN (default) or on a specific MESH node
    enum class TargetType {
        TERRAIN = 0,
        MESH = 1
    };

    TargetType target_type = TargetType::TERRAIN;
    std::string target_node_name; // If target_type == MESH, the scene node name to sample surface from
    std::string mesh_fingerprint; // Optional mesh fingerprint for determinism checks
    
    // GPU State
    int blas_id = -1;                           // GPU BLAS index
    std::vector<int> tlas_instance_ids;         // TLAS instance IDs for each instance
    bool gpu_dirty = true;                      // Needs GPU sync
    
    // Brush Settings (stored per-group for different foliage types)
    struct BrushSettings {
        float density = 1.0f;                   // Instances per m² (used for brush)
        int target_count = 1000;                // Total count for procedural scatter
        int seed = 1234;                        // Random seed for reproducibility
        float min_distance = 0.5f;              // Minimum distance between instances
        
        // Splat Map Settings
        int splat_map_channel = -1;             // -1 = None, 0=R, 1=G, 2=B, 3=A
        int exclusion_channel = -1;             // Channel to exclude (mask out)
        float exclusion_threshold = 0.5f;       // Value above which placement is prevented (For painting/spawning)
        float slope_max = 45.0f;                // Max slope angle
        float height_min = -10.0f;
        float height_max = 10.0f;
        float curvature_min = -2.0f;            // Threshold between Ridge and Flat
        float curvature_max = 2.0f;             // Threshold between Flat and Gully
        int curvature_step = 1;                 // Feature Scale (Step size for Laplacian)
        
        // Advanced Curvature Filtering
        bool allow_ridges = true;               // Allow points with curv < curvature_min
        bool allow_flats = true;                // Allow points between [min, max]
        bool allow_gullies = true;              // Allow points with curv > curvature_max
        
        float slope_direction_angle = 0.0f;     // Preferred direction (degrees, 0 = North/Z+)
        float slope_direction_influence = 0.0f; // 0 = Ignore direction, 1 = Strict placement only in direction
        
        // Global overrides (if needed, but per-source is preferred)
        bool use_global_settings = false;       // If true, overrides source settings
        float scale_min = 0.8f;
        float scale_max = 1.2f;
        float rotation_random_y = 360.0f;       // Random Y rotation (degrees)
        float rotation_random_xz = 15.0f;       // Random tilt (degrees)
        float y_offset_min = 0.0f;
        float y_offset_max = 0.0f;
        bool align_to_normal = true;            // Align Y-axis to surface normal
        float normal_influence = 1.0f;          // 0=world up, 1=full surface normal
    };
    BrushSettings brush_settings;

    // Wind Animation Settings
    struct WindSettings {
        bool enabled = false;
        float speed = 1.0f;             // Global speed multiplier
        float strength = 0.1f;          // Max shear amount
        Vec3 direction = Vec3(1,0,0);   // Primary wind direction
        float turbulence = 1.5f;        // Noise frequency/randomness
        float wave_size = 50.0f;        // Distance between wind waves
        bool use_source_profiles = true;
        bool allow_gpu_deform = true;
        float gpu_deform_max_distance = 35.0f; // CUDA only if the whole group stays near camera
        int gpu_deform_max_instances = 32;     // CUDA budget for hero foliage groups
    };
    WindSettings wind_settings;

    // Backup of original transforms (Rest Pose) to prevent drift
    std::vector<InstanceTransform> initial_instances;

    // Wind Animation Update
    void updateWind(float time); // Updates active_hittables transforms

    // Runtime Link to Scene Objects (for fast updates)
    // Used by updateWind to push changes to HittableInstances in scene.world
    std::vector<std::weak_ptr<Hittable>> active_hittables;

    
    // Statistics
    size_t getInstanceCount() const { return instances.size(); }
    size_t getTriangleCount() const {
        if (!sources.empty()) {
            size_t total = 0;
            for (const auto& inst : instances) {
                int src_idx = inst.source_index;
                if (src_idx < 0 || src_idx >= static_cast<int>(sources.size())) {
                    src_idx = 0;
                }

                const auto& source = sources[static_cast<size_t>(src_idx)];
                if (source.centered_triangles_ptr) {
                    total += source.centered_triangles_ptr->size();
                } else {
                    total += source.triangles.size();
                }
            }
            return total;
        }

        const size_t source_count = source_triangles_ptr
            ? source_triangles_ptr->size()
            : source_triangles.size();
        return source_count * instances.size();
    }
    
    // Operations
    void addInstance(const InstanceTransform& transform);
    void removeInstancesInRadius(const Vec3& center, float radius);
    void clearInstances();
    
    // Generate random transform based on brush settings
    InstanceTransform generateRandomTransform(const Vec3& position, const Vec3& normal) const;
};

