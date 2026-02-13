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
    };
    SourceSettings settings;

    ScatterSource() = default;
    ScatterSource(const std::string& n, const std::vector<std::shared_ptr<Triangle>>& tris);
    
    void computeCenter();                           // Calculate mesh center from triangles
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
    size_t getTriangleCount() const { return source_triangles.size() * instances.size(); }
    
    // Operations
    void addInstance(const InstanceTransform& transform);
    void removeInstancesInRadius(const Vec3& center, float radius);
    void clearInstances();
    
    // Generate random transform based on brush settings
    InstanceTransform generateRandomTransform(const Vec3& position, const Vec3& normal) const;
};

