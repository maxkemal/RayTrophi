#pragma once

#include "Vec3.h"
#include "Matrix4x4.h"
#include <vector>
#include <string>
#include <memory>

class Triangle;

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
};

// ═══════════════════════════════════════════════════════════════════════════════
// SCATTER SOURCE - A single mesh that can be scattered with a weight
// ═══════════════════════════════════════════════════════════════════════════════

struct ScatterSource {
    std::string name;                               // Source name (mesh node name)
    std::vector<std::shared_ptr<Triangle>> triangles;  // Source triangles
    Vec3 mesh_center;                               // Pre-computed center
    float weight = 1.0f;                            // Selection probability weight (0.0 - 1.0)
    
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
    std::vector<std::shared_ptr<Triangle>> source_triangles;  // Primary source triangles
    
    // Instances
    std::vector<InstanceTransform> instances;   // All instance transforms
    
    // GPU State
    int blas_id = -1;                           // GPU BLAS index
    std::vector<int> tlas_instance_ids;         // TLAS instance IDs for each instance
    bool gpu_dirty = true;                      // Needs GPU sync
    
    // Brush Settings (stored per-group for different foliage types)
    struct BrushSettings {
        float density = 1.0f;                   // Instances per m²
        float scale_min = 0.8f;
        float scale_max = 1.2f;
        float rotation_random_y = 360.0f;       // Random Y rotation (degrees)
        float rotation_random_xz = 15.0f;       // Random tilt (degrees)
        bool align_to_normal = true;            // Align Y-axis to surface normal
        float normal_influence = 1.0f;          // 0=world up, 1=full surface normal
    };
    BrushSettings brush_settings;
    
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
