#include "InstanceGroup.h"
#include "Triangle.h"
#include <random>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════════
// SCATTER SOURCE
// ═══════════════════════════════════════════════════════════════════════════════

ScatterSource::ScatterSource(const std::string& n, const std::vector<std::shared_ptr<Triangle>>& tris)
    : name(n), triangles(tris) {
    computeCenter();
}

void ScatterSource::computeCenter() {
    mesh_center = Vec3(0, 0, 0);
    int vertex_count = 0;
    
    for (const auto& tri : triangles) {
        mesh_center = mesh_center + tri->getV0();
        mesh_center = mesh_center + tri->getV1();
        mesh_center = mesh_center + tri->getV2();
        vertex_count += 3;
    }
    
    if (vertex_count > 0) {
        mesh_center = mesh_center / (float)vertex_count;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INSTANCE TRANSFORM
// ═══════════════════════════════════════════════════════════════════════════════

Matrix4x4 InstanceTransform::toMatrix() const {
    Matrix4x4 result;
    
    // Build TRS matrix: Translation * Rotation * Scale
    float cx = cosf(rotation.x * 3.14159f / 180.0f);
    float sx = sinf(rotation.x * 3.14159f / 180.0f);
    float cy = cosf(rotation.y * 3.14159f / 180.0f);
    float sy = sinf(rotation.y * 3.14159f / 180.0f);
    float cz = cosf(rotation.z * 3.14159f / 180.0f);
    float sz = sinf(rotation.z * 3.14159f / 180.0f);
    
    // Rotation matrix (Y * X * Z order - typical for games)
    result.m[0][0] = (cy * cz + sy * sx * sz) * scale.x;
    result.m[0][1] = (cz * sy * sx - cy * sz) * scale.x;
    result.m[0][2] = (cx * sy) * scale.x;
    result.m[0][3] = position.x;
    
    result.m[1][0] = (cx * sz) * scale.y;
    result.m[1][1] = (cx * cz) * scale.y;
    result.m[1][2] = (-sx) * scale.y;
    result.m[1][3] = position.y;
    
    result.m[2][0] = (cy * sx * sz - cz * sy) * scale.z;
    result.m[2][1] = (cy * cz * sx + sy * sz) * scale.z;
    result.m[2][2] = (cx * cy) * scale.z;
    result.m[2][3] = position.z;
    
    result.m[3][0] = 0.0f;
    result.m[3][1] = 0.0f;
    result.m[3][2] = 0.0f;
    result.m[3][3] = 1.0f;
    
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// INSTANCE GROUP
// ═══════════════════════════════════════════════════════════════════════════════

void InstanceGroup::addInstance(const InstanceTransform& transform) {
    instances.push_back(transform);
    gpu_dirty = true;
}

void InstanceGroup::removeInstancesInRadius(const Vec3& center, float radius) {
    float radius_sq = radius * radius;
    
    auto it = std::remove_if(instances.begin(), instances.end(),
        [&center, radius_sq](const InstanceTransform& t) {
            float dx = t.position.x - center.x;
            float dy = t.position.y - center.y;
            float dz = t.position.z - center.z;
            return (dx*dx + dy*dy + dz*dz) < radius_sq;
        });
    
    if (it != instances.end()) {
        instances.erase(it, instances.end());
        gpu_dirty = true;
    }
}

void InstanceGroup::clearInstances() {
    instances.clear();
    tlas_instance_ids.clear();
    gpu_dirty = true;
}

InstanceTransform InstanceGroup::generateRandomTransform(const Vec3& position, const Vec3& normal) const {
    // Thread-local RNG for performance
    thread_local std::mt19937 rng(std::random_device{}());
    
    InstanceTransform t;
    t.position = position;
    
    // Random scale
    std::uniform_real_distribution<float> scale_dist(brush_settings.scale_min, brush_settings.scale_max);
    float uniform_scale = scale_dist(rng);
    t.scale = Vec3(uniform_scale, uniform_scale, uniform_scale);
    
    // Random Y rotation
    std::uniform_real_distribution<float> rot_y_dist(0.0f, brush_settings.rotation_random_y);
    t.rotation.y = rot_y_dist(rng);
    
    // Random tilt (X/Z rotation)
    if (brush_settings.rotation_random_xz > 0.0f) {
        std::uniform_real_distribution<float> tilt_dist(-brush_settings.rotation_random_xz, brush_settings.rotation_random_xz);
        t.rotation.x = tilt_dist(rng);
        t.rotation.z = tilt_dist(rng);
    }
    
    // Align to surface normal
    if (brush_settings.align_to_normal && normal.length() > 0.01f) {
        Vec3 up(0, 1, 0);
        Vec3 n = normal.normalize();
        
        // Calculate rotation to align up vector with normal
        // Using quaternion would be better, but for simplicity:
        float influence = brush_settings.normal_influence;
        Vec3 target_up = up * (1.0f - influence) + n * influence;
        target_up = target_up.normalize();
        
        // Convert to euler (simplified - just pitch and roll)
        t.rotation.x += asinf(-target_up.z) * 180.0f / 3.14159f * influence;
        t.rotation.z += asinf(target_up.x) * 180.0f / 3.14159f * influence;
    }
    
    return t;
}
