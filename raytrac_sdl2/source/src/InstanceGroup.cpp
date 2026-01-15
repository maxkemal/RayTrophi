#include "InstanceGroup.h"
#include "Triangle.h"
#include "HittableInstance.h" // Added for wind update
#include <random>
#include <algorithm>
#include "globals.h" // For SCENE_LOG_INFO

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
    initial_instances.push_back(transform); // Store Rest Pose
    gpu_dirty = true;
}

void InstanceGroup::removeInstancesInRadius(const Vec3& center, float radius) {
    float radius_sq = radius * radius;
    
    // Synchronized compaction to keep instances and initial_instances in sync
    size_t write_idx = 0;
    bool has_initial = initial_instances.size() == instances.size();

    for (size_t read_idx = 0; read_idx < instances.size(); ++read_idx) {
        const auto& t = instances[read_idx];
        float dx = t.position.x - center.x;
        float dy = t.position.y - center.y;
        float dz = t.position.z - center.z;
        
        // Condition to KEEP: Distance >= Radius (Outside circle)
        if ((dx*dx + dy*dy + dz*dz) >= radius_sq) {
            if (write_idx != read_idx) {
                instances[write_idx] = instances[read_idx];
                if (has_initial) initial_instances[write_idx] = initial_instances[read_idx];
            }
            write_idx++;
        }
    }
    
    if (write_idx < instances.size()) {
        instances.resize(write_idx);
        if (has_initial) initial_instances.resize(write_idx);
        gpu_dirty = true;
    }
}

void InstanceGroup::clearInstances() {
    instances.clear();
    initial_instances.clear();
    tlas_instance_ids.clear();
    gpu_dirty = true;
}

InstanceTransform InstanceGroup::generateRandomTransform(const Vec3& position, const Vec3& normal) const {
    // Thread-local RNG for performance
    thread_local std::mt19937 rng(std::random_device{}());
    
    InstanceTransform t;
    t.position = position;

    // 1. SELECT SOURCE FIRST
    if (!sources.empty()) {
        if (sources.size() == 1) {
            t.source_index = 0;
        } else {
            float total_weight = 0.0f;
            for (const auto& src : sources) total_weight += src.weight;
            
            if (total_weight <= 0.001f) {
                std::uniform_int_distribution<int> idx_dist(0, (int)sources.size() - 1);
                t.source_index = idx_dist(rng);
            } else {
                std::uniform_real_distribution<float> w_dist(0.0f, total_weight);
                float r = w_dist(rng);
                float current_w = 0.0f;
                int selected_idx = 0;
                for (int i = 0; i < (int)sources.size(); ++i) {
                    current_w += sources[i].weight;
                    if (r <= current_w) {
                        selected_idx = i;
                        break;
                    }
                }
                t.source_index = selected_idx;
            }
        }
    } else {
        t.source_index = 0;
    }
    
    // 2. DETERMINE SETTINGS (Global or Local)
    float scale_min, scale_max;
    float rot_y, rot_xz;
    float y_off_min, y_off_max;
    bool align;
    float normal_inf;
    
    // Use local settings if available and not overridden
    if (!brush_settings.use_global_settings && !sources.empty() && t.source_index < sources.size()) {
        const auto& set = sources[t.source_index].settings;
        scale_min = set.scale_min;
        scale_max = set.scale_max;
        rot_y = set.rotation_random_y;
        rot_xz = set.rotation_random_xz;
        y_off_min = set.y_offset_min;
        y_off_max = set.y_offset_max;
        align = set.align_to_normal;
        normal_inf = set.normal_influence;
    } else {
        // Fallback to global brush settings
        scale_min = brush_settings.scale_min;
        scale_max = brush_settings.scale_max;
        rot_y = brush_settings.rotation_random_y;
        rot_xz = brush_settings.rotation_random_xz;
        y_off_min = brush_settings.y_offset_min;
        y_off_max = brush_settings.y_offset_max;
        align = brush_settings.align_to_normal;
        normal_inf = brush_settings.normal_influence;
    }
    
    // 3. GENERATE TRANSFORM
    
    // Random scale
    std::uniform_real_distribution<float> scale_dist(scale_min, scale_max);
    float uniform_scale = scale_dist(rng);
    t.scale = Vec3(uniform_scale, uniform_scale, uniform_scale);
    
    // Random Y rotation
    std::uniform_real_distribution<float> rot_y_dist(0.0f, rot_y);
    t.rotation.y = rot_y_dist(rng);
    
    // Random tilt (X/Z rotation)
    if (rot_xz > 0.0f) {
        std::uniform_real_distribution<float> tilt_dist(-rot_xz, rot_xz);
        t.rotation.x = tilt_dist(rng);
        t.rotation.z = tilt_dist(rng);
    }
    
    // Generate random Y offset
    if (y_off_min != 0.0f || y_off_max != 0.0f) {
        std::uniform_real_distribution<float> offset_dist(y_off_min, y_off_max);
        t.position.y += offset_dist(rng);
    }
    
    // Align to surface normal
    if (align && normal.length() > 0.01f) {
        Vec3 up(0, 1, 0);
        Vec3 n = normal.normalize();
        
        float influence = normal_inf;
        Vec3 target_up = up * (1.0f - influence) + n * influence;
        target_up = target_up.normalize();
        
        t.rotation.x += asinf(-target_up.z) * 180.0f / 3.14159f * influence;
        t.rotation.z += asinf(target_up.x) * 180.0f / 3.14159f * influence;
    }

    return t;
}

void InstanceGroup::updateWind(float time) {
    if (!wind_settings.enabled) return;

    // Wind parameters
    float t = time * wind_settings.speed;
    // Normalize direction
    Vec3 dir = wind_settings.direction;
    float len = dir.length();
    if (len > 0.001f) dir = dir / len;
    else return;

    float strength = wind_settings.strength;
    float turbulence = wind_settings.turbulence;
    float wave_size = wind_settings.wave_size;
    if (wave_size < 0.1f) wave_size = 0.1f;

    // Rotate around axis perpendicular to wind and UP
    Vec3 up(0, 1, 0);
    Vec3 axis = dir.cross(up);
    float axis_len = axis.length();
    if (axis_len < 0.001f) {
        // Wind is vertical? Use X axis fallback
        axis = Vec3(1, 0, 0);
    } else {
        axis = axis / axis_len;
    }

    // Precompute axis-angle rotation math constants
    float ax = axis.x, ay = axis.y, az = axis.z;

    // Iterate
    size_t count = std::min(instances.size(), active_hittables.size());
    if (initial_instances.size() < count) count = initial_instances.size(); 

    for (size_t i = 0; i < count; ++i) {
        if (active_hittables[i].expired()) continue;
        
        auto hittable_ptr = active_hittables[i].lock();
        if (!hittable_ptr) continue;

        auto inst = std::dynamic_pointer_cast<HittableInstance>(hittable_ptr);
        if (!inst) continue;

        const auto& initial = initial_instances[i];
        
        // Compute Sway Phase based on position
        float phase = (initial.position.x * dir.x + initial.position.z * dir.z) * (1.0f / wave_size);
        
        // Multi-frequency noise for natural movement
        float noise = sinf(t + phase) + sinf(t * turbulence + phase * 2.5f) * 0.5f;
        
        // Calculate deflection angle (degrees)
        float angle = noise * strength;

        // --- Build Rotation Matrix (Axis-Angle) ---
        float rad = angle * 3.14159f / 180.0f;
        float c = cosf(rad);
        float s = sinf(rad);
        float t_val = 1.0f - c;

        Matrix4x4 swayMat;
        swayMat.m[0][0] = t_val*ax*ax + c;    
        swayMat.m[0][1] = t_val*ax*ay - az*s; 
        swayMat.m[0][2] = t_val*ax*az + ay*s; 
        swayMat.m[0][3] = 0;
        swayMat.m[1][0] = t_val*ax*ay + az*s; swayMat.m[1][1] = t_val*ay*ay + c;    swayMat.m[1][2] = t_val*ay*az - ax*s; swayMat.m[1][3] = 0;
        swayMat.m[2][0] = t_val*ax*az - ay*s; swayMat.m[2][1] = t_val*ay*az + ax*s; swayMat.m[2][2] = t_val*az*az + c;    swayMat.m[2][3] = 0;
        swayMat.m[3][0] = 0;                  swayMat.m[3][1] = 0;                  swayMat.m[3][2] = 0;                  swayMat.m[3][3] = 1;

        // --- Apply Transform ---
        // Final = Translate(Pos) * Sway * Rotate(InitialRot) * Scale(InitialScale)
        
        // Reconstruct base matrix components
        Matrix4x4 baseMat = initial.toMatrix();
        // Remove translation to get Orientation * Scale
        Matrix4x4 localMat = baseMat;
        localMat.m[0][3] = 0; localMat.m[1][3] = 0; localMat.m[2][3] = 0;

        // Translation Matrix
        Matrix4x4 T; 
        T.m[0][3] = initial.position.x; 
        T.m[1][3] = initial.position.y; 
        T.m[2][3] = initial.position.z;

        // Multiply: T * Sway * Local
        Matrix4x4 TSway = T * swayMat; 
        Matrix4x4 finalMat = TSway * localMat;

        // Update Instance
        inst->setTransform(finalMat);
    }
}
