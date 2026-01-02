#include "InstanceManager.h"
#include "Triangle.h"
#include "globals.h"
#include <random>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════════
// GROUP MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

int InstanceManager::createGroup(const std::string& name, const std::string& source_node_name,
                                  const std::vector<std::shared_ptr<Triangle>>& source_triangles) {
    if (source_triangles.empty()) {
        SCENE_LOG_WARN("[InstanceManager] Cannot create group with empty source mesh");
        return -1;
    }
    
    InstanceGroup group;
    group.id = next_group_id++;
    group.name = name;
    group.source_node_name = source_node_name;
    group.source_triangles = source_triangles;
    group.gpu_dirty = true;
    
    groups.push_back(std::move(group));
    
    SCENE_LOG_INFO("[InstanceManager] Created group '" + name + "' (ID: " + 
                   std::to_string(group.id) + ") with " + 
                   std::to_string(source_triangles.size()) + " source triangles");
    
    return groups.back().id;
}

void InstanceManager::deleteGroup(int group_id) {
    auto it = std::find_if(groups.begin(), groups.end(),
        [group_id](const InstanceGroup& g) { return g.id == group_id; });
    
    if (it != groups.end()) {
        SCENE_LOG_INFO("[InstanceManager] Deleted group '" + it->name + "'");
        groups.erase(it);
    }
}

InstanceGroup* InstanceManager::getGroup(int group_id) {
    auto it = std::find_if(groups.begin(), groups.end(),
        [group_id](const InstanceGroup& g) { return g.id == group_id; });
    return (it != groups.end()) ? &(*it) : nullptr;
}

const InstanceGroup* InstanceManager::getGroup(int group_id) const {
    auto it = std::find_if(groups.begin(), groups.end(),
        [group_id](const InstanceGroup& g) { return g.id == group_id; });
    return (it != groups.end()) ? &(*it) : nullptr;
}

InstanceGroup* InstanceManager::findGroupByName(const std::string& name) {
    auto it = std::find_if(groups.begin(), groups.end(),
        [&name](const InstanceGroup& g) { return g.name == name; });
    return (it != groups.end()) ? &(*it) : nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BRUSH OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

int InstanceManager::paintInstances(int group_id, const Vec3& center, const Vec3& normal,
                                     float brush_radius, float density_multiplier) {
    InstanceGroup* group = getGroup(group_id);
    if (!group) return 0;
    
    // Calculate number of instances based on density
    float area = 3.14159f * brush_radius * brush_radius;
    float density = group->brush_settings.density * density_multiplier;
    int target_count = static_cast<int>(area * density);
    
    if (target_count <= 0) return 0;
    
    // Minimum distance between instances (based on density)
    float min_distance = 1.0f / sqrtf(density);
    
    // Generate poisson-distributed points
    std::vector<Vec3> points = generatePoissonPoints(center, brush_radius, min_distance);
    
    // Add instances at each point (check against existing instances)
    int added = 0;
    for (const Vec3& pos : points) {
        // Check if too close to any existing instance
        bool too_close = false;
        for (const auto& existing : group->instances) {
            float dist = (existing.position - pos).length();
            if (dist < min_distance * 0.8f) {  // 80% of min_distance for existing
                too_close = true;
                break;
            }
        }
        
        if (!too_close) {
            InstanceTransform t = group->generateRandomTransform(pos, normal);
            group->addInstance(t);
            added++;
        }
        
        if (added >= target_count) break;
    }
    
    return added;
}

int InstanceManager::eraseInstances(int group_id, const Vec3& center, float brush_radius) {
    InstanceGroup* group = getGroup(group_id);
    if (!group) return 0;
    
    size_t before = group->instances.size();
    group->removeInstancesInRadius(center, brush_radius);
    return static_cast<int>(before - group->instances.size());
}

int InstanceManager::eraseAllGroupsInRadius(const Vec3& center, float brush_radius) {
    int total_removed = 0;
    for (auto& group : groups) {
        size_t before = group.instances.size();
        group.removeInstancesInRadius(center, brush_radius);
        total_removed += static_cast<int>(before - group.instances.size());
    }
    return total_removed;
}

// ═══════════════════════════════════════════════════════════════════════════════
// POISSON DISK SAMPLING
// ═══════════════════════════════════════════════════════════════════════════════

std::vector<Vec3> InstanceManager::generatePoissonPoints(const Vec3& center, float radius,
                                                          float min_distance, int max_attempts) {
    std::vector<Vec3> points;
    std::vector<Vec3> active;
    
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist_angle(0.0f, 2.0f * 3.14159f);
    
    // Start with center point
    points.push_back(center);
    active.push_back(center);
    
    while (!active.empty()) {
        // Pick random active point
        std::uniform_int_distribution<size_t> idx_dist(0, active.size() - 1);
        size_t idx = idx_dist(rng);
        Vec3 point = active[idx];
        
        bool found = false;
        for (int attempt = 0; attempt < max_attempts; attempt++) {
            // Generate random point at distance [min_distance, 2*min_distance]
            float angle = dist_angle(rng);
            float r = min_distance * (1.0f + dist01(rng));
            
            Vec3 new_point = point + Vec3(cosf(angle) * r, 0, sinf(angle) * r);
            
            // Check if within brush radius
            float dist_from_center = (new_point - center).length();
            if (dist_from_center > radius) continue;
            
            // Check if far enough from all existing points
            bool too_close = false;
            for (const Vec3& p : points) {
                if ((new_point - p).length() < min_distance) {
                    too_close = true;
                    break;
                }
            }
            
            if (!too_close) {
                points.push_back(new_point);
                active.push_back(new_point);
                found = true;
                break;
            }
        }
        
        if (!found) {
            // Remove from active list
            active.erase(active.begin() + idx);
        }
    }
    
    return points;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU SYNCHRONIZATION (Placeholder - will integrate with OptixWrapper)
// ═══════════════════════════════════════════════════════════════════════════════

void InstanceManager::syncToGPU(OptixWrapper* optix) {
    // TODO: Implement when integrating with OptixWrapper
    // For each dirty group:
    //   - Register/update BLAS
    //   - Create/update TLAS instances
    for (auto& group : groups) {
        if (group.gpu_dirty) {
            // Will call optix->registerInstanceGroup() etc.
            group.gpu_dirty = false;
        }
    }
}

void InstanceManager::rebuildAllTLAS(OptixWrapper* optix) {
    markAllDirty();
    syncToGPU(optix);
}

void InstanceManager::markAllDirty() {
    for (auto& group : groups) {
        group.gpu_dirty = true;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATISTICS
// ═══════════════════════════════════════════════════════════════════════════════

size_t InstanceManager::getTotalInstanceCount() const {
    size_t total = 0;
    for (const auto& group : groups) {
        total += group.instances.size();
    }
    return total;
}

size_t InstanceManager::getTotalTriangleCount() const {
    size_t total = 0;
    for (const auto& group : groups) {
        total += group.getTriangleCount();
    }
    return total;
}

void InstanceManager::clearAll() {
    groups.clear();
    next_group_id = 1;
    SCENE_LOG_INFO("[InstanceManager] Cleared all instance groups");
}
