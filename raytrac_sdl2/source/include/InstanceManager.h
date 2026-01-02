#pragma once

#include "InstanceGroup.h"
#include <vector>
#include <memory>
#include <functional>

class OptixWrapper;
class SceneData;

// ═══════════════════════════════════════════════════════════════════════════════
// INSTANCE MANAGER - Central manager for all instance groups
// ═══════════════════════════════════════════════════════════════════════════════

class InstanceManager {
public:
    static InstanceManager& getInstance() {
        static InstanceManager instance;
        return instance;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // GROUP MANAGEMENT
    // ─────────────────────────────────────────────────────────────────────────
    
    // Create a new instance group from selected mesh
    int createGroup(const std::string& name, const std::string& source_node_name,
                    const std::vector<std::shared_ptr<Triangle>>& source_triangles);
    
    // Delete an instance group
    void deleteGroup(int group_id);
    
    // Get group by ID (nullptr if not found)
    InstanceGroup* getGroup(int group_id);
    const InstanceGroup* getGroup(int group_id) const;
    
    // Get all groups
    std::vector<InstanceGroup>& getGroups() { return groups; }
    const std::vector<InstanceGroup>& getGroups() const { return groups; }
    
    // Find group by name
    InstanceGroup* findGroupByName(const std::string& name);
    
    // ─────────────────────────────────────────────────────────────────────────
    // BRUSH OPERATIONS
    // ─────────────────────────────────────────────────────────────────────────
    
    // Paint instances at a location (returns number added)
    int paintInstances(int group_id, const Vec3& center, const Vec3& normal,
                       float brush_radius, float density_multiplier = 1.0f);
    
    // Erase instances in radius (returns number removed)
    int eraseInstances(int group_id, const Vec3& center, float brush_radius);
    
    // Erase ALL groups' instances in radius
    int eraseAllGroupsInRadius(const Vec3& center, float brush_radius);
    
    // ─────────────────────────────────────────────────────────────────────────
    // GPU SYNCHRONIZATION
    // ─────────────────────────────────────────────────────────────────────────
    
    // Sync all dirty groups to GPU
    void syncToGPU(OptixWrapper* optix);
    
    // Force rebuild of all TLAS instances
    void rebuildAllTLAS(OptixWrapper* optix);
    
    // Mark all groups as needing GPU sync
    void markAllDirty();
    
    // ─────────────────────────────────────────────────────────────────────────
    // STATISTICS
    // ─────────────────────────────────────────────────────────────────────────
    
    size_t getTotalInstanceCount() const;
    size_t getTotalTriangleCount() const;
    size_t getGroupCount() const { return groups.size(); }
    
    // ─────────────────────────────────────────────────────────────────────────
    // CLEANUP
    // ─────────────────────────────────────────────────────────────────────────
    
    void clearAll();
    
private:
    InstanceManager() = default;
    ~InstanceManager() = default;
    InstanceManager(const InstanceManager&) = delete;
    InstanceManager& operator=(const InstanceManager&) = delete;
    
    std::vector<InstanceGroup> groups;
    int next_group_id = 1;
    
    // Poisson disk sampling for even distribution
    std::vector<Vec3> generatePoissonPoints(const Vec3& center, float radius, 
                                            float min_distance, int max_attempts = 30);
};
