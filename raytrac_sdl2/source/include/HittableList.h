#ifndef HITTABLELIST_H
#define HITTABLELIST_H

#include "Hittable.h"
#include "ParallelBVHNode.h"
#include <memory>
#include <vector>
#include <algorithm>

/**
 * @brief Container for all renderable objects in the scene.
 * 
 * This class manages the primary object list and provides:
 * - Add/remove/clear operations
 * - Hit detection (with optional BVH acceleration)
 * - Bounding box computation
 */
class HittableList : public Hittable {
public:
    // Primary object storage - all scene objects live here
    std::vector<std::shared_ptr<Hittable>> objects;
    
    // Optional BVH acceleration structure
    std::shared_ptr<ParallelBVHNode> bvh_root;

    // =========================================================================
    // Constructors
    // =========================================================================
    HittableList();
    HittableList(std::shared_ptr<Hittable> object);

    // =========================================================================
    // Object Management
    // =========================================================================
    void clear();
    void add(std::shared_ptr<Hittable> object);
    void reserve(size_t n);
    size_t size() const;
    
    // Safe index-based access (returns nullptr if out of bounds)
    std::shared_ptr<Hittable> getHittable(size_t index) const {
        if (index < objects.size()) {
            return objects[index];
        }
        return nullptr;
    }
    
    // Remove object by pointer (returns true if found and removed)
    bool remove(std::shared_ptr<Hittable> object) {
        auto it = std::find(objects.begin(), objects.end(), object);
        if (it != objects.end()) {
            objects.erase(it);
            return true;
        }
        return false;
    }
    
    // Remove object by index (returns true if index was valid)
    bool removeAt(size_t index) {
        if (index < objects.size()) {
            objects.erase(objects.begin() + index);
            return true;
        }
        return false;
    }

    // =========================================================================
    // Hit Detection
    // =========================================================================
    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    virtual bool bounding_box(float time0, float time1, AABB& output_box) const override;

    // =========================================================================
    // BVH Management
    // =========================================================================
    void build_bvh();
    
    // Check if BVH is built and valid
    bool hasBVH() const { return bvh_root != nullptr; }
    
    // Invalidate BVH (call after geometry changes)
    void invalidateBVH() { bvh_root = nullptr; }
};

#endif // HITTABLELIST_H