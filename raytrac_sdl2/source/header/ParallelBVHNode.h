#pragma once
#include <algorithm>
#include <future>
#include <vector>
#include <memory>
#include <atomic>
#include "Hittable.h"
#include "Material.h"
#include "AABB.h"
#include "OptixWrapper.h"
#include <queue>

class ParallelBVHNode : public Hittable {
private:    
  
    static std::atomic<int> active_threads;
    static constexpr size_t MIN_OBJECTS_PER_THREAD = 8192;
    static constexpr int MAX_DEPTH = 32;
    static bool box_compare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b, int axis);
    static bool box_x_compare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b);
    static bool box_y_compare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b);
    static bool box_z_compare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b);

    int depth = 0; // Add depth as a class member

    std::shared_ptr<OptixWrapper> optix_wrapper;
public:
    AABB box;
    ParallelBVHNode* init(const std::vector<std::shared_ptr<Hittable>>& src_objects,
        size_t start, size_t end, float time0, float time1,int depth);

    bool updateTree(const std::vector<std::shared_ptr<Hittable>>& animated_objects, float time0, float time1);
    bool occluded(const Ray& ray, float t_min, float t_max) const override {
        // hýzlý versiyon: sadece t varsa true döner, detay alma
        HitRecord dummy;
        return hit(ray, t_min, t_max, dummy);
    }

    ParallelBVHNode() = default;  

    ParallelBVHNode(const std::vector<std::shared_ptr<Hittable>>& src_objects,
        size_t start, size_t end, float time0, float time1, int depth=0);
    bool isLeaf() const;
    std::shared_ptr<Hittable> left;
    std::shared_ptr<Hittable> right;
    bool bounding_box(float time0, float time1, AABB& output_box) const;
    bool use_optix;
   

     bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const ;
};