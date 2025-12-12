#include "ParallelBVHNode.h"
#include <atomic>
#include <thread>
#include <future>
#include <memory>
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>
#include <omp.h>
#include <execution>
#include <numeric>
#include <mutex>
#include <chrono>

std::atomic<int> ParallelBVHNode::active_threads(0);

ParallelBVHNode::ParallelBVHNode(const std::vector<std::shared_ptr<Hittable>>& src_objects,
    size_t start, size_t end, float time0, float time1, int depth)
{
    init(src_objects, start, end, time0, time1, depth);
}

// Onceden hesaplanmis AABB'leri saklamak icin struct
struct alignas(64) ObjectInfo {
    std::shared_ptr<Hittable> object;
    AABB box;
    Vec3 centroid;
    float surface_area;
    
    ObjectInfo() = default;
    ObjectInfo(std::shared_ptr<Hittable> obj, float time0, float time1)
        : object(obj), centroid(0, 0, 0) {
        obj->bounding_box(time0, time1, box);
        centroid = (box.min + box.max) * 0.5;
        surface_area = box.surface_area();
    }
};

constexpr float OBJECT_INTERSECTION_COST = 1.0;
inline float sah_cost(size_t num_left, const AABB& left_box,
    size_t num_right, const AABB& right_box) {
    return OBJECT_INTERSECTION_COST *
        (num_left * left_box.surface_area() +
            num_right * right_box.surface_area());
}