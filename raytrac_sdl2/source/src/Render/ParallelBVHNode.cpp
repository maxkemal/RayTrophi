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
    if (src_objects.empty() || start >= end) {
        // Safe initialization for empty node
        return;
    }
    init(src_objects, start, end, time0, time1, depth);
}

// Struct to store precomputed AABBs
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

ParallelBVHNode* ParallelBVHNode::init(const std::vector<std::shared_ptr<Hittable>>& src_objects,
    size_t start, size_t end, float time0, float time1,  int depth)
{
    const size_t object_span = end - start;
    if (depth >= MAX_DEPTH || object_span <= 1) {
        left = src_objects[start];
        right = nullptr; 
        left->bounding_box(time0, time1, box);
        return this;
    }

    std::vector<ObjectInfo> object_infos;
    object_infos.resize(object_span);
    AABB overall_box;

    // Parallel calculation of ObjectInfo
    #pragma omp parallel
    {
        AABB local_box; 
        #pragma omp for nowait
        for (int i = 0; i < static_cast<int>(object_span); ++i) {
            ObjectInfo& info = object_infos[i];
            info.object = src_objects[start + i];
            info.object->bounding_box(time0, time1, info.box);
            info.centroid = (info.box.min + info.box.max) * 0.5;
            local_box = surrounding_box(local_box, info.box);
        }
        #pragma omp critical
        {
            overall_box = surrounding_box(overall_box, local_box);
        }
    }

    // Find best split
    int best_axis = -1;
    size_t best_split_idx = 0;
    double best_cost = std::numeric_limits<double>::max();

    // Calculate SAH for each axis
    for (int axis = 0; axis < 3; ++axis) {
        std::sort(object_infos.begin(), object_infos.end(),
            [axis](const ObjectInfo& a, const ObjectInfo& b) {
                return a.centroid[axis] < b.centroid[axis];
            });

        std::vector<AABB> left_boxes(object_span);
        std::vector<AABB> right_boxes(object_span);

        AABB running_left;
        for (size_t i = 0; i < object_span - 1; ++i) {
            running_left = surrounding_box(running_left, object_infos[i].box);
            left_boxes[i] = running_left;
        }

        AABB running_right;
        for (size_t i = object_span - 1; i > 0; --i) {
            running_right = surrounding_box(running_right, object_infos[i].box);
            right_boxes[i - 1] = running_right;
        }

        // Parallel SAH calculation
        #pragma omp parallel
        {
            double local_best_cost = std::numeric_limits<double>::max();
            int local_best_axis = -1;
            size_t local_best_split = 0;

            #pragma omp for nowait
            for (int i = 1; i < static_cast<int>(object_span); ++i) {
                double cost = sah_cost(i, left_boxes[i - 1],
                    object_span - i, right_boxes[i - 1]);
                
                if (cost < local_best_cost) {
                    local_best_cost = cost;
                    local_best_axis = axis;
                    local_best_split = i;
                }
            }

            #pragma omp critical
            {
                if (local_best_cost < best_cost) {
                    best_cost = local_best_cost;
                    best_axis = local_best_axis;
                    best_split_idx = local_best_split;
                }
            }
        }
    }

    if (best_split_idx == 0 || best_split_idx == object_span || best_cost >= overall_box.surface_area() * object_span) {
        best_split_idx = object_span / 2;
    }
    std::sort(object_infos.begin(), object_infos.end(),
        [best_axis](const ObjectInfo& a, const ObjectInfo& b) {
            return a.centroid[best_axis] < b.centroid[best_axis];
        });

    std::vector<std::shared_ptr<Hittable>> left_objects;
    std::vector<std::shared_ptr<Hittable>> right_objects;
    left_objects.reserve(best_split_idx);
    right_objects.reserve(object_span - best_split_idx);

    for (size_t i = 0; i < best_split_idx; ++i) {
        left_objects.push_back(object_infos[i].object);
    }
    for (size_t i = best_split_idx; i < object_span; ++i) {
        right_objects.push_back(object_infos[i].object);
    }

    bool can_parallelize = object_span >= MIN_OBJECTS_PER_THREAD &&
        active_threads < std::thread::hardware_concurrency();

    if (can_parallelize) {
        active_threads++;
        auto future_left = std::async(std::launch::async,
            [&left_objects, time0, time1, depth]() {
                return std::make_shared<ParallelBVHNode>(
                    left_objects, 0, left_objects.size(),
                    time0, time1, depth+1);
            });

        right = std::make_shared<ParallelBVHNode>(
            right_objects, 0, right_objects.size(),
			time0, time1, depth + 1
        );

        left = future_left.get();
        active_threads--;
    }
    else {
        left = std::make_shared<ParallelBVHNode>(
            left_objects, 0, left_objects.size(),
            time0, time1, depth+1
        );
        right = std::make_shared<ParallelBVHNode>(
            right_objects, 0, right_objects.size(),
			time0, time1, depth + 1
        );
    }

    AABB left_box, right_box;
    left->bounding_box(time0, time1, left_box);
    right->bounding_box(time0, time1, right_box);
    box = surrounding_box(left_box, right_box);

    return this;

}

bool ParallelBVHNode::updateTree(const std::vector<std::shared_ptr<Hittable>>& animated_objects, float time0, float time1) {
    if (left == right) { 
        if (std::find(animated_objects.begin(), animated_objects.end(), left) != animated_objects.end()) {
            return left->bounding_box(time0, time1, box);
        }
        return false;
    }

    bool left_updated = left && std::dynamic_pointer_cast<ParallelBVHNode>(left) && std::dynamic_pointer_cast<ParallelBVHNode>(left)->updateTree(animated_objects, time0, time1);
    bool right_updated = right && std::dynamic_pointer_cast<ParallelBVHNode>(right) && std::dynamic_pointer_cast<ParallelBVHNode>(right)->updateTree(animated_objects, time0, time1);

    if (left_updated || right_updated) {
        AABB left_box, right_box;
        bool has_left = left && left->bounding_box(time0, time1, left_box);
        bool has_right = right && right->bounding_box(time0, time1, right_box);

        if (has_left && has_right) box = surrounding_box(left_box, right_box);
        else if (has_left) box = left_box;
        else if (has_right) box = right_box;
        return true;
    }
    return false;
}
bool ParallelBVHNode::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, bool ignore_volumes) const {
    if (!box.hit(r, t_min, t_max))
        return false;

    if (!right) {
        return left->hit(r, t_min, t_max, rec, ignore_volumes);
    }

    bool hit_left = left->hit(r, t_min, t_max, rec, ignore_volumes);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec, ignore_volumes);

    return hit_left || hit_right;
}
bool ParallelBVHNode::bounding_box(float time0, float time1, AABB& output_box) const {
    if (box.is_valid()) {
        output_box = box;
        return true;
    }
    return false;
}
bool ParallelBVHNode::box_compare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b, int axis) {
    AABB box_a;
    AABB box_b;
    if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b))
        std::cerr << "No bounding box in BVHNode constructor.\n";
    return box_a.min[axis] < box_b.min[axis];
}
bool ParallelBVHNode::box_x_compare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
    return box_compare(a, b, 0);
}
bool ParallelBVHNode::box_y_compare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
    return box_compare(a, b, 1);
}
bool ParallelBVHNode::box_z_compare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
    return box_compare(a, b, 2);
}

// ============================================================================
// Packet Tracing Traversal (Phase 2)
// ============================================================================

void ParallelBVHNode::hit_packet(const RayPacket& r, float t_min, float t_max, HitRecordPacket& rec, bool ignore_volumes) const {
    // 1. Packet Intersect against THIS Node's AABB
    __m256 hit_mask = box.hit_packet(r, t_min, t_max);

    // Optimization: If NO ray hits the box, stop.
    // _mm256_movemask_ps returns integer bitmask of sign bits.
    // hit_packet usually returns mask where passed/hit is 1.
    // Note: 'hit_packet' logic returns 1 if t_enter <= t_exit.
    if (_mm256_movemask_ps(hit_mask) == 0) {
        return;
    }

    // 2. Leaf Node check
    if (!right) {
        // If leaf, delegate to the object (Triangle or other)
        // But we must check if the object actually implemented hit_packet.
        // Hittable::hit_packet defaults to empty.
        // Assuming 'left' is a Triangle or Mesh that overrides it.
        left->hit_packet(r, t_min, t_max, rec, ignore_volumes); 
        // Note: Masking is handled inside object's hit_packet or via 'rec'.
        // Ideally we should pass 'hit_mask' down to avoid checking rays that missed the box.
        // But the signature doesn't take 'hit_mask'. 
        // Improvement: update HitRecordPacket::mask with hit_mask? NO.
        // 'rec.mask' tracks rays that have HIT geometry. 'hit_mask' is for bounds.
        // The object should re-check or assume valid.
        return;
    }

    // 3. Recursive Step for Inner Node
    // We traverse both children. A better approach is to check split axis direction to order traversal front-to-back.
    // For now, simpler traversal.
    
    // Check Left
    // Optimization: pass updated t_max? Ray packets have distinct t_max per ray. 
    // rec.t stores closest hit so far.
    left->hit_packet(r, t_min, t_max, rec, ignore_volumes);
    
    // Check Right
    right->hit_packet(r, t_min, t_max, rec, ignore_volumes);
}

__m256 ParallelBVHNode::occluded_packet(const RayPacket& packet, float t_min, __m256 t_max) const {
    __m256 hit_mask = box.hit_packet(packet, _mm256_set1_ps(t_min), t_max);
    if (_mm256_movemask_ps(hit_mask) == 0) return _mm256_setzero_ps();

    if (!right) {
        // Leaf node
        return left->occluded_packet(packet, t_min, t_max);
    }

    __m256 occ_left = left->occluded_packet(packet, t_min, t_max);
    
    // Optimization: if all active rays are already occluded, skip right
    __m256 combined_occ = occ_left;
    if (_mm256_movemask_ps(_mm256_andnot_ps(combined_occ, hit_mask)) == 0) {
        return combined_occ;
    }

    __m256 occ_right = right->occluded_packet(packet, t_min, t_max);
    return _mm256_or_ps(occ_left, occ_right);
}

