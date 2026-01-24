/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Hittable.h
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef HITTABLE_H
#define HITTABLE_H

#include "Ray.h"
#include "RayPacket.h"
#include "HitRecordPacket.h"
#include "AABB.h"
#include <vector>
#include <memory>
#include "Vec2.h"

// Forward declarations
class Material;
class Texture;
class Triangle;
class Triangle;
class VDBVolume;
class GasVolume;

struct HitRecord {
    Vec3 point;
    Vec3 normal;
    Vec3 neighbor_normal;
    bool has_neighbor_normal = false;
    Vec3 interpolated_normal;
    Vec3 face_normal;
  
    std::shared_ptr<Material> material;
    uint16_t materialID = 0xFFFF;
    
    float t = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    bool front_face = false;
  
    // Add reference to the hit object itself (for accessing AABB etc.)
    // const Hittable* obj = nullptr; // REMOVED: potentially circular and redundant (we use triangle ptr)
    
    const Triangle* triangle = nullptr;
   
    const VDBVolume* vdb_volume = nullptr; // Pointer to VDB Volume if hit
    const GasVolume* gas_volume = nullptr; // Pointer to Gas Volume if hit
    Vec3 tangent;
    Vec3 bitangent;
    bool has_tangent = false;
    Vec2 uv;
    bool is_instance_hit = false; // Track if the hit came from an instance (for brush filters)
  
    inline void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = Vec3::dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
   
    HitRecord() = default;
};

/**
 * @brief Abstract base class for all ray-traceable objects.
 * 
 * All renderable geometry (Triangle, Sphere, etc.) must inherit from this
 * and implement hit() and bounding_box() methods.
 */
class Hittable {
public:
    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec, bool ignore_volumes = false) const = 0;
    
    // Packet Tracing Interface (Phase 2)
    // Updates 'rec' based on active lanes in 'r' and 'rec.mask'
    virtual void hit_packet(const RayPacket& packet, float t_min, float t_max, HitRecordPacket& rec, bool ignore_volumes = false) const {
        // Default implementation: Scalar fallback (slow)
        // In reality, leaf nodes (Triangle) and BVH nodes must override this.
    }

    virtual bool bounding_box(float time0, float time1, AABB& output_box) const = 0;
    virtual ~Hittable() = default;
    
    // Optional: Collect neighbor normals for smoothing (default: no-op)
    virtual void collect_neighbor_normals(const AABB& query_box, Vec3& neighbor_normal,
        int& neighbor_count, const std::shared_ptr<Material>& current_material) const {
    }
    
    // Fast occlusion test (default: just calls hit())
    virtual bool occluded(const Ray& ray, float t_min, float t_max) const {
        HitRecord dummy;
        return hit(ray, t_min, t_max, dummy);
    }

    // Packet Occlusion Interface (Phase 2)
    // Returns a mask of occluded rays
    virtual __m256 occluded_packet(const RayPacket& packet, float t_min, __m256 t_max) const {
        // Default implementation: Scalar fallback (very slow, override in BVH/Objects)
        alignas(32) int mask_result[8] = {0};
        alignas(32) float t_max_vals[8];
        _mm256_store_ps(t_max_vals, t_max);

        for (int i = 0; i < 8; i++) {
            Ray r(Vec3(packet.orig_x.get(i), packet.orig_y.get(i), packet.orig_z.get(i)),
                  Vec3(packet.dir_x.get(i), packet.dir_y.get(i), packet.dir_z.get(i)));
            if (occluded(r, t_min, t_max_vals[i])) {
                mask_result[i] = 0xFFFFFFFF;
            }
        }
        return _mm256_load_ps((float*)mask_result);
    }

    // Visibility flag for rendering (CPU/GPU)
    bool visible = true;
};

#endif // HITTABLE_H


