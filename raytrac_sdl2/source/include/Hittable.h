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
    Material* materialPtr = nullptr; // FAST ACCESS
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
    int terrain_id = -1; // Terrain ID if hit
    Vec3 tangent;
    Vec3 bitangent;
    bool has_tangent = false;
    Vec2 uv;
    bool is_instance_hit = false; // Track if the hit came from an instance (for brush filters)
    
    // Custom Blended Data (for Terrain System)
    bool use_custom_data = false;
    Vec3 custom_albedo;
    float custom_roughness = 0.5f;
    float custom_metallic = 0.0f;
    float custom_transmission = 0.0f;
    float custom_clearcoat = 0.0f;
    float custom_clearcoat_roughness = 0.0f;
    float custom_subsurface = 0.0f;
    Vec3 custom_subsurface_color;
    float custom_translucent = 0.0f;
    float custom_ior = 1.45f;
 
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


    // Visibility flag for rendering (CPU/GPU)
    bool visible = true;
};

#endif // HITTABLE_H


