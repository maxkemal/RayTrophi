/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          AABB.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef AABB_H
#define AABB_H

#include "Vec3SIMD.h"
#include "Ray.h"
#include "RayPacket.h"

class AABB {
public:
    Vec3 min, max;
    Vec3 diagonal() const {
        return max - min;
    }
    mutable float cached_surface_area;  // Önceden hesaplanmış alan
   

    AABB() {}
    AABB(const Vec3& a, const Vec3& b)
        : min(a), max(b), cached_surface_area(-1.0f) {
    }  // İlk başta -1.0 olarak ayarla
    bool overlaps(const AABB& other) const {
        return (min.x <= other.max.x && max.x >= other.min.x) &&
            (min.y <= other.max.y && max.y >= other.min.y) &&
            (min.z <= other.max.z && max.z >= other.min.z);
    }

    bool hit(const Ray& r, float t_min, float t_max) const {
        for (int a = 0; a < 3; a++) {
            auto invD = std::abs(r.direction[a]) > 1e-6f ? 1.0f / r.direction[a] : 0.0f;
            auto t0 = (min[a] - r.origin[a]) * invD;
            auto t1 = (max[a] - r.origin[a]) * invD;
            if (invD < 0.0f)
                std::swap(t0, t1);
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min )
                return false;
        }
        return true;
    }

    bool hit_interval(const Ray& r, float t_min, float t_max, float& t_enter, float& t_exit) const {
        float t0 = t_min;
        float t1 = t_max;
        for (int a = 0; a < 3; a++) {
            auto invD = std::abs(r.direction[a]) > 1e-6f ? 1.0f / r.direction[a] : 0.0f;
            auto t_near = (min[a] - r.origin[a]) * invD;
            auto t_far = (max[a] - r.origin[a]) * invD;
            if (invD < 0.0f) std::swap(t_near, t_far);
            t0 = t_near > t0 ? t_near : t0;
            t1 = t_far < t1 ? t_far : t1;
            if (t1 <= t0) return false;
        }
        t_enter = t0;
        t_exit = t1;
        return true;
    }

    Vec3 getPositiveVertex(const Vec3& normal) const {
        return Vec3(
            normal.x >= 0 ? max.x : min.x,
            normal.y >= 0 ? max.y : min.y,
            normal.z >= 0 ? max.z : min.z
        );
    }
    int max_axis() const;
    float surface_area() const;
    bool is_valid() const {
        return (min.x <= max.x && min.y <= max.y && min.z <= max.z);
    }
    // AABB Packet Intersection (Slab Method - SIMD)
    __m256 hit_packet(const RayPacket& r, float t_min, float t_max) const {
         return hit_packet(r, _mm256_set1_ps(t_min), _mm256_set1_ps(t_max));
    }

    __m256 hit_packet(const RayPacket& r, __m256 t_min, __m256 t_max) const {
         // Box min/max components splatted to 8 lanes
         Vec3SIMD b_min_x(min.x);
         Vec3SIMD b_min_y(min.y);
         Vec3SIMD b_min_z(min.z);

         Vec3SIMD b_max_x(max.x);
         Vec3SIMD b_max_y(max.y);
         Vec3SIMD b_max_z(max.z);

         // t0 = (min - org) * inv_dir
         Vec3SIMD t0_x = (b_min_x - r.orig_x) * r.inv_dir_x;
         Vec3SIMD t0_y = (b_min_y - r.orig_y) * r.inv_dir_y;
         Vec3SIMD t0_z = (b_min_z - r.orig_z) * r.inv_dir_z;

         // t1 = (max - org) * inv_dir
         Vec3SIMD t1_x = (b_max_x - r.orig_x) * r.inv_dir_x;
         Vec3SIMD t1_y = (b_max_y - r.orig_y) * r.inv_dir_y;
         Vec3SIMD t1_z = (b_max_z - r.orig_z) * r.inv_dir_z;

         // Swap t0/t1 based on sign of direction (for correctness when dir < 0)
         Vec3SIMD tmin_x = Vec3SIMD::min(t0_x, t1_x);
         Vec3SIMD tmax_x = Vec3SIMD::max(t0_x, t1_x);

         Vec3SIMD tmin_y = Vec3SIMD::min(t0_y, t1_y);
         Vec3SIMD tmax_y = Vec3SIMD::max(t0_y, t1_y);

         Vec3SIMD tmin_z = Vec3SIMD::min(t0_z, t1_z);
         Vec3SIMD tmax_z = Vec3SIMD::max(t0_z, t1_z);

         // tnear = max(tmin_x, tmin_y, tmin_z, global_tmin)
         __m256 t_enter = _mm256_max_ps(tmin_x.data, _mm256_max_ps(tmin_y.data, _mm256_max_ps(tmin_z.data, t_min)));
         
         // tfar = min(tmax_x, tmax_y, tmax_z, global_tmax)
         __m256 t_exit = _mm256_min_ps(tmax_x.data, _mm256_min_ps(tmax_y.data, _mm256_min_ps(tmax_z.data, t_max)));

         // Hit if t_enter <= t_exit
         return _mm256_cmp_ps(t_enter, t_exit, _CMP_LE_OQ);
    }
};

AABB surrounding_box(const AABB& box0, const AABB& box1);
#endif // AABB_H

