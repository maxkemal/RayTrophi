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

#include "Ray.h"

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
};

AABB surrounding_box(const AABB& box0, const AABB& box1);
#endif // AABB_H

