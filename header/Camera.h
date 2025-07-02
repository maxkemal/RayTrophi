#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include "Vec3.h"
#include "Matrix4x4.h"
#include "Ray.h"
#include "AABB.h"
#include "ThreadLocalRNG.h"
class Camera {
private:
    struct Plane {
        Vec3 normal;
        double distance;

        Plane() : normal(Vec3()), distance(0) {}
        Plane(const Vec3& n, const Vec3& point) : normal(n.normalize()) {
            distance = -Vec3::dot(normal, point);
        }

        double distanceToPoint(const Vec3& point) const {
            return Vec3::dot(normal, point) + distance;
        }
    };

public:
    Vec3 initialLookDirection;
    std::string nodeName;
    int blade_count;
    float aperture;
    float focus_dist;
    Vec3 origin;
    Vec3 u, v, w;
    Vec3 lookfrom;
    Vec3 lookat;
    Vec3 vup;
    float aspect;
    float near_dist;
    float far_dist;
    float fov;
    float aspect_ratio;
    float vfov;
   // Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double vfov, double aspect, double aperture, double focus_dist);

    Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect, float aperture, float focus_dist, int blade_count);
    Camera();
    Ray get_ray(float s, float t) const;

    int random_int(int min, int max) const;

    void update_camera_vectors();

    void moveToTargetLocked(const Vec3& new_position);

    void setLookDirection(const Vec3& direction_normalized);

    Vec3 random_in_unit_polygon(int sides) const;

    float calculate_bokeh_intensity(const Vec3& point) const;

    Vec3 create_bokeh_shape(const Vec3& color, float intensity) const;

    bool isPointInFrustum(const Vec3& point, float size) const;
     Matrix4x4 getRotationMatrix() const ;
    bool isAABBInFrustum(const AABB& aabb) const;
    std::vector<AABB> performFrustumCulling(const std::vector<AABB>& objects) const;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    float lens_radius;
private:
    void updateFrustumPlanes();

    Vec3 getViewDirection() const;

    // Frustum culling için ek alanlar

    Plane frustum_planes[6];
};

#endif // CAMERA_H
