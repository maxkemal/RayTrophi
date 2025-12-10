#include "Box.h"

Box::Box() {}
Box::Box(const Vec3& position, double size, std::shared_ptr<Material> mat)
    : center(position), size(size), material(mat) {}

Vec3 Box::min() const {
    float s = (float)size / 2.0f;
    return center - Vec3(s, s, s);
}

Vec3 Box::max() const {
    float s = (float)size / 2.0f;
    return center + Vec3(s, s, s);
}

bool Box::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    Vec3 min_point = min();
    Vec3 max_point = max();

    float tmin = (min_point.x - r.origin.x) / r.direction.x;
    float tmax = (max_point.x - r.origin.x) / r.direction.x;

    if (tmin > tmax) std::swap(tmin, tmax);

    float tymin = (min_point.y - r.origin.y) / r.direction.y;
    float tymax = (max_point.y - r.origin.y) / r.direction.y;

    if (tymin > tymax) std::swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (min_point.z - r.origin.z) / r.direction.z;
    float tzmax = (max_point.z - r.origin.z) / r.direction.z;

    if (tzmin > tzmax) std::swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;

    if (tzmax < tmax)
        tmax = tzmax;

    if (tmin < t_max && tmin > t_min) {
        rec.t = tmin;
        rec.point = r.at(rec.t);
        Vec3 outward_normal = Vec3(0, 0, 0);
        Vec3 relative_pos = rec.point - center;
        float epsilon = 0.0001f;

        if (std::abs(relative_pos.x) > size / 2 - epsilon)
            outward_normal.x = relative_pos.x > 0 ? 1.0f : -1.0f;
        else if (std::abs(relative_pos.y) > size / 2 - epsilon)
            outward_normal.y = relative_pos.y > 0 ? 1.0f : -1.0f;
        else if (std::abs(relative_pos.z) > size / 2 - epsilon)
            outward_normal.z = relative_pos.z > 0 ? 1.0f : -1.0f;

        rec.set_face_normal(r, outward_normal);
        rec.material = material;
        return true;
    }

    return false;
}

bool Box::bounding_box(float time0, float time1, AABB& output_box) const {
    output_box = AABB(min(), max());
    return true;
}
