#include "Box.h"

Box::Box() {}
Box::Box(const Vec3& position, double size, std::shared_ptr<Material> mat)
    : center(position), size(size), material(mat) {}

Vec3 Box::min() const {
    return center - Vec3(size / 2, size / 2, size / 2);
}

Vec3 Box::max() const {
    return center + Vec3(size / 2, size / 2, size / 2);
}

bool Box::hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const {
    Vec3 min_point = min();
    Vec3 max_point = max();

    double tmin = (min_point.x - r.origin.x) / r.direction.x;
    double tmax = (max_point.x - r.origin.x) / r.direction.x;

    if (tmin > tmax) std::swap(tmin, tmax);

    double tymin = (min_point.y - r.origin.y) / r.direction.y;
    double tymax = (max_point.y - r.origin.y) / r.direction.y;

    if (tymin > tymax) std::swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    double tzmin = (min_point.z - r.origin.z) / r.direction.z;
    double tzmax = (max_point.z - r.origin.z) / r.direction.z;

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
        double epsilon = 0.0001;

        if (std::abs(relative_pos.x) > size / 2 - epsilon)
            outward_normal.x = relative_pos.x > 0 ? 1 : -1;
        else if (std::abs(relative_pos.y) > size / 2 - epsilon)
            outward_normal.y = relative_pos.y > 0 ? 1 : -1;
        else if (std::abs(relative_pos.z) > size / 2 - epsilon)
            outward_normal.z = relative_pos.z > 0 ? 1 : -1;

        rec.set_face_normal(r, outward_normal);
        rec.material = material;
        return true;
    }

    return false;
}

bool Box::bounding_box(double time0, double time1, AABB& output_box) const {
    output_box = AABB(min(), max());
    return true;
}
