#include "Sphere.h"

Sphere::Sphere() {}
Sphere::Sphere(Vec3 cen, double r, std::shared_ptr<Material> m)
    : center(cen), radius(r), material(m) {}

bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    Vec3 oc = r.origin - center;
    auto a = Vec3::dot(r.direction, r.direction);
    auto b = Vec3::dot(oc, r.direction);
    auto c = Vec3::dot(oc, oc) - radius * radius;
    auto discriminant = b * b - a * c;

    if (discriminant > 0) {
        auto root = std::sqrt(discriminant);
        auto temp = (-b - root) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.point = r.at(rec.t);
            Vec3 outward_normal = (rec.point - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.material = material;
            return true;
        }
        temp = (-b + root) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.point = r.at(rec.t);
            Vec3 outward_normal = (rec.point - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.material = material;
            return true;
        }
    }
    return false;
}

bool Sphere::bounding_box(float time0, float time1, AABB& output_box) const {
    output_box = AABB(center - Vec3(radius, radius, radius),
        center + Vec3(radius, radius, radius));
    return true;
}

Vec3 Sphere::min() const {
    return center - Vec3(radius, radius, radius);
}

Vec3 Sphere::max() const {
    return center + Vec3(radius, radius, radius);
}
