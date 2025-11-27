#ifndef SPHERE_H
#define SPHERE_H

#include "Hittable.h"
#include "Vec3.h"
#include "Material.h"

class Sphere : public Hittable {
public:
    Sphere();
    Sphere(Vec3 cen, double r, std::shared_ptr<Material> m);

    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    virtual bool bounding_box(float time0, float time1, AABB& output_box) const override;

private:
    Vec3 center;
    double radius;
    std::shared_ptr<Material> material;

    Vec3 min() const;
    Vec3 max() const;
};

#endif // SPHERE_H
