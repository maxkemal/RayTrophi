#ifndef BOX_H
#define BOX_H

#include "Hittable.h"
#include "Vec3.h"
#include "Material.h"

class Box : public Hittable {
public:
    Vec3 center;
    double size;
    std::shared_ptr<Material> material;

    Box();
    Box(const Vec3& position, double size, std::shared_ptr<Material> mat);

    Vec3 min() const;
    Vec3 max() const;

    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override;
    virtual bool bounding_box(double time0, double time1, AABB& output_box) const override;
};

#endif // BOX_H
