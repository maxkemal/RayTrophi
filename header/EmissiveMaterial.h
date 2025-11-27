#ifndef EMISSIVEMATERIAL_H
#define EMISSIVEMATERIAL_H

#include "Material.h"

class EmissiveMaterial : public Material {
public:
    EmissiveMaterial(const Vec3& emit);

    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const override;

    virtual Vec3 getEmission(const Vec2& uv, const Vec3& p) const override;
    virtual float get_opacity(const Vec2& uv) const override;
private:
    Vec3 emission;
};

#endif // EMISSIVEMATERIAL_H
