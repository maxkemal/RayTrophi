/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          DiffuseLight.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef DIFFUSELIGHT_H
#define DIFFUSELIGHT_H

#include "Material.h"

class DiffuseLight : public Material {
public:
    DiffuseLight(Vec3 c);
   
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const override;
    virtual float get_opacity(const Vec2& uv) const override;
    virtual Vec3 getEmission(const Vec2& uv, const Vec3& p) const override;
   
public:
    Vec3 emit;
};

#endif // DIFFUSELIGHT_H

