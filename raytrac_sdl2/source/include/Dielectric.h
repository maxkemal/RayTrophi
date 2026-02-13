/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Dielectric.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "Material.h"
#include "ParallelBVHNode.h"
#include <array>
#include <globals.h>
class Dielectric : public Material {
public:
    MaterialType type() const override { return MaterialType::Dielectric; }
    Dielectric(float index_of_refraction,
        const Vec3& color,
        float caustic_intensity=1.0,
        float tint_factor=0.1,
        float roughness = 0.0,
        float scratch_density = 0.0);
   
    virtual Vec3 getEmission(const Vec2& uv, const Vec3& p) const;

    Vec3 calculate_reflected_attenuation(const Vec3& base_color, const Vec3& fresnel_factor) const;

    Vec3 calculate_reflected_attenuation(const Vec3& base_color, const Vec3& fresnel_factor, const std::array<double, 3>& ior) const;

   // Vec3 calculate_reflected_attenuation(const Vec3& base_color,const Vec3& fresnel_factor) const;

    Vec3 calculate_refracted_attenuation(const Vec3& base_color, double thickness, const Vec3& fresnel_factor, const Vec3& ior) const;

    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, bool& is_specular) const override;
    virtual float getIndexOfRefraction() const override;

    Vec3 fresnel(const Vec3& incident, const Vec3& normal, const Vec3& ir_values) const;

    float get_scattering_factor() const override {
        return 0.00001f; // Example value
    }
    Vec3 color; // Base color of the glass
    float caustic_intensity; // Intensity of the caustic effect
  
    float tint_factor; // How much the glass is tinted
    float scratch_density = 1.0f; // Density of scratches on the glass
    Vec3 ir;  // Index of refraction for R, G, B
    virtual float get_opacity(const Vec2& uv) const override;
    float roughness;
private:
   
    float calculate_caustic_factor(float cos_theta, float refraction_ratio, bool is_reflected) const;
    Vec3 calculate_caustic(const Vec3& incident, const Vec3& normal, const Vec3& refracted) const;
    Vec3 apply_tint(const Vec3& color) const;
    Vec3 apply_roughness(const Vec3& direction, const Vec3& normal) const;
    float calculate_attenuation(float distance) const;
    Vec3 apply_scratches(const Vec3& color, const Vec3& point) const;
    static float reflectance(float cosine, float ref_idx);
};

#endif // DIELECTRIC_H

