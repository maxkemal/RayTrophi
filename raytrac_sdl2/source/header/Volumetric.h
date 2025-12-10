#ifndef VOLUMETRIC_H
#define VOLUMETRIC_H

#include "Material.h"
#include "Vec3SIMD.h"
#include "Ray.h"
#include <algorithm>
#include <memory>
#include "perlin.h"

class Volumetric : public Material {
public:
    Volumetric(const Vec3& a, double d, double ap, double sf, const Vec3& e, std::shared_ptr<Perlin> noiseGen);

    virtual float get_opacity(const Vec2& uv) const override;
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const override;
    virtual Vec3 getEmission(const Vec2& uv, const Vec3& p) const override;

    double calculate_density(const Vec3& point) const;
    Vec3 calculate_random_color_shift(double distance_to_center) const;
    Vec3 calculate_color_shift(double distance_to_center, double local_density) const;

    virtual MaterialType type() const override { return MaterialType::Volumetric; }
    virtual float get_scattering_factor() const override { return scattering_factor; }
    virtual double getIndexOfRefraction() const override { return 1.0; }
    virtual Vec3 getF0() const override { return Vec3(0.04f); }
    virtual bool has_normal_map() const override { return false; }
    virtual Vec3 get_normal_from_map(double u, double v) const override { return Vec3(0, 0, 1); }
    virtual float get_normal_strength() const override { return 1.0f; }
    virtual Vec3 getEmission() const override { return Vec3(0, 0, 0); }

    Vec3 sample_henyey_greenstein(const Vec3& wi, double g) const;

    void setG(double g) { this->g = g; }
    void setNoise(std::shared_ptr<Perlin> n) { noise = n; }
    std::shared_ptr<Perlin> noise; // Gürültü jeneratörü
private:
    Vec3 albedo;
    Vec3 center;
    Vec3 emission;
    double density;
    double absorption_probability;
    float scattering_factor;
    double max_distance;
    double g;

   
    double calculate_absorption(double distance) const;
    Vec3 random_in_unit_sphere() const;
};

#endif // VOLUMETRIC_H
