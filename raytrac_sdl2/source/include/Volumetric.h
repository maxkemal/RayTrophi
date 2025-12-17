#ifndef VOLUMETRIC_H
#define VOLUMETRIC_H

#include "Material.h"
#include "Vec3SIMD.h"
#include "Ray.h"
#include <algorithm>
#include <memory>
#include "perlin.h"

// Unified Volume Rendering için gerekli
#include "unified_volume.h"

class Volumetric : public Material {
public:
    Volumetric(const Vec3& a, double d, double ap, double sf, const Vec3& e, std::shared_ptr<Perlin> noiseGen);

    virtual float get_opacity(const Vec2& uv) const override;
    
    // Unified Ray Marching Scatter Logic
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const override;
    
    // Emissions
    virtual Vec3 getEmission(const Vec2& uv, const Vec3& p) const override;
    virtual Vec3 getEmission() const override;

    // Helper methods
    double calculate_density(const Vec3& point) const;

    // Special method for volumetric emission integration
    Vec3 getVolumetricEmission(const Vec3& p, const Vec3& dir) const;

    virtual MaterialType type() const override { return MaterialType::Volumetric; }
    virtual float get_scattering_factor() const override { return (float)scattering_factor; }
    virtual double getIndexOfRefraction() const override { return 1.0; }
    virtual Vec3 getF0() const override { return Vec3(0.04f); }
    virtual bool has_normal_map() const override { return false; }
    virtual Vec3 get_normal_from_map(double u, double v) const override { return Vec3(0, 0, 1); }
    virtual float get_normal_strength() const override { return 1.0f; }

    // Setters
    void setG(double g) { this->g = g; }
    void setNoise(std::shared_ptr<Perlin> n) { noise = n; }
    void setStepSize(float step) { step_size = step; }
    void setMaxSteps(int steps) { max_steps = steps; }

    std::shared_ptr<Perlin> noise;

private:
    Vec3 albedo;       // Scattering color
    Vec3 emission;     // Emission color
    double density;    // Base density multiplier
    double absorption_probability; // Absorption coefficient factor
    double scattering_factor;     // Scattering coefficient factor
    double g;          // Phase anisotropy (default 0)
    
    // Ray Marching Params
    float step_size;
    int max_steps;
    double max_distance;

    // Core Ray Marching Function
    // Returns emitted light accumulated along the ray
    // Updates out_transmittance (0..1)
    Vec3 march_volume(const Vec3& origin, const Vec3& dir, float& out_transmittance) const;
};

#endif // VOLUMETRIC_H
