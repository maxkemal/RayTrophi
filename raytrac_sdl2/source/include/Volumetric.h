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
    Vec3 getVolumetricEmission(const Vec3& p, const Vec3& dir, const Vec3& aabb_min, const Vec3& aabb_max) const;

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
    void setAlbedo(const Vec3& a) { albedo = a; }
    void setDensity(double d) { density = d; }
    void setAbsorption(double a) { absorption_probability = a; }
    void setScattering(double s) { scattering_factor = s; }
    void setEmissionColor(const Vec3& e) { emission = e; }
    
    // Getters for UI
    Vec3 getAlbedo() const { return albedo; }
    double getDensity() const { return density; }
    double getAbsorption() const { return absorption_probability; }
    double getScattering() const { return scattering_factor; }
    Vec3 getEmissionColor() const { return emission; }
    double getG() const { return g; }
    float getStepSize() const { return step_size; }
    int getMaxSteps() const { return max_steps; }
    float getNoiseScale() const { return noise_scale; }
   
    void setNoiseScale(float scale) { noise_scale = scale; }
    float getVoidThreshold() const { return void_threshold; }
    void setVoidThreshold(float t) { void_threshold = t; }

    // Multi-Scattering Getters/Setters (NEW)
    float getMultiScatter() const { return multi_scatter; }
    void setMultiScatter(float ms) { multi_scatter = ms; }
    float getGBack() const { return g_back; }
    void setGBack(float gb) { g_back = gb; }
    float getLobeMix() const { return lobe_mix; }
    void setLobeMix(float lm) { lobe_mix = lm; }
    int getLightSteps() const { return light_steps; }
    void setLightSteps(int ls) { light_steps = ls; }
    float getShadowStrength() const { return shadow_strength; }
    void setShadowStrength(float ss) { shadow_strength = ss; }

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
    float noise_scale = 1.0f; // Noise frequency multiplier
    float void_threshold = 0.0f; // Void cutoff
    double max_distance;
    
    // Multi-Scattering Parameters (NEW)
    float multi_scatter = 0.3f;   // Multi-scatter contribution (0-1)
    float g_back = -0.3f;         // Backward scattering anisotropy
    float lobe_mix = 0.7f;        // Forward/backward lobe mix
    int light_steps = 4;          // Light march steps (0=disabled)
    float shadow_strength = 0.8f; // Self-shadow intensity

    // Core Ray Marching Function
    // Returns emitted light accumulated along the ray
    // Updates out_transmittance (0..1)
    Vec3 march_volume(const Vec3& origin, const Vec3& dir, float& out_transmittance, const Vec3& aabb_min, const Vec3& aabb_max) const;
};
// Note: Moved getVolumetricEmission declaration to public section above or update it here if it's separate
// Actually, let's update the existing lines.

// ... oops, I need to match the line numbers exactly or use search.
// Let's rewrite the block around line 31 and 83.


#endif // VOLUMETRIC_H
