#ifndef VOLUMETRIC_H
#define VOLUMETRIC_H

#include "Material.h"
#include "Vec3SIMD.h"
#include "Ray.h"
#include <algorithm>
#include "perlin.h"

class Volumetric : public Material {
public:
    Volumetric(const Vec3& a, double d, double ap,  double max_d, const Vec3& e);
    virtual float get_opacity(const Vec2& uv) const override;
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const override;
    double calculate_density(const Vec3& point) const;
    Vec3 calculate_random_color_shift(double distance_to_center) const;
    //double calculate_density(double distance_to_center) const;
   
    virtual Vec3 getEmission(double u, double v, const Vec3& p) const override;

    Vec3 calculate_color_shift(double distance_to_center, double local_density) const;


    virtual MaterialType type() const override {
        return MaterialType::Volumetric;
    }

    virtual float get_scattering_factor() const override {
        return scattering_factor;
    }

    // Geri kalan gerekli override fonksiyonlar (gereksizse varsayýlan davranýþ)
    virtual double getIndexOfRefraction() const override {
        return 1.0;  // Volumetrik malzeme için kýrýlma indeksi
    }

    virtual Vec3 getF0() const override {
        return Vec3(0.04f);  // Volumetrik malzemelerde varsayýlan Fresnel deðeri
    }

    virtual bool has_normal_map() const override {
        return false;  // Volumetrik malzemeler normal haritalar kullanmaz
    }

    virtual Vec3 get_normal_from_map(double u, double v) const override {
        return Vec3(0, 0, 1);  // Normal harita kullanýlmadýðýnda varsayýlan normal
    }

    virtual float get_normal_strength() const override {
        return 1.0f;  // Varsayýlan normal harita kuvveti
    }

    virtual Vec3 getEmission() const override {
        return Vec3(0, 0, 0);  // Volumetrik malzemeler genellikle emisyon yapmaz
    }
    Vec3 sample_henyey_greenstein(const Vec3& wi, double g) const;
    void setG(double g) { this->g = g; }
    double density;
private:
    Vec3 albedo;
    Vec3 center;
    Vec3 emission;
    Perlin noise;
    double absorption_probability;
    float scattering_factor;      // Saçýlma faktörü
    double max_distance;
    double g;
    double calculate_density() const;
    Vec3 random_in_unit_sphere() const;
   
    double calculate_absorption(double distance) const;
};

#endif // VOLUMETRIC_H
