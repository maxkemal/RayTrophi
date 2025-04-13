#include "Volumetric.h"
#include "Vec3.h"
#include "Ray.h"
#include <cmath>

Volumetric::Volumetric(const Vec3& a, double d, double ap, double sf, const Vec3& e)
    : albedo(a), density(d), absorption_probability(ap), scattering_factor(sf),emission(e) {}

Vec3 Volumetric::getEmission(double u, double v, const Vec3& p) const {
    return  emission*albedo*calculate_density(p.length()); // Hacimde yoðunlukla deðiþen ýþýk emisyonu

}
float Volumetric::get_opacity(const Vec2& uv) const {
    return 1.0f;  // Dielectric materyal tamamen opak, bu yüzden 1.0 döndür
}

bool Volumetric::scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    double distance_to_center =  (rec.point - Vec3(0, 0, 0)).length();
    double local_density = calculate_density(distance_to_center);

    // Yoðunluða baðlý g parametresi
    double adaptive_g = 0.5;
    Vec3 scatter_direction = sample_henyey_greenstein(r_in.direction, adaptive_g);

    // Yoðunluk düþükse daha az sapma
    scatter_direction = Vec3::lerp(r_in.direction, scatter_direction, local_density).normalize();
    scattered = Ray(rec.point, scatter_direction);
    attenuation = albedo * local_density;
   
    return true;
}
double Volumetric::calculate_density(const Vec3& point) const {
    // Final yoðunluk
    return density ;
}

double Volumetric::calculate_absorption(double distance_to_center) const {
    double normalized_distance = distance_to_center / max_distance;
    return std::min(1.0, absorption_probability * (1.0 - normalized_distance));
}
Vec3 Volumetric::random_in_unit_sphere() const {
    while (true) {
        Vec3 p = Vec3(random_double(-1, 1), random_double(-1, 1), random_double(-1, 1));
        if (p.length_squared() < 1) return p;
    }
}

Vec3 Volumetric::sample_henyey_greenstein(const Vec3& wi, double g) const {
    double cos_theta = 1.0 - 2.0 * random_double();

    // Yoðunluða baðlý olarak g parametresini modifiye et
    double distance_to_center = wi.length();
    double local_density = calculate_density(distance_to_center);
    double modified_g = g * local_density; // Yoðunluk düþtükçe daha az yönlü saçýlma

    if (std::abs(modified_g) > 0.001) {
        cos_theta = (1.0 + modified_g * modified_g -
            std::pow((1.0 - modified_g * modified_g) /
                (1.0 + modified_g * (2.0 * random_double() - 1.0)), 2)) /
            (2.0 * modified_g);
    }

    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    const double phi = 2.0 * M_PI * random_double();
    Vec3 u = Vec3::random_unit_vector().cross(wi).normalize();
    Vec3 v = wi.cross(u);
    return (u * std::cos(phi) * sin_theta +
        v * std::sin(phi) * sin_theta +
        wi * cos_theta).normalize();
}