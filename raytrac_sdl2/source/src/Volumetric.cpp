#include "Volumetric.h"
#include "Vec3.h"
#include "Ray.h"
#include <cmath>

Volumetric::Volumetric(const Vec3& a, double d, double ap, double sf, const Vec3& e, std::shared_ptr<Perlin> noiseGen)
    : albedo(a), density(d), absorption_probability(ap), scattering_factor(sf), emission(e), noise(noiseGen) {
    max_distance = 10.0;
}

Vec3 Volumetric::getEmission(const Vec2& uv, const Vec3& p) const {
    double distance_to_center = (p - Vec3(0, 0, 0)).length();
    double local_density = calculate_density(p);
    Vec3 shifted = calculate_color_shift(distance_to_center, local_density);
    return shifted;
}

Vec3 Volumetric::calculate_color_shift(double distance_to_center, double local_density) const {
    double color_shift_factor = 0.5;
    Vec3 white_tint = Vec3(1.0, 1.0, 1.0);
    Vec3 shifted_color = Vec3::lerp(albedo, white_tint, (1.0 - local_density) * color_shift_factor);
    return shifted_color;
}

float Volumetric::get_opacity(const Vec2& uv) const {
    return 1.0f;  // Tamamen opak sayılıyor burada
}

bool Volumetric::scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    // Hacim içinde örnekleme noktası
    Vec3 inside_sample_point = rec.point + r_in.direction * Vec3::random_float(0.01, 0.05);

    // Gürültü bazlı yoğunluk
    float local_density = calculate_density(inside_sample_point);

    // 🚫 Eğer yoğunluk çok azsa ışık saçılmaz, direkt geçsin
    if (local_density < 0.01) {
        attenuation = Vec3(1.0f);  // Enerji korunur
        scattered = Ray(rec.point + r_in.direction * 0.001f, r_in.direction); // aynı yöne devam et
        return true;
    }

    // Scatter yönü hesapla
    double adaptive_g = 0.5;
    Vec3 scatter_direction = sample_henyey_greenstein(r_in.direction, adaptive_g);
    scatter_direction = Vec3::lerp(r_in.direction, scatter_direction, local_density).normalize();

    // Yeni ışın oluştur
    scattered = Ray(rec.point + scatter_direction * 0.001f, scatter_direction);

    // Renk yoğunluğa göre değişsin
    Vec3 shifted_albedo = calculate_color_shift((rec.point - Vec3(0, 0, 0)).length(), local_density);
   // attenuation = shifted_albedo;
    
    Vec3 Emission = getEmission();
    attenuation = shifted_albedo + Emission;

    return true;
}


double Volumetric::calculate_density(const Vec3& surface_point) const {
    if (!noise) return density;

    // Yüzey noktasından itibaren rastgele bir offset ile örnekleme yap
    Vec3 offset = Vec3::random_in_unit_sphere() * 0.1; // 0.1 birimlik hacimsel kayma
    Vec3 sample_point = surface_point + offset;

    double scale = 0.3;
    double base_density = noise->turb(sample_point * scale);

    double threshold = 0.4;
    double sharpness = 20.0;
    double smooth_density = 1.0 / (1.0 + std::exp(-sharpness * (base_density - threshold)));

    return density * smooth_density;
}


double Volumetric::calculate_absorption(double distance_to_center) const {
    double normalized_distance = distance_to_center / max_distance;
    return std::min(1.0, absorption_probability * (1.0 - normalized_distance));
}



Vec3 Volumetric::sample_henyey_greenstein(const Vec3& wi, double g) const {
    double cos_theta = 1.0 - 2.0 * Vec3::random_float();
    double distance_to_center = wi.length();
    double local_density = calculate_density(wi);  // bu biraz yorumlanabilir, wi pozisyona karşılık gelmiyor ama idare eder
    double modified_g = g * local_density;

    if (std::abs(modified_g) > 0.001) {
        cos_theta = (1.0 + modified_g * modified_g -
            std::pow((1.0 - modified_g * modified_g) /
                (1.0 + modified_g * (2.0 * Vec3::random_float() - 1.0)), 2)) /
            (2.0 * modified_g);
    }

    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    const double phi = 2.0 * M_PI * Vec3::random_float();
    Vec3 u = Vec3::random_unit_vector().cross(wi).normalize();
    Vec3 v = wi.cross(u);
    return (u * std::cos(phi) * sin_theta +
        v * std::sin(phi) * sin_theta +
        wi * cos_theta).normalize();
}
