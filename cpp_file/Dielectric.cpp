﻿#include "Ray.h"
#include "Hittable.h"
#include <cmath>
#include "Dielectric.h"
#include <perlin.h>

Dielectric::Dielectric(double index_of_refraction,  const Vec3& color, double caustic_intensity,
     double tint_factor, double roughness, double scratch_density)
    : color(color), caustic_intensity(caustic_intensity), 
    tint_factor(tint_factor), roughness(roughness) {
    // Slightly different IOR for each color channel to simulate dispersion
    ir = { index_of_refraction, index_of_refraction * 1.01, index_of_refraction * 1.02 };
}
float Dielectric::get_opacity(const Vec2& uv) const {
    return 1.0f;  // Dielectric materyal tamamen opak, bu yüzden 1.0 döndür
}

Vec3 Dielectric::getEmission(double u, double v, const Vec3& p) const {
    return Vec3(0.0, 0.0, 0.0);
}
Vec3 Dielectric::calculate_reflected_attenuation(const Vec3& base_color, const Vec3& fresnel_factor) const {
    Vec3 reflected_color = base_color * fresnel_factor;
   // double attenuation_factor = exp(-thickness);
    return apply_tint(reflected_color);  // Renk uygulandı
}
// Kırılan ışığın katkısını hesaplama
Vec3 Dielectric::calculate_refracted_attenuation(const Vec3& base_color, double thickness, const Vec3& fresnel_factor, const std::array<double, 3>& ior) const {
    // Fresnel hesabından gelen kırılma bileşeni
    Vec3 refracted_color = base_color * (Vec3(1.0, 1.0, 1.0) - fresnel_factor);

    // Beer-Lambert yasasına göre zayıflama
    // Malzeme için yaklaşık absorpsiyon katsayısı
    Vec3 absorption_coef(0.1, 0.08, 0.12); // RGB için farklı absorpsiyon değerleri

    // Exponential decay: e^(-absorption * thickness)
    Vec3 attenuation(
        exp(-absorption_coef.x * thickness),
        exp(-absorption_coef.y * thickness),
        exp(-absorption_coef.z * thickness)
    );

    return apply_tint(refracted_color * attenuation);
}
Vec3 Dielectric::apply_scratches(const Vec3& color, const Vec3& point) const {
    if (scratch_density <= 0.0) return color;

    static Perlin noise; // Static olarak tanımlayarak her seferinde yeni instance oluşturmasını önlüyoruz

    // Scale faktörlerini ayarla
    double base_scale = 20.0;
    Vec3 scaled_point = point * base_scale;

    // Farklı frekanslarda noise ve turbulence kullan
    double noise_val = noise.noise(scaled_point);
    double turb_val = noise.turb(scaled_point * 2.0, 4); // Daha detaylı çizikler için turbulence

    // Çiziklerin yönünü belirle
    Vec3 direction_point = point * 0.1; // Daha geniş ölçekte yön değişimi için
    double angle = noise.noise(direction_point) * M_PI;
    Vec3 scratch_direction(std::cos(angle), std::sin(angle), 0.0);

    // Ana scratch pattern'i oluştur
    double pattern = std::abs(noise_val) * 0.5 + std::abs(turb_val) * 0.5;

    // Scratch threshold'u scratch_density'e göre ayarla
    double scratch_threshold = 1.0 - scratch_density;

    // Scratch efektinin gücünü hesapla
    double scratch_strength = 0.0;
    if (pattern > scratch_threshold) {
        scratch_strength = (pattern - scratch_threshold) / (1.0 - scratch_threshold);

        // Non-linear scratch effect için üstel fonksiyon
        scratch_strength = std::pow(scratch_strength, 1.5);

        // Yöne bağlı faktörü ekle
        double directional_factor = std::abs(Vec3::dot(
            scratch_direction,
            Vec3(point.x, point.y, 0.0).normalize()
        ));
        scratch_strength *= (0.7 + 0.3 * directional_factor);

        // Turbulence ile detay ekle
        scratch_strength *= (1.0 + 0.2 * turb_val);
    }

    // Scratch rengi için temel beyaz highlight
    Vec3 scratch_color(1.0, 1.0, 1.0);

    // Derinlik etkisi için scratch rengini hafifçe karart
    scratch_color = scratch_color * (0.9 + 0.1 * pattern);

    // Final renk karışımını hesapla
    Vec3 final_color = color * (1.0 - scratch_strength * 0.4) +
        scratch_color * scratch_strength * 0.4;

    // Hafif kontrast ayarı
    final_color = Vec3(
        std::pow(final_color.x, 1.05),
        std::pow(final_color.y, 1.05),
        std::pow(final_color.z, 1.05)
    );

    return Vec3::clamp(final_color, 0.0f, 1.0f);
}
bool Dielectric::scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    // Normalizasyon
    Vec3 unit_direction = r_in.direction.normalize();
    Vec3 outward_normal = rec.interpolated_normal;
	outward_normal = outward_normal.normalize();
    // Malzeme parametreleri
    float adjusted_thickness = 0.001f;  // Daha küçük bir kalınlık değeri

    // Açı hesaplamaları
    double cos_theta = fmin(Vec3::dot(-unit_direction, outward_normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // Fresnel hesaplaması - kırılma indisi daha doğru kullanılıyor
    Vec3 current_ir = rec.front_face ? ir : std::array<double, 3>{ 1.0 / ir[0], 1.0 / ir[1], 1.0 / ir[2] };
    Vec3 fresnel_factor = fresnel(unit_direction, outward_normal, current_ir);
    double fresnel_reflect = fresnel_factor.y; // Yeşil kanal üzerinden hesaplamak daha doğru olabilir

    // Kırılma oranı (yeşil kanal için)
    double refract_ratio = rec.front_face ? (1.0 / ir[1]) : ir[1];
    bool cannot_refract = sin_theta * refract_ratio > 1.0;

    // Olasılık hesaplamaları
    double direct_trans_prob = 0.15; // %5 doğrudan geçiş
    double reflect_prob = fresnel_reflect;
    double refract_prob = cannot_refract ? 0.0 : (1.0 - fresnel_reflect) * (1.0 - direct_trans_prob);

    // Normalize
    double total_prob = reflect_prob + refract_prob + direct_trans_prob;
    reflect_prob /= total_prob;
    refract_prob /= total_prob;
    direct_trans_prob /= total_prob;

    
    // Işık yönü hesaplaması
    Vec3 direction;
    double random_val = random_double();

    if (random_val < reflect_prob) {
        // Yansıma
        direction = Vec3::reflect(unit_direction, outward_normal).normalize();
        direction = apply_roughness(direction, outward_normal).normalize(); // Pürüzlülük daha kontrollü
        attenuation += calculate_reflected_attenuation(color, fresnel_factor);
    }
    else if (random_val < reflect_prob + refract_prob) {
        // Kırılma
        Vec3 refracted_dir = Vec3::refract(unit_direction, outward_normal, refract_ratio).normalize();
        direction = apply_roughness(refracted_dir, outward_normal).normalize(); // Daha az pürüzlülük

        // Beer-Lambert yasasına göre zayıflama
        double distance = (rec.point - r_in.origin).length();  // Başlangıç noktası ile etkileşim noktası arasındaki mesafe
       attenuation *= calculate_attenuation(distance);
        //trans_color = trans_color*color; // Renkle son katmanı çarp
        attenuation += calculate_refracted_attenuation(color,adjusted_thickness,fresnel_factor,ir);
        
        // Kostikler için kontrollü ekleme
        Vec3 caustic = calculate_caustic(unit_direction, outward_normal, direction);  
        attenuation += caustic ;

    }
    else {
        // Doğrudan geçiş
        direction = unit_direction;       
        // Çok hafif zayıflama
          // Beer-Lambert yasasına göre zayıflama
        double distance = (rec.point - r_in.origin).length();
        attenuation += calculate_attenuation(distance );
    }
  
    // Yeni ışının başlangıç noktası küçük bir offsetle
    scattered = Ray(rec.point+ r_in.direction *adjusted_thickness, direction.normalize());

    return true;
}

double Dielectric::getIndexOfRefraction() const {
    return ir[1];  // Return the green channel IOR as an average
}

Vec3 Dielectric::fresnel(const Vec3& incident, const Vec3& normal, const std::array<double, 3>& ior) const {
    float cos_i = std::clamp(Vec3::dot(-incident, normal), -1.0, 1.0);

    // Havadan cama mı, camdan havaya mı?
    float eta_i = 1.0f, eta_t = ior[1];
    if (cos_i > 0.0f) std::swap(eta_i, eta_t);

    float sin_t2 = (eta_i / eta_t) * (eta_i / eta_t) * (1.0f - cos_i * cos_i);
    if (sin_t2 > 1.0f) return Vec3(1.0f, 1.0f, 1.0f);  // Tam yansıma

    float cos_t = sqrt(1.0f - sin_t2);

    float r0_r = ((ior[0] - eta_i) / (ior[0] + eta_i)) * ((ior[0] - eta_i) / (ior[0] + eta_i));
    float fr_r = r0_r + (1.0f - r0_r) * pow(1.0f - fabs(cos_i), 3.0f);

    float r0_g = ((ior[1] - eta_i) / (ior[1] + eta_i)) * ((ior[1] - eta_i) / (ior[1] + eta_i));
    float fr_g = r0_g + (1.0f - r0_g) * pow(1.0f - fabs(cos_i), 3.0f);

    float r0_b = ((ior[2] - eta_i) / (ior[2] + eta_i)) * ((ior[2] - eta_i) / (ior[2] + eta_i));
    float fr_b = r0_b + (1.0f - r0_b) * pow(1.0f - fabs(cos_i), 3.0f);

    return Vec3(fr_r, fr_g, fr_b);
}

Vec3 Dielectric::calculate_caustic(const Vec3& incident, const Vec3& normal, const Vec3& refracted) const {
    float dot_product = Vec3::dot(incident, normal);
    float refraction_angle = std::acos(std::clamp(Vec3::dot(refracted, -normal), -1.0, 1.0));
    float caustic_factor = pow(std::max(refraction_angle, 0.0f), 3.0) * caustic_intensity;
    return color * caustic_factor * (1.0 - dot_product * dot_product);
}

Vec3 Dielectric::apply_tint(const Vec3& color) const {   
    return color *(1.0 - tint_factor) + color * tint_factor;
}

Vec3 Dielectric::apply_roughness(const Vec3& dir, const Vec3& normal) const {
    double rough_factor = roughness * roughness; // Roughness karesi daha doğru sonuç verir
    if (rough_factor == 0.0) return dir; // Pürüzsüzse yön değiştirme

    // Rastgele küçük bir yön sapması ekleyelim
    Vec3 random_vec = Vec3::random_unit_vector() * rough_factor;
    Vec3 perturbed_dir = (dir + random_vec).normalize();

    return perturbed_dir;
}

double Dielectric::calculate_attenuation(double distance) const {
    const float absorption_coefficient = 0.01f;  // Daha düşük absorpsiyon
    return std::exp(-absorption_coefficient * distance);
}
