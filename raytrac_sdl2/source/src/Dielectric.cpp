#include "Ray.h"
#include "Hittable.h"
#include <cmath>
#include "Dielectric.h"
#include <perlin.h>

Dielectric::Dielectric(float index_of_refraction,  const Vec3& color, float caustic_intensity,
     float tint_factor, float roughness, float scratch_density)
    : color(color), caustic_intensity(caustic_intensity), 
    tint_factor(tint_factor), roughness(roughness) {
    // Slightly different IOR for each color channel to simulate dispersion
    ir = Vec3(index_of_refraction,
        index_of_refraction * 1.01f,
        index_of_refraction * 1.02f);
}
float Dielectric::get_opacity(const Vec2& uv) const {
    return 1.0f;  // Dielectric materyal tamamen opak, bu yüzden 1.0 döndür
}

Vec3 Dielectric::getEmission(const Vec2& uv, const Vec3& p) const {
    return Vec3(0.0f, 0.0, 0.0f);
}
Vec3 Dielectric::calculate_reflected_attenuation(const Vec3& base_color, const Vec3& fresnel_factor) const {
    Vec3 reflected_color = base_color * fresnel_factor;
   // double attenuation_factor = exp(-thickness);
    return apply_tint(reflected_color);  // Renk uygulandı
}
// Kırılan ışığın katkısını hesaplama
Vec3 Dielectric::calculate_refracted_attenuation(const Vec3& base_color, double thickness, const Vec3& fresnel_factor, const Vec3& ior) const {
    // Fresnel hesabından gelen kırılma bileşeni
    Vec3 refracted_color = base_color * (Vec3(1.0f, 1.0, 1.0f) - fresnel_factor);

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
    if (scratch_density <= 0.0f) return color;

    static Perlin noise; // Static olarak tanımlayarak her seferinde yeni instance oluşturmasını önlüyoruz

    // Scale faktörlerini ayarla
    float base_scale = 20.0f;
    Vec3 scaled_point = point * base_scale;

    // Farklı frekanslarda noise ve turbulence kullan
    float noise_val = noise.noise(scaled_point);
    float turb_val = noise.turb(scaled_point * 2.0f, 4); // Daha detaylı çizikler için turbulence

    // Çiziklerin yönünü belirle
    Vec3 direction_point = point * 0.1; // Daha geniş ölçekte yön değişimi için
    float angle = noise.noise(direction_point) * M_PI;
    Vec3 scratch_direction(std::cos(angle), std::sin(angle), 0.0f);

    // Ana scratch pattern'i oluştur
    float pattern = std::abs(noise_val) * 0.5f + std::abs(turb_val) * 0.5f;

    // Scratch threshold'u scratch_density'e göre ayarla
    float scratch_threshold = 1.0f - scratch_density;

    // Scratch efektinin gücünü hesapla
    float scratch_strength = 0.0f;
    if (pattern > scratch_threshold) {
        scratch_strength = (pattern - scratch_threshold) / (1.0f - scratch_threshold);

        // Non-linear scratch effect için üstel fonksiyon
        scratch_strength = std::pow(scratch_strength, 1.5);

        // Yöne bağlı faktörü ekle
        double directional_factor = std::abs(Vec3::dot(
            scratch_direction,
            Vec3(point.x, point.y, 0.0f).normalize()
        ));
        scratch_strength *= (0.7f + 0.3f * directional_factor);

        // Turbulence ile detay ekle
        scratch_strength *= (1.0 + 0.2f * turb_val);
    }

    // Scratch rengi için temel beyaz highlight
    Vec3 scratch_color(1.0f, 1.0, 1.0f);

    // Derinlik etkisi için scratch rengini hafifçe karart
    scratch_color = scratch_color * (0.9f + 0.1f * pattern);

    // Final renk karışımını hesapla
    Vec3 final_color = color * (1.0f - scratch_strength * 0.4) +
        scratch_color * scratch_strength * 0.4;

    // Hafif kontrast ayarı
    final_color = Vec3(
        std::pow(final_color.x, 1.05f),
        std::pow(final_color.y, 1.05f),
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
    float adjusted_thickness = 0.01f;  // Daha küçük bir kalınlık değeri

    // Açı hesaplamaları
    float cos_theta = fmin(Vec3::dot(-unit_direction, outward_normal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    // Fresnel hesaplaması - kırılma indisi daha doğru kullanılıyor
    Vec3 reversed_ir = Vec3(1.0f / ir.x, 1.0f / ir.y, 1.0f / ir.z);

    Vec3 current_ir = rec.front_face ? ir : reversed_ir;
    Vec3 fresnel_factor = fresnel(unit_direction, outward_normal, current_ir);
    float fresnel_reflect = fresnel_factor.y; // Yeşil kanal üzerinden hesaplamak daha doğru olabilir

    // Kırılma oranı (yeşil kanal için)
    float refract_ratio = rec.front_face ? (1.0f / ir[1]) : ir[1];
    bool cannot_refract = sin_theta * refract_ratio > 1.0;

    // Olasılık hesaplamaları
    float direct_trans_prob = 0.05; // %5 doğrudan geçiş
    float reflect_prob = fresnel_reflect;
    float refract_prob = cannot_refract ? 0.0 : (1.0f - fresnel_reflect) * (1.0f - direct_trans_prob);

    // Normalize
    float total_prob = reflect_prob + refract_prob + direct_trans_prob;
    reflect_prob /= total_prob;
    refract_prob /= total_prob;
    direct_trans_prob /= total_prob;

    
    // Işık yönü hesaplaması
    Vec3 direction;
    float random_val = Vec3::random_float();

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
        float distance = (rec.point - r_in.origin).length();  // Başlangıç noktası ile etkileşim noktası arasındaki mesafe
       attenuation *= calculate_attenuation(distance);       
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
        float distance = (rec.point - r_in.origin).length();
        attenuation += calculate_attenuation(distance );
    }
  
    // Yeni ışının başlangıç noktası küçük bir offsetle
    scattered = Ray(rec.point+ r_in.direction *adjusted_thickness, direction.normalize());

    return true;
}

float Dielectric::getIndexOfRefraction() const {
    return ir[1];  // Return the green channel IOR as an average
}

Vec3 Dielectric::fresnel(const Vec3& incident, const Vec3& normal, const Vec3& ir_values) const {
    float cos_i = std::clamp(Vec3::dot(-incident, normal), -1.0f, 1.0f);

    // Havadan cama mı, camdan havaya mı?
    float eta_i = 1.0f, eta_t = ir_values[1];
    if (cos_i > 0.0f) std::swap(eta_i, eta_t);

    float sin_t2 = (eta_i / eta_t) * (eta_i / eta_t) * (1.0f - cos_i * cos_i);
    if (sin_t2 > 1.0f) return Vec3(1.0f, 1.0f, 1.0f);  // Tam yansıma

    float cos_t = sqrt(1.0f - sin_t2);

    float r0_r = ((ir_values[0] - eta_i) / (ir_values[0] + eta_i)) * ((ir_values[0] - eta_i) / (ir_values[0] + eta_i));
    float fr_r = r0_r + (1.0f - r0_r) * pow(1.0f - fabs(cos_i), 3.0f);

    float r0_g = ((ir_values[1] - eta_i) / (ir_values[1] + eta_i)) * ((ir_values[1] - eta_i) / (ir_values[1] + eta_i));
    float fr_g = r0_g + (1.0f - r0_g) * pow(1.0f - fabs(cos_i), 3.0f);

    float r0_b = ((ir_values[2] - eta_i) / (ir_values[2] + eta_i)) * ((ir_values[2] - eta_i) / (ir_values[2] + eta_i));
    float fr_b = r0_b + (1.0f - r0_b) * pow(1.0f - fabs(cos_i), 3.0f);

    return Vec3(fr_r, fr_g, fr_b);
}

Vec3 Dielectric::calculate_caustic(const Vec3& incident, const Vec3& normal, const Vec3& refracted) const {
    float dot_product = Vec3::dot(incident, normal);
    float refraction_angle = std::acos(std::clamp(Vec3::dot(refracted, -normal), -1.0f, 1.0f));
    float caustic_factor = pow(std::max(refraction_angle, 0.0f), 3.0f) * caustic_intensity;
    return color * caustic_factor * (1.0f - dot_product * dot_product);
}

Vec3 Dielectric::apply_tint(const Vec3& color) const {   
    return color *(1.0f - tint_factor) + color * tint_factor;
}

Vec3 Dielectric::apply_roughness(const Vec3& dir, const Vec3& normal) const {
    float rough_factor = roughness * roughness; // Roughness karesi daha doğru sonuç verir
    if (rough_factor == 0.0f) return dir; // Pürüzsüzse yön değiştirme

    // Rastgele küçük bir yön sapması ekleyelim
    Vec3 random_vec = Vec3::random_unit_vector() * rough_factor;
    Vec3 perturbed_dir = (dir + random_vec).normalize();

    return perturbed_dir;
}

float Dielectric::calculate_attenuation(float distance) const {
    const float absorption_coefficient = 0.01f;  // Daha düşük absorpsiyon
    return std::exp(-absorption_coefficient * distance);
}


