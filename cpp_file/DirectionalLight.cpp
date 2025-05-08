#include "DirectionalLight.h"

DirectionalLight::DirectionalLight(const Vec3& dir, const Vec3& intens, double radius)
    : disk_radius(radius) {
    direction = dir.normalize();
    intensity = intens;
    
}
int DirectionalLight::getSampleCount() const  {
    return 16; // ince ışıklar için daha fazla örnek
}

Vec3 DirectionalLight::random_point() const {
    // Disk üzerindeki rastgele bir nokta
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_angle(0, 2 * M_PI);
    std::uniform_real_distribution<> dis_radius(0, disk_radius);

    double angle = dis_angle(gen);
    double r = dis_radius(gen);
    double x = r * cos(angle);
    double y = r * sin(angle);
    double z = 0; // Disk düzleminde z'nin sıfır olduğunu varsayıyoruz

    last_sampled_point = Vec3(direction) * 1000.0 + Vec3(x, y, z);
    return last_sampled_point;

}

LightType DirectionalLight::type() const {
    return LightType::Directional;
}
float DirectionalLight::pdf(const Vec3& hit_point, const Vec3& incoming_direction) const {
    // Işığın yönü ile gelen ışın yönü arasındaki açı
    float cos_angle = Vec3::dot(-direction.normalize(), incoming_direction);

    // Disk yarıçapından görünen katı açıyı hesapla
    // Uzak mesafeden bakıldığında diskin görünür yarıçapı
    float apparent_angle = atan2(disk_radius, 1000.0); // 1000 birim uzaklıkta
    float cos_epsilon = cos(apparent_angle);

    // Eğer gelen ışın ışığın konisi içindeyse
    if (cos_angle > cos_epsilon) {
        // Koni içindeki uniform dağılım için PDF
        float solid_angle = 2.0f * M_PI * (1.0f - cos_epsilon);
        return 1.0f / solid_angle;
    }
    else {
        // Koni dışında ise olasılık 0
        return 0.0f;
    }
}
