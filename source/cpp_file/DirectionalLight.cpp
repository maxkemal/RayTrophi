#include "DirectionalLight.h"

DirectionalLight::DirectionalLight(const Vec3& dir, const Vec3& intens, double radius)
    : disk_radius(radius) {
    direction = dir.normalize();
    intensity = intens;
    
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
    double z = 0; // Disk düzleminde z'nin sýfýr olduðunu varsayýyoruz

    // Disk merkezini ve yönü ekleyerek ýþýk kaynaðýnýn noktasýný belirleyin
    return Vec3(direction) * 1000.0 + Vec3(x, y, z);
}

LightType DirectionalLight::type() const {
    return LightType::Directional;
}
