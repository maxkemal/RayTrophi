#include "DirectionalLight.h"
#include <cmath>

DirectionalLight::DirectionalLight(const Vec3& dir, const Vec3& input_intens, float rad)
      {
    direction = dir.normalize();

    float power = input_intens.length();
    color = (power > 0.0f) ? input_intens / power : Vec3(1.0f);
    intensity = power;
	radius = rad;
}

int DirectionalLight::getSampleCount() const {
    return 16;
}

Vec3 DirectionalLight::getDirection(const Vec3& /*point*/) const {
    return -direction.normalize();
}

Vec3 DirectionalLight::getIntensity(const Vec3& /*point*/, const Vec3& /*light_sample_point*/) const {
    return color * intensity;
}

Vec3 DirectionalLight::random_point() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_angle(0, 2 * M_PI);
    std::uniform_real_distribution<> dis_radius(0, radius);

    double angle = dis_angle(gen);
    double r = dis_radius(gen);
    float du = static_cast<float>(r * cos(angle));
    float dv = static_cast<float>(r * sin(angle));

    // Build an orthonormal tangent frame perpendicular to the light direction
    // so the disk jitter actually spreads samples across the sun disk rather
    // than along the direction axis (parity with Vulkan/OptiX).
    Vec3 N = direction.normalize();
    Vec3 tangent = Vec3::cross(N, Vec3(0.0f, 1.0f, 0.0f));
    if (tangent.length_squared() < 1e-6f) {
        tangent = Vec3::cross(N, Vec3(1.0f, 0.0f, 0.0f));
    }
    tangent = tangent.normalize();
    Vec3 bitangent = Vec3::cross(N, tangent).normalize();

    last_sampled_point = N * 1000.0f + tangent * du + bitangent * dv;
    return last_sampled_point;
}

LightType DirectionalLight::type() const {
    return LightType::Directional;
}

float DirectionalLight::pdf(const Vec3& /*hit_point*/, const Vec3& incoming_direction) const {
    float cos_angle = Vec3::dot(-direction.normalize(), incoming_direction);
    float apparent_angle = atan2(radius, 1000.0);
    float cos_epsilon = cos(apparent_angle);

    if (cos_angle > cos_epsilon) {
        float solid_angle = 2.0f * M_PI * (1.0f - cos_epsilon);
        return 1.0f / solid_angle;
    }
    else {
        return 0.0f;
    }
}

void DirectionalLight::setDirection(const Vec3& dir) {
    direction = dir.normalize();
}
