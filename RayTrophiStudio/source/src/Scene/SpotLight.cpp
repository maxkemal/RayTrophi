#include "SpotLight.h"
#include <cmath>

SpotLight::SpotLight(const Vec3& pos, const Vec3& dir, const Vec3& input_intensity, float ang, float rad) {
    position = pos;
    direction = dir.normalize();
    float power = input_intensity.length();
    color = (power > 0.0f) ? input_intensity / power : Vec3(1.0f);
    intensity = power;

    SpotLightData sData;
    sData.radius = rad;
    sData.angle_degrees = ang;
    sData.falloff = 0.1f;
    data = sData;
}

Vec3 SpotLight::getDirection(const Vec3& point) const {
    return (point - position).normalize();
}

Vec3 SpotLight::getIntensity(const Vec3& point, const Vec3& /*light_sample_point*/) const {
    float cos_theta = Vec3::dot(direction.normalize(), getDirection(point));
    float cos_angle = std::cos(getAngleDegrees() * M_PI / 180.0f);

    if (cos_theta > cos_angle) {
        float falloff_val = std::pow(cos_theta, 2.0f);
        return color * intensity * falloff_val;
    }
    else {
        return Vec3(0.0f);
    }
}

Vec3 SpotLight::random_point() const {
    Vec3 random_offset = Vec3::random_in_unit_sphere() * getRadius();
    return position + random_offset;
}

float SpotLight::pdf(const Vec3& hit_point, const Vec3& incoming_direction) const {
    Vec3 wi = (position - hit_point).normalize();
    float cos_theta = Vec3::dot(wi, direction.normalize());
    float cos_angle = std::cos(getAngleDegrees() * M_PI / 180.0f);

    if (cos_theta < cos_angle)
        return 0.0f;

    float solid_angle = 2.0f * M_PI * (1.0f - cos_angle);
    return 1.0f / solid_angle;
}

LightType SpotLight::type() const {
    return LightType::Spot;
}

void SpotLight::setAngleDegrees(float deg) {
    if (auto sData = std::get_if<SpotLightData>(&data)) {
        sData->angle_degrees = deg;
    }
}

float SpotLight::getAngleDegrees() const {
    if (auto sData = std::get_if<SpotLightData>(&data)) {
        return sData->angle_degrees;
    }
    return 45.0f;
}

void SpotLight::setFalloff(float f) {
    if (auto sData = std::get_if<SpotLightData>(&data)) {
        sData->falloff = f;
    }
}

float SpotLight::getFalloff() const {
    if (auto sData = std::get_if<SpotLightData>(&data)) {
        return sData->falloff;
    }
    return 0.1f;
}
