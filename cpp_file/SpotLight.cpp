#include "SpotLight.h"
#include <cmath>

SpotLight::SpotLight(const Vec3& pos, const Vec3& dir, const Vec3& intens, float ang, float rad)
    : angle_degrees(ang), radius(rad) {
    position = pos;
    direction = dir.normalize();
    intensity = intens;
}

Vec3 SpotLight::getDirection(const Vec3& point) const {
    // Calculate the direction from the light to the point
    return (point - position).normalize();
}
float SpotLight::pdf(const Vec3& hit_point, const Vec3& incoming_direction) const  {
    Vec3 wi = (position - hit_point).normalize();
    float cos_angle = std::cos(angle_radians);

    float cos_theta = Vec3::dot(wi, direction.normalize());
    if (cos_theta < cos_angle)
        return 0.0f;

    // Uniform konik da��l�m
    float solid_angle = 2.0f * M_PI * (1.0f - cos_angle);
    return 1.0f / solid_angle;
}

Vec3 SpotLight::getIntensity(const Vec3& point, const Vec3& light_sample_point) const {
    // Calculate the cosine of the angle between the light direction and the direction to the point
    float cos_theta = Vec3::dot(direction, getDirection(point));

    // Check if the point is within the spotlight's cone
    if (cos_theta > std::cos(angle_degrees)) {
        // Intensity falls off as a function of the cosine of the angle
        float falloff = std::pow(cos_theta, 2.0f);  // You can adjust the exponent for different falloff rates
        return intensity * falloff;
    }
    else {
        // Outside the spotlight's cone, no intensity
        return Vec3(0.0, 0.0, 0.0);
    }
}

Vec3 SpotLight::random_point() const {
    // Generate a random point within the sphere of radius `radius` around the position
    Vec3 random_offset = Vec3::random_in_unit_sphere() * radius;
    return position + random_offset;
}

LightType SpotLight::type() const {
    return LightType::Spot;
}
