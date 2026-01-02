#ifndef SPOTLIGHT_H
#define SPOTLIGHT_H

#include "Light.h"
#include "Vec3.h"

class SpotLight : public Light {
public:
    SpotLight(const Vec3& pos, const Vec3& dir, const Vec3& input_intensity, float angle_deg, float rad);

    Vec3 getDirection(const Vec3& point) const override;
    Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const override;
    Vec3 random_point() const override;
    float pdf(const Vec3& hit_point, const Vec3& incoming_direction) const override;
    LightType type() const override;

    void setAngleDegrees(float deg) { angle_degrees = deg; }
    float getAngleDegrees() const { return angle_degrees; }

    void setFalloff(float f) { falloff = f; }
    float getFalloff() const { return falloff; }

private:
    float angle_degrees = 45.0f;
    float falloff = 0.1f;
};

#endif // SPOTLIGHT_H

