#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include "Light.h"
#include "Vec3.h"
#include <random>

class DirectionalLight : public Light {
public:
    DirectionalLight(const Vec3& dir, const Vec3& input_intens, float rad);

    int getSampleCount() const override;
    Vec3 getDirection(const Vec3& point) const override;
    Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const override;
    Vec3 random_point() const override;
    LightType type() const override;
    float pdf(const Vec3& hit_point, const Vec3& incoming_direction) const override;

    void setDirection(const Vec3& dir);
    void setDiskRadius(float r) { radius = r; }
    float getDiskRadius() const { return radius; }

private:
    
    mutable Vec3 last_sampled_point;
};

#endif // DIRECTIONAL_LIGHT_H
