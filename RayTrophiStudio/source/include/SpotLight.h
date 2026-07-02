/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          SpotLight.h
* Author:        Kemal Demirtaş
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
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

    void setAngleDegrees(float deg);
    float getAngleDegrees() const;

    void setFalloff(float f);
    float getFalloff() const;

private:
    mutable Vec3 last_sampled_point;
};

#endif // SPOTLIGHT_H
