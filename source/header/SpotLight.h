#ifndef SPOTLIGHT_H
#define SPOTLIGHT_H

#include "Light.h"
#include "Vec3SIMD.h"
#include "Vec3.h"
class SpotLight : public Light {
public:
    SpotLight(const Vec3& pos, const Vec3& dir, const Vec3& intens, float ang, float rad);
    Vec3 position;
    float angle_degrees = 30.0f;
    float angle_radians = angle_degrees * (M_PI / 180.0f);
    float radius;
    Vec3 getDirection(const Vec3& point) const override;
    Vec3 getIntensity(const Vec3& point) const override;
    Vec3 random_point() const override;
    LightType type() const override;

private:
   
   
    
};

#endif // SPOTLIGHT_H
