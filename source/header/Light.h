#ifndef LIGHT_H
#define LIGHT_H

#include "Vec3.h"

enum class LightType {
    Point,
    Directional,
    Spot,
    Area
};

class Light {
public:
    Vec3 position;
    Vec3 intensity;
    Vec3 direction;
    Vec3 u, v;       // Alan ýþýðý için düzlem vektörleri
    double width, height;  // Alan ýþýðý için boyutlar

    // Default constructor
    Light();

    // Sanal yýkýcý eklendi
    virtual ~Light() = default;
    virtual Vec3 getDirection(const Vec3& point) const = 0;
    virtual Vec3 getIntensity(const Vec3& point) const = 0;
    virtual Vec3 random_point() const = 0;
    virtual LightType type() const = 0;
};

#endif // LIGHT_H
