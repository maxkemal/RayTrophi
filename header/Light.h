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
    std::string nodeName;
    Vec3 initialDirection;
    Vec3 position;
    Vec3 intensity;
    Vec3 direction;
    Vec3 u, v;       // Alan ����� i�in d�zlem vekt�rleri
    double width, height;  // Alan ����� i�in boyutlar
	float radius; // I��k kayna��n�n yar��ap�
    virtual float pdf(const Vec3& hit_point, const Vec3& incoming_direction) const = 0;
    // Default constructor
    Light();
    virtual int getSampleCount() const { return 1; } // varsay�lan

    // Sanal y�k�c� eklendi
    virtual ~Light() = default;
    virtual Vec3 getDirection(const Vec3& point) const = 0;
    virtual Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const = 0;

    virtual Vec3 random_point() const = 0;
    virtual LightType type() const = 0;
    float intensity_magnitude = sqrt(
        pow(intensity.x, 2) +
        pow(intensity.y, 2) +
        pow(intensity.z, 2));


};

#endif // LIGHT_H
