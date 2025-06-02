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
    Vec3 position;
    Vec3 direction;
    Vec3 initialDirection;

    Vec3 color = Vec3(1.0f);    // normalize edilmi� renk (0�1 aras�)
    float intensity = 1.0f;     // toplam enerji (l�men ya da key-value olarak)

    // Alan ����� i�in
    Vec3 u, v;
    double width = 1.0, height = 1.0;
    float radius = 0.0f;

    virtual float pdf(const Vec3& hit_point, const Vec3& incoming_direction) const = 0;
    virtual Vec3 getDirection(const Vec3& point) const = 0;

    // I��k yo�unlu�unu hesaplayan fonksiyon
    virtual Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const {
        return color * intensity; // varsay�lan davran��: sabit ���k
    }

    virtual Vec3 random_point() const = 0;
    virtual LightType type() const = 0;

    virtual int getSampleCount() const { return 1; }

    Light() = default;
    virtual ~Light() = default;
};


#endif // LIGHT_H
