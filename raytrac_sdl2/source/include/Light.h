#ifndef LIGHT_H
#define LIGHT_H

#include "Vec3.h"

enum class LightType : int {
    Point = 0,
    Directional = 1,
    Spot = 2,
    Area = 3
};

class Light {
public:
    std::string nodeName;
    Vec3 position;
    Vec3 direction;
    Vec3 initialDirection;

    Vec3 color = Vec3(1.0f);    // normalize edilmiş renk (0–1 arası)
    float intensity = 1.0f;     // toplam enerji (lümen ya da key-value olarak)
    float radius = 0.0f;
    // Alan ışığı için
    Vec3 u, v;
    float width = 1.0f, height = 1.0f;
  

    virtual float pdf(const Vec3& hit_point, const Vec3& incoming_direction) const = 0;
    virtual Vec3 getDirection(const Vec3& point) const = 0;

    // Işık yoğunluğunu hesaplayan fonksiyon
    virtual Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const {
        return color * intensity; // varsayılan davranış: sabit ışık
    }
    void setRadius(float rad) { radius = rad; }
    float getRadius() const { return radius; }
    virtual Vec3 random_point() const = 0;
    virtual LightType type() const = 0;

    virtual int getSampleCount() const { return 1; }

    Light() = default;
    virtual ~Light() = default;
};


#endif // LIGHT_H

