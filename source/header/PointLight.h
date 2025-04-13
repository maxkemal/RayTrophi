#ifndef POINTLIGHT_H
#define POINTLIGHT_H

#include "Light.h"
#include "Vec3.h"
#include <random>

class PointLight : public Light {
public:
    PointLight() : position(0, 0, 0), intensity(1, 1, 1), radius(0),
        volumetricDensity(0.0f), scatteringCoef(0.5f),
        volumetricSamples(50) {
    }

    PointLight(const Vec3& pos, const Vec3& intens, float rad,
        float volDensity = 0.0f, float scatter = 0.5f, int samples = 50)
        : position(pos)
        , intensity(intens)
        , radius(rad)
        , volumetricDensity(volDensity)
        , scatteringCoef(scatter)
        , volumetricSamples(samples) {
    }

    Vec3 getDirection(const Vec3& point) const override {
        return (position - point).normalize();
    }

    Vec3 getIntensity(const Vec3& point) const override {
        float distance = (position - point).length();
        Vec3 baseIntensity = intensity / (distance * distance);

        if (volumetricDensity <= 0.0f) return baseIntensity;

        // Volumetrik efekt hesaplama
        return baseIntensity * calculateVolumetricFactor(point);
    }

    float calculateVolumetricFactor(const Vec3& point) const {
        float distance = (position - point).length();
        float accum = 0.0f;

        // Ray marching ile volumetrik efekt hesaplama
        float stepSize = distance / volumetricSamples;
        Vec3 direction = getDirection(point);

        for (int i = 0; i < volumetricSamples; ++i) {
            float t = i * stepSize;
            Vec3 samplePoint = point + direction * t;

            // Beer-Lambert yasasý ile ýþýk zayýflamasý
            float density = volumetricDensity * exp(-t * scatteringCoef);
            accum += density * stepSize;
        }

        return exp(-accum);
    }

    Vec3 random_point() const override {
      
        static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        static std::mt19937 generator(std::random_device{}());
      
        float theta = 2.0f * M_PI * dis(generator);
        float phi = acos(2.0f * dis(generator) - 1.0f);
        float r = radius * std::cbrt(dis(generator));

        float x = r * sin(phi) * cos(theta);
        float y = r * sin(phi) * sin(theta);
        float z = r * cos(phi);

        return position + Vec3(x, y, z);
    }

    LightType type() const override { return LightType::Point; }

    // Getter ve Setter metodlarý
    Vec3 getPosition() const { return position; }
    Vec3 getIntensity() const { return intensity; }
    float getRadius() const { return radius; }
    float getVolumetricDensity() const { return volumetricDensity; }
    float getScatteringCoef() const { return scatteringCoef; }
    int getVolumetricSamples() const { return volumetricSamples; }

    void setPosition(const Vec3& pos) { position = pos; }
    void setIntensity(const Vec3& intens) { intensity = intens; }
    void setRadius(float rad) { radius = rad; }
    void setVolumetricDensity(float density) { volumetricDensity = density; }
    void setScatteringCoef(float coef) { scatteringCoef = coef; }
    void setVolumetricSamples(int samples) { volumetricSamples = samples; }

public:
    Vec3 position;
    Vec3 intensity;

private:
    float radius;
    float volumetricDensity;  // Volumetrik efekt yoðunluðu
    float scatteringCoef;     // Iþýk saçýlma katsayýsý
    int volumetricSamples;    // Ray marching örnek sayýsý
};

#endif // POINTLIGHT_H