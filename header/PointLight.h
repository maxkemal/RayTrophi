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
   
  

    int getSampleCount() const override {
        return 16; // ince ýþýklar için daha fazla örnek
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

    Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const override {
        float distance = std::max(0.001f, (light_sample_point - point).length());
        return intensity / (distance * distance);
    }

    float pdf(const Vec3& hit_point, const Vec3& incoming_direction) const override {
        float area = 4.0f * M_PI * radius * radius;
        return 1.0f / area;

    }

    Vec3 random_point() const override {
        Vec3 dir = Vec3::random_unit_vector();
        last_sampled_point = position + dir * radius;
        return last_sampled_point;
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
    mutable Vec3 last_sampled_point;
};

#endif // POINTLIGHT_H