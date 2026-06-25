/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          PointLight.h
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef POINTLIGHT_H
#define POINTLIGHT_H

#include "Light.h"
#include "Vec3.h"
#include <random>
#include <cmath>

class PointLight : public Light {
public:
   
    PointLight()
        :volumetricDensity(0.0f),
        scatteringCoef(0.5f),
        volumetricSamples(50) {
        position = Vec3(0.0f);
        color = Vec3(1.0f);
        intensity = 1.0f;
		radius = 0.1f; // Varsayï¿½lan yarï¿½ï¿½ap
    }

    PointLight(const Vec3& pos, const Vec3& input_intens, float rad,
        float volDensity = 0.0f, float scatter = 0.5f, int samples = 50)
        :
        volumetricDensity(volDensity),
        scatteringCoef(scatter),
        volumetricSamples(samples) {
        position = pos;
        float power = input_intens.length();
        color = (power > 0.0f) ? input_intens / power : Vec3(1.0f);
        intensity = power;
		radius = rad;
    }

    Vec3 getDirection(const Vec3& point) const override {
        return (position - point).normalize();
    }

    Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const override {
        float distance2 = (light_sample_point - point).length_squared();
        return (color * intensity) / (std::max)(distance2, 0.0001f);
    }

    float pdf(const Vec3& /*hit_point*/, const Vec3& /*incoming_direction*/) const override {
        float area = 4.0f * M_PI * radius * radius;
        return 1.0f / (std::max)(area, 0.0001f);
    }

    Vec3 random_point() const override {
        Vec3 dir = Vec3::random_unit_vector();
        last_sampled_point = position + dir * radius;
        return last_sampled_point;
    }

    int getSampleCount() const override {
        return 16;
    }

    float calculateVolumetricFactor(const Vec3& point) const {
        float distance = (position - point).length();
        float accum = 0.0f;

        float stepSize = distance / volumetricSamples;
        Vec3 direction = getDirection(point);

        for (int i = 0; i < volumetricSamples; ++i) {
            float t = i * stepSize;
            Vec3 samplePoint = point + direction * t;
            float density = volumetricDensity * std::exp(-t * scatteringCoef);
            accum += density * stepSize;
        }

        return std::exp(-accum);
    }

    LightType type() const override { return LightType::Point; }

    // Getter ve setter'lar
    Vec3 getPosition() const { return position; }
    float getRadius() const { return radius; }
    float getVolumetricDensity() const { return volumetricDensity; }
    float getScatteringCoef() const { return scatteringCoef; }
    int getVolumetricSamples() const { return volumetricSamples; }

    void setPosition(const Vec3& pos) { position = pos; }
    void setRadius(float rad) { radius = rad; }
    void setVolumetricDensity(float density) { volumetricDensity = density; }
    void setScatteringCoef(float coef) { scatteringCoef = coef; }
    void setVolumetricSamples(int samples) { volumetricSamples = samples; }

private:   
    float volumetricDensity;
    float scatteringCoef;
    int volumetricSamples;
    mutable Vec3 last_sampled_point;
  
};

#endif // POINTLIGHT_H

