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
   
    PointLight() {
        position = Vec3(0.0f);
        color = Vec3(1.0f);
        intensity = 1.0f;
        
        PointLightData pData;
        pData.radius = 0.1f;
        pData.volumetricDensity = 0.0f;
        pData.scatteringCoef = 0.5f;
        pData.volumetricSamples = 50;
        data = pData;
    }

    PointLight(const Vec3& pos, const Vec3& input_intens, float rad,
        float volDensity = 0.0f, float scatter = 0.5f, int samples = 50) {
        position = pos;
        float power = input_intens.length();
        color = (power > 0.0f) ? input_intens / power : Vec3(1.0f);
        intensity = power;
        
        PointLightData pData;
        pData.radius = rad;
        pData.volumetricDensity = volDensity;
        pData.scatteringCoef = scatter;
        pData.volumetricSamples = samples;
        data = pData;
    }

    Vec3 getDirection(const Vec3& point) const override {
        return (position - point).normalize();
    }

    Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const override {
        float distance2 = (light_sample_point - point).length_squared();
        return (color * intensity) / (std::max)(distance2, 0.0001f);
    }

    float pdf(const Vec3& /*hit_point*/, const Vec3& /*incoming_direction*/) const override {
        float rad = getRadius();
        float area = 4.0f * M_PI * rad * rad;
        return 1.0f / (std::max)(area, 0.0001f);
    }

    Vec3 random_point() const override {
        float rad = getRadius();
        Vec3 dir = Vec3::random_unit_vector();
        last_sampled_point = position + dir * rad;
        return last_sampled_point;
    }

    int getSampleCount() const override {
        return 16;
    }

    float calculateVolumetricFactor(const Vec3& point) const {
        float distance = (position - point).length();
        float accum = 0.0f;

        int samples = getVolumetricSamples();
        float stepSize = distance / samples;
        Vec3 direction = getDirection(point);

        float densityVal = getVolumetricDensity();
        float coefVal = getScatteringCoef();

        for (int i = 0; i < samples; ++i) {
            float t = i * stepSize;
            Vec3 samplePoint = point + direction * t;
            float density = densityVal * std::exp(-t * coefVal);
            accum += density * stepSize;
        }

        return std::exp(-accum);
    }

    LightType type() const override { return LightType::Point; }

    // Getter ve setter'lar
    Vec3 getPosition() const { return position; }
    float getVolumetricDensity() const {
        if (auto pData = std::get_if<PointLightData>(&data)) {
            return pData->volumetricDensity;
        }
        return 0.0f;
    }
    float getScatteringCoef() const {
        if (auto pData = std::get_if<PointLightData>(&data)) {
            return pData->scatteringCoef;
        }
        return 0.5f;
    }
    int getVolumetricSamples() const {
        if (auto pData = std::get_if<PointLightData>(&data)) {
            return pData->volumetricSamples;
        }
        return 50;
    }

    void setPosition(const Vec3& pos) { position = pos; }
    void setVolumetricDensity(float density) {
        if (auto pData = std::get_if<PointLightData>(&data)) {
            pData->volumetricDensity = density;
        }
    }
    void setScatteringCoef(float coef) {
        if (auto pData = std::get_if<PointLightData>(&data)) {
            pData->scatteringCoef = coef;
        }
    }
    void setVolumetricSamples(int samples) {
        if (auto pData = std::get_if<PointLightData>(&data)) {
            pData->volumetricSamples = samples;
        }
    }

private:   
    mutable Vec3 last_sampled_point;
};

#endif // POINTLIGHT_H
