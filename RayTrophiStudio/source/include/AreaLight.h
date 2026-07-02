/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          AreaLight.h
* Author:        Kemal Demirtaş
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef AREA_LIGHT_H
#define AREA_LIGHT_H

#include "Light.h"
#include "Vec3.h"
#include <random>
#include <algorithm>

class AreaLight : public Light {
public:
    AreaLight(const Vec3& pos, const Vec3& u_vec, const Vec3& v_vec, float w, float h, const Vec3& input_intensity) {
        position = pos;

        // Rengi normalize et, gücü ayrı sakla
        float power = input_intensity.length();
        color = (power > 0.0f) ? input_intensity / power : Vec3(1.0f);
        intensity = power;

        AreaLightData aData;
        aData.u = u_vec.normalize();
        aData.v = v_vec.normalize();
        aData.width = w;
        aData.height = h;
        aData.area = w * h;
        data = aData;

        setUVVectors(u_vec, v_vec);
    }

    Vec3 random_point() const override {
        static std::mt19937 generator(std::random_device{}());
        static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

        float rand_u = distribution(generator);
        float rand_v = distribution(generator);

        Vec3 u_val = getU();
        Vec3 v_val = getV();
        float w = getWidth();
        float h = getHeight();

        last_sampled_point = position + u_val * (rand_u - 0.5f) * w + v_val * (rand_v - 0.5f) * h;
        return last_sampled_point;
    }

    Vec3 getDirection(const Vec3& point) const override {
        return (position - point).normalize();
    }

    Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const override {
        float distance2 = (light_sample_point - point).length_squared();
        return (color * intensity) / (std::max)(distance2, 0.0001f);
    }

    float pdf(const Vec3& hit_point, const Vec3& incoming_direction) const override {
        Vec3 wi = last_sampled_point - hit_point;
        float dist2 = wi.length_squared();
        wi = wi.normalize();

        Vec3 light_normal = direction.normalize();
        float cos_theta = std::fmax(0.0001f, Vec3::dot(-wi, light_normal));
        return dist2 / (getArea() * cos_theta);
    }

    LightType type() const override { return LightType::Area; }

    void setUVVectors(const Vec3& u_vec, const Vec3& v_vec) {
        if (auto aData = std::get_if<AreaLightData>(&data)) {
            aData->u = u_vec.normalize();
            aData->v = v_vec.normalize();
            direction = Vec3::cross(aData->u, aData->v).normalize();
            aData->v = Vec3::cross(direction, aData->u).normalize();  // v'yi yeniden hesapla
            updateArea();
        }
    }

    void setWidth(float w) {
        if (auto aData = std::get_if<AreaLightData>(&data)) {
            aData->width = w;
            updateArea();
        }
    }

    void setHeight(float h) {
        if (auto aData = std::get_if<AreaLightData>(&data)) {
            aData->height = h;
            updateArea();
        }
    }

    void setPosition(const Vec3& pos) { position = pos; }
    void setColor(const Vec3& c) { color = c; }
    void setIntensity(float i) { intensity = i; }

    Vec3 getU() const {
        if (auto aData = std::get_if<AreaLightData>(&data)) {
            return aData->u;
        }
        return Vec3(1.0f, 0.0f, 0.0f);
    }

    Vec3 getV() const {
        if (auto aData = std::get_if<AreaLightData>(&data)) {
            return aData->v;
        }
        return Vec3(0.0f, 1.0f, 0.0f);
    }

    float getWidth() const {
        if (auto aData = std::get_if<AreaLightData>(&data)) {
            return aData->width;
        }
        return 1.0f;
    }

    float getHeight() const {
        if (auto aData = std::get_if<AreaLightData>(&data)) {
            return aData->height;
        }
        return 1.0f;
    }

    float getArea() const {
        if (auto aData = std::get_if<AreaLightData>(&data)) {
            return aData->area;
        }
        return 1.0f;
    }

private:
    mutable Vec3 last_sampled_point;

    void updateArea() {
        if (auto aData = std::get_if<AreaLightData>(&data)) {
            aData->area = aData->width * aData->height;
        }
    }
};

#endif // AREA_LIGHT_H
