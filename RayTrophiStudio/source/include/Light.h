/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Light.h
* Author:        Kemal Demirtaş
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef LIGHT_H
#define LIGHT_H

#include "Vec3.h"
#include <string>
#include <variant>

enum class LightType : int {
    Point = 0,
    Directional = 1,
    Spot = 2,
    Area = 3
};

struct PointLightData {
    float radius = 0.0f;
    float volumetricDensity = 0.0f;
    float scatteringCoef = 0.5f;
    int volumetricSamples = 50;
};

struct DirectionalLightData {
    float radius = 0.0f; // disk radius for soft shadows
    Vec3 initialDirection;
};

struct SpotLightData {
    float radius = 0.0f; // source radius for sampling
    float angle_degrees = 45.0f;
    float falloff = 0.1f;
};

struct AreaLightData {
    Vec3 u = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 v = Vec3(0.0f, 0.0f, 1.0f);
    float width = 1.0f;
    float height = 1.0f;
    float area = 1.0f;
};

using LightVariantData = std::variant<PointLightData, DirectionalLightData, SpotLightData, AreaLightData>;

class Light {
public:
    std::string nodeName;
    bool visible = true;
    Vec3 position;
    Vec3 direction;
    Vec3 initialDirection;

    Vec3 color = Vec3(1.0f);    // normalize edilmiş renk (0–1 arası)
    float intensity = 1.0f;     // toplam enerji (lümen ya da key-value olarak)
    
    LightVariantData data = PointLightData{};

    virtual float pdf(const Vec3& hit_point, const Vec3& incoming_direction) const = 0;
    virtual Vec3 getDirection(const Vec3& point) const = 0;

    // Işık yoğunluğunu hesaplayan fonksiyon
    virtual Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const {
        return color * intensity; // varsayılan davranış: sabit ışık
    }
    
    void setRadius(float rad) {
        std::visit([rad](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, PointLightData> || 
                          std::is_same_v<T, DirectionalLightData> || 
                          std::is_same_v<T, SpotLightData>) {
                arg.radius = rad;
            }
        }, data);
    }
    
    float getRadius() const {
        return std::visit([](auto&& arg) -> float {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, PointLightData> || 
                          std::is_same_v<T, DirectionalLightData> || 
                          std::is_same_v<T, SpotLightData>) {
                return arg.radius;
            }
            return 0.0f;
        }, data);
    }
    
    virtual Vec3 random_point() const = 0;
    virtual LightType type() const = 0;

    virtual int getSampleCount() const { return 1; }

    Light() = default;
    virtual ~Light() = default;
};

#endif // LIGHT_H
