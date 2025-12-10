#ifndef AREA_LIGHT_H
#define AREA_LIGHT_H

#include "Light.h"
#include "Vec3.h"
#include <random>

class AreaLight : public Light {
public:
    AreaLight(const Vec3& pos, const Vec3& u_vec, const Vec3& v_vec, float w, float h, const Vec3& input_intensity) {
        position = pos;
        width = w;
        height = h;

        // Yeni yapý: rengi normalize et, gücü ayrý sakla
        float power = input_intensity.length();
        color = (power > 0.0f) ? input_intensity / power : Vec3(1.0f);
        intensity = power;

        setUVVectors(u_vec, v_vec);
        updateArea();
    }

    Vec3 random_point() const override {
        static std::mt19937 generator(std::random_device{}());
        static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

        float rand_u = distribution(generator);
        float rand_v = distribution(generator);

        last_sampled_point = position + u * (rand_u - 0.5f) * width + v * (rand_v - 0.5f) * height;
        return last_sampled_point;
    }

    Vec3 getDirection(const Vec3& point) const override {
        return (position - point).normalize();
    }

    Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const override {
        float distance2 = (light_sample_point - point).length_squared();
        return (color * intensity) / std::max(distance2, 0.0001f);
    }

    float pdf(const Vec3& hit_point, const Vec3& incoming_direction) const override {
        Vec3 wi = last_sampled_point - hit_point;
        float dist2 = wi.length_squared();
        wi = wi.normalize();

        Vec3 light_normal = direction.normalize();
        float cos_theta = std::fmax(0.0001, Vec3::dot(-wi, light_normal));
        return dist2 / (area * cos_theta);
    }

    LightType type() const override { return LightType::Area; }

    void setUVVectors(const Vec3& u_vec, const Vec3& v_vec) {
        u = u_vec.normalize();
        v = v_vec.normalize();
        direction = Vec3::cross(u, v).normalize();
        v = Vec3::cross(direction, u).normalize();  // v'yi yeniden hesapla
        updateArea();
    }

    void setWidth(float w) { width = w; updateArea(); }
    void setHeight(float h) { height = h; updateArea(); }
    void setPosition(const Vec3& pos) { position = pos; }
    void setColor(const Vec3& c) { color = c; }
    void setIntensity(float i) { intensity = i; }

    Vec3 getU() const { return u; }
    Vec3 getV() const { return v; }
    float getWidth() const { return width; }
    float getHeight() const { return height; }

private:
    Vec3 u, v;
    float width = 1.0f;
    float height = 1.0f;
    float area = 1.0f;

    mutable Vec3 last_sampled_point;

    void updateArea() {
        area = width * height;
    }
};

#endif // AREA_LIGHT_H
