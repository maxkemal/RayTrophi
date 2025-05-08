#ifndef AREA_LIGHT_H
#define AREA_LIGHT_H

#include "Light.h"
#include "Vec3.h"

class AreaLight : public Light {
public:
    AreaLight(const Vec3& pos, const Vec3& u_vec, const Vec3& v_vec, float w, float h, const Vec3& intens)
        : position(pos), width(w), height(h), intensity(intens) {
        setUVVectors(u_vec, v_vec);
        updateArea();  // Alan hesaplamasý
    }
    Vec3 position;
    // Set metodlarý
    void setPosition(const Vec3& pos) { position = pos; }
    void setU(const Vec3& u_vec) { u = u_vec; direction = u.cross(v).normalize(); }
    void setV(const Vec3& v_vec) { v = v_vec; direction = u.cross(v).normalize(); }
    void setWidth(float w) { width = w; }
    void setHeight(float h) { height = h; }
    void setIntensity(const Vec3& intens) { intensity = intens; }

    Vec3 random_point() const override {

        static std::mt19937 generator(std::random_device{}());
        static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

        float rand_u = distribution(generator);
        float rand_v = distribution(generator);

        return position + u * (rand_u - 0.5f) * width + v * (rand_v - 0.5f) * height;
    }

    // Get metodlarý
    Vec3 getPosition() const { return position; }
    Vec3 getU() const { return u; }
    Vec3 getV() const { return v; }
    double getWidth() const { return width; }
    double getHeight() const { return height; }
    Vec3 getIntensity() const { return intensity; }
    Vec3 getDirection(const Vec3& point) const {
        return (position - point).normalize();
    }

    Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const override {
        float distance = std::max(0.001f, (light_sample_point - point).length());
        return intensity / (distance * distance);
    }

    float pdf(const Vec3& hit_point, const Vec3& incoming_direction) const override {
        Vec3 light_point = last_sampled_point; // random_point() ile belirlenmiþ olmalý
        Vec3 wi = light_point - hit_point;
        float dist2 = wi.length_squared();
        wi = wi.normalize();

        // Yüzey normali
        Vec3 light_normal = light_normal.normalize();
        float cos_theta = std::max(0.0001, Vec3::dot(-wi, light_normal));

        float area = width * height; // Dikdörtgen için

        return dist2 / (area * cos_theta);
    }

    LightType type() const override { return LightType::Area; }
    // UV vektörlerinin güncellenmesi için güvenli metod
    void setUVVectors(const Vec3& u_vec, const Vec3& v_vec) {
        // Vektörlerin normalize edilmesi
        u = u_vec.normalize();
        v = v_vec.normalize();

        // Ortogonalliðin saðlanmasý
        direction = Vec3::cross(u, v).normalize();
        v = Vec3::cross(direction, u).normalize();  // v'yi yeniden hesapla

        updateArea();
    }
private:
	mutable Vec3 last_sampled_point; // Son örneklenen nokta
    Vec3 u;
    Vec3 v;
    float width;
    float height;
    Vec3 intensity;
    Vec3 direction;
    float area;    // Önhesaplanmýþ alan
    // Alan hesaplamasý için yardýmcý fonksiyon
    void updateArea() {
        area = width * height;
    }

    // Iþýðýn etki alanýný hesaplamak için yardýmcý fonksiyon
    float calculateInfluenceRadius() const {
        // Iþýk þiddetine ve alana baðlý olarak etki yarýçapý
        return std::sqrt(std::max(intensity.x, std::max(intensity.y, intensity.z)) * area);
    }
};

#endif // AREA_LIGHT_H
