#ifndef AREA_LIGHT_H
#define AREA_LIGHT_H

#include "Light.h"
#include "Vec3.h"

class AreaLight : public Light {
public:
    AreaLight(const Vec3& pos, const Vec3& u_vec, const Vec3& v_vec, float w, float h, const Vec3& intens)
        : position(pos), width(w), height(h), intensity(intens) {
        setUVVectors(u_vec, v_vec);
        updateArea();  // Alan hesaplamas�
    }
    Vec3 position;
    // Set metodlar�
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

    // Get metodlar�
    Vec3 getPosition() const { return position; }
    Vec3 getU() const { return u; }
    Vec3 getV() const { return v; }
    double getWidth() const { return width; }
    double getHeight() const { return height; }
    Vec3 getIntensity() const { return intensity; }
    Vec3 getDirection(const Vec3& point) const {
        return (position - point).normalize();
    }

    Vec3 getIntensity(const Vec3& point) const {
        Vec3 to_point = point - position;
        float distance = to_point.length();

        // Y�n� normalize et
        Vec3 light_dir = to_point / distance; // ya da to_point.normalize()

        // cos_theta hesaplamas�
        float cos_theta = Vec3::dot(direction, light_dir);
        // Negatif de�erleri s�f�ra zorla
        cos_theta = cos_theta > 0.0f ? cos_theta : 0.0f;

        // Mesafe ve a��ya ba�l� zay�flama
        return intensity * (cos_theta / (distance * distance));
    }

    LightType type() const override { return LightType::Area; }
    // UV vekt�rlerinin g�ncellenmesi i�in g�venli metod
    void setUVVectors(const Vec3& u_vec, const Vec3& v_vec) {
        // Vekt�rlerin normalize edilmesi
        u = u_vec.normalize();
        v = v_vec.normalize();

        // Ortogonalli�in sa�lanmas�
        direction = Vec3::cross(u, v).normalize();
        v = Vec3::cross(direction, u).normalize();  // v'yi yeniden hesapla

        updateArea();
    }
private:
  
    Vec3 u;
    Vec3 v;
    float width;
    float height;
    Vec3 intensity;
    Vec3 direction;
    float area;    // �nhesaplanm�� alan
    // Alan hesaplamas� i�in yard�mc� fonksiyon
    void updateArea() {
        area = width * height;
    }

    // I����n etki alan�n� hesaplamak i�in yard�mc� fonksiyon
    float calculateInfluenceRadius() const {
        // I��k �iddetine ve alana ba�l� olarak etki yar��ap�
        return std::sqrt(std::max(intensity.x, std::max(intensity.y, intensity.z)) * area);
    }
};

#endif // AREA_LIGHT_H
