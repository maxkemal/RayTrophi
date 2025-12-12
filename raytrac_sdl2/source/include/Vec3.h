#ifndef VEC3_H
#define VEC3_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <array>
#include <limits>
#include <stdexcept> // Hata yönetimi için

// M_PI'yi float olarak tanýmla
#define M_PI 3.14159265358979323846f

class Vec3 {
public:
    // Data members
    float x, y, z;

    // Constructors
    Vec3();
    Vec3(float value);
    Vec3(float x, float y, float z);
    // double constructor'u kaldýrýldý, float array constructor'ý eklenebilir.

    // Access operators
    float operator[](int i) const;
    float& operator[](int i);

    // Arithmetic operators
    Vec3 operator-() const;
    Vec3 operator+(const Vec3& v) const;
    Vec3 operator-(const Vec3& v) const;
    Vec3 operator*(const Vec3& v) const;
    Vec3 operator*(float t) const;
    Vec3 operator/(float scalar) const;
    Vec3 operator/(const Vec3& v) const;
    Vec3 operator+(float scalar) const;
    Vec3 operator-(float scalar) const;

    // Compound assignment operators
    Vec3& operator+=(const Vec3& v);
    Vec3& operator-=(const Vec3& v);
    Vec3& operator*=(float t);
    Vec3& operator*=(const Vec3& v);
    Vec3& operator/=(float t);

    // Comparison operators (Hassasiyet için)
    bool operator==(const Vec3& other) const;
    bool operator!=(const Vec3& other) const;

    // Vector operations
    float length() const;
    float length_squared() const;
    Vec3 normalize() const;
    Vec3 cross(const Vec3& v) const;
    Vec3 abs() const;
    Vec3 orient(const Vec3& local) const;
    Vec3 cwiseProduct(const Vec3& v) const;

    // Component-wise operations
    float max_component() const;
    bool near_zero() const;
    float luminance() const;
    std::string toString() const;

    // Color aliases
    float& r() { return x; }
    float& g() { return y; }
    float& b() { return z; }
    const float& r() const { return x; }
    const float& g() const { return y; }
    const float& b() const { return z; }

    // Static utility functions
    inline float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    static float dot(const Vec3& v1, const Vec3& v2);
    static Vec3 cross(const Vec3& v1, const Vec3& v2);
    static Vec3 reflect(const Vec3& v, const Vec3& n);
    // refract metodunu float parametreler kullanacak þekilde düzenle
    static Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat);
    static Vec3 min(const Vec3& a, const Vec3& b);
    static Vec3 max(const Vec3& a, const Vec3& b);
    static Vec3 clamp(const Vec3& vec, float min, float max);
    static Vec3 lerp(const Vec3& a, const Vec3& b, float t);
    static Vec3 lerp(const Vec3& a, const Vec3& b, const Vec3& t);
    static float lerpf(float a, float b, float t);
    static Vec3 mix(const Vec3& a, const Vec3& b, float t);
    static Vec3 mix(const Vec3& a, const Vec3& b, const Vec3& t);
    static Vec3 exp(const Vec3& v);
    static float average(const Vec3& v);

    // Random generation functions (Hepsi float kullanacak)
    static Vec3 random();
    static Vec3 random(float min, float max); // double -> float
    static Vec3 random_in_unit_sphere();
    static Vec3 random_in_hemisphere(const Vec3& normal);
    static Vec3 random_unit_vector();
    static Vec3 random_in_unit_disk();
    static Vec3 random_cosine_direction(const Vec3& normal);
    static Vec3 sample_hemisphere_cosine_weighted(const Vec3& normal, float u, float v);
    static Vec3 sphericalDirection(float sinTheta, float cosTheta, float phi);
    static Vec3 from_spherical(float theta, float phi, float r = 1.0f); // double -> float

    // Random number utilities
    static float random_float(float min = 0.0f, float max = 1.0f); // double -> float

    // Conversion operators (Kaldýrýldý veya float array'e çevrildi)
    // operator std::array<double, 3>() const; // Kaldýrýldý

    // Friend functions
    friend std::ostream& operator<<(std::ostream& os, const Vec3& v);
    friend Vec3 operator*(float t, const Vec3& v); // double -> float
    friend Vec3 operator/(float scalar, const Vec3& v);
};

// Non-member functions
Vec3 unit_vector(const Vec3& v);
// double operatör kaldýrýldý.

#endif // VEC3_H