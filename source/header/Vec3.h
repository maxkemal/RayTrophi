#ifndef VEC3_H
#define VEC3_H

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <algorithm> // std::clamp için
#include <random>
#include <string>
#include <array>
#define M_PI 3.14159265358979323846

template <typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}



template<typename T>
T clamp(const T& value, const T& min, const T& max) {
    return std::max(min, std::min(max, value));
}

class Vec3 {
public:
    float x, y, z;
    // Vec3 / float operatörü
    Vec3 operator/(float scalar) const {
        return Vec3(x / scalar, y / scalar, z / scalar);
    }
    // std::array<double, 3> tipine dönüþüm operatörü
    operator std::array<double, 3>() const {
        return { x, y, z };
    }
    // std::array<double, 3> tipinden Vec3'e dönüþüm için konstruktör
    Vec3(const std::array<double, 3>& arr) : x(arr[0]), y(arr[1]), z(arr[2]) {}
    // float / Vec3 operatörü
    friend Vec3 operator/(float scalar, const Vec3& v) {
        return Vec3(scalar / v.x, scalar / v.y, scalar / v.z);
    }
    friend std::ostream& operator<<(std::ostream& os, const Vec3& vec) {
        os << vec.x << ", " << vec.y << ", " << vec.z;
        return os;
    }
    static float average(const Vec3& v) {
        return (v.x + v.y + v.z) / 3.0f;
    }
    static Vec3 mix(const Vec3& a, const Vec3& b, float t) {
        return Vec3(
            a.x * (1.0f - t) + b.x * t,
            a.y * (1.0f - t) + b.y * t,
            a.z * (1.0f - t) + b.z * t
        );
    }
    static Vec3 mix(const Vec3& a, const Vec3& b, const Vec3& t) {
        return Vec3(
            a.x * (1.0f - t.x) + b.x * t.x,
            a.y * (1.0f - t.y) + b.y * t.y,
            a.z * (1.0f - t.z) + b.z * t.z
        );
    }
    static Vec3 exp(const Vec3& v) {
        return Vec3(std::exp(v.x), std::exp(v.y), std::exp(v.z));
    }
    std::string toString() const {
        return "(" + std::to_string(x) + ", " +
            std::to_string(y) + ", " +
            std::to_string(z) + ")";
    }
    // Bu tanýmlamalar, r, g, b isimleriyle ayný iþlevi görecek.
    float& r() { return x; }
    float& g() { return y; }
    float& b() { return z; }
    const float& r() const { return x; }
    const float& g() const { return y; }
    const float& b() const { return z; }
    static Vec3 clamp(const Vec3& vec, float min, float max) {
        if (min > max) std::swap(min, max);
        return Vec3(std::clamp(vec.x, min, max),
            std::clamp(vec.y, min, max),
            std::clamp(vec.z, min, max));
    }
   
    float luminance() const {
        return 0.2126f * x + 0.7152f * y + 0.0722f * z;
    }

    // Yeni kurucu: tek float deðerle baþlatma
    Vec3(float value);
    Vec3 operator*(const Vec3& other) const;
    Vec3();
    Vec3(float x, float y, float z);
    Vec3 orient(const Vec3& local) const {
        Vec3 tangent = (std::abs(z) > 0.999f) ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(0.0f, 0.0f, 1.0f);
        Vec3 bitangent = cross(tangent).normalize();
        tangent = bitangent.cross(*this); // *this = N

        return (tangent * local.x + bitangent * local.y + (*this) * local.z).normalize();
    }

    bool near_zero() const {
        const auto s = std::numeric_limits<float>::epsilon();
        return (fabs(x) < s) && (fabs(y) < s) && (fabs(z) < s);
    }
    Vec3 operator-(float scalar) const {
        return Vec3(x - scalar, y - scalar, z - scalar);
    }
    Vec3 operator+(float scalar) const {
        return Vec3(x + scalar, y + scalar, z + scalar);
    }
    Vec3 operator/(const Vec3& v) const {
        return Vec3(x / v.x, y / v.y, z / v.z);
    }
    Vec3 operator-() const;
    Vec3& operator+=(const Vec3& v);
    Vec3& operator*=(const float t);
    Vec3& operator/=(const float t);
    Vec3 operator+(const Vec3& v) const;
    Vec3 operator-(const Vec3& v) const;
    //Vec3 operator*(const Vec3& v) const;
    Vec3 operator*(float t) const;
    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }
    static Vec3 min(const Vec3& a, const Vec3& b);
    static Vec3 max(const Vec3& a, const Vec3& b);
    Vec3& operator*=(const Vec3& other);
    //Vec3& operator*=(const Vec3& rhs);
    float length_squared() const;
    float operator[](int i) const;
    float& operator[](int i);
    float max_component() const;
    static inline Vec3 random_cosine_direction(const Vec3& normal) {
        double r1 = random_double();
        double r2 = random_double();

        // Cosinüs aðýrlýklý hemisphere sampling için küresel koordinatlar
        double phi = 2 * M_PI * r1;
        double cos_theta = sqrt(1.0f - r2);  // r2 yerine (1-r2) kullanarak daha iyi daðýlým
        double sin_theta = sqrt(r2);

        // Kartezyen koordinatlara dönüþüm
        double x = cos(phi) * sin_theta;
        double y = sin(phi) * sin_theta;
        double z = cos_theta;

        // Koordinat sistemi oluþturma (daha stabil bir versiyon)
        Vec3 u;
        if (fabs(normal.x) > fabs(normal.y)) {
            u = Vec3(-normal.z, 0, normal.x);
            u /= sqrt(normal.x * normal.x + normal.z * normal.z);
        }
        else {
            u = Vec3(0, normal.z, -normal.y);
            u /= sqrt(normal.y * normal.y + normal.z * normal.z);
        }
        Vec3 v = normal.cross(u);

        // Dünya koordinat sistemine dönüþüm
        return (u * x + v * y + normal * z).normalize();
    }
    static Vec3 from_spherical(double theta, double phi, double r);
    static Vec3 random_in_unit_sphere();
    static  Vec3 random_in_hemisphere(const Vec3& normal) ;
    inline double dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Vec3 cwiseProduct(const Vec3& v) const;
    //Vec3 clamp(double min, double max) const;
    float length() const;
    Vec3 normalize() const;

    static Vec3 random();
    static Vec3 random(double min, double max);
//static Vec3 random_in_unit_sphere();
    static Vec3 random_unit_vector();
    static Vec3 reflect(const Vec3& v, const Vec3& n);
    static Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat);
   static Vec3 random_in_unit_disk();
    static Vec3 random_in_unit_hemisphere(const Vec3& normal) {
        Vec3 random_vec = random_in_unit_sphere();
        // Eðer normal ile ayný yönde deðilse ters çevir
        if (dot(random_vec, normal) < 0.0)
            return -random_vec;
        return random_vec;
    }
    
    static double random_double(double min = 0.0, double max = 1.0) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<double> dis(0.0, 1.0);
        return min + (max - min) * dis(gen);
    }
    static  inline Vec3 sample_hemisphere_cosine_weighted(const Vec3& normal, float u, float v) {
        float r = std::sqrt(u);
        float theta = 2.0f * M_PI * v;

        float x = r * std::cos(theta);
        float y = r * std::sin(theta);
        float z = std::sqrt(std::max(0.0f, 1.0f - u)); // Ensures z is always positive

        // Orthonormal Basis Construction (Tangent Space)
        Vec3 up = std::abs(normal.y) > 0.99f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
        Vec3 tangent = up.cross(normal).normalize();
        Vec3 bitangent = normal.cross(tangent);

        // Convert local sample to world space
        return (tangent * x) + (bitangent * y) + (normal * z);
    }


   static Vec3 sphericalDirection(float sinTheta, float cosTheta, float phi) {
        float x = sinTheta * cos(phi);
        float y = sinTheta * sin(phi);
        float z = cosTheta;
        return Vec3(x, y, z);
    }

    // Baðýmsýz fonksiyon olarak unit_vector
    friend Vec3 unit_vector(const Vec3& v);
   
    static Vec3 lerp(const Vec3& a, const Vec3& b, const Vec3& t) {
        return a * (Vec3(1.0) - t) + b * t;
    }

    static double dot(const Vec3& v1, const Vec3& v2);
    static Vec3 cross(const Vec3& v1, const Vec3& v2);

    friend std::ostream& operator<<(std::ostream& os, const Vec3& v);
    Vec3 cross(const Vec3& v) const {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
   Vec3 abs() const {
        return Vec3(std::fabs(x), std::fabs(y), std::fabs(z));
    }
    static Vec3 lerp(const Vec3& a, const Vec3& b, float t) {
        return Vec3(
            a.x + t * (b.x - a.x),
            a.y + t * (b.y - a.y),
            a.z + t * (b.z - a.z)
        );
    }
    static double lerp(double a, double b, double t) {
        return a * (1.0 - t) + b * t;
    }

    // Vec3 sýnýfýna "==" operatörü ekleme
     bool operator==(const Vec3& other) const {
        // Epsilon deðeri (çok küçük farklar için tolerans)
        const float epsilon = std::numeric_limits<float>::epsilon() * 100.0f;

        // x, y ve z bileþenlerinin yaklaþýk olarak eþit olup olmadýðýný kontrol et
        return (std::abs(x - other.x) < epsilon &&
            std::abs(y - other.y) < epsilon &&
            std::abs(z - other.z) < epsilon);
    }

   
    bool operator!=(const Vec3& other) const {
        return !(*this == other);
    }
   
};
Vec3 unit_vector(const Vec3& v);
Vec3 operator*(double t, const Vec3& v);
// Function declarations
double random_double(double min, double max);
double random_double();
Vec3 operator*(double t, const Vec3& v);
#endif // VEC3_H
