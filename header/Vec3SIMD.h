#ifndef VEC3SIMD_H
#define VEC3SIMD_H

#include <immintrin.h> // AVX header
#include "Vec3.h" // Assuming Vec3 class is defined here
#include <iostream>
#include <random>
class Vec3SIMD {
public:
    __m256 data;

    Vec3SIMD();
    Vec3SIMD(__m256 d);
    Vec3SIMD(const Vec3& v);
    Vec3SIMD(float x, float y, float z);

    float get(int index) const;
    double operator[](int i) const {
        if (i == 0) return x();
        if (i == 1) return y();
        if (i == 2) return z();
        throw std::out_of_range("Vec3 index out of range");
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec3SIMD& vec);
    operator Vec3() const;

    static Vec3SIMD mix(const Vec3SIMD& a, const Vec3SIMD& b, float t) ;
    Vec3SIMD operator-(double scalar) const {
        // Double'ý vektör boyutuna dönüþtür
        __m256 scalar_vec = _mm256_set1_ps(static_cast<float>(scalar));
        // Çýkarma iþlemi
        __m256 result = _mm256_sub_ps(data, scalar_vec);
        return Vec3SIMD(result);
    }
 
    // Vektör ile double arasýndaki toplama iþlemi
    Vec3SIMD operator+(double scalar) const {
        // Double'ý vektör boyutuna dönüþtür
        __m256 scalar_vec = _mm256_set1_ps(static_cast<float>(scalar));
        // Toplama iþlemi
        __m256 result = _mm256_add_ps(data, scalar_vec);
        return Vec3SIMD(result);
    }
    // Eþitlik operatörü
    bool operator==(const Vec3SIMD& other) const {
        // Epsilon deðeri, küçük sayýsal farklýlýklarý tolere etmek için kullanýlýr
        const float epsilon = 1e-6f;
        return std::abs(x() - other.x()) < epsilon &&
            std::abs(y() - other.y()) < epsilon &&
            std::abs(z() - other.z()) < epsilon;
    }

    // Eþit olmama operatörü
    bool operator!=(const Vec3SIMD& other) const {
        return !(*this == other);
    }

    static Vec3SIMD clamp(const Vec3SIMD& v, float minVal, float maxVal) {
        float clampedX = std::max(minVal, std::min(v.x(), maxVal));
        float clampedY = std::max(minVal, std::min(v.y(), maxVal));
        float clampedZ = std::max(minVal, std::min(v.z(), maxVal));

        return Vec3SIMD(clampedX, clampedY, clampedZ);
    }

    static Vec3SIMD pow(const Vec3SIMD& v, float exponent) {
        return Vec3SIMD(
            std::pow(v.x(), exponent),
            std::pow(v.y(), exponent),
            std::pow(v.z(), exponent)
        );
    }

    // Add element-wise maximum operation
    static Vec3SIMD max(const Vec3SIMD& v, float scalar) {
        return Vec3SIMD(
            std::max(v.x(), scalar),
            std::max(v.y(), scalar),
            std::max(v.z(), scalar)
        );
    }


    // Vektör ile double arasýndaki çarpma iþlemi
    Vec3SIMD operator*(double scalar) const {
        // Double'ý vektör boyutuna dönüþtür
        __m256 scalar_vec = _mm256_set1_ps(static_cast<float>(scalar));
        // Çarpma iþlemi
        __m256 result = _mm256_mul_ps(data, scalar_vec);
        return Vec3SIMD(result);
    }
    float x() const;
    float y() const;
    float z() const;
    Vec3SIMD(const float arr[8]) : data(_mm256_loadu_ps(arr)) {}
    float to_float() const;
    // Method to load a scalar value into all elements of the vector
    static Vec3SIMD set1(float val) {
        return Vec3SIMD{ _mm256_set1_ps(val) };
    }
    Vec3SIMD operator-() const;
    Vec3SIMD& operator+=(const Vec3SIMD& v);
    Vec3SIMD& operator-=(const Vec3SIMD& v);
    Vec3SIMD& operator*=(const Vec3SIMD& v);
    Vec3SIMD& operator/=(const Vec3SIMD& v);
    Vec3SIMD& operator*=(float t);
    Vec3SIMD& operator/=(float t);

    float length() const;
    __m256 length_vec() const;
    float length_squared() const;
    // Square root operation
    Vec3SIMD sqrt() const {
        return Vec3SIMD{ _mm256_sqrt_ps(data) };
    }
    // Absolute value operation
    Vec3SIMD abs() const {
        return Vec3SIMD{ _mm256_andnot_ps(_mm256_set1_ps(-0.0f), data) };
    }

    Vec3SIMD cross(const Vec3SIMD& v) const;
    Vec3SIMD safe_normalize(const Vec3SIMD& fallback) const;
    Vec3SIMD normalize() const;
    Vec3SIMD yxz() const;
    float max_component() const;

    static Vec3SIMD to_vec3simd(const Vec3& vec);

    friend Vec3SIMD operator+(const Vec3SIMD& u, const Vec3SIMD& v);
    friend Vec3SIMD operator-(const Vec3SIMD& u, const Vec3SIMD& v);
    friend Vec3SIMD operator*(const Vec3SIMD& u, const Vec3SIMD& v);
    friend Vec3SIMD operator/(const Vec3SIMD& u, const Vec3SIMD& v);
    friend Vec3SIMD operator*(float t, const Vec3SIMD& v);
    friend Vec3SIMD operator*(const Vec3SIMD& v, float t);
    friend Vec3SIMD operator/(const Vec3SIMD& v, float t);

    static Vec3SIMD lerp(const Vec3SIMD& a, const Vec3SIMD& b, float t);
    static Vec3SIMD reflect(const Vec3SIMD& v, const Vec3SIMD& n);
    static Vec3SIMD refract(const Vec3SIMD& uv, const Vec3SIMD& n, float etai_over_etat);
    static Vec3SIMD dot(const Vec3SIMD& u, const Vec3SIMD& v);

     float dot(const Vec3SIMD& other) const;
   
    static float dotfloat(const Vec3SIMD& u, const Vec3SIMD& v);
    bool near_zero() const;
    static Vec3SIMD cross(const Vec3SIMD& u, const Vec3SIMD& v);
    static Vec3SIMD random_in_unit_sphere();
    static Vec3SIMD random(double min, double max);
    static Vec3SIMD random_in_unit_disk();
    static float random_double();
    static Vec3SIMD random_unit_vector();
    static Vec3SIMD random_cosine_direction(const Vec3SIMD& normal) ;
    static Vec3SIMD random_in_unit_hemisphere(const Vec3SIMD& normal);
    static Vec3SIMD random_in_hemisphere(const Vec3SIMD& normal);
private:
    static std::mt19937 rng; // Static random number generator
};

#endif // VEC3SIMD_H