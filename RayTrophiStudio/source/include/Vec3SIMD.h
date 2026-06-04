/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Vec3SIMD.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef VEC3SIMD_H
#define VEC3SIMD_H

#include <immintrin.h> // AVX header
#include <cmath>
#include <random>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <limits>

// M_PI'yi float olarak tanımla
#define M_PI 3.14159265358979323846f

// ÖNEMLİ: Vec3SIMD, 8 farklı float'ı (örneğin 8 farklı ışının X bileşenini) tutar.
class Vec3SIMD {
public:
    __m256 data;

    // --- Constructors ---
    Vec3SIMD();
    Vec3SIMD(__m256 d);
    Vec3SIMD(float val); // set1

    // Tek bir 3D Vektörü AVX'e Yükler (İlk 3 bileşeni doldurur, diğerleri 0)
    // DİKKAT: Yavaşlamaya neden olabilir, sadece skaler entegrasyon için.
    Vec3SIMD(float x, float y, float z);

    // Array Constructor
    Vec3SIMD(const float arr[8]);

    // --- Access Operators (Yavaş, Skaler Kod ile Uyum İçin Korundu) ---
    float x() const;
    float y() const;
    float z() const;
    float get(int index) const; // Genel indeksli erişim
    float operator[](int i) const { return get(i); } // [] operatörü

    // --- Aritmetik Operatörler (8x Paralel) ---
    Vec3SIMD operator-() const;

    Vec3SIMD& operator+=(const Vec3SIMD& v);
    Vec3SIMD& operator-=(const Vec3SIMD& v);
    Vec3SIMD& operator*=(const Vec3SIMD& v);
    Vec3SIMD& operator/=(const Vec3SIMD& v);

    Vec3SIMD& operator*=(float t);
    Vec3SIMD& operator/=(float t);

    Vec3SIMD operator-(float scalar) const;
    Vec3SIMD operator+(float scalar) const;
    Vec3SIMD operator*(float scalar) const;
    Vec3SIMD operator/(float scalar) const;

    // --- Karşılaştırma Operatörleri (8x Paralel, __m256 Maskesi Döndürür) ---
    __m256 operator==(const Vec3SIMD& other) const;
    __m256 operator!=(const Vec3SIMD& other) const;

    // --- Temel Matematik (8x Paralel) ---
    Vec3SIMD abs() const;
    Vec3SIMD sqrt() const;

    // --- Statik Yardımcı Metotlar (Paket Tabanlı) ---
    static Vec3SIMD set1(float val);
    static Vec3SIMD setZero();

    // Çoğu bu statik metotları çağırır, 3 bileşenin hepsini işler:

    // 8x Nokta Çarpım: Geriye 8 sonucu içeren __m256 döndürür.
    static __m256 dot_product_8x(const Vec3SIMD& u_x, const Vec3SIMD& u_y, const Vec3SIMD& u_z,
        const Vec3SIMD& v_x, const Vec3SIMD& v_y, const Vec3SIMD& v_z);

    // 8x Uzunluk Karesi: Geriye 8 sonucu içeren __m256 döndürür.
    static __m256 length_squared_8x(const Vec3SIMD& u_x, const Vec3SIMD& u_y, const Vec3SIMD& u_z);

    // 8x Normalizasyon: Geriye normalize edilmiş 3 bileşeni döndürür.
    static void normalize_8x(const Vec3SIMD& u_x, const Vec3SIMD& u_y, const Vec3SIMD& u_z,
        Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z);

    // 8x Çapraz Çarpım (Cross Product): Geriye 3 bileşeni döndürür.
    static void cross_8x(const Vec3SIMD& u_x, const Vec3SIMD& u_y, const Vec3SIMD& u_z,
        const Vec3SIMD& v_x, const Vec3SIMD& v_y, const Vec3SIMD& v_z,
        Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z);

    // 8x Yansıma (Reflect)
    static void reflect_8x(const Vec3SIMD& v_x, const Vec3SIMD& v_y, const Vec3SIMD& v_z,
        const Vec3SIMD& n_x, const Vec3SIMD& n_y, const Vec3SIMD& n_z,
        Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z);

    // 8x Kırılma (Refract)
    static void refract_8x(const Vec3SIMD& uv_x, const Vec3SIMD& uv_y, const Vec3SIMD& uv_z,
        const Vec3SIMD& n_x, const Vec3SIMD& n_y, const Vec3SIMD& n_z,
        const Vec3SIMD& etai_over_etat,
        Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z);


    // --- Skaler Uyum ve Legacy Metotlar (Yavaş) ---
    // Bu metotlar AVX'in performansını düşürür, sadece tekil test/kullanım için saklanmıştır.
    float length() const;
    float length_squared() const;
    float dot(const Vec3SIMD& other) const; // Tek bir skaler değer döndürür (Hadd gerektirir)
    bool near_zero() const; // Tek bir vektör için kontrol
    float max_component() const; // Tek bir vektör için kontrol

    // --- Skaler-SIMD Uyum Metotları ---
    static Vec3SIMD max(const Vec3SIMD& v, float scalar);
    static Vec3SIMD min(const Vec3SIMD& a, const Vec3SIMD& b); // Added
    static Vec3SIMD max(const Vec3SIMD& a, const Vec3SIMD& b); // Added
    static Vec3SIMD clamp(const Vec3SIMD& v, float minVal, float maxVal);
    static Vec3SIMD pow(const Vec3SIMD& v, float exponent);

    // --- AVX2 Math Intrinsics (Phase 1 Upgrade) ---
    // Trigonometry & Power Functions using Polynomial Approximation
    static __m256 sin_256(const __m256& x);
    static __m256 cos_256(const __m256& x);
    static void sincos_256(const __m256& x, __m256* s, __m256* c);
    static __m256 acos_256(const __m256& x);
    static __m256 atan2_256(const __m256& y, const __m256& x);
    static __m256 pow_256(const __m256& x, const __m256& y);
    static __m256 exp_256(const __m256& x);
    static __m256 log_256(const __m256& x);
    // --- MIS & Helpers ---
    static __m256 power_heuristic_8x(__m256 f, __m256 g);

    // --- Skaler Random Metotlar (Yavaş) ---
    static float random_float();
    static __m256 random_float_8x();
    static void random_unit_vector_8x(Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z);

private:
    // static std::mt19937 rng; // REMOVED: Thread-unsafe and slow
    // static std::uniform_real_distribution<float> dist; // REMOVED
};

// Represents 8 three-dimensional vectors (SoA layout)
struct Vec3Packet {
    Vec3SIMD x, y, z;
    Vec3Packet() = default;
    Vec3Packet(const Vec3SIMD& _x, const Vec3SIMD& _y, const Vec3SIMD& _z) : x(_x), y(_y), z(_z) {}
    Vec3Packet(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};

// --- Friend Fonksiyonlar (8x Paralel) ---
Vec3SIMD operator+(const Vec3SIMD& u, const Vec3SIMD& v);
Vec3SIMD operator-(const Vec3SIMD& u, const Vec3SIMD& v);
Vec3SIMD operator*(const Vec3SIMD& u, const Vec3SIMD& v);
Vec3SIMD operator/(const Vec3SIMD& u, const Vec3SIMD& v);
Vec3SIMD operator*(float t, const Vec3SIMD& v);
Vec3SIMD operator/(const Vec3SIMD& v, float t);

#endif // VEC3SIMD_H
