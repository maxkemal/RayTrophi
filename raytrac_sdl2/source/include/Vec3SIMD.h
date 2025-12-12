#ifndef VEC3SIMD_H
#define VEC3SIMD_H

#include <immintrin.h> // AVX header
#include <cmath>
#include <random>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <limits>

// M_PI'yi float olarak tanýmla
#define M_PI 3.14159265358979323846f

// ÖNEMLÝ: Vec3SIMD, 8 farklý float'ý (örneðin 8 farklý ýþýnýn X bileþenini) tutar.
class Vec3SIMD {
public:
    __m256 data;

    // --- Constructors ---
    Vec3SIMD();
    Vec3SIMD(__m256 d);
    Vec3SIMD(float val); // set1

    // Tek bir 3D Vektörü AVX'e Yükler (Ýlk 3 bileþeni doldurur, diðerleri 0)
    // DÝKKAT: Yavaþlamaya neden olabilir, sadece skaler entegrasyon için.
    Vec3SIMD(float x, float y, float z);

    // Array Constructor
    Vec3SIMD(const float arr[8]);

    // --- Access Operators (Yavaþ, Skaler Kod ile Uyum Ýçin Korundu) ---
    float x() const;
    float y() const;
    float z() const;
    float get(int index) const; // Genel indeksli eriþim
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

    // --- Karþýlaþtýrma Operatörleri (8x Paralel, __m256 Maskesi Döndürür) ---
    __m256 operator==(const Vec3SIMD& other) const;
    __m256 operator!=(const Vec3SIMD& other) const;

    // --- Temel Matematik (8x Paralel) ---
    Vec3SIMD abs() const;
    Vec3SIMD sqrt() const;

    // --- Statik Yardýmcý Metotlar (Paket Tabanlý) ---
    static Vec3SIMD set1(float val);
    static Vec3SIMD setZero();

    // Çoðu bu statik metotlarý çaðýrýr, 3 bileþenin hepsini iþler:

    // 8x Nokta Çarpým: Geriye 8 sonucu içeren __m256 döndürür.
    static __m256 dot_product_8x(const Vec3SIMD& u_x, const Vec3SIMD& u_y, const Vec3SIMD& u_z,
        const Vec3SIMD& v_x, const Vec3SIMD& v_y, const Vec3SIMD& v_z);

    // 8x Uzunluk Karesi: Geriye 8 sonucu içeren __m256 döndürür.
    static __m256 length_squared_8x(const Vec3SIMD& u_x, const Vec3SIMD& u_y, const Vec3SIMD& u_z);

    // 8x Normalizasyon: Geriye normalize edilmiþ 3 bileþeni döndürür.
    static void normalize_8x(const Vec3SIMD& u_x, const Vec3SIMD& u_y, const Vec3SIMD& u_z,
        Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z);

    // 8x Çapraz Çarpým (Cross Product): Geriye 3 bileþeni döndürür.
    static void cross_8x(const Vec3SIMD& u_x, const Vec3SIMD& u_y, const Vec3SIMD& u_z,
        const Vec3SIMD& v_x, const Vec3SIMD& v_y, const Vec3SIMD& v_z,
        Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z);

    // 8x Yansýma (Reflect)
    static void reflect_8x(const Vec3SIMD& v_x, const Vec3SIMD& v_y, const Vec3SIMD& v_z,
        const Vec3SIMD& n_x, const Vec3SIMD& n_y, const Vec3SIMD& n_z,
        Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z);

    // 8x Kýrýlma (Refract)
    static void refract_8x(const Vec3SIMD& uv_x, const Vec3SIMD& uv_y, const Vec3SIMD& uv_z,
        const Vec3SIMD& n_x, const Vec3SIMD& n_y, const Vec3SIMD& n_z,
        const Vec3SIMD& etai_over_etat,
        Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z);


    // --- Skaler Uyum ve Legacy Metotlar (Yavaþ) ---
    // Bu metotlar AVX'in performansýný düþürür, sadece tekil test/kullaným için saklanmýþtýr.
    float length() const;
    float length_squared() const;
    float dot(const Vec3SIMD& other) const; // Tek bir skaler deðer döndürür (Hadd gerektirir)
    bool near_zero() const; // Tek bir vektör için kontrol
    float max_component() const; // Tek bir vektör için kontrol

    // --- Skaler-SIMD Uyum Metotlarý ---
    static Vec3SIMD max(const Vec3SIMD& v, float scalar);
    static Vec3SIMD clamp(const Vec3SIMD& v, float minVal, float maxVal);
    static Vec3SIMD pow(const Vec3SIMD& v, float exponent);

    // --- Skaler Random Metotlar (Yavaþ) ---
    static float random_float();
    static void random_unit_vector_8x(Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z);

private:
    static std::mt19937 rng;
    static std::uniform_real_distribution<float> dist;
};

// --- Friend Fonksiyonlar (8x Paralel) ---
Vec3SIMD operator+(const Vec3SIMD& u, const Vec3SIMD& v);
Vec3SIMD operator-(const Vec3SIMD& u, const Vec3SIMD& v);
Vec3SIMD operator*(const Vec3SIMD& u, const Vec3SIMD& v);
Vec3SIMD operator/(const Vec3SIMD& u, const Vec3SIMD& v);
Vec3SIMD operator*(float t, const Vec3SIMD& v);
Vec3SIMD operator/(const Vec3SIMD& v, float t);

#endif // VEC3SIMD_H