#ifndef MATRIX4X4_H
#define MATRIX4X4_H

#include "Vec3.h"  // Vec3 s�n�f� i�in gerekli ba�l�k dosyas�
#include "Vec3SIMD.h"
#include <immintrin.h> // AVX header

class Mat3x3 {
public:
    float m[3][3];

    // Varsay�lan yap�land�r�c� (birim matris olu�turur)  
    Mat3x3() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] = (i == j) ? 1.0f : 0.0f; // Identity matrix  
    }



    // Vec3 kullanan yap�land�r�c�  
    Mat3x3(Vec3 tangent, Vec3 bitangent, Vec3 normal) {
        m[0][0] = tangent.x; m[0][1] = bitangent.x; m[0][2] = normal.x;
        m[1][0] = tangent.y; m[1][1] = bitangent.y; m[1][2] = normal.y;
        m[2][0] = tangent.z; m[2][1] = bitangent.z; m[2][2] = normal.z;
    }

    // 9 float alan yap�land�r�c�  
    Mat3x3(float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
    }
    Mat3x3 transpose() const {
        Mat3x3 result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result.m[i][j] = m[j][i]; // Sat�r ve s�tunlar� de�i�tir  
            }
        }
        return result;
    }
    // Determinant method  
    float determinant() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }
    // Vec3 ile �arpma i�lemi  
    Vec3 operator*(const Vec3& vec) const {
        return Vec3(
            m[0][0] * vec.x + m[0][1] * vec.y + m[0][2] * vec.z,
            m[1][0] * vec.x + m[1][1] * vec.y + m[1][2] * vec.z,
            m[2][0] * vec.x + m[2][1] * vec.y + m[2][2] * vec.z
        );
    }

    // Vec3 vekt�r�n� d�n��t�rmek i�in transform_vector metodu  
    Vec3 transform_vector(const Vec3& vec) const {
        return (*this) * vec;
    }
};
#define MATRIX4X4_H



class Vec4 {
public:
    float x, y, z, w;

    Vec4() : x(0), y(0), z(0), w(0) {}
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    Vec4(const Vec3& v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}

    Vec3 xyz() const { return Vec3(x, y, z); }
    float& operator[](int i) { return (&x)[i]; }
    const float& operator[](int i) const { return (&x)[i]; }
};

class Matrix4x4 {
public:
    float  m[4][4];
    Matrix4x4() {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m[i][j] = (i == j) ? 1.0f : 0.0f; // Identity matrix
    }
    Matrix4x4(
        float m11, float m12, float m13, float m14,
        float m21, float m22, float m23, float m24,
        float m31, float m32, float m33, float m34,
        float m41, float m42, float m43, float m44
    ) {
        m[0][0] = m11; m[0][1] = m12; m[0][2] = m13; m[0][3] = m14;
        m[1][0] = m21; m[1][1] = m22; m[1][2] = m23; m[1][3] = m24;
        m[2][0] = m31; m[2][1] = m32; m[2][2] = m33; m[2][3] = m34;
        m[3][0] = m41; m[3][1] = m42; m[3][2] = m43; m[3][3] = m44;
    }


    Matrix4x4(Vec3 tangent, Vec3 bitangent, Vec3 normal);
    // Matrix4x4(); // Varsay�lan yap�c�
     // Vec3 ile Matrix4x4 �arpma operat�r�n� arkada� fonksiyon olarak tan�mlay�n
    void setRow(int row, const Vec4& v) {
        m[row][0] = v.x;
        m[row][1] = v.y;
        m[row][2] = v.z;
        m[row][3] = v.w;
    }

    Vec4 getRow(int row) const {
        return Vec4(m[row][0], m[row][1], m[row][2], m[row][3]);
    }

    Vec4 multiplyVector(const Vec4& v) const {
        Vec4 result;
        for (int i = 0; i < 4; i++) {
            result[i] = m[i][0] * v.x + m[i][1] * v.y +
                m[i][2] * v.z + m[i][3] * v.w;
        }
        return result;
    }
    // SIMD optimizasyonu i�in
#ifdef __SSE__
    __m128 getRowSIMD(int row) const {
        return _mm_load_ps(&m[row][0]);
    }

    void setRowSIMD(int row, __m128 v) {
        _mm_store_ps(&m[row][0], v);
    }
#endif
    friend Vec3 operator*(const Matrix4x4& mat, const Vec3& vec);
    static Matrix4x4 identity() {
        Matrix4x4 m;
        m.m[0][0] = 1.0f; m.m[0][1] = 0.0f; m.m[0][2] = 0.0f; m.m[0][3] = 0.0f;
        m.m[1][0] = 0.0f; m.m[1][1] = 1.0f; m.m[1][2] = 0.0f; m.m[1][3] = 0.0f;
        m.m[2][0] = 0.0f; m.m[2][1] = 0.0f; m.m[2][2] = 1.0f; m.m[2][3] = 0.0f;
        m.m[3][0] = 0.0f; m.m[3][1] = 0.0f; m.m[3][2] = 0.0f; m.m[3][3] = 1.0f;
        return m;
    }

    double minor(int row, int col) const;
    double cofactor(int row, int col) const;
    double determinant() const;
    // Transpose metodu
    Matrix4x4 transpose() const;


    Matrix4x4 inverse() const;
    // Matris �arp�m� operat�r�
    Matrix4x4 operator*(const Matrix4x4& other) const;

    Vec3 transform_point(const Vec3& p) const; // Nokta d�n���m�
    Vec3 transform_vector(const Vec3& v) const; // Vekt�r d�n���m�
    Matrix4x4 multiply(const Matrix4x4& other) const {
        Matrix4x4 result;
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                result.m[row][col] = 0.0f;
                for (int k = 0; k < 4; k++) {
                    result.m[row][col] += m[row][k] * other.m[k][col];
                }
            }
        }
        return result;
    }

    // Statik matris olu�turucular�
    static Matrix4x4 translation(const Vec3& t);
    static Matrix4x4 scaling(const Vec3& s);
    static Matrix4x4 rotation_x(double angle);
    // Y ve Z eksenleri i�in benzer rotasyon fonksiyonlar� eklenebilir
    static Matrix4x4 translation(double x, double y, double z);
    static Matrix4x4 scaling(double x, double y, double z);
    static Matrix4x4 rotationX(double angle);
    static Matrix4x4 rotationY(double angle);
    static Matrix4x4 rotationZ(double angle);
};

#endif // MATRIX4X4_H

