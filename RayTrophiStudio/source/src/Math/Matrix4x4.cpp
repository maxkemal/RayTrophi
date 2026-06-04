#include "Matrix4x4.h"
#include <cmath> // cos ve sin fonksiyonları için

// Varsayılan yapıcı
// Birim matris oluşturma

Matrix4x4::Matrix4x4(Vec3 tangent, Vec3 bitangent, Vec3 normal) {
    // İlk üç satır ve sütunu TBN matrisine göre ayarla
    m[0][0] = tangent.x;
    m[0][1] = tangent.y;
    m[0][2] = tangent.z;
    m[0][3] = 0.0f;

    m[1][0] = bitangent.x;
    m[1][1] = bitangent.y;
    m[1][2] = bitangent.z;
    m[1][3] = 0.0f;

    m[2][0] = normal.x;
    m[2][1] = normal.y;
    m[2][2] = normal.z;
    m[2][3] = 0.0f;

    // Son satırı ve sütunu birim matrise göre ayarla
    m[3][0] = 0.0f;
    m[3][1] = 0.0f;
    m[3][2] = 0.0f;
    m[3][3] = 1.0f;
}
// Matrix4x4 ve Vec3 çarpma operatörü tanımı
Vec3 operator*(const Matrix4x4& mat, const Vec3& vec) {
    Vec3 result;
    result.x = mat.m[0][0] * vec.x + mat.m[0][1] * vec.y + mat.m[0][2] * vec.z + mat.m[0][3];
    result.y = mat.m[1][0] * vec.x + mat.m[1][1] * vec.y + mat.m[1][2] * vec.z + mat.m[1][3];
    result.z = mat.m[2][0] * vec.x + mat.m[2][1] * vec.y + mat.m[2][2] * vec.z + mat.m[2][3];
    return result;
}

Matrix4x4 Matrix4x4::transpose() const {
    Matrix4x4 result;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.m[i][j] = m[j][i];
        }
    }
    return result;
}
// Matris çarpımı operatörü
Matrix4x4 Matrix4x4::operator*(const Matrix4x4& other) const {
    Matrix4x4 result;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.m[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                result.m[i][j] += m[i][k] * other.m[k][j];
            }
        }
    }
    return result;
}

// Nokta dönüşümü
Vec3 Matrix4x4::transform_point(const Vec3& point) const {
    float x = m[0][0] * point.x + m[0][1] * point.y + m[0][2] * point.z + m[0][3];
    float y = m[1][0] * point.x + m[1][1] * point.y + m[1][2] * point.z + m[1][3];
    float z = m[2][0] * point.x + m[2][1] * point.y + m[2][2] * point.z + m[2][3];
    float w = m[3][0] * point.x + m[3][1] * point.y + m[3][2] * point.z + m[3][3];

    if (w != 1.0f && w != 0.0f) {
        return Vec3(x / w, y / w, z / w);
    }
    return Vec3(x, y, z);
}

// Vektör dönüşümü
Vec3 Matrix4x4::transform_vector(const Vec3& v) const {
    float x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z;
    float y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z;
    float z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z;
    return Vec3(x, y, z);
}

// Statik matris oluşturucuları
Matrix4x4 Matrix4x4::translation(const Vec3& t) {
    Matrix4x4 mat;
    mat.m[0][3] = t.x;
    mat.m[1][3] = t.y;
    mat.m[2][3] = t.z;
    return mat;
}

Matrix4x4 Matrix4x4::scaling(const Vec3& s) {
    Matrix4x4 mat;
    mat.m[0][0] = s.x;
    mat.m[1][1] = s.y;
    mat.m[2][2] = s.z;
    return mat;
}
float Matrix4x4::cofactor(int row, int col) const {
    return ((row + col) % 2 == 0 ? 1 : -1) * minor(row, col);
}

float Matrix4x4::determinant() const {
    // 4x4 matris için determinant hesaplama
    // Bu basit bir implementasyondur, daha verimli metotlar kullanılabilir
    return m[0][0] * cofactor(0, 0) - m[0][1] * cofactor(0, 1) + m[0][2] * cofactor(0, 2) - m[0][3] * cofactor(0, 3);
}
float Matrix4x4::minor(int row, int col) const {
    float minor[3][3];
    int r = 0, c = 0;
    for (int i = 0; i < 4; i++) {
        if (i == row) continue;
        c = 0;
        for (int j = 0; j < 4; j++) {
            if (j == col) continue;
            minor[r][c] = m[i][j];
            c++;
        }
        r++;
    }
    return minor[0][0] * (minor[1][1] * minor[2][2] - minor[1][2] * minor[2][1]) -
        minor[0][1] * (minor[1][0] * minor[2][2] - minor[1][2] * minor[2][0]) +
        minor[0][2] * (minor[1][0] * minor[2][1] - minor[1][1] * minor[2][0]);
}
Matrix4x4 Matrix4x4::inverse() const {
    // Bu, basit bir tersi alma implementasyonudur.
    // Daha karmaşık ve verimli bir implementasyon gerekebilir.
    Matrix4x4 result;
    float det = determinant();
    if (std::abs(det) < 1e-25) { // MUCH lower threshold for tiny scales (0.001 scale^3 = 1e-9)
        // Matris tekil, tersi alınamaz
        return Matrix4x4(); // Birim matris döndür
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.m[i][j] = cofactor(i, j) / det;
        }
    }

    return result.transpose();
}
// X ekseni etrafında rotasyon matrisi oluşturma
Matrix4x4 Matrix4x4::rotation_x(float angle) {
    Matrix4x4 mat;
    return mat;
}
Matrix4x4 Matrix4x4::translation(float x, float y, float z) {
    Matrix4x4 mat;
    mat.identity();
    mat.m[0][3] = x;
    mat.m[1][3] = y;
    mat.m[2][3] = z;
    return mat;
}

Matrix4x4 Matrix4x4::scaling(float x, float y, float z) {
    Matrix4x4 mat;
    mat.identity();
    mat.m[0][0] = x;
    mat.m[1][1] = y;
    mat.m[2][2] = z;
    return mat;
}

Matrix4x4 Matrix4x4::rotationX(float angle) {
    Matrix4x4 mat;
    mat.identity();
    float c = cosf(angle);
    float s = sinf(angle);
    mat.m[1][1] = c;
    mat.m[1][2] = -s;
    mat.m[2][1] = s;
    mat.m[2][2] = c;
    return mat;
}

Matrix4x4 Matrix4x4::rotationY(float angle) {
    Matrix4x4 mat;
    mat.identity();
    float c = cosf(angle);
    float s = sinf(angle);
    mat.m[0][0] = c;
    mat.m[0][2] = s;
    mat.m[2][0] = -s;
    mat.m[2][2] = c;
    return mat;
}

Matrix4x4 Matrix4x4::rotationZ(float angle) {
    Matrix4x4 mat;
    mat.identity();
    float c = cosf(angle);
    float s = sinf(angle);
    mat.m[0][0] = c;
    mat.m[0][1] = -s;
    mat.m[1][0] = s;
    mat.m[1][1] = c;
    return mat;
}
