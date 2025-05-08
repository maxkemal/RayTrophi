#pragma once
#include "Vec3.h"
#include "Matrix4x4.h"
class Quaternion {
public:
    float w, x, y, z;

    // Varsayýlan yapýcý
    Quaternion() : w(1), x(0), y(0), z(0) {}

    // Parametreli yapýcý
    Quaternion(double w, double x, double y, double z) : w(w), x(x), y(y), z(z) {}

    // Quaternion'u birim yapma (normalize etme)
    void normalize() {
        double mag = sqrt(w * w + x * x + y * y + z * z);
        w /= mag;
        x /= mag;
        y /= mag;
        z /= mag;
    }
    static Quaternion slerp(const Quaternion& start, const Quaternion& end, float t) {
        Quaternion result;
        float dot = start.x * end.x + start.y * end.y + start.z * end.z + start.w * end.w;

        if (dot < 0.0f) {
            // Quaternions ters iþaretlenmiþse, bunu düzelt
            result.x = -end.x;
            result.y = -end.y;
            result.z = -end.z;
            result.w = -end.w;
            dot = -dot;
        }
        else {
            result = end;
        }

        float theta_0 = acos(dot);  // Baþlangýç açýsý
        float sin_theta_0 = sin(theta_0);  // Sinüs(theta_0)

        if (fabs(sin_theta_0) > 1e-6) {
            float theta = theta_0 * t;  // t için interpolasyon açýsý
            float sin_theta = sin(theta);
            float s0 = cos(theta) - dot * sin_theta / sin_theta_0;
            float s1 = sin_theta / sin_theta_0;
            result.x = s0 * start.x + s1 * result.x;
            result.y = s0 * start.y + s1 * result.y;
            result.z = s0 * start.z + s1 * result.z;
            result.w = s0 * start.w + s1 * result.w;
        }
        else {
            // Eðer açýlar çok yakýnsa, düz interpolasyon yap
            result.x = (1.0f - t) * start.x + t * result.x;
            result.y = (1.0f - t) * start.y + t * result.y;
            result.z = (1.0f - t) * start.z + t * result.z;
            result.w = (1.0f - t) * start.w + t * result.w;
        }
        return result;
    }
    // Quaternion ile vektör döndürme
    Vec3 rotate(const Vec3& v) const {
        Quaternion qv(0, v.x, v.y, v.z);
        Quaternion inv = conjugate();
        Quaternion result = (*this) * qv * inv;
        return Vec3(result.x, result.y, result.z);
    }

    // Conjugate (ters) alma
    Quaternion conjugate() const {
        return Quaternion(w, -x, -y, -z);
    }

    // Quaternion çarpýmý
    Quaternion operator*(const Quaternion& other) const {
        return Quaternion(
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w
        );
    }

    // Quaternion'dan dönüþüm matrisi oluþturma
    Matrix4x4 toMatrix() const {
        Matrix4x4 mat;
        mat.m[0][0] = 1 - 2 * (y * y + z * z);
        mat.m[0][1] = 2 * (x * y - z * w);
        mat.m[0][2] = 2 * (x * z + y * w);
        mat.m[0][3] = 0;

        mat.m[1][0] = 2 * (x * y + z * w);
        mat.m[1][1] = 1 - 2 * (x * x + z * z);
        mat.m[1][2] = 2 * (y * z - x * w);
        mat.m[1][3] = 0;

        mat.m[2][0] = 2 * (x * z - y * w);
        mat.m[2][1] = 2 * (y * z + x * w);
        mat.m[2][2] = 1 - 2 * (x * x + y * y);
        mat.m[2][3] = 0;

        mat.m[3][0] = 0;
        mat.m[3][1] = 0;
        mat.m[3][2] = 0;
        mat.m[3][3] = 1;

        return mat;
    }

    // Quaternion'dan euler açýlarý oluþturma
    Vec3 toEuler() const {
        Vec3 euler;
        euler.x = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
        euler.y = asin(2.0 * (w * y - z * x));
        euler.z = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
        return euler;
    }
};
