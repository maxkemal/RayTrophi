/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Quaternion.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include "Vec3.h"
#include "Matrix4x4.h"
class Quaternion {
public:
    float w, x, y, z;

    // Create a quaternion representing the rotation from 'start' to 'dest' vectors
    static Quaternion rotationBetween(const Vec3& start, const Vec3& dest) {
        Vec3 startNorm = start.normalize();
        Vec3 destNorm = dest.normalize();
    
        float cosTheta = Vec3::dot(startNorm, destNorm);
        Vec3 rotationAxis;
    
        if (cosTheta < -1.0f + 1e-6f) {
            // Vectors are opposite, rotate 180 degrees around any orthogonal axis
            rotationAxis = Vec3::cross(Vec3(0.0f, 0.0f, 1.0f), startNorm);
            if (rotationAxis.length_squared() < 1e-6f) // parallel to z-axis
                rotationAxis = Vec3::cross(Vec3(1.0f, 0.0f, 0.0f), startNorm);
            
            rotationAxis = rotationAxis.normalize();
            return Quaternion(0.0f, rotationAxis.x, rotationAxis.y, rotationAxis.z);
        }
    
        rotationAxis = Vec3::cross(startNorm, destNorm);
    
        float s = sqrtf((1.0f + cosTheta) * 2.0f);
        float invs = 1.0f / s;
    
        return Quaternion(
            s * 0.5f,
            rotationAxis.x * invs,
            rotationAxis.y * invs,
            rotationAxis.z * invs
        );
    }

    // VarsayÃ½lan yapÃ½cÃ½
    Quaternion() : w(1), x(0), y(0), z(0) {}

    // Parametreli yapÃ½cÃ½
    Quaternion(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}

    // Quaternion'u birim yapma (normalize etme)
    void normalize() {
        float mag = sqrtf(w * w + x * x + y * y + z * z);
        w /= mag;
        x /= mag;
        y /= mag;
        z /= mag;
    }
    static Quaternion slerp(const Quaternion& start, const Quaternion& end, float t) {
        Quaternion result;
        float dot = start.x * end.x + start.y * end.y + start.z * end.z + start.w * end.w;

        if (dot < 0.0f) {
            // Quaternions ters iÃ¾aretlenmiÃ¾se, bunu dÃ¼zelt
            result.x = -end.x;
            result.y = -end.y;
            result.z = -end.z;
            result.w = -end.w;
            dot = -dot;
        }
        else {
            result = end;
        }

        float theta_0 = acos(dot);  // BaÃ¾langÃ½Ã§ aÃ§Ã½sÃ½
        float sin_theta_0 = sin(theta_0);  // SinÃ¼s(theta_0)

        if (fabs(sin_theta_0) > 1e-6) {
            float theta = theta_0 * t;  // t iÃ§in interpolasyon aÃ§Ã½sÃ½
            float sin_theta = sin(theta);
            float s0 = cos(theta) - dot * sin_theta / sin_theta_0;
            float s1 = sin_theta / sin_theta_0;
            result.x = s0 * start.x + s1 * result.x;
            result.y = s0 * start.y + s1 * result.y;
            result.z = s0 * start.z + s1 * result.z;
            result.w = s0 * start.w + s1 * result.w;
        }
        else {
            // EÃ°er aÃ§Ã½lar Ã§ok yakÃ½nsa, dÃ¼z interpolasyon yap
            result.x = (1.0f - t) * start.x + t * result.x;
            result.y = (1.0f - t) * start.y + t * result.y;
            result.z = (1.0f - t) * start.z + t * result.z;
            result.w = (1.0f - t) * start.w + t * result.w;
        }
        return result;
    }
    // Quaternion ile vektÃ¶r dÃ¶ndÃ¼rme
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

    // Quaternion Ã§arpÃ½mÃ½
    Quaternion operator*(const Quaternion& other) const {
        return Quaternion(
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w
        );
    }

    // Quaternion'dan dÃ¶nÃ¼Ã¾Ã¼m matrisi oluÃ¾turma
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

    // Matristen Quaternion oluşturma (Decomposition)
    static Quaternion fromMatrix(const Matrix4x4& m) {
        float trace = m.m[0][0] + m.m[1][1] + m.m[2][2];
        float S = 0;

        if (trace > 0) {
            S = sqrtf(trace + 1.0f) * 2;
            return Quaternion(
                0.25f * S,
                (m.m[2][1] - m.m[1][2]) / S,
                (m.m[0][2] - m.m[2][0]) / S,
                (m.m[1][0] - m.m[0][1]) / S
            );
        } else if (m.m[0][0] > m.m[1][1] && m.m[0][0] > m.m[2][2]) {
            S = sqrtf(1.0f + m.m[0][0] - m.m[1][1] - m.m[2][2]) * 2;
            return Quaternion(
                (m.m[2][1] - m.m[1][2]) / S,
                0.25f * S,
                (m.m[0][1] + m.m[1][0]) / S,
                (m.m[0][2] + m.m[2][0]) / S
            );
        } else if (m.m[1][1] > m.m[2][2]) {
            S = sqrtf(1.0f + m.m[1][1] - m.m[0][0] - m.m[2][2]) * 2;
            return Quaternion(
                (m.m[0][2] - m.m[2][0]) / S,
                (m.m[0][1] + m.m[1][0]) / S,
                0.25f * S,
                (m.m[1][2] + m.m[2][1]) / S
            );
        } else {
            S = sqrtf(1.0f + m.m[2][2] - m.m[0][0] - m.m[1][1]) * 2;
            return Quaternion(
                (m.m[1][0] - m.m[0][1]) / S,
                (m.m[0][2] + m.m[2][0]) / S,
                (m.m[1][2] + m.m[2][1]) / S,
                0.25f * S
            );
        }
    }

    // Quaternion'dan euler aÃ§Ã½larÃ½ oluÃ¾turma
    Vec3 toEuler() const {
        Vec3 euler;
        euler.x = atan2f(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y));
        euler.y = asinf(2.0f * (w * y - z * x));
        euler.z = atan2f(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
        return euler;
    }
};

