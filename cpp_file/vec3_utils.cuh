#pragma once
#include <cuda_runtime.h>
#ifndef M_PIf
#define M_PIf 3.1415927f
#endif

// === Arithmetic Operators ===
__host__ __device__ inline float length3(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float3 operator-(const float3& v) {
    return make_float3(-v.x, -v.y, -v.z);
}
__device__ inline float3 exp_componentwise(float3 v) {
    return make_float3(expf(v.x), expf(v.y), expf(v.z));
}
__device__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ inline float3 operator*(const float3& a, float t) {
    return make_float3(a.x * t, a.y * t, a.z * t);
}
__device__ inline float3 operator*(float t, const float3& a) {
    return a * t;
}

__device__ inline float3 operator/(const float3& a, float t) {
    return make_float3(a.x / t, a.y / t, a.z / t);
}
__device__ inline float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

// === Compound Assignment ===

__device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}
__device__ inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}
__device__ inline float3& operator*=(float3& a, float t) {
    a.x *= t; a.y *= t; a.z *= t;
    return a;
}
__device__ inline float3& operator*=(float3& a, const float3& b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
    return a;
}
__device__ inline float3& operator/=(float3& a, float t) {
    a.x /= t; a.y /= t; a.z /= t;
    return a;
}

// === float2 operators ===

__device__ inline float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ inline float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ inline float2 operator*(const float2& a, const float2& b) {
    return make_float2(a.x * b.x, a.y * b.y);
}
__device__ inline float2 operator*(float a, const float2& b) {
    return make_float2(a * b.x, a * b.y);
}
__device__ inline float2 operator*(const float2& a, float b) {
    return make_float2(a.x * b, a.y * b);
}
__device__ inline float2 operator/(const float2& a, float b) {
    return make_float2(a.x / b, a.y / b);
}

// === Math ===

__device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}
__device__ inline float3 normalize(const float3& v) {
    return v / length(v);
}
__device__ inline float3 reflect(const float3& v, const float3& n) {
    return v - 2.0f * dot(v, n) * n;
}
__device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
__device__ inline float clamp(float val, float minVal, float maxVal) {
    return fminf(fmaxf(val, minVal), maxVal);
}
__device__ inline float3 clamp(const float3& v, float minVal, float maxVal) {
    return make_float3(
        fminf(fmaxf(v.x, minVal), maxVal),
        fminf(fmaxf(v.y, minVal), maxVal),
        fminf(fmaxf(v.z, minVal), maxVal)
    );
}
__device__ __host__ inline float2 clamp(const float2& v, float minVal, float maxVal) {
    return make_float2(
        fminf(fmaxf(v.x, minVal), maxVal),
        fminf(fmaxf(v.y, minVal), maxVal)
    );
}
__device__ inline float lerp(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

__device__ inline float3 lerp(const float3& a, const float3& b, float t) {
    return a * (1.0f - t) + b * t;
}
__device__ inline float3 max(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
__device__ inline float3 min(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
__device__ inline float3 abs(const float3& a) {
    return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}

// === Color & Utility ===

__device__ inline uchar4 make_color(const float3& c) {

    float3 gamma_corrected = make_float3(
        powf(fminf(c.x, 1.0f), 1.0f / 1.4f),
        powf(fminf(c.y, 1.0f), 1.0f / 1.4f),
        powf(fminf(c.z, 1.0f), 1.0f / 1.4f)
    );

    return make_uchar4(
        static_cast<unsigned char>(255.0f * gamma_corrected.x),
        static_cast<unsigned char>(255.0f * gamma_corrected.y),
        static_cast<unsigned char>(255.0f * gamma_corrected.z),
        255
    );
}

__device__ inline float3 to_float3(float4 f4) {
    return make_float3(f4.x, f4.y, f4.z);
}
__device__ inline float3 decode_normal(float4 n) {
    float3 norm = make_float3(n.x, n.y, n.z);
    return norm * 2.0f - make_float3(1.0f, 1.0f, 1.0f);
}
__device__ void build_coordinate_system(const float3& N, float3& U, float3& V) {
    if (fabsf(N.x) > fabsf(N.z)) {
        U = normalize(make_float3(-N.y, N.x, 0.0f));
    }
    else {
        U = normalize(make_float3(0.0f, -N.z, N.y));
    }
    V = cross(N, U);
}

