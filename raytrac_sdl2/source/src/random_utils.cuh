#pragma once
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__device__ inline float random_float(curandState* state) {
    return curand_uniform(state);
}
__device__ float random_float(curandState* rng, float min, float max) {
    return min + (max - min) * curand_uniform(rng);
}

__device__ inline float random_float_range(curandState* state, float min, float max) {
    return min + (max - min) * random_float(state);
}

__device__ float3 random_in_unit_sphere(curandState* rng) {
    while (true) {
        float x = random_float(rng, -1.0f, 1.0f);
        float y = random_float(rng, -1.0f, 1.0f);
        float z = random_float(rng, -1.0f, 1.0f);
        float3 p = make_float3(x, y, z);
        if (dot(p, p) < 1.0f) return p;
    }
}

__device__ inline float3 random_unit_vector(curandState* state) {
    return normalize(random_in_unit_sphere(state));
}

__device__ inline float3 random_cosine_direction(curandState* state) {
    float r1 = random_float(state);
    float r2 = random_float(state);
    float z = sqrtf(1.0f - r2);

    float phi = 2.0f * M_PI * r1;
    float x = cosf(phi) * sqrtf(r2);
    float y = sinf(phi) * sqrtf(r2);

    return make_float3(x, y, z);
}
__device__ float3 cosine_sample_hemisphere(curandState* rng, const float3& N) {
    float u1 = random_float(rng);
    float u2 = random_float(rng);

    float r = sqrtf(u1);
    float theta = 2.0f * M_PIf * u2;

    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(max(0.0f, 1.0f - u1));

    float3 sample = make_float3(x, y, z);

    // Dönüştür: Z ekseni yukarı, N'ye göre yönlendir
    float3 up = fabsf(N.z) < 0.999f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
    float3 tangentX = normalize(cross(up, N));
    float3 tangentY = cross(N, tangentX);

    return normalize(tangentX * sample.x + tangentY * sample.y + N * sample.z);
}
__device__ float2 random_in_unit_disk(curandState* rng) {
    float2 p;
    do {
        p = make_float2(
            random_float(rng) * 2.0f - 1.0f,
            random_float(rng) * 2.0f - 1.0f
        );
    } while ((p.x * p.x + p.y * p.y) >= 1.0f);
    return p;
}

