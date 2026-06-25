#pragma once
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================================================
// FAST RANDOM NUMBER GENERATION (Wang Hash + PCG)
// ============================================================================
// These functions are ~10-100x faster than curand_init + curand_uniform

// Wang Hash - Fast integer hash with good distribution
__device__ __forceinline__ unsigned int wang_hash(unsigned int seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

// PCG Random State - Lightweight alternative to curandState
struct PCGState {
    unsigned int state;
};

// Initialize PCG state from seed
__device__ __forceinline__ void pcg_init(PCGState* rng, unsigned int seed) {
    rng->state = wang_hash(seed);
}

// Generate next random unsigned int
__device__ __forceinline__ unsigned int pcg_next(PCGState* rng) {
    unsigned int state = rng->state;
    rng->state = state * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate random float in [0, 1)
__device__ __forceinline__ float pcg_float(PCGState* rng) {
    return pcg_next(rng) * (1.0f / 4294967296.0f);
}

// Generate random float in [min, max)
__device__ __forceinline__ float pcg_float_range(PCGState* rng, float min, float max) {
    return min + (max - min) * pcg_float(rng);
}

// Random in unit sphere using PCG
__device__ float3 pcg_random_in_unit_sphere(PCGState* rng) {
    while (true) {
        float x = pcg_float_range(rng, -1.0f, 1.0f);
        float y = pcg_float_range(rng, -1.0f, 1.0f);
        float z = pcg_float_range(rng, -1.0f, 1.0f);
        float3 p = make_float3(x, y, z);
        if (dot(p, p) < 1.0f) return p;
    }
}

// Random unit vector using PCG
__device__ __forceinline__ float3 pcg_random_unit_vector(PCGState* rng) {
    return normalize(pcg_random_in_unit_sphere(rng));
}

// Random in unit disk using PCG
__device__ float2 pcg_random_in_unit_disk(PCGState* rng) {
    float2 p;
    do {
        p = make_float2(
            pcg_float(rng) * 2.0f - 1.0f,
            pcg_float(rng) * 2.0f - 1.0f
        );
    } while ((p.x * p.x + p.y * p.y) >= 1.0f);
    return p;
}

// ============================================================================
// LEGACY CURAND-BASED FUNCTIONS (kept for compatibility)
// ============================================================================

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

// Sample a direction within a cone around the target direction
__device__ inline float3 sample_sphere_cap(const float3& dir, float angle_rad, curandState* rng) {
    if (angle_rad < 1e-4f) return dir;
    
    float r1 = curand_uniform(rng);
    float r2 = curand_uniform(rng);
    
    float cos_theta = 1.0f - r1 * (1.0f - cosf(angle_rad));
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * M_PI * r2;
    
    float3 local_dir = make_float3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);
    
    // Create local coordinate system
    float3 up = (fabsf(dir.y) < 0.999f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    float3 tangent = normalize(cross(up, dir));
    float3 bitangent = cross(dir, tangent);
    
    return tangent * local_dir.x + bitangent * local_dir.y + dir * local_dir.z;
}
