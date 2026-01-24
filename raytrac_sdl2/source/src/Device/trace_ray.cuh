// ===== trace_ray.cuh =====
#pragma once
#include <optix.h>
#include "payload.h"
#include "params.h"
#include "ray.h"

__device__ void trace_ray(const Ray& ray, OptixHitResult* payload, float tmin, float tmax) {
    unsigned int p0 = 0, p1 = 0;
    packPayload(payload, p0, p1);

    optixTrace(
        optixLaunchParams.handle,
        ray.origin,
        ray.direction,
        tmin,
        tmax,
        0.0f,
        0xFF,
        OPTIX_RAY_FLAG_NONE,
        0, 
        1, 
        0,
        p0, p1
    );
}




// Revert to simple signature without seed
__device__ void trace_shadow_ray(const Ray& ray, OptixHitResult* payload, float tmin, float tmax) {
    // Pack pointers to 32-bit registers
    unsigned int p0, p1;
    packPayload(payload, p0, p1);

    optixTrace(
        optixLaunchParams.handle,
        ray.origin,
        ray.direction,
        tmin,
        tmax,
        0.0f,
        0xFF,
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        0, // Ray type
        1, // Stride
        0, // Miss index
        p0, p1
    );
}
