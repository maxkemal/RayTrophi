// ===== trace_ray.cuh =====
#pragma once
#include <optix.h>
#include "params.h"
#include "payload.h"
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
        0,                 // SBT offset (Radiance = 0)
        0,                 // SBT stride (Must be 0 to share records among all primitives!)
        0,                 // Miss index
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
        1,                 // SBT offset (Shadow = 1)
        0,                 // SBT stride (Must be 0!)
        1,                 // Miss index (SHADOW_RAY_TYPE)
        p0, p1
    );
}
