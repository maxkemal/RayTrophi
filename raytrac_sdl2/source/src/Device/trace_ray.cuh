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

// Lightweight shadow trace: returns 1 if any geometry is hit before tmax,
// 0 otherwise. Uses a single OptiX payload register (no pointer-pack, no
// stack-allocated OptixHitResult). __closesthit__shadow only needs to flip
// a flag, so the previous 400-byte payload was pure overhead — one shadow
// ray per bounce per surface NEE, this is hot.
__device__ unsigned int trace_shadow_ray(const Ray& ray, float tmin, float tmax) {
    unsigned int hit = 0;
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
        hit                // payload register 0
    );
    return hit;
}
