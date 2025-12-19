// ===== trace_ray.cuh =====
#pragma once
#include <optix.h>
#include "payload.h"
#include "params.h"
#include "ray.h"

__device__ void trace_ray(const Ray& ray, OptixHitResult* result, float tmin = 0.01f, float tmax = 1e16f) {
    unsigned int p0, p1;
    packPayload(result, p0, p1);

    optixTrace(
        optixLaunchParams.handle,
        ray.origin,
        ray.direction,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, // ← ray_type = 0 (primary / bounce ray)
        1, // ← toplam ray tipi (primary + shadow)
        0,
        p0, p1
    );
}


__device__ void trace_shadow_ray(const Ray& ray, OptixHitResult* payload, float tmin, float tmax) {
    unsigned int p0 = 0, p1 = 0;
    packPayload<OptixHitResult>(payload, p0, p1);
    optixTrace(
        optixLaunchParams.handle,
        ray.origin,
        ray.direction,
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, // ← RayType artık primary ile aynı
        1, // toplam materyal sayısı
        0,
        p0, p1
    );
}




