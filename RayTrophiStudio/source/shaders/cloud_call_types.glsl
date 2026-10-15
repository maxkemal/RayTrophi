// Interface of cloud_callable.rcall: the ONE stage that holds the cloud
// transport (marcher, reference walk, precipitation, cirrus) and the sun's
// cloud shadow.
//
// Why a callable (2026-10-13): included in every hit and miss stage, that
// code was compiled by the driver five times (closesthit, sphere_closesthit,
// volume_closesthit, hair_closesthit, miss), each copy fully inlined (-O).
// The RT pipeline compile went from ~3 to ~11 minutes over the atmosphere
// batches. A callable stage is compiled once.
#ifndef CLOUD_CALL_TYPES_GLSL
#define CLOUD_CALL_TYPES_GLSL

// Callable data location (its own namespace; kept apart from the ray payload
// locations 0/1 anyway for readability) and the callable SBT record.
#define CLOUD_CALL_LOCATION 7
#define CLOUD_CALL_SBT_INDEX 0

// op
#define CLOUD_CALL_RENDER 0u   // cloud transport along [0, tLimit]
#define CLOUD_CALL_SUN_T  1u   // cloud transmittance toward the sun (surface NEE)
// Near-camera drops in front of [0, tLimit] of a camera ray (near_precip.glsl):
// in = origin (camera), dir, tLimit (surface hit), pixelAngle (rad / pixel);
// out = inscatter (drop radiance in front), T (what lies behind is * T).
#define CLOUD_CALL_NEAR_PRECIP 2u

// flags (CLOUD_CALL_RENDER)
#define CLOUD_CALL_CAMERA_RAY  1u   // camera segment: full steps / reference walk
// Sky light on the clouds = the miss shader's sky without the sun disk.
// Unset: the sky-view LUT texel (hit segments). The two differ by the
// multi-scatter boost and the environment overlay; kept as they were.
#define CLOUD_CALL_MISS_AMBIENT 2u

struct CloudCall {
    vec3  origin;
    float tLimit;
    vec3  dir;
    uint  seed;
    vec3  inscatter;   // out
    float T;           // out: transmittance (CLOUD_CALL_SUN_T: the result)
    float depth;       // out: first-collision / weighted depth, -1 = none
    uint  op;
    uint  flags;
    float pixelAngle;  // CLOUD_CALL_NEAR_PRECIP only
};

#endif // CLOUD_CALL_TYPES_GLSL
