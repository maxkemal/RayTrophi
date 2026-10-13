// Caller side of cloud_callable.rcall (see cloud_call_types.glsl). Include in
// a stage that may call executeCallableEXT (raygen, closest-hit, miss).
#ifndef CLOUD_CALL_GLSL
#define CLOUD_CALL_GLSL

#include "cloud_call_types.glsl"

layout(location = CLOUD_CALL_LOCATION) callableDataEXT CloudCall cloudCall;

void cloudCallRender(vec3 o, vec3 d, float tLimit, bool cameraRay, bool missAmbient, uint seed,
                     out vec3 inscatter, out float T, out float depth) {
    cloudCall.origin = o;
    cloudCall.tLimit = tLimit;
    cloudCall.dir = d;
    cloudCall.seed = seed;
    cloudCall.op = CLOUD_CALL_RENDER;
    cloudCall.flags = (cameraRay ? CLOUD_CALL_CAMERA_RAY : 0u) |
                      (missAmbient ? CLOUD_CALL_MISS_AMBIENT : 0u);
    executeCallableEXT(CLOUD_CALL_SBT_INDEX, CLOUD_CALL_LOCATION);
    inscatter = cloudCall.inscatter;
    T = cloudCall.T;
    depth = cloudCall.depth;
}

// Transmittance of the cloud field from p toward the sun (direction wi).
float cloudCallSunTransmittance(vec3 p, vec3 wi, uint seed) {
    cloudCall.origin = p;
    cloudCall.dir = wi;
    cloudCall.seed = seed;
    cloudCall.op = CLOUD_CALL_SUN_T;
    cloudCall.flags = 0u;
    executeCallableEXT(CLOUD_CALL_SBT_INDEX, CLOUD_CALL_LOCATION);
    return cloudCall.T;
}

#endif // CLOUD_CALL_GLSL
