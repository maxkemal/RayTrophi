// Finite cloud transport in front of a real RT hit. Include after the world,
// atmosphere samplers and location-0 payload declarations. The transport runs
// in cloud_callable.rcall (cloud_call_types.glsl); this stage keeps only the
// flags and the cheap bounds test.
#ifndef CLOUD_HIT_SEGMENT_GLSL
#define CLOUD_HIT_SEGMENT_GLSL

#include "cloud_rt.glsl"
#include "cloud_call.glsl"

void cloudApplyHitSegment(float distance) {
    if (worldData.w.mode != 2 || !cloudRtEnabled() || distance <= 0.0 ||
        (payload.primaryMeta & PL_NO_SKY_RADIANCE) != 0u) return;
    vec3 direction = normalize(gl_WorldRayDirectionEXT);
    distance = min(distance, CLOUD_RT_MAX_DIST);
    if (!cloudSegmentMayContain(gl_WorldRayOriginEXT, direction, distance)) return;
    bool straightPass = plIsTransparentPass(payload, direction);
    vec3 inscatter;
    float T, depth;
    bool cameraRay = (payload.primaryMeta & PL_CAMERA_SEGMENT) != 0u;
    cloudCallRender(gl_WorldRayOriginEXT, direction, distance, cameraRay, false, payload.seed,
                    inscatter, T, depth);
    payload.radiance = payload.radiance * T + inscatter;
    payload.attenuation *= T;
    if (straightPass) payload.bounceType = BOUNCE_MEDIUM_PASS;
}

void cloudApplyVolumeSegment() {
    float distance = gl_HitTEXT;
    if (payload.scattered) {
        // Straight medium exits and solid handoffs publish their true endpoint.
        // A same-origin retrace must defer the prefix to the next real hit.
        distance = max(0.0, dot(payload.scatterOrigin - gl_WorldRayOriginEXT,
                                normalize(gl_WorldRayDirectionEXT)));
    }
    cloudApplyHitSegment(distance);
}

#endif
