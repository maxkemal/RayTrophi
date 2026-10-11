// Finite cloud transport in front of a real RT hit. Include after the world,
// atmosphere samplers and location-0 payload declarations.
#ifndef CLOUD_HIT_SEGMENT_GLSL
#define CLOUD_HIT_SEGMENT_GLSL

#define CLOUD_RT_LIGHTING
#include "cloud_rt.glsl"

vec3 cloudSkyAmbient(vec3 direction) {
    if (worldData.w._pad5 == 0) return vec3(0.0);
    float u = atan(direction.z, direction.x) / (2.0 * CLOUD_PI);
    if (u < 0.0) u += 1.0;
    float v = (1.0 - clamp(direction.y, -1.0, 1.0)) * 0.5;
    return textureLod(atmosphereLUTs[1], vec2(u, v), 0.0).rgb;
}

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
    cloudRenderLimited(gl_WorldRayOriginEXT, direction, distance, cameraRay, payload.seed,
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
