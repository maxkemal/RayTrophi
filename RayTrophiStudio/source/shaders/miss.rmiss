/*
 * RayTrophi Studio — Vulkan Miss Shader
 * Procedural Sky — Preetham/Gradient hybrid, HDR-correct
 *
 * v3 — Ghost düzeltmesi:
 *   - Push constant struct raygen v3 ile TAM eşleşme (pad[3] → pad0/pad1/pad2)
 *   - Ufuk altı ground rengi: scatter=false, radiance=sabit koyu renk
 *   - payload.attenuation sıfırlandı — miss sonrası stale değer kalmaz
 *   - skyColor fonksiyonu dışına çıkarıldı, main içinde temiz akış
 */

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : require

// ============================================================
// Push Constants — raygen v3 ile BİREBİR eşleşmeli
// pad[3] array değil, pad0/pad1/pad2 ayrı field
// ============================================================
layout(push_constant) uniform CameraPC {
    vec4  origin;
    vec4  lowerLeft;
    vec4  horizontal;
    vec4  vertical;
    uint  frameCount;
    uint  minSamples;
    uint  lightCount;
    float varianceThreshold;
    uint  maxSamples;
    float exposure_factor;
} cam;

// ============================================================
// Payload — shared ABI, single source of truth
// ============================================================
#include "rt_payload.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

#include "rt_sky.glsl"
// Clouds: flags and bounds only; the transport runs in cloud_callable.rcall.
#include "cloud_rt.glsl"
#include "cloud_call.glsl"

vec3 skyColor(vec3 dir) { return skyColorImpl(dir, true, gl_WorldRayOriginEXT); }

// ============================================================
// Main
// ============================================================
void main() {
    payload.skipGasVolumes = false;
    vec3 dir = normalize(gl_WorldRayDirectionEXT);

    // ----------------------------------------------------------
    // Payload'u tamamen sıfırla
    // Miss sonrası closesthit'ten kalan stale attenuation
    // bir sonraki bounce'ta (olmayacak ama) throughput'u bozmasın
    // ----------------------------------------------------------
    payload.attenuation   = vec3(0.0);   // Miss: artık throughput yok
    payload.scatterOrigin = vec3(0.0);
    payload.scatterDir    = vec3(0.0);
    payload.scattered     = false;        // Bounce durur

    // ----------------------------------------------------------
    // Sky radiance — raygen: radiance += throughput * payload.radiance
    // ----------------------------------------------------------
    vec3 sky = skyColor(dir);
    // Clouds (Faz 3b): only the Nishita sky has them; the adapter clears the
    // render flag in every other world mode and while the textures build.
    if (worldData.w.mode == 2 && cloudRtEnabled() &&
        (payload.primaryMeta & PL_NO_SKY_RADIANCE) == 0u) {
        bool cameraRay = (payload.primaryMeta & PL_CAMERA_SEGMENT) != 0u;
        vec3 inscatter;
        float T, depth;
        cloudCallRender(gl_WorldRayOriginEXT, dir, CLOUD_RT_MAX_DIST, cameraRay, true,
                        payload.seed, inscatter, T, depth);
        sky = sky * T + inscatter;
    }
    payload.radiance = sky;
}
