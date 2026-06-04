// procedural_detail.cuh
// CUDA/OptiX port of procedural_detail.glsl — identical algorithms, CUDA types.
// Include after cuda_runtime.h.
//
// API  (all __device__ __forceinline__)
//   float2 pd_tileBreak   (float2 uv, float3 worldPos, float strength)
//   float  pd_dirt        (float3 worldPos, float scale)          → [0,1]
//   float  pd_roughnessVar(float3 worldPos, float scale)          → [-0.5, +0.5]

#pragma once
#include <cuda_runtime.h>

// ─── Hash ─────────────────────────────────────────────────────────────────────

__device__ __forceinline__ float pd_h2(float2 p) {
    float v = sinf(p.x * 127.1f + p.y * 311.7f) * 43758.5453123f;
    return v - floorf(v);
}

__device__ __forceinline__ float pd_h3(float3 p) {
    // fract(p * vec3(443.897, 441.423, 437.195)) — matches GLSL pd_hash3
    float tx = p.x * 443.897f; float qx = tx - floorf(tx);
    float ty = p.y * 441.423f; float qy = ty - floorf(ty);
    float tz = p.z * 437.195f; float qz = tz - floorf(tz);
    // p += dot(p, p.yzx + 19.19) — scalar broadcast
    float d = qx*(qy + 19.19f) + qy*(qz + 19.19f) + qz*(qx + 19.19f);
    qx += d; qy += d; qz += d;
    float r = (qx + qy) * qz;
    return r - floorf(r);
}

// ─── Value Noise ──────────────────────────────────────────────────────────────

__device__ __forceinline__ float pd_vnoise2(float2 p) {
    float2 i = make_float2(floorf(p.x), floorf(p.y));
    float2 f = make_float2(p.x - i.x, p.y - i.y);
    // quintic: eliminates block-edge C1/C2 discontinuities
    f.x = f.x * f.x * f.x * (f.x * (f.x * 6.0f - 15.0f) + 10.0f);
    f.y = f.y * f.y * f.y * (f.y * (f.y * 6.0f - 15.0f) + 10.0f);

    float a = pd_h2(i);
    float b = pd_h2(make_float2(i.x + 1.0f, i.y));
    float c = pd_h2(make_float2(i.x,         i.y + 1.0f));
    float d = pd_h2(make_float2(i.x + 1.0f,  i.y + 1.0f));
    return a + (b - a) * f.x + (c - a) * f.y + (a - b - c + d) * f.x * f.y;
}

__device__ __forceinline__ float pd_vnoise3(float3 p) {
    float3 i = make_float3(floorf(p.x), floorf(p.y), floorf(p.z));
    float3 f = make_float3(p.x - i.x, p.y - i.y, p.z - i.z);
    // quintic: eliminates block-edge C1/C2 discontinuities
    f.x = f.x * f.x * f.x * (f.x * (f.x * 6.0f - 15.0f) + 10.0f);
    f.y = f.y * f.y * f.y * (f.y * (f.y * 6.0f - 15.0f) + 10.0f);
    f.z = f.z * f.z * f.z * (f.z * (f.z * 6.0f - 15.0f) + 10.0f);

    float n000 = pd_h3(i);
    float n100 = pd_h3(make_float3(i.x+1,i.y,  i.z  ));
    float n010 = pd_h3(make_float3(i.x,  i.y+1,i.z  ));
    float n110 = pd_h3(make_float3(i.x+1,i.y+1,i.z  ));
    float n001 = pd_h3(make_float3(i.x,  i.y,  i.z+1));
    float n101 = pd_h3(make_float3(i.x+1,i.y,  i.z+1));
    float n011 = pd_h3(make_float3(i.x,  i.y+1,i.z+1));
    float n111 = pd_h3(make_float3(i.x+1,i.y+1,i.z+1));

    float x00 = n000*(1-f.x) + n100*f.x;
    float x10 = n010*(1-f.x) + n110*f.x;
    float x01 = n001*(1-f.x) + n101*f.x;
    float x11 = n011*(1-f.x) + n111*f.x;
    float y0  = x00*(1-f.y)  + x10*f.y;
    float y1  = x01*(1-f.y)  + x11*f.y;
    return y0*(1-f.z) + y1*f.z;
}

// ─── fBm (4-octave, domain-rotated from original p) ──────────────────────────
// All rotations applied to the ORIGINAL p — not cascaded through scaled/offset
// intermediates (which would produce asymmetric block artifacts).
// Two rotation angles for maximum de-correlation between octaves.
// cos(0.5)=0.8776 sin(0.5)=0.4794  |  cos(1.0)=0.5403 sin(1.0)=0.8415

__device__ __forceinline__ float pd_fbm(float3 p) {
    float rx1 = p.x*0.8776f - p.z*0.4794f;
    float rz1 = p.x*0.4794f + p.z*0.8776f;
    float rx2 = p.x*0.5403f - p.z*0.8415f;
    float rz2 = p.x*0.8415f + p.z*0.5403f;
    float3 p2 = make_float3(rx1*2.0f+1.7f, p.y*2.0f+9.2f, rz1*2.0f+3.5f);
    float3 p3 = make_float3(rx2*4.0f+8.3f, p.y*4.0f+2.8f, rz2*4.0f+5.1f);
    float3 p4 = make_float3(rx1*8.0f+4.1f, p.y*8.0f+6.7f, rz2*8.0f+2.3f);
    return pd_vnoise3(p)  * 0.5000f
         + pd_vnoise3(p2) * 0.2500f
         + pd_vnoise3(p3) * 0.1250f
         + pd_vnoise3(p4) * 0.0625f;
}

// ─── Tile-Break ───────────────────────────────────────────────────────────────
// Perturbs UV by world-position noise to break repeating texture tiling seams.
// strength: 0.0–0.35 recommended.

__device__ __forceinline__ float2 pd_tileBreak(float2 uv, float3 worldPos, float strength) {
    float2 seed0 = make_float2(uv.x*4.17f + worldPos.x*0.13f,
                                uv.y*4.17f + worldPos.z*0.13f);
    float2 seed1 = make_float2(uv.x*4.17f + worldPos.y*0.13f + 3.3f,
                                uv.y*4.17f + worldPos.z*0.13f + 7.1f);
    float n0 = pd_vnoise2(seed0);
    float n1 = pd_vnoise2(seed1);
    return make_float2(uv.x + (n0 - 0.5f) * strength,
                       uv.y + (n1 - 0.5f) * strength);
}

// ─── Procedural Dirt ──────────────────────────────────────────────────────────
// Returns [0,1]. Multiply albedo: albedo = mix(albedo, albedo*dirtColor, dirt*str)
// scale: world-space frequency (1–5 typical)

__device__ __forceinline__ float pd_dirt(float3 worldPos, float scale) {
    float3 sp = make_float3(worldPos.x*scale, worldPos.y*scale, worldPos.z*scale);
    float n = pd_fbm(sp);
    // smoothstep(0.62, 0.28, n): 1 when n<=0.28, 0 when n>=0.62
    float t = (n - 0.62f) / (0.28f - 0.62f);
    t = fmaxf(0.0f, fminf(1.0f, t));
    return t * t * (3.0f - 2.0f * t);
}

// ─── Roughness Variation ──────────────────────────────────────────────────────
// Returns [-0.5, +0.5]. Add to base roughness and clamp.

__device__ __forceinline__ float pd_roughnessVar(float3 worldPos, float scale) {
    float3 sp = make_float3(worldPos.x*scale*2.5f + 5.5f,
                             worldPos.y*scale*2.5f + 3.1f,
                             worldPos.z*scale*2.5f + 8.9f);
    return pd_vnoise3(sp) - 0.5f;
}
