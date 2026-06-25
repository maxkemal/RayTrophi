/*
 * RayTrophi Studio — Vulkan Foam Sphere Closest-Hit Shader
 *
 * Shades foam / fluid-splat point spheres (one analytic sphere GAS per foam
 * type — see sphere_intersection.rint). Foam materials are plain PrincipledBSDF
 * presets (no textures), so this is a COMPACT path-tracing shader, not the full
 * 2600-line surface closesthit: one stochastic dielectric-vs-diffuse split
 * reproduces all three looks from a single material field (transmission):
 *   Spray  (transmission ≈ 1.0) → smooth dielectric water droplet (IOR 1.33)
 *   Bubble (transmission ≈ 0.65)→ mostly refractive air bubble + faint emission
 *   Foam   (transmission = 0.0) → rough white scatter (surface whitewater)
 *
 * No next-event estimation here (the raygen path-tracing loop gathers light by
 * continuing the path) — this keeps the shader self-contained and low-risk.
 * Geometry comes from the intersection shader's hit attribute; the per-sphere
 * material index comes from the combined foam buffer (binding 18) indexed by
 * gl_PrimitiveID — the whole foam cloud is ONE AABB BLAS / ONE TLAS instance.
 *
 * Payload + Material layouts MUST match raygen.rgen / closesthit.rchit.
 */
#version 460
#extension GL_EXT_ray_tracing         : require
#extension GL_EXT_scalar_block_layout : require

const float PI         = 3.14159265358979323846;
const float RAY_OFFSET  = 1e-3;
const uint  BOUNCE_SPECULAR     = 0u;
const uint  BOUNCE_DIFFUSE      = 1u;
const uint  BOUNCE_TRANSMISSION = 2u;

// ─── Payload (must match raygen.rgen) ────────────────────────────────────────
struct RayPayload {
    vec3     radiance;
    vec3     attenuation;
    vec3     scatterOrigin;
    vec3     scatterDir;
    uint     seed;
    bool     scattered;
    bool     hitEmissive;
    uint     occluded;
    bool     skipAABBs;
    vec3     primaryAlbedo;
    vec3     primaryNormal;
    uint     primaryHit;
    float    primaryTransmission;
    float    primaryMetallic;
    uint     bounceType;
    uint     primaryMaterialId;
};
layout(location = 0) rayPayloadInEXT RayPayload payload;

// ─── Hit attribute from sphere_intersection.rint ─────────────────────────────
hitAttributeEXT vec3 sphereNormal;   // world-space outward normal

// ─── Material (binding 2) — full layout for correct scalar stride ────────────
// Material struct — single source of truth shared by every material-reading shader.
#include "material_struct.glsl"
layout(set = 0, binding = 2, scalar) readonly buffer MaterialBuffer { Material m[]; } materials;

// ─── Combined foam sphere buffer (binding 18) — per-sphere material index ─────
struct FoamSphereGPU {
    vec4 centerRadius;
    uint matId;
    uint _p0;
    uint _p1;
    uint _p2;
};
layout(set = 0, binding = 18, scalar) readonly buffer FoamSphereSSBO { FoamSphereGPU s[]; } foamSpheres;

// ─── RNG (PCG, matches closesthit.rchit) ─────────────────────────────────────
uint pcgNext(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float rnd(inout uint seed) { return float(pcgNext(seed)) * (1.0 / 4294967296.0); }

void buildONB(in vec3 n, out vec3 tangent, out vec3 bitangent) {
    float sign_ = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (sign_ + n.z);
    float b = n.x * n.y * a;
    tangent   = vec3(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
    bitangent = vec3(b, sign_ + n.y * n.y * a, -n.y);
}

vec3 cosineSampleHemisphere(vec3 n, inout uint seed) {
    float u1 = rnd(seed);
    float u2 = rnd(seed);
    float r   = sqrt(u1);
    float phi = 2.0 * PI * u2;
    vec3 t, b;
    buildONB(n, t, b);
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0, 1.0 - u1));
    return normalize(t * x + b * y + n * z);
}

void main()
{
    uint matIdx = foamSpheres.s[uint(gl_PrimitiveID)].matId;
    Material mat = materials.m[matIdx];

    vec3  albedo       = vec3(mat.albedo_r, mat.albedo_g, mat.albedo_b);
    vec3  emission     = vec3(mat.emission_r, mat.emission_g, mat.emission_b) * mat.emission_strength;
    float transmission = clamp(mat.transmission, 0.0, 1.0);
    float ior          = (mat.ior > 1.0) ? mat.ior : 1.33;

    vec3 rayDir   = normalize(gl_WorldRayDirectionEXT);
    vec3 Ng       = normalize(sphereNormal);
    bool frontFace = dot(Ng, rayDir) < 0.0;
    vec3 N        = frontFace ? Ng : -Ng;          // faces the incoming ray
    vec3 hitPos   = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;

    uint seed = payload.seed;

    // Emission (foam/bubble glow) — add, keep tracing. attenuation is 1.0 on entry
    // (raygen resets it per bounce), so this is the full material emission, matching
    // closesthit.rchit's `radiance = emColor * emStrength`.
    payload.radiance   += payload.attenuation * emission;
    payload.hitEmissive = (dot(emission, emission) > 1e-6);

    // Primary AOV (first surface hit only) for the denoiser / stylize pass.
    if (payload.primaryHit == 0u) {
        payload.primaryAlbedo       = albedo;
        payload.primaryNormal       = N;
        payload.primaryHit          = 1u;
        payload.primaryTransmission = transmission;
        payload.primaryMetallic     = 0.0;
        payload.primaryMaterialId   = matIdx;
    }

    if (rnd(seed) < transmission) {
        // ── Dielectric (Fresnel reflect / refract) ──
        float eta  = frontFace ? (1.0 / ior) : ior;
        float cosi = clamp(dot(-rayDir, N), 0.0, 1.0);
        float r0   = (1.0 - ior) / (1.0 + ior); r0 *= r0;
        float fres = r0 + (1.0 - r0) * pow(1.0 - cosi, 5.0);
        vec3  refr = refract(rayDir, N, eta);
        vec3  dir;
        if (refr == vec3(0.0) || rnd(seed) < fres) {
            dir = reflect(rayDir, N);                 // TIR or Fresnel reflect
            payload.bounceType = BOUNCE_SPECULAR;
        } else {
            dir = refr;                               // refract
            payload.bounceType = BOUNCE_TRANSMISSION;
        }
        dir = normalize(dir);
        payload.scatterOrigin = hitPos + dir * RAY_OFFSET;
        payload.scatterDir    = dir;
        payload.attenuation  *= albedo;               // white foam = no tint
        payload.scattered     = true;
    } else {
        // ── Rough white scatter (surface foam) ──
        vec3 dir = cosineSampleHemisphere(N, seed);
        payload.scatterOrigin = hitPos + N * RAY_OFFSET;
        payload.scatterDir    = dir;
        payload.attenuation  *= albedo;
        payload.scattered     = true;
        payload.bounceType    = BOUNCE_DIFFUSE;
    }

    payload.seed = seed;
}
