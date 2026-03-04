/*
 * RayTrophi Studio — Vulkan Hair Closest-Hit Shader
 * Marschner Hair BSDF (R + TT + TRT + MS lobes)
 *
 * Features:
 *   - Marschner et al. 2003 with d'Eon/Chiang improvements
 *   - Fresnel dielectric reflection, cuticle angle shifts
 *   - Color modes: Direct, Melanin (physical), Absorption, Root UV Map
 *   - Specular tint, artistic tint, coat/fur layer  
 *   - Root-to-tip color gradient
 *   - Diffuse softness (multiple scattering weight)
 *   - Emission support
 *   - Shadow rays for all lights
 *   - Synced with OptiX hair_bsdf.cuh / HairBSDF.cpp
 *
 * SSBO binding 10: HairSegmentGPU[]  (materialID lookup)
 * SSBO binding 11: HairGpuMaterial[] (144 bytes each, scalar layout)
 */

#version 460
#extension GL_EXT_ray_tracing                          : require
#extension GL_EXT_scalar_block_layout                  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// ─── Constants ───────────────────────────────────────────────────────────────
const float PI      = 3.14159265358979323846;
const float INV_PI  = 0.31830988618379067154;
const float EPSILON = 1e-4;

// ─── Push Constants ──────────────────────────────────────────────────────────
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

// ─── Payload ─────────────────────────────────────────────────────────────────
struct RayPayload {
    vec3  radiance;
    vec3  attenuation;
    vec3  scatterOrigin;
    vec3  scatterDir;
    uint  seed;
    bool  scattered;
    bool  hitEmissive;
    uint  occluded;
    bool  skipAABBs;    // set by volume_closesthit when a solid surface is found inside
};

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT   bool       shadowOccluded;

// ─── Hit Attribute (hair_intersection.rint) ──────────────────────────────────
hitAttributeEXT vec4 hairAttrib;

// ─── Descriptor Bindings ─────────────────────────────────────────────────────
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

struct LightData {
    vec4 position;   // xyz=pos/dir, w=type (0=point,1=dir,2=spot)
    vec4 color;      // rgb=color,   w=intensity
    vec4 params;     // x=radius,    yzw=...
    vec4 direction;  // xyz=dir,     w=outerAngle
};
layout(set = 0, binding = 3, scalar) readonly buffer LightBuffer {
    LightData l[];
} lights;

// ─── Binding 10: Hair Segment SSBO ──────────────────────────────────────────
struct HairSegmentGPU {
    vec4 cp0;
    vec4 cp1;
    vec4 cp2;
    vec4 cp3;
    uint strandID;
    uint groomID;
    uint materialID;
    uint padding;
};
layout(set = 0, binding = 10, scalar) readonly buffer HairSegmentSSBO {
    HairSegmentGPU hairSegs[];
};

// ─── Binding 11: Hair Material SSBO (144 bytes, scalar) ─────────────────────
struct HairGpuMaterial {
    // Block 1: Color & Roughness
    vec3  baseColor;         // offset  0
    float roughness;          // offset 12
    // Block 2: Physical
    float melanin;            // offset 16
    float melaninRedness;     // offset 20
    float ior;                // offset 24
    float cuticleAngle;       // offset 28
    // Block 3: Mode & Surface
    uint  colorMode;          // offset 32
    float radialRoughness;    // offset 36
    float specularTint;       // offset 40
    float diffuseSoftness;    // offset 44
    // Block 4: Tint
    vec3  tintColor;          // offset 48
    float tint;               // offset 60
    // Block 5: Coat
    vec3  coatTint;           // offset 64
    float coat;               // offset 76
    // Block 6: Emission
    vec3  emission;           // offset 80
    float emissionStrength;   // offset 92
    // Block 7: Root-Tip Gradient
    vec3  tipColor;           // offset 96
    float rootTipBalance;     // offset 108
    // Block 8: Absorption & Gradient Flag
    vec3  absorption;         // offset 112
    uint  enableGradient;     // offset 124
    // Block 9: Random & ID
    float randomHue;          // offset 128
    float randomValue;        // offset 132
    uint  groomID;            // offset 136
    float pad;                // offset 140
};
layout(set = 0, binding = 11, scalar) readonly buffer HairMaterialSSBO {
    HairGpuMaterial hairMats[];
};

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions (synced with hair_bsdf.cuh)
// ═══════════════════════════════════════════════════════════════════════════════

float sqr(float x) { return x * x; }

float safe_sqrt(float x) { return sqrt(max(0.0, x)); }

float safe_asin(float x) { return asin(clamp(x, -1.0, 1.0)); }

// Exact Fresnel for dielectric
float fresnel_dielectric(float cosThetaI, float eta) {
    float sinThetaI = safe_sqrt(1.0 - cosThetaI * cosThetaI);
    float sinThetaT = sinThetaI / eta;
    if (sinThetaT >= 1.0) return 1.0; // Total internal reflection
    float cosThetaT = safe_sqrt(1.0 - sinThetaT * sinThetaT);
    float rs = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    return 0.5 * (rs * rs + rp * rp);
}

// Melanin to absorption coefficient (synced with CPU/OptiX)
vec3 melanin_to_absorption(float melanin, float redness) {
    float eumelanin  = melanin * (1.0 - redness * 0.5);
    float pheomelanin = melanin * redness;
    vec3 eumelaninSigma  = vec3(0.419, 0.697, 1.37);
    vec3 pheomelaninSigma = vec3(0.187, 0.4, 1.05);
    return eumelaninSigma * eumelanin * 8.0 + pheomelaninSigma * pheomelanin * 8.0;
}

// Gaussian distribution
float gaussian(float x, float variance) {
    return exp(-x * x / (2.0 * variance)) / sqrt(2.0 * PI * variance);
}

// Logistic distribution (Disney 2016)
float logistic_pdf(float x, float s) {
    float e = exp(-abs(x) / s);
    return e / (s * sqr(1.0 + e));
}

// Normalized azimuthal scattering (N term)
float eval_N(float phi, float s, float phiTarget) {
    float diff = phi - phiTarget;
    diff -= 2.0 * PI * floor(diff * (0.5 * INV_PI) + 0.5);
    float norm = (1.0 / (1.0 + exp(-PI / s))) - (1.0 / (1.0 + exp(PI / s)));
    return logistic_pdf(diff, s) / max(norm, 0.1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Marschner Hair BSDF Evaluation
// ═══════════════════════════════════════════════════════════════════════════════

vec3 hair_bsdf_eval(
    vec3 wo, vec3 wi, vec3 tangent,
    HairGpuMaterial mat, vec3 sigma_a,
    float h, float vStrand
) {
    // 1. Hair coordinates
    float sinThetaO = dot(wo, tangent);
    float sinThetaI = dot(wi, tangent);
    float cosThetaO = safe_sqrt(1.0 - sinThetaO * sinThetaO);
    float cosThetaI = safe_sqrt(1.0 - sinThetaI * sinThetaI);

    if (cosThetaO < 1e-4 || cosThetaI < 1e-4)
        return vec3(0.0);

    // Azimuthal angle
    vec3 wo_perp = normalize(wo - sinThetaO * tangent);
    vec3 wi_perp = normalize(wi - sinThetaI * tangent);
    float cosPhi = dot(wo_perp, wi_perp);
    float sinPhi = dot(cross(wo_perp, wi_perp), tangent);
    float phi = atan(sinPhi, cosPhi);

    float alpha = mat.cuticleAngle;
    float eta   = mat.ior;

    // Root-to-tip gradient
    if (mat.enableGradient != 0u) {
        vec3 tipSigma = -log(max(mat.tipColor, vec3(0.001))) * 0.5;
        float t = vStrand * mat.rootTipBalance;
        sigma_a = mix(sigma_a, tipSigma, t);
    }

    // Gamma angles for path length / azimuthal shifts
    float gammaO = safe_asin(h);
    float gammaT = safe_asin(h / eta);
    float cosGammaO = cos(gammaO);
    float cosGammaT = cos(gammaT);

    // Precompute lobe variances from roughness (matching convertToGpu)
    float baseR = max(mat.roughness, 0.08);
    float v_R   = sqr(baseR * 1.2);
    float v_TT  = sqr(baseR);
    float v_TRT = sqr(baseR);
    float s_R   = baseR * 1.8 * 0.5;
    float s_TT  = baseR * 0.7 * 0.5;
    float s_TRT = baseR * 2.2 * 0.5;
    float s_MS  = max(baseR * 0.7 * 10.0, 0.2) * 0.5;

    // ════════════════════════════════════════════════════════════════════════
    // R Lobe (Primary Specular)
    // ════════════════════════════════════════════════════════════════════════
    float F_R = fresnel_dielectric(cosThetaO, eta);
    float sinThetaSum_R = sinThetaI + sinThetaO - 2.0 * alpha;
    float M_R = gaussian(sinThetaSum_R, v_R);
    float phi_R = -2.0 * gammaO;
    float N_R = eval_N(phi, s_R, phi_R);

    // Specular tint: white → hair body color
    vec3 hairBodyColor = exp(-sigma_a * 0.5);
    vec3 specColor = mix(vec3(1.0), hairBodyColor, mat.specularTint);
    vec3 R = specColor * (F_R * M_R * N_R);

    // ════════════════════════════════════════════════════════════════════════
    // TT Lobe (Transmission — back-lit highlight)
    // ════════════════════════════════════════════════════════════════════════
    float cosThetaD = cos((asin(sinThetaO) - asin(sinThetaI)) * 0.5);
    float path_L = 2.0 * cosGammaT / max(cosThetaD, 0.1);
    vec3 A = exp(-sigma_a * path_L);

    float F_TT = (1.0 - F_R) * (1.0 - fresnel_dielectric(cosGammaT, 1.0 / eta));
    float sinThetaSum_TT = sinThetaI + sinThetaO + alpha;
    float M_TT = gaussian(sinThetaSum_TT, 0.5 * v_TT);
    float phi_TT = PI + 2.0 * gammaT - 2.0 * gammaO;
    float N_TT = eval_N(phi, s_TT, phi_TT);
    vec3 TT = A * (F_TT * M_TT * N_TT);

    // ════════════════════════════════════════════════════════════════════════
    // TRT Lobe (Internal Reflection — front-lit colored specular)
    // ════════════════════════════════════════════════════════════════════════
    float F_internal = fresnel_dielectric(cosGammaT, 1.0 / eta);
    float F_TRT = (1.0 - F_R) * F_internal * (1.0 - fresnel_dielectric(cosGammaT, 1.0 / eta));
    float sinThetaSum_TRT = sinThetaI + sinThetaO - 4.0 * alpha;
    float M_TRT = gaussian(sinThetaSum_TRT, 2.0 * v_TRT);
    float phi_TRT = 4.0 * gammaT - 2.0 * gammaO;
    float N_TRT = eval_N(phi, s_TRT, phi_TRT);
    vec3 TRT = (A * A) * (F_TRT * M_TRT * N_TRT);

    // ════════════════════════════════════════════════════════════════════════
    // Multiple Scattering (Bulk body color — controlled by diffuseSoftness)
    // ════════════════════════════════════════════════════════════════════════
    float N_MS = eval_N(phi, s_MS, 0.0);
    float msWeight = mat.diffuseSoftness * 1.2;
    vec3 MS = (A * A) * (msWeight * N_MS);

    // Combine lobes
    vec3 bsdf = R + TT + TRT + MS;

    // ─── Artistic Tint ───────────────────────────────────────────────────
    if (mat.tint > 0.0) {
        vec3 tinted = bsdf * mat.tintColor;
        bsdf = mix(bsdf, tinted, mat.tint);
    }

    // ─── Coat / Fur Layer ────────────────────────────────────────────────
    if (mat.coat > 0.0) {
        float coatIOR = 1.33; // Water film
        float coatFresnel = fresnel_dielectric(cosThetaO, coatIOR) * mat.coat;
        float coatRough = max(mat.roughness * 0.3, 0.02);
        float coatVar = coatRough * coatRough;
        float M_coat = gaussian(sinThetaI + sinThetaO, coatVar);
        float s_coat = max(coatRough * 0.8 * 0.5, 0.01);
        float N_coat = eval_N(phi, s_coat, -2.0 * gammaO);
        vec3 coatSpec = mat.coatTint * (coatFresnel * M_coat * N_coat);
        bsdf = bsdf * (1.0 - coatFresnel) + coatSpec;
    }

    // ─── Emission ────────────────────────────────────────────────────────
    if (mat.emissionStrength > 0.0) {
        bsdf += mat.emission * mat.emissionStrength;
    }

    // Final normalization
    float denom = max(cosThetaD * cosThetaD, 0.001);
    bsdf /= denom;

    // Firefly clamp
    bsdf = min(bsdf, vec3(100.0));

    return bsdf;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

void main()
{
    vec3 hitPoint = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 tangent  = normalize(hairAttrib.xyz);
    float vStrand = hairAttrib.w; // 0=root, 1=tip

    vec3 V = -normalize(gl_WorldRayDirectionEXT);

    // Material lookup
    uint segIdx = uint(gl_PrimitiveID);
    uint matID  = hairSegs[segIdx].materialID;
    HairGpuMaterial mat = hairMats[matID];

    // ─── Compute sigma_a based on color mode ─────────────────────────────
    vec3 sigma_a;
    vec3 hairColor;
    if (mat.colorMode == 1u) {
        // Melanin (physical)
        sigma_a = melanin_to_absorption(mat.melanin, mat.melaninRedness);
        hairColor = exp(-sigma_a * 0.5);
    } else if (mat.colorMode == 2u) {
        // Explicit absorption
        sigma_a = mat.absorption;
        hairColor = exp(-sigma_a * 0.5);
    } else {
        // Direct color (0) or Root UV Map (3) — derive sigma from color
        hairColor = max(mat.baseColor, vec3(0.001));
        vec3 c = clamp(hairColor, vec3(0.001), vec3(0.99));
        sigma_a = -log(c) * 0.5;
    }

    // ─── Compute h (azimuthal offset) ────────────────────────────────────
    vec3 toHit = hitPoint - gl_WorldRayOriginEXT;
    vec3 planeN = normalize(cross(tangent, toHit));
    vec3 normal = -normalize(cross(planeN, tangent));
    vec3 wo_perp = V - dot(V, tangent) * tangent;
    vec3 bitangent = cross(tangent, normalize(wo_perp));
    float h = clamp(dot(normal, bitangent), -1.0, 1.0);

    // ─── Ambient ─────────────────────────────────────────────────────────
    vec3 radiance = hairColor * 0.07;

    // ─── Light Loop ──────────────────────────────────────────────────────
    uint numLights = cam.lightCount;
    for (uint li = 0u; li < numLights; li++) {
        LightData light = lights.l[li];
        vec3  lightColor = light.color.rgb * light.color.w;
        vec3  lightDir;
        float lightDist;
        float atten = 1.0;

        int lightType = int(light.position.w);
        if (lightType == 1) {
            lightDir  = -normalize(light.direction.xyz);
            lightDist = 1e6;
        } else {
            vec3 toLight = light.position.xyz - hitPoint;
            lightDist = length(toLight);
            if (lightDist < 1e-4) continue;
            lightDir = toLight / lightDist;
            float r = max(light.params.x, 0.001);
            atten = 1.0 / (1.0 + (lightDist * lightDist) / (r * r));
        }
        if (dot(lightDir, lightDir) < 0.5) continue;

        // Shadow ray
        float dotVT   = dot(V, tangent);
        vec3 sideNorm = normalize(V - dotVT * tangent);
        vec3 shadowOrig = hitPoint + sideNorm * max(mat.roughness, 5e-4);

        shadowOccluded = true;
        traceRayEXT(
            topLevelAS,
            gl_RayFlagsTerminateOnFirstHitEXT |
            gl_RayFlagsSkipClosestHitShaderEXT |
            gl_RayFlagsOpaqueEXT,
            // mask 0x01 = triangles only, skip volume AABBs (mask 0x02)
            0x01, 0, 0, 1,
            shadowOrig, 0.001, lightDir, lightDist - 0.002,
            1
        );
        if (shadowOccluded) continue;

        // Marschner BSDF evaluation
        vec3 bsdf = hair_bsdf_eval(V, lightDir, tangent, mat, sigma_a, h, vStrand);
        radiance += max(bsdf * lightColor * atten, vec3(0.0));
    }

    // Exposure
    radiance *= cam.exposure_factor;

    // Payload
    payload.radiance    = radiance;
    payload.attenuation = vec3(0.0);
    payload.scattered   = false;
    payload.hitEmissive = false;
    payload.occluded    = 0u;
}
