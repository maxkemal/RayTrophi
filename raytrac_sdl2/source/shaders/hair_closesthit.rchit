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
#extension GL_EXT_nonuniform_qualifier : require

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
    vec3  primaryAlbedo;
    vec3  primaryNormal;
    uint  primaryHit;
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

// ─── Binding 7: World Data (Atmosphere/Sky) ──────────────────────────────────
struct VkWorldDataExtended {
    vec3  sunDir;
    int   mode;
    vec3  sunColor;
    float sunIntensity;
    float sunSize;
    float mieAnisotropy;
    float rayleighDensity;
    float mieDensity;
    float humidity;
    float temperature;
    float ozoneAbsorptionScale;
    float _pad0;
    float airDensity;
    float dustDensity;
    float ozoneDensity;
    float altitude;
    float planetRadius;
    float atmosphereHeight;
    float _pad1;
    float _pad2;
    int   cloudsEnabled;
    float cloudCoverage;
    float cloudDensity;
    float cloudScale;
    float cloudHeightMin;
    float cloudHeightMax;
    float cloudOffsetX;
    float cloudOffsetZ;
    float cloudQuality;
    float cloudDetail;
    int   cloudBaseSteps;
    int   cloudLightSteps;
    float cloudShadowStrength;
    float cloudAmbientStrength;
    float cloudSilverIntensity;
    float cloudAbsorption;
    float cloudAnisotropy;
    float cloudAnisotropyBack;
    float cloudLobeMix;
    float cloudEmissiveIntensity;
    vec3  cloudEmissiveColor;
    float _pad3;
    int   fogEnabled;
    float fogDensity;
    float fogHeight;
    float fogFalloff;
    float fogDistance;
    float fogSunScatter;
    vec3  fogColor;
    float _pad4;
    int   godRaysEnabled;
    float godRaysIntensity;
    float godRaysDensity;
    int   godRaysSamples;
    int   aerialEnabled;
    float aerialMinDistance;
    float aerialMaxDistance;
    float _pad5_aerial;
    int   envTexSlot;
    float envIntensity;
    float envRotation;
    int   _pad5;
};
layout(set = 0, binding = 7, scalar) readonly buffer WorldBuffer { VkWorldDataExtended w; } worldData;

// --- Binding 6: Material textures (Environment map) ---
layout(set = 0, binding = 6) uniform sampler2D materialTextures[];

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

uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand_float(uint seed) {
    return float(pcg_hash(seed)) / 4294967296.0;
}

vec3 rgb_to_hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv_to_rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// --- Robust Ray Offsetting (Parity with OptiX) ---
vec3 offset_ray(vec3 p, vec3 n) {
    const float origin = 1.0 / 32.0;
    const float float_scale = 1.0 / 65536.0;
    const float int_scale = 256.0;

    ivec3 of_i = ivec3(int_scale * n.x, int_scale * n.y, int_scale * n.z);
    vec3 p_i = vec3(
        intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z))
    );

    return vec3(
        abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
        abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
        abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z
    );
}

const float HAIR_SHADOW_TMIN = 1e-3;

// Simplified sky lookup for ambient matching
vec3 get_ambient_sky(vec3 dir) {
    if (worldData.w.mode == 0) return worldData.w.sunColor * worldData.w.envIntensity;
    if (worldData.w.mode == 1) {
        int envSlot = worldData.w.envTexSlot;
        if (envSlot > 0) {
            float u = 0.5 + atan(dir.z, dir.x) / (2.0 * PI);
            float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
            return texture(materialTextures[nonuniformEXT(envSlot)], vec2(u, v)).rgb * worldData.w.envIntensity;
        }
    }
    // Mode 2: Nishita (simplified directional gradient for ambient)
    float cosAlt = dir.y;
    vec3 horizon = vec3(0.7, 0.85, 1.0);
    vec3 zenith = vec3(0.1, 0.3, 0.9);
    vec3 sky = mix(horizon, zenith, pow(clamp(cosAlt, 0.0, 1.0), 0.6)) * (worldData.w.sunIntensity / 10.0);
    if (cosAlt < 0.0) sky = mix(sky, vec3(0.02, 0.015, 0.01), smoothstep(0.0, -0.15, cosAlt));
    return sky;
}

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
    float baseRadialR = max(mat.radialRoughness, 0.08);
    float v_R   = sqr(baseR * 1.2);
    float v_TT  = sqr(baseR);
    float v_TRT = sqr(baseR);
    float s_R   = baseRadialR * 1.8 * 0.5;
    float s_TT  = baseRadialR * 0.7 * 0.5;
    float s_TRT = baseRadialR * 2.2 * 0.5;
    float s_MS  = max(baseRadialR * 0.7 * 10.0, 0.2) * 0.5;

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
    uint segCount = hairSegs.length();
    if (segIdx >= segCount) {
        payload.radiance      = vec3(0.0);
        payload.attenuation   = vec3(0.0);
        payload.scatterOrigin = hitPoint;
        payload.scatterDir    = normalize(gl_WorldRayDirectionEXT);
        payload.scattered     = false;
        payload.hitEmissive   = false;
        payload.occluded      = 0u;
        payload.skipAABBs     = false;
        return;
    }
    HairSegmentGPU seg = hairSegs[segIdx];
    uint matID  = seg.materialID;
    uint matCount = hairMats.length();
    if (matCount == 0u) {
        payload.radiance      = vec3(0.0);
        payload.attenuation   = vec3(0.0);
        payload.scatterOrigin = hitPoint;
        payload.scatterDir    = normalize(gl_WorldRayDirectionEXT);
        payload.scattered     = false;
        payload.hitEmissive   = false;
        payload.occluded      = 0u;
        payload.skipAABBs     = false;
        return;
    }
    if (matID >= matCount) matID = 0u;
    HairGpuMaterial mat = hairMats[matID];

    // ─── Per-strand Randomization ─────────────────────────────────────────
    uint strandID = seg.strandID;
    float r1 = rand_float(strandID * 1973u);
    float r2 = rand_float(strandID * 9277u + 83492791u);

    // ─── Compute sigma_a based on color mode ─────────────────────────────
    vec3 sigma_a;
    vec3 hairColor;
    if (mat.colorMode == 1u) {
        // Melanin (physical)
        float m = clamp(mat.melanin + (r1 - 0.5) * mat.randomValue, 0.0, 1.0);
        float red = clamp(mat.melaninRedness + (r2 - 0.5) * mat.randomHue, 0.0, 1.0);
        sigma_a = melanin_to_absorption(m, red);
        hairColor = exp(-sigma_a * 0.5);
    } else if (mat.colorMode == 2u) {
        // Explicit absorption
        sigma_a = mat.absorption;
        hairColor = exp(-sigma_a * 0.5);
    } else {
        // Direct color (0) or Root UV Map (3) — derive sigma from color
        hairColor = max(mat.baseColor, vec3(0.001));
        // Apply HSV variation
        vec3 hsv = rgb_to_hsv(hairColor);
        hsv.x = fract(hsv.x + (r1 - 0.5) * mat.randomHue);
        hsv.z = clamp(hsv.z + (r2 - 0.5) * mat.randomValue, 0.0, 2.0);
        hairColor = hsv_to_rgb(hsv);

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

    // ─── Ambient (Parity with OptiX) ─────────────────────────────────────
    vec3 skyDir_amb = normalize(normal + vec3(0.0, 1.0, 0.0));
    vec3 skyColor = get_ambient_sky(skyDir_amb);
    vec3 ambient_bsdf = hairColor * (0.318309886); // 1.0 / PI
    vec3 radiance = ambient_bsdf * skyColor * 0.15;
    radiance += 0.01 * hairColor; 

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
            lightDir  = normalize(light.direction.xyz);
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
        vec3 shadowOrig = offset_ray(hitPoint, normal); 
        shadowOccluded = true;
        traceRayEXT(
            topLevelAS,
            gl_RayFlagsTerminateOnFirstHitEXT |
            gl_RayFlagsSkipClosestHitShaderEXT |
            gl_RayFlagsOpaqueEXT,
            // mask 0x01 = triangles only, skip volume AABBs (mask 0x02)
            0x01, 0, 0, 1,
            shadowOrig, HAIR_SHADOW_TMIN, lightDir, lightDist - 0.002,
            1
        );
        if (shadowOccluded) continue;

        // Marschner BSDF evaluation
        vec3 bsdf = hair_bsdf_eval(V, lightDir, tangent, mat, sigma_a, h, vStrand);
        radiance += max(bsdf * lightColor * atten, vec3(0.0));
    }

    // Exposure
    radiance *= cam.exposure_factor;

    if (payload.primaryHit == 0u) {
        payload.primaryAlbedo = hairColor;
        payload.primaryNormal = normal;
        payload.primaryHit = 1u;
    }

    // Payload
    payload.radiance    = radiance;
    payload.attenuation = vec3(0.0);
    payload.scatterOrigin = offset_ray(hitPoint, normal);
    payload.scatterDir    = normalize(gl_WorldRayDirectionEXT);
    payload.scattered   = false;
    payload.hitEmissive = false;
    payload.occluded    = 0u;
    payload.skipAABBs   = false;
}
