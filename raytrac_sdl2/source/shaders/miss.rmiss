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
// Payload — raygen v3 + closesthit v2 ile tam eşleşme
// Alan sırası ve tipleri birebir aynı olmalı
// ============================================================
struct RayPayload {
    vec3  radiance;
    vec3  attenuation;
    vec3  scatterOrigin;
    vec3  scatterDir;
    uint  seed;
    bool  scattered;
    bool  hitEmissive;
    uint  occluded;
    bool  skipAABBs;
    vec3  primaryAlbedo;
    vec3  primaryNormal;
    uint  primaryHit;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

// Array of combined image samplers for uploaded textures (match other shaders)
layout(set = 0, binding = 6) uniform sampler2D materialTextures[];

// ════════════════════════════════════════════════════════════════════════════════
// EXTENDED WORLD DATA — Full Nishita Sky Model + Atmosphere LUT
// ════════════════════════════════════════════════════════════════════════════════
struct VkWorldDataExtended {
    // ════════════════════════════ CORE MODE & SUN TINT (32 bytes)
    vec3  sunDir;
    int   mode;
    vec3  sunColor;
    float sunIntensity;
    
    // ════════════════════════════ NISHITA SUN PARAMETERS (32 bytes)
    float sunSize;
    float mieAnisotropy;
    float rayleighDensity;
    float mieDensity;
    float humidity;
    float temperature;
    float ozoneAbsorptionScale;
    float _pad0;
    
    // ════════════════════════════ ATMOSPHERE DENSITY (32 bytes)
    float airDensity;
    float dustDensity;
    float ozoneDensity;
    float altitude;
    float planetRadius;
    float atmosphereHeight;
    float _pad1;
    float _pad2;
    
    // ════════════════════════════ CLOUD LAYER 1 PARAMETERS (64 bytes)
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
    
    // ════════════════════════════ ADVANCED CLOUD SCATTERING (32 bytes)
    float cloudAnisotropy;
    float cloudAnisotropyBack;
    float cloudLobeMix;
    float cloudEmissiveIntensity;
    vec3  cloudEmissiveColor;
    float _pad3;
    
    // ════════════════════════════ FOG PARAMETERS (40 bytes)
    int   fogEnabled;
    float fogDensity;
    float fogHeight;
    float fogFalloff;
    float fogDistance;
    float fogSunScatter;
    vec3  fogColor;
    float _pad4;   // padding — aligns fogColor[3]+pad to 16 bytes, matches C++ struct
    
    // ════════════════════════════ GOD RAYS (16 bytes)
    int   godRaysEnabled;
    float godRaysIntensity;
    float godRaysDensity;
    int   godRaysSamples;
    
    // ════════════════════════════ AERIAL PERSPECTIVE (16 bytes) — matches OptiX AtmosphereAdvanced
    int   aerialEnabled;        // 1 = apply aerial perspective
    float aerialMinDistance;    // No haze below this (meters)
    float aerialMaxDistance;    // Full haze at this (meters)
    float _pad5_aerial;
    
    // ════════════════════════════ ENVIRONMENT & LUT REFS (32 bytes)
    int   envTexSlot;
    float envIntensity;
    float envRotation;
    int   _pad5;
    uvec2 transmittanceLUT;      // 64-bit handle as uvec2
    uvec2 skyviewLUT;            // 64-bit handle as uvec2
    uvec2 multiScatterLUT;       // 64-bit handle as uvec2
    uvec2 aerialPerspectiveLUT;  // 64-bit handle as uvec2
};

layout(set = 0, binding = 7, scalar) readonly buffer WorldBuffer { VkWorldDataExtended w; } worldData;
// Atmosphere LUT samplers: [0]=transmittance, [1]=skyview, [2]=multi_scatter, [3]=aerial_perspective
layout(set = 0, binding = 8) uniform sampler2D atmosphereLUTs[4];

// ============================================================
// Sabitler
// ============================================================
const float PI     = 3.14159265358979323846;
const float TWO_PI = 6.28318530717958647692;

// ============================================================
// Sky Yardımcı Fonksiyonlar
// ============================================================

vec3 rayleighColor(float cosAlt) {
    float t       = clamp(cosAlt, 0.0, 1.0);
    vec3  zenith  = vec3(0.1,  0.3,  0.9);   // derin mavi
    vec3  horizon = vec3(0.7,  0.85, 1.0);   // açık beyazımsı mavi
    return mix(horizon, zenith, pow(t, 0.6));
}

float atmosphereDensity(float cosAlt) {
    return exp(-max(cosAlt, 0.0) * 3.5);
}

float sunContribution(vec3 rayDir, vec3 sunDir) {
    float cosAngle = dot(rayDir, sunDir);
    float diskEdge = 0.9998;
    float disk     = smoothstep(diskEdge - 0.0005, diskEdge, cosAngle);
    float halo     = pow(max(cosAngle, 0.0), 8.0) * 0.3;
    return disk + halo;
}

vec3 skyColor(vec3 dir) {
    int   mode   = worldData.w.mode;
    float cosAlt = dir.y;

    // ─── Mode 0: Solid Color ─────────────────────────────────────────────────
    if (mode == 0) {
        return worldData.w.sunColor * worldData.w.envIntensity;
    }

    // ─── Mode 1: HDRI Environment ────────────────────────────────────────────
    if (mode == 1) {
        int envSlot = worldData.w.envTexSlot;
        if (envSlot > 0) {
            float u = 0.5 + atan(dir.z, dir.x) / TWO_PI;
            float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
            return texture(materialTextures[nonuniformEXT(envSlot)], vec2(u, v)).rgb
                   * worldData.w.envIntensity;
        }
        return vec3(0.0); // No HDRI loaded → black
    }

    // ─── Mode 2: Nishita Spectral Sky ────────────────────────────────────────
    vec3 sunDir = normalize(worldData.w.sunDir);

    vec3 sky;

    bool hasAtmosLUTs = (worldData.w.transmittanceLUT.x | worldData.w.transmittanceLUT.y) != 0u;
    if (hasAtmosLUTs) {
        // Sample the full-sphere LUT (covers cosTheta -1..+1, i.e. below horizon too).
        // This lets the LUT provide natural warm/orange horizon glow just like OptiX ray-sphere
        // integration does — no early-return that would create a hard horizon edge.
        float az    = atan(dir.z, dir.x);
        float u_sky = az / TWO_PI;
        if (u_sky < 0.0) u_sky += 1.0;
        // v_sky: dir.y=+1(up)→0.0, dir.y=0(horizon)→0.5, dir.y=-1(down)→1.0
        float v_sky = (1.0 - clamp(dir.y, -1.0, 1.0)) * 0.5;
        sky = texture(atmosphereLUTs[1], vec2(u_sky, v_sky)).rgb;

        // Blend to dark ground color well below horizon so geometry doesn't show sky on underside.
        // Use a wide, smooth fade starting from slightly below horizon — matches OptiX behavior
        // where very steep downward rays get near-zero radiance from the atmosphere.
        if (cosAlt < 0.0) {
            // t=0 at horizon, t=1 at cosAlt=-0.15 (≈8.6° below)
            float t     = clamp(-cosAlt / 0.15, 0.0, 1.0);
            float blend = smoothstep(0.0, 1.0, t);
            vec3  ground = vec3(0.02, 0.015, 0.01);
            sky = mix(sky, ground, blend);
        }
    } else {
        // LUT not yet uploaded — simple Rayleigh fallback with smooth horizon transition.
        // No hard cutoff at cosAlt=0.
        float cosAltClamped = max(cosAlt, -0.15);
        float skyBright = clamp(worldData.w.sunIntensity / 10.0, 0.0, 2.0);
        sky = rayleighColor(max(cosAltClamped, 0.0)) * skyBright;
        // air/dustDensity UI range 0-10 → divide by 10 to normalize
        float airN  = clamp(worldData.w.airDensity  / 10.0, 0.0, 1.0);
        float dustN = clamp(worldData.w.dustDensity / 10.0, 0.0, 1.0);
        float atmFallback = clamp(airN * 0.8 + dustN * 0.4, 0.0, 1.0);
        sky = mix(sky, vec3(0.8, 0.88, 1.0) * skyBright,
                  atmosphereDensity(max(cosAltClamped, 0.0)) * 0.4 * atmFallback);
        if (cosAlt < 0.0) {
            float t     = clamp(-cosAlt / 0.15, 0.0, 1.0);
            float blend = smoothstep(0.0, 1.0, t);
            sky = mix(sky, vec3(0.02, 0.015, 0.01), blend);
        }
    }

    // Procedural sun disk — matches OptiX sky_model.cuh exactly
    // Only when sun is active
    if (worldData.w.sunIntensity > 0.0) {
        // ── Sun disk (OptiX-matched) ──────────────────────────────────────────
        // sunSize is stored as angular diameter (degrees).
        // OptiX uses * 0.5 to convert to radius before radians → match here.
        float sunSizeDeg = worldData.w.sunSize;

        // Elevation factor: matches OptiX elevation broadening near horizon
        float elevDeg = degrees(asin(clamp(worldData.w.sunDir.y, -1.0, 1.0)));
        float elevFactor = 1.0;
        if (elevDeg < 15.0) {
            elevFactor = 1.0 + (15.0 - max(elevDeg, -10.0)) * 0.04;
        }
        sunSizeDeg *= elevFactor;

        // Angular radius (NOT diameter) — key fix vs old code which used full diameter
        float sunRadius  = radians(sunSizeDeg * 0.5);
        float cosThresh  = cos(sunRadius);
        float sunCos     = dot(dir, sunDir);

        if (sunCos > cosThresh) {
            // Radial position on disk: 0 = center, 1 = edge  (matches OptiX)
            float angDist   = acos(min(1.0, sunCos));
            float radialPos = angDist / sunRadius;

            // Limb darkening (OptiX: u=0.6)
            float u_limb        = 0.6;
            float cosine_mu     = sqrt(max(0.0, 1.0 - radialPos * radialPos));
            float limbDarkening = 1.0 - u_limb * (1.0 - cosine_mu);

            // Smooth edge (OptiX: same cubic smoothstep)
            float edge_t    = clamp((radialPos - 0.85) / 0.15, 0.0, 1.0);
            float edgeSoft  = 1.0 - edge_t * edge_t * (3.0 - 2.0 * edge_t);

            // Brightness: OptiX uses * 1000.0f.
            // LUT has sun_intensity baked in; disk is an additive on top.
            sky += worldData.w.sunColor * worldData.w.sunIntensity * 1000.0
                   * limbDarkening * edgeSoft;
        }

        // ── Mie phase: full (uncapped) + excess glow ─────────────────────────
        // The LUT clamps phaseM at 2.0 during precompute (AtmosphereLUT.cpp).
        // The sharp forward-scatter peak above that cap is the "glow" halo around the sun.
        // CPU calculateNishitaSky uses: excessPhase = max(0, phaseM - 2.0)
        // and adds: transmittance * mie_scattering * mie_density * 0.15 * excessPhase * sun_intensity
        // This is what makes OptiX look softer/fluffier near the disk — we replicate it here.
        float g           = clamp(worldData.w.mieAnisotropy, 0.0, 0.99);
        float mu          = dot(dir, sunDir);
        float phaseM_full = (1.0 - g*g) / (4.0 * PI * pow(max(1.0 + g*g - 2.0*g*mu, 0.0001), 1.5));
        float excessPhase = max(0.0, phaseM_full - 2.0); // sharp peak removed from LUT
        float phaseM_cap  = min(phaseM_full, 2.0);       // capped version for background halo

        // ── Excess phase glow (the missing "fluffy" corona) ──────────────────
        if (excessPhase > 0.0 && hasAtmosLUTs) {
            // Transmittance LUT [binding 8, slot 0]
            // UV matches AtmosphereLUT::sampleTransmittance:
            //   u = (cosTheta + 0.2) / 1.2,  v = altitude / atmosphereHeight
            float cosSun  = max(0.01, worldData.w.sunDir.y);
            float u_trans = (cosSun + 0.2) / 1.2;
            float v_trans = clamp(worldData.w.altitude / max(1.0, worldData.w.atmosphereHeight), 0.0, 1.0);
            vec3  transSun = texture(atmosphereLUTs[0], vec2(u_trans, v_trans)).rgb;

            // mie_scattering default = vec3(3.996e-6) — physical constant, hardcoded
            // since it is not stored in VkWorldDataExtended.
            // mieScat = mie_scattering * mieDensity * 0.15  (matches CPU exactly)
            const float MIE_SCAT = 3.996e-6;
            vec3 mieScat = vec3(MIE_SCAT) * worldData.w.mieDensity * 0.15;
            sky += transSun * (mieScat * excessPhase * worldData.w.sunIntensity);
        }

        // ── Broad Mie background halo (capped, contributes to horizon glow) ──
        // dustDensity UI range 0-10.  At dust=1 → 0.015, at dust=10 → 0.15
        float mie_scale = clamp(worldData.w.dustDensity * 0.015, 0.0, 0.15);
        sky += worldData.w.sunColor * worldData.w.sunIntensity * phaseM_cap * mie_scale;

        // NOTE: No air/dust post-tint here when LUT is loaded — LUT already has
        // air_density baked in from AtmosphereLUT.cpp precompute.
    }

    return sky;
}

// ============================================================
// Main
// ============================================================
void main() {
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
    payload.hitEmissive   = false;
    payload.occluded      = 0u;

    // ----------------------------------------------------------
    // Sky radiance — raygen: radiance += throughput * payload.radiance
    // ----------------------------------------------------------
    payload.radiance = skyColor(dir);
}
