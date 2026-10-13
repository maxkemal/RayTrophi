// Ray-traced sky: the world/atmosphere bindings and the sky radiance of the
// miss shader. Shared by miss.rmiss and cloud_callable.rcall (the clouds of
// the miss path are lit by this sky without the sun disk). Needs
// GL_EXT_scalar_block_layout and GL_EXT_nonuniform_qualifier.
#ifndef RT_SKY_GLSL
#define RT_SKY_GLSL

#include "sky_sun_corona.glsl"

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
    float atmosphereIntensity;
    
    // ════════════════════════════ ATMOSPHERE DENSITY (32 bytes)
    float airDensity;
    float dustDensity;
    float ozoneDensity;
    float altitude;
    float planetRadius;
    float atmosphereHeight;
    int   multiScatterEnabled;
    float multiScatterFactor;
    
    
    // ════════════════════════════ FOG PARAMETERS (40 bytes)
    int   fogEnabled;
    float fogDensity;
    float fogHeight;
    float fogFalloff;
    float fogDistance;
    float fogAnisotropy;
    vec3  fogAlbedo;
    float _pad4;   // padding — aligns fogAlbedo[3]+pad to 16 bytes, matches C++ struct
    
    // ════════════════════════════ GOD RAYS (16 bytes)
    int   godRaysEnabled;
    float godRaysIntensity;
    float godRaysDensity;
    int   godRaysSamples;
    
    // ════════════════════════════ AERIAL PERSPECTIVE (16 bytes) — matches OptiX AtmosphereAdvanced
    int   aerialFroxelReady;    // 1 = raygen applies the aerial froxel (binding 8 slot 3)
    float _aerialPad0;
    float _aerialPad1;
    float _aerialPad2;

    // Weather payload (passive until weather rendering is enabled)
    int   weatherEnabled;
    int   weatherType;
    float weatherIntensity;
    float weatherDensity;
    vec3  weatherWindDirection;
    float weatherWindSpeed;
    float weatherPrecipitationScale;
    float weatherVisibility;
    float weatherSurfaceWetness;
    float weatherSurfaceAccumulation;
    float weatherSurfaceSettling;
    float weatherSurfaceHeight;
    int   weatherVisualMode;
    int   weatherSurfaceResponseEnabled;
    
    // ════════════════════════════ ENVIRONMENT & LUT REFS (32 bytes)
    int   envTexSlot;
    float envIntensity;
    float envRotation;
    int   _pad5;                 // nishitaLutReady: Vulkan binding 8 has valid LUT samplers
    int   envOverlayEnabled;
    int   envOverlayBlendMode;
    float envOverlayIntensity;
    float envOverlayRotation;
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

// Fast weather is composed once on the camera segment in raygen.
vec3 blendEnvironmentOverlay(vec3 base, vec3 sampled, float intensity, int blendMode) {
    float strength = max(intensity, 0.0);
    float amount = min(strength, 1.0);
    vec3 overlay = sampled * strength;

    if (blendMode == 1) {
        return base * mix(vec3(1.0), sampled, amount);
    }
    if (blendMode == 2) {
        return base + overlay;
    }
    if (blendMode == 3) {
        return overlay;
    }
    return mix(base, overlay, amount);
}

vec3 sampleEnvironmentLatLong(int envSlot, vec3 dir, float rotationRad) {
    float u = 0.5 + atan(dir.z, dir.x) / TWO_PI;
    u = fract(u - rotationRad / TWO_PI);
    float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
    return texture(materialTextures[nonuniformEXT(envSlot)], vec2(u, v)).rgb;
}

// withSun = false: sky light only (cloud lighting gets the sun through NEE).
vec3 skyColorImpl(vec3 dir, bool withSun, vec3 origin) {
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
            return sampleEnvironmentLatLong(envSlot, dir, worldData.w.envRotation)
                 * worldData.w.envIntensity;
        }
        return vec3(0.0); // No HDRI loaded → black
    }

    // ─── Mode 2: Nishita Spectral Sky ────────────────────────────────────────
    vec3 sunDir = normalize(worldData.w.sunDir);

    vec3 sky;

    bool hasAtmosLUTs = worldData.w._pad5 != 0;
    if (hasAtmosLUTs) {
        // SkyView LUT — UV matches AtmosphereLUT::sampleSkyView exactly:
        //   u = azimuth / 2π    (atan2(z, x), wrapped to [0,1])
        //   v = (1 - dir.y) * 0.5
        float az    = atan(dir.z, dir.x);
        float u_sky = az / TWO_PI;
        if (u_sky < 0.0) u_sky += 1.0;
        float v_sky = (1.0 - clamp(dir.y, -1.0, 1.0)) * 0.5;
        sky = texture(atmosphereLUTs[1], vec2(u_sky, v_sky)).rgb;
    } else {
        // LUT not yet uploaded — simple Rayleigh fallback with smooth horizon transition.
        // No hard cutoff at cosAlt=0.
        float cosAltClamped = max(cosAlt, -0.15);
        float skyBright = clamp(worldData.w.atmosphereIntensity / 10.0, 0.0, 2.0);
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

    // Ray-origin altitude (matches CPU calculateNishitaSky):
    //   p = origin + (0, Rg, 0);  alt = max(0, length(p) - Rg)
    // Used by transmittance LUT v-coord so high cameras get proper attenuation.
    float Rg            = worldData.w.planetRadius;
    vec3  pOrigin       = origin + vec3(0.0, Rg, 0.0);
    float currentAlt    = max(0.0, length(pOrigin) - Rg);
    float v_trans       = clamp(currentAlt / max(1.0, worldData.w.atmosphereHeight), 0.0, 1.0);

    // ── Multi-scatter (analytic, matches World::calculateNishitaSky) ──────────
    // Gated by AtmosphereAdvanced::multi_scatter_enabled (UI checkbox).
    if (worldData.w.multiScatterEnabled != 0) {
        vec3  scatteringAlbedo = vec3(0.8, 0.85, 0.9);
        float mf               = worldData.w.multiScatterFactor;
        vec3  secondOrder      = sky * scatteringAlbedo * 0.5 * exp(-0.5 * 0.3);
        vec3  thirdOrder       = secondOrder * scatteringAlbedo * 0.25 * exp(-0.5 * 0.1);
        sky = sky + secondOrder * mf + thirdOrder * (mf * 0.5);
    }

    // Procedural sun disk — matches World::calculateNishitaSky exactly
    if (withSun && worldData.w.sunIntensity > 0.0) {
        float sunSizeDeg = worldData.w.sunSize;

        // Elevation broadening near horizon
        float elevDeg = degrees(asin(clamp(worldData.w.sunDir.y, -1.0, 1.0)));
        float elevFactor = 1.0;
        if (elevDeg < 15.0) {
            elevFactor = 1.0 + (15.0 - max(elevDeg, -10.0)) * 0.04;
        }
        sunSizeDeg *= elevFactor;

        float sunRadius  = radians(sunSizeDeg * 0.5);
        float cosThresh  = cos(sunRadius);
        float mu         = dot(dir, sunDir);  // also used by excessPhase below

        // ── Sun disk ─────────────────────────────────────────────────────────
        if (mu > cosThresh) {
            float angDist   = acos(min(1.0, mu));
            float radialPos = angDist / sunRadius;

            // Limb darkening (u = 0.6)
            float cosine_mu     = sqrt(max(0.0, 1.0 - radialPos * radialPos));
            float limbDarkening = 1.0 - 0.6 * (1.0 - cosine_mu);

            // Cubic smoothstep edge
            float edge_t    = clamp((radialPos - 0.85) / 0.15, 0.0, 1.0);
            float edgeSoft  = 1.0 - edge_t * edge_t * (3.0 - 2.0 * edge_t);

            vec3 transSun = vec3(1.0);
            if (hasAtmosLUTs) {
                float cosSun  = max(0.01, worldData.w.sunDir.y);
                float u_trans = (cosSun + 0.2) / 1.2;
                transSun = texture(atmosphereLUTs[0], vec2(u_trans, v_trans)).rgb;
            }

            // CPU uses 80000.0; keep until sun-multiplier unification (PR-3).
            sky += transSun * worldData.w.sunIntensity * 80000.0
                 * limbDarkening * edgeSoft;
        }

        // ── Excess-phase corona (sky_sun_corona.glsl, shared with RayFusion) ──
        if (hasAtmosLUTs) {
            float cosSun  = max(0.01, worldData.w.sunDir.y);
            float u_trans = (cosSun + 0.2) / 1.2;
            vec3  transSun = texture(atmosphereLUTs[0], vec2(u_trans, v_trans)).rgb;
            sky += skyMieCorona(mu, worldData.w.mieAnisotropy, worldData.w.mieDensity,
                                worldData.w.atmosphereIntensity, transSun);
        }

        // NOTE: No air/dust post-tint here when LUT is loaded — LUT already has
        // air_density baked in from AtmosphereLUT.cpp precompute.
    }

    if (worldData.w.envOverlayEnabled != 0 && worldData.w.envTexSlot > 0) {
        vec3 overlay = sampleEnvironmentLatLong(worldData.w.envTexSlot, dir, worldData.w.envOverlayRotation);
        sky = blendEnvironmentOverlay(
            sky,
            overlay,
            worldData.w.envOverlayIntensity,
            worldData.w.envOverlayBlendMode);
    }

    return sky;
}

#endif // RT_SKY_GLSL
