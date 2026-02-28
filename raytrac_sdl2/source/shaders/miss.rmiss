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
    
    // ════════════════════════════ FOG PARAMETERS (32 bytes)
    int   fogEnabled;
    float fogDensity;
    float fogHeight;
    float fogFalloff;
    float fogDistance;
    float fogSunScatter;
    vec3  fogColor;
    
    // ════════════════════════════ GOD RAYS (16 bytes)
    int   godRaysEnabled;
    float godRaysIntensity;
    float godRaysDensity;
    int   godRaysSamples;
    
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
    vec3  sunDir     = normalize(worldData.w.sunDir);
    int   mode       = worldData.w.mode;
    float cosAlt     = dir.y;

    // Yer altı — koyu, düz renk
    // Ghost önleme: sabit, düşük değer → Welford hızlı converge eder
    if (cosAlt < 0.0) {
        // Expand horizon blending region to reduce temporal aliasing when camera moves
        float t = clamp((-cosAlt) / 0.5, 0.0, 1.0); // maps cosAlt=0->0, cosAlt=-0.5->1
        float blend = smoothstep(0.0, 1.0, t);
        vec3  ground  = vec3(0.05, 0.04, 0.03);
        vec3  horiCol = rayleighColor(0.0);
        return mix(horiCol, ground, blend);
    }

    // Gökyüzü (Rayleigh + atmosferik sis)
    vec3  sky       = rayleighColor(cosAlt);
    float fogAmt    = atmosphereDensity(cosAlt) * 0.4;
    vec3  fogColor  = vec3(0.8, 0.88, 1.0);
    sky             = mix(sky, fogColor, fogAmt);

    // Debug: force sampling the precomputed skyview LUT (temporary)
    // If a precomputed skyview LUT is available, prefer sampling it for sky color
    if (worldData.w.skyviewLUT.x != 0u) {
        float az = atan(dir.z, dir.x);
        float u_sky = az / TWO_PI;
        if (u_sky < 0.0) u_sky += 1.0;
        float v_sky = (1.0 - clamp(dir.y, -1.0, 1.0)) * 0.5; // match CPU precompute mapping
        vec3 lutSky = texture(atmosphereLUTs[1], vec2(u_sky, v_sky)).rgb;
        sky = lutSky;
    }

    // World modes: 0=color, 1=hdri, 2=nishita (procedural)
    // Use world sun color/intensity when adding sun contribution
    // Sun disk size and halo controlled by world parameters
    float sunSizeDeg = worldData.w.sunSize;
    float sunSizeRad = radians(max(sunSizeDeg, 0.0001));
    float diskEdge = cos(sunSizeRad);
    float sun       = sunContribution(dir, sunDir);
    vec3  sunCol    = worldData.w.sunColor;
    // Halo strength influenced by dust (mie) density
    float haloScale = 0.3 * (1.0 + worldData.w.dustDensity);
    float disk     = smoothstep(diskEdge - 0.0005, diskEdge, dot(dir, sunDir));
    float halo     = pow(max(dot(dir, sunDir), 0.0), 8.0) * haloScale;
    sky            += (disk + halo) * sunCol * worldData.w.sunIntensity;

    // If an environment texture was uploaded, sample it and blend
    int envSlot = worldData.w.envTexSlot;
    if (mode == 1 && envSlot > 0) {
        // Convert direction to lat-long UV (equirectangular)
            float u = 0.5 + atan(dir.z, dir.x) / TWO_PI;
        float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
        vec3 envCol = texture(materialTextures[nonuniformEXT(envSlot)], vec2(u, v)).rgb;
        sky = mix(sky, envCol * worldData.w.envIntensity, clamp(worldData.w.envIntensity, 0.0, 1.0));
    }

    // If Color mode, override sky with a flat color scaled by envIntensity
    if (mode == 0) {
        sky = worldData.w.sunColor * worldData.w.envIntensity;
    }
        // Apply simple atmosphere density tint (Rayleigh-like)
        float air = clamp(worldData.w.airDensity, 0.0, 100.0);
        float dust = clamp(worldData.w.dustDensity, 0.0, 100.0);
        float atm = clamp(air * 0.02 + dust * 0.01, 0.0, 1.0);
        sky = mix(sky, vec3(0.8,0.88,1.0), atm);

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
