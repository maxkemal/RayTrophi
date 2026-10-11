#ifndef WEATHER_SURFACE_GLSL
#define WEATHER_SURFACE_GLSL
#include "weather_render_data.glsl"
#ifndef WEATHER_SURFACE_AVAILABLE
#define WEATHER_SURFACE_AVAILABLE true
#endif
vec3 weatherSafeNormalize(vec3 v, vec3 fallback) {
    return dot(v, v) > 1e-8 ? normalize(v) : fallback;
}
bool weatherSurfaceActive() {
    if (!WEATHER_SURFACE_AVAILABLE) return false;
    if (WEATHER_DATA.state.x == 0 || WEATHER_DATA.state.y == 0 ||
        WEATHER_DATA.flags.x == 0) {
        return false;
    }

    float surfaceSignal = 0.0;
    if (WEATHER_DATA.state.y == 1) {
        surfaceSignal = WEATHER_DATA.surface.x;
    } else if (WEATHER_DATA.state.y == 2 || WEATHER_DATA.state.y == 3) {
        surfaceSignal = WEATHER_DATA.surface.y;
    }

    return surfaceSignal > 0.001 || (WEATHER_DATA.state.z > 0.0 && WEATHER_DATA.state.w > 0.0);
}

float weatherSurfaceGeometricSupport(vec3 supportNormal) {
    vec3 macroNormal = weatherSafeNormalize(supportNormal, vec3(0.0, 1.0, 0.0));
    float support = 0.0;
    if (WEATHER_DATA.state.y == 2) {
        support = clamp((macroNormal.y - 0.02) / 0.72, 0.0, 1.0);
        support *= support;
    } else {
        support = clamp((macroNormal.y - 0.02) / 0.78, 0.0, 1.0);
    }
    return support * support * (3.0 - 2.0 * support);
}

float weatherSurfaceExposure(vec3 hitPos, vec3 normal) {
    float upMask = smoothstep(0.12, 0.90, normal.y);
    float windAmount = clamp(WEATHER_DATA.wind.w / 35.0, 0.0, 1.0);
    vec3 windRaw = WEATHER_DATA.wind.xyz;
    vec3 windDir = (dot(windRaw, windRaw) > 1e-8) ? normalize(windRaw) : vec3(1.0, 0.0, 0.0);
    vec3 incoming = normalize(vec3(0.0, 1.0, 0.0) - windDir * windAmount);
    float windFacing = clamp(dot(weatherSafeNormalize(normal, vec3(0.0, 1.0, 0.0)), incoming), 0.0, 1.0);
    float exposure = clamp(upMask * (1.0 - windAmount * 0.78) + windFacing * (0.12 + windAmount * 1.22), 0.0, 1.0);
    float scale = max(WEATHER_DATA.visual.x, 0.1);
    float n = pd_vnoise3(vec3(hitPos.xz * scale * 0.18, 13.1));
    float breakup = clamp(n * 1.35 - 0.18, 0.0, 1.0);
    return exposure * mix(0.45, 1.0, breakup);
}

float weatherSurfaceSettling(vec3 hitPos, vec3 normal, vec3 supportNormal);

float weatherSurfaceAccumulation(vec3 hitPos, vec3 normal, vec3 supportNormal) {
    if (WEATHER_DATA.state.y != 2 && WEATHER_DATA.state.y != 3) {
        return 0.0;
    }

    float baseAccum = clamp(WEATHER_DATA.surface.y, 0.0, 1.0);
    float intensity = clamp(WEATHER_DATA.state.z, 0.0, 1.0);
    float density = clamp(WEATHER_DATA.state.w, 0.0, 1.0);
    float geomSupport = weatherSurfaceGeometricSupport(supportNormal);
    float intensityResponse = 0.80 + intensity * 0.70;
    float densityResponse = 0.35 + density * 1.15;
    float typeBoost = (WEATHER_DATA.state.y == 2) ? 1.10 : 0.90;
    float directAccum = baseAccum * intensityResponse * weatherSurfaceExposure(hitPos, normal) * densityResponse * typeBoost * geomSupport;
    float settling = weatherSurfaceSettling(hitPos, normal, supportNormal);
    return clamp(directAccum + (1.0 - clamp(directAccum, 0.0, 1.0)) * settling, 0.0, 1.0);
}

float weatherSurfaceSettling(vec3 hitPos, vec3 normal, vec3 supportNormal) {
    if (WEATHER_DATA.state.y != 2 && WEATHER_DATA.state.y != 3) {
        return 0.0;
    }

    float settlingAmount = clamp(WEATHER_DATA.surface.z, 0.0, 1.0);
    if (settlingAmount <= 1e-4) {
        return 0.0;
    }

    vec3 shadingNormal = weatherSafeNormalize(normal, vec3(0.0, 1.0, 0.0));
    vec3 macroNormal = weatherSafeNormalize(supportNormal, shadingNormal);
    float support = weatherSurfaceGeometricSupport(supportNormal);
    float supportGate = clamp((support - 0.02) / 0.58, 0.0, 1.0);
    if (supportGate <= 1e-4) {
        return 0.0;
    }
    float exposure = weatherSurfaceExposure(hitPos, shadingNormal);
    float cavity = clamp((1.0 - dot(shadingNormal, macroNormal)) * 3.8 + (1.0 - support) * 0.10, 0.0, 1.0);
    vec3 windFlat = vec3(WEATHER_DATA.wind.xyz.x, 0.0, WEATHER_DATA.wind.xyz.z);
    vec3 leeDir = dot(windFlat, windFlat) > 1e-8 ? weatherSafeNormalize(vec3(-windFlat.x, 0.28, -windFlat.z), vec3(0.0, 1.0, 0.0)) : vec3(0.0, 1.0, 0.0);
    float lee = clamp(dot(macroNormal, leeDir) * 0.85 + cavity * 0.35, 0.0, 1.0);
    float shelter = clamp((1.0 - exposure) * 0.52 + cavity * 0.26 + (1.0 - support) * 0.22 + lee * 0.42, 0.0, 1.0);
    float pocketNoise = pd_vnoise3(hitPos * 0.085 + vec3(31.4, 9.7, 54.2));
    float pocketMask = clamp(cavity * 0.92 + pocketNoise * 0.26, 0.0, 1.0);
    float slopeBase = clamp((support - 0.16) / 0.54, 0.0, 1.0);
    float density = clamp(WEATHER_DATA.state.w, 0.0, 1.0);
    float typeBoost = (WEATHER_DATA.state.y == 2) ? 1.34 : 1.04;
    float anchor = max(pocketMask, slopeBase * 0.30 + cavity * 0.40 + lee * 0.30);
    return clamp(settlingAmount * supportGate * shelter * anchor * (0.76 + density * 1.10) * typeBoost, 0.0, 1.0);
}

float weatherSurfaceHeight(vec3 hitPos) {
    float scale = max(WEATHER_DATA.visual.x, 0.1);
    float heightBoost = 0.25 + clamp(WEATHER_DATA.surface.w, 0.0, 1.0) * 3.75;
    if (WEATHER_DATA.state.y == 2) {
        vec2 windXZ = WEATHER_DATA.wind.xyz.xz;
        float windLen = length(windXZ);
        vec2 along = windLen > 1e-4 ? windXZ / windLen : vec2(1.0, 0.0);
        vec2 across = vec2(-along.y, along.x);
        vec2 uv = vec2(dot(hitPos.xz, along), dot(hitPos.xz, across));
        vec3 p = vec3(uv.x * scale * 0.12, hitPos.y * scale * 0.03, uv.y * scale * 0.12);
        float broad = pd_vnoise3(p * 0.55 + vec3(17.3, 9.1, 41.7));
        float drift = 1.0 - abs(pd_vnoise3(vec3(p.x * 1.45, p.y * 0.8, p.z * 0.58) + vec3(3.7, 29.4, 11.8)) * 2.0 - 1.0);
        drift *= drift;
        float clumps = 1.0 - abs(pd_vnoise3(p * 2.90 + vec3(61.2, 7.5, 18.9)) * 2.0 - 1.0);
        float micro = pd_vnoise3(p * 7.40 + vec3(8.3, 51.7, 27.4));
        return (broad * 0.22 + drift * 0.36 + clumps * 0.27 + micro * 0.15) * heightBoost;
    }

    vec3 p = hitPos * (scale * 0.18);
    float wisps = pd_vnoise3(p + vec3(19.7, 5.3, 27.1));
    float grain = pd_vnoise3(p * 2.85 + vec3(4.1, 37.8, 12.4));
    float streak = pd_vnoise3(p * 1.65 + vec3(44.5, 14.2, 7.6));
    return (wisps * 0.30 + grain * 0.45 + streak * 0.25) * heightBoost;
}

vec3 weatherSurfaceNormal(vec3 hitPos, vec3 normal, vec3 supportNormal) {
    vec3 baseNormal = weatherSafeNormalize(normal, vec3(0.0, 1.0, 0.0));
    if (!weatherSurfaceActive()) return baseNormal;
    if (WEATHER_DATA.state.y != 2 && WEATHER_DATA.state.y != 3) return baseNormal;

    float accumulation = weatherSurfaceAccumulation(hitPos, baseNormal, supportNormal);
    if (accumulation <= 1e-4) return baseNormal;

    float geomSupport = weatherSurfaceGeometricSupport(supportNormal);
    if (geomSupport <= 1e-4) return baseNormal;

    float settling = weatherSurfaceSettling(hitPos, baseNormal, supportNormal);
    float detailCapture = 0.45 + 0.55 * clamp((baseNormal.y - 0.04) / 0.82, 0.0, 1.0);
    float heightResponse = 0.12 + clamp(WEATHER_DATA.surface.w, 0.0, 1.0) * 0.95;
    float buildup = clamp(accumulation + settling * 0.85, 0.0, 1.0);
    float normalStrength = buildup * detailCapture * heightResponse * (WEATHER_DATA.state.y == 2 ? 0.42 : 0.15);
    if (normalStrength <= 1e-4) return baseNormal;

    vec3 wind = WEATHER_DATA.wind.xyz;
    vec3 tangent = wind - baseNormal * dot(wind, baseNormal);
    if (dot(tangent, tangent) <= 1e-8) {
        vec3 helper = abs(baseNormal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        tangent = cross(helper, baseNormal);
    }
    tangent = weatherSafeNormalize(tangent, vec3(1.0, 0.0, 0.0));
    vec3 bitangent = weatherSafeNormalize(cross(baseNormal, tangent), vec3(0.0, 0.0, 1.0));
    tangent = weatherSafeNormalize(cross(bitangent, baseNormal), tangent);

    float sampleStep = (WEATHER_DATA.state.y == 2 ? 0.62 : 0.90) / max(WEATHER_DATA.visual.x, 0.35);
    float heightCenter = weatherSurfaceHeight(hitPos);
    float heightT = weatherSurfaceHeight(hitPos + tangent * sampleStep);
    float heightB = weatherSurfaceHeight(hitPos + bitangent * sampleStep);
    float gradT = clamp((heightT - heightCenter) / sampleStep, -0.28, 0.28);
    float gradB = clamp((heightB - heightCenter) / sampleStep, -0.28, 0.28);

    vec3 perturbed = weatherSafeNormalize(baseNormal - tangent * (gradT * normalStrength) - bitangent * (gradB * normalStrength), baseNormal);
    if (dot(perturbed, baseNormal) < 0.05) {
        perturbed = weatherSafeNormalize(mix(baseNormal, perturbed, 0.35), baseNormal);
    }
    if (dot(perturbed, supportNormal) < 0.55) {
        perturbed = weatherSafeNormalize(mix(supportNormal, perturbed, 0.05), supportNormal);
    }
    return perturbed;
}

void applyWeatherSurface(vec3 hitPos, vec3 normal, vec3 supportNormal, inout vec3 albedo, inout float roughness, inout float metallic) {
    if (!weatherSurfaceActive()) return;

    float exposed = weatherSurfaceExposure(hitPos, normal);

    if (WEATHER_DATA.state.y == 1) {
        float wet = clamp(WEATHER_DATA.surface.x, 0.0, 1.0) *
                    mix(0.35, 1.0, exposed);
        albedo = mix(albedo, albedo * 0.50, wet * 0.62);
        roughness = max(0.012, roughness * (1.0 - wet * 0.78));
        metallic = max(0.0, metallic - wet * 0.05);
    } else if (WEATHER_DATA.state.y == 2) {
        float acc = weatherSurfaceAccumulation(hitPos, normal, supportNormal);
        float settling = weatherSurfaceSettling(hitPos, normal, supportNormal);
        float heightLift = clamp(WEATHER_DATA.surface.w, 0.0, 1.0);
        float cover = clamp(acc + settling * 0.84 + heightLift * (acc * 0.08 + settling * 0.30), 0.0, 1.0);
        albedo = mix(albedo, vec3(0.88, 0.91, 0.96), cover * 0.72);
        roughness = min(1.0, roughness + cover * (0.45 + heightLift * 0.10));
        metallic *= (1.0 - cover * 0.8);
    } else if (WEATHER_DATA.state.y == 3) {
        float acc = weatherSurfaceAccumulation(hitPos, normal, supportNormal);
        float settling = weatherSurfaceSettling(hitPos, normal, supportNormal);
        float heightLift = clamp(WEATHER_DATA.surface.w, 0.0, 1.0);
        float cover = clamp(acc + settling * 0.90 + heightLift * settling * 0.22, 0.0, 1.0);
        albedo = mix(albedo, vec3(0.58, 0.46, 0.30), cover * 0.55);
        roughness = min(1.0, roughness + cover * (0.35 + heightLift * 0.08));
        metallic *= (1.0 - cover * 0.55);
    }
}




#endif
