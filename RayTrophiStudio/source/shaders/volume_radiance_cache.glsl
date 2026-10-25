// Volume radiance cache: per-vertex source radiance for BOUNCE rays through gas.
// docs/dev/RT_GAS_SECONDARY_MARCH.md §13.
//
// The cloud renderer never marches a cloud for a secondary ray: it reads a
// cloud-aware sky panorama (CloudQuality::rt_secondary_full_march = false).
// This is the gas equivalent. Four extra channels of the light cache grid hold,
// per cell vertex, the sampled density and the source radiance J (in-scatter
// with an ISOTROPIC phase + emission) such that a march segment contributes
//     T * J * (1 - exp(-sigma_t * dt))
// exactly like the full march's `transmittance * (inscatter + emis) *
// one_minus_sampleT`. A bounce ray then crosses the box in cell-sized steps of
// trilinear lookups: no NanoVDB walk, no per-sample lighting, no shadow march.
//
// Approximations, all on bounce rays only (the camera segment never reads it):
//   * isotropic phase (bounce directions are diffuse; the march blends toward
//     isotropic with multi-scatter anyway);
//   * area lights and lights past index 4 are not in J (no cache channel);
//   * ambient shadow from two upward density taps, not the ambient march;
//   * no emitter-NEE term (fire lighting its own smoke) and no material program;
//     volumes with a program or a companion never use the cache.
// A miss is never an error: the caller runs the full march for that ray.
//
// Requires volume_light_cache.glsl + volume_light_cache_fill.glsl, cam, lights,
// sampleSkyAmbient, sampleTransmittanceLUT, volSampleLight, sampleTemperature,
// volumeEmissionAt, volumeEmissionSource, volumeHigherOrderSkyVisibility and
// clampVolumeRadiance.
#ifndef RAYTROPHI_VOLUME_RADIANCE_CACHE_GLSL
#define RAYTROPHI_VOLUME_RADIANCE_CACHE_GLSL

const uint VRC_CH_DENSITY = 6u;   // published LAST: it carries the fill claim
const uint VRC_CH_SOURCE_R = 7u;
const uint VRC_CH_SOURCE_G = 8u;
const uint VRC_CH_SOURCE_B = 9u;

// Lighting the source radiance depends on: sun, sky mode and the cached scene
// lights. Hashed once per invocation (private global).
uint vrcLightingStamp = 0u;
bool vrcLightingStampValid = false;

uint vrcMixVec(uint h, vec3 v) {
    h = vlcMix(h, floatBitsToUint(v.x));
    h = vlcMix(h, floatBitsToUint(v.y));
    return vlcMix(h, floatBitsToUint(v.z));
}

uint volumeRadianceLightingStamp() {
    if (vrcLightingStampValid) return vrcLightingStamp;
    uint h = vlcMix(0x52414443u, uint(worldData.w.mode));
    h = vrcMixVec(h, worldData.w.sunDir);
    h = vrcMixVec(h, worldData.w.sunColor);
    h = vlcMix(h, floatBitsToUint(worldData.w.sunIntensity));
    int n = min(int(cam.lightCount), int(VLC_LIGHT_CHANNELS) - 1);
    h = vlcMix(h, uint(n));
    for (int i = 0; i < n; ++i) {
        LightData light = lights.l[i];
        h = vrcMixVec(h, light.position.xyz);
        h = vlcMix(h, floatBitsToUint(light.position.w));
        h = vrcMixVec(h, light.direction.xyz);
        h = vrcMixVec(h, light.color.rgb);
        h = vlcMix(h, floatBitsToUint(light.color.a));
    }
    vrcLightingStamp = h;
    vrcLightingStampValid = true;
    return h;
}

// Material terms J depends on beyond the density the slot key already covers.
uint volumeRadianceMaterialStamp(VkVolumeInstance vol) {
    uint h = vlcMix(0x4d41544cu, floatBitsToUint(vol.scatter_coefficient));
    h = vlcMix(h, floatBitsToUint(vol.absorption_coefficient));
    h = vrcMixVec(h, vol.absorption_color);
    h = vrcMixVec(h, vol.scatter_color);
    h = vlcMix(h, floatBitsToUint(vol.scatter_multi));
    h = vlcMix(h, floatBitsToUint(vol.shadow_strength));
    h = vlcMix(h, uint(vol.shadow_steps));
    h = vlcMix(h, uint(vol.emission_mode));
    h = vrcMixVec(h, vol.emission_color);
    return vlcMix(h, floatBitsToUint(vol.emission_intensity));
}

uint volumeRadianceTag(uint key, uint channel, uint lightingStamp, uint materialStamp) {
    uint h = vlcMix(key, channel);
    h = vlcMix(h, lightingStamp);
    h = vlcMix(h, materialStamp);
    return (h & ~1u) | 2u;
}

// Density and source radiance at one world position (a cell vertex).
// False when a light cache lookup it needs is not available yet.
bool vrcComputeVertex(uint slot, VkVolumeInstance vol, vec3 p,
                      pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
                      inout pnanovdb_readaccessor_t acc,
                      out float density, out vec3 source) {
    density = sampleDensityAcc(vol, p, buf, mapH, acc);
    source = vec3(0.0);
    if (density <= 0.0) return true;
    vec3 absColor = max(vol.absorption_color, vec3(0.0));
    float absPeak = max(absColor.r, max(absColor.g, absColor.b));
    vec3 absWeights = absPeak > 1e-5 ? absColor / absPeak : vec3(1.0);
    float sigmaS = density * vol.scatter_coefficient;
    vec3 sigmaT = vec3(sigmaS) + density * vol.absorption_coefficient * absWeights;
    vec3 albedo = vec3(sigmaS) / max(sigmaT, vec3(EPSILON));
    float albedoLuma = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
    const float pIso = 1.0 / (4.0 * PI);
    vec3 inscatter = vec3(0.0);
    if (sigmaS > 0.0) {
        int n = min(int(cam.lightCount), int(VLC_LIGHT_CHANNELS) - 1);
        for (int i = 0; i < n; ++i) {
            LightData light = lights.l[i];
            if (int(light.position.w + 0.5) == 2) continue;   // area: no channel
            vec3 wi;
            float dist, atten;
            bool inverseSquare;
            if (!volSampleLight(light, p, 0.5, 0.5, wi, dist, atten, inverseSquare)) continue;
            if (inverseSquare) atten /= max(dist * dist, 1e-4);
            float integral;
            if (!volumeLightCacheFetch(slot, vol, i, p, buf, mapH, acc, integral)) return false;
            inscatter += light.color.rgb * light.color.a * atten * pIso *
                         volumeShadowFromIntegral(vol, integral) * vol.scatter_color * albedo;
        }
        if (worldData.w.mode == 2) {
            vec3 sunDir = normalize(worldData.w.sunDir);
            float integral;
            if (!volumeLightCacheFetch(slot, vol, -1, p, buf, mapH, acc, integral)) return false;
            vec3 sunLi = sampleTransmittanceLUT(p, sunDir) * worldData.w.sunColor *
                         worldData.w.sunIntensity;
            inscatter += sunLi * pIso * volumeShadowFromIntegral(vol, integral) *
                         vol.scatter_color * albedo;
        }
        // Ambient: sky from above, shadowed by two density taps toward the box
        // top (the cloud renderer's top/bottom AO taps, one-sided).
        vec3 up = vec3(0.0, 1.0, 0.0);
        float h = volumeExitDistance(vol, p, up);
        float tau = 0.0;
        if (h > 1e-4) {
            float sigmaTLuma = vol.scatter_coefficient +
                vol.absorption_coefficient * dot(absWeights, vec3(0.2126, 0.7152, 0.0722));
            tau = (sampleDensityAcc(vol, p + up * (0.25 * h), buf, mapH, acc) +
                   sampleDensityAcc(vol, p + up * (0.75 * h), buf, mapH, acc)) *
                  0.5 * h * sigmaTLuma;
        }
        float skyT = exp(-min(tau, 20.0));
        vec3 sky = sampleSkyAmbient(up);
        inscatter += sky * vol.scatter_color * albedo *
                     (skyT + volumeHigherOrderSkyVisibility(skyT, vol.scatter_multi, albedoLuma));
        inscatter = clampVolumeRadiance(inscatter, 8.0);
    }
    vec3 emis = vec3(0.0);
    if (vol.emission_mode == 1) {
        emis = volumeEmissionSource(vol.emission_color, 1.0, density, 1.0,
            vol.emission_intensity, vol._retired_cloud, false);
    } else if (vol.emission_mode >= 2) {
        emis = volumeEmissionAt(vol, p, density, sampleTemperature(vol, p));
    }
    source = inscatter + emis;
    return true;
}

// Trilinear density + source radiance at a world position, filling missing
// vertices within the shared per-invocation budget. False = incomplete.
bool volumeRadianceCacheFetch(uint slot, VkVolumeInstance vol, VlcSlot s, uvec4 tags,
                              vec3 worldPos, pnanovdb_buf_t buf,
                              pnanovdb_map_handle_t mapH, inout pnanovdb_readaccessor_t acc,
                              out float density, out vec3 source) {
    density = 0.0;
    source = vec3(0.0);
    uvec3 c0, c1;
    vec3 f;
    vlcCorners(vol, s, worldPos, c0, c1, f);
    VlcEntries entries = VlcEntries(s.entries);
    VlcWords words = VlcWords(s.entries);
    vec3 extent = max(vol.aabb_max - vol.aabb_min, vec3(1e-6));
    for (uint k = 0u; k < 8u; ++k) {
        uvec3 corner = vlcCorner(c0, c1, k);
        uint iD = vlcEntryIndex(s, corner, VRC_CH_DENSITY);
        uint64_t wD = entries.e[iD];
        uint64_t wR = entries.e[iD + 1u];
        uint64_t wG = entries.e[iD + 2u];
        uint64_t wB = entries.e[iD + 3u];
        float d;
        vec3 j;
        if (uint(wD) == tags.x && uint(wR) == tags.y && uint(wG) == tags.z && uint(wB) == tags.w) {
            d = uintBitsToFloat(uint(wD >> 32));
            j = vec3(uintBitsToFloat(uint(wR >> 32)), uintBitsToFloat(uint(wG >> 32)),
                     uintBitsToFloat(uint(wB >> 32)));
        } else {
            uint seen = uint(wD);
            uint filling = tags.x | 1u;
            if (seen == filling || vlcFillBudget == 0u ||
                atomicCompSwap(words.w[iD * 2u], seen, filling) != seen) return false;
            vlcFillBudget--;
            vec3 p = vlcLocalToWorld(vol, vol.aabb_min + (vec3(corner) + 0.5) / vec3(s.dims) * extent);
            if (!vrcComputeVertex(slot, vol, p, buf, mapH, acc, d, j)) {
                // Release the claim: a light this vertex needs is not cached yet.
                atomicCompSwap(words.w[iD * 2u], filling, seen);
                return false;
            }
            // Source channels first, density last: its tag publishes the vertex.
            entries.e[iD + 1u] = (uint64_t(floatBitsToUint(j.r)) << 32) | uint64_t(tags.y);
            entries.e[iD + 2u] = (uint64_t(floatBitsToUint(j.g)) << 32) | uint64_t(tags.z);
            entries.e[iD + 3u] = (uint64_t(floatBitsToUint(j.b)) << 32) | uint64_t(tags.w);
            entries.e[iD] = (uint64_t(floatBitsToUint(d)) << 32) | uint64_t(tags.x);
        }
        float w = vlcCornerWeight(f, k);
        density += w * d;
        source += w * j;
    }
    return true;
}

// Bounce-ray march over the cached grid: cell-sized steps of trilinear
// lookups. Writes the segment's radiance and transmittance only on success;
// false leaves both untouched and the caller runs the full march.
bool volumeRadianceCacheMarch(uint slot, VkVolumeInstance vol, vec3 o, vec3 d,
                              float t0, float t1, float jitter,
                              pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
                              inout pnanovdb_readaccessor_t acc,
                              inout vec3 radiance, inout vec3 transmittance) {
    VlcSlot s;
    if (t1 <= t0 || !volumeLightCacheSlot(slot, s)) return false;
    uint lightingStamp = volumeRadianceLightingStamp();
    uint materialStamp = volumeRadianceMaterialStamp(vol);
    uvec4 tags = uvec4(
        volumeRadianceTag(s.key, VRC_CH_DENSITY, lightingStamp, materialStamp),
        volumeRadianceTag(s.key, VRC_CH_SOURCE_R, lightingStamp, materialStamp),
        volumeRadianceTag(s.key, VRC_CH_SOURCE_G, lightingStamp, materialStamp),
        volumeRadianceTag(s.key, VRC_CH_SOURCE_B, lightingStamp, materialStamp));
    vec3 cellWorld = abs(vlcLocalToWorld(vol, vol.aabb_max) - vlcLocalToWorld(vol, vol.aabb_min)) /
                     vec3(s.dims);
    float cell = max(min(cellWorld.x, min(cellWorld.y, cellWorld.z)), 1e-4);
    float len = t1 - t0;
    int n = clamp(int(ceil(len / cell)), 1, 256);
    float h = len / float(n);
    vec3 absColor = max(vol.absorption_color, vec3(0.0));
    float absPeak = max(absColor.r, max(absColor.g, absColor.b));
    vec3 absWeights = absPeak > 1e-5 ? absColor / absPeak : vec3(1.0);
    vec3 sigmaTPerDensity = vec3(vol.scatter_coefficient) + vol.absorption_coefficient * absWeights;
    vec3 rad = vec3(0.0);
    vec3 tr = vec3(1.0);
    for (int i = 0; i < n; ++i) {
        float density;
        vec3 source;
        if (!volumeRadianceCacheFetch(slot, vol, s, tags, o + d * (t0 + (float(i) + jitter) * h),
                                      buf, mapH, acc, density, source)) return false;
        if (density <= 0.0) continue;
        vec3 segT = exp(-sigmaTPerDensity * density * h);
        rad += tr * source * (vec3(1.0) - segT);
        tr *= segT;
        if (max(tr.r, max(tr.g, tr.b)) < 0.001) break;
    }
    radiance = rad;
    transmittance = tr;
    return true;
}

#endif
