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
//     volumes with a program never use the cache.
// A miss is never an error: the caller runs the full march for that ray.
//
// Two coincident gas media (resolveGasOverlapSegment's companion, 2026-10-28):
// each medium keeps its OWN slot and vertex values, and a step combines them
//     sigma = sigma_1 + sigma_2,   J = (sigma_1 J_1 + sigma_2 J_2) / sigma,
// which is exact for media that superpose. It used to refuse the overlap and
// march it in full: in the 4-source scene bounce rays then took 3.3 density
// samples per path instead of 0 and the frame went 451 -> 964 ms.
// Beyond the single-medium cache: each J is shadowed by its own medium only
// (the full march multiplies the companion's light march in), and the
// companion's own emission is in its J, which the full camera march omits
// (volume_overlap_selection.glsl limit; the emissive volume is the primary).
// One vertex value per medium is kept on purpose: a companion-only variant
// would share the medium's cells and the two would refill each other.
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

// True when all eight density vertices of this position are published and zero.
// The step then adds nothing, so the three source channels and the accessor
// work behind them are skipped. A bounce ray crosses mostly empty space, and a
// full fetch costs 32 incoherent loads per step, four times a density sample.
bool vrcProvenEmpty(VkVolumeInstance vol, VlcSlot s, uint densityTag, vec3 worldPos) {
    uvec3 c0, c1;
    vec3 f;
    vlcCorners(vol, s, worldPos, c0, c1, f);
    VlcEntries entries = VlcEntries(s.entries);
    for (uint k = 0u; k < 8u; ++k) {
        uint64_t word = entries.e[vlcEntryIndex(s, vlcCorner(c0, c1, k), VRC_CH_DENSITY)];
        if (uint(word) != densityTag || uintBitsToFloat(uint(word >> 32)) > 0.0) {
            return false;
        }
    }
    return true;
}

// Per-medium constants of a cached march.
struct VrcMedium {
    uint slot;
    VlcSlot s;
    uvec4 tags;
    vec3 sigmaTPerDensity;
    float cell;
};

bool vrcMediumSetup(uint slot, VkVolumeInstance vol, uint lightingStamp, out VrcMedium m) {
    m.slot = slot;
    if (!volumeLightCacheSlot(slot, m.s)) return false;
    uint materialStamp = volumeRadianceMaterialStamp(vol);
    m.tags = uvec4(
        volumeRadianceTag(m.s.key, VRC_CH_DENSITY, lightingStamp, materialStamp),
        volumeRadianceTag(m.s.key, VRC_CH_SOURCE_R, lightingStamp, materialStamp),
        volumeRadianceTag(m.s.key, VRC_CH_SOURCE_G, lightingStamp, materialStamp),
        volumeRadianceTag(m.s.key, VRC_CH_SOURCE_B, lightingStamp, materialStamp));
    vec3 cellWorld = abs(vlcLocalToWorld(vol, vol.aabb_max) - vlcLocalToWorld(vol, vol.aabb_min)) /
                     vec3(m.s.dims);
    m.cell = max(min(cellWorld.x, min(cellWorld.y, cellWorld.z)), 1e-4);
    vec3 absColor = max(vol.absorption_color, vec3(0.0));
    float absPeak = max(absColor.r, max(absColor.g, absColor.b));
    vec3 absWeights = absPeak > 1e-5 ? absColor / absPeak : vec3(1.0);
    m.sigmaTPerDensity = vec3(vol.scatter_coefficient) + vol.absorption_coefficient * absWeights;
    return true;
}

// Bounce-ray march over the cached grid: cell-sized steps of trilinear
// lookups, over a prepared pair or all general-mixture candidates.
// Writes the segment's radiance and transmittance only on success; false
// leaves both untouched and the caller runs the full march.
// The media are walked by ONE fetch call site in a bounded member loop, and
// each reads its instance from the volume table: two call sites would inline
// the vertex fill (light loop + light cache) twice into this stage.
bool volumeRadianceCacheMarch(uint slot, uint companionSlot, uint slots, uint mixtureCount, vec3 o, vec3 d,
                              float t0, float t1, float jitter,
                              pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
                              inout pnanovdb_readaccessor_t acc,
                              pnanovdb_buf_t compBuf, pnanovdb_map_handle_t compMapH,
                              inout pnanovdb_readaccessor_t compAcc,
                              inout vec3 radiance, inout vec3 transmittance) {
    if (t1 <= t0) return false;
    uint lightingStamp = volumeRadianceLightingStamp();
    // Scalar setup avoids dynamically indexed aggregate arrays in the RT driver.
    float cell = 1e30;
    // Keyed by slot, not loop position: an SDF slot between two gases would
    // otherwise leave the second gas to rehash every tag at every step.
    const uint VRC_NO_SLOT = 0xffffffffu;
    VrcMedium firstMedium, secondMedium;
    uint firstSlot = VRC_NO_SLOT;
    uint secondSlot = VRC_NO_SLOT;
    uint memberCount = mixtureCount > 0u ? min(mixtureCount, 16u) : slots;
    for (uint m = 0u; m < memberCount; ++m) {
        uint index = mixtureCount > 0u ? m : (m == 0u ? slot : companionSlot);
        VkVolumeInstance v = volumes.v[index];
        if (mixtureCount > 0u && !gasOverlapCandidate(v)) continue;
        VrcMedium medium;
        if (!vrcMediumSetup(index, v, lightingStamp, medium)) return false;
        cell = min(cell, medium.cell);
        if (firstSlot == VRC_NO_SLOT) {
            firstMedium = medium;
            firstSlot = index;
        } else if (secondSlot == VRC_NO_SLOT) {
            secondMedium = medium;
            secondSlot = index;
        }
    }
    // The authored bounce LOD applies here too (RT_GAS_SECONDARY_MARCH.md §8):
    // bounce rays reach this march, not the full one that used to read the word.
    uint marchLod = worldData.w.volumeMarchLod;
    float bounceStep = cell * float(((marchLod >> 8) & 15u) + 1u);
    int bounceCap = int((marchLod >> 12) & 511u);
    float len = t1 - t0;
    int n = clamp(int(ceil(len / bounceStep)), 1, bounceCap > 0 ? min(bounceCap, 256) : 256);
    float h = len / float(n);
    vec3 rad = vec3(0.0);
    vec3 tr = vec3(1.0);
    for (int i = 0; i < n; ++i) {
        vec3 p = o + d * (t0 + (float(i) + jitter) * h);
        vec3 sigma = vec3(0.0);
        vec3 emitted = vec3(0.0);   // sum of sigma_m * J_m
        for (uint m = 0u; m < memberCount; ++m) {
            uint mSlot = mixtureCount > 0u ? m : (m == 0u ? slot : companionSlot);
            if (mixtureCount > 0u && !gasOverlapCandidate(volumes.v[mSlot])) continue;
            if (mixtureCount > 0u || m == 1u) {
                // Outside its box the companion is empty; the corner lookup
                // would clamp to its edge cells instead.
                vec3 local = vlcWorldToLocal(volumes.v[mSlot], p);
                if (any(lessThan(local, volumes.v[mSlot].aabb_min)) ||
                    any(greaterThan(local, volumes.v[mSlot].aabb_max))) continue;
            }
            VrcMedium medium;
            if (mSlot == firstSlot) medium = firstMedium;
            else if (mSlot == secondSlot) medium = secondMedium;
            else if (!vrcMediumSetup(mSlot, volumes.v[mSlot], lightingStamp, medium)) return false;
            if (vrcProvenEmpty(volumes.v[mSlot], medium.s, medium.tags.x, p)) continue;
            float density;
            vec3 source;
            pnanovdb_buf_t mBuf = m == 0u ? buf : compBuf;
            pnanovdb_map_handle_t mMap = m == 0u ? mapH : compMapH;
            pnanovdb_readaccessor_t mAcc = m == 0u ? acc : compAcc;
            if (mixtureCount > 0u && mSlot == slot) {
                mBuf = buf;
                mMap = mapH;
                mAcc = acc;
            } else if (mixtureCount > 0u && slots > 1u && mSlot == companionSlot) {
                mBuf = compBuf;
                mMap = compMapH;
                mAcc = compAcc;
            } else if (mixtureCount > 0u) {
                // General mixtures can contain more than the prepared pair.
                mBuf.address = volumes.v[mSlot].volume_type == 2
                    ? volumes.v[mSlot].vdb_grid_address : uint64_t(0);
                mMap.address.byte_offset = 0u;
                pnanovdb_root_handle_t root;
                root.address.byte_offset = 0u;
                if (mBuf.address != uint64_t(0)) {
                    pnanovdb_grid_handle_t grid;
                    grid.address.byte_offset = 0u;
                    root = pnanovdb_tree_get_root(mBuf, pnanovdb_grid_get_tree(mBuf, grid));
                    mMap = pnanovdb_grid_get_map(mBuf, grid);
                }
                pnanovdb_readaccessor_init(mAcc, root);
            }
            bool ok = volumeRadianceCacheFetch(mSlot, volumes.v[mSlot], medium.s,
                                               medium.tags, p, mBuf, mMap, mAcc,
                                               density, source);
            if (mSlot == slot) acc = mAcc;
            else if (slots > 1u && mSlot == companionSlot) compAcc = mAcc;
            if (!ok) return false;
            if (density <= 0.0) continue;
            vec3 s = medium.sigmaTPerDensity * density;
            sigma += s;
            emitted += s * source;
        }
        if (max(sigma.r, max(sigma.g, sigma.b)) <= 0.0) continue;
        vec3 segT = exp(-sigma * h);
        rad += tr * (emitted / max(sigma, vec3(EPSILON))) * (vec3(1.0) - segT);
        tr *= segT;
        if (max(tr.r, max(tr.g, tr.b)) < 0.001) break;
    }
    radiance = rad;
    transmittance = tr;
    return true;
}

#endif
