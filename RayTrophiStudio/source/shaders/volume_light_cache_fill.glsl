// Volume light cache, volume closest-hit side: lookup that FILLS missing cells.
// docs/dev/VOLUME_LIGHT_CACHE.md. Requires volume_light_cache.glsl,
// sampleDensityAcc and volumeExitDistance.
//
// ★ The fill uses sampleDensityAcc - the very sampler the march shades with -
// so the shadow can never be cast by a different density than the one that is
// drawn. That is the reason this is a lazy fill inside the RT shader and not a
// compute pass: a compute pass would need its own copy of NanoVDB, dense gas,
// Volume Graph noise and cutoff feathering, and two copies drift silently.
#ifndef RAYTROPHI_VOLUME_LIGHT_CACHE_FILL_GLSL
#define RAYTROPHI_VOLUME_LIGHT_CACHE_FILL_GLSL

// Fill budget of ONE closest-hit invocation, shared by every lookup it makes.
// A per-lookup budget would let a cold cache charge every march step of every
// light: ~26 steps x 3 lights x 2 cells x 32 samples on frame one. Per
// invocation the worst case is 4 cells (<= 128 samples) on top of the march it
// would have run anyway; a few million rays per sample still fill a 64^3 grid
// within the first samples. (Private global: one copy per invocation.)
const uint VLC_FILLS_PER_INVOCATION = 4u;
uint vlcFillBudget = VLC_FILLS_PER_INVOCATION;

// Density integral from a cell centre toward the light, deterministic: fixed
// midpoint lattice, step count from the cell size only. Two invocations that
// fill the same cell write the same bits.
float vlcFillCell(VkVolumeInstance vol, VlcSlot s, uvec3 c, bool positional, vec3 geometry,
                  pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
                  inout pnanovdb_readaccessor_t acc, out uint sampleCount) {
    vec3 extent = max(vol.aabb_max - vol.aabb_min, vec3(1e-6));
    vec3 p = vlcLocalToWorld(vol, vol.aabb_min + (vec3(c) + 0.5) / vec3(s.dims) * extent);
    vec3 dir = geometry;
    float lightDist = 1e30;
    if (positional) {
        vec3 toLight = geometry - p;
        lightDist = length(toLight);
        dir = lightDist > 1e-6 ? toLight / lightDist : vec3(0.0, 1.0, 0.0);
    }
    float chord = min(lightDist, volumeExitDistance(vol, p, dir));
    sampleCount = 0u;
    if (chord <= 1e-6) return 0.0;
    vec3 boxWorld = vlcLocalToWorld(vol, vol.aabb_max) - vlcLocalToWorld(vol, vol.aabb_min);
    float cellWorld = max(length(boxWorld) / length(vec3(s.dims)), 1e-6);
    // One sample per cell crossed, 4..32. The march this replaces takes at most
    // 12-16 adaptive samples, so the cached chord is never sampled coarser.
    int n = clamp(int(ceil(chord / cellWorld)), 4, 32);
    float h = chord / float(n);
    float sum = 0.0;
    for (int k = 0; k < n; ++k) {
        sum += sampleDensityAcc(vol, p + dir * ((float(k) + 0.5) * h), buf, mapH, acc);
    }
    sampleCount = uint(n);
    return sum * h;
}

// True when all eight corners are valid after this call (some possibly filled
// by it). False = the caller runs its march for this sample, as before.
bool volumeLightCacheFetch(uint slot, VkVolumeInstance vol, int lightIndex, vec3 worldPos,
                           pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
                           inout pnanovdb_readaccessor_t acc, out float integral) {
    integral = 0.0;
    VlcSlot s;
    if (!volumeLightCacheSlot(slot, s)) return false;
    uint channel;
    bool positional;
    vec3 geometry;
    if (!volumeLightCacheChannel(lightIndex, channel, positional, geometry)) return false;
    uint tag = volumeLightCacheTag(s.key, channel, positional, geometry);
    uint filling = tag | 1u;
    uvec3 c0, c1;
    vec3 f;
    vlcCorners(vol, s, worldPos, c0, c1, f);
    VlcEntries entries = VlcEntries(s.entries);
    VlcWords words = VlcWords(s.entries);
    float sum = 0.0;
    bool complete = true;
    uint fills = 0u;
    uint fillSamples = 0u;
    for (uint k = 0u; k < 8u; ++k) {
        uvec3 corner = vlcCorner(c0, c1, k);
        uint index = vlcEntryIndex(s, corner, channel);
        uint64_t word = entries.e[index];
        uint seen = uint(word);
        if (seen == tag) {
            sum += vlcCornerWeight(f, k) * uintBitsToFloat(uint(word >> 32));
            continue;
        }
        // Someone else is filling it for THIS tag: do not duplicate the work.
        // A stale "filling" mark from another tag is reclaimable (it never
        // completed for the current content).
        if (seen != filling && vlcFillBudget > 0u &&
            atomicCompSwap(words.w[index * 2u], seen, filling) == seen) {
            uint n;
            float value = vlcFillCell(vol, s, corner, positional, geometry, buf, mapH, acc, n);
            // One 64-bit store publishes tag and value together; a reader can
            // never pair this tag with the previous content's value.
            entries.e[index] = (uint64_t(floatBitsToUint(value)) << 32) | uint64_t(tag);
            fills++;
            vlcFillBudget--;
            fillSamples += n;
            sum += vlcCornerWeight(f, k) * value;
            continue;
        }
        complete = false;
    }
    volumeRecordLightCache(complete, fills, fillSamples);
    if (!complete) return false;
    integral = sum;
    return true;
}

// Same transmittance rule as lightMarchAcc, from a cached integral: the
// colour-weighted extinction and the 0.92 * shadow_strength blend, so the
// cached and marched answers differ only by how the integral was sampled.
float volumeShadowFromIntegral(VkVolumeInstance vol, float integral) {
    if (vol.shadow_steps <= 0) return 1.0;
    vec3 shadowAbsColor = max(vol.absorption_color, vec3(0.0));
    float shadowAbsPeak = max(shadowAbsColor.r, max(shadowAbsColor.g, shadowAbsColor.b));
    vec3 shadowAbsWeights = shadowAbsPeak > 1e-5 ? shadowAbsColor / shadowAbsPeak : vec3(1.0);
    float sigma_t = vol.scatter_coefficient +
        vol.absorption_coefficient * dot(shadowAbsWeights, vec3(0.2126, 0.7152, 0.0722));
    if (sigma_t <= EPSILON) return 1.0;
    float physTrans = exp(-min(integral * sigma_t, 10.0));
    float shadowStrength = clamp(vol.shadow_strength * 0.92, 0.0, 1.0);
    return 1.0 - shadowStrength * (1.0 - physTrans);
}

#endif
