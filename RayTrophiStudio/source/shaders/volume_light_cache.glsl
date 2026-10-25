// Volume light cache: per-cell density integral toward a light, shared by the
// volume closest-hit (reads AND fills) and the surface closest-hit (reads only).
// Design: docs/dev/VOLUME_LIGHT_CACHE.md.
//
// Requires before inclusion: VkVolumeInstance, LightData, `lights`, `worldData`
// (with volumeLightCacheLo/Hi) and volume_instrumentation.glsl.
//
// ★ The cache stores I = integral of the SAMPLED density along the chord from a
// cell centre toward the light, NOT optical depth. Each consumer multiplies by
// its own extinction and applies its own shadow-strength rule, exactly as the
// march it replaces did. Storing sigma_t * I would tie the cache to one of the
// two formulas and invalidate it on every scatter/absorption edit.
//
// ★ A miss is never an error and never "unshadowed": every caller falls back to
// the march it always ran. The cache only ever replaces work, never a result.
#ifndef RAYTROPHI_VOLUME_LIGHT_CACHE_GLSL
#define RAYTROPHI_VOLUME_LIGHT_CACHE_GLSL

// Entry layout per cell: 0 = Nishita sun, 1..5 = scene lights 0..4 (density
// integrals), 6..9 = radiance cache (density, source R/G/B;
// volume_radiance_cache.glsl). VLC_LIGHT_CHANNELS bounds the LIGHT mapping:
// it must not follow VLC_CHANNELS, or lights 5..8 would map onto radiance
// channels.
const uint VLC_CHANNELS = 10u;
const uint VLC_LIGHT_CHANNELS = 6u;
const uint VLC_MAX_SLOTS = 16u;
const uint VLC_ENTRY_BASE_BYTES = 512u;

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer VlcHeader {
    uint slotCount;
    uint channelCount;
    uint reserved0;
    uint reserved1;
    uvec4 slots[];   // x = entry offset, y = dims 10:10:10, z = volume key (0 = off)
};
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer VlcEntries {
    uint64_t e[];    // low 32 = tag, high 32 = float bits of the integral
};
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer VlcWords {
    uint w[];        // 32-bit view of the same entries, for the claim CAS
};

struct VlcSlot {
    uint64_t entries;    // address of entry 0 of this slot
    uvec3 dims;
    uint key;
};

uint64_t volumeLightCacheBase() {
    return (uint64_t(worldData.w.volumeLightCacheHi) << 32) |
           uint64_t(worldData.w.volumeLightCacheLo);
}

bool volumeLightCacheSlot(uint slot, out VlcSlot s) {
    s.entries = 0ul;
    s.dims = uvec3(0u);
    s.key = 0u;
    uint64_t base = volumeLightCacheBase();
    if (base == 0ul || slot >= VLC_MAX_SLOTS) return false;
    VlcHeader h = VlcHeader(base);
    if (slot >= h.slotCount || h.channelCount != VLC_CHANNELS) return false;
    uvec4 r = h.slots[slot];
    if (r.z == 0u) return false;
    s.dims = uvec3(r.y & 1023u, (r.y >> 10) & 1023u, (r.y >> 20) & 1023u);
    if (any(lessThan(s.dims, uvec3(2u)))) return false;
    s.entries = base + uint64_t(VLC_ENTRY_BASE_BYTES) + uint64_t(r.x) * 8ul;
    s.key = r.z;
    return true;
}

uint vlcMix(uint h, uint v) {
    h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
    h *= 0x85ebca6bu;
    return h ^ (h >> 13);
}

// Channel and light geometry for a light. lightIndex -1 = the Nishita sun.
// Returns false for anything the cache does not model (area lights, lights past
// the channel budget, no sun); the caller keeps its march for those.
bool volumeLightCacheChannel(int lightIndex, out uint channel, out bool positional,
                             out vec3 geometry) {
    channel = 0u;
    positional = false;
    geometry = vec3(0.0);
    if (lightIndex < 0) {
        if (worldData.w.mode != 2) return false;
        geometry = normalize(worldData.w.sunDir);
        return true;
    }
    if (lightIndex >= int(VLC_LIGHT_CHANNELS) - 1) return false;
    LightData light = lights.l[lightIndex];
    int type = int(light.position.w + 0.5);
    if (type == 1) {
        geometry = normalize(light.direction.xyz);
    } else if (type == 0 || type == 3) {
        geometry = light.position.xyz;
        positional = true;
    } else {
        return false;   // area: the sampled point on the shape moves per sample
    }
    channel = uint(lightIndex) + 1u;
    return true;
}

// Tag of one channel of one slot. bit0 = 0 (bit0 marks "being filled"),
// bit1 = 1 (so zero-initialised memory can never match).
uint volumeLightCacheTag(uint key, uint channel, bool positional, vec3 geometry) {
    uint h = vlcMix(key, channel);
    h = vlcMix(h, positional ? 1u : 0u);
    h = vlcMix(h, floatBitsToUint(geometry.x));
    h = vlcMix(h, floatBitsToUint(geometry.y));
    h = vlcMix(h, floatBitsToUint(geometry.z));
    return (h & ~1u) | 2u;
}

vec3 vlcWorldToLocal(VkVolumeInstance vol, vec3 p) {
    return vec3(
        vol.inv_transform[0] * p.x + vol.inv_transform[1] * p.y + vol.inv_transform[2] * p.z + vol.inv_transform[3],
        vol.inv_transform[4] * p.x + vol.inv_transform[5] * p.y + vol.inv_transform[6] * p.z + vol.inv_transform[7],
        vol.inv_transform[8] * p.x + vol.inv_transform[9] * p.y + vol.inv_transform[10] * p.z + vol.inv_transform[11]);
}

vec3 vlcLocalToWorld(VkVolumeInstance vol, vec3 p) {
    return vec3(
        vol.transform[0] * p.x + vol.transform[1] * p.y + vol.transform[2] * p.z + vol.transform[3],
        vol.transform[4] * p.x + vol.transform[5] * p.y + vol.transform[6] * p.z + vol.transform[7],
        vol.transform[8] * p.x + vol.transform[9] * p.y + vol.transform[10] * p.z + vol.transform[11]);
}

uint vlcEntryIndex(VlcSlot s, uvec3 c, uint channel) {
    uint cell = c.x + s.dims.x * (c.y + s.dims.y * c.z);
    return cell * VLC_CHANNELS + channel;
}

// Trilinear corners of a world position. Cell centres sit at
// aabb_min + (i + 0.5) * cellSize; outside the box the coordinate clamps to the
// border cells, which is what a surface point just outside the domain needs.
void vlcCorners(VkVolumeInstance vol, VlcSlot s, vec3 worldPos,
                out uvec3 c0, out uvec3 c1, out vec3 f) {
    vec3 extent = max(vol.aabb_max - vol.aabb_min, vec3(1e-6));
    vec3 u = (vlcWorldToLocal(vol, worldPos) - vol.aabb_min) / extent * vec3(s.dims) - 0.5;
    u = clamp(u, vec3(0.0), vec3(s.dims) - 1.0);
    vec3 fl = floor(u);
    f = u - fl;
    c0 = uvec3(fl);
    c1 = min(c0 + 1u, s.dims - 1u);
}

uvec3 vlcCorner(uvec3 c0, uvec3 c1, uint k) {
    return uvec3((k & 1u) != 0u ? c1.x : c0.x,
                 (k & 2u) != 0u ? c1.y : c0.y,
                 (k & 4u) != 0u ? c1.z : c0.z);
}

float vlcCornerWeight(vec3 f, uint k) {
    return ((k & 1u) != 0u ? f.x : 1.0 - f.x) *
           ((k & 2u) != 0u ? f.y : 1.0 - f.y) *
           ((k & 4u) != 0u ? f.z : 1.0 - f.z);
}

// Read-only lookup (surface closest-hit). True only when all eight corners
// carry this tag; then `integral` is their trilinear blend.
bool volumeLightCacheRead(uint slot, VkVolumeInstance vol, int lightIndex,
                          vec3 worldPos, out float integral) {
    integral = 0.0;
    VlcSlot s;
    if (!volumeLightCacheSlot(slot, s)) return false;
    uint channel;
    bool positional;
    vec3 geometry;
    if (!volumeLightCacheChannel(lightIndex, channel, positional, geometry)) return false;
    uint tag = volumeLightCacheTag(s.key, channel, positional, geometry);
    uvec3 c0, c1;
    vec3 f;
    vlcCorners(vol, s, worldPos, c0, c1, f);
    VlcEntries entries = VlcEntries(s.entries);
    float sum = 0.0;
    for (uint k = 0u; k < 8u; ++k) {
        uint64_t word = entries.e[vlcEntryIndex(s, vlcCorner(c0, c1, k), channel)];
        if (uint(word) != tag) {
            volumeRecordLightCache(false, 0u, 0u);
            return false;
        }
        sum += vlcCornerWeight(f, k) * uintBitsToFloat(uint(word >> 32));
    }
    integral = sum;
    volumeRecordLightCache(true, 0u, 0u);
    return true;
}

#endif
