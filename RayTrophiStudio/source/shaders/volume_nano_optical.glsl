#ifndef VOLUME_NANO_OPTICAL_GLSL
#define VOLUME_NANO_OPTICAL_GLSL

#include "volume_optical_activity.glsl"

// One affine transform per straight ray, shared by traversal and interpolation.
struct VolumeNanoRay {
    vec3 indexOrigin;
    vec3 indexDirection;
    vec3 localOrigin;
    vec3 localDirection;
};

// Invocation-local raw-cell cache. Never cache remapped density: two instances
// may share a grid but have different optical materials. Grid memory is read-only
// during tracing, and the address plus integer cell identifies all eight values.
// Fixed scalar slots avoid dynamic buffer/accessor arrays in the heavy RT stage.
uint64_t volumeCellGridA = uint64_t(0);
uint64_t volumeCellGridB = uint64_t(0);
ivec3 volumeCellA;
ivec3 volumeCellB;
vec4 volumeCellLowA;
vec4 volumeCellHighA;
vec4 volumeCellLowB;
vec4 volumeCellHighB;

void volumeNanoCorners(pnanovdb_buf_t buf, ivec3 cell,
    inout pnanovdb_readaccessor_t acc, out vec4 low, out vec4 high) {
    if (volumeCellGridA == buf.address && all(equal(volumeCellA, cell))) {
        low = volumeCellLowA;
        high = volumeCellHighA;
        return;
    }
    if (volumeCellGridB == buf.address && all(equal(volumeCellB, cell))) {
        low = volumeCellLowB;
        high = volumeCellHighB;
        return;
    }
    pnanovdb_coord_t first;
    first.x = cell.x;
    first.y = cell.y;
    first.z = cell.z;
    uint level = 0u;
    pnanovdb_address_t firstAddress = pnanovdb_readaccessor_get_value_address_and_level(
        PNANOVDB_GRID_TYPE_FLOAT, buf, acc, first, level);
    float firstValue = pnanovdb_read_float(buf, firstAddress);
    // Every corner stays in one 8^3 leaf unless an integer coordinate ends
    // in seven. Bit masking also handles negative NanoVDB coordinates.
    bool sameLeaf = all(lessThan(cell & ivec3(7), ivec3(7)));
    low = vec4(firstValue);
    high = vec4(firstValue);
    for (int i = 1; i < 8; ++i) {
        // A non-leaf lookup is a constant tile/background. All corners within
        // the same 8^3 region necessarily resolve to that very same value.
        if (sameLeaf && level != 0u) break;
        pnanovdb_coord_t ijk;
        ijk.x = cell.x + (i & 1);
        ijk.y = cell.y + ((i >> 1) & 1);
        ijk.z = cell.z + ((i >> 2) & 1);
        pnanovdb_address_t address;
        if (sameLeaf) {
            address = pnanovdb_leaf_get_value_address(
                PNANOVDB_GRID_TYPE_FLOAT, buf, acc.leaf, ijk);
        } else {
            address = pnanovdb_readaccessor_get_value_address(
                PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk);
        }
        float value = pnanovdb_read_float(buf, address);
        if (i < 4) low[i] = value;
        else high[i - 4] = value;
    }
    // Reserve a slot per grid, so alternating primary/companion lookups do not
    // evict each other. Additional grids retain the uncached scalar fallback.
    if (volumeCellGridA == uint64_t(0) || volumeCellGridA == buf.address) {
        volumeCellGridA = buf.address;
        volumeCellA = cell;
        volumeCellLowA = low;
        volumeCellHighA = high;
    } else if (volumeCellGridB == uint64_t(0) || volumeCellGridB == buf.address) {
        volumeCellGridB = buf.address;
        volumeCellB = cell;
        volumeCellLowB = low;
        volumeCellHighB = high;
    }
}

bool volumeNanoIndexEligible(VkVolumeInstance vol, pnanovdb_buf_t buf) {
    return vol.volume_type == 2 && buf.address != uint64_t(0) &&
        vol._ext_reserved[7] < 0.5 && (vol._reserved[1] <= 0.5 ||
            matProgramOffset(uint(vol._reserved[1] - 1.0)) == MATPROG_NONE);
}

VolumeNanoRay volumeNanoRay(VkVolumeInstance vol, vec3 p, vec3 d,
    pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH) {
    VolumeNanoRay ray;
    ray.localOrigin = vec3(
        dot(vec3(vol.inv_transform[0], vol.inv_transform[1], vol.inv_transform[2]), p),
        dot(vec3(vol.inv_transform[4], vol.inv_transform[5], vol.inv_transform[6]), p),
        dot(vec3(vol.inv_transform[8], vol.inv_transform[9], vol.inv_transform[10]), p)) +
        vec3(vol.inv_transform[3], vol.inv_transform[7], vol.inv_transform[11]);
    ray.localDirection = vec3(
        dot(vec3(vol.inv_transform[0], vol.inv_transform[1], vol.inv_transform[2]), d),
        dot(vec3(vol.inv_transform[4], vol.inv_transform[5], vol.inv_transform[6]), d),
        dot(vec3(vol.inv_transform[8], vol.inv_transform[9], vol.inv_transform[10]), d));
    vec3 origin = ray.localOrigin -
        vec3(vol.pivot_offset[0], vol.pivot_offset[1], vol.pivot_offset[2]);
    pnanovdb_vec3_t a = pnanovdb_vec3_uniform(0.0);
    a.x = origin.x; a.y = origin.y; a.z = origin.z;
    pnanovdb_vec3_t b = pnanovdb_vec3_uniform(0.0);
    b.x = origin.x + ray.localDirection.x;
    b.y = origin.y + ray.localDirection.y;
    b.z = origin.z + ray.localDirection.z;
    pnanovdb_vec3_t ia = pnanovdb_map_apply_inverse(buf, mapH, a);
    pnanovdb_vec3_t ib = pnanovdb_map_apply_inverse(buf, mapH, b);
    ray.indexOrigin = vec3(ia.x, ia.y, ia.z);
    ray.indexDirection = vec3(ib.x - ia.x, ib.y - ia.y, ib.z - ia.z);
    return ray;
}

float volumeNanoRemap(VkVolumeInstance vol, float raw) {
    return volumeDensityCurve(max((raw - vol.density_remap_low) /
        max(vol.density_remap_high - vol.density_remap_low, EPSILON), 0.0),
        vol.density_curve);
}

// Exact same eight corners and optical transform as sampleDensityAccMode.
// Call only for a NanoVDB field without material density noise or a graph.
float volumeNanoDensity(VkVolumeInstance vol, VolumeNanoRay ray, float t,
    pnanovdb_buf_t buf, inout pnanovdb_readaccessor_t acc) {
    if (vol.density_multiplier == 0.0) return 0.0;
    vec3 local = ray.localOrigin + ray.localDirection * t;
    if (any(lessThan(local, vol.aabb_min)) || any(greaterThan(local, vol.aabb_max))) {
        return 0.0;
    }
    vec3 index = ray.indexOrigin + ray.indexDirection * t;
    vec3 cell = floor(index);
    vec3 f = index - cell;
    vec4 low, high;
    volumeNanoCorners(buf, ivec3(cell), acc, low, high);
    float raw = mix(mix(mix(low.x, low.y, f.x),
                       mix(low.z, low.w, f.x), f.y),
                   mix(mix(high.x, high.y, f.x),
                       mix(high.z, high.w, f.x), f.y), f.z);
    if (isnan(raw) || isinf(raw)) return 0.0;
    float density = volumeNanoRemap(vol, raw);
    float cutoff = max(vol._reserved[0], 0.0);
    if (density <= cutoff) return 0.0;
    float fade = cutoff > 0.0 ? smoothstep(cutoff, cutoff * 2.0, density) : 1.0;
    return density * vol.density_multiplier * fade;
}

float nanoEmptyIndexStep(VkVolumeInstance vol, vec3 index, vec3 direction,
    float baseStep, pnanovdb_buf_t buf, inout pnanovdb_readaccessor_t acc,
    out uint skipKind) {
    skipKind = 0u;
    if (buf.address == 0) return baseStep;
    if (vol._ext_reserved[7] >= 0.5 || !(vol._reserved[1] <= 0.5) ||
        vol.source_type == 4 || vol.volume_type == 3) return baseStep;
    pnanovdb_coord_t ijk;
    ijk.x = int(floor(index.x));
    ijk.y = int(floor(index.y));
    ijk.z = int(floor(index.z));
    uint dim = pnanovdb_readaccessor_get_dim(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk);
    vec3 boxMin;
    vec3 boxMax;
    bool leaf = dim <= 1u;
    if (leaf) {
        // Only the dense-to-sparse live producer guarantees zero background
        // and inactive values below its copyFromDense tolerance. Arbitrary VDB
        // assets may carry inactive values excluded from NanoVDB statistics.
        if (vol._uvw_pad[0] < 0.5) return baseStep;
        // Trilinear support must stay inside this leaf. In its last voxel slab
        // the adjacent leaf may be visible, so preserve the normal sampler.
        boxMin = floor(index / 8.0) * 8.0;
        boxMax = boxMin + vec3(7.0);
        if (any(greaterThanEqual(index, boxMax)) || vol._ext_reserved[7] >= 0.5) {
            return baseStep;
        }
        pnanovdb_grid_handle_t grid;
        grid.address.byte_offset = 0u;
        bool hasBounds = (pnanovdb_grid_get_flags(buf, grid) &
            uint(PNANOVDB_GRID_FLAGS_HAS_MIN_MAX)) != 0u;
        volumeRecordMajorantQuery(hasBounds);
        if (!hasBounds) return baseStep;
        float rawMax = pnanovdb_read_float(buf, pnanovdb_leaf_get_max_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, acc.leaf));
        if (isnan(rawMax) || isinf(rawMax) ||
            !(volumeNanoRemap(vol, max(rawMax, 1e-6)) <= max(vol._reserved[0], 0.0))) {
            return baseStep;
        }
    } else {
        // Active bits are topology, not an optical emptiness proof. A constant
        // tile may be active and zero, or inactive with a nonzero background.
        // Read its single value; never fetch eight corners for a proven zero.
        float rawTile = pnanovdb_read_float(buf,
            pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk));
        if (isnan(rawTile) || isinf(rawTile) ||
            !(volumeNanoRemap(vol, rawTile) <= max(vol._reserved[0], 0.0))) return baseStep;
        boxMin = floor(index / float(dim)) * float(dim);
        // The final voxel slab interpolates with the next tile/leaf. It must
        // retain normal sampling, even when the current tile is exactly zero.
        boxMax = boxMin + vec3(float(dim) - 1.0);
        if (any(greaterThanEqual(index, boxMax))) return baseStep;
    }
    vec3 boundary = mix(boxMin, boxMax, greaterThan(direction, vec3(0.0)));
    vec3 exitT = vec3(1e30);
    for (int axis = 0; axis < 3; ++axis) {
        if (abs(direction[axis]) > 1e-8) {
            exitT[axis] = (boundary[axis] - index[axis]) / direction[axis];
        }
    }
    float span = min(exitT.x, min(exitT.y, exitT.z));
    // Stay inside proven support. The outer sample loop floors skipped steps.
    float guard = leaf ? max(1e-6, span * 1e-5) : max(vol.voxel_size, baseStep);
    if (span - guard <= baseStep) return baseStep;
    skipKind = leaf ? 2u : 1u;
    return span - guard;
}

float nanoEmptyTileStep(VkVolumeInstance vol, vec3 p, vec3 d, float baseStep,
    pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc, out uint skipKind) {
    skipKind = 0u;
    if (buf.address == 0) return baseStep;
    VolumeNanoRay ray = volumeNanoRay(vol, p, d, buf, mapH);
    return nanoEmptyIndexStep(vol, ray.indexOrigin, ray.indexDirection,
        baseStep, buf, acc, skipKind);
}
#endif
