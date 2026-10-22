#ifndef VOLUME_NANO_OPTICAL_GLSL
#define VOLUME_NANO_OPTICAL_GLSL

// One affine transform per straight ray, shared by traversal and interpolation.
struct VolumeNanoRay {
    vec3 indexOrigin;
    vec3 indexDirection;
    vec3 localOrigin;
    vec3 localDirection;
};

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
    vec3 local = ray.localOrigin + ray.localDirection * t;
    if (any(lessThan(local, vol.aabb_min)) || any(greaterThan(local, vol.aabb_max))) {
        return 0.0;
    }
    vec3 index = ray.indexOrigin + ray.indexDirection * t;
    vec3 cell = floor(index);
    vec3 f = index - cell;
    float corners[8];
    for (int i = 0; i < 8; ++i) {
        pnanovdb_coord_t ijk;
        ijk.x = int(cell.x) + (i & 1);
        ijk.y = int(cell.y) + ((i >> 1) & 1);
        ijk.z = int(cell.z) + ((i >> 2) & 1);
        corners[i] = pnanovdb_read_float(buf,
            pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk));
    }
    float raw = mix(mix(mix(corners[0], corners[1], f.x),
                       mix(corners[2], corners[3], f.x), f.y),
                   mix(mix(corners[4], corners[5], f.x),
                       mix(corners[6], corners[7], f.x), f.y), f.z);
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
            volumeNanoRemap(vol, max(rawMax, 1e-6)) > max(vol._reserved[0], 0.0)) {
            return baseStep;
        }
    } else {
        if (pnanovdb_readaccessor_is_active(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk)) {
            return baseStep;
        }
        boxMin = floor(index / float(dim)) * float(dim);
        boxMax = boxMin + vec3(float(dim));
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
