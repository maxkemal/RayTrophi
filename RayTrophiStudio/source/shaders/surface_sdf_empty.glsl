#ifndef SURFACE_SDF_EMPTY_GLSL
#define SURFACE_SDF_EMPTY_GLSL

struct SurfaceSdfRegionCache {
    ivec3 base;
    uint width; // zero = uninitialized
    float upper;
};

// A surface proxy is NOT an optical density. Never use fog remap, cutoff or
// multiplier here. Prove the complete trilinear footprint remains below iso.
// Existing pore/alpha modifiers only subtract, so this exterior upper bound
// remains conservative even when either modifier is enabled. Interior exits
// retain the original sampler: modifiers can open holes inside a full cell.
int surfaceSdfEmptySteps(VkVolumeInstance v, VolumeNanoRay ray, float t,
    float stepSize, float endT, int budget, bool startedInside, float previous,
    pnanovdb_buf_t buf, inout pnanovdb_readaccessor_t acc,
    inout SurfaceSdfRegionCache cache, inout uint queries) {
    if (startedInside || !(previous < 0.5) || v.source_type != 4 ||
        v.volume_type != 2 || buf.address == uint64_t(0) || budget < 2 ||
        !(stepSize > 0.0)) return 0;
    vec3 local = ray.localOrigin + ray.localDirection * t;
    if (any(lessThan(local, v.aabb_min)) || any(greaterThan(local, v.aabb_max))) return 0;
    vec3 index = ray.indexOrigin + ray.indexDirection * t;
    pnanovdb_coord_t ijk;
    ijk.x = int(floor(index.x));
    ijk.y = int(floor(index.y));
    ijk.z = int(floor(index.z));
    bool cached = cache.width > 0u &&
        all(greaterThanEqual(index, vec3(cache.base))) &&
        all(lessThan(index, vec3(cache.base) + vec3(float(cache.width))));
    if (!cached) {
        queries++;
        uint dim = pnanovdb_readaccessor_get_dim(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk);
        cache.width = dim <= 1u ? 8u : dim;
        cache.base = ivec3(floor(index / float(cache.width)) * float(cache.width));
        cache.upper = 1e30; // failed/unknown bounds retain normal stepping
        if (dim <= 1u) {
            // Imported VDB statistics may exclude nonzero inactive values. Only
            // the live sparse producer certifies their <=1e-6 upper bound.
            if (v._uvw_pad[0] < 0.5) return 0;
            pnanovdb_grid_handle_t grid;
            grid.address.byte_offset = 0u;
            if ((pnanovdb_grid_get_flags(buf, grid) &
                    uint(PNANOVDB_GRID_FLAGS_HAS_MIN_MAX)) == 0u) return 0;
            float bound = pnanovdb_read_float(buf, pnanovdb_leaf_get_max_address(
                PNANOVDB_GRID_TYPE_FLOAT, buf, acc.leaf));
            if (isnan(bound) || isinf(bound)) return 0;
            cache.upper = max(bound, 1e-6);
        } else {
            // Constant tile/background: topology activity is irrelevant.
            cache.upper = pnanovdb_read_float(buf, pnanovdb_readaccessor_get_value_address(
                PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk));
        }
    }
    vec3 boxMin = vec3(cache.base);
    vec3 boxMax = boxMin + vec3(float(cache.width) - 1.0);
    if (isnan(cache.upper) || isinf(cache.upper) || !(cache.upper < 0.5 - 1e-5) ||
        any(greaterThanEqual(index, boxMax))) return 0;
    vec3 boundary = mix(boxMin, boxMax, greaterThan(ray.indexDirection, vec3(0.0)));
    float span = endT - t;
    for (int axis = 0; axis < 3; ++axis) {
        if (abs(ray.indexDirection[axis]) > 1e-8) {
            span = min(span, (boundary[axis] - index[axis]) / ray.indexDirection[axis]);
        }
    }
    // Advance by whole ORIGINAL sample intervals, leaving a rounding guard.
    // The caller samples the endpoint to preserve its next crossing bracket.
    span -= max(1e-6, abs(span) * 1e-5);
    return int(clamp(floor(span / stepSize), 0.0, float(budget)));
}

#endif
