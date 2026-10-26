#ifndef VOLUME_MIXTURE_ADAPTIVE_GLSL
#define VOLUME_MIXTURE_ADAPTIVE_GLSL

// Invocation-local policy. Nothing is retained after a ray or a material edit.
// density_curve[9] transports the existing quality.adaptive_stepping switch;
// curve knots occupy [0..8], so the shared 672-byte record does not grow.
const int MF_ADAPTIVE_LIMIT = 8192;
bool mfStepBudgetHit;

float mfCurveSlope(VkVolumeInstance v) {
    float slope = 1.0;
    if (v.density_curve[0] > 0.5) {
        slope = max(v.density_curve[8], 0.0);
        for (int knot = 1; knot < 8; ++knot) {
            slope = max(slope, 7.0 * abs(v.density_curve[knot + 1] - v.density_curve[knot]));
        }
    }
    // The cutoff fade d*smoothstep(c,2c,d) has derivative at most four.
    return slope * (v._reserved[0] > 0.0 ? 4.0 : 1.0) *
        abs(v.density_multiplier) /
        max(v.density_remap_high - v.density_remap_low, EPSILON);
}

float mfRawSlope(vec4 low, vec4 high, vec3 direction) {
    // Trilinear partial derivatives are convex combinations of edge differences.
    float dx = max(max(abs(low.y - low.x), abs(low.w - low.z)),
        max(abs(high.y - high.x), abs(high.w - high.z)));
    float dy = max(max(abs(low.z - low.x), abs(low.w - low.y)),
        max(abs(high.z - high.x), abs(high.w - high.y)));
    float dz = max(max(abs(high.x - low.x), abs(high.y - low.y)),
        max(abs(high.z - low.z), abs(high.w - low.w)));
    return dot(vec3(dx, dy, dz), abs(direction));
}

ivec3 mfForwardCell(vec3 index, vec3 direction) {
    ivec3 cell = ivec3(floor(index));
    for (int axis = 0; axis < 3; ++axis) {
        if (direction[axis] < 0.0 && index[axis] == float(cell[axis])) {
            cell[axis]--;
        }
    }
    return cell;
}

float mfCellExit(vec3 index, vec3 direction, ivec3 cell) {
    float span = 1e30;
    for (int axis = 0; axis < 3; ++axis) {
        if (abs(direction[axis]) <= 1e-12) {
            continue;
        }
        float edge = float(cell[axis]) + (direction[axis] > 0.0 ? 1.0 : 0.0);
        span = min(span, max((edge - index[axis]) / direction[axis], 0.0));
    }
    return span;
}

// Bounds are from all eight corners, not just the previous density sample.
// A narrow contributor cannot be skipped because another volume looks empty.
float mfAdaptiveSpan(vec3 origin, vec3 direction, float t, float remaining,
    float baseStep) {
    float span = min(remaining, baseStep * 4.0);
    float slope = 0.0;
    bool member = false;
    for (uint index = 0u; index < mfCount; ++index) {
        VkVolumeInstance v = volumes.v[index];
        if (!gasOverlapCandidate(v)) {
            continue;
        }
        float lo, hi;
        if (!volumeRayInterval(v, origin, direction, lo, hi) || hi <= t) {
            continue;
        }
        if (lo > t) {
            span = min(span, lo - t);
            continue;
        }
        member = true;
        // Noise, graphs and non-NanoVDB fields have no usable corner bound.
        // Any fixed-policy member vetoes coarsening for the entire mixture.
        if (v.density_curve[9] < 0.5 || v.volume_type != 2 ||
            v.vdb_grid_address == uint64_t(0) || v._ext_reserved[7] >= 0.5 ||
            (v._reserved[1] > 0.5 &&
             matProgramOffset(uint(v._reserved[1] - 1.0)) != MATPROG_NONE)) {
            return min(remaining, baseStep);
        }
        pnanovdb_buf_t fieldBuffer;
        pnanovdb_map_handle_t fieldMap;
        pnanovdb_readaccessor_t fieldAccessor;
        VolumeNanoRay ray;
        vec4 low, high;
        if (index == mfCacheIndexA) {
            ray = volumeNanoRay(v, origin, direction, mfCacheBufferA, mfCacheMapA);
            vec3 point = ray.indexOrigin + ray.indexDirection * t;
            ivec3 cell = mfForwardCell(point, ray.indexDirection);
            volumeNanoCorners(mfCacheBufferA, cell, mfCacheAccessorA, low, high);
            span = min(span, mfCellExit(point, ray.indexDirection, cell));
        } else if (index == mfCacheIndexB) {
            ray = volumeNanoRay(v, origin, direction, mfCacheBufferB, mfCacheMapB);
            vec3 point = ray.indexOrigin + ray.indexDirection * t;
            ivec3 cell = mfForwardCell(point, ray.indexDirection);
            volumeNanoCorners(mfCacheBufferB, cell, mfCacheAccessorB, low, high);
            span = min(span, mfCellExit(point, ray.indexDirection, cell));
        } else {
            mfInitializeNano(v, fieldBuffer, fieldMap, fieldAccessor);
            ray = volumeNanoRay(v, origin, direction, fieldBuffer, fieldMap);
            vec3 point = ray.indexOrigin + ray.indexDirection * t;
            ivec3 cell = mfForwardCell(point, ray.indexDirection);
            volumeNanoCorners(fieldBuffer, cell, fieldAccessor, low, high);
            span = min(span, mfCellExit(point, ray.indexDirection, cell));
        }
        span = min(span, hi - t);
        if (any(isnan(low)) || any(isnan(high)) || any(isinf(low)) || any(isinf(high))) {
            return min(remaining, baseStep);
        }
        float scatter = max(v.scatter_coefficient, 0.0);
        float meanG = mix(v.scatter_anisotropy_back, v.scatter_anisotropy, v.scatter_lobe_mix);
        if (v.random_walk_enabled > 0.5 && meanG > 0.05 &&
            mfEventOrdinal >= uint(clamp(int(v.random_walk_exact_events), 1, 512))) {
            scatter *= 1.0 - meanG;
        }
        slope += mfRawSlope(low, high, ray.indexDirection) * mfCurveSlope(v) *
            (scatter + max(v.absorption_coefficient, 0.0));
        // Temperature/ramp emission is an independent field. Do not coarsen it
        // based only on a smooth density field; keep its authored resolution.
        float largestRaw = max(max(max(low.x, low.y), max(low.z, low.w)),
            max(max(high.x, high.y), max(high.z, high.w)));
        bool densityPresent = volumeNanoRemap(v, largestRaw) > max(v._reserved[0], 0.0);
        if (v.emission_mode != 0 && densityPresent) {
            span = min(span, baseStep);
        }
    }
    if (!member) {
        return min(remaining, max(span, max(baseStep * 1e-5, abs(t) * 2e-7)));
    }
    // Midpoint optical-depth error <= slope * length^2 / 4 inside this cell.
    // Local target 0.01; refinement is limited to 4x the authored resolution.
    // The safety floor is a sampling-quality bound, not an opacity threshold.
    float opticalSpan = slope > 1e-12 ? sqrt(0.04 / slope) : baseStep * 4.0;
    opticalSpan = max(opticalSpan, baseStep * 0.25);
    float progress = max(baseStep * 1e-5, abs(t) * 2e-7);
    return min(remaining, max(min(span, opticalSpan), progress));
}

float mfBudgetedSpan(vec3 origin, vec3 direction, float t, float remaining,
    float baseStep, int iteration) {
    if (iteration == MF_ADAPTIVE_LIMIT - 1) {
        // Preserve complete geometric coverage even at the hard work limit.
        // Report the quality fallback instead of claiming an adaptive bound.
        mfStepBudgetHit = true;
        return remaining;
    }
    return mfAdaptiveSpan(origin, direction, t, remaining, baseStep);
}
#endif
