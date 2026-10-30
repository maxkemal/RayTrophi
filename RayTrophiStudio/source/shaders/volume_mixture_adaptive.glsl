#ifndef VOLUME_MIXTURE_ADAPTIVE_GLSL
#define VOLUME_MIXTURE_ADAPTIVE_GLSL

// Invocation-local policy. Nothing is retained after a ray or a material edit.
// density_curve[9] transports the existing quality.adaptive_stepping switch;
// curve knots occupy [0..8], so the shared 672-byte record does not grow.
const int MF_ADAPTIVE_LIMIT = 8192;
bool mfStepBudgetHit;

// Largest d(density)/d(raw) over this cell's raw value range. A trilinear value
// stays inside its corner range and the curve is monotone, so only the segments
// that range touches can steepen the field. The whole-curve maximum made a
// saturated, flat interior refine as if it sat on the curve's steepest segment.
float mfCellCurveSlope(uint index, float rawLow, float rawHigh) {
    float remapLow = volumes.v[index].density_remap_low;
    float remapRange = max(volumes.v[index].density_remap_high - remapLow, EPSILON);
    float low = max((rawLow - remapLow) / remapRange, 0.0);
    float high = max((rawHigh - remapLow) / remapRange, 0.0);
    float cutoff = max(volumes.v[index]._reserved[0], 0.0);
    float densityLow = volumeDensityCurve(low, volumes.v[index].density_curve);
    float densityHigh = volumeDensityCurve(high, volumes.v[index].density_curve);
    if (densityHigh <= cutoff) {
        return 0.0;
    }
    float slope = 1.0;
    if (volumes.v[index].density_curve[0] > 0.5) {
        slope = high >= 1.0 ? max(volumes.v[index].density_curve[8], 0.0) : 0.0;
        if (low < 1.0) {
            int last = min(int(high * 7.0), 6);
            for (int segment = min(int(low * 7.0), 6); segment <= last; ++segment) {
                slope = max(slope, 7.0 * abs(volumes.v[index].density_curve[segment + 2] -
                    volumes.v[index].density_curve[segment + 1]));
            }
        }
    }
    // The cutoff fade d*smoothstep(c,2c,d) has derivative at most four, and
    // exactly one above its top end.
    bool fading = cutoff > 0.0 && densityLow < cutoff * 2.0;
    return slope * (fading ? 4.0 : 1.0) * abs(volumes.v[index].density_multiplier) / remapRange;
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

bool mfAdaptiveCellBounds(uint index, ivec3 cell, vec3 direction,
    pnanovdb_buf_t fieldBuffer, inout pnanovdb_readaccessor_t accessor, out vec2 bound) {
    if (mfLoadCellBounds(index, cell, bound)) return true;
    vec4 low, high;
    volumeNanoCorners(fieldBuffer, cell, accessor, low, high);
    if (any(isnan(low)) || any(isnan(high)) || any(isinf(low)) || any(isinf(high))) {
        return false;
    }
    bound.y = max(max(max(low.x, low.y), max(low.z, low.w)),
        max(max(high.x, high.y), max(high.z, high.w)));
    float lowest = min(min(min(low.x, low.y), min(low.z, low.w)),
        min(min(high.x, high.y), min(high.z, high.w)));
    bound.x = mfRawSlope(low, high, direction) * mfCellCurveSlope(index, lowest, bound.y);
    mfStoreCellBounds(index, cell, bound);
    return true;
}

// The cell edge limits how far the corner bound is proven, not how fine the
// authored lattice is: clipping to it left many sliver steps below the spacing
// a single gas uses. Only the optical bound may refine past the authored step.
float mfRefinedSpan(float cellSpan, float opticalSpan, float baseStep, float progress) {
    return max(max(min(cellSpan, opticalSpan), min(baseStep, opticalSpan)), progress);
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
        // The straight camera segment prepared these affine rays and box
        // intervals once. A random-walk flight has a new origin/direction and
        // must compute its own interval and map instead.
        if (!mfWalking && index == mfCacheIndexA) {
            lo = mfCameraIntervalA.x;
            hi = mfCameraIntervalA.y;
        } else if (!mfWalking && index == mfCacheIndexB) {
            lo = mfCameraIntervalB.x;
            hi = mfCameraIntervalB.y;
        } else if (!volumeRayInterval(v, origin, direction, lo, hi)) {
            continue;
        }
        if (hi <= t) continue;
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
        vec2 bound;
        if (index == mfCacheIndexA) {
            ray = mfCameraRayA;
            if (mfWalking) {
                ray = volumeNanoRay(v, origin, direction, mfCacheBufferA, mfCacheMapA);
            }
            vec3 point = ray.indexOrigin + ray.indexDirection * t;
            ivec3 cell = mfForwardCell(point, ray.indexDirection);
            if (!mfAdaptiveCellBounds(index, cell, ray.indexDirection,
                    mfCacheBufferA, mfCacheAccessorA, bound)) {
                return min(remaining, baseStep);
            }
            span = min(span, mfCellExit(point, ray.indexDirection, cell));
        } else if (index == mfCacheIndexB) {
            ray = mfCameraRayB;
            if (mfWalking) {
                ray = volumeNanoRay(v, origin, direction, mfCacheBufferB, mfCacheMapB);
            }
            vec3 point = ray.indexOrigin + ray.indexDirection * t;
            ivec3 cell = mfForwardCell(point, ray.indexDirection);
            if (!mfAdaptiveCellBounds(index, cell, ray.indexDirection,
                    mfCacheBufferB, mfCacheAccessorB, bound)) {
                return min(remaining, baseStep);
            }
            span = min(span, mfCellExit(point, ray.indexDirection, cell));
        } else {
            mfInitializeNano(v.volume_type == 2 ? v.vdb_grid_address : uint64_t(0),
                fieldBuffer, fieldMap, fieldAccessor);
            ray = volumeNanoRay(v, origin, direction, fieldBuffer, fieldMap);
            vec3 point = ray.indexOrigin + ray.indexDirection * t;
            ivec3 cell = mfForwardCell(point, ray.indexDirection);
            if (!mfAdaptiveCellBounds(index, cell, ray.indexDirection,
                    fieldBuffer, fieldAccessor, bound)) {
                return min(remaining, baseStep);
            }
            span = min(span, mfCellExit(point, ray.indexDirection, cell));
        }
        span = min(span, hi - t);
        float scatter = max(v.scatter_coefficient, 0.0);
        float meanG = mix(v.scatter_anisotropy_back, v.scatter_anisotropy, v.scatter_lobe_mix);
        if (v.random_walk_enabled > 0.5 && meanG > 0.05 &&
            mfEventOrdinal >= uint(clamp(int(v.random_walk_exact_events), 1, 512))) {
            scatter *= 1.0 - meanG;
        }
        slope += bound.x * (scatter + max(v.absorption_coefficient, 0.0));
        // Temperature/ramp emission is an independent field. Do not coarsen it
        // based only on a smooth density field; keep its authored resolution.
        bool densityPresent = volumeNanoRemap(v, bound.y) > max(v._reserved[0], 0.0);
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
    return min(remaining, mfRefinedSpan(span, opticalSpan, baseStep, progress));
}

float mfBudgetedSpan(vec3 origin, vec3 direction, float t, float remaining,
    float baseStep, int iteration) {
    if (iteration == MF_ADAPTIVE_LIMIT - 1) {
        // Preserve complete geometric coverage even at the hard work limit.
        // Report the quality fallback instead of claiming an adaptive bound.
        mfStepBudgetHit = true;
        return remaining;
    }
    if (mfSharedWalk) {
        return mfSharedCameraSpan(t, remaining, baseStep);
    }
    return mfAdaptiveSpan(origin, direction, t, remaining, baseStep);
}
// All members are evaluated jointly at each camera sample, so the smallest
// authored work ceiling governs an overlap. Unrelated/off-ray slots do not.
int mfCameraStepBudget(vec3 origin, vec3 direction, float nearT, float farT) {
    int budget = 2048;
    for (uint i = 0u; i < mfCount; ++i) {
        VkVolumeInstance v = volumes.v[i];
        if (!gasOverlapCandidate(v)) continue;
        float lo, hi;
        if (!volumeRayInterval(v, origin, direction, lo, hi) ||
            hi <= nearT || lo >= farT) continue;
        budget = min(budget, clamp(v.max_steps, 1, 2048));
    }
    return budget;
}

float mfCameraBudgetedSpan(vec3 origin, vec3 direction, float t, float remaining,
    float baseStep, int iteration, int samplesLeft) {
    float requested = mfBudgetedSpan(origin, direction, t, remaining, baseStep, iteration);
    // Budget overrides the local refinement target. Reserve enough coverage
    // for every remaining sample; the last sample integrates the full tail.
    float coverageFloor = remaining / float(max(samplesLeft, 1));
    if (requested < coverageFloor) mfStepBudgetHit = true;
    return min(remaining, max(requested, coverageFloor));
}

#endif
