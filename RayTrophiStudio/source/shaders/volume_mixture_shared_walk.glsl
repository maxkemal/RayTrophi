#ifndef VOLUME_MIXTURE_SHARED_WALK_GLSL
#define VOLUME_MIXTURE_SHARED_WALK_GLSL

// Each field owns its temperature buffer, so equal contents cannot be proven and
// a read is never shared between them. What is shared is the straight ray: one
// transform and a live accessor, instead of a tree descent per corner per step.
VkVolumeInstance mfSharedVolumeA;
VkVolumeInstance mfSharedVolumeB;
// Leaf last proven to hold density. Its emptiness query has one answer for every
// step inside it, and a march crosses a leaf in many steps.
ivec3 mfOccupiedLeaf;
bool mfOccupiedLeafValid;
bool mfTemperaturePreparedA;
bool mfTemperaturePreparedB;
VolumeNanoRay mfTemperatureRayA;
VolumeNanoRay mfTemperatureRayB;
pnanovdb_buf_t mfTemperatureBufferA;
pnanovdb_buf_t mfTemperatureBufferB;
pnanovdb_readaccessor_t mfTemperatureAccessorA;
pnanovdb_readaccessor_t mfTemperatureAccessorB;

void mfPrepareTemperature(VkVolumeInstance v, vec3 origin, vec3 direction,
    out bool prepared, out VolumeNanoRay ray, out pnanovdb_buf_t fieldBuffer,
    out pnanovdb_readaccessor_t accessor) {
    pnanovdb_map_handle_t fieldMap;
    mfInitializeNano(v.vdb_temp_address, fieldBuffer, fieldMap, accessor);
    prepared = v.emission_mode >= 2 && v.vdb_temp_address != uint64_t(0);
    if (prepared) {
        ray = volumeNanoRay(v, origin, direction, fieldBuffer, fieldMap);
    }
}

// Only use one cell traversal when both prepared fields have exactly the same
// camera coordinates and support. Never infer alignment from overlapping boxes.
// Independent materials remain independent; only geometric work is shared.
void mfConfigureSharedWalk(vec3 origin, vec3 direction) {
    mfSharedWalk = false;
    mfSharedDensityReady = false;
    if (mfWalking || mfCacheIndexA == 0xffffffffu || mfCacheIndexB == 0xffffffffu) {
        return;
    }
    uint members = 0u;
    for (uint i = 0u; i < mfCount; ++i) {
        if (gasOverlapCandidate(volumes.v[i])) {
            members++;
        }
    }
    if (members != 2u) {
        return;
    }
    VkVolumeInstance a = volumes.v[mfCacheIndexA];
    VkVolumeInstance b = volumes.v[mfCacheIndexB];
    if ((a.emission_mode > 0 && volumeEmissionIndependent(a._retired_cloud[0])) ||
        (b.emission_mode > 0 && volumeEmissionIndependent(b._retired_cloud[0]))) return;
    if (a.source_type != 6 || b.source_type != 6 ||
        !volumeNanoIndexEligible(a, mfCacheBufferA) ||
        !volumeNanoIndexEligible(b, mfCacheBufferB) ||
        a.density_curve[9] < 0.5 || b.density_curve[9] < 0.5 ||
        any(notEqual(a.aabb_min, b.aabb_min)) ||
        any(notEqual(a.aabb_max, b.aabb_max)) ||
        any(notEqual(mfCameraIntervalA, mfCameraIntervalB)) ||
        any(notEqual(mfCameraRayA.localOrigin, mfCameraRayB.localOrigin)) ||
        any(notEqual(mfCameraRayA.localDirection, mfCameraRayB.localDirection)) ||
        any(notEqual(mfCameraRayA.indexOrigin, mfCameraRayB.indexOrigin)) ||
        any(notEqual(mfCameraRayA.indexDirection, mfCameraRayB.indexDirection))) {
        return;
    }
    mfSharedExtinction = vec2(
        max(a.scatter_coefficient, 0.0) + max(a.absorption_coefficient, 0.0),
        max(b.scatter_coefficient, 0.0) + max(b.absorption_coefficient, 0.0));
    mfSharedVolumeA = a;
    mfSharedVolumeB = b;
    mfOccupiedLeafValid = false;
    mfSharedUnitA = mfExtinctionTerms(mfCacheIndexA, mfReadOpticalResponse(mfCacheIndexA), 1.0);
    mfSharedUnitB = mfExtinctionTerms(mfCacheIndexB, mfReadOpticalResponse(mfCacheIndexB), 1.0);
    mfPrepareTemperature(a, origin, direction, mfTemperaturePreparedA,
        mfTemperatureRayA, mfTemperatureBufferA, mfTemperatureAccessorA);
    mfPrepareTemperature(b, origin, direction, mfTemperaturePreparedB,
        mfTemperatureRayB, mfTemperatureBufferB, mfTemperatureAccessorB);
    mfSharedWalk = true;
}

float mfSharedCameraSpan(float t, float remaining, float baseStep) {
    float span = min(remaining, baseStep * 4.0);
    float progress = max(baseStep * 1e-5, abs(t) * 2e-7);
    if (mfCameraIntervalA.y <= t) {
        return min(remaining, max(span, progress));
    }
    if (mfCameraIntervalA.x > t) {
        return min(remaining, max(min(span, mfCameraIntervalA.x - t), progress));
    }
    // One coordinate transform, integer cell and exit for all aligned fields.
    vec3 point = mfCameraRayA.indexOrigin + mfCameraRayA.indexDirection * t;
    ivec3 cell = mfForwardCell(point, mfCameraRayA.indexDirection);
    vec2 boundA, boundB;
    if (!mfAdaptiveCellBounds(mfCacheIndexA, cell, mfCameraRayA.indexDirection,
            mfCacheBufferA, mfCacheAccessorA, boundA) ||
        !mfAdaptiveCellBounds(mfCacheIndexB, cell, mfCameraRayA.indexDirection,
            mfCacheBufferB, mfCacheAccessorB, boundB)) {
        return min(remaining, baseStep);
    }
    span = min(span, mfCellExit(point, mfCameraRayA.indexDirection, cell));
    span = min(span, mfCameraIntervalA.y - t);
    float slope = 0.0;
    slope += boundA.x * mfSharedExtinction.x;
    slope += boundB.x * mfSharedExtinction.y;
    if ((mfSharedVolumeA.emission_mode != 0 && volumeNanoSupportRemap(mfSharedVolumeA, boundA.y) >
            max(mfSharedVolumeA._reserved[0], 0.0)) ||
        (mfSharedVolumeB.emission_mode != 0 && volumeNanoSupportRemap(mfSharedVolumeB, boundB.y) >
            max(mfSharedVolumeB._reserved[0], 0.0))) {
        span = min(span, baseStep);
    }
    float opticalSpan = slope > 1e-12 ? sqrt(0.04 / slope) : baseStep * 4.0;
    opticalSpan = max(opticalSpan, baseStep * 0.25);
    return min(remaining, mfRefinedSpan(span, opticalSpan, baseStep, progress));
}

float mfSharedRaw(vec4 low, vec4 high, vec3 f) {
    return mix(mix(mix(low.x, low.y, f.x), mix(low.z, low.w, f.x), f.y),
        mix(mix(high.x, high.y, f.x), mix(high.z, high.w, f.x), f.y), f.z);
}

// Also the ordinary march's temperature read: same ray, same interpolation.
float mfPreparedTemperature(VolumeNanoRay ray, pnanovdb_buf_t fieldBuffer,
    inout pnanovdb_readaccessor_t accessor, float t) {
    vec3 point = ray.indexOrigin + ray.indexDirection * t;
    vec3 cell = floor(point);
    vec4 low, high;
    volumeNanoCorners(fieldBuffer, ivec3(cell), accessor, low, high);
    return mfSharedRaw(low, high, point - cell);
}

// A sample with density is inside the box: sampleTemperature's bounds test and
// its grid read reduce to this interpolation along the prepared ray.
float mfSharedTemperature(uint index, vec3 position) {
    if (index == mfCacheIndexA && mfTemperaturePreparedA) {
        return mfPreparedTemperature(mfTemperatureRayA, mfTemperatureBufferA,
            mfTemperatureAccessorA, mfCameraT);
    }
    if (index == mfCacheIndexB && mfTemperaturePreparedB) {
        return mfPreparedTemperature(mfTemperatureRayB, mfTemperatureBufferB,
            mfTemperatureAccessorB, mfCameraT);
    }
    return sampleTemperature(volumes.v[index], position);
}

float mfSharedEmptyStep(float baseStep) {
    float lo = mfCameraIntervalA.x - mfCameraT;
    float hi = mfCameraIntervalA.y - mfCameraT;
    if (hi <= 0.0) {
        return 1e30;
    }
    if (lo > 0.0) {
        return max(lo, baseStep);
    }
    vec3 point = mfCameraRayA.indexOrigin + mfCameraRayA.indexDirection * mfCameraT;
    ivec3 leaf = ivec3(floor(point / 8.0));
    if (mfOccupiedLeafValid && all(equal(leaf, mfOccupiedLeaf))) {
        return baseStep;
    }
    uint kind = 0u;
    float spanA = nanoEmptyIndexStep(mfSharedVolumeA, point,
        mfCameraRayA.indexDirection, baseStep, mfCacheBufferA, mfCacheAccessorA, kind);
    if (kind == 3u) {
        mfOccupiedLeaf = leaf;
        mfOccupiedLeafValid = true;
    }
    float safe = min(spanA, hi);
    if (safe <= baseStep) {
        return baseStep;
    }
    // Both fields must prove emptiness. One empty source never hides another.
    float spanB = nanoEmptyIndexStep(mfSharedVolumeB, point,
        mfCameraRayA.indexDirection, baseStep, mfCacheBufferB, mfCacheAccessorB, kind);
    if (kind == 3u) {
        mfOccupiedLeaf = leaf;
        mfOccupiedLeafValid = true;
    }
    return max(min(safe, min(spanB, hi)), baseStep);
}

void mfPrepareSharedDensity(float t) {
    mfSharedDensityReady = false;
    if (!mfSharedWalk) {
        return;
    }
    vec3 local = mfCameraRayA.localOrigin + mfCameraRayA.localDirection * t;
    mfSharedDensity = vec2(0.0);
    if (!any(lessThan(local, mfSharedVolumeA.aabb_min)) &&
        !any(greaterThan(local, mfSharedVolumeA.aabb_max))) {
        vec3 point = mfCameraRayA.indexOrigin + mfCameraRayA.indexDirection * t;
        vec3 cell = floor(point);
        vec3 f = point - cell;
        vec4 low, high;
        // Retain each grid's own values and remap. In particular, summing raw
        // density or averaging temperature before material evaluation is wrong.
        if (mfSharedVolumeA.density_multiplier != 0.0) {
            volumeNanoCorners(mfCacheBufferA, ivec3(cell), mfCacheAccessorA, low, high);
            mfSharedDensity.x = volumeNanoFinishDensity(mfSharedVolumeA, mfSharedRaw(low, high, f));
        }
        if (mfSharedVolumeB.density_multiplier != 0.0) {
            volumeNanoCorners(mfCacheBufferB, ivec3(cell), mfCacheAccessorB, low, high);
            mfSharedDensity.y = volumeNanoFinishDensity(mfSharedVolumeB, mfSharedRaw(low, high, f));
        }
    }
    mfSharedDensityReady = true;
}

#endif
