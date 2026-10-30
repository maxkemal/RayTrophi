#ifndef VOLUME_MIXTURE_LIGHT_CACHE_GLSL
#define VOLUME_MIXTURE_LIGHT_CACHE_GLSL

// Invocation-local eligibility. The persistent cache is the existing canonical
// per-volume density-integral cache, never a merged density/temperature field.
bool mfDirectLightCacheEligible = false;
bool mfDirectLightPositional = false;

void mfPrepareDirectLightCache() {
    mfDirectLightCacheEligible = false;
    if (mfWalking || cam.lightCount == 0u) {
        return;
    }
    uint channel;
    vec3 geometry;
    if (!volumeLightCacheChannel(mfLightIndex, channel, mfDirectLightPositional, geometry)) {
        return;
    }
    for (uint index = 0u; index < mfCount; ++index) {
        VkVolumeInstance v = volumes.v[index];
        if (!gasOverlapCandidate(v) || v.shadow_steps <= 0 || v.shadow_strength <= 0.0) {
            continue;
        }
        // Graphs can change extinction independently of sampled density. Noise,
        // additional unprepared fields and random walk keep the exact old path.
        bool prepared = index == mfCacheIndexA
            ? volumeNanoIndexEligible(v, mfCacheBufferA)
            : (index == mfCacheIndexB && volumeNanoIndexEligible(v, mfCacheBufferB));
        VlcSlot slot;
        if (!prepared || !volumeLightCacheSlot(index, slot)) {
            return;
        }
    }
    mfDirectLightCacheEligible = true;
}

bool mfTryDirectLightCache(vec3 position, vec3 direction, float distance,
    out vec3 visibility) {
    visibility = vec3(1.0);
    if (!mfDirectLightCacheEligible) {
        return false;
    }
    for (uint index = 0u; index < mfCount; ++index) {
        VkVolumeInstance v = volumes.v[index];
        if (!gasOverlapCandidate(v) || v.shadow_steps <= 0 || v.shadow_strength <= 0.0) {
            continue;
        }
        float lo, hi;
        if (!volumeRayInterval(v, position, direction, lo, hi)) {
            continue;
        }
        lo = max(lo, 0.0);
        if (min(hi, distance) <= lo) {
            continue;
        }
        // Directional cache chords extend to the box exit. A finite clipped
        // directional query is not represented by that integral.
        if (!mfDirectLightPositional && hi > distance) {
            return false;
        }
        vec3 entry = position + direction * lo;
        float integral;
        bool found;
        if (index == mfCacheIndexA) {
            found = volumeLightCacheFetch(index, v, mfLightIndex, entry,
                mfCacheBufferA, mfCacheMapA, mfCacheAccessorA, integral);
        } else {
            found = volumeLightCacheFetch(index, v, mfLightIndex, entry,
                mfCacheBufferB, mfCacheMapB, mfCacheAccessorB, integral);
        }
        if (!found) {
            return false;
        }
        // Do not use volumeShadowFromIntegral: that helper collapses absorption
        // to scalar luminance. Mixtures keep each contributor's RGB extinction
        // and authored strength, then multiply the independent transmissions.
        MfTerms unit = mfSharedWalk
            ? (index == mfCacheIndexA ? mfSharedUnitA : mfSharedUnitB)
            : mfExtinctionTerms(index, mfReadOpticalResponse(index), 1.0);
        vec3 physical = exp(-max(integral, 0.0) * unit.extinction);
        float strength = clamp(v.shadow_strength * 0.92, 0.0, 1.0);
        visibility *= mix(vec3(1.0), physical, strength);
    }
    return true;
}

vec3 mfDirectLightShadow(vec3 position, vec3 direction, float distance) {
    vec3 visibility;
    if (mfTryDirectLightCache(position, direction, distance, visibility)) {
        return visibility;
    }
    // Never use a partial aggregate when any contributor misses. The cache's
    // existing per-invocation fill budget bounds cold-cache work.
    return mfShadow(position, direction, distance);
}

#endif
