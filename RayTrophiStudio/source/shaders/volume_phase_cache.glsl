#ifndef VOLUME_PHASE_CACHE_GLSL
#define VOLUME_PHASE_CACHE_GLSL

struct VolumePhaseCache {
    vec3 values;
    bvec3 valid;
};

VolumePhaseCache volumePhaseCache() {
    return VolumePhaseCache(vec3(0.0), bvec3(false));
}

// Slots 0/1 belong to the ray's fixed scene-light pair; slot 2 is the sun.
// Only a fixed direction with an unmodified single-medium phase may reuse it.
// The density-dependent multiple-scattering/albedo blend remains per sample.
float volumeCachedPhase(inout VolumePhaseCache cache, int slot, bool canReuse,
    float cosine, float forwardG, float backwardG, float lobeMix) {
    if (canReuse && cache.valid[slot]) return cache.values[slot];
    float phase = dualLobeHG(cosine, forwardG, backwardG, lobeMix);
    if (canReuse) {
        cache.values[slot] = phase;
        cache.valid[slot] = true;
    }
    return phase;
}
#endif
