#ifndef RT_VOLUME_DENSITY_CHANNELS_GLSL
#define RT_VOLUME_DENSITY_CHANNELS_GLSL

#include "volume_density_curve.glsl"
#include "../include/VolumeEmissionTransport.h"

float volumeDensityCutoff(float density, float cutoff) {
    if (density <= cutoff) return 0.0;
    float fade = cutoff > 0.0 ? smoothstep(cutoff, cutoff * 2.0, density) : 1.0;
    return density * fade;
}

// Both channels share ONE field lookup. The smoke curve shapes extinction;
// independent emission and its mask retain the pre-curve density differences.
// Remap, noise and cutoff remain shared authoring controls.
vec2 volumeDensityChannels(float normalized, float curve[12], float cutoff,
    float multiplier, float flags) {
    float optical = volumeDensityCutoff(volumeDensityCurve(normalized, curve), cutoff);
    float source = volumeEmissionIndependent(flags)
        ? volumeDensityCutoff(normalized, cutoff) : optical;
    return vec2(optical, source) * multiplier;
}

float volumeDensitySupport(float normalized, float curve[12], float flags) {
    float optical = volumeDensityCurve(normalized, curve);
    return volumeEmissionIndependent(flags) ? max(optical, normalized) : optical;
}

#endif
