#ifndef RAYTROPHI_VOLUME_TEMPORAL_GLSL
#define RAYTROPHI_VOLUME_TEMPORAL_GLSL

const uint VOLUME_TEMPORAL_ACCEPTED               = 0u;
const uint VOLUME_TEMPORAL_REJECT_NO_HISTORY     = 1u;
const uint VOLUME_TEMPORAL_REJECT_OUTSIDE_VIEW   = 2u;
const uint VOLUME_TEMPORAL_REJECT_POSITION       = 3u;
const uint VOLUME_TEMPORAL_REJECT_TRANSMITTANCE  = 4u;
const uint VOLUME_TEMPORAL_REJECT_CLASSIFICATION = 5u;

// Metadata: xyz = representative world position; w = transmittance.
// A negative w marks a pixel without a volume interaction.
bool volumeTemporalProject(
    vec3 worldPosition,
    mat4 previousViewProjection,
    ivec2 imageSize,
    out ivec2 previousPixel)
{
    vec4 clip = previousViewProjection * vec4(worldPosition, 1.0);
    if (clip.w <= 0.0) {
        return false;
    }

    vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) {
        return false;
    }

    previousPixel = clamp(ivec2(uv * vec2(imageSize)), ivec2(0), imageSize - 1);
    return true;
}

uint volumeTemporalValidate(
    vec4 previousMetadata,
    vec4 currentMetadata,
    float positionTolerance,
    float transmittanceTolerance)
{
    bool previousHasVolume = previousMetadata.w >= 0.0;
    bool currentHasVolume = currentMetadata.w >= 0.0;
    if (previousHasVolume != currentHasVolume) {
        return VOLUME_TEMPORAL_REJECT_CLASSIFICATION;
    }
    if (!currentHasVolume) {
        return VOLUME_TEMPORAL_ACCEPTED;
    }
    if (distance(previousMetadata.xyz, currentMetadata.xyz) > positionTolerance) {
        return VOLUME_TEMPORAL_REJECT_POSITION;
    }
    if (abs(previousMetadata.w - currentMetadata.w) > transmittanceTolerance) {
        return VOLUME_TEMPORAL_REJECT_TRANSMITTANCE;
    }
    return VOLUME_TEMPORAL_ACCEPTED;
}

vec3 volumeTemporalBlend(
    vec3 currentColor,
    vec3 historyColor,
    float historyLength,
    float maximumHistory,
    float variance)
{
    float stableHistory = clamp(historyLength, 0.0, max(maximumHistory, 1.0));
    float historyWeight = stableHistory / (stableHistory + 1.0);
    historyWeight *= 1.0 / (1.0 + max(variance, 0.0) * 8.0);
    return mix(currentColor, historyColor, clamp(historyWeight, 0.0, 0.95));
}

vec4 volumeTemporalLoadColor(uint slot, ivec2 pixel) {
    return (slot == 0u)
        ? imageLoad(volumeTemporalColor0, pixel)
        : imageLoad(volumeTemporalColor1, pixel);
}

vec4 volumeTemporalLoadMetadata(uint slot, ivec2 pixel) {
    return (slot == 0u)
        ? imageLoad(volumeTemporalMetadata0, pixel)
        : imageLoad(volumeTemporalMetadata1, pixel);
}

void volumeTemporalStore(uint slot, ivec2 pixel, vec4 color, vec4 metadata) {
    if (slot == 0u) {
        imageStore(volumeTemporalColor0, pixel, color);
        imageStore(volumeTemporalMetadata0, pixel, metadata);
    } else {
        imageStore(volumeTemporalColor1, pixel, color);
        imageStore(volumeTemporalMetadata1, pixel, metadata);
    }
}

#endif
