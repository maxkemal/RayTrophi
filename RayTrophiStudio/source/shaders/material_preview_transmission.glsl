#ifndef MATERIAL_PREVIEW_TRANSMISSION_GLSL
#define MATERIAL_PREVIEW_TRANSMISSION_GLSL

// Bounded raster counterparts of the Vulkan RT glass/resin surface contract.
// This module owns no descriptors and performs no scene lookup; the caller
// supplies the canonical world radiance so it can later be replaced by the
// scene-color/depth refraction pass without changing material semantics.

const uint PREVIEW_MAT_FLAG_BUBBLE = (1u << 19);

float previewDielectricF0(float ior) {
    float n = max(ior, 1.0001);
    float r = (n - 1.0) / (n + 1.0);
    return r * r;
}

float previewDielectricFresnel(float cosTheta, float ior) {
    float f0 = previewDielectricF0(ior);
    return f0 + (1.0 - f0) * pow(1.0 - clamp(cosTheta, 0.0, 1.0), 5.0);
}

vec3 previewBeerExtinction(vec3 tint) {
    vec3 clampedTint = clamp(tint, vec3(0.0), vec3(1.0));
    float tintMax = max(clampedTint.r, max(clampedTint.g, clampedTint.b));
    // Same coefficient model used by closesthit.rchit for resin/stone.
    return (vec3(1.0) - clampedTint) * 1.35 +
           vec3(0.22 * (1.0 - tintMax));
}

float previewHash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float previewValueNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = previewHash13(i + vec3(0,0,0));
    float n100 = previewHash13(i + vec3(1,0,0));
    float n010 = previewHash13(i + vec3(0,1,0));
    float n110 = previewHash13(i + vec3(1,1,0));
    float n001 = previewHash13(i + vec3(0,0,1));
    float n101 = previewHash13(i + vec3(1,0,1));
    float n011 = previewHash13(i + vec3(0,1,1));
    float n111 = previewHash13(i + vec3(1,1,1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

float previewFbm(vec3 p) {
    float value = 0.0;
    float amplitude = 0.55;
    for (int octave = 0; octave < 4; ++octave) {
        value += previewValueNoise(p) * amplitude;
        p = p * 2.03 + vec3(17.1, 9.2, 13.7);
        amplitude *= 0.48;
    }
    return value;
}

vec3 previewBubbleFilm(float opticalDepth) {
    float phase = opticalDepth * 6.2831853;
    return vec3(0.55 + 0.45 * cos(phase),
                0.55 + 0.45 * cos(phase + 2.0944),
                0.55 + 0.45 * cos(phase + 4.1888));
}

struct PreviewInteriorSample {
    vec3 transmittance;
    vec3 scatter;
};

PreviewInteriorSample previewResinInterior(
    vec3 anchor,
    vec3 direction,
    float thickness,
    vec3 extinction,
    float inclusion,
    float dirt,
    float shard,
    float scale,
    vec3 dustA,
    vec3 dustB,
    vec3 dirtColor,
    float shardHue,
    uint dustStyle,
    uint shardShape)
{
    PreviewInteriorSample result;
    float safeThickness = max(thickness, 0.0);
    float safeScale = max(scale, 0.01);
    vec3 middle = anchor + direction * (safeThickness * 0.5);
    vec3 dustPoint = middle * safeScale;
    float cloud;
    if (dustStyle == 2u) {
        // RT Wispy: horizontally stretched ridged filaments.
        float n = previewFbm(dustPoint * vec3(2.4, 0.55, 2.4));
        cloud = pow(1.0 - abs(2.0 * n - 1.0), 3.0);
    } else if (dustStyle == 3u) {
        // RT Paint Swirl: domain-warped fbm. Keep the same three offsets so
        // authored presets retain their characteristic flow direction.
        vec3 warp = vec3(previewFbm(dustPoint * 0.5),
                         previewFbm(dustPoint * 0.5 + vec3(19.7)),
                         previewFbm(dustPoint * 0.5 + vec3(47.3))) - 0.5;
        cloud = previewFbm(dustPoint + warp * 2.6);
    } else {
        // Nebula and Billow share RT's density field; they differ in colour.
        cloud = previewFbm(dustPoint) *
                (0.6 + 0.8 * previewValueNoise(dustPoint * 3.1));
    }
    float dustCover = smoothstep(0.48, 0.82, cloud) * clamp(inclusion, 0.0, 1.0);
    float dirtField = previewValueNoise(middle * safeScale * 3.7 + vec3(31.0));
    float dirtCover = smoothstep(0.92, 0.995, dirtField) * clamp(dirt, 0.0, 1.0);
    vec3 shardPoint = middle * safeScale * 1.9 + vec3(71.0);
    if (shardShape == 1u)
        shardPoint *= vec3(0.38, 1.0, 1.0); // crystal: elongated chip field
    float shardField = pow(max(previewValueNoise(shardPoint) - 0.62, 0.0), 2.0);
    float shardCover = clamp(shardField * 9.0 * shard, 0.0, 1.0);

    vec3 dustTint;
    if (dustStyle == 0u) {
        vec3 baseTint = max(dustA, vec3(0.0));
        dustTint = mix(baseTint, baseTint.gbr,
                       smoothstep(0.25, 0.75,
                           previewValueNoise(dustPoint * 0.6 + vec3(31.7))));
    } else {
        dustTint = mix(max(dustA, vec3(0.0)), max(dustB, vec3(0.0)),
                       smoothstep(0.30, 0.70, cloud));
    }
    vec3 shardTint = 0.55 + 0.45 * cos(6.2831853 *
        (vec3(0.0, 0.333333, 0.666667) + shardHue));
    result.transmittance = exp(-safeThickness *
        (max(extinction, vec3(0.0)) + vec3(dustCover * 1.8 + dirtCover * 5.0)));
    result.scatter = dustTint * dustCover * 0.35 +
                     max(dirtColor, vec3(0.0)) * dirtCover * 0.55 +
                     shardTint * shardCover * 0.45;
    return result;
}

#endif
