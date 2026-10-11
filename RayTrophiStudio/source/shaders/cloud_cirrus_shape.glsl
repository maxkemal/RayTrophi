// Continuous, softly broken ice wisps. Shared by Vulkan RT and RayFusion.
#ifndef CLOUD_CIRRUS_SHAPE_GLSL
#define CLOUD_CIRRUS_SHAPE_GLSL

float cloudCirrusWisps(vec4 broad, vec4 detail) {
    float coverage = clamp(cloudParams.cirrus.z, 0.0, 1.0);
    if (coverage <= 0.0) return 0.0;
    // The gradient channel provides soft open-ended patches. A cell union or
    // ellipse would turn these into recognisable leaf/stamp silhouettes.
    float field = broad.a;
    float threshold = (1.0 - coverage) * 0.65;
    float cloudMask = smoothstep(threshold - 0.18, threshold + 0.22, field);
    // Stretched continuous noise supplies filaments without periodic ribs,
    // ridged/absolute-value contours or hard thresholded white stripes.
    float strand = smoothstep(0.15, 0.85, detail.a);
    float fibres = strand * strand;
    float breakup = mix(0.35, 1.0, smoothstep(0.15, 0.75, broad.a));
    float veil = smoothstep(0.65, 1.0, coverage) * 0.08 * field;
    return clamp(coverage * cloudMask * (fibres * breakup + veil), 0.0, 1.0);
}

#endif
