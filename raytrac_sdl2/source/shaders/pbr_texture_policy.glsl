// Shared packed PBR texture policy.
// Grayscale textures remain compatible, while packed ORM-style textures use:
//   G -> roughness
//   B -> metallic

float samplePackedRoughness(vec4 texel, float fallbackValue) {
    return clamp(texel.g, fallbackValue, 1.0);
}

float samplePackedMetallic(vec4 texel) {
    return clamp(texel.b, 0.0, 1.0);
}
