// Shared packed PBR texture policy.
// Default ORM-style packed layout:
//   G -> roughness
//   B -> metallic
// When a texture has been baked to BC4 single-channel cache, the data lives
// in .r and the C++ side sets a per-material flag bit so the shader knows
// to read .r instead of .g/.b.
//
// Flag bits (must match C++ side in VulkanBackend.cpp uploadMaterials):
//   bit 9  -> roughness texture is in .r (BC4 cache or grayscale R8 upload)
//   bit 10 -> metallic  texture is in .r (BC4 cache or grayscale R8 upload)
//   bit 11 -> normal map is BC5-encoded (RG only, Z must be reconstructed)

const uint VK_MAT_FLAG_ROUGHNESS_IN_R = (1u << 9);
const uint VK_MAT_FLAG_METALLIC_IN_R  = (1u << 10);
const uint VK_MAT_FLAG_NORMAL_BC5     = (1u << 11);
// Bits 12-13 (metallic) / 14-15 (roughness): explicit USER channel override —
// 0 = Auto (policy above), 1 = R, 2 = G, 3 = B. Set from the material UI for
// non-ORM packings (RMA/MRA, DirectX metal-in-R maps), which under Auto read
// the wrong channel and shade the whole surface as metal.

float pbrSelectChannel(vec4 texel, uint sel) {
    return (sel == 1u) ? texel.r : ((sel == 2u) ? texel.g : texel.b);
}

// Decode an encoded normal map texel ([0,1] range) to a tangent-space vector.
// For BC5-compressed normals only RG carry data and B is always 0; Z must be
// derived from the unit-length constraint instead of being read from .b.
vec3 decodeNormalMapSample(vec3 nmSample, uint matFlags) {
    if ((matFlags & VK_MAT_FLAG_NORMAL_BC5) != 0u) {
        vec2 xy = nmSample.xy * 2.0 - 1.0;
        float z = sqrt(max(0.0, 1.0 - dot(xy, xy)));
        return vec3(xy, z);
    }
    return nmSample * 2.0 - 1.0;
}

float samplePackedRoughness(vec4 texel, float fallbackValue, uint matFlags) {
    uint sel = (matFlags >> 14) & 3u;
    float v = (sel != 0u) ? pbrSelectChannel(texel, sel)
            : (((matFlags & VK_MAT_FLAG_ROUGHNESS_IN_R) != 0u) ? texel.r : texel.g);
    return clamp(v, fallbackValue, 1.0);
}

float samplePackedMetallic(vec4 texel, uint matFlags) {
    uint sel = (matFlags >> 12) & 3u;
    float v = (sel != 0u) ? pbrSelectChannel(texel, sel)
            : (((matFlags & VK_MAT_FLAG_METALLIC_IN_R) != 0u) ? texel.r : texel.b);
    return clamp(v, 0.0, 1.0);
}
