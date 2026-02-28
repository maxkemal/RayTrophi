#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Push Constants (match other shaders)
layout(push_constant) uniform CameraPC {
    vec4 origin;
    vec4 lowerLeft;
    vec4 horizontal;
    vec4 vertical;
    uint frameCount;
    uint minSamples;
    uint lightCount;
    float varianceThreshold;
    uint maxSamples;
    float exposure_factor;
} cam;

// Payload structure (match closesthit)
struct RayPayload {
    vec3     radiance;
    vec3     attenuation;
    vec3     scatterOrigin;
    vec3     scatterDir;
    uint     seed;
    bool     scattered;
    bool     hitEmissive;
    uint     occluded;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadInEXT bool shadowOccluded;

// Material struct definition (must be before buffer declaration)
struct Material {
    // Block 1: Albedo + opacity
    float albedo_r, albedo_g, albedo_b, opacity;
    // Block 2: Emission + strength
    float emission_r, emission_g, emission_b, emission_strength;
    // Block 3: PBR properties
    float roughness, metallic, ior, transmission;
    // Block 4: Subsurface color + amount
    float subsurface_r, subsurface_g, subsurface_b, subsurface_amount;
    // Block 5: Subsurface radius + scale
    float subsurface_radius_r, subsurface_radius_g, subsurface_radius_b, subsurface_scale;
    // Block 6: Coatings & Translucency
    float clearcoat, clearcoat_roughness, translucent, subsurface_anisotropy;
    // Block 7: Additional properties
    float anisotropic, sheen, sheen_tint;
    uint flags;
    // Block 8: Water/Extra params
    float fft_amplitude, fft_time_scale, micro_detail_strength, micro_detail_scale;
    // Block 9: Extra water params
    float foam_threshold, fft_ocean_size, fft_choppiness, fft_wind_speed;
    // Block 10: Standard Textures (first 4)
    uint albedo_tex;
    uint normal_tex;
    uint roughness_tex;
    uint metallic_tex;
    // Block 11: Standard Textures (second 4)
    uint emission_tex;
    uint height_tex;
    uint opacity_tex;
    uint transmission_tex;
    // Block 12: Reserved
    uint _reserved_0, _reserved_1, _reserved_2, _reserved_3;
};

// Descriptor bindings (minimal, mirror main shader)
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 2, scalar) readonly buffer MaterialBuffer  { Material     m[]; } materials;
layout(set = 0, binding = 6) uniform sampler2D materialTextures[];

// Minimal geometric data (instance only, no per-vertex access)
struct VkInstanceData {
    uint materialIndex;
    uint blasIndex;
};

layout(set = 0, binding = 5, scalar) readonly buffer InstanceBuffer  { VkInstanceData  i[]; } instances;

// Hit attributes (barycentrics)
hitAttributeEXT vec2 baryCoord;

// World data (minimal) — must match miss.rmiss / C++ VkWorldDataSimple
struct WorldDataSimple {
    vec3 sunDir; float pad0;
    vec3 sunColor; float sunIntensity;
    float sunSize; float airDensity; float dustDensity; float mieAnisotropy;
    int  mode;
    int  envTexSlot;
    float envIntensity;
    float pad1;
};
layout(set = 0, binding = 7) readonly buffer WorldBuffer { WorldDataSimple w[]; } world;

void main() {
    // Get instance and material data
    VkInstanceData inst = instances.i[gl_InstanceID];
    uint matIndex = inst.materialIndex;
    
    Material mat = materials.m[matIndex];
    
    // Check opacity to determine if ray is occluded
    float finalOpacity = mat.opacity;
    
    // Sample opacity texture if present (assuming hitUV = 0.5)
    int opacityTexID = int(mat.opacity_tex);
    if (opacityTexID > 0) {
        vec2 hitUV = vec2(0.5, 0.5);
        float texOpacity = texture(materialTextures[nonuniformEXT(opacityTexID)], hitUV).r;
        finalOpacity *= texOpacity;
    }
    
    // Stochastic transparency for shadow rays
    // Probability of ray continuing (not blocked) = 1 - finalOpacity
    // This allows semi-transparent geometry (leaves, etc.) to transmit shadows
    // Generate deterministic pseudo-random value from hit position + ray direction
    uint seed = uint(gl_LaunchIDEXT.x) ^ uint(gl_LaunchIDEXT.y);
    seed ^= floatBitsToUint(gl_HitTEXT);
    seed = seed * 1103515245u + 12345u;  // Simple LCG
    float rand = float((seed >> 8u) & 0xFFFFFFu) / float(0x01000000u);
    
    // If random value > finalOpacity, ignore hit (ray continues)
    if (rand > finalOpacity) {
        // Treat as transparent - shadow ray continues
        return;
    }
    
    // Otherwise, ray is blocked (shadow)
    shadowOccluded = true;
}
