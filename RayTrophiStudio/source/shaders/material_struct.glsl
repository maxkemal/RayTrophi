// =============================================================================
// material_struct.glsl — SINGLE SOURCE OF TRUTH for the Vulkan PBR material.
// =============================================================================
// Mirrors VulkanRT::VkGpuMaterial (include/Backend/vulkan_material_types.h)
// byte-for-byte: 20 blocks x 16 bytes = 320 bytes. std430 / scalar array stride
// is therefore 320.
//
// CRITICAL: every shader that reads the `materials` buffer MUST use THIS struct
// (via #include) and nothing else. A private/shorter copy makes materials[idx]
// for any idx >= 1 read a SHIFTED offset, which previously caused:
//   - missing shadows on multi-material objects (shadow_anyhit had a 256-byte copy)
//   - wrong textures + vanished objects in material preview (frag had a 256-byte copy)
//
// When adding/removing a field, update ONLY: this file, VkGpuMaterial, and the
// CPU-side static_assert(sizeof(VkGpuMaterial) == 320) — never per-shader copies.
// =============================================================================
#ifndef MATERIAL_STRUCT_GLSL
#define MATERIAL_STRUCT_GLSL

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
    // Block 10: Extra water animation params
    float micro_anim_speed, micro_morph_speed, foam_noise_scale, fft_wind_direction;
    // Block 11: UV transform core
    float uv_scale_x, uv_scale_y, uv_offset_x, uv_offset_y;
    // Block 12: UV transform extra
    float uv_rotation_degrees, uv_tiling_x, uv_tiling_y;
    uint uv_wrap_mode;
    // Block 13: Standard Textures (first 4)
    uint albedo_tex;
    uint normal_tex;
    uint roughness_tex;
    uint metallic_tex;
    // Block 14: Standard Textures (second 4)
    uint emission_tex;
    uint height_tex;
    uint opacity_tex;
    uint transmission_tex;
    // Block 15: Reserved + terrain layer index
    float subsurface_ior;
    uint _terrain_layer_idx;
    float normal_strength;
    float tile_break_strength;
    // Block 16: Specular controls
    float specular;
    uint specular_tex;
    float bubble_ior;
    float bubble_film;
    // Block 17: Resin interior absorption (thick glass / glass-marble depth)
    float resin_color_r;
    float resin_color_g;
    float resin_color_b;
    float transmission_density;
    // Block 18: Resin coat layer params + spectral dispersion
    float resin_roughness;
    float dispersion;      // spectral dispersion strength (0 = off; repurposed _resin_pad0)
    float _resin_pad1;
    float _resin_pad2;
    // Block 19: Resin internal inclusions (dust/dirt march)
    float resin_inclusion;
    float resin_dirt;
    float resin_inclusion_scale;
    float resin_dirt_color_r;
    // Block 20: Resin dirt colour tail + iridescent clearcoat
    float resin_dirt_color_g;
    float resin_dirt_color_b;
    float clearcoat_iridescence;
    float clearcoat_film_thickness;
};

#endif // MATERIAL_STRUCT_GLSL
