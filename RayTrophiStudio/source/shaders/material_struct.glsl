// =============================================================================
// material_struct.glsl — SINGLE SOURCE OF TRUTH for the Vulkan PBR material.
// =============================================================================
// The material record is SPLIT into two SSBOs so a hit only fetches what it
// uses (the old monolithic struct had grown to 352 bytes and every hit paid
// the whole record):
//
//   `Material`    — HOT core, binding 2. 10 blocks x 16 B = 160 B stride.
//                   Read by every closesthit / shadow any-hit invocation.
//                   Mirrors VulkanRT::VkGpuMaterialCore byte-for-byte.
//   `MaterialExt` — COLD extension, binding 24 (RT set) / binding 4 (material
//                   preview set). 13 blocks x 16 B = 208 B stride. Read ONLY
//                   inside feature-gated branches (SSS lobe, water fast path,
//                   bubble, resin/interior volume, iridescent clearcoat).
//                   Mirrors VulkanRT::VkGpuMaterialExt byte-for-byte.
//
// CRITICAL: every shader that reads these buffers MUST use THESE structs (via
// #include) and nothing else. A private/shorter copy makes materials[idx] for
// any idx >= 1 read a SHIFTED offset (wrong textures / missing shadows /
// vanished objects — see bugfix history 2026-06-22).
//
// When adding/removing a field, update ONLY: this file, VkGpuMaterialCore/Ext +
// splitGpuMaterial() (include/Backend/vulkan_material_types.h), and the
// static_asserts there — never per-shader copies.
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
    // Block 4: Lobe probabilities (read every scatter decision)
    float clearcoat, clearcoat_roughness, translucent, subsurface_amount;
    // Block 5: Specular / normal / dispersion + flags
    float specular, normal_strength, dispersion;
    uint  flags;
    // Block 6: UV transform core
    float uv_scale_x, uv_scale_y, uv_offset_x, uv_offset_y;
    // Block 7: UV transform extra
    float uv_rotation_degrees, uv_tiling_x, uv_tiling_y;
    uint  uv_wrap_mode;
    // Block 8: Standard textures (first 4)
    uint albedo_tex;
    uint normal_tex;
    uint roughness_tex;
    uint metallic_tex;
    // Block 9: Standard textures (second 4)
    uint emission_tex;
    uint opacity_tex;
    uint transmission_tex;
    uint specular_tex;
    // Block 10: Tile break + terrain layer + height/FFT texture slot
    float tile_break_strength;
    uint  _terrain_layer_idx;
    uint  height_tex;
    float _core_pad0;
};

struct MaterialExt {
    // Block 1-2: Subsurface scattering
    float subsurface_r, subsurface_g, subsurface_b, subsurface_scale;
    float subsurface_radius_r, subsurface_radius_g, subsurface_radius_b, subsurface_anisotropy;
    // Block 3: Legacy water controls (anisotropic=wave_speed, sheen=wave_strength,
    //          sheen_tint=wave_frequency) + reserved
    float anisotropic, sheen, sheen_tint, subsurface_ior;
    // Block 4-6: Water FFT / micro detail
    float fft_amplitude, fft_time_scale, micro_detail_strength, micro_detail_scale;
    float foam_threshold, fft_ocean_size, fft_choppiness, fft_wind_speed;
    float micro_anim_speed, micro_morph_speed, foam_noise_scale, fft_wind_direction;
    // Block 7-10: Bubble + resin / interior volume
    float bubble_ior, bubble_film, resin_roughness, resin_shard;
    float resin_shard_hue, resin_color_r, resin_color_g, resin_color_b;
    float transmission_density, resin_inclusion, resin_dirt, resin_inclusion_scale;
    float resin_dirt_color_r, resin_dirt_color_g, resin_dirt_color_b, clearcoat_iridescence;
    // Block 11-13: Iridescence tail + interior dust
    float clearcoat_film_thickness, dust_color_a_r, dust_color_a_g, dust_color_a_b;
    float dust_style, dust_color_b_r, dust_color_b_g, dust_color_b_b;
    float shard_shape, _ext_pad0, _ext_pad1, _ext_pad2;
};

#endif // MATERIAL_STRUCT_GLSL
