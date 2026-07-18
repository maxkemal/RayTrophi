#pragma once
#include <cstdint>

/**
 * @file vulkan_material_types.h
 * @brief Fully isolated CPU-host material structures for Vulkan backend.
 */

#ifdef _MSC_VER
    #define VK_GPU_ALIGN(n) __declspec(align(n))
#else
    #define VK_GPU_ALIGN(n) alignas(n)
#endif

namespace VulkanRT {

// For GLSL compatibility, use uint32_t for texture indices
// 64-bit handles are handled on CPU side, GPU receives 32-bit indices
typedef uint32_t VkGpuTextureHandle;

/**
 * @brief Vulkan-Specific GPU Material struct.
 * Matches the layout in closesthit.rchit exactly.
 * Texture handles are uint32_t indices for GLSL compatibility.
 */
struct VK_GPU_ALIGN(16) VkGpuMaterial {
    // Block 1: Albedo + opacity (16 bytes)
    float albedo_r, albedo_g, albedo_b, opacity;
    
    // Block 2: Emission + strength (16 bytes)
    float emission_r, emission_g, emission_b, emission_strength;
    
    // Block 3: PBR properties (16 bytes)
    float roughness, metallic, ior, transmission;
    
    // Block 4: Subsurface color + amount (16 bytes)
    float subsurface_r, subsurface_g, subsurface_b, subsurface_amount;
    
    // Block 5: Subsurface radius + scale (16 bytes)
    float subsurface_radius_r, subsurface_radius_g, subsurface_radius_b, subsurface_scale;
    
    // Block 6: Coatings & Translucency (16 bytes)
    float clearcoat, clearcoat_roughness, translucent, subsurface_anisotropy;
    
    // Block 7: Additional properties (16 bytes)
    float anisotropic, sheen, sheen_tint; 
    uint32_t flags;

    // Block 8: Water/Extra params (16 bytes)
    float fft_amplitude, fft_time_scale, micro_detail_strength, micro_detail_scale;

    // Block 9: Extra water params (16 bytes) 
    float foam_threshold, fft_ocean_size, fft_choppiness, fft_wind_speed;

    // Block 10: Extra water animation params (16 bytes)
    float micro_anim_speed, micro_morph_speed, foam_noise_scale, fft_wind_direction;

    // Block 11: UV transform core (16 bytes)
    float uv_scale_x, uv_scale_y, uv_offset_x, uv_offset_y;

    // Block 12: UV transform extra (16 bytes)
    float uv_rotation_degrees, uv_tiling_x, uv_tiling_y;
    uint32_t uv_wrap_mode;

    // Block 13: Standard Textures (first 4) (16 bytes)
    VkGpuTextureHandle albedo_tex;
    VkGpuTextureHandle normal_tex;
    VkGpuTextureHandle roughness_tex;
    VkGpuTextureHandle metallic_tex;

    // Block 14: Standard Textures (second 4) (16 bytes)
    VkGpuTextureHandle emission_tex;
    VkGpuTextureHandle height_tex; 
    VkGpuTextureHandle opacity_tex;
    VkGpuTextureHandle transmission_tex;

    // Block 15: Terrain layer index + subsurface IOR (16 bytes)
    // When FLAG_TERRAIN (bit 16) is set in flags, _terrain_layer_idx is the index
    // into the TerrainLayerBuffer (binding 12) for splat-based layer blending.
    float subsurface_ior;
    uint32_t _terrain_layer_idx; // terrain layer buffer index (valid when FLAG_TERRAIN set)
    float normal_strength;
    float tile_break_strength;   // UV tile-break strength (0=off, 0.1–0.3 typical)

    // Block 16: Specular controls (16 bytes)
    float specular;              // Blender-style dielectric F0 = 0.08 * specular
    VkGpuTextureHandle specular_tex;
    float bubble_ior;            // Thin-shell bubble rim Fresnel IOR (when VK_MAT_FLAG_BUBBLE set)
    float bubble_film;           // Thin-film interference strength (soap iridescence)

    // Block 17: Resin (thick glass / glass-marble) interior absorption (16 bytes)
    float resin_color_r;         // resin absorption tint (separate from albedo); white = clear
    float resin_color_g;
    float resin_color_b;
    float transmission_density;  // 0 = legacy constant-thickness glass; >0 = real-distance Beer-Lambert

    // Block 18: Resin coat layer params + spectral dispersion (16 bytes)
    float resin_roughness;       // resin coat gloss (reflect-lobe roughness), independent of base
    float dispersion;            // spectral dispersion strength (0 = off; repurposed _resin_pad0)
    float resin_shard;           // colored glass-shard amount (repurposed _resin_pad1)
    float resin_shard_hue;       // shard base hue 0..1; <0 = rainbow (repurposed _resin_pad2)

    // Block 19: Resin internal inclusions — dust/dirt march (16 bytes)
    float resin_inclusion;       // dust cloudiness amount (0 = off, analytic fallback)
    float resin_dirt;            // dirt speck amount (opaque early-return)
    float resin_inclusion_scale; // procedural feature size
    float resin_dirt_color_r;

    // Block 20: Resin dirt colour tail + iridescent clearcoat (16 bytes)
    float resin_dirt_color_g;
    float resin_dirt_color_b;
    float clearcoat_iridescence;    // thin-film tint on clearcoat lobe (0 = plain white)
    float clearcoat_film_thickness; // hue cycle / film thickness (OPD scale)

    // Block 21: Interior Volume dust colour A + style (16 bytes)
    float dust_color_a_r;           // paint/nebula colour pole A
    float dust_color_a_g;
    float dust_color_a_b;
    float dust_style;               // 0=Nebula(auto) 1=Billow 2-colour 2=Wispy 3=Paint swirl

    // Block 22: Interior Volume dust colour B + shard shape (16 bytes)
    float dust_color_b_r;           // paint/nebula colour pole B
    float dust_color_b_g;
    float dust_color_b_b;
    float shard_shape;              // 0=round chips, 1=elongated faceted crystals
};

// AUTHORING struct only — VkGpuMaterial is what the CPU fill code writes, but it
// is NO LONGER uploaded as-is. splitGpuMaterial() below derives the two buffers
// the GPU actually reads: VkGpuMaterialCore (binding 2, hot per-hit fields) and
// VkGpuMaterialExt (binding 24, feature-gated cold fields). Keeping the fill code
// on this single struct means adding a material feature stays a one-place edit;
// only the split decides which buffer the new field lands in.
static_assert(sizeof(VkGpuMaterial) == 352, "VkGpuMaterial (authoring) must stay 352 bytes (22x16)");

/**
 * @brief HOT material fields — binding 2, read by every closesthit/any-hit
 *        invocation. Mirrors `Material` in shaders/material_struct.glsl exactly.
 *        Kept to 10 blocks (160 B) so a hit touches 1-3 cache lines instead of
 *        the whole 352 B authoring record; the growth of feature fields (resin,
 *        dust, water FFT…) had turned every hit into a full-struct fetch.
 */
struct VK_GPU_ALIGN(16) VkGpuMaterialCore {
    // Block 1
    float albedo_r, albedo_g, albedo_b, opacity;
    // Block 2
    float emission_r, emission_g, emission_b, emission_strength;
    // Block 3
    float roughness, metallic, ior, transmission;
    // Block 4 — lobe probabilities (read every scatter decision)
    float clearcoat, clearcoat_roughness, translucent, subsurface_amount;
    // Block 5
    float specular, normal_strength, dispersion;
    uint32_t flags;
    // Block 6-7: UV transform (closesthit + shadow any-hit, every textured hit)
    float uv_scale_x, uv_scale_y, uv_offset_x, uv_offset_y;
    float uv_rotation_degrees, uv_tiling_x, uv_tiling_y;
    uint32_t uv_wrap_mode;
    // Block 8-9: texture slots
    VkGpuTextureHandle albedo_tex, normal_tex, roughness_tex, metallic_tex;
    VkGpuTextureHandle emission_tex, opacity_tex, transmission_tex, specular_tex;
    // Block 10
    float tile_break_strength;
    uint32_t _terrain_layer_idx;
    VkGpuTextureHandle height_tex;
    float _core_pad0;
};
static_assert(sizeof(VkGpuMaterialCore) == 160, "VkGpuMaterialCore must stay 160 bytes (10x16); update shaders/material_struct.glsl `Material` to match");

/**
 * @brief COLD material fields — binding 24, read only inside feature-gated
 *        branches (SSS lobe, water fast path, bubble, resin/interior volume,
 *        iridescent clearcoat). Mirrors `MaterialExt` in material_struct.glsl.
 */
struct VK_GPU_ALIGN(16) VkGpuMaterialExt {
    // Block 1-2: SSS
    float subsurface_r, subsurface_g, subsurface_b, subsurface_scale;
    float subsurface_radius_r, subsurface_radius_g, subsurface_radius_b, subsurface_anisotropy;
    // Block 3: legacy water controls (anisotropic=wave_speed, sheen=wave_strength, sheen_tint=wave_freq)
    float anisotropic, sheen, sheen_tint, subsurface_ior;
    // Block 4-6: water FFT / micro detail
    float fft_amplitude, fft_time_scale, micro_detail_strength, micro_detail_scale;
    float foam_threshold, fft_ocean_size, fft_choppiness, fft_wind_speed;
    float micro_anim_speed, micro_morph_speed, foam_noise_scale, fft_wind_direction;
    // Block 7-10: bubble + resin/interior volume
    float bubble_ior, bubble_film, resin_roughness, resin_shard;
    float resin_shard_hue, resin_color_r, resin_color_g, resin_color_b;
    float transmission_density, resin_inclusion, resin_dirt, resin_inclusion_scale;
    float resin_dirt_color_r, resin_dirt_color_g, resin_dirt_color_b, clearcoat_iridescence;
    // Block 11-13: iridescence tail + interior dust
    float clearcoat_film_thickness, dust_color_a_r, dust_color_a_g, dust_color_a_b;
    float dust_style, dust_color_b_r, dust_color_b_g, dust_color_b_b;
    float shard_shape, _ext_pad0, _ext_pad1, _ext_pad2;
};
static_assert(sizeof(VkGpuMaterialExt) == 208, "VkGpuMaterialExt must stay 208 bytes (13x16); update shaders/material_struct.glsl `MaterialExt` to match");

/// Derive the two GPU records from one authoring record. Field-for-field copy —
/// keep in sync with both structs above (a missed field reads as 0 on the GPU,
/// which the compiler cannot catch; group edits here with struct edits).
inline void splitGpuMaterial(const VkGpuMaterial& m, VkGpuMaterialCore& c, VkGpuMaterialExt& e) {
    c.albedo_r = m.albedo_r; c.albedo_g = m.albedo_g; c.albedo_b = m.albedo_b; c.opacity = m.opacity;
    c.emission_r = m.emission_r; c.emission_g = m.emission_g; c.emission_b = m.emission_b; c.emission_strength = m.emission_strength;
    c.roughness = m.roughness; c.metallic = m.metallic; c.ior = m.ior; c.transmission = m.transmission;
    c.clearcoat = m.clearcoat; c.clearcoat_roughness = m.clearcoat_roughness; c.translucent = m.translucent; c.subsurface_amount = m.subsurface_amount;
    c.specular = m.specular; c.normal_strength = m.normal_strength; c.dispersion = m.dispersion; c.flags = m.flags;
    c.uv_scale_x = m.uv_scale_x; c.uv_scale_y = m.uv_scale_y; c.uv_offset_x = m.uv_offset_x; c.uv_offset_y = m.uv_offset_y;
    c.uv_rotation_degrees = m.uv_rotation_degrees; c.uv_tiling_x = m.uv_tiling_x; c.uv_tiling_y = m.uv_tiling_y; c.uv_wrap_mode = m.uv_wrap_mode;
    c.albedo_tex = m.albedo_tex; c.normal_tex = m.normal_tex; c.roughness_tex = m.roughness_tex; c.metallic_tex = m.metallic_tex;
    c.emission_tex = m.emission_tex; c.opacity_tex = m.opacity_tex; c.transmission_tex = m.transmission_tex; c.specular_tex = m.specular_tex;
    c.tile_break_strength = m.tile_break_strength; c._terrain_layer_idx = m._terrain_layer_idx; c.height_tex = m.height_tex; c._core_pad0 = 0.0f;

    e.subsurface_r = m.subsurface_r; e.subsurface_g = m.subsurface_g; e.subsurface_b = m.subsurface_b; e.subsurface_scale = m.subsurface_scale;
    e.subsurface_radius_r = m.subsurface_radius_r; e.subsurface_radius_g = m.subsurface_radius_g; e.subsurface_radius_b = m.subsurface_radius_b; e.subsurface_anisotropy = m.subsurface_anisotropy;
    e.anisotropic = m.anisotropic; e.sheen = m.sheen; e.sheen_tint = m.sheen_tint; e.subsurface_ior = m.subsurface_ior;
    e.fft_amplitude = m.fft_amplitude; e.fft_time_scale = m.fft_time_scale; e.micro_detail_strength = m.micro_detail_strength; e.micro_detail_scale = m.micro_detail_scale;
    e.foam_threshold = m.foam_threshold; e.fft_ocean_size = m.fft_ocean_size; e.fft_choppiness = m.fft_choppiness; e.fft_wind_speed = m.fft_wind_speed;
    e.micro_anim_speed = m.micro_anim_speed; e.micro_morph_speed = m.micro_morph_speed; e.foam_noise_scale = m.foam_noise_scale; e.fft_wind_direction = m.fft_wind_direction;
    e.bubble_ior = m.bubble_ior; e.bubble_film = m.bubble_film; e.resin_roughness = m.resin_roughness; e.resin_shard = m.resin_shard;
    e.resin_shard_hue = m.resin_shard_hue; e.resin_color_r = m.resin_color_r; e.resin_color_g = m.resin_color_g; e.resin_color_b = m.resin_color_b;
    e.transmission_density = m.transmission_density; e.resin_inclusion = m.resin_inclusion; e.resin_dirt = m.resin_dirt; e.resin_inclusion_scale = m.resin_inclusion_scale;
    e.resin_dirt_color_r = m.resin_dirt_color_r; e.resin_dirt_color_g = m.resin_dirt_color_g; e.resin_dirt_color_b = m.resin_dirt_color_b; e.clearcoat_iridescence = m.clearcoat_iridescence;
    e.clearcoat_film_thickness = m.clearcoat_film_thickness; e.dust_color_a_r = m.dust_color_a_r; e.dust_color_a_g = m.dust_color_a_g; e.dust_color_a_b = m.dust_color_a_b;
    e.dust_style = m.dust_style; e.dust_color_b_r = m.dust_color_b_r; e.dust_color_b_g = m.dust_color_b_g; e.dust_color_b_b = m.dust_color_b_b;
    e.shard_shape = m.shard_shape; e._ext_pad0 = 0.0f; e._ext_pad1 = 0.0f; e._ext_pad2 = 0.0f;
}

// Flag bits for VkGpuMaterial::flags
static constexpr uint32_t VK_MAT_FLAG_TERRAIN = (1u << 16); // Splat-blended terrain material
static constexpr uint32_t VK_MAT_FLAG_WATER   = (1u << 17); // Explicit water surface material
static constexpr uint32_t VK_MAT_FLAG_WATER_FFT_READY = (1u << 18); // height/normal slots contain FFT textures
static constexpr uint32_t VK_MAT_FLAG_BUBBLE  = (1u << 19); // Thin-shell bubble (Fresnel rim + pass-through)
static constexpr uint32_t VK_MAT_FLAG_MARBLE_VOLUME = (1u << 20); // Glass marble full-volume medium march (raygen)
static constexpr uint32_t VK_MAT_FLAG_RESIN_OBJ_SPACE = (1u << 21); // Interior volume anchored in OBJECT space (moves with the mesh)
static constexpr uint32_t VK_MAT_FLAG_WATER_LAKE = (1u << 22); // Fetch-limited inland water profile
static constexpr uint32_t VK_MAT_FLAG_WATER_RIVER = (1u << 23); // UV-flow-aligned river profile

/**
 * @brief Per-terrain splat-layer descriptor uploaded to binding 12.
 *        Contains up to 4 material layer indices, per-layer UV scales,
 *        and the splat map texture handle used for blending weights.
 */
struct VK_GPU_ALIGN(16) VkTerrainLayerData {
    uint32_t layer_mat_id[4];   // Material buffer indices for layers 0-3
    float    layer_uv_scale[4]; // UV tiling scales for layers 0-3
    uint32_t splat_map_tex;     // Combined-image-sampler slot for the RGBA splat map
    uint32_t layer_count;       // Number of active layers (1-4)
    uint32_t _pad[2];           // Padding to 48 bytes
};
static_assert(sizeof(VkTerrainLayerData) == 48, "VkTerrainLayerData size mismatch");

/**
 * @brief Vulkan-Specific GPU Light struct.
 */
struct VK_GPU_ALIGN(16) VkGpuLight {
    float position[4];  // xyz + type
    float color[4];     // rgb + intensity
    float params[4];    // radius, width, height, inner_angle
    float direction[4]; // xyz + outer_angle
    float area_u[4];    // xyz: AreaLight u-axis (unit), w: pad
    float area_v[4];    // xyz: AreaLight v-axis (unit), w: pad
};

} // namespace VulkanRT
