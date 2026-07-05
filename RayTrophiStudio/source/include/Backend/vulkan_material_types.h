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
    float _resin_pad1;           // reserved: resin roughness texture handle
    float _resin_pad2;           // reserved: resin normal texture handle

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
};

// Single source of truth for the GPU material stride. Every material-reading shader
// includes shaders/material_struct.glsl which must mirror THIS struct exactly. If
// this assert fires after adding a field, update material_struct.glsl too — a shader
// copy that is shorter makes materials[idx] for idx>=1 read a shifted offset (wrong
// textures / missing shadows / vanished objects). See bugfix history 2026-06-22.
static_assert(sizeof(VkGpuMaterial) == 320, "VkGpuMaterial must stay 320 bytes (20x16); update shaders/material_struct.glsl to match");

// Flag bits for VkGpuMaterial::flags
static constexpr uint32_t VK_MAT_FLAG_TERRAIN = (1u << 16); // Splat-blended terrain material
static constexpr uint32_t VK_MAT_FLAG_WATER   = (1u << 17); // Explicit water surface material
static constexpr uint32_t VK_MAT_FLAG_WATER_FFT_READY = (1u << 18); // height/normal slots contain FFT textures
static constexpr uint32_t VK_MAT_FLAG_BUBBLE  = (1u << 19); // Thin-shell bubble (Fresnel rim + pass-through)
static constexpr uint32_t VK_MAT_FLAG_MARBLE_VOLUME = (1u << 20); // Glass marble full-volume medium march (raygen)

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
