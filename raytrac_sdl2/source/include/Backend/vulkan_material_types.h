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
 * Matches the layout in closesthit.rchit exactly (12 blocks of 16 bytes).
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

    // Block 10: Standard Textures (first 4) (16 bytes)
    VkGpuTextureHandle albedo_tex;
    VkGpuTextureHandle normal_tex;
    VkGpuTextureHandle roughness_tex;
    VkGpuTextureHandle metallic_tex;

    // Block 11: Standard Textures (second 4) (16 bytes)
    VkGpuTextureHandle emission_tex;
    VkGpuTextureHandle height_tex; 
    VkGpuTextureHandle opacity_tex;
    VkGpuTextureHandle transmission_tex;

    // Block 12: Reserved for future use (16 bytes)
    uint32_t _reserved[4];
};

/**
 * @brief Vulkan-Specific GPU Light struct.
 */
struct VK_GPU_ALIGN(16) VkGpuLight {
    float position[4];  // xyz + type
    float color[4];     // rgb + intensity
    float params[4];    // radius, width, height, inner_angle
    float direction[4]; // xyz + outer_angle
};

} // namespace VulkanRT
