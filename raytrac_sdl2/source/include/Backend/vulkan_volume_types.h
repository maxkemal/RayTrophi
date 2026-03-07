/*
 * File: vulkan_volume_types.h
 * Description: Vulkan GPU Volume data structures — OptiX-compatible layout
 * 
 * Mirrors OptiX HitGroupData volumetric fields and GpuVDBVolume/GpuGasVolume
 * for Vulkan ray tracing backend. Uploaded as SSBO at binding 9.
 *
 * Struct layout matches GLSL VkVolumeInstance in volume_closesthit.rchit
 */

#pragma once
#include <cstdint>

#ifdef _MSC_VER
    #define VK_VOL_ALIGN(n) __declspec(align(n))
#else
    #define VK_VOL_ALIGN(n) alignas(n)
#endif

namespace VulkanRT {

/**
 * @struct VkVolumeInstance
 * @brief Single volumetric object for Vulkan RT pipeline.
 *
 * Compatible with OptiX GpuVDBVolume / GpuGasVolume / HitGroupData volumetric fields.
 * Each instance describes a bounded volume with density, scattering, absorption
 * and emission properties for ray-march integration.
 *
 * Size: 256 bytes (16-byte aligned, cache-friendly)
 */
struct VK_VOL_ALIGN(16) VkVolumeInstance {
    // ═══════════════════════════ TRANSFORM (48 bytes) ═══════════════════════
    // Row-major 3x4 affine transform (object → world)
    float transform[12];    // [0..2] = row0, [3..5] = row1, [6..8] = row2, [9..11] = translation
    
    // ═══════════════════════════ BOUNDS (24 bytes) ══════════════════════════
    // VDB native (original file) world-space AABB — NOT the gizmo-moved scene AABB.
    // Used by volume_closesthit.rchit to remap localPos [-0.5,0.5] → VDB world space
    // before calling pnanovdb_map_apply_inverse, so NanoVDB sampling is correct even
    // after the volume is moved/rotated/scaled with the scene gizmo.
    // Populated from GpuVDBVolume::local_bbox_min / local_bbox_max.
    float aabb_min[3];      // VDB native world-space bounding box minimum
    float aabb_max[3];      // VDB native world-space bounding box maximum
    
    // ═══════════════════════════ DENSITY (16 bytes) ═════════════════════════
    float density_multiplier;   // Base density scale (maps to OptiX vol_density)
    float density_remap_low;    // Density remap input low  (default 0.0)
    float density_remap_high;   // Density remap input high (default 1.0)
    float noise_scale;          // Procedural noise frequency (maps to vol_noise_scale)
    
    // ═══════════════════════════ SCATTERING (32 bytes) ══════════════════════
    float scatter_color[3];     // Scattering albedo (maps to vol_albedo)
    float scatter_coefficient;  // Sigma_s (maps to vol_scattering)
    float scatter_anisotropy;   // Henyey-Greenstein g forward (maps to vol_g)
    float scatter_anisotropy_back; // Backward lobe g (maps to vol_g_back)
    float scatter_lobe_mix;     // Forward/backward mix (maps to vol_lobe_mix)
    float scatter_multi;        // Multi-scatter contribution (maps to vol_multi_scatter)
    
    // ═══════════════════════════ ABSORPTION (16 bytes) ══════════════════════
    float absorption_color[3];  // Absorption tint
    float absorption_coefficient; // Sigma_a (maps to vol_absorption)
    
    // ═══════════════════════════ EMISSION (16 bytes) ════════════════════════
    float emission_color[3];    // Volume emission color (maps to vol_emission)
    float emission_intensity;   // Emission strength
    
    // ═══════════════════════════ RAY MARCH PARAMS (16 bytes) ════════════════
    float step_size;            // Ray march step (maps to vol_step_size)
    int   max_steps;            // Max ray march iterations (maps to vol_max_steps)
    int   shadow_steps;         // Light march steps (maps to vol_light_steps)
    float shadow_strength;      // Self-shadow intensity (maps to vol_shadow_strength)
    
    // ═══════════════════════════ FLAGS & PADDING (16 bytes) ═════════════════
    int   volume_type;          // 0 = homogeneous, 1 = procedural noise, 2 = 3D texture (future)
    int   is_active;             // 1 = enabled, 0 = skip
    float voxel_size;           // Voxel size for adaptive stepping
    int   _pad0;                // Alignment padding
    
    // ═══════════════════════════ INVERSE TRANSFORM (48 bytes) ═══════════════
    // Row-major 3x4 inverse affine transform (world → object)
    float inv_transform[12];
    
    // ═══════════════════════════ RESERVED (24 bytes) ════════════════════════
    uint64_t vdb_grid_address;   // NanoVDB density grid device address
    uint64_t vdb_temp_address;   // NanoVDB temperature grid device address (fire/blackbody)
    float    _reserved[2];       // padding

    // ═══════════════════════════ EMISSION EXTENSION (256 bytes) ═════════════
    // Blackbody / color-ramp emission — matches GpuVDBVolume fields.
    // emission_mode: 0=off, 1=plain color, 2=blackbody/color-ramp
    int   emission_mode;
    float temperature_scale;      // temperature multiplier for blackbody
    float blackbody_intensity;    // emission strength for blackbody mode
    float max_temperature;        // reference max temperature (unused in shader, kept for parity)
    int   color_ramp_enabled;     // 1 = use color ramp instead of pure blackbody
    int   ramp_stop_count;        // active stops in ramp (0..8)
    int   _ramp_pad[2];           // alignment
    float ramp_positions[8];      // stop positions [0..1]
    float ramp_colors_r[8];       // stop R components
    float ramp_colors_g[8];       // stop G components
    float ramp_colors_b[8];       // stop B components
    float pivot_offset[3];        // Pivot correction identical to OptiX
    float _ext_reserved[21];      // padding to reach 512 total bytes
};

// Compile-time size check (512 bytes = 8 cache lines)
static_assert(sizeof(VkVolumeInstance) == 512, "VkVolumeInstance must be 512 bytes");

/**
 * @struct VkVolumeParams
 * @brief Global volume rendering parameters (part of push constants or world data)
 */
struct VK_VOL_ALIGN(16) VkVolumeParams {
    int   volume_count;         // Number of active volume instances
    float global_density_scale; // Global density multiplier
    int   max_volume_bounces;   // Max volume scattering events per path
    int   _pad0;
};

} // namespace VulkanRT
