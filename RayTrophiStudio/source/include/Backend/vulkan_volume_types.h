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
    int   volume_type;          // 0=homogeneous, 1=noise, 2=NanoVDB, 3=cloud, 4=live dense gas
    int   is_active;             // 1 = enabled, 0 = skip
    float voxel_size;           // Voxel size for adaptive stepping
    int   shadow_stride;        // Reuse self-shadow across N primary samples
    
    // ═══════════════════════════ INVERSE TRANSFORM (48 bytes) ═══════════════
    // Row-major 3x4 inverse affine transform (world → object)
    float inv_transform[12];
    
    // ═══════════════════════════ RESERVED (24 bytes) ════════════════════════
    uint64_t vdb_grid_address;   // NanoVDB density grid device address
    uint64_t vdb_temp_address;   // NanoVDB temperature grid device address (fire/blackbody)
    float    _reserved[2];       // [0] density cutoff, [1] reserved

    // ═══════════════════════════ EMISSION EXTENSION (256 bytes) ═════════════
    // Blackbody / color-ramp emission — matches GpuVDBVolume fields.
    // emission_mode: 0=off, 1=plain color, 2=blackbody/color-ramp
    int   emission_mode;
    float temperature_scale;      // temperature multiplier for blackbody
    float blackbody_intensity;    // emission strength for blackbody mode
    float max_temperature;        // reference max temperature
    int   color_ramp_enabled;     // 1 = use color ramp instead of pure blackbody
    int   ramp_stop_count;        // active stops in ramp (0..8)
    int   _ramp_pad[2];           // alignment
    float ramp_positions[8];      // stop positions [0..1]
    float ramp_colors_r[8];       // stop R components
    float ramp_colors_g[8];       // stop G components
    float ramp_colors_b[8];       // stop B components
    float pivot_offset[3];        // Pivot correction identical to OptiX
    int   source_type;            // 0=NanoVDB, 3=cloud, 4=surface SDF, 5=live dense gas
    float cloud_coverage;
    float cloud_detail;
    float cloud_erosion;
    float cloud_base_scale;
    float cloud_edge_fade;
    float cloud_offset_x;
    float cloud_offset_z;
    float cloud_seed;
    // [0..6] = isosurface/foam data for source_type 4. For other volume
    // types [6] stores authored minimum emission temperature. [7..11] =
    // bounded Material Graph density-noise program.
    float _ext_reserved[12];

    // ═══════════════════════════ ACCELERATION (64 bytes) ════════════════════
    // Appended at the END so every existing offset is untouched. GROWING this
    // struct means EVERY shader that declares VkVolumeInstance must be updated
    // in the same commit — the SSBO stride is per-declaration, so one stale copy
    // shifts every instance after the first and the volume table silently reads
    // garbage. Current declarations: volume_closesthit.rchit, closesthit.rchit,
    // raygen.rgen, volume_intersection.rint.
    //
    // Per-block maximum density for live dense gas (source_type 5). The RT march
    // skips a whole block when its maximum is below the density cutoff; a dense
    // domain has no other empty-space acceleration, since the NanoVDB hierarchy
    // skip only applies to volume_type 2. Address 0 = unavailable, and the
    // shader must then march every step: a missing majorant may never be read as
    // "empty", or the skip erases real smoke.
    uint64_t majorant_address;
    float    majorant_dim[3];     // block-grid resolution
    float    majorant_block;      // cells per block edge (kGasMajorantBlock)
    // Combustion reaction field (GridFluid's bounded `interaction` channel) for
    // live dense gas. Temperature alone cannot separate a flame from the hot
    // smoke above it — both are hot — so without this the fire core has no
    // distinct emission. 0 when the domain publishes no flame grid.
    uint64_t flame_address;
    // Emitting-block list for volume emission NEE: [0] = count, [1..] = block
    // indices into the majorant grid. Lets a scatter sample aim straight at the
    // fire instead of waiting for a random bounce to land in it.
    uint64_t emissive_list_address;
    float    emissive_capacity;   // entries the list can hold (0 = unavailable)
    // SDF isosurface material: 1-based MaterialManager id, 0 = none. Claimed from
    // the accel headroom rather than _reserved[1] so the volume-VM meaning of that
    // slot is untouched, and rather than growing the struct so the 576-byte ABI
    // (mirrored by five shader declarations) stays put.
    float    iso_material_index;
    // Headroom for the next pass (velocity grid for volume motion blur).
    float    _accel_reserved[4];
};

// Compile-time size check (576 bytes = 9 cache lines)
static_assert(sizeof(VkVolumeInstance) == 576, "VkVolumeInstance must be 576 bytes");

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
