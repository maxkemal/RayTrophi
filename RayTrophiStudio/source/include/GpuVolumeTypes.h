#pragma once

// Shared volume publication data for Vulkan and offline render backends.
// These records carry backend-owned handles/addresses; their names, field order
// and layouts are preserved from params.h. This header does not require OptiX.
#include "GpuInteropTypes.h"
#include <cstdint>

// ═══════════════════════════════════════════════════════════════════════════════
// GPU VDB VOLUME (Industry-Standard VDB Rendering)
// Independent volume objects with transform and multi-grid support
// ═══════════════════════════════════════════════════════════════════════════════
struct GpuVDBVolume {
    // ─────────────────────────────────────────────────────────────────────────
    // GRID POINTERS (NanoVDB device pointers)
    // ─────────────────────────────────────────────────────────────────────────
    void* density_grid;       // FloatGrid* - Required
    void* temperature_grid;   // FloatGrid* - Optional (fire)
    void* velocity_grid;      // Vec3fGrid* - Optional (motion blur)
    void* emission_grid;      // FloatGrid* - Optional (custom emission)
    
    // ─────────────────────────────────────────────────────────────────────────
    // TRANSFORM (Object → World, row-major 3x4)
    // ─────────────────────────────────────────────────────────────────────────
    float transform[12];      // Object to world
    float inv_transform[12];  // World to object (for ray transform)
    
    // ─────────────────────────────────────────────────────────────────────────
    // BOUNDS
    // ─────────────────────────────────────────────────────────────────────────
    float3 world_bbox_min;    // World-space bounds (after transform)
    float3 world_bbox_max;
    float3 local_bbox_min;    // Native VDB bounds (before transform)
    float3 local_bbox_max;
    
    // ─────────────────────────────────────────────────────────────────────────
    // DENSITY SHADER
    // ─────────────────────────────────────────────────────────────────────────
    float density_multiplier;
    float density_remap_low;
    float density_remap_high;
    float density_pad;
    
    // ─────────────────────────────────────────────────────────────────────────
    // SCATTERING SHADER
    // ─────────────────────────────────────────────────────────────────────────
    float3 scatter_color;
    float scatter_coefficient;
    float scatter_anisotropy;       // G forward (-1 to 1)
    float scatter_anisotropy_back;  // G backward
    float scatter_lobe_mix;         // Forward/back blend
    float scatter_multi;            // Multi-scatter approximation
    
    // ─────────────────────────────────────────────────────────────────────────
    // ABSORPTION SHADER
    // ─────────────────────────────────────────────────────────────────────────
    float3 absorption_color;
    float absorption_coefficient;
    
    // ─────────────────────────────────────────────────────────────────────────
    // EMISSION SHADER (Fire/Explosions)
    // ─────────────────────────────────────────────────────────────────────────
    int emission_mode;  // 0=None, 1=Constant, 2=Blackbody, 3=Channel
    float3 emission_color;
    float emission_intensity;
    float temperature_scale;
    float blackbody_intensity;
    float emission_pad;

    // ─────────────────────────────────────────────────────────────────────────
    // COLOR RAMP (Gradient)
    // ─────────────────────────────────────────────────────────────────────────
    int color_ramp_enabled;       // 0=Off, 1=On
    int ramp_stop_count;          // Number of active stops (max 8)
    float ramp_positions[8];      // Stop positions (0-1)
    float3 ramp_colors[8];        // Stop colors
    float ramp_pad;               // Padding for alignment
    
    // ─────────────────────────────────────────────────────────────────────────
    // RAY MARCHING QUALITY
    // ─────────────────────────────────────────────────────────────────────────
    float step_size;
    float voxel_size;  // Added for precision
    int max_steps;
    int shadow_steps;
    int shadow_stride;
    float shadow_strength;
    float max_temperature; // For normalization
    
    // ─────────────────────────────────────────────────────────────────────────
    // PIVOT & ANIMATION
    // ─────────────────────────────────────────────────────────────────────────
    float pivot_offset[3];
    int motion_blur_enabled;
    float velocity_scale;
    int vdb_id;

    // Procedural density source. 0 = NanoVDB/default, 3 = procedural cloud.
    int source_type;
    float cloud_coverage;
    float cloud_detail;
    float cloud_erosion;
    float cloud_base_scale;
    float cloud_edge_fade;
    float cloud_offset_x;
    float cloud_offset_z;
    float cloud_seed;
    // Isosurface dielectric IOR (source_type==4 fluid surface). Default water.
    float ior;
    // Isosurface surface roughness 0..1 (GGX). 0 = mirror-smooth.
    float surface_roughness;
    // Isosurface whitewater/foam strength 0..1 (curvature-driven).
    float surface_foam;
    // Particle-foam (whitewater) appearance for the SurfaceSDF single-volume
    // path: foam rides this volume's temperature channel and the iso shader
    // marches it. foam_color = scattering tint, foam_opacity = extinction
    // multiplier (foam_shader density × scatter). 0 → shader defaults.
    float3 foam_color;
    float  foam_opacity;

    // Material Graph density-field modulation (Vulkan spatial slice).
    int   density_noise_enabled;
    float density_noise_scale;
    float density_noise_strength;
    int   density_noise_detail;
    int   density_noise_seed;
    int   material_program_index = -1; // Vulkan Volume Graph table index; -1 = none
    // MaterialManager id shading an SDF isosurface through the full Principled
    // BSDF; -1 = built-in dielectric. Separate from material_program_index above
    // even though both are material-table ids: that one is gated on the graph
    // actually driving VOLUME closure slots, so a plain surface material resolves
    // to -1 there and would silently never reach the boundary.
    int   iso_material_id = -1;
    // Procedural porosity for the SDF isosurface — see
    // SimulationGridDomainDesc::fluid_surface_pore_*. Rides
    // VkVolumeInstance::_accel_reserved[0..2] (that block was declared in all
    // five ABI mirrors and never claimed by anything).
    float pore_amount = 0.0f;
    float pore_scale  = 0.05f;
    float pore_detail = 0.5f;
    // Which coordinate the isosurface addresses patterns in.
    // 0 = Material (carried by the liquid), 1 = Domain (carried by the vessel),
    // 2 = World (nailed to the room — the pre-material-coordinate behaviour).
    // Rides VkVolumeInstance::_accel_reserved[3], the last free float in that
    // block, so no ABI growth: the coordinate field's own address block already
    // took the struct from 576 to 608 and one enum does not justify moving it
    // again across five declarations.
    int   surface_coord_space = 0;
    // Path-traced multiple scattering (VolumeShader::ScatteringSettings::
    // random_walk). Vulkan RT only; rides VkVolumeInstance::random_walk_*,
    // carved from the retired cloud block, so no ABI growth. OptiX ignores it.
    int   random_walk = 0;
    int   random_walk_max_events = 32;
    int   random_walk_exact_events = 4;
    // Material coordinate (UVW) RESIDUAL field for the SDF isosurface — HOST
    // pointer to dense interleaved xyz triples at sim-grid resolution holding
    // (uvw - cell centre), plus its cell counts. The backend uploads it and
    // publishes a device address; nothing downstream of the upload reads this
    // pointer.
    //
    // ★ A displacement, not a coordinate: the consumer adds worldPos back. Named
    // for it so a consumer written against the old absolute meaning fails to
    // compile instead of rendering a liquid collapsed onto one texel.
    //
    // ★ Null (or any zero dimension) means "no coordinate this frame", and every
    // consumer must fall back to world anchoring. It may NEVER be treated as a
    // zero coordinate: the whole surface would map to one texel, which reads as
    // a flat-tinted liquid rather than as a missing feature.
    const float* uvw_residual_grid = nullptr;
    // Composition field, SAME grid/origin/voxel as the residual above (it is
    // gathered from the same particles with the same kernel), so it needs no
    // placement of its own. Null = nothing to blend.
    const float* composition_grid = nullptr;
    int uvw_dim[3] = { 0, 0, 0 };
    // World origin of cell (0,0,0) and the cell size, from the producer's grid.
    // ★ A zero voxel size means "not published" and the consumer must fall back
    // to world anchoring — it may never be substituted with the volume's render
    // bounds, which are a different box at a different resolution.
    float uvw_origin[3] = { 0.0f, 0.0f, 0.0f };
    float uvw_voxel = 0.0f;
    // Bumped by the producer on every refill. The backend re-uploads only when
    // it changes — the field is several megabytes and does not change on the
    // frames the surface was not rebuilt.
    unsigned int uvw_version = 0u;

    // Vulkan live gas-domain dense fields. Host-only transport metadata; OptiX
    // ignores these fields and continues to consume NanoVDB pointers above.
    uint64_t dense_density_address;
    uint64_t dense_temperature_address;
    uint64_t dense_fuel_address;
    uint64_t dense_flame_address;
    int dense_resolution_x;
    int dense_resolution_y;
    int dense_resolution_z;
    int dense_fields_valid;
    float3 dense_origin;
    // Per-block density maximum for the Vulkan RT empty-space skip
    // (dense_majorant_block^3 cells per entry). 0 = unavailable, and the shader
    // must then march every step — a missing majorant may never be read as
    // "empty" or the skip erases real smoke.
    uint64_t dense_majorant_address;
    int dense_majorant_dim[3];
    int dense_majorant_block;
    // Emitting-block list for volume emission NEE (0 = none).
    uint64_t dense_emissive_list_address;
    int dense_emissive_capacity;
    // ★★★ HOST MIRROR of the dense grids, for the consumer that CANNOT use the
    // addresses above. They are raw VkDeviceAddresses owned by the simulation
    // compute device; a second VkDevice (the dedicated raster viewport backend)
    // reads them as zero density, which is indistinguishable from empty smoke.
    // When set, the backend uploads these into ITS OWN buffer and publishes
    // that address instead — the same per-adapter pattern uvw_residual_grid
    // already uses. nullptr = this consumer owns the device and needs no copy.
    const float* dense_density_host;
    const float* dense_temperature_host;
    // Producer's dense_version. The backend re-uploads only when it changes:
    // the grid is megabytes and does not change on frames the sim did not step.
    uint64_t dense_host_version;
    int internal_light_steps = 0;
    int internal_light_stride = 8;
    float emission_response[5] = {};
    float density_curve[12] = {};
    float emission_mask[4] = {};
    float soot_burnout[3] = {};
    float temperature_gamma = 1.0f;
};

// ═══════════════════════════════════════════════════════════════════════════════
// GPU GAS VOLUME (Dense 3D Texture)
// ═══════════════════════════════════════════════════════════════════════════════
struct GpuGasVolume {
    // ─────────────────────────────────────────────────────────────────────────
    // DENSITY TEXTURE (3D Texture Object)
    // ─────────────────────────────────────────────────────────────────────────
    cudaTextureObject_t density_texture;
    cudaTextureObject_t temperature_texture;
    cudaTextureObject_t velocity_texture;
    int has_texture;
    int pad0;
    
    // ─────────────────────────────────────────────────────────────────────────
    // TRANSFORM (Object → World, row-major 3x4)
    // ─────────────────────────────────────────────────────────────────────────
    float transform[12];      // Object to world
    float inv_transform[12];  // World to object (for ray transform)
    
    // ─────────────────────────────────────────────────────────────────────────
    // BOUNDS
    // ─────────────────────────────────────────────────────────────────────────
    float3 world_bbox_min;    // World-space bounds
    float3 world_bbox_max;
    float3 local_bbox_min;    // Local bounds
    float3 local_bbox_max;
    
    // ─────────────────────────────────────────────────────────────────────────
    // SHADER PARAMETERS (Matches VDB Volume)
    // ─────────────────────────────────────────────────────────────────────────
    float density_multiplier;
    float density_remap_low;
    float density_remap_high;
    float density_pad;
    
    // Scattering
    float3 scatter_color;
    float scatter_coefficient;
    float scatter_anisotropy;       // G forward (-1 to 1)
    float scatter_anisotropy_back;  // G backward
    float scatter_lobe_mix;         // Forward/back blend
    float scatter_multi;            // Multi-scatter approximation
    
    // Absorption
    float3 absorption_color;
    float absorption_coefficient;
    
    // Emission (Fire/Explosions)
    int emission_mode;              // 0=None, 1=Constant, 2=Blackbody, 3=Channel
    float3 emission_color;
    float emission_intensity;
    float temperature_scale;
    float blackbody_intensity;
    float emission_pad;

    // Color Ramp (Gradient)
    int color_ramp_enabled;         // 0=Off, 1=On
    int ramp_stop_count;            // Number of active stops (max 8)
    float ramp_positions[8];        // Stop positions (0-1)
    float3 ramp_colors[8];          // Stop colors
    float ramp_pad;                 // Padding
    
    // Quality
    float step_size;
    int max_steps;
    int shadow_steps;
    float shadow_strength;
    float max_temperature; // For normalization
    float pad;
    int internal_light_steps = 0;
    int internal_light_stride = 8;
    float emission_response[5] = {};
    float density_curve[12] = {};
    float emission_mask[4] = {};
    float soot_burnout[3] = {};
    float temperature_gamma = 1.0f;
};

