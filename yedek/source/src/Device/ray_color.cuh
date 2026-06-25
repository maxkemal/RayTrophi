#pragma once
#include "trace_ray.cuh"
#include <math_constants.h>

#ifndef SCENE_EPSILON
#define SCENE_EPSILON 0.001f
#endif

#ifndef M_1_PIf
#define M_1_PIf 0.318309886183790671538f
#endif
#include "material_scatter.cuh"
#include "random_utils.cuh"
#include "CloudNoise.cuh"
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include "ray.h"
#include "scatter_volume_step.h"
#include "params.h"  // For GpuVDBVolume
#include "GodRaysModel.h"

// =============================================================================
// PHASE FUNCTIONS AND VOLUMETRIC HELPERS
// =============================================================================

// GPU helper: Dual-lobe Henyey-Greenstein phase function
__device__ float gpu_phase_dual_hg(float cos_theta, float g_forward, float g_back, float lobe_mix) {
    // Forward lobe
    float g2_fwd = g_forward * g_forward;
    float denom_fwd = 1.0f + g2_fwd - 2.0f * g_forward * cos_theta;
    float phase_fwd = (1.0f - g2_fwd) / (4.0f * M_PIf * powf(fmaxf(denom_fwd, 0.0001f), 1.5f));
    
    // Backward lobe
    float g2_back = g_back * g_back;
    float denom_back = 1.0f + g2_back - 2.0f * g_back * cos_theta;
    float phase_back = (1.0f - g2_back) / (4.0f * M_PIf * powf(fmaxf(denom_back, 0.0001f), 1.5f));
    
    return lobe_mix * phase_fwd + (1.0f - lobe_mix) * phase_back;
}

// GPU helper: Multi-scatter transmittance approximation (beer-style)
__device__ float gpu_multiscatter_transmittance(float sigma_t, float distance, float multi_scatter, float albedo_avg) {
    float T_single = expf(-sigma_t * distance);
    float T_multi = expf(-sigma_t * distance * 0.25f);
    float blend = multi_scatter * albedo_avg;
    return T_single * (1.0f - blend) + T_multi * blend;
}

// GPU helper: Powder effect for volume
__device__ float gpu_powder_effect(float density, float cos_theta) {
    float powder = 1.0f - expf(-density * 2.0f);
    float forward_bias = 0.5f + 0.5f * fmaxf(0.0f, cos_theta);
    return powder * forward_bias;
}

// Forward declaration for ambient sky radiance
__device__ float3 gpu_get_sky_radiance(const WorldData& world, const float3& dir);
__device__ float3 gpu_get_ambient_radiance_volume(const WorldData& world, const float3& dir);
__device__ float3 gpu_get_volume_ambient_dir(const WorldData& world, const float3& view_dir);
__device__ float sample_procedural_cloud_density(const GpuVDBVolume& vol, const float3& local_pos);

// GPU helper: Get transmittance from LUT (Pfe-computed)
__device__ float3 gpu_get_transmittance(const WorldData& world, float3 pos, float3 sunDir) {
    if (world.lut.transmittance_lut == 0) return make_float3(1.0f, 1.0f, 1.0f);
    
    float Rg = world.nishita.planet_radius;
    
    // COORDINATE SYNC: Camera Y=0 is planet surface. Center is at (0, -Rg, 0)
    float3 p = pos + make_float3(0, Rg, 0);
    float altitude = length(p) - Rg;
    float3 up = p / (Rg + altitude);
    float cosTheta = dot(up, sunDir);
    
    // UV Mapping (Matches Kernel: cosTheta = -0.2 + u * 1.2)
    float u = (cosTheta + 0.2f) / 1.2f; 
    float v = altitude / world.nishita.atmosphere_height;
    
    float4 tex = tex2D<float4>(world.lut.transmittance_lut, u, v);
    return make_float3(tex.x, tex.y, tex.z);
}

// ═══════════════════════════════════════════════════════════════════════════════
// VDB VOLUME RAY MARCHING (Industry-Standard)
// Independent volume objects with transform and NanoVDB sampling
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Convert temperature to blackbody RGB color (Planck approximation)
 * @param kelvin Temperature in Kelvin (1000K - 40000K)
 * @return RGB color normalized to 0-1
 */
__device__ float3 blackbody_to_rgb(float kelvin) {
    kelvin = fmaxf(1000.0f, fminf(40000.0f, kelvin));
    float temp = kelvin / 100.0f;
    
    float red, green, blue;
    
    // Red
    if (temp <= 66.0f) {
        red = 255.0f;
    } else {
        red = temp - 60.0f;
        red = 329.698727446f * powf(red, -0.1332047592f);
    }
    
    // Green
    if (temp <= 66.0f) {
        green = temp;
        green = 99.4708025861f * logf(green) - 161.1195681661f;
    } else {
        green = temp - 60.0f;
        green = 288.1221695283f * powf(green, -0.0755148492f);
    }
    
    // Blue
    if (temp >= 66.0f) {
        blue = 255.0f;
    } else if (temp <= 19.0f) {
        blue = 0.0f;
    } else {
        blue = temp - 10.0f;
        blue = 138.5177312231f * logf(blue) - 305.0447927307f;
    }
    
    return make_float3(
        fmaxf(0.0f, fminf(1.0f, red / 255.0f)),
        fmaxf(0.0f, fminf(1.0f, green / 255.0f)),
        fmaxf(0.0f, fminf(1.0f, blue / 255.0f))
    );
}

/**
 * @brief Stefan-Boltzmann intensity from temperature
 */
__device__ float blackbody_intensity(float kelvin, float scale) {
    float t_normalized = kelvin / 3000.0f;  // Normalize around flame temp
    return scale * t_normalized * t_normalized * t_normalized * t_normalized;
}

/**
 * @brief Transform a point using a 3x4 affine matrix (row-major)
 */
__device__ float3 transform_point_affine(const float3& p, const float* m) {
    return make_float3(
        m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
        m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
        m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]
    );
}

/**
 * @brief Transform a direction vector using a 3x4 affine matrix (ignores translation)
 */
__device__ float3 transform_vector_affine(const float3& v, const float* m) {
    return make_float3(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[4] * v.x + m[5] * v.y + m[6] * v.z,
        m[8] * v.x + m[9] * v.y + m[10] * v.z
    );
}

/**
 * @brief Ray-AABB intersection
 */
__device__ bool intersect_aabb_vdb(
    const float3& origin, const float3& dir,
    const float3& aabb_min, const float3& aabb_max,
    float& t_enter, float& t_exit
) {
    float3 inv_dir = make_float3(
        fabsf(dir.x) > 1e-6f ? 1.0f / dir.x : 1e6f * (dir.x >= 0 ? 1 : -1),
        fabsf(dir.y) > 1e-6f ? 1.0f / dir.y : 1e6f * (dir.y >= 0 ? 1 : -1),
        fabsf(dir.z) > 1e-6f ? 1.0f / dir.z : 1e6f * (dir.z >= 0 ? 1 : -1)
    );
    
    float3 t0 = (aabb_min - origin) * inv_dir;
    float3 t1 = (aabb_max - origin) * inv_dir;
    
    float3 tmin_v = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    float3 tmax_v = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
    
    t_enter = fmaxf(fmaxf(tmin_v.x, tmin_v.y), tmin_v.z);
    t_exit = fminf(fminf(tmax_v.x, tmax_v.y), tmax_v.z);
    
    return t_exit >= t_enter && t_exit > 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────
// VDB Occlusion Helper (for Shadow Rays)
// ─────────────────────────────────────────────────────────────────────────
// Helper to get local extinction (absorption + scattering) at a specific world position
__device__ float get_local_volumetric_extinction(float3 world_pos) {
    float total_extinction = 0.0f;
    
    // 1. VDB Volumes
    if (optixLaunchParams.vdb_volumes) {
        for (int v = 0; v < optixLaunchParams.vdb_volume_count; ++v) {
            const GpuVDBVolume& vol = optixLaunchParams.vdb_volumes[v];
            const bool procedural_cloud = (vol.source_type == 3);
            if (!procedural_cloud && !vol.density_grid) continue;
            
            float3 local_pos = transform_point_affine(world_pos, vol.inv_transform);
            
            // Check AABB
            if (local_pos.x >= vol.local_bbox_min.x && local_pos.x <= vol.local_bbox_max.x &&
                local_pos.y >= vol.local_bbox_min.y && local_pos.y <= vol.local_bbox_max.y &&
                local_pos.z >= vol.local_bbox_min.z && local_pos.z <= vol.local_bbox_max.z) {
                
                // Pivot offset adjustment
                float3 sample_pos = local_pos;
                sample_pos.x -= vol.pivot_offset[0];
                sample_pos.y -= vol.pivot_offset[1];
                sample_pos.z -= vol.pivot_offset[2];

                float d = 0.0f;
                if (procedural_cloud) {
                    d = sample_procedural_cloud_density(vol, sample_pos);
                } else {
                    nanovdb::FloatGrid* grid = (nanovdb::FloatGrid*)vol.density_grid;
                    nanovdb::math::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1> sampler(grid->tree());
                    nanovdb::Vec3f idx = grid->worldToIndexF(nanovdb::Vec3f(sample_pos.x, sample_pos.y, sample_pos.z));
                    d = sampler(idx);
                }
                float density = fmaxf((d - vol.density_remap_low) / (vol.density_remap_high - vol.density_remap_low + 1e-6f), 0.0f) * vol.density_multiplier;
                total_extinction += density * (vol.scatter_coefficient + vol.absorption_coefficient);
            }
        }
    }
    
    return total_extinction;
}

// Stochastic probe to find if a ray is occluded by volumes (Russian Roulette)
__device__ float get_stochastic_volumetric_occlusion(
    float3 ray_origin,
    float3 ray_dir,
    float max_dist,
    curandState* rng,
    float bias
) {
    float t_min = max_dist;
    
    // 1. VDB Volumes
    if (optixLaunchParams.vdb_volumes) {
        for (int v = 0; v < optixLaunchParams.vdb_volume_count; ++v) {
            const GpuVDBVolume& vol = optixLaunchParams.vdb_volumes[v];
            const bool procedural_cloud = (vol.source_type == 3);
            if (!procedural_cloud && !vol.density_grid) continue;
            
            float3 local_origin = transform_point_affine(ray_origin, vol.inv_transform);
            float3 local_dir = transform_vector_affine(normalize(ray_dir), vol.inv_transform);
            float dir_len = length(local_dir);
            if (dir_len < 1e-6f) continue;
            local_dir /= dir_len;
            
            float t0, t1;
            if (intersect_aabb_vdb(local_origin, local_dir, vol.local_bbox_min, vol.local_bbox_max, t0, t1)) {
                t0 = fmaxf(t0, 0.0f);
                t1 = fminf(t1, max_dist * dir_len);
                if (t1 > t0) {
                    float probe_t = t0 + (vol.step_size * 2.0f) * curand_uniform(rng);
                    float world_step = vol.step_size * 8.0f; 
                    float local_step = world_step * dir_len;
                    
                    float density_scale = fmaxf(0.01f, 1.0f - bias * 10.0f);
                    float p_factor = vol.density_multiplier * density_scale * (vol.scatter_coefficient + vol.absorption_coefficient) * world_step;

                    nanovdb::FloatGrid* grid = procedural_cloud ? nullptr : (nanovdb::FloatGrid*)vol.density_grid;

                    while (probe_t < t1) {
                        float3 pos = local_origin + local_dir * probe_t;
                        
                        // Pivot offset adjustment
                        pos.x -= vol.pivot_offset[0];
                        pos.y -= vol.pivot_offset[1];
                        pos.z -= vol.pivot_offset[2];

                        float d = 0.0f;
                        if (procedural_cloud) {
                            d = sample_procedural_cloud_density(vol, pos);
                        } else {
                            nanovdb::math::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1> sampler(grid->tree());
                            nanovdb::Vec3f idx = grid->worldToIndexF(nanovdb::Vec3f(pos.x, pos.y, pos.z));
                            d = sampler(idx);
                        }
                        float raw_d = fmaxf(0.0f, d - vol.density_remap_low);
                        
                        if (raw_d > 1e-5f) {
                            // --- SMOOTH HYBRID OCCLUSION ---
                            // To prevent light leaking through thick clouds while keeping soft edges:
                            // 1. Calculate Beer's Law probability for this step
                            float prob = (1.0f - expf(-raw_d * p_factor));
                            
                            // 2. Smoothly boost probability towards 100% as density increases
                            // This ensures thick clouds block DETERMINISTICALLY while thin edges remain STOCHASTIC.
                            // Using a fixed threshold for hybrid occlusion logic
                            float stochastic_threshold = 0.1f;
                            float boost = fminf(1.0f, raw_d / fmaxf(1e-5f, stochastic_threshold));
                            
                            // Using quadratic boost for a more cinematic falloff
                            float final_prob = lerp(prob, 1.0f, boost * boost);

                            if (curand_uniform(rng) < final_prob) {
                                return probe_t / dir_len;
                            }
                        }
                        probe_t += local_step;
                    }
                }
            }
        }
    }
    
    return t_min;
}

__device__ float calculate_vdb_occlusion(
    const float3& ray_origin,
    const float3& ray_dir,
    float max_dist,
    curandState* rng
) {
    float total_transmittance = 1.0f;
    
    if (!optixLaunchParams.vdb_volumes || optixLaunchParams.vdb_volume_count == 0) return 1.0f;

    for (int v = 0; v < optixLaunchParams.vdb_volume_count; ++v) {
        const GpuVDBVolume& vol = optixLaunchParams.vdb_volumes[v];
        const bool procedural_cloud = (vol.source_type == 3);
        if (!procedural_cloud && !vol.density_grid) continue;
        
        // Transform ray to VDB local space
        float3 local_origin = transform_point_affine(ray_origin, vol.inv_transform);
        float3 local_dir = transform_vector_affine(ray_dir, vol.inv_transform);
        float dir_length = length(local_dir);
        if (dir_length < 1e-8f) continue;
        
        local_dir = local_dir / dir_length;
        
        // Ray-AABB intersection
        float t0, t1;
        if (!intersect_aabb_vdb(local_origin, local_dir, vol.local_bbox_min, vol.local_bbox_max, t0, t1)) continue;
        
        // Clip t1 with max_dist (converted to local scale)
        float local_max_dist = max_dist * dir_length;
        t1 = fminf(t1, local_max_dist);
        
        if (t0 >= t1) continue;
        
        t0 = fmaxf(t0, 0.0f);
        
        // Step size logic
        float world_step = fmaxf(vol.step_size, (vol.world_bbox_max.x - vol.world_bbox_min.x) / (float)max(8, vol.shadow_steps));
        float local_step = world_step * dir_length;
        
        // Jitter
        float t = t0 + local_step * random_float(rng); 
        
        nanovdb::FloatGrid* grid = procedural_cloud ? nullptr : (nanovdb::FloatGrid*)vol.density_grid;
        
        float remap_range = fmaxf(0.001f, vol.density_remap_high - vol.density_remap_low);
        float density_sum = 0.0f;
        int steps = 0;
        int max_s = max(16, vol.shadow_steps * 2);

        // Ray Marching Loop for Shadow
        while (t < t1 && steps < max_s) {
            float3 pos = local_origin + local_dir * t;
            
            // Pivot offset adjustment
            pos.x -= vol.pivot_offset[0];
            pos.y -= vol.pivot_offset[1];
            pos.z -= vol.pivot_offset[2];

            float d = 0.0f;
            if (procedural_cloud) {
                d = sample_procedural_cloud_density(vol, pos);
            } else {
                auto acc = grid->getAccessor();
                nanovdb::math::SampleFromVoxels<nanovdb::FloatGrid::AccessorType, 1, false> sampler(acc);
                nanovdb::Vec3f idx = grid->worldToIndexF(nanovdb::Vec3f(pos.x, pos.y, pos.z));
                d = sampler(idx);
            }
            float density = fmaxf((d - vol.density_remap_low) / remap_range, 0.0f) * vol.density_multiplier;
            
            // Attenuate by scatter + absorption (sigma_t)
            float sigma_t = density * (vol.scatter_coefficient + vol.absorption_coefficient);
            density_sum += sigma_t * world_step;
            
            if (density_sum > 10.0f) break; // Fully occluded
            
            t += local_step;
            steps++;
        }
        
        // Boost shadow strength to match CPU "solid" behavior for God Rays
        float tr = expf(-density_sum * vol.shadow_strength * 2.0f);
        total_transmittance *= tr;
        
        if (total_transmittance < 0.01f) return 0.0f;
    }
    
    return total_transmittance;
}

// Helper to sample GPU Color Ramp
__device__ float3 sample_color_ramp(const GpuVDBVolume& vol, float t) {
    if (vol.ramp_stop_count == 0) return make_float3(1.0f);
    
    // Clamp t to first/last stop
    if (t <= vol.ramp_positions[0]) return vol.ramp_colors[0];
    if (t >= vol.ramp_positions[vol.ramp_stop_count - 1]) return vol.ramp_colors[vol.ramp_stop_count - 1];
    
    // Linear search (max 8 stops, fast enough)
    for (int i = 1; i < vol.ramp_stop_count; ++i) {
        if (t <= vol.ramp_positions[i]) {
            float t0 = vol.ramp_positions[i-1];
            float t1 = vol.ramp_positions[i];
            float blend = (t - t0) / (t1 - t0);
            return vol.ramp_colors[i-1] * (1.0f - blend) + vol.ramp_colors[i] * blend;
        }
    }
    return vol.ramp_colors[vol.ramp_stop_count - 1];
}

// Helper to sample GPU Color Ramp for Legacy Gas
__device__ float3 sample_color_ramp_gas(const GpuGasVolume& vol, float t) {
    if (vol.ramp_stop_count == 0) return make_float3(1.0f);
    if (t <= vol.ramp_positions[0]) return vol.ramp_colors[0];
    if (t >= vol.ramp_positions[vol.ramp_stop_count - 1]) return vol.ramp_colors[vol.ramp_stop_count - 1];
    for (int i = 1; i < vol.ramp_stop_count; ++i) {
        if (t <= vol.ramp_positions[i]) {
            float t0 = vol.ramp_positions[i-1];
            float t1 = vol.ramp_positions[i];
            float blend = (t - t0) / (t1 - t0);
            return vol.ramp_colors[i-1] * (1.0f - blend) + vol.ramp_colors[i] * blend;
        }
    }
    return vol.ramp_colors[vol.ramp_stop_count - 1];
}

__device__ float3 clamp_volume_radiance(const float3& c, float max_luma) {
    float l = c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f;
    if (l > max_luma && l > 1e-6f) {
        return c * (max_luma / l);
    }
    return c;
}

__device__ float sample_procedural_cloud_density(const GpuVDBVolume& vol, const float3& local_pos) {
    float3 extent = vol.local_bbox_max - vol.local_bbox_min;
    float3 norm_pos = (local_pos - vol.local_bbox_min) / make_float3(
        fmaxf(extent.x, 1e-5f),
        fmaxf(extent.y, 1e-5f),
        fmaxf(extent.z, 1e-5f));

    if (norm_pos.x < 0.0f || norm_pos.x > 1.0f ||
        norm_pos.y < 0.0f || norm_pos.y > 1.0f ||
        norm_pos.z < 0.0f || norm_pos.z > 1.0f) {
        return 0.0f;
    }

    float base_scale = fmaxf(vol.cloud_base_scale, 1.0f);
    float3 cloud_pos = make_float3(
        norm_pos.x * base_scale + vol.cloud_offset_x,
        norm_pos.y * 1.35f,
        norm_pos.z * base_scale + vol.cloud_offset_z);
    cloud_pos += make_float3(vol.cloud_seed * 0.137f, vol.cloud_seed * 0.317f, vol.cloud_seed * 0.719f);

    float coverage = fmaxf(0.0f, fminf(1.0f, vol.cloud_coverage));
    float detail = fmaxf(0.0f, fminf(1.0f, vol.cloud_detail));
    float erosion = fmaxf(0.0f, fminf(1.0f, vol.cloud_erosion));

    float warp_x = fbm(make_float3(cloud_pos.x * 0.38f, cloud_pos.y * 0.16f, cloud_pos.z * 0.38f) + make_float3(11.0f, 0.0f, 7.0f), 2) - 0.5f;
    float warp_z = fbm(make_float3(cloud_pos.x * 0.38f, cloud_pos.y * 0.16f, cloud_pos.z * 0.38f) + make_float3(41.0f, 3.0f, 23.0f), 2) - 0.5f;
    float3 warped = cloud_pos + make_float3(warp_x * 1.35f, 0.0f, warp_z * 1.35f);

    float base = fbm(make_float3(warped.x * 0.52f, warped.y * 0.28f, warped.z * 0.52f), 4);
    float billow = 1.0f - fabsf(fbm(make_float3(warped.x * 1.15f, warped.y * 0.5f, warped.z * 1.15f) + make_float3(17.0f, 3.0f, 11.0f), 4) * 2.0f - 1.0f);
    float detail_noise = fbm(warped * lerp(2.8f, 7.0f, detail) + make_float3(31.0f, 7.0f, 19.0f), 2);

    float puffy = smoothstep(0.32f, 0.88f, billow);
    float shape = lerp(base, base * 0.45f + puffy * 0.75f, 0.72f);
    shape -= detail_noise * lerp(0.06f, 0.28f, erosion);

    float threshold = lerp(0.78f, 0.30f, coverage);
    float density = fmaxf((shape - threshold) / fmaxf(1.0f - threshold, 1e-4f), 0.0f);

    float bottom = smoothstep(0.12f, 0.42f, norm_pos.y);
    float top = 1.0f - smoothstep(0.72f, 1.02f, norm_pos.y);
    float3 edge = make_float3(0.5f) - make_float3(
        fabsf(norm_pos.x - 0.5f),
        fabsf(norm_pos.y - 0.5f),
        fabsf(norm_pos.z - 0.5f));
    float edge_falloff = smoothstep(0.0f, fmaxf(vol.cloud_edge_fade, 0.02f), fminf(edge.x, edge.z));
    return density * density * bottom * top * edge_falloff * 4.6f;
}

/**
 * @brief VDB Volume Ray Marching (GPU Kernel)
 * 
 * High-performance GPU path for OpenVDB/NanoVDB volumetrics.
 * Uses Trilinear sampling for maximum efficiency in both viewport and final render.
 */
__device__ float3 raymarch_vdb_volume(
    const GpuVDBVolume& vol,
    const float3& ray_origin,
    const float3& ray_dir,
    const float3& sun_dir,
    float sun_intensity,
    float& out_transmittance,
    float max_t,
    curandState* rng,
    // Fluid-surface (source_type==4) dielectric redirect. When the function
    // hits an isosurface it fills these so the bounce loop can continue the
    // path along the REFRACTED ray (true through-water distortion) instead of
    // treating the surface as opaque. Returned colour is the reflection lobe
    // (sky * Fresnel) to add immediately; out_surface_throughput is the
    // (1-Fresnel)*depth-tint multiplier for the refracted continuation.
    bool*   out_surface_redirect = nullptr,
    float3* out_surface_origin   = nullptr,
    float3* out_surface_dir      = nullptr,
    float3* out_surface_throughput = nullptr
) {
    if (out_surface_redirect) *out_surface_redirect = false;
    const bool use_procedural_cloud = (vol.source_type == 3);
    if (!use_procedural_cloud && vol.density_grid == nullptr) {
        out_transmittance = 1.0f;
        return make_float3(0.0f);
    }
    
    // Transform ray to VDB local space
    float3 local_origin = transform_point_affine(ray_origin, vol.inv_transform);
    float3 local_dir_world = transform_vector_affine(ray_dir, vol.inv_transform);
    float dir_length = length(local_dir_world);
    if (dir_length < 1e-8f) {
        out_transmittance = 1.0f;
        return make_float3(0.0f);
    }
    float3 local_dir = local_dir_world / dir_length;
    float3 ambient_dir = gpu_get_volume_ambient_dir(optixLaunchParams.world, ray_dir);
    
    // Ray-AABB intersection in local space
    float t_enter, t_exit;
    if (!intersect_aabb_vdb(local_origin, local_dir, vol.local_bbox_min, vol.local_bbox_max, t_enter, t_exit)) {
        out_transmittance = 1.0f;
        return make_float3(0.0f);
    }
    
    // Convert world max_t to local space distance
    float local_max_t = max_t * dir_length;
    t_exit = fminf(t_exit, local_max_t);
    t_enter = fmaxf(t_enter, 0.001f);
    
    if (t_enter >= t_exit) {
        out_transmittance = 1.0f;
        return make_float3(0.0f);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // FLUID SURFACE (source_type == 4) — simplified isosurface mode
    // ══════════════════════════════════════════════════════════════════════════
    // The density channel is the SDF proxy band (0..1). Walk for the first
    // iso=0.5 crossing, build a Fresnel-tinted opaque surface colour. True
    // Snell refraction lives in the Vulkan path (closesthit can redirect the
    // payload); this OptiX path accumulates along a single ray so we model
    // the surface as an opaque-with-fresnel boundary. Visually the fluid
    // reads as a solid volume; through-glass distortion is a follow-up that
    // needs the bounce loop to be restructured (memory: optix-perf-ceiling).
    if (vol.source_type == 4 && !use_procedural_cloud && vol.density_grid != nullptr) {
        // The main raymarch creates its own `grid` variable later; we need our
        // own here (iso branch returns before reaching it).
        const nanovdb::FloatGrid* iso_grid =
            reinterpret_cast<const nanovdb::FloatGrid*>(vol.density_grid);
        const float ISO_THRESH = 0.5f;
        // Walk quality: step = max(fine, cover) so max_steps always reach
        // t_exit — a fixed fine step + low cap truncates the walk and skips the
        // far side of the fluid. Mirrors the Vulkan iso path.
        const float local_extent = t_exit - t_enter;
        int   iso_cap = vol.max_steps;
        if (iso_cap < 32) iso_cap = 32; else if (iso_cap > 2048) iso_cap = 2048;
        float qlo = vol.voxel_size * 0.1f, qhi = vol.voxel_size * 0.5f;
        float fine_step = fminf(fmaxf(vol.step_size, qlo), qhi);
        float cover_step = local_extent / (float)iso_cap;
        const float local_step = fmaxf(0.001f, fmaxf(fine_step, cover_step));
        int   max_steps_iso = (int)(local_extent / local_step) + 2;
        if (max_steps_iso > iso_cap + 2) max_steps_iso = iso_cap + 2;

        using AccTIso = nanovdb::FloatGrid::AccessorType;
        AccTIso accIso = iso_grid->getAccessor();
        nanovdb::math::SampleFromVoxels<AccTIso, 1, false> samplerIso(accIso);

        // Apply the same density remap + multiplier the Vulkan iso path uses
        // (sampleDensityAcc returns remapped*multiplier), so the UI density
        // value moves the iso=0.5 crossing identically on both backends. Without
        // this OptiX sampled the raw 0..1 proxy and ignored the density slider.
        const float iso_remap_range = vol.density_remap_high - vol.density_remap_low + 1e-6f;
        const float iso_mult = (vol.density_multiplier > 0.0f) ? vol.density_multiplier : 1.0f;
        auto sample_at_local = [&](const float3& lp) -> float {
            float3 pp = lp;
            pp.x -= vol.pivot_offset[0];
            pp.y -= vol.pivot_offset[1];
            pp.z -= vol.pivot_offset[2];
            nanovdb::Vec3f idx = iso_grid->worldToIndexF(nanovdb::Vec3f(pp.x, pp.y, pp.z));
            float v = samplerIso(idx);
            if (!isfinite(v)) return 0.0f;
            float remapped = fmaxf((v - vol.density_remap_low) / iso_remap_range, 0.0f);
            return remapped * iso_mult;
        };

        float t_iso = t_enter;
        float3 p0 = local_origin + local_dir * t_iso;
        float start_d = sample_at_local(p0);
        bool  start_inside = start_d > ISO_THRESH;   // ray began inside the fluid
        float prev_d = start_d;
        float hit_t = -1.0f;
        for (int s = 0; s < max_steps_iso; ++s) {
            float next_t = fminf(t_iso + local_step, t_exit);
            float3 pn = local_origin + local_dir * next_t;
            float cur_d = sample_at_local(pn);
            bool crossed = start_inside
                ? (prev_d >= ISO_THRESH && cur_d < ISO_THRESH)   // exit
                : (prev_d <  ISO_THRESH && cur_d >= ISO_THRESH);  // enter
            if (crossed) {
                float denom = cur_d - prev_d;
                float frac = (fabsf(denom) > 1e-6f) ? ((ISO_THRESH - prev_d) / denom) : 0.5f;
                if (frac < 0.0f) frac = 0.0f;
                if (frac > 1.0f) frac = 1.0f;
                hit_t = t_iso + frac * (next_t - t_iso);
                break;
            }
            prev_d = cur_d;
            t_iso = next_t;
            if (t_iso >= t_exit) break;
        }

        // ── Whitewater foam composited in FRONT of the surface ──────────────
        // Foam rides this same volume's TEMPERATURE channel (scaled by
        // FOAM_TEMP_SCALE on upload; see SceneData::syncSimulationRenderVolumes),
        // so there is exactly ONE volume on the domain AABB — no coincident-
        // volume sort, correct on both backends. March it as a bright white
        // single-scatter medium over the span in front of the iso surface and
        // pre-multiply everything behind it by the foam transmittance. NO self-
        // shadow term: foam is bright, and self-shadowing is exactly what turned
        // the old separate fog volume into a black cube on Vulkan.
        float3 foam_inscatter = make_float3(0.0f);
        float  foam_T = 1.0f;
        const nanovdb::FloatGrid* foam_grid =
            reinterpret_cast<const nanovdb::FloatGrid*>(vol.temperature_grid);
        if (foam_grid != nullptr) {
            const float FOAM_TEMP_SCALE = 10000.0f;
            // Extinction multiplier + tint ride the volume (set from the domain
            // foam shader at sync) so the "Foam Volume Shader" panel drives the
            // look; fall back to sane defaults when unset.
            const float FOAM_OPTICAL = (vol.foam_opacity > 1e-3f) ? vol.foam_opacity : 8.0f;
            float3 foam_albedo = (vol.foam_color.x + vol.foam_color.y + vol.foam_color.z > 1e-3f)
                ? vol.foam_color : make_float3(0.95f, 0.97f, 1.0f);
            using AccTF = nanovdb::FloatGrid::AccessorType;
            AccTF accF = foam_grid->getAccessor();
            nanovdb::math::SampleFromVoxels<AccTF, 1, false> samplerF(accF);
            auto sample_foam = [&](const float3& lp) -> float {
                float3 pp = lp;
                pp.x -= vol.pivot_offset[0];
                pp.y -= vol.pivot_offset[1];
                pp.z -= vol.pivot_offset[2];
                nanovdb::Vec3f fidx = foam_grid->worldToIndexF(nanovdb::Vec3f(pp.x, pp.y, pp.z));
                float v = samplerF(fidx);
                if (!isfinite(v) || v <= 0.0f) return 0.0f;
                return v * (1.0f / FOAM_TEMP_SCALE);
            };
            const float foam_end = (hit_t < 0.0f) ? t_exit : hit_t;
            const float foam_len = foam_end - t_enter;
            if (foam_len > 1e-5f) {
                int fsteps = (int)(foam_len / fmaxf(vol.voxel_size, foam_len / 48.0f)) + 1;
                if (fsteps > 64) fsteps = 64;
                const float fstep = foam_len / (float)fsteps;
                const float3 up = make_float3(0.0f, 1.0f, 0.0f);
                const float3 sky_amb = gpu_get_sky_radiance(optixLaunchParams.world, up);
                const float jitf = curand_uniform(rng);
                // ── Ray-constant shading terms (HOISTED out of the per-step loop) ──
                // The view/sun geometry is identical for every foam sample on this
                // ray, so the HG phase + thin-film cos() are evaluated ONCE per ray
                // instead of per sample (was per-step before — the OptiX cost the
                // user noticed). Only fd / position terms stay in the loop.
                const float3 sunDir = optixLaunchParams.world.nishita.sun_direction;
                const float  vlen   = fmaxf(dir_length, 1e-6f);
                const float  cosVS  = (ray_dir.x * sunDir.x + ray_dir.y * sunDir.y + ray_dir.z * sunDir.z) / vlen;
                const float  g = 0.35f, g2 = g * g;
                const float  hgd       = 1.0f + g2 - 2.0f * g * cosVS;
                const float  foamPhase = (1.0f - g2) / fmaxf(4.0f * 3.14159265f * hgd * sqrtf(fmaxf(hgd, 1e-6f)), 1e-6f);
                const float  powderCos = 0.5f + 0.5f * fmaxf(0.0f, cosVS);   // constant part of powder
                // Thin-film iridescence (soap-bubble shimmer): pearlescent tint on the
                // silver lining — also ray-constant.
                const float3 silver = make_float3(
                    1.0f - 0.15f + 0.15f * (0.5f + 0.5f * cosf(cosVS * 9.0f + 0.0f)),
                    1.0f - 0.15f + 0.15f * (0.5f + 0.5f * cosf(cosVS * 9.0f + 2.0f)),
                    1.0f - 0.15f + 0.15f * (0.5f + 0.5f * cosf(cosVS * 9.0f + 4.0f)));
                // Foam shares the volume's Edge Cutoff (density_pad) so the faint
                // low-density foam fringe is clipped just like the water density —
                // otherwise it leaves a grey haze the density cutoff can't reach.
                const float foamCutoff = fmaxf(vol.density_pad, 1e-4f);
                for (int fs = 0; fs < fsteps && foam_T > 0.01f; ++fs) {
                    float ft = t_enter + ((float)fs + jitf) * fstep;
                    float fd = sample_foam(local_origin + local_dir * ft);
                    if (fd <= foamCutoff) continue;
                    float a = 1.0f - expf(-fd * FOAM_OPTICAL * (fstep / fmaxf(dir_length, 1e-6f)));
                    // Soft edge falloff over [cutoff, 2·cutoff] so the clip is a clean
                    // fade, not a hard ring (smoothstep, manual for CUDA).
                    float fe = fminf(fmaxf((fd - foamCutoff) / fmaxf(foamCutoff, 1e-6f), 0.0f), 1.0f);
                    a *= fe * fe * (3.0f - 2.0f * fe);
                    float3 wp = ray_origin + ray_dir * (ft / fmaxf(dir_length, 1e-6f));
                    float3 sun_tr = gpu_get_transmittance(optixLaunchParams.world, wp, sunDir);
                    // Forward-scatter silver lining (HG toward sun → backlit crests
                    // glow) + gentle powder (edge-dark/core-bright). Sky ambient is an
                    // un-powdered fill so foam never goes black.
                    float powder   = 0.75f + 0.25f * ((1.0f - expf(-fd * 2.0f)) * powderCos);
                    float sunBoost = sun_intensity * 0.5f * (1.0f + foamPhase * 4.0f) * powder;
                    // Multiple scattering: packed bubbles diffuse light through the mass
                    // → thick foam reads as a bright FILLED white, not a thin wisp.
                    float ms = 1.0f - expf(-fd * 6.0f);                  // 0 thin → 1 thick
                    float3 msL = make_float3(
                        sky_amb.x + 0.4f * sun_tr.x * sun_intensity,
                        sky_amb.y + 0.4f * sun_tr.y * sun_intensity,
                        sky_amb.z + 0.4f * sun_tr.z * sun_intensity);
                    float3 Li = make_float3(
                        foam_albedo.x * (sky_amb.x + sun_tr.x * sunBoost * silver.x + 1.2f * ms * msL.x),
                        foam_albedo.y * (sky_amb.y + sun_tr.y * sunBoost * silver.y + 1.2f * ms * msL.y),
                        foam_albedo.z * (sky_amb.z + sun_tr.z * sunBoost * silver.z + 1.2f * ms * msL.z));
                    foam_inscatter.x += foam_T * a * Li.x;
                    foam_inscatter.y += foam_T * a * Li.y;
                    foam_inscatter.z += foam_T * a * Li.z;
                    foam_T *= (1.0f - a);
                }
            }
        }

        if (hit_t < 0.0f) {
            // No surface, but airborne foam/spray in this segment still shows.
            out_transmittance = foam_T;
            return foam_inscatter;
        }

        // Central-difference gradient -> surface normal in local space.
        float h_iso = fmaxf(0.001f, vol.voxel_size);
        float3 hit_local = local_origin + local_dir * hit_t;
        float sxp = sample_at_local(make_float3(hit_local.x + h_iso, hit_local.y, hit_local.z));
        float sxm = sample_at_local(make_float3(hit_local.x - h_iso, hit_local.y, hit_local.z));
        float syp = sample_at_local(make_float3(hit_local.x, hit_local.y + h_iso, hit_local.z));
        float sym = sample_at_local(make_float3(hit_local.x, hit_local.y - h_iso, hit_local.z));
        float szp = sample_at_local(make_float3(hit_local.x, hit_local.y, hit_local.z + h_iso));
        float szm = sample_at_local(make_float3(hit_local.x, hit_local.y, hit_local.z - h_iso));
        float gx = sxp - sxm, gy = syp - sym, gz = szp - szm;
        float glen = sqrtf(gx * gx + gy * gy + gz * gz);
        float3 N_local;
        if (glen > 1e-6f) {
            float inv = -1.0f / glen;   // -gradient points OUT of denser interior
            N_local = make_float3(gx * inv, gy * inv, gz * inv);
        } else {
            N_local = make_float3(-local_dir.x, -local_dir.y, -local_dir.z);
        }

        // Foam / whitewater from SDF curvature (Laplacian). White light added
        // immediately (not redirected). Reuses the 6 gradient samples + centre.
        float3 foam_add = make_float3(0.0f, 0.0f, 0.0f);
        {
            float foam_strength = vol.surface_foam;
            if (foam_strength < 0.0f) foam_strength = 0.0f;
            else if (foam_strength > 1.0f) foam_strength = 1.0f;
            if (foam_strength > 1e-3f) {
                float dc = sample_at_local(hit_local);
                float lap = fabsf((sxp + sxm + syp + sym + szp + szm) - 6.0f * dc);
                float ft = (lap - 0.15f) / (0.7f - 0.15f);
                ft = fminf(fmaxf(ft, 0.0f), 1.0f);
                ft = ft * ft * (3.0f - 2.0f * ft);   // smoothstep
                float foam = foam_strength * ft * 0.9f;
                foam_add = make_float3(foam, foam, foam);
            }
        }

        const float ior_w = (vol.ior > 1.0f) ? vol.ior : 1.33f;

        // World-space surface normal (vol.transform is rigid/uniform-scale for
        // fluid domains), oriented against the incoming ray.
        float3 N_world = normalize(transform_vector_affine(N_local, vol.transform));
        if (dot(N_world, ray_dir) > 0.0f) N_world = make_float3(-N_world.x, -N_world.y, -N_world.z);

        // GGX roughness perturbation (vol.surface_roughness). Jitter the normal
        // inside a microfacet lobe so reflection AND refraction blur. 0 = mirror.
        float rough = vol.surface_roughness;
        if (rough < 0.0f) rough = 0.0f; else if (rough > 1.0f) rough = 1.0f;
        if (rough > 1e-3f) {
            float a = rough * rough;
            float u1 = curand_uniform(rng);
            float u2 = curand_uniform(rng);
            float phi = 6.2831853f * u1;
            float cosT = sqrtf(fmaxf(0.0f, (1.0f - u2) / (1.0f + (a * a - 1.0f) * u2)));
            float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
            float3 hT = make_float3(sinT * cosf(phi), sinT * sinf(phi), cosT);
            float3 up = (fabsf(N_world.z) < 0.999f) ? make_float3(0.0f, 0.0f, 1.0f)
                                                    : make_float3(1.0f, 0.0f, 0.0f);
            float3 T = normalize(cross(up, N_world));
            float3 B = cross(N_world, T);
            float3 Np = normalize(make_float3(hT.x * T.x + hT.y * B.x + hT.z * N_world.x,
                                              hT.x * T.y + hT.y * B.y + hT.z * N_world.y,
                                              hT.x * T.z + hT.y * B.z + hT.z * N_world.z));
            if (dot(ray_dir, Np) < 0.0f) N_world = Np;
        }

        // Schlick Fresnel with the (possibly perturbed) normal.
        float cos_t = fabsf(dot(ray_dir, N_world));
        if (cos_t > 1.0f) cos_t = 1.0f;
        float r0 = (1.0f - ior_w) / (1.0f + ior_w);
        r0 = r0 * r0;
        float fres = r0 + (1.0f - r0) * powf(1.0f - cos_t, 5.0f);

        // Beer-Lambert over the segment just traversed inside the fluid (only
        // when the ray started inside — exit / internal event).
        float3 seg_tp = make_float3(1.0f, 1.0f, 1.0f);
        if (start_inside) {
            float depth_world = (hit_t - t_enter) / fmaxf(dir_length, 1e-6f);
            seg_tp = make_float3(
                expf(-vol.absorption_color.x * vol.absorption_coefficient * depth_world),
                expf(-vol.absorption_color.y * vol.absorption_coefficient * depth_world),
                expf(-vol.absorption_color.z * vol.absorption_coefficient * depth_world));
        }

        // Fresnel importance-sampled dielectric: reflect (scene/sky) vs refract
        // (through). The chosen branch is redirected through the bounce loop, so
        // reflection traces the real scene — not just the sky — exactly like the
        // Vulkan path. Per-branch weight is 1 (probability cancels Fresnel).
        if (out_surface_redirect && out_surface_origin && out_surface_dir && out_surface_throughput) {
            float eta = start_inside ? ior_w : (1.0f / ior_w);
            float3 refr_dir;
            bool ok = refract(ray_dir, N_world, eta, &refr_dir);
            float3 cont_dir;
            float3 branch_tp;
            if (!ok || curand_uniform(rng) < fres) {
                cont_dir = reflect(ray_dir, N_world);     // scene/sky reflection
                branch_tp = make_float3(1.0f, 1.0f, 1.0f);
            } else {
                cont_dir = normalize(refr_dir);            // through the water
                branch_tp = vol.scatter_color;             // mild surface cast
            }
            float hit_world_t = hit_t / fmaxf(dir_length, 1e-6f);
            float3 hit_world = ray_origin + ray_dir * hit_world_t;
            *out_surface_origin = hit_world + cont_dir * 1e-3f;
            *out_surface_dir = cont_dir;
            // Foam in front dims whatever the redirected lobe brings back.
            *out_surface_throughput = make_float3(seg_tp.x * branch_tp.x * foam_T,
                                                  seg_tp.y * branch_tp.y * foam_T,
                                                  seg_tp.z * branch_tp.z * foam_T);
            *out_surface_redirect = true;
            out_transmittance = 1.0f;          // throughput handled by the caller
            return foam_add + foam_inscatter;   // surface sheen + particle whitewater
        }

        // Fallback (no redirect out-params): opaque Fresnel + sky reflection.
        float3 refl_dir = reflect(ray_dir, N_world);
        float3 sky_refl = gpu_get_sky_radiance(optixLaunchParams.world, refl_dir);
        float3 depth_tint = make_float3(vol.scatter_color.x * seg_tp.x,
                                        vol.scatter_color.y * seg_tp.y,
                                        vol.scatter_color.z * seg_tp.z);
        float3 result;
        result.x = (depth_tint.x * (1.0f - fres) + sky_refl.x * fres) * foam_T + foam_add.x + foam_inscatter.x;
        result.y = (depth_tint.y * (1.0f - fres) + sky_refl.y * fres) * foam_T + foam_add.y + foam_inscatter.y;
        result.z = (depth_tint.z * (1.0f - fres) + sky_refl.z * fres) * foam_T + foam_add.z + foam_inscatter.z;
        out_transmittance = 0.05f;
        return result;
    }

    // Step size - ADAPTIVE PRECISION
    float world_t_enter = t_enter / dir_length;
    float world_t_exit = t_exit / dir_length;
    float world_vol_extent = world_t_exit - world_t_enter;

    // Safety Step Calculation: Ensures we reach the exit before hitting max_steps
    float min_step_to_cover = world_vol_extent / fmaxf(1.0f, (float)vol.max_steps - 1.0f);
    float base_step = fmaxf(vol.step_size, min_step_to_cover);
    
    // Quality clamp: Ensure we don't use a step larger than the voxel size if requested small
    float world_scale = 1.0f / fmaxf(1e-6f, length(make_float3(vol.inv_transform[0], vol.inv_transform[1], vol.inv_transform[2])));
    float world_voxel_size = vol.voxel_size * world_scale;
    base_step = fmaxf(0.0001f, fminf(base_step, fmaxf(world_voxel_size, world_vol_extent / 16.0f)));
    
    // Re-verify that step will actually cover the volume (final safety)
    base_step = fmaxf(base_step, min_step_to_cover);
    float min_step = fmaxf(base_step * 0.25f, 0.0001f);
    const float tau_max = 0.2f;

    // Raymarching State
    float jitter = curand_uniform(rng) * base_step;
    float t = world_t_enter + jitter;
    float3 transmittance = make_float3(1.0f);
    float3 accumulated_color = make_float3(0.0f);
    
    // Grid Setup
    const nanovdb::FloatGrid* grid = use_procedural_cloud ? nullptr : reinterpret_cast<const nanovdb::FloatGrid*>(vol.density_grid);
    
    float remap_range = vol.density_remap_high - vol.density_remap_low + 1e-6f;
    float3 local_sun_dir = normalize(transform_vector_affine(sun_dir, vol.inv_transform));
    float3 local_ray_dir = normalize(transform_vector_affine(ray_dir, vol.inv_transform));
    
    const nanovdb::FloatGrid* temp_grid = (vol.temperature_grid && vol.emission_mode >= 2) ? 
        reinterpret_cast<const nanovdb::FloatGrid*>(vol.temperature_grid) : nullptr;
    
    int steps = 0;
    while (t < world_t_exit && steps < vol.max_steps && (transmittance.x + transmittance.y + transmittance.z) > 0.015f) {
        float3 world_pos = ray_origin + ray_dir * t;
        float3 local_pos = transform_point_affine(world_pos, vol.inv_transform);
        
        // Pivot offset correction
        local_pos.x -= vol.pivot_offset[0];
        local_pos.y -= vol.pivot_offset[1];
        local_pos.z -= vol.pivot_offset[2];
        
        float raw_density = 0.0f;
        if (use_procedural_cloud) {
            raw_density = sample_procedural_cloud_density(vol, local_pos);
        } else {
            using AccT = nanovdb::FloatGrid::AccessorType;
            AccT acc = grid->getAccessor();
            nanovdb::math::SampleFromVoxels<AccT, 1, false> sampler(acc);
            nanovdb::Vec3f idx = grid->worldToIndexF(nanovdb::Vec3f(local_pos.x, local_pos.y, local_pos.z));
            raw_density = sampler(idx);
        }
        if (!isfinite(raw_density)) raw_density = 0.0f;
        
        float density = fmaxf((raw_density - vol.density_remap_low) / remap_range, 0.0f) * vol.density_multiplier;
        float sigma_a = density * vol.absorption_coefficient;
        float sigma_s = density * vol.scatter_coefficient;
        float sigma_t = sigma_a + sigma_s;
        float sparse_cutoff = (vol.density_pad > 0.0f) ? vol.density_pad : 0.04f;
        float scatter_keep = fminf(1.0f, fmaxf(0.0f, (sigma_s * base_step) / sparse_cutoff));
        if (curand_uniform(rng) <= scatter_keep) {
            if (sigma_t <= 1e-8f) { t += base_step; steps++; continue; }
            float step = fminf(base_step, tau_max / sigma_t);
            step = fmaxf(step, min_step);
            step = fminf(step, world_t_exit - t);
            if (step <= 1e-8f) break;
            float albedo_avg = vol.scatter_color.x * 0.2126f + vol.scatter_color.y * 0.7152f + vol.scatter_color.z * 0.0722f;
            
            // --- BLENDED MULTI-SCATTER TRANSMITTANCE (Matches CPU Scalar Model) ---
            float T_single = expf(-sigma_t * step);
            float step_transmittance = T_single;
            
            // Only use multi-scatter softening if there is actually scattering happening
            if (vol.scatter_coefficient > 1e-6f && vol.scatter_multi > 1e-6f) {
                float T_multi_p = expf(-sigma_t * step * 0.25f);
                step_transmittance = T_single * (1.0f - vol.scatter_multi * albedo_avg) + T_multi_p * (vol.scatter_multi * albedo_avg);
            }
            float one_minus_T = 1.0f - step_transmittance;
            
            // Emission
            float3 emission = make_float3(0.0f);
            if (vol.emission_mode == 1) emission = vol.emission_color * vol.emission_intensity * density;
            else if (vol.emission_mode >= 2) {
                float temperature = density;
                if (temp_grid) {
                    nanovdb::FloatGrid::AccessorType temp_acc = temp_grid->getAccessor();
                    nanovdb::math::SampleFromVoxels<nanovdb::FloatGrid::AccessorType, 1, false> temp_sampler(temp_acc);
                    nanovdb::Vec3f temp_idx = temp_grid->worldToIndexF(nanovdb::Vec3f(local_pos.x, local_pos.y, local_pos.z));
                    temperature = temp_sampler(temp_idx);
                }
                if (!isfinite(temperature)) temperature = density;
                
                float kelvin = (temperature > 20.0f) ? (temperature * vol.temperature_scale) : ((temperature * 3000.0f + 1000.0f) * vol.temperature_scale);
                float t_ramp;
                if (temperature > 20.0f) {
                    float ramp_min = (vol.emission_pad > 20.0f) ? vol.emission_pad : 0.0f;
                    float ramp_max = (vol.max_temperature > ramp_min + 1.0f) ? vol.max_temperature : 6000.0f;
                    t_ramp = (temperature - ramp_min) / fmaxf(ramp_max - ramp_min, 1.0f);
                } else {
                    t_ramp = temperature;
                }
                
                float ramp_t_clamped = fminf(fmaxf(t_ramp, 0.0f), 1.0f);
                float3 e_color = vol.color_ramp_enabled ? sample_color_ramp(vol, ramp_t_clamped) : blackbody_to_rgb(kelvin);
                float emit_gate = (vol.color_ramp_enabled && temperature > 20.0f && t_ramp <= 0.0f) ? 0.0f : 1.0f;
                emission = clamp_volume_radiance(e_color * (density * vol.blackbody_intensity * emit_gate), 64.0f);
            }
            
            // Lighting (Sun + Scene)
            float3 total_light = make_float3(0.0f);
            
            // 1. Sun Lighting
            // 1. Sun Lighting with Atmosphere Transmittance Parity
            {
                float3 sun_trans = gpu_get_transmittance(optixLaunchParams.world, world_pos, optixLaunchParams.world.nishita.sun_direction);
                float3 sun_color = sun_trans * sun_intensity;
                
                float cos_theta = dot(local_ray_dir, local_sun_dir);
                float phase = gpu_phase_dual_hg(cos_theta, vol.scatter_anisotropy, vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
                float powder = gpu_powder_effect(density, cos_theta);
                phase *= (1.0f + powder * 0.5f);

                float shadow = 1.0f;
                if (vol.shadow_steps > 0) {
                    int s_steps = max(1, vol.shadow_steps);
                    float tau_hint = fmaxf(density, 0.0f) * (vol.absorption_coefficient + vol.scatter_coefficient) * world_vol_extent;
                    if (s_steps > 8) {
                        float step_scale = fminf(1.0f, fmaxf(0.25f, sqrtf(fmaxf(tau_hint, 0.0f))));
                        s_steps = (int)ceilf((float)s_steps * step_scale);
                        s_steps = max(3, min(s_steps, min(vol.shadow_steps, 16)));
                    }
                    float s_step_world = world_vol_extent / (float)max(1, s_steps);
                    float s_jitter = curand_uniform(rng);

                    float s_trans = 0.0f;
                    for (int ls = 0; ls < s_steps && s_trans < 10.0f; ++ls) {
                        float sw = ((float)ls + s_jitter + 0.5f) * s_step_world;
                        float3 curr_local_s_pos = local_pos + local_sun_dir * (sw * dir_length);
                        if (curr_local_s_pos.x < vol.local_bbox_min.x || curr_local_s_pos.x > vol.local_bbox_max.x || 
                            curr_local_s_pos.y < vol.local_bbox_min.y || curr_local_s_pos.y > vol.local_bbox_max.y || 
                            curr_local_s_pos.z < vol.local_bbox_min.z || curr_local_s_pos.z > vol.local_bbox_max.z) break;
                        
                        float sd = 0.0f;
                        if (use_procedural_cloud) {
                            sd = sample_procedural_cloud_density(vol, curr_local_s_pos);
                        } else {
                            nanovdb::FloatGrid::AccessorType s_acc = grid->getAccessor();
                            nanovdb::math::SampleFromVoxels<nanovdb::FloatGrid::AccessorType, 1, false> s_sampler(s_acc);
                            nanovdb::Vec3f s_idx = grid->worldToIndexF(nanovdb::Vec3f(curr_local_s_pos.x, curr_local_s_pos.y, curr_local_s_pos.z));
                            sd = s_sampler(s_idx);
                        }
                        if (isfinite(sd)) {
                            float s_rem = fmaxf((sd - vol.density_remap_low) / remap_range, 0.0f);
                            s_trans += s_rem * vol.density_multiplier * (vol.absorption_coefficient + vol.scatter_coefficient) * s_step_world;
                        }
                    }
                    
                    // --- MULTI-OCTAVE SHADOWING (Matches CPU 1.0 - strength * (1-T) ) ---
                    float beers = expf(-s_trans);
                    float beers_soft = expf(-s_trans * 0.25f);
                    float albedo_lum = vol.scatter_color.x * 0.2126f + vol.scatter_color.y * 0.7152f + vol.scatter_color.z * 0.0722f;
                    float phys_trans = beers * (1.0f - vol.scatter_multi * albedo_lum) + beers_soft * (vol.scatter_multi * albedo_lum);
                    float shadow_strength = fminf(fmaxf(vol.shadow_strength * 1.08f, 0.0f), 1.0f);
                    shadow = 1.0f - shadow_strength * (1.0f - phys_trans);
                }
                total_light += sun_color * shadow * phase;
            }
            
            // 2. Extra Lights (Stable)
            if (optixLaunchParams.lights && optixLaunchParams.light_count > 0) {
                for (int j = 0; j < optixLaunchParams.light_count; ++j) {
                    const LightGPU& light = optixLaunchParams.lights[j];
                    float3 L_world = (light.type == 1) ? -light.direction : normalize(light.position - world_pos);
                    float3 local_L_dir = normalize(transform_vector_affine(L_world, vol.inv_transform));
                    float cos_th = dot(local_ray_dir, local_L_dir);
                    float ph = gpu_phase_dual_hg(cos_th, vol.scatter_anisotropy, vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
                    total_light += light.color * light.intensity * ph; 
                }
            }
            
            // 3. Sky/Ambient Lighting (CPU Parity). Gated by
            // volume_atmosphere_ambient (default OFF) — the raw Nishita sky
            // over-lights the volume vs Vulkan's LUT ambient.
            float thin_scatter = scatter_keep * scatter_keep;
            if (optixLaunchParams.world.volume_atmosphere_ambient != 0) {
                total_light += gpu_get_ambient_radiance_volume(optixLaunchParams.world, ambient_dir) * (0.15f * thin_scatter) * optixLaunchParams.world.nishita.atmosphere_intensity;
            }

            float3 albedo = vol.scatter_color;
            float3 ms_boost = make_float3(1.0f) + albedo * vol.scatter_multi * (2.0f * thin_scatter);
            float3 inscatter = (albedo * total_light * sigma_s * ms_boost);
            
            // CPU/Vulkan parity integration
            accumulated_color += transmittance * (inscatter + emission) * one_minus_T;
            transmittance *= step_transmittance;
            t += step;
            steps++;
            continue;
        }
        t += base_step;
        steps++;
    }
    
    out_transmittance = (transmittance.x + transmittance.y + transmittance.z) / 3.0f;
    return accumulated_color;
}

// ═══════════════════════════════════════════════════════════════════════════
// ATMOSPHERIC FOG - Height-based exponential fog with sun scattering
// ═══════════════════════════════════════════════════════════════════════════
__device__ float calculate_height_fog_factor(
    float3 rayOrigin, 
    float3 rayDir, 
    float distance,
    float fogDensity, 
    float fogHeight, 
    float fogFalloff
) {
    // Exponential height fog - denser at ground level
    // Based on: https://iquilezles.org/articles/fog/
    float a = fogDensity * expf(-fogFalloff * rayOrigin.y);
    float b = fogFalloff * rayDir.y;
    
    // Avoid division by zero for horizontal rays
    if (fabsf(b) < 1e-5f) {
        return 1.0f - expf(-a * distance);
    }
    
    // Analytical integral of exponential height fog
    float fogAmount = a * (1.0f - expf(-b * distance)) / b;
    return 1.0f - expf(-fogAmount);
}

__device__ float3 apply_atmospheric_fog(
    const WorldData& world,
    float3 sceneColor,
    float3 rayDir,
    float distance
) {
    if (!world.nishita.fog_enabled || world.nishita.fog_density <= 0.0f) {
        return sceneColor;
    }
    
    // Camera position (at altitude)
    float3 rayOrigin = make_float3(0.0f, world.nishita.altitude, 0.0f);
    
    // Clamp distance to fog range
    distance = fminf(distance, world.nishita.fog_distance);
    
    // Calculate fog factor with height falloff
    float fogFactor = calculate_height_fog_factor(
        rayOrigin, rayDir, distance,
        world.nishita.fog_density,
        world.nishita.fog_height,
        world.nishita.fog_falloff
    );
    
    // Base fog color
    float3 fogColor = world.nishita.fog_color;
    
    // Sun scattering in fog (makes fog glow towards sun)
    float3 sunDir = normalize(world.nishita.sun_direction);
    float sunDot = fmaxf(0.0f, dot(rayDir, sunDir));
    float sunScatter = powf(sunDot, 8.0f) * world.nishita.fog_sun_scatter;
    
    // Add sun color to fog when looking towards sun
    float3 sunColor = make_float3(1.0f, 0.9f, 0.7f) * world.nishita.sun_intensity * 0.1f;
    fogColor = fogColor + sunColor * sunScatter;
    
    // Blend scene with fog
    return lerp(sceneColor, fogColor, fogFactor);
}

__device__ float3 rc_weather_tint_color(int type) {
    if (type == WEATHER_RAIN) return make_float3(0.50f, 0.56f, 0.62f);
    if (type == WEATHER_SNOW) return make_float3(0.86f, 0.90f, 0.96f);
    if (type == WEATHER_DUST) return make_float3(0.74f, 0.58f, 0.38f);
    if (type == WEATHER_MIST) return make_float3(0.70f, 0.76f, 0.82f);
    return make_float3(0.0f);
}

__device__ bool rc_weather_active(const WeatherParams& weather) {
    return weather.enabled != 0 && weather.type != WEATHER_NONE &&
           weather.intensity > 0.0f && weather.density > 0.0f;
}

__device__ bool rc_weather_visual_active(const WeatherParams& weather) {
    return rc_weather_active(weather) && weather.visual_mode != WEATHER_VISUAL_SURFACE_ONLY;
}

__device__ float3 rc_apply_weather_sky(const WeatherParams& weather, float3 sky, float3 rayDir) {
    if (!rc_weather_visual_active(weather)) return sky;

    float visibilityLoss = fmaxf(0.0f, fminf(1.0f, 1.0f - weather.visibility));
    float horizon = powf(fmaxf(0.0f, 1.0f - fabsf(rayDir.y)), 0.65f);
    float amount = weather.intensity * (0.25f + weather.density * 0.75f + visibilityLoss * 0.65f);
    amount = fmaxf(0.0f, fminf(0.85f, amount * (0.35f + horizon * 0.65f)));

    float3 tint = rc_weather_tint_color(weather.type);
    float3 dimmed = sky;
    if (weather.type == WEATHER_RAIN) {
        dimmed *= 0.72f;
    } else if (weather.type == WEATHER_DUST) {
        dimmed *= 0.82f;
    } else if (weather.type == WEATHER_SNOW || weather.type == WEATHER_MIST) {
        dimmed = dimmed * 0.90f + tint * 0.10f;
    }
    return dimmed * (1.0f - amount) + tint * amount;
}

__device__ float3 rc_apply_weather_atmosphere(
    const WeatherParams& weather,
    float3 color,
    float3 rayDir,
    float distance
) {
    if (!rc_weather_visual_active(weather)) return color;

    float vis = fmaxf(0.02f, fminf(1.0f, weather.visibility));
    float sigma = weather.intensity * weather.density * (0.00016f + (1.0f - vis) * 0.00034f);
    float amount = 1.0f - expf(-fmaxf(distance, 0.0f) * sigma);
    amount = fmaxf(0.0f, fminf(0.78f, amount));

    float3 tint = rc_weather_tint_color(weather.type);
    if (weather.type == WEATHER_RAIN) {
        tint *= 0.72f;
    } else if (weather.type == WEATHER_DUST) {
        tint *= 1.12f;
    }

    float windLen2 = dot(weather.wind_direction, weather.wind_direction);
    float3 windDir = (windLen2 > 1e-8f) ? normalize(weather.wind_direction) : make_float3(1.0f, 0.0f, 0.0f);
    float forward = powf(fmaxf(0.0f, dot(normalize(rayDir), windDir)), 4.0f);
    amount = fminf(0.90f, amount + forward * weather.intensity * weather.density * 0.08f);
    return color * (1.0f - amount) + tint * amount;
}

__device__ float rc_precip_hash(float x, float y) {
    float v = sinf(x * 127.1f + y * 311.7f) * 43758.5453123f;
    return v - floorf(v);
}

__device__ float rc_precip_noise(float x, float y) {
    float cell_x = floorf(x);
    float cell_y = floorf(y);
    float frac_x = x - cell_x;
    float frac_y = y - cell_y;
    float smooth_x = frac_x * frac_x * (3.0f - 2.0f * frac_x);
    float smooth_y = frac_y * frac_y * (3.0f - 2.0f * frac_y);
    float n00 = rc_precip_hash(cell_x, cell_y);
    float n10 = rc_precip_hash(cell_x + 1.0f, cell_y);
    float n01 = rc_precip_hash(cell_x, cell_y + 1.0f);
    float n11 = rc_precip_hash(cell_x + 1.0f, cell_y + 1.0f);
    float nx0 = lerp(n00, n10, smooth_x);
    float nx1 = lerp(n01, n11, smooth_x);
    return lerp(nx0, nx1, smooth_y);
}

__device__ float rc_precip_smoothstep(float edge0, float edge1, float x) {
    float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / fmaxf(edge1 - edge0, 1e-6f)));
    if (edge1 < edge0) {
        t = fmaxf(0.0f, fminf(1.0f, (edge0 - x) / fmaxf(edge0 - edge1, 1e-6f)));
    }
    return t * t * (3.0f - 2.0f * t);
}

__device__ float rc_precip_line(float u, float v, float windX, float windY, float time, float density, float scale) {
    float windDrive = 1.0f + density * 0.85f;
    float px = u * 74.0f / scale + windX * time * (6.5f + density * 4.0f);
    float py = v * 22.0f / scale - time * (34.0f + density * 10.0f) + windY * time * (7.5f + density * 4.5f);
    float cx = floorf(px);
    float cy = floorf(py);
    float fx = px - cx;
    float fy = py - cy;
    float rnd = rc_precip_hash(cx, cy);
    float spawn = rc_precip_smoothstep(0.94f - density * 0.10f, 1.0f, rnd);
    float x = fabsf(fx - 0.5f - (rnd - 0.5f) * (0.35f + 0.18f * windDrive));
    float core = rc_precip_smoothstep(0.055f - density * 0.016f, 0.0f, x);
    float trail = rc_precip_smoothstep(1.0f, 0.04f, fy);
    return spawn * core * trail;
}

__device__ float rc_precip_flake(float u, float v, float windX, float windY, float time, float density, float scale) {
    float px = u * 46.0f / scale + windX * time * (2.6f + density * 2.5f);
    float py = v * 30.0f / scale - time * (2.8f + density * 0.9f) + windY * time * (1.1f + density * 0.9f);
    float cx = floorf(px);
    float cy = floorf(py);
    float fx = px - cx;
    float fy = py - cy;
    float rnd = rc_precip_hash(cx, cy);
    float spawn = rc_precip_smoothstep(0.84f - density * 0.22f, 1.0f, rnd);
    float ox = rc_precip_hash(cx + 13.7f, cy + 13.7f) + sinf(time * 1.7f + rnd * 6.2831f) * (0.12f + density * 0.08f) + windX * 0.12f;
    float oy = rc_precip_hash(cx + 41.3f, cy + 41.3f) + windY * 0.05f;
    float radius = lerp(0.035f, 0.120f + density * 0.025f, rc_precip_hash(cx + 7.1f, cy + 7.1f));
    float dx = fx - ox;
    float dy = fy - oy;
    return spawn * rc_precip_smoothstep(radius, 0.0f, sqrtf(dx * dx + dy * dy));
}

__device__ float rc_precip_dust(float u, float v, float windX, float windY, float time, float density, float scale) {
    float advected_u = u / scale + windX * time * (0.24f + density * 0.08f);
    float advected_v = v / scale + windY * time * (0.18f + density * 0.10f);
    float streaks = rc_precip_noise(advected_u * 18.0f, advected_v * 7.5f);
    float wisps = rc_precip_noise(advected_u * 33.0f + 19.4f, advected_v * 12.0f + 7.1f);
    float grain = rc_precip_noise(advected_u * 95.0f + 3.7f, advected_v * 42.0f + 17.3f);
    float elongated = rc_precip_smoothstep(0.52f, 0.98f, streaks * 0.72f + wisps * 0.28f);
    float soft_grain = rc_precip_smoothstep(0.38f, 0.88f, grain);
    return (elongated * (0.72f + density * 0.28f) + soft_grain * 0.18f) * density;
}

__device__ float3 rc_apply_weather_precipitation_overlay(
    const WeatherParams& weather,
    float3 color,
    float3 rayDir,
    float distance,
    float time
) {
    if (!rc_weather_visual_active(weather)) return color;

    float density = fmaxf(0.0f, fminf(1.0f,
        powf(fmaxf(0.0f, fminf(1.0f, weather.intensity)), 0.82f) *
        (0.28f + fmaxf(0.0f, fminf(1.0f, weather.density)) * 1.22f)));
    if (density <= 0.001f) return color;

    rayDir = normalize(rayDir);
    float u = atan2f(rayDir.z, rayDir.x) * (1.0f / (2.0f * M_PIf)) + 0.5f;
    float v = acosf(fmaxf(-1.0f, fminf(1.0f, rayDir.y))) * (1.0f / M_PIf);
    float scale = fmaxf(weather.precipitation_scale, 0.25f);
    float depthFade = rc_precip_smoothstep(0.6f, 18.0f, fmaxf(distance, 0.0f));
    float horizonFade = fmaxf(0.35f, fminf(1.0f, 1.0f - fmaxf(rayDir.y, 0.0f) * 0.35f));
    float windLen2 = weather.wind_direction.x * weather.wind_direction.x + weather.wind_direction.z * weather.wind_direction.z;
    float invWindLen = windLen2 > 1e-8f ? rsqrtf(windLen2) : 0.0f;
    float windAmount = fmaxf(0.0f, fminf(1.0f, weather.wind_speed / 35.0f));
    float windX = (windLen2 > 1e-8f ? weather.wind_direction.x * invWindLen : 1.0f) * windAmount;
    float windY = (windLen2 > 1e-8f ? weather.wind_direction.z * invWindLen : 0.0f) * windAmount;
    float windVisual = 0.65f + windAmount * 0.9f;
    float amount = 0.0f;
    float3 tint = rc_weather_tint_color(weather.type);

    if (weather.type == WEATHER_RAIN) {
        amount = fminf(1.0f, rc_precip_line(u, v, windX, windY, time, density, scale) * (0.85f + density * 0.85f + windAmount * 0.35f));
        color = lerp(color, color * 0.84f, amount * density * 0.24f * depthFade);
        color += make_float3(0.45f, 0.55f, 0.66f) * amount * density * 0.28f * depthFade * horizonFade * windVisual;
    } else if (weather.type == WEATHER_SNOW) {
        amount = fminf(1.0f, rc_precip_flake(u, v, windX, windY, time, density, scale) * (0.80f + density * 0.95f + windAmount * 0.22f));
        color = lerp(color, tint, amount * density * 0.34f * depthFade * horizonFade);
        color += make_float3(0.85f, 0.92f, 1.0f) * amount * density * 0.16f * depthFade * windVisual;
    } else if (weather.type == WEATHER_DUST) {
        amount = fminf(1.0f, rc_precip_dust(u, v, windX, windY, time, density, scale) * (0.75f + density * 1.05f + windAmount * 0.45f));
        color = lerp(color, tint, amount * 0.16f * depthFade * windVisual);
        color += tint * amount * 0.075f * depthFade * windVisual;
    } else if (weather.type == WEATHER_MIST) {
        amount = fminf(1.0f, rc_precip_dust(u, v, windX * 0.25f, windY * 0.25f, time * 0.35f, density, scale * 1.4f) * (0.65f + density * 0.85f));
        color = lerp(color, tint, amount * 0.11f * depthFade);
    }

    return color;
}

// ═══════════════════════════════════════════════════════════════════════════
// VOLUMETRIC GOD RAYS - Ray-marched light shafts with proper occlusion
// Objects will block god rays creating shadows in the volumetric effect
// ═══════════════════════════════════════════════════════════════════════════

// Forward declaration for shadow test
__device__ bool trace_shadow_test(float3 origin, float3 direction, float maxDist);

// ═══════════════════════════════════════════════════════════════════════════
// NEW: Combined Transmittance for God Rays (Solid + Volumetric)
// ═══════════════════════════════════════════════════════════════════════════
__device__ float calculate_sun_transmittance(
    float3 origin, 
    float3 direction, 
    float maxDist, 
    curandState* rng
) {
    // JITTERED SUN SAMPLING: Check visibility against the actual sun disk area. 
    // This prevents "leaking" at mountain edges by smoothing out precision errors 
    // and correctly handling partial occlusion.
    float3 jitteredSunDir = direction;
    if (rng) {
        float sunRadius = optixLaunchParams.world.nishita.sun_size * (M_PIf / 180.0f) * 0.25f;
        jitteredSunDir = normalize(direction + random_in_unit_sphere(rng) * sunRadius);
    }

    // 1. Check Solid Occlusion (Binary) - Increased distance to world-scale infinity
    if (trace_shadow_test(origin, jitteredSunDir, maxDist)) {
        return 0.0f; // Fully occluded by solid geometry
    }
    
    float transmittance = 1.0f;
    
    // 2. Check VDB Volume Occlusion
    if (optixLaunchParams.vdb_volumes && optixLaunchParams.vdb_volume_count > 0) {
        transmittance *= calculate_vdb_occlusion(origin, jitteredSunDir, maxDist, rng);
    }
    
    // 3. Check Gas Volume Occlusion (Texture3D)
    if (optixLaunchParams.gas_volumes && optixLaunchParams.gas_volume_count > 0) {
        for (int i = 0; i < optixLaunchParams.gas_volume_count; ++i) {
            const GpuGasVolume& vol = optixLaunchParams.gas_volumes[i];
            if (!vol.density_texture || vol.shadow_strength <= 0.0f) continue;
            
            float3 l_origin = transform_point_affine(origin, vol.inv_transform);
            float3 l_dir = transform_vector_affine(jitteredSunDir, vol.inv_transform);
            float dir_len = length(l_dir);
            if (dir_len < 1e-6f) continue;
            l_dir /= dir_len;
            
            float t_enter, t_exit;
            if (intersect_aabb_vdb(l_origin, l_dir, vol.local_bbox_min, vol.local_bbox_max, t_enter, t_exit)) {
                t_enter = fmaxf(t_enter, 0.0f);
                float dist = t_exit - t_enter;
                if (dist > 0.0f) {
                    float shadow_density_sum = 0.0f;
                    int s_steps = min(vol.shadow_steps, 4);
                    float s_step_size = dist / (float)s_steps;
                    float3 remap = make_float3(1.0f) / (vol.local_bbox_max - vol.local_bbox_min);
                    
                    for (int s = 0; s < s_steps; ++s) {
                        float st = t_enter + s_step_size * (curand_uniform(rng));
                        float3 lp = l_origin + l_dir * st;
                        float3 uv = (lp - vol.local_bbox_min) * remap;
                        float d = tex3D<float>(vol.density_texture, uv.x, uv.y, uv.z);
                        d = fmaxf(0.0f, d - vol.density_remap_low) / (vol.density_remap_high - vol.density_remap_low + 1e-6f);
                        shadow_density_sum += d * vol.density_multiplier * s_step_size;
                    }
                    transmittance *= expf(-shadow_density_sum * vol.shadow_strength);
                }
            }
            if (transmittance < 0.01f) return 0.0f;
        }
    }

    return transmittance;
}

// Shadow test for god rays - MUST be before calculate_volumetric_god_rays
__device__ bool trace_shadow_test(float3 origin, float3 direction, float maxDist) {
    Ray shadow_ray(origin, direction);
    return trace_shadow_ray(shadow_ray, SCENE_EPSILON, maxDist) != 0u;
}

__device__ float3 calculate_volumetric_god_rays(
    const WorldData& world,
    float3 rayOrigin,
    float3 rayDir,
    float maxDistance,
    curandState* rng
) {
    // 1. EARLY EXIT & PARAMETERS
    if (!GodRaysModel::isEnabled(
            world.nishita.godrays_enabled,
            world.nishita.godrays_intensity,
            world.nishita.godrays_density)) {
        return make_float3(0.0f);
    }
    
    float3 sunDir = normalize(world.nishita.sun_direction);
    float sunDot = dot(rayDir, sunDir);

    // Mie scattering is extremely forward-heavy. We fade based on anisotropy.
    float g = world.nishita.mie_anisotropy;
    float anisotropyFade = GodRaysModel::anisotropyFade(sunDot, g);
    if (anisotropyFade < 1e-6f) return make_float3(0.0f);
    
    if (GodRaysModel::isSunBelowHorizon(sunDir.y)) return make_float3(0.0f);

    // 2. STOCHASTIC ADAPTIVE STEPPING
    float marchDistance = GodRaysModel::computeMarchDistance(maxDistance, world.nishita.fog_distance);
    
    int numSteps = GodRaysModel::computeStepCount(sunDot, world.nishita.godrays_samples);
    
    float stepSize = GodRaysModel::computeStepSize(marchDistance, numSteps);
    float3 godRayColor = make_float3(0.0f);
    float transmittance = 1.0f;

    // 3. PHASE & DENSITY
    float phase = GodRaysModel::computeMiePhase(sunDot, g);
    float solarCoreFade = GodRaysModel::computeSolarCoreFade(sunDot, world.nishita.sun_size);
    
    float mediaDensity = GodRaysModel::computeMediaDensity(world.nishita.godrays_density);
    float3 sunRadianceBase = make_float3(1.0f, 0.98f, 0.95f) * world.nishita.sun_intensity * (GodRaysModel::kSunRadianceScale * 1.35f);

    // 4. RAYMARCHING LOOP
    float jitter = curand_uniform(rng);
    float t = jitter * stepSize;
    
    for (int i = 0; i < numSteps; ++i) {
        if (t > marchDistance) break;
        
        float nearFade = GodRaysModel::computeNearFade(t);
        if (nearFade > 0.001f) {
            float3 samplePos = rayOrigin + rayDir * t;
            
            float heightFactor = GodRaysModel::computeHeightFactor(samplePos.y, world.nishita.altitude);
            
            float sigma_s = mediaDensity * heightFactor;
            float sigma_t = sigma_s;
            float stepTrans = GodRaysModel::computeStepTransmittance(sigma_t, stepSize);
            
            float3 sunTrans = gpu_get_transmittance(world, samplePos, sunDir);
            float3 currentSunRadiance = sunRadianceBase * sunTrans;

            // Keep the shaft direction stable here. calculate_sun_transmittance already
            // performs soft/jittered visibility against the solar disk, and double-jittering
            // makes OptiX shafts noticeably weaker than Vulkan.
            float occlusion = calculate_sun_transmittance(samplePos, sunDir, 100000.0f, rng);
            
            if (occlusion > 0.001f && sigma_t > 1e-6f) {
                float3 inscatter = currentSunRadiance * phase * occlusion * (sigma_s / sigma_t) * nearFade * solarCoreFade;
                godRayColor += transmittance * inscatter * (1.0f - stepTrans);
            }
            
            transmittance *= stepTrans;
        }
        
        if (transmittance < GodRaysModel::kTransmittanceCutoff) break;
        t += stepSize;
    }
    
    return godRayColor * world.nishita.godrays_intensity;
}

// trace_shadow_test moved above calculate_volumetric_god_rays

// ═══════════════════════════════════════════════════════════════════════════
// MULTI-SCATTERING - Improved atmosphere realism (Frostbite-style)
// ═══════════════════════════════════════════════════════════════════════════
__device__ float3 apply_multi_scattering(
    float3 singleScatter,
    float opticalDepth,
    float3 scatteringAlbedo,
    float multiFactor
) {
    // Multi-scattering approximation:
    // Each bounce adds dimmer contribution of scattered light
    // This brightens the horizon and makes sky more uniform
    
    float3 ms = singleScatter;
    
    // Second order scattering
    float3 secondOrder = singleScatter * scatteringAlbedo * 0.5f * expf(-opticalDepth * 0.3f);
    ms = ms + secondOrder * multiFactor;
    
    // Third order (very subtle)
    float3 thirdOrder = secondOrder * scatteringAlbedo * 0.25f * expf(-opticalDepth * 0.1f);
    ms = ms + thirdOrder * multiFactor * 0.5f;
    
    return ms;
}

__device__ float3 gpu_get_sky_radiance(const WorldData& world, const float3& dir) {
    if (world.mode != 2 || world.lut.skyview_lut == 0) return make_float3(0,0,0);
    
    float azimuth = atan2f(dir.z, dir.x);
    if (azimuth < 0.0f) azimuth += 2.0f * M_PIf;
    
    float u = azimuth / (2.0f * M_PIf);
    float v = (1.0f - dir.y) * 0.5f; // Matches cosTheta mapping on CPU
    
    float4 tex = tex2D<float4>(world.lut.skyview_lut, u, v);
    return make_float3(tex.x, tex.y, tex.z);
}

// Unified helper for ambient lighting in volumes (excl clouds)
__device__ float3 gpu_get_ambient_radiance_volume(const WorldData& world, const float3& dir) {
    if (world.mode == 0) return world.color * world.color_intensity;
    if (world.mode == 1 && world.env_texture) {
        float theta = acosf(dir.y);
        float phi = atan2f(-dir.z, dir.x) + M_PIf;
        float u = phi * (0.5f * M_1_PIf);
        float v = theta * M_1_PIf;
        u -= world.env_rotation / (2.0f * M_PIf);
        u -= floorf(u);
        float4 tex = tex2D<float4>(world.env_texture, u, v);
        return make_float3(tex.x, tex.y, tex.z) * world.env_intensity;
    }
    if (world.mode == 2) return gpu_get_sky_radiance(world, dir);
    return make_float3(0,0,0);
}

__device__ float3 gpu_get_volume_ambient_dir(const WorldData& world, const float3& view_dir) {
    const float3 up = make_float3(0.0f, 1.0f, 0.0f);
    float3 d = normalize(view_dir * 0.45f + up * 0.55f + world.nishita.sun_direction * 0.15f);
    if (!isfinite(d.x) || !isfinite(d.y) || !isfinite(d.z)) return up;
    return d;
}

__device__ float3 gpu_get_aerial_perspective(const WorldData& world, float3 color, float3 origin, float3 dir, float dist) {
    if (world.mode != 2 || world.lut.transmittance_lut == 0 || dist > 1e6f) return color;
    
    if (!world.advanced.aerial_perspective) return color;

    float Rg = world.nishita.planet_radius;
    float3 p = origin + make_float3(0, Rg, 0);
    float altitude = length(p) - Rg;
    float3 up = p / (Rg + altitude);
    
    float cosTheta = fmaxf(0.01f, dot(up, dir)); 
    float u = (cosTheta + 0.2f) / 1.2f;
    float v = fminf(1.0f, fmaxf(0.0f, altitude / world.nishita.atmosphere_height));
    
    float4 trans4 = tex2D<float4>(world.lut.transmittance_lut, u, v);
    float3 transmittance = make_float3(trans4.x, trans4.y, trans4.z);
    
    const float min_dist = world.advanced.aerial_min_distance;
    const float max_dist = world.advanced.aerial_max_distance;
    float ramp = (dist < min_dist) ? 0.0f : fminf(1.0f, (dist - min_dist) / fmaxf(1.0f, max_dist - min_dist));
    
    float aerialDensity = fmaxf(0.0f, world.advanced.aerial_density);
    float atmosphereDensity = fmaxf(0.001f, world.nishita.air_density * 0.60f + world.nishita.dust_density * 0.40f);
    float densityFactor = aerialDensity * atmosphereDensity * (1.0f + world.nishita.fog_density * 120.0f);
    float distFactor = (1.0f - expf(-(dist / 10000.0f) * densityFactor)) * (ramp * ramp);
    
    float3 finalTrans = make_float3(powf(transmittance.x, distFactor), powf(transmittance.y, distFactor), powf(transmittance.z, distFactor));
    
    float3 lookupDir = dir;
    if (lookupDir.y < 0.0f) {
        lookupDir.y = 0.0f;
        lookupDir = normalize(lookupDir);
    }
    float3 skyRadiance = gpu_get_sky_radiance(world, lookupDir);
    
    // Blend with scene
    float3 res = color * finalTrans + skyRadiance * (make_float3(1,1,1) - finalTrans);

    // DYNAMIC HEIGHT FOG OVERLAY: If Nishita is active and fog is enabled, 
    // overlay the height-based fog for extra artistic control.
    if (world.nishita.fog_enabled && world.nishita.fog_density > 0.01f) {
        float fAmount = calculate_height_fog_factor(
            origin, dir, dist,
            world.nishita.fog_density * 0.5f,
            world.nishita.fog_height,
            world.nishita.fog_falloff
        );
        res = lerp(res, world.nishita.fog_color, fAmount);
    }

    return res;
}

__device__ float3 rc_blend_environment_overlay(const float3& base, const float3& sampled, float intensity, int blendMode) {
    float strength = fmaxf(0.0f, intensity);
    float amount = fminf(strength, 1.0f);
    float3 overlay = sampled * strength;

    if (blendMode == 1) {
        return base * (make_float3(1.0f) * (1.0f - amount) + sampled * amount);
    }
    if (blendMode == 2) {
        return base + overlay;
    }
    if (blendMode == 3) {
        return overlay;
    }
    return base * (1.0f - amount) + overlay * amount;
}

__device__ float3 rc_apply_nishita_environment_overlay(const WorldData& world, const float3& base, const float3& dir) {
    if (!world.advanced.env_overlay_enabled || !world.advanced.env_overlay_tex) {
        return base;
    }

    float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
    float phi = atan2f(-dir.z, dir.x) + M_PIf;
    float u = phi * (0.5f * M_1_PIf);
    float v = theta * M_1_PIf;
    u -= world.advanced.env_overlay_rotation / 360.0f;
    u -= floorf(u);

    float4 tex = tex2D<float4>(world.advanced.env_overlay_tex, u, v);
    float3 sampled = make_float3(tex.x, tex.y, tex.z);
    return rc_blend_environment_overlay(
        base,
        sampled,
        world.advanced.env_overlay_intensity,
        world.advanced.env_overlay_blend_mode);
}

__device__ float3 evaluate_background(const WorldData& world, const float3& origin, const float3& dir, curandState* rng, bool skip_cloud_overlay = false) {
    if (world.mode == 0) { // WORLD_MODE_COLOR
        return rc_apply_weather_sky(world.weather, world.color, dir);
    }
    else if (world.mode == 1) { // WORLD_MODE_HDRI
        if (world.env_texture) {
            float theta = acosf(dir.y);
            float phi = atan2f(-dir.z, dir.x) + M_PIf;
            
            float u = phi * (0.5f * M_1_PIf); // 0..1
            float v = theta * M_1_PIf;        // 0..1
            
            u -= world.env_rotation / (2.0f * M_PIf);
            u -= floorf(u);
            
            float4 tex = tex2D<float4>(world.env_texture, u, v);
            float3 hdri = make_float3(tex.x, tex.y, tex.z) * world.env_intensity;
            return rc_apply_weather_sky(world.weather, hdri, dir);
        }
        return rc_apply_weather_sky(world.weather, world.color, dir);
    }
    else if (world.mode == 2) { // WORLD_MODE_NISHITA
        // High-Quality LUT based Sky radiance
        float3 radiance = gpu_get_sky_radiance(world, dir);
        
        // --- PROCEDURAL SUN GLOW (Improved Transition) ---
        float3 sunDir = normalize(world.nishita.sun_direction);
        float mu = dot(dir, sunDir);
        
        // Removed hard mu > 0.99f check to allow smoother mie halo transitions
        float g_mie = world.nishita.mie_anisotropy;
        float phaseM = (1.0f - g_mie * g_mie) / (4.0f * 3.14159f * powf(1.0f + g_mie * g_mie - 2.0f * g_mie * mu, 1.5f));
        
        // Soft halo contribution (tight excess-phase corona around disk)
        float excessPhase = fmaxf(0.0f, phaseM - 2.0f);
        if (excessPhase > 0.0f) {
            // HALO LEAK PROTECTION: Add shadow test for halo to prevent bleeding through solid objects
            if (!trace_shadow_test(origin, sunDir, 100000.0f)) {
                float3 transToSun = gpu_get_transmittance(world, origin, sunDir);
                float3 haloTint = make_float3(0.35f) + transToSun * 0.65f;
                float3 mieScat = make_float3(world.nishita.mie_density, world.nishita.mie_density, world.nishita.mie_density); 
                mieScat = mieScat * world.nishita.mie_scattering * (0.15f * 2.5f);
                radiance += haloTint * mieScat * excessPhase * world.nishita.atmosphere_intensity;
            }
        }

        // ── Broad Mie background halo (sunset/horizon scatter glow) ──────────
        // Vulkan miss.rmiss adds: sunColor * sunIntensity * phaseM_cap * dust*0.015
        // This term was missing in OptiX causing the horizon to stay blue while
        // only the disk turned orange/red at sunset.
        // We use gpu_get_transmittance(sunDir) for physical reddening instead of a
        // fixed sunColor — spectrum is naturally orange/red when sun is near horizon.
        {
            float phaseM_cap = fminf(phaseM, 2.0f);
            float mie_scale  = fminf(world.nishita.dust_density * 0.0225f, 0.225f);
            if (mie_scale > 0.0f) {
                float3 transToSun = gpu_get_transmittance(world, origin, sunDir);
                float3 haloTint = make_float3(0.35f) + transToSun * 0.65f;
                radiance += haloTint * world.nishita.atmosphere_intensity * phaseM_cap * mie_scale;
            }
        }

        // Multi-scattering (Advanced Selection)
        if (world.advanced.multi_scatter_enabled) {
            float3 scatteringAlbedo = make_float3(0.8f, 0.85f, 0.9f);
            float3 ms = apply_multi_scattering(radiance, 0.5f, scatteringAlbedo, world.advanced.multi_scatter_factor); 
            radiance = ms;
        }

        // --- SUN DISK ---
        float sunSizeDeg = world.nishita.sun_size;
        
        // Apply horizon magnification effect
        if (world.nishita.sun_elevation < 15.0f) {
            sunSizeDeg *= 1.0f + (15.0f - fmaxf(world.nishita.sun_elevation, -10.0f)) * 0.04f;
        }
        
        float sun_radius = sunSizeDeg * (M_PIf / 180.0f) * 0.5f;
        if (dot(dir, sunDir) > cosf(sun_radius)) {
            // Procedural Limb Darkening & Edge Softening (Matches CPU Exactly)
            float mu_disk = dot(dir, sunDir);
            float angular_dist = acosf(fminf(1.0f, mu_disk));
            float radial_pos = angular_dist / fmaxf(1e-6f, sun_radius);
            
            float u_limb = 0.6f;
            float cosine_mu = sqrtf(fmaxf(0.0f, 1.0f - radial_pos * radial_pos));
            float limbDarkening = 1.0f - u_limb * (1.0f - cosine_mu);
            
            float edge_t = fmaxf(0.0f, fminf(1.0f, (radial_pos - 0.85f) / 0.15f));
            float edgeSoftness = 1.0f - edge_t * edge_t * (3.0f - 2.0f * edge_t);

            float t_occ = get_stochastic_volumetric_occlusion(origin, dir, 200000.0f, rng, 0.0f);
            if (t_occ >= 200000.0f) {
                float3 transSun = gpu_get_transmittance(world, origin, sunDir);
                // BOOSTER: Sun disk needs much more power to punch through high exposure atmosphere
                radiance += transSun * world.nishita.sun_intensity * 80000.0f * limbDarkening * edgeSoftness;
            }
        }

        // Clouds are a direct background layer.  For glass/refraction paths they
        // can read like dirt painted onto the material, while Vulkan's miss shader
        // returns only sky radiance here.
        radiance = rc_apply_nishita_environment_overlay(world, radiance, dir);
        return rc_apply_weather_sky(world.weather, radiance, dir);
    }

    return rc_apply_weather_sky(world.weather, make_float3(0,0,0), dir);
}

__device__ float power_heuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + 1e-4f);
}
__device__ float balance_heuristic(float pdf_a, float pdf_b) {
    return pdf_a / (pdf_a + pdf_b + 1e-4f);
}
__device__ float luminance(const float3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ float3 soft_clip_fireflies(const float3& c, float max_luma) {
    float l = luminance(c);
    if (l <= max_luma) return c;
    return c * (max_luma / fmaxf(l, 1e-6f));
}

__device__ float3 glass_background_response(const float3& bg) {
    float l = luminance(bg);
    float3 neutral = make_float3(l, l, l);
    float3 desaturated = neutral * 0.32f + bg * 0.68f;
    float max_luma = 10.0f;
    float dl = luminance(desaturated);
    if (dl > max_luma) {
        desaturated *= max_luma / fmaxf(dl, 1e-6f);
    }
    return desaturated;
}

__device__ float resolve_surface_opacity(const GpuMaterial& mat, const OptixHitResult& payload) {
    float opacity = fminf(fmaxf(mat.opacity, 0.0f), 1.0f);
    if (payload.has_opacity_tex && payload.opacity_tex) {
        float2 uv = apply_material_uv_transform(mat, payload.uv);
        float4 tex = tex2D<float4>(payload.opacity_tex, uv.x, uv.y);
        float mask = payload.opacity_has_alpha ? tex.w : tex.x;
        opacity *= fminf(fmaxf(mask, 0.0f), 1.0f);
    }
    return fminf(fmaxf(opacity, 0.0f), 1.0f);
}

__device__ float resolve_surface_transmission(const GpuMaterial& mat, const OptixHitResult& payload) {
    float transmission = fminf(fmaxf(mat.transmission, 0.0f), 1.0f);
    if (!payload.use_blended_data && payload.has_transmission_tex && payload.transmission_tex) {
        float2 uv = apply_material_uv_transform(mat, payload.uv);
        transmission = tex2D<float4>(payload.transmission_tex, uv.x, uv.y).x;
        transmission = fminf(fmaxf(transmission, 0.0f), 1.0f);
    }

    float opacity = resolve_surface_opacity(mat, payload);
    if (opacity < 0.99f && mat.metallic < 0.1f && transmission < 0.01f) {
        transmission = 1.0f - opacity;
    }
    return fminf(fmaxf(transmission, 0.0f), 1.0f);
}

__device__ int pick_smart_light(const float3& hit_position, curandState* rng, float* pdf_out = nullptr) {
    int light_count = optixLaunchParams.light_count;
    if (light_count == 0) {
        if (pdf_out) *pdf_out = 0.0f;
        return -1;
    }

    // --- Vulkan parity: Directional ışık olasılığını oransal hesapla ---
    // Sabit 0.33 yerine dir_count/light_count kullanarak unbiased estimator sağla
    int dir_count = 0;
    for (int i = 0; i < light_count; i++) {
        if (optixLaunchParams.lights[i].type == 1) dir_count++;
    }

    float prob_to_reach_weighted = 1.0f;
    if (dir_count > 0) {
        float dir_prob = float(dir_count) / float(light_count);
        float rng_val = random_float(rng);
        if (rng_val < dir_prob) {
            // Directional ışıklar arasından seç
            float step = dir_prob / float(dir_count);
            int sel = int(rng_val / step);
            int found = 0;
            for (int i = 0; i < light_count; i++) {
                if (optixLaunchParams.lights[i].type == 1) {
                    if (found == sel) {
                        if (pdf_out) *pdf_out = dir_prob / float(dir_count);
                        return i;
                    }
                    found++;
                }
            }
        }
        // Directional seçilmedi, weighted selection'a geç
        prob_to_reach_weighted = 1.0f - dir_prob;
    }

    // --- Tüm ışık türleri için akıllı seçim (directional hariç) ---
    float weights[128];
    float total_weight = 0.0f;

    for (int i = 0; i < light_count; i++) {
        const LightGPU& light = optixLaunchParams.lights[i];
        
        if (light.type == 1) { // Directional'lar weighted seçimde 0 ağırlık
            weights[i] = 0.0f;
            continue;
        }
        
        float dist = length(light.position - hit_position);
        dist = fmaxf(dist, 1.0f);
        
        float falloff = 1.0f / (dist * dist);
        float intensity = luminance(light.color * light.intensity);
        
        if (light.type == 0) { // Point Light
            float area = 4.0f * M_PIf * light.radius * light.radius;
            weights[i] = falloff * intensity * area;
        }
        else if (light.type == 2) { // Area Light
            float area = light.area_width * light.area_height;
            weights[i] = falloff * intensity * fminf(area, 10.0f);
        }
        else if (light.type == 3) { // Spot Light
            weights[i] = falloff * intensity * 0.8f;
        }
        else {
            weights[i] = 0.0f;
        }
        
        total_weight += weights[i];
    }

    // --- Eğer total_weight çok düşükse fallback ---
    if (total_weight < 1e-6f) {
        int idx = clamp(int(random_float(rng) * light_count), 0, light_count - 1);
        if (pdf_out) *pdf_out = prob_to_reach_weighted * (1.0f / float(light_count));
        return idx;
    }

    // --- Weighted seçim ---
    float r = random_float(rng) * total_weight;
    float accum = 0.0f;
    int selected = light_count - 1;
    for (int i = 0; i < light_count; i++) {
        accum += weights[i];
        if (r <= accum) {
            selected = i;
            break;
        }
    }

    if (pdf_out) *pdf_out = prob_to_reach_weighted * (weights[selected] / total_weight);
    return selected;
}

__device__ float3 sample_directional_light(const LightGPU& light, const float3& hit_pos, curandState* rng, float3& wi_out) {
    float3 L = normalize(light.direction);
    // Build tangent frame: check raw cross product length BEFORE normalize.
    // normalize(zero) produces NaN, and NaN<threshold is false → fallback would never fire.
    float3 tangent_raw = cross(L, make_float3(0.0f, 1.0f, 0.0f));
    if (dot(tangent_raw, tangent_raw) < 1e-6f) {
        tangent_raw = cross(L, make_float3(1.0f, 0.0f, 0.0f));
    }
    float3 tangent = normalize(tangent_raw);
    float3 bitangent = normalize(cross(L, tangent));

    float2 disk_sample = random_in_unit_disk(rng);
    float3 offset = (tangent * disk_sample.x + bitangent * disk_sample.y) * light.radius;

    float3 light_pos = hit_pos + L * 1000 + offset;
    wi_out = normalize(light_pos - hit_pos);
    return wi_out;
}

// AreaLight için rastgele nokta örnekleme
__device__ float3 sample_area_light(const LightGPU& light, curandState* rng) {
    float rand_u = random_float(rng) - 0.5f;
    float rand_v = random_float(rng) - 0.5f;
    return light.position 
        + light.area_u * rand_u * light.area_width 
        + light.area_v * rand_v * light.area_height;
}

// SpotLight için cone falloff hesabı
__device__ float spot_light_falloff(const LightGPU& light, const float3& wi) {
    float cos_theta = dot(-wi, normalize(light.direction));
    if (cos_theta < light.outer_cone_cos) return 0.0f;
    if (cos_theta > light.inner_cone_cos) return 1.0f;
    // Smooth falloff between inner and outer cone
    float t = (cos_theta - light.outer_cone_cos) / (light.inner_cone_cos - light.outer_cone_cos + 1e-6f);
    return t * t;  // Quadratic falloff
}

__device__ float3 calculate_light_contribution(
    const LightGPU& light,
    const GpuMaterial& material,
    const OptixHitResult& payload,
    const float3& wo,
    curandState* rng,
    float pdf_select = 1.0f,
    bool   resin_active  = false,
    float  resin_density = 0.0f,
    float3 resin_ext     = make_float3(0.0f, 0.0f, 0.0f)
) {
    float3 wi;
    float distance = 1.0f;
    float attenuation = 1.0f;
   // const float shadow_bias = SCENE_EPSILON;

    if (light.type == 0) { // Point Light
        float3 L = light.position - payload.position;
        distance = length(L);
        if (distance < 1e-3f) return make_float3(0.0f, 0.0f, 0.0f);
        float3 dir = normalize(L);
        float3 jitter = light.radius * random_in_unit_sphere(rng);
        wi = normalize(dir * distance + jitter);
        attenuation = 1.0f / (distance * distance);
    }
    else if (light.type == 1) { // Directional Light
        wi = sample_directional_light(light, payload.position, rng, wi);
        attenuation = 1.0f;
        distance = 1e8f;
    }
    else if (light.type == 2) { // Area Light
        float3 light_sample = sample_area_light(light, rng);
        float3 L = light_sample - payload.position;
        distance = length(L);
        if (distance < 1e-3f) return make_float3(0.0f, 0.0f, 0.0f);
        wi = normalize(L);
        
        // Cosine falloff based on light normal
        float3 light_normal = normalize(cross(light.area_u, light.area_v));
        float cos_light = fmaxf(dot(-wi, light_normal), 0.0f);
        attenuation = cos_light / (distance * distance);
    }
    else if (light.type == 3) { // Spot Light
        float3 L = light.position - payload.position;
        distance = length(L);
        if (distance < 1e-3f) return make_float3(0.0f, 0.0f, 0.0f);
        wi = normalize(L);
        
        // Spot cone falloff
        float falloff = spot_light_falloff(light, wi);
        if (falloff < 1e-4f) return make_float3(0.0f, 0.0f, 0.0f);
        attenuation = falloff / (distance * distance);
    }
    else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float NdotL = max(dot(payload.normal, wi),0.0001);
    //if (NdotL <= 0.001f) return make_float3(0.0f, 0.0f, 0.0f);

    // Terrain is a special case:
    // using the smoothed shading normal for shadow origin caused large-scale terrains
    // to self-shadow in a pixelated / low-resolution-looking way, especially as area size grew.
    // Keep terrain on geometric shadow normals; only regular meshes use the softened path.
    float3 shadow_normal = payload.is_terrain
        ? safe_normalize(payload.geom_normal, make_float3(0.0f, 1.0f, 0.0f))
        : safe_normalize(payload.normal, make_float3(0.0f, 1.0f, 0.0f));
    float3 origin = payload.position + shadow_normal * SCENE_EPSILON;
    Ray shadow_ray(origin, wi);

    unsigned int shadow_hit = trace_shadow_ray(shadow_ray, SCENE_EPSILON, distance);
    float shadow_visibility = shadow_hit ? 0.0f : 1.0f;
    if (shadow_visibility <= 1e-4f) return make_float3(0.0f, 0.0f, 0.0f);
    
    // Check VDB Occlusion (Volumetric Shadow)
    float vdb_transmittance = calculate_vdb_occlusion(origin, wi, distance, rng);
    if (vdb_transmittance < 0.001f) return make_float3(0.0f, 0.0f, 0.0f);

    float3 f = evaluate_brdf(material, payload, wo, wi);
    float pdf_brdf_val = pdf_brdf(material, wo, wi, payload.normal);
    float pdf_brdf_val_mis = clamp(pdf_brdf_val, 0.001f, 5000.0f);

    float3 Li = light.color * light.intensity * attenuation * vdb_transmittance * shadow_visibility;

    // Resin: the direct light also travels through the coat to reach the base.
    // Absorb it over its ENTRY path (light-angle slant) so thick/tinted resin
    // visibly dims direct lighting too — parity with the Vulkan NEE block.
    if (resin_active) {
        float cosL  = fmaxf(NdotL, 0.05f);
        float pLen  = resin_density / cosL;
        Li = Li * make_float3(expf(-pLen * resin_ext.x),
                              expf(-pLen * resin_ext.y),
                              expf(-pLen * resin_ext.z));
    }

    // Vulkan parity: Delta ışıklar (point/directional) için MIS uygulanmaz
    bool isDelta = (light.type == 0 || light.type == 1);
    if (isDelta) {
        // Delta light: mis_weight = 1.0, pdf_select ile bölünecek (caller'da)
        return f * Li * NdotL;
    }

    // Area/Spot ışıklar için Vulkan parity MIS: combined PDF = pdf_geo * pdf_select
    float pdf_light_geo = 1.0f;
    if (light.type == 2) { // Area Light
        float area = light.area_width * light.area_height;
        pdf_light_geo = 1.0f / fmaxf(area, 1e-4f);
    }
    else if (light.type == 3) { // Spot Light
        float solid_angle = 2.0f * M_PIf * (1.0f - light.outer_cone_cos);
        pdf_light_geo = 1.0f / fmaxf(solid_angle, 1e-4f);
    }

    float pdf_combined = pdf_light_geo * fmaxf(pdf_select, 1e-6f);
    float mis_weight = power_heuristic(pdf_combined, pdf_brdf_val_mis);
    // Vulkan: contrib = f * Li * NdotL * mis_weight / pdf_combined
    // Caller divides by pdf_select, so return f * Li * NdotL * mis_weight / pdf_light_geo
    return (f * Li * NdotL) * mis_weight / fmaxf(pdf_light_geo, 1e-6f);
}

__device__ float3 calculate_direct_lighting(
    const OptixHitResult& payload,
    const float3& wo,
    curandState* rng
) {
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    GpuMaterial mat = optixLaunchParams.materials[payload.material_id];

    int light_count = optixLaunchParams.light_count;
    if (light_count == 0) return result;

    // ------ YENİ: Rastgele bir ışık seç ------
    int light_index = clamp((int)(random_float(rng) * light_count), 0, light_count - 1);
    const LightGPU& light = optixLaunchParams.lights[light_index];

    float pdf_light_select = 1.0f / light_count;

    float3 wi;
    float distance = 1.0f;
    float attenuation = 1.0f;
    const float shadow_bias = 1e-3f; // Match CPU bias (was 1e-2f)


    // ==== Light sampling ====
    if (light.type == 0) { // Point Light
        float3 L = light.position - payload.position;
        distance = length(L);
        if (distance < 1e-3f) return result;

        float3 dir = normalize(L);
        float3 jitter = light.radius * random_in_unit_sphere(rng);
        wi = normalize(dir * distance + jitter);
        attenuation = 1.0f / (distance * distance);
    }
    else if (light.type == 1) { // Directional Light
        wi = sample_directional_light(light, payload.position, rng, wi);
        attenuation = 1.0f;
        distance = 1e8f;
    }
    else if (light.type == 2) { // Area Light
        float3 light_sample = sample_area_light(light, rng);
        float3 L = light_sample - payload.position;
        distance = length(L);
        if (distance < 1e-3f) return result;
        wi = normalize(L);
        
        float3 light_normal = normalize(cross(light.area_u, light.area_v));
        float cos_light = fmaxf(dot(-wi, light_normal), 0.0f);
        attenuation = cos_light / (distance * distance);
    }
    else if (light.type == 3) { // Spot Light
        float3 L = light.position - payload.position;
        distance = length(L);
        if (distance < 1e-3f) return result;
        wi = normalize(L);
        
        float falloff = spot_light_falloff(light, wi);
        if (falloff < 1e-4f) return result;
        attenuation = falloff / (distance * distance);
    }
    else {
        return result;
    }

    float NdotL = dot(payload.normal, wi);
    if (NdotL <= 0.001f) return result;

    // ==== Shadow ray ====
    // Terrain is a special case:
    // using the smoothed shading normal for shadow origin caused large-scale terrains
    // to self-shadow in a pixelated / low-resolution-looking way, especially as area size grew.
    // Keep terrain on geometric shadow normals; only regular meshes use the softened path.
    float3 shadow_normal = payload.is_terrain
        ? safe_normalize(payload.geom_normal, make_float3(0.0f, 1.0f, 0.0f))
        : safe_normalize(payload.normal, make_float3(0.0f, 1.0f, 0.0f));
    float3 origin = payload.position + shadow_normal * shadow_bias;
    Ray shadow_ray(origin, wi);

    if (trace_shadow_ray(shadow_ray, shadow_bias, distance) != 0u) return result;

    // ==== BRDF & PDF ====
    float3 f = evaluate_brdf(mat, payload, wo, wi);
    float pdf_brdf_val = pdf_brdf(mat, wo, wi, payload.normal);
    float pdf_brdf_val_mis = clamp(pdf_brdf_val, 0.01f, 5000.0f);

    // ==== Light PDF ====
    float pdf_light = 1.0f;
    if (light.type == 0) {
        float area = 4.0f * M_PIf * light.radius * light.radius;
        pdf_light = (1.0f / area)* pdf_light_select;
    }
    else if (light.type == 1) {
        float apparent_angle = atan2(light.radius, 1000.0f);
        float cos_epsilon = cos(apparent_angle);
        float solid_angle = 2.0f * M_PIf * (1.0f - cos_epsilon);
        pdf_light = (1.0f / solid_angle) * pdf_light_select;
    }
    else if (light.type == 2) { // Area Light
        float area = light.area_width * light.area_height;
        pdf_light = (1.0f / fmaxf(area, 1e-4f)) * pdf_light_select;
    }
    else if (light.type == 3) { // Spot Light
        float solid_angle = 2.0f * M_PIf * (1.0f - light.outer_cone_cos);
        pdf_light = (1.0f / fmaxf(solid_angle, 1e-4f)) * pdf_light_select;
    }

    float mis_weight = power_heuristic(pdf_light, pdf_brdf_val_mis);

    float3 Li = light.color * light.intensity * attenuation;
    result += (f * Li * NdotL) * mis_weight * light_count;
    return result;
}

__device__ float3 calculate_brdf_mis(
    const OptixHitResult& payload,
    const float3& wo,
    const Ray& scattered,
    const GpuMaterial& mat,
    const float pdf,
    curandState* rng
)
{
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    float3 wi = normalize(scattered.direction);

    float pdf_brdf_val_mis = clamp(pdf, 0.1f, 5000.0f);

    // ------ YENİ: Rastgele bir ışık seç ------
    int light_count = optixLaunchParams.light_count;
    if (light_count == 0) return result;  // Işık yoksa katkı yok.

    int light_index = clamp((int)(random_float(rng) * light_count), 0, light_count - 1);
    const LightGPU& light = optixLaunchParams.lights[light_index];

    // --- PDF light seçim katsayısı ---
    float pdf_light_select = 1.0f / light_count;

    // ---------- ESKİ KODLAR --------------
    if (light.type == 1) { // Directional
        float3 L = normalize(light.direction);
        float alignment = dot(wi, L);
        if (alignment > 0.999f) {
            float apparent_angle = atan2(light.radius, 1000.0f);
            float cos_epsilon = cos(apparent_angle);
            float solid_angle = 2.0f * M_PIf * (1.0f - cos_epsilon);
            float pdf_light = (1.0f / solid_angle) * pdf_light_select;  // Dikkat: pdf ışık seçimiyle bölünüyor

            float mis_weight = power_heuristic(pdf_brdf_val_mis, pdf_light);

            float3 f = evaluate_brdf(mat, payload, wo, wi);
            float NdotL = fmaxf(dot(payload.normal, wi), 0.0f);
            float3 Li = light.color * light.intensity ;
          
            result += (f * Li * NdotL) * mis_weight * light_count; // Light_count çarpılıyor, çünkü sadece bir ışık örneklendi.
        }
    }

    if (light.type == 0) { // Point Light
        float3 delta = light.position - payload.position;
        float dist = length(delta);
        if (dist < light.radius * 1.05f) {
            float area = 4.0f * M_PIf * light.radius * light.radius;
            float pdf_light = (1.0f / area) * pdf_light_select;

            float mis_weight = power_heuristic(pdf_brdf_val_mis, pdf_light);

            float3 f = evaluate_brdf(mat, payload, wo, wi);
            float NdotL = fmaxf(dot(payload.normal, wi), 0.0f);
            float3 Li = light.color * light.intensity / (dist * dist);
			
            result += (f * Li * NdotL) * mis_weight * light_count; // Sadece seçilen ışık örneklendiği için çarpım.
        }
    }

    return result;
}

// =============================================================================
// VOLUMETRIC OBJECT RAY MARCHING - WITH MULTI-SCATTERING
// Renders volumetric materials inside object bounds using ray marching
// Features: Dual-lobe phase, light marching, multi-scatter transmittance
// =============================================================================

__device__ float3 raymarch_volumetric_object(
    const float3& ray_origin,
    const float3& ray_dir,
    const float3& aabb_min,
    const float3& aabb_max,
    float vol_density,
    float vol_absorption,
    float vol_scattering,
    const float3& vol_albedo,
    const float3& vol_emission,
    float vol_g,
    float step_size,
    int max_steps,
    float noise_scale,
    // Multi-scattering parameters
    float multi_scatter,
    float g_back,
    float lobe_mix,
    int light_steps,
    float shadow_strength,
    // NanoVDB parameters
    void* nanovdb_grid,
    int has_nanovdb,
    float& out_transmittance,
    curandState* rng
) {
    float3 ambient_dir = gpu_get_volume_ambient_dir(optixLaunchParams.world, ray_dir);
    // Ray-AABB intersection
    float3 inv_dir = make_float3(
        fabsf(ray_dir.x) > 1e-6f ? 1.0f / ray_dir.x : 1e6f,
        fabsf(ray_dir.y) > 1e-6f ? 1.0f / ray_dir.y : 1e6f,
        fabsf(ray_dir.z) > 1e-6f ? 1.0f / ray_dir.z : 1e6f
    );
    
    float3 t0 = (aabb_min - ray_origin) * inv_dir;
    float3 t1 = (aabb_max - ray_origin) * inv_dir;
    
    float3 tmin_v = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    float3 tmax_v = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
    
    float t_enter = fmaxf(fmaxf(tmin_v.x, tmin_v.y), tmin_v.z);
    float t_exit = fminf(fminf(tmax_v.x, tmax_v.y), tmax_v.z);
    
    // No intersection
    if (t_exit < t_enter || t_exit < 0.0f) {
        out_transmittance = 1.0f;
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    // Clamp entry point
    if (t_enter < 0.0f) t_enter = 0.0f;
    
    // Initialize
    float3 accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
    float3 transmittance = make_float3(1.0f);
    
    // Raymarching span
    float volume_extent = t_exit - t_enter;
    
    // Safety Step Calculation: Ensures we reach the exit before hitting max_steps
    float min_step_to_cover = volume_extent / fmaxf(1.0f, (float)max_steps - 1.0f);
    float actual_step_size = fmaxf(step_size, min_step_to_cover);
    
    // Temporal jitter to reduce banding
    float jitter = curand_uniform(rng) * actual_step_size;
    float t = t_enter + jitter;
    int steps = 0;
    
    // Get sun direction for lighting
    float3 sun_dir = normalize(optixLaunchParams.world.nishita.sun_direction);
    float sun_intensity = optixLaunchParams.world.nishita.sun_intensity;
    
    // Precompute light march step size
    float light_step_size = volume_extent / fmaxf((float)light_steps, 1.0f);
    
    while (t < t_exit && steps < max_steps && (transmittance.x + transmittance.y + transmittance.z) > 0.03f) {
        // --- STOCHASTIC EDGE SMOOTHING ---
        
        
        float3 pos = ray_origin + ray_dir * t;
        
        // Density at this point
        float local_density = vol_density;
        
        // ═══════════════════════════════════════════════════════════
        // DENSITY SOURCE: NanoVDB or Procedural Noise
        // ═══════════════════════════════════════════════════════════
        if (has_nanovdb && nanovdb_grid) {
             // Cast the void pointer to FloatGrid (GPU pointer)
             const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(nanovdb_grid);
             
             // Get accessor - lightweight object, safe to create on stack
             // Note: using linear interpolation (order=1) for smooth clouds
             // If SampleFromVoxels is not available, we could use acc.getValue(coord) but that's nearest neighbor
             auto acc = grid->getAccessor();
             auto sampler = nanovdb::math::createSampler<1>(acc);
             
             // Convert world position to index space (float coordinates)
             nanovdb::Vec3f pos_vdb(pos.x, pos.y, pos.z);
             nanovdb::Vec3f index_coord = grid->worldToIndexF(pos_vdb);
             
             // Sample density
             local_density = sampler(index_coord) * vol_density;
        }
        else if (noise_scale > 0.01f) {
            // Apply procedural noise modulation
            float3 aabb_size = aabb_max - aabb_min;
            float3 local_pos = (pos - aabb_min) / fmaxf(fmaxf(aabb_size.x, aabb_size.y), fmaxf(aabb_size.z, 0.001f));
            float3 noise_pos = local_pos * noise_scale;
            
            // Simple FBM-like noise with 3 octaves
            float noise_val = 0.0f;
            float amplitude = 0.5f;
            float frequency = 1.0f;
            for (int i = 0; i < 3; i++) {
                float3 p = noise_pos * frequency;
                float3 pi = make_float3(floorf(p.x), floorf(p.y), floorf(p.z));
                float3 pf = make_float3(p.x - pi.x, p.y - pi.y, p.z - pi.z);
                float3 pf2 = pf * pf * (make_float3(3.0f, 3.0f, 3.0f) - 2.0f * pf);
                
                float n = pi.x + pi.y * 57.0f + pi.z * 113.0f;
                float v1 = fmodf(sinf(n) * 43758.5453f, 1.0f);
                float v2 = fmodf(sinf(n + 1.0f) * 43758.5453f, 1.0f);
                float v3 = fmodf(sinf(n + 57.0f) * 43758.5453f, 1.0f);
                float v4 = fmodf(sinf(n + 58.0f) * 43758.5453f, 1.0f);
                float v5 = fmodf(sinf(n + 113.0f) * 43758.5453f, 1.0f);
                float v6 = fmodf(sinf(n + 114.0f) * 43758.5453f, 1.0f);
                float v7 = fmodf(sinf(n + 170.0f) * 43758.5453f, 1.0f);
                float v8 = fmodf(sinf(n + 171.0f) * 43758.5453f, 1.0f);
                
                v1 = v1 + (v2 - v1) * pf2.x;
                v3 = v3 + (v4 - v3) * pf2.x;
                v5 = v5 + (v6 - v5) * pf2.x;
                v7 = v7 + (v8 - v7) * pf2.x;
                v1 = v1 + (v3 - v1) * pf2.y;
                v5 = v5 + (v7 - v5) * pf2.y;
                v1 = v1 + (v5 - v1) * pf2.z;
                
                noise_val += v1 * amplitude;
                amplitude *= 0.5f;
                frequency *= 2.0f;
            }
            
            // Simple density modulation: noise_val ~0.3-0.7, scale to 0.0-1.0 range
            local_density *= noise_val;
        }
        float edge_factor = 1.0f;
        // ═══════════════════════════════════════════════════════════
        // EDGE FALLOFF - Smooth transition at AABB boundaries
        // ═══════════════════════════════════════════════════════════
        {
            float3 aabb_size = aabb_max - aabb_min;
            // Falloff distance as percentage of volume size (10% each side)
            float falloff_dist = fminf(fminf(aabb_size.x, aabb_size.y), aabb_size.z) * 0.15f;
            
            // Distance to each face of the AABB
            float dx_min = pos.x - aabb_min.x;
            float dx_max = aabb_max.x - pos.x;
            float dy_min = pos.y - aabb_min.y;
            float dy_max = aabb_max.y - pos.y;
            float dz_min = pos.z - aabb_min.z;
            float dz_max = aabb_max.z - pos.z;
            
            // Combined edge falloff (minimum distance to any face)
            float d_edge = fminf(fminf(fminf(dx_min, dx_max), fminf(dy_min, dy_max)), fminf(dz_min, dz_max));
            
            // Smooth falloff using smoothstep
           
            if (d_edge < falloff_dist && falloff_dist > 0.001f) {
                float t_f = d_edge / falloff_dist;
                // Smoothstep: 3t^2 - 2t^3
                edge_factor = t_f * t_f * (3.0f - 2.0f * t_f);
            }
            
            local_density *= edge_factor;
        }
        float threshold = curand_uniform(rng) * 0.01f * edge_factor;
        if (local_density > threshold) {
            // Compute extinction coefficient
            float sigma_a = local_density * vol_absorption; 
            float sigma_s = local_density * vol_scattering;
            float sigma_t = sigma_a + sigma_s;
            float albedo_avg = vol_albedo.x * 0.2126f + vol_albedo.y * 0.7152f + vol_albedo.z * 0.0722f;
            
            // --- BLENDED MULTI-SCATTER TRANSMITTANCE (Matches CPU Scalar Model) ---
            float T_single = expf(-sigma_t * actual_step_size);
            float T_multi_p = expf(-sigma_t * actual_step_size * 0.25f);
            float step_transmittance = T_single * (1.0f - multi_scatter * albedo_avg) + T_multi_p * (multi_scatter * albedo_avg);
            
            float shadow_trans = 1.0f;
            if (light_steps > 0) {
                float density_accum = 0.0f;
                for (int j = 1; j <= light_steps; ++j) {
                    float3 light_pos = pos + sun_dir * (light_step_size * (float)j);
                    if (light_pos.x < aabb_min.x || light_pos.x > aabb_max.x || light_pos.y < aabb_min.y || light_pos.y > aabb_max.y || light_pos.z < aabb_min.z || light_pos.z > aabb_max.z) break;

                    float3 aabb_size_l = aabb_max - aabb_min;
                    float falloff_l = fminf(fminf(aabb_size_l.x, aabb_size_l.y), aabb_size_l.z) * 0.15f;
                    float dx_l = fminf(light_pos.x - aabb_min.x, aabb_max.x - light_pos.x);
                    float dy_l = fminf(light_pos.y - aabb_min.y, aabb_max.y - light_pos.y);
                    float dz_l = fminf(light_pos.z - aabb_min.z, aabb_max.z - light_pos.z);
                    float d_edge_l = fminf(fminf(dx_l, dy_l), dz_l);
                    float t_f_l = fminf(1.0f, d_edge_l / fmaxf(falloff_l, 0.001f));
                    float light_fade = t_f_l * t_f_l * (3.0f - 2.0f * t_f_l);

                    density_accum += local_density * light_fade * (vol_absorption + vol_scattering) * light_step_size;
                    if (density_accum > 5.0f) break;
                }
                // --- STABLE SHADOWING (Matches CPU 1.0 - strength * (1-T)) ---
                float beers = expf(-density_accum);
                float phys_trans = beers;
                
                // Multi-scatter softening guard
                if (sigma_s > 1e-6f && multi_scatter > 1e-6f) {
                    float beers_soft = expf(-density_accum * 0.25f);
                    float albedo_p = vol_albedo.x * 0.2126f + vol_albedo.y * 0.7152f + vol_albedo.z * 0.0722f;
                    phys_trans = beers * (1.0f - multi_scatter * albedo_p) + beers_soft * (multi_scatter * albedo_p);
                }
                shadow_trans = 1.0f - shadow_strength * (1.0f - phys_trans);
            }
            
            float cos_theta = dot(ray_dir, sun_dir);
            float phase = gpu_phase_dual_hg(cos_theta, vol_g, g_back, lobe_mix);
            float powder = gpu_powder_effect(local_density, cos_theta);
            phase *= (1.0f + powder * 0.5f);

            float3 sun_trans = gpu_get_transmittance(optixLaunchParams.world, pos, sun_dir);
            float3 sun_color = sun_trans * sun_intensity;
            float3 shadow_radiance = sun_color * shadow_trans * phase;
            
            // Physical Parity: Ambient sky light should scale with atmosphere intensity.
            // Gated by volume_atmosphere_ambient (default OFF, Vulkan parity).
            float3 ambient = make_float3(0.0f, 0.0f, 0.0f);
            if (optixLaunchParams.world.volume_atmosphere_ambient != 0) {
                ambient = gpu_get_ambient_radiance_volume(optixLaunchParams.world, ambient_dir) * 0.15f * optixLaunchParams.world.nishita.atmosphere_intensity;
            }
            float3 total_light = shadow_radiance + ambient;

            float3 ms_boost = make_float3(1.0f) + vol_albedo * multi_scatter * 2.0f;
            float3 source = (vol_albedo * total_light * sigma_s * ms_boost + vol_emission * local_density);
            
            // Energy-stable integration (Parity with Renderer.cpp)
            float one_minus_T = 1.0f - step_transmittance;
            accumulated_color += source * (one_minus_T * transmittance);
            transmittance *= step_transmittance;
        }
        t += actual_step_size;
        steps++;
    }
    out_transmittance = (transmittance.x + transmittance.y + transmittance.z) / 3.0f;
    return accumulated_color;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GAS VOLUME RAY MARCHING (Texture3D)
// Simple dense grid sampling from Texture Object
// ═══════════════════════════════════════════════════════════════════════════════
__device__ float3 raymarch_gas_volume(
    const GpuGasVolume& vol,
    const float3& ray_origin,
    const float3& ray_dir,
    const float3& sun_dir,
    float sun_intensity,
    float& out_transmittance,
    float max_t,
    curandState* rng
) {
    if (!vol.density_texture) {
         out_transmittance = 1.0f;
         return make_float3(0.0f);
    }

    // Transform ray to local space
    float3 local_origin = transform_point_affine(ray_origin, vol.inv_transform);
    float3 local_dir = transform_vector_affine(ray_dir, vol.inv_transform);
    float3 ambient_dir = gpu_get_volume_ambient_dir(optixLaunchParams.world, ray_dir);
    float dir_length = length(local_dir);
    if (dir_length < 1e-8f) return make_float3(0.0f);
    local_dir /= dir_length;

    // Ray-AABB intersection
    float t_enter, t_exit;
    if (!intersect_aabb_vdb(local_origin, local_dir, vol.local_bbox_min, vol.local_bbox_max, t_enter, t_exit)) {
        out_transmittance = 1.0f;
        return make_float3(0.0f);
    }
    
    // Clip against max_t
    float local_max_t = max_t * dir_length;
    t_exit = fminf(t_exit, local_max_t);
    t_enter = fmaxf(t_enter, 0.001f);
    
    if (t_enter >= t_exit) {
        out_transmittance = 1.0f;
        return make_float3(0.0f);
    }

    float3 accumulated_color = make_float3(0.0f);
    float3 transmittance = make_float3(1.0f);
    
    // Convert step size to world space logic
    float3 vol_size = vol.local_bbox_max - vol.local_bbox_min;
    float world_t_enter = t_enter / dir_length;
    float world_t_exit = t_exit / dir_length;
    
    // Safety Step Calculation: Ensures we reach the exit before hitting max_steps
    float volume_extent = world_t_exit - world_t_enter;
    float min_step_to_cover = volume_extent / fmaxf(1.0f, (float)vol.max_steps - 1.0f);
    float base_step = fmaxf(vol.step_size <= 0.0f ? 0.05f : vol.step_size, min_step_to_cover);
    float min_step = fmaxf(base_step * 0.25f, 0.0001f);
    const float tau_max = 0.2f;
    
    // Jitter
    float jitter = curand_uniform(rng) * base_step;
    float t = world_t_enter + jitter;

    int steps = 0;
    
    while (t < world_t_exit && steps < vol.max_steps && (transmittance.x + transmittance.y + transmittance.z) > 0.03f) {
        // --- STOCHASTIC EDGE SMOOTHING ---
        float threshold = curand_uniform(rng) * 0.01f;
        
        float3 world_pos = ray_origin + ray_dir * t;
        float3 local_pos = transform_point_affine(world_pos, vol.inv_transform);
        
        // Normalize local_pos to texture coordinates [0,1]
        // Assuming local_bbox matches the grid dimensions
        float3 tex_coord = (local_pos - vol.local_bbox_min) / (vol.local_bbox_max - vol.local_bbox_min);
        
        // Sample Density
        float density = tex3D<float>(vol.density_texture, tex_coord.x, tex_coord.y, tex_coord.z);
        if (!isfinite(density)) density = 0.0f;
        
        // Remap
        float remap_range = vol.density_remap_high - vol.density_remap_low + 1e-6f;
        density = (density - vol.density_remap_low) / remap_range;
        density = fmaxf(density, 0.0f) * vol.density_multiplier;
        
        float sigma_a = density * vol.absorption_coefficient;
        float sigma_s = density * vol.scatter_coefficient;
        float sigma_t = sigma_a + sigma_s;
        float sparse_cutoff = (vol.density_pad > 0.0f) ? vol.density_pad : 0.04f;
        float scatter_keep = fminf(1.0f, fmaxf(0.0f, (sigma_s * base_step) / sparse_cutoff));
        if (curand_uniform(rng) <= scatter_keep) {
            if (sigma_t <= 1e-8f) { t += base_step; steps++; continue; }
            float step = fminf(base_step, tau_max / sigma_t);
            step = fmaxf(step, min_step);
            step = fminf(step, world_t_exit - t);
            if (step <= 1e-8f) break;
            float albedo_avg = vol.scatter_color.x * 0.2126f + vol.scatter_color.y * 0.7152f + vol.scatter_color.z * 0.0722f;
            
            // --- BLENDED MULTI-SCATTER TRANSMITTANCE (Matches CPU Scalar Model) ---
            float T_single = expf(-sigma_t * step);
            float T_multi_p = expf(-sigma_t * step * 0.25f);
            float step_transmittance = T_single * (1.0f - vol.scatter_multi * albedo_avg) + T_multi_p * (vol.scatter_multi * albedo_avg);
            float one_minus_T = 1.0f - step_transmittance;
            
            float3 total_radiance = make_float3(0.0f);
            float cos_theta = dot(ray_dir, sun_dir);
            float phase = gpu_phase_dual_hg(cos_theta, vol.scatter_anisotropy, vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
            
            float shadow_trans = 1.0f;
            if (vol.shadow_steps > 0) {
                 int s_steps = max(1, vol.shadow_steps);
                 float tau_hint = fmaxf(density, 0.0f) * (vol.absorption_coefficient + vol.scatter_coefficient) * volume_extent;
                 if (s_steps > 8) {
                     float step_scale = fminf(1.0f, fmaxf(0.25f, sqrtf(fmaxf(tau_hint, 0.0f))));
                     s_steps = (int)ceilf((float)s_steps * step_scale);
                     s_steps = max(3, min(s_steps, min(vol.shadow_steps, 16)));
                 }
                 float shadow_step = volume_extent / (float)max(1, s_steps);
                 float s_jitter = curand_uniform(rng);
                 float shadow_density_sum = 0.0f;
                 for(int ls=0; ls < s_steps; ++ls) {
                     float3 shadow_pos = world_pos + sun_dir * (shadow_step * ((float)ls + s_jitter + 0.5f));
                     float3 l_shadow = transform_point_affine(shadow_pos, vol.inv_transform);
                     if (l_shadow.x < vol.local_bbox_min.x || l_shadow.x > vol.local_bbox_max.x ||
                         l_shadow.y < vol.local_bbox_min.y || l_shadow.y > vol.local_bbox_max.y ||
                         l_shadow.z < vol.local_bbox_min.z || l_shadow.z > vol.local_bbox_max.z) break;
                     float3 stcind = (l_shadow - vol.local_bbox_min) / (vol.local_bbox_max - vol.local_bbox_min);
                     float sd = tex3D<float>(vol.density_texture, stcind.x, stcind.y, stcind.z);
                     sd = (sd - vol.density_remap_low)/(vol.density_remap_high - vol.density_remap_low + 1e-6f) * vol.density_multiplier;
                     if(sd > 0.0f) shadow_density_sum += sd;
                     if(shadow_density_sum > 10.0f) break; 
                 }
                 // --- STABLE SHADOWING (Matches CPU 1.0 - strength * (1-T)) ---
                 float beers = expf(-shadow_density_sum * shadow_step * (vol.absorption_coefficient + vol.scatter_coefficient));
                 float phys_trans = beers;
                 
                 // Multi-scatter softening guard: Only soften shadow if scattering is present
                 if (vol.scatter_coefficient > 1e-6f && vol.scatter_multi > 1e-6f) {
                     float beers_soft = expf(-shadow_density_sum * shadow_step * (vol.absorption_coefficient + vol.scatter_coefficient) * 0.25f);
                     float albedo_lum = vol.scatter_color.x * 0.2126f + vol.scatter_color.y * 0.7152f + vol.scatter_color.z * 0.0722f;
                     phys_trans = beers * (1.0f - vol.scatter_multi * albedo_lum) + beers_soft * (vol.scatter_multi * albedo_lum);
                 }
                 float shadow_strength = fminf(fmaxf(vol.shadow_strength * 1.08f, 0.0f), 1.0f);
                 shadow_trans = 1.0f - shadow_strength * (1.0f - phys_trans);
            }
            float3 sun_trans = gpu_get_transmittance(optixLaunchParams.world, world_pos, sun_dir);
            float3 sun_color = sun_trans * sun_intensity;
            total_radiance = sun_color * shadow_trans * phase;

            // --- RESTORED PARITY FEATURES (Powder + Ambient) ---
            float powder = gpu_powder_effect(density, cos_theta);
            total_radiance = total_radiance * (1.0f + powder * 0.5f);
            
            float thin_scatter = scatter_keep * scatter_keep;
            // Gated by volume_atmosphere_ambient (default OFF, Vulkan parity).
            if (optixLaunchParams.world.volume_atmosphere_ambient != 0) {
                float3 ambient = gpu_get_ambient_radiance_volume(optixLaunchParams.world, ambient_dir) * (0.15f * thin_scatter) * optixLaunchParams.world.nishita.atmosphere_intensity;
                total_radiance += ambient;
            }
            
            float3 emission = make_float3(0.0f);
            if (vol.emission_mode == 1) { // Constant
                emission = vol.emission_color * vol.emission_intensity * density;
            }
            else if (vol.emission_mode >= 2) { // Blackbody / Color Ramp / Channel-driven
                float temperature = density; 
                if (vol.temperature_texture) {
                    temperature = tex3D<float>(vol.temperature_texture, tex_coord.x, tex_coord.y, tex_coord.z);
                    if (!isfinite(temperature)) temperature = density; 
                }
                
                float3 e_color; float kelvin; float t_ramp_val;
                if (temperature > 20.0f) { // Likely physical Kelvin
                    kelvin = temperature * vol.temperature_scale;
                    float ramp_min = (vol.emission_pad > 20.0f) ? vol.emission_pad : 0.0f;
                    float ramp_max = (vol.max_temperature > ramp_min + 1.0f) ? vol.max_temperature : 6000.0f;
                    t_ramp_val = (temperature - ramp_min) / fmaxf(ramp_max - ramp_min, 1.0f);
                } else { // Likely normalized 0-1
                    kelvin = (temperature * 3000.0f + 1000.0f) * vol.temperature_scale;
                    t_ramp_val = temperature;
                }
                float ramp_t_clamped = fminf(fmaxf(t_ramp_val, 0.0f), 1.0f);
                if (vol.color_ramp_enabled) e_color = sample_color_ramp_gas(vol, ramp_t_clamped);
                else e_color = blackbody_to_rgb(kelvin);
                float emit_gate = (vol.color_ramp_enabled && temperature > 20.0f && t_ramp_val <= 0.0f) ? 0.0f : 1.0f;
                emission = clamp_volume_radiance(e_color * density * vol.blackbody_intensity * emit_gate, 64.0f);
            }

            float3 ms_boost = make_float3(1.0f) + vol.scatter_color * vol.scatter_multi * (2.0f * thin_scatter);
            float3 source = (vol.scatter_color * total_radiance * sigma_s * ms_boost + emission);
            
            // CPU/Vulkan parity integration
            accumulated_color += source * (one_minus_T * transmittance);
            transmittance *= step_transmittance;
            t += step;
            steps++;
            continue;
        }
        
        t += base_step;
        steps++;
    }

    out_transmittance = (transmittance.x + transmittance.y + transmittance.z) / 3.0f;
    return accumulated_color;
}

__device__ float3 ray_color(Ray ray, curandState* rng, float3* primary_albedo_out = nullptr, float3* primary_normal_out = nullptr, int* primary_hit_out = nullptr,
                            float3* primary_world_pos_out = nullptr, int* primary_material_id_out = nullptr) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    // max_depth = total path bounces (must be >= 1). bounce=1 ⇒ primary
    // hit + direct lighting only; bounce=N ⇒ primary + N-1 indirect.
    // The UI clamps the slider minimum to 1 so this loop always runs at
    // least once on a hit. Both Vulkan and OptiX backends share this
    // convention.
    const int max_depth = optixLaunchParams.max_depth;
    int light_count = optixLaunchParams.light_count;
    int light_index = -1;
    
    float first_hit_t = -1.0f;
    float first_hit_transmission = 0.0f;
    bool path_touched_transmissive = false;
    float vol_depth = -1.0f;
    float vol_trans_accum = 1.0f;
    float3 first_ray_origin = ray.origin;
    float3 first_ray_dir = ray.direction;
    const int max_diffuse_depth = max(0, optixLaunchParams.diffuse_depth);
    const int max_transmission_depth = max(0, optixLaunchParams.transmission_depth);
    int diffuse_depth = 0;
    int transmission_depth = 0;
    // Resin is energy-preserving (thin/white coats don't decay throughput), so RR
    // can't kill its paths and a full-depth GI run gets very expensive. Cap resin
    // interactions to a small budget — base keeps full direct lighting (NEE); only
    // barely-visible deep indirect GI under the glossy coat is dropped. Vulkan parity.
    int resin_depth = 0;
    const int RESIN_MAX_DEPTH = 2;
    
    // Firefly önleme için maksimum katkı limiti (Vulkan parity: 1e4)
    const float MAX_CONTRIBUTION = 10000.0f;

    for (int bounce = 0; bounce < max_depth; ++bounce) {
        OptixHitResult payload = {};
        float t_min = (bounce == 0) ? optixLaunchParams.clip_near : SCENE_EPSILON;
        float t_max = (bounce == 0) ? optixLaunchParams.clip_far : 1e16f;

        // Path regularization hint: scatter_material / evaluate_brdf read this
        // to clamp roughness on indirect bounces (Müller 2018). Written BEFORE
        // trace_ray so closesthit can also see it and skip the primary-ray-only
        // AOV outputs (primary_albedo / primary_normal / primary_hit + their
        // albedo texture fetch) on indirect bounces — those values are only
        // consumed by the denoiser path at bounce==0.
        payload.bounce_index = bounce;

        trace_ray(ray, &payload, t_min, t_max);

        // --- 1. HANDLE HAIR (Unified Path) ---
        if (payload.hit && payload.is_hair) {
            if (bounce == 0) {
                if (primary_albedo_out) *primary_albedo_out = payload.primary_albedo;
                if (primary_normal_out) *primary_normal_out = payload.primary_normal;
                if (primary_hit_out) *primary_hit_out = payload.primary_hit;
                // Stylize AOV: world hit position; hair has no material index → leave -1 (unknown)
                if (primary_world_pos_out) *primary_world_pos_out = ray.origin + ray.direction * payload.t;
                if (primary_material_id_out) *primary_material_id_out = -1;
            }
            color += throughput * payload.color;
            
            // For now, hair is treated as opaque to keep it simple and visible
            // If it's a primary ray, we still need this for fog/picking
            if (bounce == 0) {
                first_hit_t = payload.t;
                if (optixLaunchParams.pick_buffer != nullptr && optixLaunchParams.frame_number == 0) {
                    const uint3 launch_idx = optixGetLaunchIndex();
                    int pixel_idx = launch_idx.y * optixLaunchParams.image_width + launch_idx.x;
                    optixLaunchParams.pick_buffer[pixel_idx] = payload.object_id;
                }
            }
            break; // Stop path at hair
        }

        if (bounce == 0 && payload.hit) {
            first_hit_t = payload.t;
            
            // ═══════════════════════════════════════════════════════════
            // GPU PICKING - Write object ID to pick buffer on primary hit
            // This enables O(1) viewport object selection from GPU render
            // ═══════════════════════════════════════════════════════════
            if (optixLaunchParams.pick_buffer != nullptr) {
                const uint3 launch_idx = optixGetLaunchIndex();
                int pixel_idx = launch_idx.y * optixLaunchParams.image_width + launch_idx.x;
                
                // Only update on first sample (frame_number == 0 or sample pass 0)
                // This avoids race conditions and ensures stable pick results
                if (optixLaunchParams.frame_number == 0) {
                    optixLaunchParams.pick_buffer[pixel_idx] = payload.object_id;
                    if (optixLaunchParams.pick_depth_buffer != nullptr) {
                        optixLaunchParams.pick_depth_buffer[pixel_idx] = payload.t;
                    }
                }
            }
        } else if (bounce == 0 && !payload.hit) {
            if (primary_albedo_out) *primary_albedo_out = make_float3(0.0f);
            if (primary_normal_out) *primary_normal_out = make_float3(0.0f);
            if (primary_hit_out) *primary_hit_out = 0;
            if (primary_world_pos_out) *primary_world_pos_out = make_float3(0.0f);
            if (primary_material_id_out) *primary_material_id_out = -1;
            // Miss on primary ray - write -1 to pick buffer (no object)
            if (optixLaunchParams.pick_buffer != nullptr && optixLaunchParams.frame_number == 0) {
                const uint3 launch_idx = optixGetLaunchIndex();
                int pixel_idx = launch_idx.y * optixLaunchParams.image_width + launch_idx.x;
                optixLaunchParams.pick_buffer[pixel_idx] = -1;
                if (optixLaunchParams.pick_depth_buffer != nullptr) {
                    optixLaunchParams.pick_depth_buffer[pixel_idx] = -1.0f;
                }
            }
        }

        if (optixLaunchParams.vdb_volumes && optixLaunchParams.vdb_volume_count > 0) {
            float3 sun_dir = normalize(optixLaunchParams.world.nishita.sun_direction);
            float sun_intensity = optixLaunchParams.world.nishita.sun_intensity;

            // ═══════════════════════════════════════════════════════════
            // INDUSTRY-STANDARD: Per-ray depth sorting for correct compositing
            // Compute entry distances, sort front-to-back, then render
            // ═══════════════════════════════════════════════════════════
            const int MAX_VDB_SORT = 16;  // Max VDBs to sort per ray
            int vdb_count = min(optixLaunchParams.vdb_volume_count, MAX_VDB_SORT);

            // Store VDB indices and their entry distances
            int   sorted_indices[MAX_VDB_SORT];
            float entry_distances[MAX_VDB_SORT];
            int   valid_count = 0;

            float3 ray_dir_norm = normalize(ray.direction);

            // Step 1: Compute entry distances for all VDBs
            for (int v = 0; v < vdb_count; ++v) {
                const GpuVDBVolume& vdb = optixLaunchParams.vdb_volumes[v];
                // Transform ray to local space and compute AABB intersection
                float3 local_origin = transform_point_affine(ray.origin, vdb.inv_transform);
                float3 local_dir = transform_vector_affine(ray_dir_norm, vdb.inv_transform);
                float dir_len = length(local_dir);
                if (dir_len < 1e-8f) continue;
                local_dir = local_dir / dir_len;

                float t_enter, t_exit;
                if (intersect_aabb_vdb(local_origin, local_dir, vdb.local_bbox_min, vdb.local_bbox_max, t_enter, t_exit)) {
                    // Convert to world space distance
                    float world_t_enter = t_enter / dir_len;
                    if (world_t_enter < 0.0f) world_t_enter = 0.0f;

                    // Bias iso-surface (source_type==4) volumes slightly back so a
                    // COINCIDENT fog volume (e.g. whitewater foam sharing the fluid
                    // domain AABB) is marched FIRST — its scattering is added before
                    // the surface redirect breaks the volume loop. Without this the
                    // foam only contributes a shadow on the water (no white showing).
                    if (vdb.source_type == 4) world_t_enter += 1e-3f;

                    sorted_indices[valid_count] = v;
                    entry_distances[valid_count] = world_t_enter;
                    valid_count++;
                }
            }

            // Step 2: Sort by entry distance (insertion sort - efficient for small N)
            for (int i = 1; i < valid_count; ++i) {
                int   key_idx = sorted_indices[i];
                float key_dist = entry_distances[i];
                int j = i - 1;
                while (j >= 0 && entry_distances[j] > key_dist) {
                    sorted_indices[j + 1] = sorted_indices[j];
                    entry_distances[j + 1] = entry_distances[j];
                    j--;
                }
                sorted_indices[j + 1] = key_idx;
                entry_distances[j + 1] = key_dist;
            }

            // Step 3: Render in sorted order (front-to-back)
            bool fluid_redirect = false;
            float3 fluid_new_origin = make_float3(0.0f);
            float3 fluid_new_dir = make_float3(0.0f);
            for (int i = 0; i < valid_count; ++i) {
                const GpuVDBVolume& vdb = optixLaunchParams.vdb_volumes[sorted_indices[i]];

                // Determine occlusion distance (depth clipping)
                float max_dist = payload.hit ? payload.t : 1e16f;
                if (bounce == 0 && max_dist > optixLaunchParams.clip_far) max_dist = optixLaunchParams.clip_far;

                float vol_transmittance = 1.0f;
                bool   surf_redirect = false;
                float3 surf_o = make_float3(0.0f), surf_d = make_float3(0.0f), surf_tp = make_float3(1.0f);
                float3 vol_color = raymarch_vdb_volume(
                    vdb,
                    ray.origin,
                    ray_dir_norm,
                    sun_dir,
                    sun_intensity,
                    vol_transmittance,
                    max_dist,
                    rng,
                    &surf_redirect, &surf_o, &surf_d, &surf_tp
                );

                // Accumulate volume contribution (reflection lobe for a fluid
                // surface; participating-media colour otherwise).
                color += throughput * vol_color;

                // Fluid surface in front of the geometry hit: redirect the path
                // along the refracted ray and continue the bounce (true through-
                // water distortion). Only when the surface is actually the
                // nearest thing — otherwise geometry in front would be skipped.
                if (surf_redirect && entry_distances[i] <= (payload.hit ? payload.t : 1e16f)) {
                    throughput *= surf_tp;
                    fluid_new_origin = surf_o;
                    fluid_new_dir = surf_d;
                    fluid_redirect = true;
                    if (bounce == 0) {
                        if (vol_depth < 0.0f || entry_distances[i] < vol_depth)
                            vol_depth = entry_distances[i];
                        // Aerial perspective fix: when the primary ray refracts THROUGH
                        // the fluid surface, the water IS the nearest hit. first_hit_t
                        // was only set from the geometry behind (farther) — or left -1
                        // when nothing is behind, which the post-process turns into a
                        // 10000 fallback. Either way Nishita aerial perspective then
                        // fogs the surface at a huge distance and the water washes out /
                        // disappears. Pin the fog distance to the actual surface entry
                        // (matches the CPU reference backend's SDF dielectric fix).
                        if (first_hit_t < 0.0f || entry_distances[i] < first_hit_t)
                            first_hit_t = entry_distances[i];
                        // ...and mark it FULLY transmissive (1.0) — EXACT parity with
                        // the Vulkan path, whose fluid-surface closesthit sets
                        // payload.primaryTransmission = 1.0, so Vulkan's raygen scales
                        // its aerial/fog distance by (1 - 1) = 0 and applies NO aerial
                        // perspective to the water. OptiX must do the same or the post-
                        // process hazes the water (more so at grazing side views with a
                        // large entry distance) and it reads THINNER than Vulkan's
                        // filled body. aerial_dist *= (1 - first_hit_transmission) = 0 →
                        // water stays fully visible at every angle, matching Vulkan.
                        first_hit_transmission = fmaxf(first_hit_transmission, 1.0f);
                    }
                    break;  // stop processing further volumes this bounce
                }

                throughput *= vol_transmittance;

                // NEW: Update depth for fogging calculation
                // If this is the primary ray and we hit substantive volume, pull the fog distance forward.
                if (bounce == 0) {
                    if (vol_transmittance < 0.999f) {
                        if (vol_depth < 0.0f || entry_distances[i] < vol_depth) {
                            vol_depth = entry_distances[i];
                        }
                    }
                    vol_trans_accum *= vol_transmittance;
                }

                // Early termination if fully absorbed
                if (throughput.x < 0.001f && throughput.y < 0.001f && throughput.z < 0.001f) {
                    return color;
                }
            }

            // Continue the path along the refracted ray: skip the geometry
            // shading + remaining volume passes for this bounce and re-trace
            // from the bent ray next iteration. This is what makes the
            // background visibly refract through the water on OptiX.
            if (fluid_redirect) {
                if (throughput.x < 0.001f && throughput.y < 0.001f && throughput.z < 0.001f) {
                    return color;
                }
                ray.origin = fluid_new_origin;
                ray.direction = fluid_new_dir;
                continue;
            }
        }
        // ═══════════════════════════════════════════════════════════
        // VOLUMETRIC GOD RAYS - Only on primary ray for performance
        // God rays are accumulated to the point of first hit or infinity
        // ═══════════════════════════════════════════════════════════
        if (bounce == 0 && optixLaunchParams.world.nishita.godrays_enabled) {
            float maxDist = payload.hit ? payload.t : 200000.0f;
            
            // --- STOCHASTIC DEPTH PROBE (Film Quality Occlusion) ---
            // Probe for near volumetric occluders, but avoid overly-aggressive
            // clamps caused by tiny dithered probes (which produce only a
            // local halo). Require a sensible minimum distance before clamping.
            float t_vol_min = get_stochastic_volumetric_occlusion(
                ray.origin, ray.direction, maxDist, rng,
                0.0f
            );

            // If probe returned a very small distance (noise/dither), ignore it.
            const float kMinProbeDistance = 5.0f; // meters — prevent tiny clamps
            if (t_vol_min > kMinProbeDistance && t_vol_min < maxDist * 0.98f) {
                maxDist = fminf(maxDist, t_vol_min);
            }

            float3 godRayContribution = calculate_volumetric_god_rays(
                optixLaunchParams.world,
                ray.origin,
                normalize(ray.direction),
                maxDist,
                rng
            );
            color += godRayContribution;
        }
        
        // ═══════════════════════════════════════════════════════════
        // VDB VOLUME RAY MARCHING (Independent volumes)
        // Each VDB is rendered as a participating medium
        // ═══════════════════════════════════════════════════════════
     



        // ═══════════════════════════════════════════════════════════
        // GAS VOLUME RAY MARCHING (Texture3D)
        // ═══════════════════════════════════════════════════════════
        if (optixLaunchParams.gas_volumes && optixLaunchParams.gas_volume_count > 0) {
            float3 sun_dir = normalize(optixLaunchParams.world.nishita.sun_direction);
            float sun_intensity = optixLaunchParams.world.nishita.sun_intensity;
             
            for (int i = 0; i < optixLaunchParams.gas_volume_count; ++i) {
                const GpuGasVolume& vol = optixLaunchParams.gas_volumes[i];
                if(!vol.density_texture) continue;

                // Determine max dist
                float max_dist = payload.hit ? payload.t : 1e16f;
                if (bounce == 0 && max_dist > optixLaunchParams.clip_far) max_dist = optixLaunchParams.clip_far;
                
                float vol_transmittance = 1.0f;
                float3 vol_color = raymarch_gas_volume(
                    vol,
                    ray.origin,
                    normalize(ray.direction),
                    sun_dir,
                    sun_intensity,
                    vol_transmittance,
                    max_dist,
                    rng
                );
                
                color += throughput * vol_color;
                throughput *= vol_transmittance;

                // NEW: Update depth for fogging calculation
                if (bounce == 0) {
                    float3 local_origin = transform_point_affine(ray.origin, vol.inv_transform);
                    float3 local_dir = transform_vector_affine(normalize(ray.direction), vol.inv_transform);
                    float dir_len = length(local_dir);
                    if (dir_len > 1e-8f) {
                        local_dir /= dir_len;
                        float t0, t1;
                        if (intersect_aabb_vdb(local_origin, local_dir, vol.local_bbox_min, vol.local_bbox_max, t0, t1)) {
                            float current_t = fmaxf(t0 / dir_len, 0.0f);
                            if (vol_transmittance < 0.999f) {
                                if (vol_depth < 0.0f || current_t < vol_depth) {
                                    vol_depth = current_t;
                                }
                            }
                        }
                    }
                    vol_trans_accum *= vol_transmittance;
                }

                if (throughput.x < 0.001f && throughput.y < 0.001f && throughput.z < 0.001f) {
                     return color;
                }
            }
        }

        if (!payload.hit) {
            // --- Arka plan rengi ---
            float3 bg_color = evaluate_background(
                optixLaunchParams.world,
                ray.origin,
                ray.direction,
                rng,
                path_touched_transmissive
            );

            // --- Infinite Grid Logic (GPU) ---
            if (optixLaunchParams.grid_enabled && optixLaunchParams.is_final_render == 0 && ray.direction.y < -0.0001f) {
                 float t = -ray.origin.y / ray.direction.y;
                 if (t > 0.0f) {
                     float3 p = ray.origin + ray.direction * t;
                     float scale_major = 1.0f;
                     float line_width = 0.02f;
                     float x_mod = fabsf(fmodf(p.x, scale_major));
                     float z_mod = fabsf(fmodf(p.z, scale_major));
                     bool x_line = x_mod < line_width || x_mod > (scale_major - line_width);
                     bool z_line = z_mod < line_width || z_mod > (scale_major - line_width);
                     bool x_axis = fabsf(p.z) < line_width * 2.0f;
                     bool z_axis = fabsf(p.x) < line_width * 2.0f;
                     float3 grid_color = make_float3(0.0f, 0.0f, 0.0f);
                     bool hit_grid = false;
                     if (x_axis) { grid_color = make_float3(0.8f, 0.2f, 0.2f); hit_grid = true; }
                     else if (z_axis) { grid_color = make_float3(0.2f, 0.8f, 0.2f); hit_grid = true; }
                     else if (x_line || z_line) { grid_color = make_float3(0.4f, 0.4f, 0.4f); hit_grid = true; }
                     if (hit_grid) {
                         float dist = t;
                         float alpha = 1.0f - fminf(fmaxf((dist - 5.0f) / (optixLaunchParams.grid_fade_distance - 5.0f), 0.0f), 1.0f);
                         bg_color = bg_color * (1.0f - alpha) + grid_color * alpha;
                     }
                 }
            }
            color += soft_clip_fireflies(throughput * bg_color, 64.0f);
            break;
        }

        float3 wo = -normalize(ray.direction);
        Ray scattered;
        float3 attenuation;
        float pdf;
        bool is_specular;
        int bounce_type = PATH_BOUNCE_SPECULAR;
        
        GpuMaterial mat = optixLaunchParams.materials[payload.material_id];

        // --- GPU Terrain Blending Override ---
        if (payload.use_blended_data) {
             if (payload.water_surface_active) {
                 // CPU parity: water has TWO color channels.
                 //   - direct lighting BRDF albedo  = water_color (depth+foam blended)
                 //     evaluate_brdf reads payload.blended_albedo directly, so NEE
                 //     already gets the right tint without overriding mat.albedo here.
                 //   - transmission scatter tint   = constant deep_color (mat.albedo)
                 //     CPU PrincipledBSDF::scatter passes albedoProperty.color (deep_color)
                 //     to Dielectric, which is the unblended water material albedo.
                 // Keep mat.albedo at its original deep_color so transmission_scatter
                 // applies CPU-parity Beer-Lambert tint instead of the depth-blended
                 // water_color (which collapses to near-black through Beer's law).
                 mat.roughness = payload.blended_roughness;
                 mat.transmission = payload.blended_transmission;
                 mat.emission = payload.blended_emission;
                 mat.ior = payload.blended_ior;
             } else {
                 mat.albedo = payload.blended_albedo;
                 mat.roughness = payload.blended_roughness;
                 mat.metallic = payload.blended_metallic;
                 mat.clearcoat = payload.blended_clearcoat;
                 mat.clearcoat_roughness = payload.blended_clearcoat_roughness;
                 mat.subsurface = payload.blended_subsurface;
                 mat.subsurface_color = payload.blended_subsurface_color;
                 mat.transmission = payload.blended_transmission;
                 mat.translucent = payload.blended_translucent;
                 mat.emission = payload.blended_emission;
                 mat.ior = payload.blended_ior;
             }
        }
        if (bounce == 0) {
            first_hit_transmission = resolve_surface_transmission(mat, payload);
            if (first_hit_transmission < 0.5f) {
                if (primary_albedo_out) *primary_albedo_out = payload.primary_albedo;
                if (primary_normal_out) *primary_normal_out = payload.primary_normal;
                if (primary_hit_out) *primary_hit_out = payload.primary_hit;
                // Stylize AOV: world hit position + real material id (for outline boundaries)
                if (primary_world_pos_out) *primary_world_pos_out = ray.origin + ray.direction * payload.t;
                if (primary_material_id_out) *primary_material_id_out = payload.material_id;
            } else {
                if (primary_albedo_out) *primary_albedo_out = make_float3(0.0f);
                if (primary_normal_out) *primary_normal_out = make_float3(0.0f);
                if (primary_hit_out) *primary_hit_out = 0;
                if (primary_world_pos_out) *primary_world_pos_out = make_float3(0.0f);
                if (primary_material_id_out) *primary_material_id_out = -1;
            }
        }

        // --- Emission hesabı: scatter'dan ÖNCE (Vulkan parity) ---
        // Vulkan closesthit: payload.radiance = emColor * emStrength (scatter kararından önce)
        // Vulkan raygen: radiance += throughput * payload.radiance (scatter kontrolünden önce)
        // Böylece scatter başarısız olsa bile emissive yüzeyler ışık katkısı yapar.
        float3 emission = mat.emission;
        if (payload.has_emission_tex) {
            float4 tex = tex2D<float4>(payload.emission_tex, payload.uv.x, payload.uv.y);
            // Emission texture is authoritative for color (matching Vulkan behavior).
            // mat.emission is pre-multiplied (emColor * emStrength), so extract
            // approximate strength via max component to avoid double color tint.
            float em_strength = fmaxf(mat.emission.x, fmaxf(mat.emission.y, mat.emission.z));
            emission = make_float3(tex.x, tex.y, tex.z) * em_strength;
        }

        // ----------------------------------------------------------
        // RESIN (Vulkan parity): a refractive ABSORBING coat over an OPAQUE base.
        // Fresnel-split — the reflect lobe is the glossy resin top (specular, skips
        // NEE); the base lobe tints the LOCAL albedo by the coat absorption (round
        // trip) and shades as a plain diffuse surface, so the base gets full direct
        // (NEE) + indirect light. The NEE direct contribution is additionally
        // absorbed over its own light-angle entry path (resin_* handed to
        // calculate_light_contribution below). Keeps OptiX == Vulkan.
        // ----------------------------------------------------------
        bool   resin_active  = false;
        float  resin_density = 0.0f;
        float3 resin_ext     = make_float3(0.0f, 0.0f, 0.0f);
        if (mat.transmission_density > 1e-4f) {
            float3 N    = payload.normal;
            float3 unit = normalize(ray.direction);
            float3 V    = -unit;
            float  effIor = fmaxf(mat.ior, 1.45f);
            float  cosT = clamp(dot(V, N), 0.0f, 1.0f);
            float  fres = schlick(cosT, effIor);
            if (random_float(rng) < fres) {
                // Glossy resin top — specular reflection, no NEE/diffuse.
                float3 refl;
                if (mat.roughness < 0.02f) {
                    refl = reflect(unit, N);
                } else {
                    float alpha = fmaxf(mat.roughness * mat.roughness, 1e-4f);
                    refl = ggxSampleVNDF(N, V, alpha, random_float(rng), random_float(rng));
                    if (dot(refl, N) <= 0.0f) refl = reflect(unit, N);
                }
                // Emissive resin still emits (usually 0); white specular throughput.
                color += soft_clip_fireflies(throughput * emission, 64.0f);
                if (++resin_depth > RESIN_MAX_DEPTH) break;  // bound resin cost
                ray = Ray(offset_ray(payload.position, N), normalize(refl));
                continue;   // skip scatter_material + NEE this bounce
            }
            // Base under the resin: absorb over the thickness (round trip) into albedo,
            // then shade as opaque diffuse. A small base extinction (0.25) darkens with
            // Resin Depth even for white resin; resin_color tints which channels survive.
            float3 ct      = clamp3f(mat.resin_color, 0.0f, 1.0f);
            float  cosV    = fmaxf(fabsf(cosT), 0.25f);
            float  pathLen = 2.0f * mat.transmission_density / cosV;
            resin_ext = make_float3((1.0f - ct.x) + 0.25f,
                                    (1.0f - ct.y) + 0.25f,
                                    (1.0f - ct.z) + 0.25f);
            mat.albedo = mat.albedo * make_float3(expf(-pathLen * resin_ext.x),
                                                  expf(-pathLen * resin_ext.y),
                                                  expf(-pathLen * resin_ext.z));
            resin_density            = mat.transmission_density;
            mat.roughness            = 1.0f;
            mat.metallic             = 0.0f;
            mat.transmission         = 0.0f;
            mat.transmission_density = 0.0f;  // now a plain diffuse base for scatter + NEE
            resin_active             = true;
        }

        // --- Scatter başarısızsa: emission'ı ekle ve çık ---
        if (!scatter_material(mat, payload, ray, rng, &scattered, &attenuation, &pdf, &is_specular, &bounce_type)) {
            // Vulkan parity: emissive-only yüzeyler hala ışık yayar.
            // raygen.rgen payload.radiance'ı payload.scattered kontrolünden ÖNCE toplar.
            color += soft_clip_fireflies(throughput * emission, 64.0f);
            break;
        }

        // Save pre-attenuation throughput for NEE/emission accumulation.
        // Emission is self-emitted light (not reflected), so it must NOT be
        // attenuated by the surface BRDF.  Direct lighting already includes
        // evaluate_brdf() internally, same reasoning.  This matches the
        // Vulkan path where payload.radiance (emission + direct) is
        // accumulated with the original throughput before scatter attenuation.
        float3 throughput_for_nee = throughput;

        throughput *= attenuation;
        throughput = soft_clip_fireflies(throughput, 128.0f);
        float surface_transmission = resolve_surface_transmission(mat, payload);
        float surface_opacity = resolve_surface_opacity(mat, payload);
        bool alpha_passthrough = surface_opacity < 0.999f
            && fabsf(dot(normalize(scattered.direction), normalize(ray.direction))) > 0.999f
            && attenuation.x > 0.999f && attenuation.y > 0.999f && attenuation.z > 0.999f;
        if (is_specular && (surface_transmission > 0.01f || alpha_passthrough)) {
            path_touched_transmissive = true;
        }

        // --- GPU VOLUMETRIC RENDERING ---
        // Check if this is a volumetric material using the proper flag
        if (payload.is_volumetric) {
            // Use the new raymarch_volumetric_object function with multi-scattering
            float vol_transmittance = 1.0f;
            float3 vol_color = raymarch_volumetric_object(
                ray.origin,
                ray.direction,
                payload.aabb_min,
                payload.aabb_max,
                payload.vol_density,
                payload.vol_absorption,
                payload.vol_scattering,
                payload.vol_albedo,
                payload.vol_emission,
                payload.vol_g,
                payload.vol_step_size,
                payload.vol_max_steps,
                payload.vol_noise_scale,
                // Multi-scattering parameters
                payload.vol_multi_scatter,
                payload.vol_g_back,
                payload.vol_lobe_mix,
                payload.vol_light_steps,
                payload.vol_shadow_strength,
                // NanoVDB parameters
                payload.nanovdb_grid,
                payload.has_nanovdb,
                vol_transmittance,
                rng
            );
            
            // Accumulate volumetric contribution
            color += throughput * vol_color;
            
            // Apply transmittance to throughput
            throughput *= make_float3(vol_transmittance, vol_transmittance, vol_transmittance);
            
            // If fully absorbed, stop
            if (vol_transmittance < 0.01f) {
                break;
            }
            
            // Continue ray beyond the volume
            // Find exit point and continue
            Ray exit_ray(payload.position + ray.direction * SCENE_EPSILON, ray.direction);
            OptixHitResult exit_payload = {};
            trace_ray(exit_ray, &exit_payload, SCENE_EPSILON, 1e16f);
            
            if (exit_payload.hit) {
                scattered = Ray(exit_payload.position + ray.direction * 0.01f, ray.direction);
            } else {
                // Ray exited scene through volume
                float3 bg_color = evaluate_background(
                    optixLaunchParams.world,
                    ray.origin,
                    ray.direction,
                    rng,
                    path_touched_transmissive
                );
                float bg_factor = (bounce == 0) ? 1.0f : fmaxf(0.1f, 1.0f / (1.0f + bounce * 0.5f));
                color += soft_clip_fireflies(throughput * bg_color * bg_factor, 64.0f);
                break;
            }
            
            ray = scattered;
            continue; // Skip surface shading
        }
        
        // Clamp extreme future-bounce throughput before expensive continuation work.
        float max_throughput = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        if (max_throughput > MAX_CONTRIBUTION) {
            throughput *= (MAX_CONTRIBUTION / max_throughput);
        }

        // emission zaten scatter'dan önce hesaplandı (yukarıda)
        // --- Eğer hiç ışık yoksa sadece emissive katkı yap ---
        if (light_count == 0) {
            // Use pre-attenuation throughput: emission is self-emitted, not BRDF-filtered
            color += soft_clip_fireflies(throughput_for_nee * emission, 64.0f);
            if (resin_active) {
                if (++resin_depth > RESIN_MAX_DEPTH) break;
            } else {
                if (bounce_type == PATH_BOUNCE_DIFFUSE && ++diffuse_depth > max_diffuse_depth) break;
                if (bounce_type == PATH_BOUNCE_TRANSMISSION && ++transmission_depth > max_transmission_depth) break;
            }
            ray = scattered;
            continue;
        }

        float pdf_select = 1.0f;
        light_index = pick_smart_light(payload.position, rng, &pdf_select);
       
        // --- Direkt ışık katkısı ---
        float3 direct = make_float3(0.0f, 0.0f, 0.0f);
        if (!is_specular && light_index >= 0 && pdf_select > 1e-6f) {
            direct = calculate_light_contribution(
                optixLaunchParams.lights[light_index], mat, payload, wo, rng, pdf_select,
                resin_active, resin_density, resin_ext
            );
            // Vulkan parity: pdf_select ile böl (unbiased Monte Carlo estimator)
            // Işık daha az seçiliyorsa katkısı orantılı artırılmalı
            direct /= fmaxf(pdf_select, 1e-4f);
            // Firefly kontrolü - aşırı parlak direkt katkıları sınırla
            float direct_lum = luminance(direct);
            if (direct_lum > MAX_CONTRIBUTION) {
                direct *= (MAX_CONTRIBUTION / direct_lum);
            }
        }

        // BRDF-side MIS for area/spot lights removed: the previous implementation
        // skipped the geometric ray-light intersection check and treated every BRDF
        // sample as if it hit the light's center. On low-roughness surfaces the GGX
        // peak (huge f) combined with mis_weight≈1 produced energy explosions.
        // Vulkan and CPU only do light-side (NEE) MIS for area/spot — match that.
        float3 brdf_mis = make_float3(0.0f, 0.0f, 0.0f);
      
        // --- Toplam katkı ---
        float3 total_contribution = direct + brdf_mis + emission;
        
        // Son firefly kontrolü
        float total_lum = luminance(total_contribution);
        if (total_lum > MAX_CONTRIBUTION * 2.0f) {
            total_contribution *= (MAX_CONTRIBUTION * 2.0f / total_lum);
        }
        
        // Use pre-attenuation throughput: emission is self-emitted light
        // and direct/brdf_mis already contain evaluate_brdf() internally.
        // Multiplying by post-attenuation throughput would double-apply the
        // surface BRDF, causing oversaturated albedo-tinted emission.
        // This matches the Vulkan raygen loop where radiance is accumulated
        // before throughput *= payload.attenuation.
        color += soft_clip_fireflies(throughput_for_nee * total_contribution, 64.0f);

        // Russian roulette must happen after the current vertex contribution.
        // Otherwise surviving paths stay unbiased, but killed paths lose the
        // direct/emissive light that belongs to this hit. That increases noise
        // and biases OptiX against Vulkan, whose closest-hit radiance is added
        // before its raygen loop applies RR to the continuation throughput.
        if (bounce > 2) {
            float p = fminf(fmaxf(fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)), 0.05f), 1.0f);
            if (random_float(rng) > p) break;
            throughput /= p;
        }

        if (resin_active) {
            if (++resin_depth > RESIN_MAX_DEPTH) break;
        } else {
            if (bounce_type == PATH_BOUNCE_DIFFUSE && ++diffuse_depth > max_diffuse_depth) break;
            if (bounce_type == PATH_BOUNCE_TRANSMISSION && ++transmission_depth > max_transmission_depth) break;
        }

        ray = scattered;
    }

    // ═══════════════════════════════════════════════════════════
    // POST-PROCESS: Aerial Perspective (Photo-realistic atmosphere)
    // ═══════════════════════════════════════════════════════════
    const WorldData& world = optixLaunchParams.world;
    if (world.mode == 2) {
        // Balance Aerial Perspective: Background already has atmosphere logic.
        // If we apply full distance fog (100km+) to background, sun and sky get washed out.
        // Solution: If it's a miss (dist > 1M), we use a shorter effective distance for 'atmospheric feel'
        // without killing the sun disk.
        // --- WEIGHTED FOG DISTANCE ---
        // Logic: if we have volumes, blend the fog distance based on cloud opacity.
        // This ensures thin cloud edges use background haze, while opaque centers use cloud-depth haze.
        float aerial_dist = first_hit_t;
        if (vol_depth > 0.0f) {
            float background_t = (first_hit_t > 0.0f) ? first_hit_t : 10000.0f;
            float weight = fminf(fmaxf(1.0f - vol_trans_accum, 0.0f), 1.0f);
            aerial_dist = background_t * (1.0f - weight) + vol_depth * weight;
        }
        else if (aerial_dist <= 0.0f) {
            aerial_dist = 10000.0f;
        }
        aerial_dist *= (1.0f - first_hit_transmission);
        // Prepare ray origin/dir for aerial perspective sampling (use the first ray)
        float3 rayOrigin = first_ray_origin;
        float3 rayDir = normalize(first_ray_dir);
        color = gpu_get_aerial_perspective(world, color, rayOrigin, rayDir, aerial_dist);
    }
    else if (world.nishita.fog_enabled && world.nishita.fog_density > 0.0f) {
        // Fallback to simple height fog for non-Nishita modes
        float fogDistance = (first_hit_t > 0.0f) ? first_hit_t : world.nishita.fog_distance * 0.8f;
        fogDistance *= (1.0f - first_hit_transmission);
        float3 rayOrigin = first_ray_origin;
        float3 rayDir = normalize(first_ray_dir);
        
        float fogFactor = calculate_height_fog_factor(
            rayOrigin, rayDir, fogDistance,
            world.nishita.fog_density,
            world.nishita.fog_height,
            world.nishita.fog_falloff
        );
        
        float3 fogColor = world.nishita.fog_color;
        float3 sunDir = normalize(world.nishita.sun_direction);
        float sunDot = fmaxf(0.0f, dot(rayDir, sunDir));
        float sunScatter = powf(sunDot, 8.0f) * world.nishita.fog_sun_scatter;
        float3 sunColor = make_float3(1.0f, 0.9f, 0.7f) * world.nishita.sun_intensity * 0.05f;
        fogColor = fogColor + sunColor * sunScatter;
        color = lerp(color, fogColor, fogFactor);
    }

    if (rc_weather_active(world.weather)) {
        float weatherDistance = (first_hit_t > 0.0f) ? first_hit_t : 12000.0f;
        if (vol_depth > 0.0f) weatherDistance = fminf(weatherDistance, vol_depth);
        weatherDistance *= (1.0f - first_hit_transmission);
        color = rc_apply_weather_atmosphere(
            world.weather,
            color,
            normalize(first_ray_dir),
            weatherDistance);
        color = rc_apply_weather_precipitation_overlay(
            world.weather,
            color,
            normalize(first_ray_dir),
            weatherDistance,
            optixLaunchParams.water_time);
    }

    // Final clamp - NaN ve Inf kontrolü
    color.x = isfinite(color.x) ? fminf(fmaxf(color.x, 0.0f), 100.0f) : 0.0f;
    color.y = isfinite(color.y) ? fminf(fmaxf(color.y, 0.0f), 100.0f) : 0.0f;
    color.z = isfinite(color.z) ? fminf(fmaxf(color.z, 0.0f), 100.0f) : 0.0f;

    return color;
}
