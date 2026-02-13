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


// Helper function to render a single cloud layer
// Returns ONLY the cloud color contribution (not blended with background)
// Modifies transmittance based on cloud density encountered
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

__device__ inline float smoothstep_cloud(float edge0, float edge1, float x) {
    float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0f - 2.0f * t);
}

// Forward declaration for ambient sky radiance
__device__ float3 gpu_get_sky_radiance(const WorldData& world, const float3& dir);
__device__ float3 gpu_get_ambient_radiance_volume(const WorldData& world, const float3& dir);

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

__device__ float3 render_cloud_layer(
    const WorldData& world, 
    const float3& rayDir, 
    float3 bg_color,  // Used for ambient calculation only
    float cloudMinY, float cloudMaxY,
    float scale, float coverage, float densityMult, float detail,
    float& transmittance  // In/out parameter
) {
    // Use actual camera Y position
    float camY = (world.camera_y != 0.0f) ? world.camera_y : world.nishita.altitude;
    float3 cloudCamPos = make_float3(0.0f, camY, 0.0f);
    
    // Ray-Plane Intersection
    float t_enter, t_exit;
    
    // No color contribution for all these early-exit cases
    float3 noCloud = make_float3(0.0f, 0.0f, 0.0f);
    
    if (cloudCamPos.y < cloudMinY) {
        if (rayDir.y <= 0.0f) return noCloud;  // Looking down, can't see clouds above
        t_enter = (cloudMinY - cloudCamPos.y) / rayDir.y;
        t_exit = (cloudMaxY - cloudCamPos.y) / rayDir.y;
    }
    else if (cloudCamPos.y > cloudMaxY) {
        if (rayDir.y >= 0.0f) return noCloud;  // Looking up, can't see clouds below
        t_enter = (cloudMaxY - cloudCamPos.y) / rayDir.y;
        t_exit = (cloudMinY - cloudCamPos.y) / rayDir.y;
    }
    else {
        t_enter = 0.0f;
        if (rayDir.y > 0.001f) {
            t_exit = (cloudMaxY - cloudCamPos.y) / rayDir.y;
        } else if (rayDir.y < -0.001f) {
            t_exit = (cloudMinY - cloudCamPos.y) / rayDir.y;
        } else {
            t_exit = 500000.0f;
        }
    }
    
    if (t_exit <= 0.0f || t_exit <= t_enter) return noCloud;  // No valid intersection
    
    // Performance Optimization: Distance Culling (Frustum/Range Limit)
    // Extended to 500km to support realistic horizon and high-altitude views
    const float MAX_CLOUD_DIST = 500000.0f;
    if (t_enter > MAX_CLOUD_DIST) return noCloud;
    t_exit = fminf(t_exit, MAX_CLOUD_DIST);
    
    if (t_enter < 0.0f) t_enter = 0.0f;
    
    // Horizon fade - MORE GENEROUS (approx 0.5 degree)
    // Prevents hard cutoff at horizon but allows clouds to be seen lower
    float h_val = rayDir.y / 0.008f;
    float h_t = fmaxf(0.0f, fminf(1.0f, fabsf(h_val)));
    float horizonFade = h_t * h_t * (3.0f - 2.0f * h_t);
    
    // Additional fade out at max distance (starts 100km before edge)
    float distFade = 1.0f - fmaxf(0.0f, (t_enter - (MAX_CLOUD_DIST - 100000.0f)) / 100000.0f);
    horizonFade *= distFade;

    // Quality-based step count - BUFFED 3X (User controllable base)
    float quality = fmaxf(0.1f, fminf(3.0f, world.nishita.cloud_quality));
    int baseSteps = (int)((float)world.nishita.cloud_base_steps * quality); 
    int numSteps = baseSteps + (int)((float)baseSteps * (1.0f - h_t));
    
    float stepSize = (t_exit - t_enter) / (float)numSteps;
    float3 cloudColor = make_float3(0.0f, 0.0f, 0.0f);
    float t = t_enter;
    
    float localDensityMult = densityMult * horizonFade;
    
    float3 ambientSky = bg_color * 0.3f;
    float3 sunDirection = normalize(world.nishita.sun_direction);
    
    // Cloud Anisotropy (G-Factors)
    float cloudG = fmaxf(0.0f, fminf(0.99f, world.nishita.cloud_anisotropy));
    float cloudG_back = fmaxf(-0.99f, fminf(0.0f, world.nishita.cloud_anisotropy_back));
    float lobeMix = fmaxf(0.0f, fminf(1.0f, world.nishita.cloud_lobe_mix));
    
    // Emissive
    float3 cloudEmissive = world.nishita.cloud_emissive_color * world.nishita.cloud_emissive_intensity;
    
    for (int i = 0; i < numSteps; ++i) {
        float jitterSeed = (float)i + (rayDir.x * 53.0f + rayDir.z * 91.0f) * 10.0f;
        float3 pos = cloudCamPos + rayDir * (t + stepSize * hash(jitterSeed));
        
        float heightFraction = (pos.y - cloudMinY) / (cloudMaxY - cloudMinY);
        
        // CUMULUS PROFILE (Strict Flat Bottom)
        float heightGradient = smoothstep_cloud(0.0f, 0.05f, heightFraction) * smoothstep_cloud(1.0f, 0.3f, heightFraction);
        if (heightFraction < 0.2f) {
           heightGradient = fmaxf(heightGradient, smoothstep_cloud(0.0f, 0.02f, heightFraction));
        }
        
        float3 offsetPos = pos + make_float3(world.nishita.cloud_offset_x, 0.0f, world.nishita.cloud_offset_z);
        float3 noisePos = offsetPos * scale;
        float effectiveCoverage = coverage;
        
        // FFT Based Coverage Modulation
        if (world.nishita.cloud_use_fft && world.nishita.cloud_fft_map) {
            float uvScale = 0.002f; 
            float u = offsetPos.x * uvScale;
            float v = offsetPos.z * uvScale;
            u = u - floorf(u);
            v = v - floorf(v);
            float4 fftData = tex2D<float4>(world.nishita.cloud_fft_map, u, v);
            effectiveCoverage = fmaxf(0.0f, fminf(1.0f, effectiveCoverage + fftData.x * 0.15f));
        }

        float rawDensity = cloud_shape(noisePos, effectiveCoverage, detail);
        
        float density = rawDensity * heightGradient;
        
        if (density > 0.003f) {
            density *= localDensityMult;
            
            // ═══════════════════════════════════════════════════════════
            // LIGHT MARCHING (Self-Shadowing) - Controllable via UI
            // ═══════════════════════════════════════════════════════════
            float lightTransmittance = 1.0f;
            int lightSteps = world.nishita.cloud_light_steps;  // UI controlled
            
            if (lightSteps > 0 && sunDirection.y > -0.1f) {
                // Adaptive step size based on density
                float lightStepSize = (cloudMaxY - pos.y) / fmaxf(0.1f, fabsf(sunDirection.y)) / (float)lightSteps;
                lightStepSize = fminf(lightStepSize, 200.0f); // Clamp max step
                
                // ═══════════════════════════════════════════════════════════
                // DUAL SCATTERING LIGHT MARCHING
                // ═══════════════════════════════════════════════════════════
                float lightDensitySum = 0.0f;

                for (int j = 1; j <= lightSteps; ++j) {
                    float3 lightPos = pos + sunDirection * (lightStepSize * (float)j);
                    
                    if (lightPos.y > cloudMaxY || lightPos.y < cloudMinY) break;
                    
                    float3 lightOffsetPos = lightPos + make_float3(world.nishita.cloud_offset_x, 0.0f, world.nishita.cloud_offset_z);
                    
                    float3 lightNoisePos = lightOffsetPos * scale;
                    // Use fast_cloud_shape for performance in shadow rays
                    float lightDensity = fast_cloud_shape(lightNoisePos, coverage);

                    float lh = (lightPos.y - cloudMinY) / (cloudMaxY - cloudMinY);
                    // Match the profile used in main loop for consistency (Sharp Cumulus)
                    float lightHeightGrad = smoothstep_cloud(0.0f, 0.05f, lh) * smoothstep_cloud(1.0f, 0.3f, lh);
                    if (lh < 0.2f) {
                       lightHeightGrad = fmaxf(lightHeightGrad, smoothstep_cloud(0.0f, 0.02f, lh));
                    }
                    
                    lightDensity *= lightHeightGrad * localDensityMult;
                    
                    lightDensitySum += lightDensity * lightStepSize;
                    
                    if (lightDensitySum > 10.0f) break; 
                }

                // Beer's Law (Primary Absorption)
                float beersLaw = expf(-lightDensitySum * 0.02f * world.nishita.cloud_absorption);
                
                // Secondary Absorption (Softer, simulates scattered light) - Dual Scattering
                float beersLaw2 = expf(-lightDensitySum * 0.02f * world.nishita.cloud_absorption * 0.2f);
                
                // Combine Terms with Silver intensity
                lightTransmittance = beersLaw * 0.3f + beersLaw2 * 0.7f * world.nishita.cloud_silver_intensity;
                
                // Shadow stength from UI
                lightTransmittance = lerp(1.0f, lightTransmittance, world.nishita.cloud_shadow_strength);
            }
            
            
            // ═══════════════════════════════════════════════════════════
            // ADVANCED COLOR CALCULATION
            // ═══════════════════════════════════════════════════════════
            float cosTheta = dot(rayDir, sunDirection);
            
            // Henyey-Greenstein phase function (Dual-Lobe for Back Scattering)
            // Lobe 1: Forward scattering (strong peak around sun)
            float phase1 = (1.0f - cloudG * cloudG) / (4.0f * 3.14159f * powf(1.0f + cloudG * cloudG - 2.0f * cloudG * cosTheta, 1.5f));
            
            // Lobe 2: Backward scattering (softer peak opposite to sun)
            float phase2 = (1.0f - cloudG_back * cloudG_back) / (4.0f * 3.14159f * powf(1.0f + cloudG_back * cloudG_back - 2.0f * cloudG_back * cosTheta, 1.5f));
            
            // Mix lobes
            float phase = lerp(phase2, phase1, lobeMix);
            
            // Powder effect
            float powder = powderEffect(density, cosTheta);
            
            // ═══════════════════════════════════════════════════════════
            // SUN COLOR - IMPROVED GRADIENT (Sunset/Sunrise)
            // ═══════════════════════════════════════════════════════════
            float sunElevation = sunDirection.y;
            
            // Layered color transitions
            float3 sunColor;
            if (sunElevation > 0.5f) {
                // High sun - warm white
                sunColor = make_float3(1.0f, 0.98f, 0.95f);
            } else if (sunElevation > 0.2f) {
                // Golden hour
                float t = (sunElevation - 0.2f) / 0.3f;
                float3 goldenColor = make_float3(1.0f, 0.85f, 0.6f);
                float3 whiteColor = make_float3(1.0f, 0.98f, 0.95f);
                sunColor = goldenColor * (1.0f - t) + whiteColor * t;
            } else if (sunElevation > 0.0f) {
                // Sunset/sunrise - orange to red
                float t = sunElevation / 0.2f;
                float3 orangeColor = make_float3(1.0f, 0.6f, 0.3f);
                float3 goldenColor = make_float3(1.0f, 0.85f, 0.6f);
                sunColor = orangeColor * (1.0f - t) + goldenColor * t;
            } else {
                // Below horizon - deep orange/red
                float t = fmaxf(0.0f, 1.0f + sunElevation * 5.0f);
                float3 redColor = make_float3(0.8f, 0.3f, 0.1f);
                float3 orangeColor = make_float3(1.0f, 0.6f, 0.3f);
                sunColor = redColor * (1.0f - t) + orangeColor * t;
            }
            
            // ═══════════════════════════════════════════════════════════
            // DIRECT LIGHTING (with self-shadowing)
            // ═══════════════════════════════════════════════════════════
            float directIntensity = world.nishita.sun_intensity * phase * lightTransmittance * 5.0f;
            float3 directLight = sunColor * directIntensity;
            
            // Silver lining (UI controlled intensity)
            float silverBase = fmaxf(0.0f, cosTheta) * powder * lightTransmittance;
            float silverLining = silverBase * 4.0f * world.nishita.cloud_silver_intensity;
            directLight += sunColor * silverLining * world.nishita.sun_intensity;
            
            // ═══════════════════════════════════════════════════════════
            // AMBIENT / MULTI-SCATTERING (UI controlled)
            // ═══════════════════════════════════════════════════════════
            float multiScatter = 0.25f * (1.0f - expf(-density * 4.0f));
            
            // Shadow color gradient - more blue in shadows
            float shadowAmount = 1.0f - lightTransmittance;
            float3 shadowColor = make_float3(0.12f, 0.18f, 0.35f);  // Deep blue shadow
            
            // Height-based ambient (upper parts brighter)
            float heightFactor = (pos.y - cloudMinY) / (cloudMaxY - cloudMinY);
            float ambientBoost = 1.0f + heightFactor * 0.3f;
            
            float3 ambient = ambientSky * 0.35f * world.nishita.cloud_ambient_strength * ambientBoost;
            ambient += shadowColor * world.nishita.sun_intensity * 0.12f * shadowAmount;
            ambient += sunColor * multiScatter * world.nishita.sun_intensity * 0.4f;
            
            // ═══════════════════════════════════════════════════════════
            // FINAL COLOR - Energy conserving
            // ═══════════════════════════════════════════════════════════
            float3 lightColor = directLight + ambient + cloudEmissive;
            
            float3 stepColor = lightColor * density;
            // INCREASED ABSORPTION MULTIPLIER (from 0.012 to 0.08) for better sun occlusion
            float absorption = density * stepSize * 0.08f * world.nishita.cloud_absorption;
            float stepTransmittance = expf(-absorption);
            
            cloudColor += stepColor * transmittance * (1.0f - stepTransmittance);
            transmittance *= stepTransmittance;
            
            if (transmittance < 0.01f) break;
        }
        t += stepSize;
    }
    
    return cloudColor;
}

// Main function to render all cloud layers
__device__ float3 render_clouds(const WorldData& world, const float3& rayDir, float3 bg_color) {
    if (!world.nishita.clouds_enabled && !world.nishita.cloud_layer2_enabled) {
        return bg_color;
    }

    float transmittance = 1.0f;
    float3 cloudColor = make_float3(0.0f, 0.0f, 0.0f);
    
    // === LAYER 1 (Primary clouds) ===
    if (world.nishita.clouds_enabled) {
        // IMPROVED SCALE: UI Scale 1.0 = moderate size. 
        // We use a non-linear mapping so that small UI values give huge clouds 
        // and large UI values give dense small clouds without losing texture.
        float scale = 0.003f / fmaxf(0.01f, world.nishita.cloud_scale);
        float3 layer1 = render_cloud_layer(
            world, rayDir, bg_color,
            world.nishita.cloud_height_min, world.nishita.cloud_height_max,
            scale, world.nishita.cloud_coverage, world.nishita.cloud_density,
            world.nishita.cloud_detail,
            transmittance
        );
        cloudColor += layer1;
    }
    
    // === LAYER 2 (Secondary clouds - e.g., high cirrus) ===
    if (world.nishita.cloud_layer2_enabled) {
        float scale2 = 0.003f / fmaxf(0.01f, world.nishita.cloud2_scale);
        float3 layer2 = render_cloud_layer(
            world, rayDir, bg_color,
            world.nishita.cloud2_height_min, world.nishita.cloud2_height_max,
            scale2, world.nishita.cloud2_coverage, world.nishita.cloud2_density,
            world.nishita.cloud_detail, // Use same detail for layer 2
            transmittance
        );
        cloudColor += layer2;
    }
    
    // Final blend with background
    return bg_color * transmittance + cloudColor;
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
            if (!vol.density_grid) continue;
            
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

                nanovdb::FloatGrid* grid = (nanovdb::FloatGrid*)vol.density_grid;
                nanovdb::math::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1> sampler(grid->tree());
                nanovdb::Vec3f idx = grid->worldToIndexF(nanovdb::Vec3f(sample_pos.x, sample_pos.y, sample_pos.z));
                
                float d = sampler(idx);
                float density = fmaxf((d - vol.density_remap_low) / (vol.density_remap_high - vol.density_remap_low + 1e-6f), 0.0f) * vol.density_multiplier;
                total_extinction += density * (vol.scatter_coefficient + vol.absorption_coefficient);
            }
        }
    }
    
    // 2. Procedural Clouds
    if (optixLaunchParams.world.nishita.clouds_enabled || optixLaunchParams.world.nishita.cloud_layer2_enabled) {
        float height = world_pos.y;
        for (int layer = 0; layer < 2; ++layer) {
            bool enabled = (layer == 0) ? optixLaunchParams.world.nishita.clouds_enabled : optixLaunchParams.world.nishita.cloud_layer2_enabled;
            if (!enabled) continue;
            
            float minH = (layer == 0) ? optixLaunchParams.world.nishita.cloud_height_min : optixLaunchParams.world.nishita.cloud2_height_min;
            float maxH = (layer == 0) ? optixLaunchParams.world.nishita.cloud_height_max : optixLaunchParams.world.nishita.cloud2_height_max;
            
            if (height >= minH && height <= maxH) {
                // Simplified cloud sampling (similar to calculate_cloud_transmittance)
                // Match the improved scale logic from main renderer
                float scale = 0.003f / fmaxf(0.01f, (layer == 0) ? optixLaunchParams.world.nishita.cloud_scale : optixLaunchParams.world.nishita.cloud2_scale);
                float coverage = (layer == 0) ? optixLaunchParams.world.nishita.cloud_coverage : optixLaunchParams.world.nishita.cloud2_coverage;
                float density_mult = (layer == 0) ? optixLaunchParams.world.nishita.cloud_density : optixLaunchParams.world.nishita.cloud2_density;
                float detail = optixLaunchParams.world.nishita.cloud_detail;
                
                float3 noise_pos = world_pos * scale;
                float d = cloud_shape(noise_pos, coverage, detail);
                
                // Height gradient fade - match the cumulus profile roughly
                float h_norm = (height - minH) / (maxH - minH);
                float height_gradient = smoothstep_cloud(0.0f, 0.05f, h_norm) * smoothstep_cloud(1.0f, 0.3f, h_norm);
                
                total_extinction += d * density_mult * height_gradient;
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
            if (!vol.density_grid) continue;
            
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

                    nanovdb::FloatGrid* grid = (nanovdb::FloatGrid*)vol.density_grid;
                    nanovdb::math::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1> sampler(grid->tree());

                    while (probe_t < t1) {
                        float3 pos = local_origin + local_dir * probe_t;
                        
                        // Pivot offset adjustment
                        pos.x -= vol.pivot_offset[0];
                        pos.y -= vol.pivot_offset[1];
                        pos.z -= vol.pivot_offset[2];

                        nanovdb::Vec3f idx = grid->worldToIndexF(nanovdb::Vec3f(pos.x, pos.y, pos.z));
                        float d = sampler(idx);
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
    
    // 2. Procedural Clouds (Dithered entry)
    if (optixLaunchParams.world.nishita.clouds_enabled || optixLaunchParams.world.nishita.cloud_layer2_enabled) {
        float3 rDir = normalize(ray_dir);
        if (fabsf(rDir.y) > 1e-4f) {
            for (int layer = 0; layer < 2; ++layer) {
                bool enabled = (layer == 0) ? optixLaunchParams.world.nishita.clouds_enabled : optixLaunchParams.world.nishita.cloud_layer2_enabled;
                if (enabled) {
                    float minH = (layer == 0) ? optixLaunchParams.world.nishita.cloud_height_min : optixLaunchParams.world.nishita.cloud2_height_min;
                    float maxH = (layer == 0) ? optixLaunchParams.world.nishita.cloud_height_max : optixLaunchParams.world.nishita.cloud2_height_max;
                    
                    float t_enter = (minH - ray_origin.y) / rDir.y;
                    float t_exit = (maxH - ray_origin.y) / rDir.y;
                    if (t_enter > t_exit) { float tmp = t_enter; t_enter = t_exit; t_exit = tmp; }
                    
                    if (t_exit > 0.0f && t_enter < max_dist) {
                        float t_hit = fmaxf(t_enter, 0.0f);
                        float dither = (curand_uniform(rng) - 0.5f) * 50.0f; // 50m scale dither
                        t_min = fminf(t_min, t_hit + dither);
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
        if (!vol.density_grid) continue;
        
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
        
        // NanoVDB Sampler (Trilinear) with Accessor for speed
        nanovdb::FloatGrid* grid = (nanovdb::FloatGrid*)vol.density_grid;
        auto acc = grid->getAccessor();
        nanovdb::math::SampleFromVoxels<nanovdb::FloatGrid::AccessorType, 1, false> sampler(acc);
        
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

            nanovdb::Vec3f idx = grid->worldToIndexF(nanovdb::Vec3f(pos.x, pos.y, pos.z));
            
            float d = sampler(idx);
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
    curandState* rng
) {
    // Skip if no density grid
    if (!vol.density_grid) {
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
    
    // Step size - ADAPTIVE PRECISION
    float world_t_enter = t_enter / dir_length;
    float world_t_exit = t_exit / dir_length;
    float world_vol_extent = world_t_exit - world_t_enter;

    // Safety Step Calculation: Ensures we reach the exit before hitting max_steps
    float min_step_to_cover = world_vol_extent / fmaxf(1.0f, (float)vol.max_steps - 1.0f);
    float step = fmaxf(vol.step_size, min_step_to_cover);
    
    // Quality clamp: Ensure we don't use a step larger than the voxel size if requested small
    float world_scale = 1.0f / fmaxf(1e-6f, length(make_float3(vol.inv_transform[0], vol.inv_transform[1], vol.inv_transform[2])));
    float world_voxel_size = vol.voxel_size * world_scale;
    step = fmaxf(0.0001f, fminf(step, fmaxf(world_voxel_size, world_vol_extent / 16.0f)));
    
    // Re-verify that step will actually cover the volume (final safety)
    step = fmaxf(step, min_step_to_cover);

    // Raymarching State
    float jitter = curand_uniform(rng) * step;
    float t = world_t_enter + jitter;
    float3 transmittance = make_float3(1.0f);
    float3 accumulated_color = make_float3(0.0f);
    
    // Grid Setup
    const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(vol.density_grid);
    using AccT = nanovdb::FloatGrid::AccessorType;
    AccT acc = grid->getAccessor();
    
    // Unified fast Trilinear Sampler (Order 1)
    nanovdb::math::SampleFromVoxels<AccT, 1, false> sampler(acc);
    
    float remap_range = vol.density_remap_high - vol.density_remap_low + 1e-6f;
    float3 local_sun_dir = normalize(transform_vector_affine(sun_dir, vol.inv_transform));
    float3 local_ray_dir = normalize(transform_vector_affine(ray_dir, vol.inv_transform));
    
    const nanovdb::FloatGrid* temp_grid = (vol.temperature_grid && vol.emission_mode == 2) ? 
        reinterpret_cast<const nanovdb::FloatGrid*>(vol.temperature_grid) : nullptr;
    
    int steps = 0;
    while (t < world_t_exit && steps < vol.max_steps && (transmittance.x + transmittance.y + transmittance.z) > 0.015f) {
        float3 world_pos = ray_origin + ray_dir * t;
        float3 local_pos = transform_point_affine(world_pos, vol.inv_transform);
        
        // Pivot offset correction
        local_pos.x -= vol.pivot_offset[0];
        local_pos.y -= vol.pivot_offset[1];
        local_pos.z -= vol.pivot_offset[2];
        
        nanovdb::Vec3f idx = grid->worldToIndexF(nanovdb::Vec3f(local_pos.x, local_pos.y, local_pos.z));
        float raw_density = sampler(idx);
        if (!isfinite(raw_density)) raw_density = 0.0f;
        
        float density = fmaxf((raw_density - vol.density_remap_low) / remap_range, 0.0f) * vol.density_multiplier;
        
        if (density > curand_uniform(rng) * 0.01f) {
            float sigma_a = density * vol.absorption_coefficient;
            float sigma_s = density * vol.scatter_coefficient;
            float sigma_t = sigma_a + sigma_s;
            float albedo_avg = vol.scatter_color.x * 0.2126f + vol.scatter_color.y * 0.7152f + vol.scatter_color.z * 0.0722f;
            
            // --- BLENDED MULTI-SCATTER TRANSMITTANCE (Matches CPU Scalar Model) ---
            float T_single = expf(-sigma_t * step);
            float T_multi_p = expf(-sigma_t * step * 0.25f);
            float step_transmittance = T_single * (1.0f - vol.scatter_multi * albedo_avg) + T_multi_p * (vol.scatter_multi * albedo_avg);
            
            // Emission
            float3 emission = make_float3(0.0f);
            if (vol.emission_mode == 1) emission = vol.emission_color * vol.emission_intensity * density;
            else if (vol.emission_mode == 2 && temp_grid) {
                AccT temp_acc = temp_grid->getAccessor();
                nanovdb::math::SampleFromVoxels<AccT, 1, false> temp_sampler(temp_acc);
                nanovdb::Vec3f temp_idx = temp_grid->worldToIndexF(nanovdb::Vec3f(local_pos.x, local_pos.y, local_pos.z));
                float temperature = temp_sampler(temp_idx);
                if (!isfinite(temperature)) temperature = 293.0f;
                
                float kelvin = (temperature > 20.0f) ? (temperature * vol.temperature_scale) : ((temperature * 3000.0f + 1000.0f) * vol.temperature_scale);
                float t_ramp = (temperature > 20.0f) ? ((vol.max_temperature > 20.0f) ? (temperature/vol.max_temperature) : (temperature/6000.0f)) : temperature;
                
                float3 e_color = vol.color_ramp_enabled ? sample_color_ramp(vol, t_ramp * vol.temperature_scale) : blackbody_to_rgb(kelvin);
                emission = e_color * (density * vol.blackbody_intensity);
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
                    float s_step_world = world_vol_extent / (float)(vol.shadow_steps * 2);
                    float3 local_s_step_vec = local_sun_dir * (s_step_world * dir_length);
                    float3 curr_local_s_pos = local_pos;

                    float s_trans = 0.0f;
                    for (int ls = 0; ls < vol.shadow_steps && s_trans < 5.0f; ++ls) {
                        curr_local_s_pos += local_s_step_vec;
                        if (curr_local_s_pos.x < vol.local_bbox_min.x || curr_local_s_pos.x > vol.local_bbox_max.x || 
                            curr_local_s_pos.y < vol.local_bbox_min.y || curr_local_s_pos.y > vol.local_bbox_max.y || 
                            curr_local_s_pos.z < vol.local_bbox_min.z || curr_local_s_pos.z > vol.local_bbox_max.z) break;
                        
                        nanovdb::Vec3f s_idx = grid->worldToIndexF(nanovdb::Vec3f(curr_local_s_pos.x, curr_local_s_pos.y, curr_local_s_pos.z));
                        float sd = sampler(s_idx);
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
                    shadow = 1.0f - vol.shadow_strength * (1.0f - phys_trans);
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
            
            // 3. Sky/Ambient Lighting (CPU Parity)
            total_light += gpu_get_ambient_radiance_volume(optixLaunchParams.world, make_float3(0, 1, 0)) * 0.15f;

            float3 albedo = vol.scatter_color;
            float3 ms_boost = make_float3(1.0f) + albedo * vol.scatter_multi * 2.0f;
            float3 source = (albedo * total_light * sigma_s * ms_boost + emission);
            
            // 4. Energy-Stable Integration (Matches CPU line 3376)
            float one_minus_T = 1.0f - step_transmittance;
            accumulated_color += source * (one_minus_T * transmittance);
            transmittance *= step_transmittance;
        }
        t += step;
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

// ═══════════════════════════════════════════════════════════════════════════
// VOLUMETRIC GOD RAYS - Ray-marched light shafts with proper occlusion
// Objects will block god rays creating shadows in the volumetric effect
// ═══════════════════════════════════════════════════════════════════════════

// Forward declaration for shadow test
__device__ bool trace_shadow_test(float3 origin, float3 direction, float maxDist);

// Simplified cloud transmittance for shadow/godray rays - OPTIMIZED
__device__ float calculate_cloud_transmittance(const WorldData& world, float3 origin, float3 dir, float maxDist, curandState* rng) {
    if (!world.nishita.clouds_enabled && !world.nishita.cloud_layer2_enabled) return 1.0f;
    
    // Performance: Fast exit if coverage is zero
    if (world.nishita.cloud_coverage < 0.01f && world.nishita.cloud2_coverage < 0.01f) return 1.0f;
    
    if (fabsf(dir.y) < 1e-6f) return 1.0f;

    float cloud_trans = 1.0f;
    
    for (int layer = 0; layer < 2; ++layer) {
        bool enabled = (layer == 0) ? world.nishita.clouds_enabled : world.nishita.cloud_layer2_enabled;
        if (!enabled) continue;
        
        float minH = (layer == 0) ? world.nishita.cloud_height_min : world.nishita.cloud2_height_min;
        float maxH = (layer == 0) ? world.nishita.cloud_height_max : world.nishita.cloud2_height_max;
        
        float scale = 0.003f / fmaxf(0.1f, (layer == 0) ? world.nishita.cloud_scale : world.nishita.cloud2_scale);
        float coverage = (layer == 0) ? world.nishita.cloud_coverage : world.nishita.cloud2_coverage;
        float density_mult = (layer == 0) ? world.nishita.cloud_density : world.nishita.cloud2_density;
        
        if (coverage < 0.01f) continue;
        
        float t0 = (minH - origin.y) / dir.y;
        float t1 = (maxH - origin.y) / dir.y;
        float t_enter = fminf(t0, t1);
        float t_exit = fmaxf(t0, t1);
        
        t_enter = fmaxf(t_enter, 0.0f);
        t_exit = fminf(t_exit, maxDist);
        
        if (t_exit > t_enter) {
            float dist = t_exit - t_enter;
            // REDUCED STEPS for sun transmittance (was 12, now 4-8 adaptive)
            int steps = (world.nishita.cloud_quality > 1.0f) ? 8 : 4; 
            float stepSize = dist / (float)steps;
            float density_sum = 0.0f;
            
            for (int i = 0; i < steps; ++i) {
                float t = t_enter + stepSize * (i + curand_uniform(rng));
                float3 p = origin + dir * t;
                float3 offsetP = p + make_float3(world.nishita.cloud_offset_x, 0.0f, world.nishita.cloud_offset_z);
                float h_frac = (p.y - minH) / (maxH - minH);
                
                float h_grad = smoothstep_cloud(0.0f, 0.05f, h_frac) * smoothstep_cloud(1.0f, 0.3f, h_frac);
                if (h_frac < 0.2f) {
                     h_grad = fmaxf(h_grad, smoothstep_cloud(0.0f, 0.02f, h_frac));
                }
                
                float d = fast_cloud_shape(offsetP * scale, coverage);
                density_sum += d * h_grad * density_mult * stepSize;
            }
            cloud_trans *= expf(-density_sum * 15.0f * world.nishita.cloud_absorption);
        }
    }
    return cloud_trans;
}

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

    // 4. Check Procedural Cloud Occlusion
    if (transmittance > 0.01f && (optixLaunchParams.world.nishita.clouds_enabled || optixLaunchParams.world.nishita.cloud_layer2_enabled)) {
        transmittance *= calculate_cloud_transmittance(optixLaunchParams.world, origin, jitteredSunDir, maxDist, rng);
    }
    
    return transmittance;
}

// Shadow test for god rays - MUST be before calculate_volumetric_god_rays
__device__ bool trace_shadow_test(float3 origin, float3 direction, float maxDist) {
    Ray shadow_ray(origin, direction);
    
    OptixHitResult shadow_payload = {};
    trace_shadow_ray(shadow_ray, &shadow_payload, SCENE_EPSILON, maxDist);
    
    return shadow_payload.hit;
}

__device__ float3 calculate_volumetric_god_rays(
    const WorldData& world,
    float3 rayOrigin,
    float3 rayDir,
    float maxDistance,
    curandState* rng
) {
    // 1. EARLY EXIT & PARAMETERS
    if (!world.nishita.godrays_enabled || world.nishita.godrays_intensity <= 0.001f || world.nishita.godrays_density <= 0.0f) {
        return make_float3(0.0f);
    }
    
    float3 sunDir = normalize(world.nishita.sun_direction);
    float sunDot = dot(rayDir, sunDir);

    // Mie scattering is extremely forward-heavy. We fade based on anisotropy.
    float g = world.nishita.mie_anisotropy;
    float anisotropyFade = powf(fmaxf(0.0f, sunDot), 1.0f + (1.0f - g) * 10.0f);
    if (anisotropyFade < 0.001f) return make_float3(0.0f);
    
    if (sunDir.y < -0.05f) return make_float3(0.0f);

    // 2. STOCHASTIC ADAPTIVE STEPPING (Performance Focus)
    float marchDistance = fminf(maxDistance, world.nishita.fog_distance);
    marchDistance = fminf(marchDistance, 5000.0f); // Balanced for most scenes
    
    // Reduced steps for major performance gain (was 8-48, now 8-24)
    int numSteps = (sunDot > 0.98f) ? world.nishita.godrays_samples : (world.nishita.godrays_samples / 2);
    numSteps = clamp(numSteps, 8, 24); 
    
    float stepSize = marchDistance / (float)numSteps;
    float3 godRayColor = make_float3(0.0f);
    float transmittance = 1.0f;

    // 3. PHASE & DENSITY - Physically Plausible Scale
    float g2 = g * g;
    float phase = (1.0f - g2) / (4.0f * M_PI * powf(1.0f + g2 - 2.0f * g * sunDot, 1.5f));
    phase = fminf(phase, 6.0f); // Toned down Peak
    
    // REDUCED DENSITY (0.03 -> 0.002): Now scales correctly for kilometer-scale mountains
    float mediaDensity = world.nishita.godrays_density * 0.002f; 
    float3 sunRadianceBase = make_float3(1.0f, 0.98f, 0.95f) * world.nishita.sun_intensity * 0.15f;

    // 4. RAYMARCHING LOOP
    float jitter = curand_uniform(rng);
    float t = jitter * stepSize;
    
    for (int i = 0; i < numSteps; ++i) {
        if (t > marchDistance) break;
        
        float nearFade = fminf(1.0f, fmaxf(0.0f, (t - 0.2f) * 4.0f));
        if (nearFade > 0.001f) {
            float3 samplePos = rayOrigin + rayDir * t;
            
            float h = fmaxf(0.0f, samplePos.y + world.nishita.altitude);
            float heightFactor = expf(-h * 0.0002f); 
            
            float sigma_s = mediaDensity * heightFactor;
            float sigma_t = sigma_s; // Decay removed for simplicity
            float stepTrans = expf(-sigma_t * stepSize);
            
            float3 sunTrans = gpu_get_transmittance(world, samplePos, sunDir);
            float3 currentSunRadiance = sunRadianceBase * sunTrans;

            // SOFT SHADOWS: Jitter sun direction sample based on sun size for God Rays
            // This prevents sharp 'leaking' lines on mountain ridges.
            float3 jitteredSunDir = sunDir;
            if (world.nishita.sun_size > 0.1f) {
                float3 spread = sample_sphere_cap(sunDir, world.nishita.sun_size * 0.01745f, rng); // deg to rad
                jitteredSunDir = normalize(spread);
            }
            
            float occlusion = calculate_sun_transmittance(samplePos, jitteredSunDir, 100000.0f, rng);
            
            if (occlusion > 0.001f && sigma_t > 1e-6f) {
                float3 inscatter = currentSunRadiance * phase * occlusion * (sigma_s / sigma_t) * nearFade;
                godRayColor += transmittance * inscatter * (1.0f - stepTrans);
            }
            
            transmittance *= stepTrans;
        }
        
        if (transmittance < 0.01f) break;
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
    
    // INCREASED FOG IMPACT (100 -> 300) - Toned down from 1000 to prevent whitening
    float densityFactor = 1.0f + world.nishita.fog_density * 300.0f;
    float effectiveDist = dist * densityFactor;
    
    const float min_dist = world.advanced.aerial_min_distance;
    const float max_dist = world.advanced.aerial_max_distance;
    float ramp = (dist < min_dist) ? 0.0f : fminf(1.0f, (dist - min_dist) / fmaxf(1.0f, max_dist - min_dist));
    
    // Adjusted horizon scaling (20km -> 10km) to make haze more apparent
    float distFactor = fminf(1.0f, effectiveDist / 10000.0f);
    distFactor *= (ramp * ramp); 
    
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

__device__ float3 evaluate_background(const WorldData& world, const float3& origin, const float3& dir, curandState* rng) {
    if (world.mode == 0) { // WORLD_MODE_COLOR
        return render_clouds(world, dir, world.color);
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
            return render_clouds(world, dir, make_float3(tex.x, tex.y, tex.z) * world.env_intensity);
        }
        return render_clouds(world, dir, world.color);
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
        
        // Soft halo contribution
        float excessPhase = fmaxf(0.0f, phaseM - 2.0f);
        if (excessPhase > 0.0f) {
            // HALO LEAK PROTECTION: Add shadow test for halo to prevent bleeding through solid objects
            if (!trace_shadow_test(origin, sunDir, 100000.0f)) {
                float3 transToSun = gpu_get_transmittance(world, origin, sunDir);
                float3 mieScat = make_float3(world.nishita.mie_density, world.nishita.mie_density, world.nishita.mie_density); 
                mieScat = mieScat * world.nishita.mie_scattering * 0.15f;
                radiance += transToSun * mieScat * excessPhase * world.nishita.sun_intensity;
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

        // Clouds are rendered on top of the sky radiance
        return render_clouds(world, dir, radiance);
    }

    return make_float3(0,0,0);
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

__device__ int pick_smart_light(const float3& hit_position, curandState* rng) {
    int light_count = optixLaunchParams.light_count;
    if (light_count == 0) return -1;

    // --- Öncelik: Directional light --- (örnekleme olasılığı %33)
    for (int i = 0; i < light_count; i++) {
        if (optixLaunchParams.lights[i].type == 1) {
            if (random_float(rng) < 0.33f)
                return i;
        }
    }

    // --- Tüm ışık türleri için akıllı seçim ---
    float weights[128];
    float total_weight = 0.0f;

    for (int i = 0; i < light_count; i++) {
        const LightGPU& light = optixLaunchParams.lights[i];
        float dist = length(light.position - hit_position);
        dist = fmaxf(dist, 1.0f);
        
        float falloff = 1.0f / (dist * dist);
        float intensity = luminance(light.color * light.intensity);
        
        if (light.type == 0) { // Point Light
            weights[i] = falloff * intensity;
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
    if (total_weight < 1e-6f)
        return clamp(int(random_float(rng) * light_count), 0, light_count - 1);

    // --- Weighted seçim ---
    float r = random_float(rng) * total_weight;
    float accum = 0.0f;
    for (int i = 0; i < light_count; i++) {
        accum += weights[i];
        if (r <= accum)
            return i;
    }

    // --- Güvenli fallback ---
    return clamp(int(random_float(rng) * light_count), 0, light_count - 1);
}

__device__ float3 sample_directional_light(const LightGPU& light, const float3& hit_pos, curandState* rng, float3& wi_out) {
    float3 L = normalize(light.direction);
    float3 tangent = normalize(cross(L, make_float3(0.0f, 1.0f, 0.0f)));
    if (length(tangent) < 1e-3f) tangent = normalize(cross(L, make_float3(1.0f, 0.0f, 0.0f)));
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
    curandState* rng
) {
    float3 wi;
    float distance = 1.0f;
    float attenuation = 1.0f;
    const float shadow_bias = SCENE_EPSILON;

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

    float3 origin = payload.position + payload.normal * shadow_bias;
    Ray shadow_ray(origin, wi);
    OptixHitResult shadow_payload = {};
    
    trace_shadow_ray(shadow_ray, &shadow_payload, SCENE_EPSILON, distance);
    if (shadow_payload.hit) return make_float3(0.0f, 0.0f, 0.0f);
    
    // Check VDB Occlusion (Volumetric Shadow)
    float vdb_transmittance = calculate_vdb_occlusion(origin, wi, distance, rng);
    if (vdb_transmittance < 0.001f) return make_float3(0.0f, 0.0f, 0.0f);

    float3 f = evaluate_brdf(material, payload, wo, wi);
    float pdf_brdf_val = pdf_brdf(material, wo, wi, payload.normal);
    float pdf_brdf_val_mis = clamp(pdf_brdf_val, 0.001f, 5000.0f);

    float pdf_light = 1.0f;
    if (light.type == 0) {
        float area = 4.0f * M_PIf * light.radius * light.radius;
        pdf_light = 1.0f / area;
    }
    else if (light.type == 1) {
        float apparent_angle = atan2(light.radius, 1000.0f);
        float cos_epsilon = cos(apparent_angle);
        float solid_angle = 2.0f * M_PIf * (1.0f - cos_epsilon);
        pdf_light = 1.0f / solid_angle;
    }
    else if (light.type == 2) { // Area Light
        float area = light.area_width * light.area_height;
        pdf_light = 1.0f / fmaxf(area, 1e-4f);
    }
    else if (light.type == 3) { // Spot Light
        float solid_angle = 2.0f * M_PIf * (1.0f - light.outer_cone_cos);
        pdf_light = 1.0f / fmaxf(solid_angle, 1e-4f);
    }

    float mis_weight = power_heuristic(pdf_light, pdf_brdf_val_mis);
    float3 Li = light.color * light.intensity * attenuation * vdb_transmittance;
    return (f * Li * NdotL) * mis_weight;
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
        float3 L = normalize(light.direction);
        float3 tangent = normalize(cross(L, make_float3(0.0f, 1.0f, 0.0f)));
        if (length(tangent) < 1e-3f) tangent = normalize(cross(L, make_float3(1.0f, 0.0f, 0.0f)));
        float3 bitangent = normalize(cross(L, tangent));

        float2 disk_p = random_in_unit_disk(rng);
        float3 offset = (tangent * disk_p.x + bitangent * disk_p.y) * light.radius;

        float3 light_pos = L * 1000.0f + offset;
        wi = normalize(light_pos);
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
    float3 origin = payload.position + payload.normal * shadow_bias;
    Ray shadow_ray(origin, wi);

    OptixHitResult shadow_payload = {};
    trace_shadow_ray(shadow_ray, &shadow_payload, shadow_bias, distance);
    if (shadow_payload.hit) return result;

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
        float threshold = curand_uniform(rng) * 0.01f;
        
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
            float edge_factor = 1.0f;
            if (d_edge < falloff_dist && falloff_dist > 0.001f) {
                float t_f = d_edge / falloff_dist;
                // Smoothstep: 3t^2 - 2t^3
                edge_factor = t_f * t_f * (3.0f - 2.0f * t_f);
            }
            
            local_density *= edge_factor;
        }
        
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
                    density_accum += local_density * (vol_absorption + vol_scattering) * light_step_size;
                    if (density_accum > 5.0f) break;
                }
                // --- STABLE SHADOWING (Matches CPU 1.0 - strength * (1-T)) ---
                float beers = expf(-density_accum);
                float beers_soft = expf(-density_accum * 0.25f);
                float albedo_p = vol_albedo.x * 0.2126f + vol_albedo.y * 0.7152f + vol_albedo.z * 0.0722f;
                float phys_trans = beers * (1.0f - multi_scatter * albedo_p) + beers_soft * (multi_scatter * albedo_p);
                shadow_trans = 1.0f - shadow_strength * (1.0f - phys_trans);
            }
            
            float cos_theta = dot(ray_dir, sun_dir);
            float phase = gpu_phase_dual_hg(cos_theta, vol_g, g_back, lobe_mix);
            float powder = gpu_powder_effect(local_density, cos_theta);
            phase *= (1.0f + powder * 0.5f);

            float3 sun_trans = gpu_get_transmittance(optixLaunchParams.world, pos, sun_dir);
            float3 sun_color = sun_trans * sun_intensity;
            float3 shadow_radiance = sun_color * shadow_trans * phase;
            float3 ambient = gpu_get_ambient_radiance_volume(optixLaunchParams.world, make_float3(0, 1, 0)) * 0.15f;
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
    float step = fmaxf(vol.step_size <= 0.0f ? 0.05f : vol.step_size, min_step_to_cover);
    
    // Jitter
    float jitter = curand_uniform(rng) * step;
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
        
        if (density > threshold) {
            float sigma_a = density * vol.absorption_coefficient;
            float sigma_s = density * vol.scatter_coefficient;
            float sigma_t = sigma_a + sigma_s;
            float albedo_avg = vol.scatter_color.x * 0.2126f + vol.scatter_color.y * 0.7152f + vol.scatter_color.z * 0.0722f;
            
            // --- BLENDED MULTI-SCATTER TRANSMITTANCE (Matches CPU Scalar Model) ---
            float T_single = expf(-sigma_t * step);
            float T_multi_p = expf(-sigma_t * step * 0.25f);
            float step_transmittance = T_single * (1.0f - vol.scatter_multi * albedo_avg) + T_multi_p * (vol.scatter_multi * albedo_avg);
            
            float3 total_radiance = make_float3(0.0f);
            float cos_theta = dot(ray_dir, sun_dir);
            float phase = gpu_phase_dual_hg(cos_theta, vol.scatter_anisotropy, vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
            
            float shadow_trans = 1.0f;
            if (vol.shadow_steps > 0) {
                 float shadow_step = vol.step_size * 2.0f; 
                 float shadow_density_sum = 0.0f;
                 for(int ls=0; ls < vol.shadow_steps; ++ls) {
                     float3 shadow_pos = world_pos + sun_dir * (shadow_step * (float)(ls + 1));
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
                 float beers_soft = expf(-shadow_density_sum * shadow_step * (vol.absorption_coefficient + vol.scatter_coefficient) * 0.25f);
                 float albedo_lum = vol.scatter_color.x * 0.2126f + vol.scatter_color.y * 0.7152f + vol.scatter_color.z * 0.0722f;
                 float phys_trans = beers * (1.0f - vol.scatter_multi * albedo_lum) + beers_soft * (vol.scatter_multi * albedo_lum);
                 shadow_trans = 1.0f - vol.shadow_strength * (1.0f - phys_trans);
            }
            float3 sun_trans = gpu_get_transmittance(optixLaunchParams.world, world_pos, sun_dir);
            float3 sun_color = sun_trans * sun_intensity;
            total_radiance = sun_color * shadow_trans * phase;

            // --- RESTORED PARITY FEATURES (Powder + Ambient) ---
            float powder = gpu_powder_effect(density, cos_theta);
            total_radiance = total_radiance * (1.0f + powder * 0.5f);
            
            float3 ambient = gpu_get_ambient_radiance_volume(optixLaunchParams.world, make_float3(0, 1, 0)) * 0.15f;
            total_radiance += ambient;
            
            float3 emission = make_float3(0.0f);
            if (vol.emission_mode == 1) { // Constant
                emission = vol.emission_color * vol.emission_intensity * density;
            }
            else if (vol.emission_mode == 2) { // Blackbody / Color Ramp
                float temperature = density; 
                if (vol.temperature_texture) {
                    temperature = tex3D<float>(vol.temperature_texture, tex_coord.x, tex_coord.y, tex_coord.z);
                    if (!isfinite(temperature)) temperature = density; 
                }
                
                float3 e_color; float kelvin; float t_ramp_val;
                if (temperature > 20.0f) { // Likely physical Kelvin
                    kelvin = temperature * vol.temperature_scale;
                    t_ramp_val = (vol.max_temperature > 20.0f) ? (temperature / vol.max_temperature) : (temperature / 6000.0f);
                } else { // Likely normalized 0-1
                    kelvin = (temperature * 3000.0f + 1000.0f) * vol.temperature_scale;
                    t_ramp_val = temperature;
                }
                if (vol.color_ramp_enabled) e_color = sample_color_ramp_gas(vol, t_ramp_val * vol.temperature_scale);
                else e_color = blackbody_to_rgb(kelvin);
                emission = e_color * density * vol.blackbody_intensity;
            }

            float3 ms_boost = make_float3(1.0f) + vol.scatter_color * vol.scatter_multi * 2.0f;
            float3 source = (vol.scatter_color * total_radiance * sigma_s * ms_boost + emission);
            
            // Energy-stable integration (Parity with Renderer.cpp)
            float one_minus_T = 1.0f - step_transmittance;
            accumulated_color += source * (one_minus_T * transmittance);
            transmittance *= step_transmittance;
        }
        
        t += step;
        steps++;
    }

    out_transmittance = (transmittance.x + transmittance.y + transmittance.z) / 3.0f;
    return accumulated_color;
}

__device__ float3 ray_color(Ray ray, curandState* rng) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float3 current_medium_absorb = make_float3(0.0f, 0.0f, 0.0f); // Default: Air (no absorption)

    const int max_depth = optixLaunchParams.max_depth;
    int light_count = optixLaunchParams.light_count;
    int light_index = (light_count > 0) ? pick_smart_light(ray.origin, rng) : -1;
    
    float first_hit_t = -1.0f;
    float vol_depth = -1.0f;
    float vol_trans_accum = 1.0f;
    float3 first_ray_origin = ray.origin;
    float3 first_ray_dir = ray.direction;
    
    // Firefly önleme için maksimum katkı limiti
    const float MAX_CONTRIBUTION = 100.0f;

    for (int bounce = 0; bounce < max_depth; ++bounce) {
        OptixHitResult payload = {};
        float t_min = (bounce == 0) ? optixLaunchParams.clip_near : SCENE_EPSILON;
        float t_max = (bounce == 0) ? optixLaunchParams.clip_far : 1e16f;
        
        trace_ray(ray, &payload, t_min, t_max);
        
        // --- 1. HANDLE HAIR (Unified Path) ---
        if (payload.hit && payload.is_hair) {
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

        // === VOLUMETRIC ABSORPTION (Beer's Law) ===
        // Apply absorption based on the distance traveled in the current medium
        if (payload.hit && (current_medium_absorb.x > 0.0f || current_medium_absorb.y > 0.0f || current_medium_absorb.z > 0.0f)) {
            float dist = payload.t;
            float3 transmission = make_float3(
                expf(-current_medium_absorb.x * dist),
                expf(-current_medium_absorb.y * dist),
                expf(-current_medium_absorb.z * dist)
            );
            throughput *= transmission;
        }
        
        // ═══════════════════════════════════════════════════════════
        // VOLUMETRIC GOD RAYS - Only on primary ray for performance
        // God rays are accumulated to the point of first hit or infinity
        // ═══════════════════════════════════════════════════════════
        if (bounce == 0 && optixLaunchParams.world.nishita.godrays_enabled) {
            float maxDist = payload.hit ? payload.t : 200000.0f;
            
            // --- STOCHASTIC DEPTH PROBE (Film Quality Occlusion) ---
            float t_vol_min = get_stochastic_volumetric_occlusion(
                ray.origin, ray.direction, maxDist, rng, 
                0.0f
            );
            
            // Clamp God Ray march to the visual surface
            maxDist = fminf(maxDist, t_vol_min);

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
                if (!vdb.density_grid) continue;
                
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
            for (int i = 0; i < valid_count; ++i) {
                const GpuVDBVolume& vdb = optixLaunchParams.vdb_volumes[sorted_indices[i]];
                
                // Determine occlusion distance (depth clipping)
                float max_dist = payload.hit ? payload.t : 1e16f;
                if (bounce == 0 && max_dist > optixLaunchParams.clip_far) max_dist = optixLaunchParams.clip_far;

                float vol_transmittance = 1.0f;
                float3 vol_color = raymarch_vdb_volume(
                    vdb,
                    ray.origin,
                    ray_dir_norm,
                    sun_dir,
                    sun_intensity,
                    vol_transmittance,
                    max_dist,
                    rng
                );
                
                // Accumulate volume contribution
                color += throughput * vol_color;
                throughput *= vol_transmittance;
                
                // NEW: Update depth for fogging calculation
                // If this is the primary ray and we hit substantive volume, pull the fog distance forward.
                if (bounce == 0) {
                    if (vol_depth < 0.0f || entry_distances[i] < vol_depth) {
                        vol_depth = entry_distances[i];
                    }
                    vol_trans_accum *= vol_transmittance;
                }
                
                // Early termination if fully absorbed
                if (throughput.x < 0.001f && throughput.y < 0.001f && throughput.z < 0.001f) {
                    return color;
                }
            }
        }



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
                            if (vol_depth < 0.0f || current_t < vol_depth) {
                                vol_depth = current_t;
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
            float3 bg_color = evaluate_background(optixLaunchParams.world, ray.origin, ray.direction, rng);

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
            color += throughput * bg_color;
            break;
        }

        float3 wo = -normalize(ray.direction);
        Ray scattered;
        float3 attenuation;
        float pdf;
        bool is_specular;
        
        GpuMaterial mat = optixLaunchParams.materials[payload.material_id];

        // --- Scatter başarısızsa çık ---
        if (!scatter_material(mat, payload, ray, rng, &scattered, &attenuation, &pdf, &is_specular))
            break;
       
        throughput *= attenuation;

        // --- Volumetric Medium Tracking (for Beer's Law) ---
        if (is_specular && mat.transmission > 0.01f) {
            // Check if we are entering or exiting based on normal direction
            // Ray direction is incoming, Normal is surface normal
            float NdotD = dot(normalize(ray.direction), payload.normal);
            bool entering = NdotD < 0.0f; 
            
            if (entering) {
                // Entering medium: Set absorption coefficient
                // sigma_a = (1 - color) * density
                float3 absorb_base = make_float3(1.0f, 1.0f, 1.0f) - mat.subsurface_color;
                // Clamp to avoid negative values
                absorb_base = make_float3(fmaxf(absorb_base.x, 0.0f), fmaxf(absorb_base.y, 0.0f), fmaxf(absorb_base.z, 0.0f));
                
                current_medium_absorb = absorb_base * mat.subsurface_scale;
            } else {
                // Exiting medium: Reset to air (no absorption)
                current_medium_absorb = make_float3(0.0f, 0.0f, 0.0f);
            }
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
            trace_ray(exit_ray, &exit_payload,t_min,t_max);
            
            if (exit_payload.hit) {
                scattered = Ray(exit_payload.position + ray.direction * 0.01f, ray.direction);
            } else {
                // Ray exited scene through volume
                float3 bg_color = evaluate_background(optixLaunchParams.world, ray.origin, ray.direction, rng);
                float bg_factor = (bounce == 0) ? 1.0f : fmaxf(0.1f, 1.0f / (1.0f + bounce * 0.5f));
                color += throughput * bg_color * bg_factor;
                break;
            }
            
            ray = scattered;
            continue; // Skip surface shading
        }
        
        // --- Throughput clamp - aşırı parlak yansımaları önle ---
        float max_throughput = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        if (max_throughput > MAX_CONTRIBUTION) {
            throughput *= (MAX_CONTRIBUTION / max_throughput);
        }
        
        // --- Russian roulette - bounce > 2'den sonra ---
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        p = clamp(p, 0.05f, 0.95f);  // Daha sıkı sınırlar
        if (bounce > 2) {
            if (random_float(rng) > p)
                break;
            throughput /= p;
            
            // Russian roulette sonrası tekrar clamp
            max_throughput = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (max_throughput > MAX_CONTRIBUTION) {
                throughput *= (MAX_CONTRIBUTION / max_throughput);
            }
        }
        float3 emission = mat.emission;
        if (payload.has_emission_tex) {
            float4 tex = tex2D<float4>(payload.emission_tex, payload.uv.x, payload.uv.y);
            emission = make_float3(tex.x, tex.y, tex.z) * mat.emission; // Tint with material emission color
        }
       // emission *= 0.5f; // User optimization: add at half-rate to prevent over-emission
        // --- Eğer hiç ışık yoksa sadece emissive katkı yap ---
        if (light_count == 0) {           
           
            color += throughput * emission;
            ray = scattered;
            continue;
        }

        light_index = pick_smart_light(payload.position, rng);
       
        // --- Direkt ışık katkısı ---
        float3 direct = make_float3(0.0f, 0.0f, 0.0f);
        if (!is_specular && light_index >= 0) {
            direct = calculate_light_contribution(
                optixLaunchParams.lights[light_index], mat, payload, wo, rng
            );
            // Firefly kontrolü - aşırı parlak direkt katkıları sınırla
            float direct_lum = luminance(direct);
            if (direct_lum > MAX_CONTRIBUTION) {
                direct *= (MAX_CONTRIBUTION / direct_lum);
            }
        }

        // --- BRDF yönünde MIS katkı ---
        float3 brdf_mis = make_float3(0.0f, 0.0f, 0.0f);
        if (!is_specular && light_index >= 0) {
            const LightGPU& light = optixLaunchParams.lights[light_index];
            float3 wi = normalize(scattered.direction);
            float pdf_brdf_val_mis = clamp(pdf, 0.1f, 5000.0f);
            float pdf_light = 1.0f;
            float NdotL = fmaxf(dot(payload.normal, wi), 0.0f);

            if (light.type == 1) { // Directional
                float3 L = normalize(light.direction);
                float apparent_angle = atan2(light.radius, 1000.0f);
                float cos_epsilon = cos(apparent_angle);
                if (dot(wi, L) > cos_epsilon) {
                    float solid_angle = 2.0f * M_PIf * (1.0f - cos_epsilon);
                    pdf_light = 1.0f / solid_angle;
                    float mis_weight = power_heuristic(pdf_brdf_val_mis, pdf_light);
                    float3 f = evaluate_brdf(mat, payload, wo, wi);
                    brdf_mis += f * light.intensity * light.color * NdotL * mis_weight;
                }
            }
            if (light.type == 0) { // Point
                float3 delta = light.position - payload.position;
                float dist = length(delta);
                if (dist < light.radius * 1.05f) {
                    float area = 4.0f * M_PIf * light.radius * light.radius;
                    pdf_light = 1.0f / area;
                    float mis_weight = power_heuristic(pdf_brdf_val_mis, pdf_light);
                    float3 f = evaluate_brdf(mat, payload, wo, wi);
                    brdf_mis += f * (light.intensity * light.color / (dist * dist)) * NdotL * mis_weight;
                }
            }
            
            // Firefly kontrolü - aşırı parlak BRDF MIS katkılarını sınırla
            float brdf_lum = luminance(brdf_mis);
            if (brdf_lum > MAX_CONTRIBUTION) {
                brdf_mis *= (MAX_CONTRIBUTION / brdf_lum);
            }
        }
      
        // --- Toplam katkı ---
        float3 total_contribution = direct + brdf_mis + emission;
        
        // Son firefly kontrolü
        float total_lum = luminance(total_contribution);
        if (total_lum > MAX_CONTRIBUTION * 2.0f) {
            total_contribution *= (MAX_CONTRIBUTION * 2.0f / total_lum);
        }
        
        color += throughput * total_contribution;
       
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
        float dist = first_hit_t;
        
        // --- WEIGHTED FOG DISTANCE ---
        // Logic: if we have volumes, blend the fog distance based on cloud opacity.
        // This ensures thin cloud edges use background haze, while opaque centers use cloud-depth haze.
        if (vol_depth > 0.0f) {
            float background_t = (first_hit_t > 0.0f) ? first_hit_t : 10000.0f;
            float weight = 1.0f - vol_trans_accum;
            dist = lerp(background_t, vol_depth, weight);
        }
        else if (dist <= 0.0f) {
             dist = 10000.0f; 
        }
        
        color = gpu_get_aerial_perspective(world, color, first_ray_origin, first_ray_dir, dist);
    }
    else if (world.nishita.fog_enabled && world.nishita.fog_density > 0.0f) {
        // Fallback to simple height fog for non-Nishita modes
        float fogDistance = (first_hit_t > 0.0f) ? first_hit_t : world.nishita.fog_distance * 0.8f;
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

    // Final clamp - NaN ve Inf kontrolü
    color.x = isfinite(color.x) ? fminf(fmaxf(color.x, 0.0f), 100.0f) : 0.0f;
    color.y = isfinite(color.y) ? fminf(fmaxf(color.y, 0.0f), 100.0f) : 0.0f;
    color.z = isfinite(color.z) ? fminf(fmaxf(color.z, 0.0f), 100.0f) : 0.0f;

    return color;
}
