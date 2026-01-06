#pragma once
#include "trace_ray.cuh"
#include <math_constants.h>

#ifndef M_1_PIf
#define M_1_PIf 0.318309886183790671538f
#endif
#include "material_scatter.cuh"
#include "random_utils.cuh"
#include "CloudNoise.cuh"
#include "ray.h"
#include "scatter_volume_step.h"

// Helper function to render a single cloud layer
// Returns ONLY the cloud color contribution (not blended with background)
// Modifies transmittance based on cloud density encountered
__device__ inline float smoothstep_cloud(float edge0, float edge1, float x) {
    float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0f - 2.0f * t);
}

__device__ float3 render_cloud_layer(
    const WorldData& world, 
    const float3& rayDir, 
    float3 bg_color,  // Used for ambient calculation only
    float cloudMinY, float cloudMaxY,
    float scale, float coverage, float densityMult,
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
            t_exit = 30000.0f;
        }
    }
    
    if (t_exit <= 0.0f || t_exit <= t_enter) return noCloud;  // No valid intersection
    
    // Performance Optimization: Distance Culling (Frustum/Range Limit)
    // Extended to 300km to support realistic horizon (User Request)
    const float MAX_CLOUD_DIST = 300000.0f;
    if (t_enter > MAX_CLOUD_DIST) return noCloud;
    t_exit = fminf(t_exit, MAX_CLOUD_DIST);
    
    if (t_enter < 0.0f) t_enter = 0.0f;
    
    // Horizon fade - ONLY fade at the very edge (approx 1 degree)
    // Was 0.15f (~8.5 degrees) which cut off clouds too high up
    float h_val = rayDir.y / 0.02f;
    float h_t = fmaxf(0.0f, fminf(1.0f, fabsf(h_val)));
    float horizonFade = h_t * h_t * (3.0f - 2.0f * h_t);
    
    // Additional fade out at max distance (starts 50km before edge)
    float distFade = 1.0f - fmaxf(0.0f, (t_enter - (MAX_CLOUD_DIST - 50000.0f)) / 50000.0f);
    horizonFade *= distFade;

    // Quality-based step count
    float quality = fmaxf(0.1f, fminf(3.0f, world.nishita.cloud_quality));
    int baseSteps = (int)(16.0f * quality);
    int numSteps = baseSteps + (int)((float)baseSteps * (1.0f - h_t));
    
    float stepSize = (t_exit - t_enter) / (float)numSteps;
    float3 cloudColor = make_float3(0.0f, 0.0f, 0.0f);
    float t = t_enter;
    
    float localDensityMult = densityMult * horizonFade;
    
    float3 ambientSky = bg_color * 0.3f;
    float3 sunDirection = normalize(world.nishita.sun_direction);
    float g = fmaxf(0.0f, fminf(0.95f, world.nishita.mie_anisotropy));
    
    for (int i = 0; i < numSteps; ++i) {
        float jitterSeed = (float)i + (rayDir.x * 53.0f + rayDir.z * 91.0f) * 10.0f;
        float3 pos = cloudCamPos + rayDir * (t + stepSize * hash(jitterSeed));
        
        float heightFraction = (pos.y - cloudMinY) / (cloudMaxY - cloudMinY);
        
        // CUMULUS PROFILE (Strict Flat Bottom)
        // 1. Sharp rise at bottom (0% to 5% of height) creates the "cut" look
        // 2. Round falloff at top (starting at 30% height)
        // 3. No noise on the floor to ensure it's perfectly flat
        float heightGradient = smoothstep_cloud(0.0f, 0.05f, heightFraction) * smoothstep_cloud(1.0f, 0.3f, heightFraction);
        
        // Boost density at the bottom to make it look "solid" immediately
        // Gradient power to make the bottom 20% very dense
        if (heightFraction < 0.2f) {
           heightGradient = fmaxf(heightGradient, smoothstep_cloud(0.0f, 0.02f, heightFraction));
        }
        
        float3 offsetPos = pos + make_float3(world.nishita.cloud_offset_x, 0.0f, world.nishita.cloud_offset_z);
        
        // Use 3D texture if available, otherwise fallback to procedural
        // FORCE CINEMATIC QUALITY (Procedural)
        float3 noisePos = offsetPos * scale;
        
        float effectiveCoverage = coverage;
        
        // FFT Based Coverage Modulation
        if (world.nishita.cloud_use_fft && world.nishita.cloud_fft_map) {
            // Apply similar scaling to UVs based on cloud scale
            // FFT ocean tiles, so we can tile it across the sky
            // INCREASED TILING for more detail
            float uvScale = 0.002f; 
            float u = offsetPos.x * uvScale;
            float v = offsetPos.z * uvScale;
            
            // Wrap UVs
            u = u - floorf(u);
            v = v - floorf(v);
            
            // Sample FFT texture (assuming standard float texture)
            // FFT texture typically contains displacement. We use .x as height/intensity.
            float4 fftData = tex2D<float4>(world.nishita.cloud_fft_map, u, v);
            
            // ADDITIVE BIAS instead of multiplicative
            // This preserves existing clouds but adds FFT wave influence
            // FFT height is usually -2 to +2. We scale it down.
            float fftBias = fftData.x * 0.15f; 
            
            effectiveCoverage = fmaxf(0.0f, fminf(1.0f, effectiveCoverage + fftBias));
        }

        float rawDensity = cloud_shape(noisePos, effectiveCoverage);
        
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
            float phase1 = (1.0f - g * g) / (4.0f * 3.14159f * powf(1.0f + g * g - 2.0f * g * cosTheta, 1.5f));
            
            // Lobe 2: Backward scattering (softer peak opposite to sun)
            float g2 = -0.4f; // Typical back-scatter value for clouds
            float phase2 = (1.0f - g2 * g2) / (4.0f * 3.14159f * powf(1.0f + g2 * g2 - 2.0f * g2 * cosTheta, 1.5f));
            
            // Mix lobes (mostly forward, but some backward to light up the front of clouds)
            float phase = lerp(phase2, phase1, 0.7f); // 70% forward, 30% backward
            
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
            float3 lightColor = directLight + ambient;
            
            float3 stepColor = lightColor * density;
            float absorption = density * stepSize * 0.012f * world.nishita.cloud_absorption;
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
        float scale = 0.003f / fmaxf(0.1f, world.nishita.cloud_scale);
        float3 layer1 = render_cloud_layer(
            world, rayDir, bg_color,
            world.nishita.cloud_height_min, world.nishita.cloud_height_max,
            scale, world.nishita.cloud_coverage, world.nishita.cloud_density,
            transmittance
        );
        cloudColor += layer1;
    }
    
    // === LAYER 2 (Secondary clouds - e.g., high cirrus) ===
    if (world.nishita.cloud_layer2_enabled) {
        float scale2 = 0.003f / fmaxf(0.1f, world.nishita.cloud2_scale);
        float3 layer2 = render_cloud_layer(
            world, rayDir, bg_color,
            world.nishita.cloud2_height_min, world.nishita.cloud2_height_max,
            scale2, world.nishita.cloud2_coverage, world.nishita.cloud2_density,
            transmittance
        );
        cloudColor += layer2;
    }
    
    // Final blend with background
    // No artificial minimum - clouds should be fully transparent where there's no density
    return bg_color * transmittance + cloudColor;
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
// Forward declaration for shadow test
__device__ bool trace_shadow_test(float3 origin, float3 direction, float maxDist);

__device__ float3 calculate_volumetric_god_rays(
    const WorldData& world,
    float3 rayOrigin,
    float3 rayDir,
    float maxDistance,  // Distance to first hit (or far distance for miss)
    curandState* rng
) {
    if (!world.nishita.godrays_enabled || world.nishita.godrays_intensity <= 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    float3 sunDir = normalize(world.nishita.sun_direction);
    float sunDot = dot(rayDir, sunDir);
    
    // Early exit if not looking somewhat towards sun
    if (sunDot < 0.3f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    // Ray march parameters from world settings
    int numSteps = world.nishita.godrays_samples;
    numSteps = max(4, min(numSteps, 64)); // Clamp between 4-64
    
    // Limit march distance for performance
    float marchDistance = fminf(maxDistance, world.nishita.fog_distance * 0.5f);
    float stepSize = marchDistance / (float)numSteps;
    
    float3 godRayColor = make_float3(0.0f, 0.0f, 0.0f);
    float transmittance = 1.0f;
    
    // Sun color based on elevation
    float sunElevation = sunDir.y;
    float3 sunColor;
    if (sunElevation < 0.1f) {
        sunColor = make_float3(1.0f, 0.5f, 0.2f); // Orange at sunset
    } else if (sunElevation < 0.3f) {
        sunColor = make_float3(1.0f, 0.8f, 0.5f); // Golden
    } else {
        sunColor = make_float3(1.0f, 0.95f, 0.9f); // Warm white
    }
    
    // Phase function (Henyey-Greenstein for forward scattering)
    float g = 0.85f;
    float g2 = g * g;
    float phase = (1.0f - g2) / (4.0f * M_PIf * powf(1.0f + g2 - 2.0f * g * sunDot, 1.5f));
    
    // Elevation factor (stronger at low sun angles)
    float elevationFactor = 1.0f - fminf(1.0f, fmaxf(0.0f, sunElevation * 3.0f));
    elevationFactor = fmaxf(0.1f, elevationFactor);
    
    // Media density
    float mediaDensity = world.nishita.godrays_density * 0.01f;
    
    // Jitter start position to reduce banding
    float jitter = 0.0f;
    if (rng) {
        jitter = curand_uniform(rng) * stepSize;
    }
    
    for (int i = 0; i < numSteps; ++i) {
        float t = jitter + stepSize * (float)i + stepSize * 0.5f;
        if (t > marchDistance) break;
        
        float3 samplePos = rayOrigin + rayDir * t;
        
        // Height-based density falloff
        float height = samplePos.y;
        float heightFactor = expf(-fmaxf(0.0f, height) * 0.0005f); // Fog denser near ground
        
        // Check if this point is lit by the sun (not in shadow)
        bool inSunlight = !trace_shadow_test(samplePos, sunDir, 10000.0f);
        
        if (inSunlight) {
            // Accumulate god ray contribution
            float localDensity = mediaDensity * heightFactor;
            float scattering = localDensity * phase * world.nishita.godrays_intensity;
            
            // Distance-based decay
            float decay = powf(world.nishita.godrays_decay, t * 0.01f);
            
            // Add contribution
            godRayColor += sunColor * scattering * transmittance * decay * stepSize * elevationFactor;
        }
        
        // Absorption through the medium
        transmittance *= expf(-mediaDensity * heightFactor * stepSize * 0.5f);
        
        // Early exit if fully absorbed
        if (transmittance < 0.01f) break;
    }
    
    // Final intensity scaling
    return godRayColor * world.nishita.sun_intensity * 0.5f;
}

// Shadow test for god rays
__device__ bool trace_shadow_test(float3 origin, float3 direction, float maxDist) {
    Ray shadow_ray(origin, direction);
    
    OptixHitResult shadow_payload = {};
    trace_shadow_ray(shadow_ray, &shadow_payload, 0.1f, maxDist);
    
    return shadow_payload.hit;
}

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

__device__ float3 evaluate_background(const WorldData& world, const float3& dir) {
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
        // Nishita Sky Model (Single Scattering) - Blender compatible
        float3 sunDir = normalize(world.nishita.sun_direction);
        float Rg = world.nishita.planet_radius;                     // Ground radius (meters)
        float Rt = Rg + world.nishita.atmosphere_height;            // Top of atmosphere (meters)
        
        // Camera position with altitude (meters)
        float3 camPos = make_float3(0.0f, Rg + world.nishita.altitude, 0.0f);
        float3 rayDir = normalize(dir);
        
        // Ray-Sphere Intersection (Atmosphere)
        float a = dot(rayDir, rayDir);
        float b = 2.0f * dot(rayDir, camPos);
        float c = dot(camPos, camPos) - Rt * Rt;
        float delta = b * b - 4.0f * a * c;
        
        if (delta < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
        
        float t1 = (-b - sqrtf(delta)) / (2.0f * a);
        float t2 = (-b + sqrtf(delta)) / (2.0f * a);
        float t = (t1 >= 0.0f) ? t1 : t2;
        if (t < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
        
        int numSamples = 8;
        float stepSize = t / (float)numSamples;
        
        float3 totalRayleigh = make_float3(0.0f, 0.0f, 0.0f);
        float3 totalMie = make_float3(0.0f, 0.0f, 0.0f);
        
        float opticalDepthRayleigh = 0.0f;
        float opticalDepthMie = 0.0f;
        
        // Phase Functions
        float mu = dot(rayDir, sunDir);
        float phaseR = 3.0f / (16.0f * M_PIf) * (1.0f + mu * mu);
        float g = world.nishita.mie_anisotropy;
        float phaseM = 3.0f / (8.0f * M_PIf) * ((1.0f - g * g) * (1.0f + mu * mu)) / ((2.0f + g * g) * powf(1.0f + g * g - 2.0f * g * mu, 1.5f));
        
        float currentT = 0.0f;
        
        for (int i = 0; i < numSamples; ++i) {
            float3 samplePos = camPos + rayDir * (currentT + stepSize * 0.5f);
            float height = length(samplePos) - Rg;
            if (height < 0.0f) height = 0.0f;
            
            float hr = expf(-height / world.nishita.rayleigh_density);
            float hm = expf(-height / world.nishita.mie_density);
            
            opticalDepthRayleigh += hr * stepSize;
            opticalDepthMie += hm * stepSize;
            
            // Optical depth to sun
            float b_light = 2.0f * dot(sunDir, samplePos);
            float c_light = dot(samplePos, samplePos) - Rt * Rt;
            float delta_light = b_light * b_light - 4.0f * c_light;
            
            if (delta_light >= 0.0f) {
                float t_light = (-b_light + sqrtf(delta_light)) / 2.0f;
                
                int numLightSamples = 4;
                float lightStep = t_light / (float)numLightSamples;
                float lightOpticalRayleigh = 0.0f;
                float lightOpticalMie = 0.0f;
                
                for(int j=0; j<numLightSamples; ++j) {
                    float3 lightSamplePos = samplePos + sunDir * (lightStep * (j + 0.5f));
                    float lightHeight = length(lightSamplePos) - Rg;
                    if(lightHeight < 0.0f) lightHeight = 0.0f;
                    
                    lightOpticalRayleigh += expf(-lightHeight / world.nishita.rayleigh_density) * lightStep;
                    lightOpticalMie += expf(-lightHeight / world.nishita.mie_density) * lightStep;
                }
                
                // Apply air and dust density multipliers
                float3 rayleighScatter = world.nishita.rayleigh_scattering * world.nishita.air_density;
                float3 mieScatter = world.nishita.mie_scattering * world.nishita.dust_density;
                
                float3 tau = rayleighScatter * (opticalDepthRayleigh + lightOpticalRayleigh) + 
                             mieScatter * 1.1f * (opticalDepthMie + lightOpticalMie);
                             
                float3 attenuation = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
                
                totalRayleigh += attenuation * hr * stepSize;
                totalMie += attenuation * hm * stepSize;
            }
            currentT += stepSize;
        }
        
        // Apply air and dust density multipliers
        float3 rayleighScatter = world.nishita.rayleigh_scattering * world.nishita.air_density;
        float3 mieScatter = world.nishita.mie_scattering * world.nishita.dust_density;
        
        float3 L = (totalRayleigh * rayleighScatter * phaseR + 
                totalMie * mieScatter * phaseM) * world.nishita.sun_intensity;
        
        // Apply ozone (affects blue channel - simple approximation)
        float ozoneFactor = world.nishita.ozone_density;
        L.x *= (1.0f + 0.1f * ozoneFactor);   // Slightly affect red
        L.z *= (1.0f + 0.3f * ozoneFactor);   // Boost blue

        // Add Sun Disk using sun_size (in degrees)
        // Apply horizon magnification effect
        float sunSizeDeg = world.nishita.sun_size;
        float elevationFactor = 1.0f;
        if (world.nishita.sun_elevation < 15.0f) {
            elevationFactor = 1.0f + (15.0f - fmaxf(world.nishita.sun_elevation, -10.0f)) * 0.04f;
        }
        sunSizeDeg *= elevationFactor;
        
        float sun_radius = sunSizeDeg * (M_PIf / 180.0f) * 0.5f; // Half angle in radians
        if (dot(rayDir, sunDir) > cosf(sun_radius)) {
            // Transmittance to space (accumulated optical depth)
            float3 tau = rayleighScatter * opticalDepthRayleigh + 
                         mieScatter * 1.1f * opticalDepthMie;
            float3 transmittance = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
            L += transmittance * world.nishita.sun_intensity * 1000.0f; // Direct sun
        }
        
        // ═══════════════════════════════════════════════════════════
        // REALISTIC NIGHT SKY with TWILIGHT GRADIENT
        // ═══════════════════════════════════════════════════════════
        float nightVal = (world.nishita.sun_elevation + 5.0f) / 20.0f;
        float nightFactor = 1.0f - fminf(1.0f, fmaxf(0.0f, nightVal));
        
        // ════════════════════════════════════════════════════════════════
        // TWILIGHT HORIZON GRADIENT (Blue-Orange-Purple)
        // Civil twilight: -6°, Nautical: -12°, Astronomical: -18°
        // ════════════════════════════════════════════════════════════════
        if (world.nishita.sun_elevation < 5.0f && world.nishita.sun_elevation > -18.0f) {
            float twilightPhase = (-world.nishita.sun_elevation + 5.0f) / 23.0f; // 0 at sunset, 1 at -18°
            twilightPhase = fminf(1.0f, fmaxf(0.0f, twilightPhase));
            
            float3 sunDir = normalize(world.nishita.sun_direction);
            float sunAngle = dot(rayDir, sunDir);
            float horizonFade = expf(-fabsf(rayDir.y) * 4.0f); // Strong at horizon
            float sunProximity = fmaxf(0.0f, sunAngle);
            
            // Twilight colors based on phase
            float3 horizonColor;
            if (twilightPhase < 0.3f) {
                // Civil twilight - Orange/Red at sun, Blue away
                float3 sunSideColor = make_float3(1.0f, 0.5f, 0.2f);  // Orange
                float3 awaySideColor = make_float3(0.3f, 0.4f, 0.7f); // Deep blue
                horizonColor = lerp(awaySideColor, sunSideColor, powf(sunProximity, 2.0f));
            } else if (twilightPhase < 0.6f) {
                // Nautical twilight - Purple/Deep blue gradient
                float3 sunSideColor = make_float3(0.6f, 0.3f, 0.4f);  // Purple-red
                float3 awaySideColor = make_float3(0.15f, 0.2f, 0.5f); // Navy blue
                horizonColor = lerp(awaySideColor, sunSideColor, powf(sunProximity, 3.0f));
            } else {
                // Astronomical twilight - Dark blue with hint of purple
                float3 sunSideColor = make_float3(0.2f, 0.15f, 0.35f);  // Dark purple
                float3 awaySideColor = make_float3(0.05f, 0.08f, 0.2f); // Near-black blue
                horizonColor = lerp(awaySideColor, sunSideColor, powf(sunProximity, 4.0f));
            }
            
            // Blend intensity based on twilight depth
            float twilightIntensity = (1.0f - twilightPhase) * 0.4f;
            L += horizonColor * horizonFade * twilightIntensity;
        }
        
        // ════════════════════════════════════════════════════════════════
        // ENVIRONMENT TEXTURE OVERLAY (HDR/EXR blending with procedural)
        // ════════════════════════════════════════════════════════════════
        if (world.nishita.env_overlay_enabled && world.nishita.env_overlay_tex != 0) {
            float3 starDir = normalize(rayDir);
            
            // Apply rotation to UV calculation
            float rotation = world.nishita.env_overlay_rotation * M_PIf / 180.0f;
            float sinR = sinf(rotation);
            float cosR = cosf(rotation);
            
            // Rotated direction
            float3 rotatedDir = make_float3(
                starDir.x * cosR - starDir.z * sinR,
                starDir.y,
                starDir.x * sinR + starDir.z * cosR
            );
            
            // Convert direction to equirectangular UV
            float u = 0.5f + atan2f(rotatedDir.z, rotatedDir.x) / (2.0f * M_PIf);
            float v = 0.5f - asinf(fmaxf(-1.0f, fminf(1.0f, rotatedDir.y))) / M_PIf;
            
            // Sample environment texture
            float4 envSample = tex2D<float4>(world.nishita.env_overlay_tex, u, v);
            float3 envColor = make_float3(envSample.x, envSample.y, envSample.z);
            envColor = envColor * world.nishita.env_overlay_intensity;
            
            // Blend mode application
            int blendMode = world.nishita.env_overlay_blend_mode;
            float blendFactor = fminf(1.0f, world.nishita.env_overlay_intensity);
            
            if (blendMode == 0) {
                // Mix - lerp between Nishita and texture
                L = L * (1.0f - blendFactor) + envColor;
            } else if (blendMode == 1) {
                // Multiply - use env as multiplier (normalized around 1.0)
                float3 normalizedEnv = envColor / fmaxf(0.001f, world.nishita.env_overlay_intensity);
                L = L * (normalizedEnv * 0.5f + make_float3(0.5f, 0.5f, 0.5f));
            } else if (blendMode == 2) {
                // Screen - brightens without washing out
                L = make_float3(1.0f, 1.0f, 1.0f) - (make_float3(1.0f, 1.0f, 1.0f) - L) * (make_float3(1.0f, 1.0f, 1.0f) - envColor);
            } else if (blendMode == 3) {
                // Replace - environment texture ONLY (ignore Nishita completely)
                L = envColor;
            }
        }
        
        // ════════════════════════════════════════════════════════════════
        // NIGHT SKY AMBIENT (when dark, no texture overlay)
        // ════════════════════════════════════════════════════════════════
        if (nightFactor > 0.3f && !world.nishita.env_overlay_enabled) {
            // Deep blue night sky base (never pure black)
            float3 nightSkyBase = make_float3(0.01f, 0.015f, 0.04f);
            L += nightSkyBase * nightFactor;
            
            // Light pollution gradient at horizon
            float lightPollution = expf(-fabsf(rayDir.y) * 6.0f);
            L += make_float3(0.02f, 0.015f, 0.01f) * lightPollution * nightFactor * 0.3f;
        }
        
        // ═══════════════════════════════════════════════════════════════
        // MULTI-SCATTERING - Brightens horizon and makes sky more uniform
        // ═══════════════════════════════════════════════════════════════
        if (world.nishita.multi_scatter_enabled && world.nishita.multi_scatter_factor > 0.0f) {
            // Use average optical depth for multi-scatter calculation
            float avgOpticalDepth = (opticalDepthRayleigh + opticalDepthMie) * 0.5f;
            
            // Scattering albedo (how much light is scattered vs absorbed)
            float3 scatterAlbedo = make_float3(0.8f, 0.85f, 0.9f);
            
            L = apply_multi_scattering(L, avgOpticalDepth, scatterAlbedo, world.nishita.multi_scatter_factor);
        }
        
        // NOTE: Volumetric god rays are now handled in ray_color() with proper occlusion
        // They need access to scene geometry for shadow testing
        
        // Global volumetric clouds handled by return statement below
        return render_clouds(world, dir, L);
    }
    return make_float3(0.0f, 0.0f, 0.0f);
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
    const float shadow_bias = 1e-2f;

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
    
    trace_shadow_ray(shadow_ray, &shadow_payload, 0.01f, distance);
    if (shadow_payload.hit) return make_float3(0.0f, 0.0f, 0.0f);

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
    float3 Li = light.color * light.intensity * attenuation;
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
        float3 jitter = light.radius * random_in_unit_sphere(rng);
        wi = normalize(light.direction + jitter);
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

// GPU helper: Dual-lobe Henyey-Greenstein phase function
__device__ float gpu_phase_dual_hg(float cos_theta, float g_forward, float g_back, float lobe_mix) {
    // Forward lobe
    float g2_fwd = g_forward * g_forward;
    float phase_fwd = (1.0f - g2_fwd) / (4.0f * 3.14159f * powf(1.0f + g2_fwd - 2.0f * g_forward * cos_theta, 1.5f));
    
    // Backward lobe
    float g2_back = g_back * g_back;
    float phase_back = (1.0f - g2_back) / (4.0f * 3.14159f * powf(1.0f + g2_back - 2.0f * g_back * cos_theta, 1.5f));
    
    return lobe_mix * phase_fwd + (1.0f - lobe_mix) * phase_back;
}

// GPU helper: Multi-scatter transmittance approximation
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
    // Multi-scattering parameters (NEW)
    float multi_scatter,
    float g_back,
    float lobe_mix,
    int light_steps,
    float shadow_strength,
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
    float transmittance = 1.0f;
    
    // Adaptive step size based on volume size
    float volume_size = length(aabb_max - aabb_min);
    float actual_step_size = fmaxf(step_size, volume_size / (float)max_steps);
    
    // Temporal jitter to reduce banding
    float jitter = random_float(rng) * actual_step_size;
    float t = t_enter + jitter;
    int steps = 0;
    
    // Get sun direction for lighting
    float3 sun_dir = normalize(optixLaunchParams.world.nishita.sun_direction);
    float sun_intensity = optixLaunchParams.world.nishita.sun_intensity;
    
    // Precompute light march step size
    float light_step_size = volume_size / fmaxf((float)light_steps, 1.0f);
    
    while (t < t_exit && steps < max_steps && transmittance > 0.01f) {
        float3 pos = ray_origin + ray_dir * t;
        
        // Density at this point - start with base density
        float local_density = vol_density;
        
        // Apply procedural noise modulation if enabled
        if (noise_scale > 0.01f) {
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
            // This creates variations without cutting content
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
                float t = d_edge / falloff_dist;
                // Smoothstep: 3t^2 - 2t^3
                edge_factor = t * t * (3.0f - 2.0f * t);
            }
            
            local_density *= edge_factor;
        }
        
        if (local_density > 0.001f) {
            // Compute extinction coefficient
            float sigma_a = local_density * vol_absorption;
            float sigma_s = local_density * vol_scattering;
            float sigma_t = sigma_a + sigma_s;
            
            // Albedo average for multi-scatter
            float albedo_avg = (vol_albedo.x + vol_albedo.y + vol_albedo.z) / 3.0f;
            
            // Multi-scatter transmittance
            float step_transmittance = gpu_multiscatter_transmittance(sigma_t, actual_step_size, multi_scatter, albedo_avg);
            
            // ═══════════════════════════════════════════════════════════
            // LIGHT MARCHING (Self-Shadowing)
            // ═══════════════════════════════════════════════════════════
            float light_transmittance = 1.0f;
            if (light_steps > 0) {
                float density_accum = 0.0f;
                for (int j = 1; j <= light_steps; ++j) {
                    float3 light_pos = pos + sun_dir * (light_step_size * (float)j);
                    
                    // Check if still in AABB
                    if (light_pos.x < aabb_min.x || light_pos.x > aabb_max.x ||
                        light_pos.y < aabb_min.y || light_pos.y > aabb_max.y ||
                        light_pos.z < aabb_min.z || light_pos.z > aabb_max.z) {
                        break;
                    }
                    
                    // Simple density sample (skip noise for performance)
                    float light_density = vol_density;
                    density_accum += light_density * vol_absorption * light_step_size;
                    
                    if (density_accum > 5.0f) break;
                }
                
                // Beer's Law + multi-scatter
                float beers = expf(-density_accum);
                float beers_soft = expf(-density_accum * 0.25f);
                light_transmittance = beers * (1.0f - multi_scatter * albedo_avg) + 
                                      beers_soft * multi_scatter * albedo_avg;
                light_transmittance = 1.0f - shadow_strength * (1.0f - light_transmittance);
            }
            
            // ═══════════════════════════════════════════════════════════
            // IN-SCATTERING (Dual-lobe phase + powder effect)
            // ═══════════════════════════════════════════════════════════
            float cos_theta = dot(ray_dir, sun_dir);
            
            // Dual-lobe Henyey-Greenstein phase function
            float phase = gpu_phase_dual_hg(cos_theta, vol_g, g_back, lobe_mix);
            
            // Powder effect
            float powder = gpu_powder_effect(local_density, cos_theta);
            
            // Light contribution with self-shadowing
            float3 Li = make_float3(sun_intensity, sun_intensity, sun_intensity);
            float3 inscatter = vol_albedo * Li * phase * sigma_s * light_transmittance;
            inscatter = inscatter * (1.0f + powder * 0.5f);
            
            // Emission
            float3 emit = vol_emission;
            
            // Accumulate
            float3 step_color = (inscatter + emit) * transmittance * (1.0f - step_transmittance);
            accumulated_color += step_color;
            
            transmittance *= step_transmittance;
        }
        
        t += actual_step_size;
        steps++;
    }
    
    out_transmittance = transmittance;
    return accumulated_color;
}

__device__ float3 ray_color(Ray ray, curandState* rng) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float3 current_medium_absorb = make_float3(0.0f, 0.0f, 0.0f); // Default: Air (no absorption)

    const int max_depth = optixLaunchParams.max_depth;
    int light_count = optixLaunchParams.light_count;
    int light_index = (light_count > 0) ? pick_smart_light(ray.origin, rng) : -1;
    
    // Firefly önleme için maksimum katkı limiti
    const float MAX_CONTRIBUTION = 100.0f;

    for (int bounce = 0; bounce < max_depth; ++bounce) {

        OptixHitResult payload = {};
        
        // Use Viewport Clipping for primary rays
        float t_min = (bounce == 0) ? optixLaunchParams.clip_near : 0.01f;
        float t_max = (bounce == 0) ? optixLaunchParams.clip_far : 1e16f;
        
        trace_ray(ray, &payload, t_min, t_max);

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
            float maxDist = payload.hit ? payload.t : 10000.0f;
            float3 godRayContribution = calculate_volumetric_god_rays(
                optixLaunchParams.world,
                ray.origin,
                normalize(ray.direction),
                maxDist,
                rng
            );
            color += godRayContribution;
        }

        if (!payload.hit) {
            // --- Arka plan rengi ---
            float3 bg_color = evaluate_background(optixLaunchParams.world, ray.direction);

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
                         float fade_start = 5.0f;
                         float fade_end = optixLaunchParams.grid_fade_distance;
                         float alpha = 1.0f - fminf(fmaxf((dist - fade_start) / (fade_end - fade_start), 0.0f), 1.0f);
                         
                         bg_color = bg_color * (1.0f - alpha) + grid_color * alpha;
                     }
                 }
            }
           
            // Bounce bazlı arka plan azaltma - ilk bounce tam, sonrakiler azaltılmış
            // Bu, yansımalarda arka plan renginin yüzeyleri boyamasını önler
            float bg_factor = (bounce == 0) ? 1.0f : fmaxf(0.1f, 1.0f / (1.0f + bounce * 0.5f));
            float3 bg_contribution = bg_color * bg_factor;
            
            color += throughput * bg_contribution;
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
                // Multi-scattering parameters (NEW)
                payload.vol_multi_scatter,
                payload.vol_g_back,
                payload.vol_lobe_mix,
                payload.vol_light_steps,
                payload.vol_shadow_strength,
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
            Ray exit_ray(payload.position + ray.direction * 0.01f, ray.direction);
            OptixHitResult exit_payload = {};
            trace_ray(exit_ray, &exit_payload,t_min,t_max);
            
            if (exit_payload.hit) {
                scattered = Ray(exit_payload.position + ray.direction * 0.01f, ray.direction);
            } else {
                // Ray exited scene through volume
                float3 bg_color = evaluate_background(optixLaunchParams.world, ray.direction);
                color += throughput * bg_color;
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
                if (dot(wi, L) > 0.999f) {
                    float solid_angle = 2.0f * M_PIf * (1.0f - cos(atan2(light.radius, 1000.0f)));
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
    // POST-PROCESS: Atmospheric Fog (applied to final result)
    // ═══════════════════════════════════════════════════════════
    // Fog is applied to the accumulated color based on scene depth
    // This creates depth-based atmospheric perspective
    const WorldData& world = optixLaunchParams.world;
    if (world.nishita.fog_enabled && world.nishita.fog_density > 0.0f) {
        // Use a reasonable default distance for miss rays (far away)
        // For proper fog, we need first hit distance - approximated here
        float fogDistance = world.nishita.fog_distance * 0.8f; // Assume far objects
        
        // Camera position for fog calculation
        float3 rayOrigin = make_float3(0.0f, world.nishita.altitude, 0.0f);
        float3 rayDir = normalize(ray.direction);
        
        // Calculate fog factor with height falloff
        float fogFactor = calculate_height_fog_factor(
            rayOrigin, rayDir, fogDistance,
            world.nishita.fog_density,
            world.nishita.fog_height,
            world.nishita.fog_falloff
        );
        
        // Get fog color with sun scattering
        float3 fogColor = world.nishita.fog_color;
        float3 sunDir = normalize(world.nishita.sun_direction);
        float sunDot = fmaxf(0.0f, dot(rayDir, sunDir));
        float sunScatter = powf(sunDot, 8.0f) * world.nishita.fog_sun_scatter;
        float3 sunColor = make_float3(1.0f, 0.9f, 0.7f) * world.nishita.sun_intensity * 0.05f;
        fogColor = fogColor + sunColor * sunScatter;
        
        // Blend final color with fog
        color = lerp(color, fogColor, fogFactor * 0.5f); // 50% max fog to preserve detail
    }

    // Final clamp - NaN ve Inf kontrolü
    color.x = isfinite(color.x) ? fminf(fmaxf(color.x, 0.0f), 100.0f) : 0.0f;
    color.y = isfinite(color.y) ? fminf(fmaxf(color.y, 0.0f), 100.0f) : 0.0f;
    color.z = isfinite(color.z) ? fminf(fmaxf(color.z, 0.0f), 100.0f) : 0.0f;

    return color;
}
