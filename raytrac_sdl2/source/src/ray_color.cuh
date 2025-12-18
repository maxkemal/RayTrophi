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
    if (t_enter < 0.0f) t_enter = 0.0f;
    
    // Horizon fade
    float h_val = rayDir.y / 0.15f;
    float h_t = fmaxf(0.0f, fminf(1.0f, fabsf(h_val)));
    float horizonFade = h_t * h_t * (3.0f - 2.0f * h_t);
    
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
        float heightGradient = 4.0f * heightFraction * (1.0f - heightFraction);
        heightGradient = fmaxf(0.0f, fminf(1.0f, heightGradient));
        
        float3 offsetPos = pos + make_float3(world.nishita.cloud_offset_x, 0.0f, world.nishita.cloud_offset_z);
        
        // Use 3D texture if available, otherwise fallback to procedural
        // FORCE CINEMATIC QUALITY (Procedural)
        // This provides infinite detail and better quality than the 256^3 texture
        float3 noisePos = offsetPos * scale;
        float rawDensity = cloud_shape(noisePos, coverage);
        

        float density = rawDensity * heightGradient;
        
        if (density > 0.003f) {
            density *= localDensityMult;
            
            // ═══════════════════════════════════════════════════════════
            // LIGHT MARCHING (Self-Shadowing) - Controllable via UI
            // ═══════════════════════════════════════════════════════════
            float lightTransmittance = 1.0f;
            int lightSteps = world.nishita.cloud_light_steps;  // UI controlled
            
            if (lightSteps > 0 && sunDirection.y > 0.01f) {
                float lightStepSize = (cloudMaxY - pos.y) / fmaxf(0.01f, sunDirection.y) / (float)lightSteps;
                lightStepSize = fminf(lightStepSize, 500.0f);
                
                for (int j = 1; j <= lightSteps; ++j) {
                    float3 lightPos = pos + sunDirection * (lightStepSize * (float)j);
                    
                    if (lightPos.y > cloudMaxY || lightPos.y < cloudMinY) break;
                    
                    float3 lightOffsetPos = lightPos + make_float3(world.nishita.cloud_offset_x, 0.0f, world.nishita.cloud_offset_z);
                    
                    // Use texture LOD for light marching
                    // Use faster procedural noise for shadows to save performance
                    // Shadows don't need the micro-details of the main shape
                    float3 lightNoisePos = lightOffsetPos * scale;
                    float lightDensity = fast_cloud_shape(lightNoisePos, coverage);


                    
                    float lh = (lightPos.y - cloudMinY) / (cloudMaxY - cloudMinY);
                    float lightHeightGrad = 4.0f * lh * (1.0f - lh);
                    lightDensity *= lightHeightGrad * localDensityMult;
                    
                    // Use absorption parameter
                    lightTransmittance *= expf(-lightDensity * lightStepSize * 0.015f * world.nishita.cloud_absorption);
                    
                    if (lightTransmittance < 0.05f) break;
                }
            }
            
            // Apply shadow strength (UI controlled)
            lightTransmittance = 1.0f - (1.0f - lightTransmittance) * world.nishita.cloud_shadow_strength;
            
            // ═══════════════════════════════════════════════════════════
            // ADVANCED COLOR CALCULATION
            // ═══════════════════════════════════════════════════════════
            float cosTheta = dot(rayDir, sunDirection);
            
            // Henyey-Greenstein phase function
            float phase = (1.0f - g * g) / (4.0f * 3.14159f * powf(1.0f + g * g - 2.0f * g * cosTheta, 1.5f));
            
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
        // Night Sky: Stars and Moon (visible when sun is low)
        // ═══════════════════════════════════════════════════════════
        float nightVal = (world.nishita.sun_elevation + 5.0f) / 20.0f;
        float nightFactor = 1.0f - fminf(1.0f, fmaxf(0.0f, nightVal));
        
        if (nightFactor > 0.01f && world.nishita.stars_intensity > 0.0f) {
            // Procedural stars using hash
            float3 starDir = rayDir;
            // Grid-based star positions
            float starScale = 1000.0f;
            float3 gridPos = make_float3(
                floorf(starDir.x * starScale),
                floorf(starDir.y * starScale),
                floorf(starDir.z * starScale)
            );
            
            // Simple hash function for star placement
            float hash = fmodf(sinf(gridPos.x * 12.9898f + gridPos.y * 78.233f + gridPos.z * 45.164f) * 43758.5453f, 1.0f);
            if (hash < 0.0f) hash = -hash;
            
            // Only some cells have stars
            float starThreshold = 1.0f - world.nishita.stars_density * 0.1f;
            if (hash > starThreshold && rayDir.y > 0.0f) {
                // Star color (slightly varied)
                float colorVar = fmodf(hash * 7.0f, 1.0f);
                float3 starColor;
                if (colorVar < 0.3f) {
                    starColor = make_float3(1.0f, 0.9f, 0.8f); // Warm white
                } else if (colorVar < 0.6f) {
                    starColor = make_float3(0.9f, 0.95f, 1.0f); // Cool white
                } else if (colorVar < 0.8f) {
                    starColor = make_float3(1.0f, 0.7f, 0.5f); // Orange
                } else {
                    starColor = make_float3(0.7f, 0.8f, 1.0f); // Blue
                }
                
                // Twinkle effect based on hash
                float twinkle = 0.5f + 0.5f * sinf(hash * 100.0f);
                float starBrightness = (hash - starThreshold) / (1.0f - starThreshold);
                starBrightness = powf(starBrightness, 2.0f) * twinkle;
                
                L += starColor * starBrightness * world.nishita.stars_intensity * nightFactor * 0.1f;
            }
        }
        
        // Moon rendering
        if (world.nishita.moon_enabled && nightFactor > 0.01f && world.nishita.moon_intensity > 0.0f) {
            float moonElevRad = world.nishita.moon_elevation * M_PIf / 180.0f;
            float moonAzimRad = world.nishita.moon_azimuth * M_PIf / 180.0f;
            float3 moonDir = make_float3(
                cosf(moonElevRad) * sinf(moonAzimRad),
                sinf(moonElevRad),
                cosf(moonElevRad) * cosf(moonAzimRad)
            );
            
            // Horizon magnification effect (like sun)
            float moonSizeDeg = world.nishita.moon_size;
            float moonElevFactor = 1.0f;
            if (world.nishita.moon_elevation < 15.0f) {
                moonElevFactor = 1.0f + (15.0f - fmaxf(world.nishita.moon_elevation, -10.0f)) * 0.04f;
            }
            moonSizeDeg *= moonElevFactor;
            
            float moon_radius = moonSizeDeg * (M_PIf / 180.0f) * 0.5f;
            float moonDot = dot(rayDir, moonDir);
            
            if (moonDot > cosf(moon_radius)) {
                // Base moon color (slightly blue-white)
                float3 moonColor = make_float3(0.9f, 0.9f, 0.95f);
                
                // Horizon color shift (orange/red when low)
                if (world.nishita.moon_elevation < 20.0f) {
                    float horizonBlend = (20.0f - fmaxf(world.nishita.moon_elevation, -5.0f)) / 25.0f;
                    horizonBlend = fminf(1.0f, fmaxf(0.0f, horizonBlend));
                    // Shift to warm orange color
                    float3 horizonColor = make_float3(1.0f, 0.7f, 0.4f);
                    moonColor.x = moonColor.x * (1.0f - horizonBlend) + horizonColor.x * horizonBlend;
                    moonColor.y = moonColor.y * (1.0f - horizonBlend) + horizonColor.y * horizonBlend;
                    moonColor.z = moonColor.z * (1.0f - horizonBlend) + horizonColor.z * horizonBlend;
                }
                
                // Atmospheric dimming near horizon
                float atmosphericDim = 1.0f;
                if (world.nishita.moon_elevation < 10.0f) {
                    atmosphericDim = 0.3f + 0.7f * fmaxf(0.0f, world.nishita.moon_elevation / 10.0f);
                }
                
                // Phase shading (0 = new moon, 0.5 = full, 1 = new)
                float phase = world.nishita.moon_phase;
                float phaseFactor = fabsf(phase - 0.5f) * 2.0f;
                float brightness = 1.0f - phaseFactor * 0.9f;
                
                L += moonColor * brightness * atmosphericDim * world.nishita.moon_intensity * nightFactor * 10.0f;
            }
        }
        
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

__device__ float3 ray_color(Ray ray, curandState* rng) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    const int max_depth = optixLaunchParams.max_depth;
    int light_count = optixLaunchParams.light_count;
    int light_index = (light_count > 0) ? pick_smart_light(ray.origin, rng) : -1;
    
    // Firefly önleme için maksimum katkı limiti
    const float MAX_CONTRIBUTION = 100.0f;

    for (int bounce = 0; bounce < max_depth; ++bounce) {

        OptixHitResult payload = {};
        trace_ray(ray, &payload);

        if (!payload.hit) {
            // --- Arka plan rengi ---
            float3 bg_color = evaluate_background(optixLaunchParams.world, ray.direction);
           
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

        // --- GPU VOLUMETRIC RENDERING (SMOKE) ---
        if (mat.anisotropic > 0.9f) { // Flagged as Volumetric
            // Ray Marching only if entering the volume
            if (dot(ray.direction, payload.normal) < 0.0f) {
                // Find exit point
                Ray march_ray(payload.position + ray.direction * 0.01f, ray.direction);
                OptixHitResult exit_payload = {};
                trace_ray(march_ray, &exit_payload);

                if (exit_payload.hit) {
                    float dist = length(exit_payload.position - payload.position);
                    // dist limiting to avoid infinite march
                    if(dist > 20.0f) dist = 20.0f; 

                    int steps = 12; // Low step count for performance
                    float step_size = dist / steps;
                    float3 current_pos = payload.position;
                    float total_density = 0.0f;

                    for (int i = 0; i < steps; i++) {
                        current_pos += ray.direction * step_size * (random_float(rng) * 0.5f + 0.75f); // Jittered step

                        // Simple procedural noise for smoke
                        float3 s = current_pos * 3.5f; 
                        float noise = fabsf(sinf(s.x) * sinf(s.y + s.z * 0.5f) * cosf(s.z));
                        float density = fmaxf(0.0f, noise - 0.2f) * 4.0f; // Hardcoded parameters matching "Test Smoke"
                        
                        // Fade edges (approximate based on assumptions, hard on pure ray march without SDF)
                        
                        total_density += density * step_size;
                    }
                    
                    // Beer's Law (Transmittance)
                    float3 volume_albedo = make_float3(0.8f, 0.8f, 0.8f); // Grey smoke
                    float absorption = 0.2f;
                    float3 transmittance = make_float3(
                        expf(-total_density * absorption * (1.0f - volume_albedo.x)),
                        expf(-total_density * absorption * (1.0f - volume_albedo.y)),
                        expf(-total_density * absorption * (1.0f - volume_albedo.z))
                    );

                    throughput *= transmittance;
                    
                    // Add some scattered light (ambient approximation)
                    // color += throughput * make_float3(0.05f) * total_density; 

                    // Skip the volume interior by moving ray to exit point
                    scattered = Ray(exit_payload.position + ray.direction * 0.001f, ray.direction);
                }
            }
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
        
        // --- Eğer hiç ışık yoksa sadece emissive katkı yap ---
        if (light_count == 0) {
            float3 emission = payload.emission;
            if (payload.has_emission_tex) {
                float4 tex = tex2D<float4>(payload.emission_tex, payload.uv.x, payload.uv.y);
                emission = make_float3(tex.x, tex.y, tex.z) * mat.emission;
            }
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
      
        float3 emission = payload.emission;
        
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

    // Final clamp - NaN ve Inf kontrolü
    color.x = isfinite(color.x) ? fminf(fmaxf(color.x, 0.0f), 100.0f) : 0.0f;
    color.y = isfinite(color.y) ? fminf(fmaxf(color.y, 0.0f), 100.0f) : 0.0f;
    color.z = isfinite(color.z) ? fminf(fmaxf(color.z, 0.0f), 100.0f) : 0.0f;

    return color;
}
