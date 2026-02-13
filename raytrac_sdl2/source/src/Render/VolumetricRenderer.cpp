/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          VolumetricRenderer.cpp
* =========================================================================
*/
#include "VolumetricRenderer.h"
#include "VDBVolume.h"
#include "VDBVolumeManager.h"
#include "GasVolume.h"
#include "VolumeShader.h"
#include "renderer.h" 
#include "globals.h"
#include <cmath>
#include <algorithm> // For std::min, std::max
#include "AtmosphereLUT.h"
// Helper for cloud transmittance (placeholder implementation based on original code)
static float determine_cloud_transmittance(const WorldData& world, const Vec3& origin, const Vec3& dir, float maxDist) {
    if (!world.nishita.clouds_enabled && !world.nishita.cloud_layer2_enabled) return 1.0f;
    if (std::abs(dir.y) < 1e-6f) return 1.0f;

    float cloud_trans = 1.0f;

    // Check layer 1
    if (world.nishita.clouds_enabled) {
        float minH = world.nishita.cloud_height_min;
        float maxH = world.nishita.cloud_height_max;
        float t0 = (minH - origin.y) / dir.y;
        float t1 = (maxH - origin.y) / dir.y;
        float t_enter = std::min(t0, t1);
        float t_exit = std::max(t0, t1);

        t_enter = std::max(t_enter, 0.0f);
        t_exit = std::min(t_exit, maxDist);

        if (t_exit > t_enter) {
            // CPU simplified check: just check middle point to save perf
            float mid_t = (t_enter + t_exit) * 0.5f;
            Vec3 p = origin + dir * mid_t;
            float scale = 0.003f / std::max(0.1f, world.nishita.cloud_scale);

            // Simple noise approximation
            float coverage = world.nishita.cloud_coverage;
            float val = (std::sin(p.x * scale * 0.5f) + std::sin(p.z * scale * 0.5f)) * 0.5f + 0.5f; // Very rough

            if (val < coverage) {
                float dist = t_exit - t_enter;
                cloud_trans *= std::exp(-dist * world.nishita.cloud_density * 0.5f);
            }
        }
    }
    return cloud_trans;
}

static Vec3 get_sun_color(float sun_elevation) {
    if (sun_elevation < -0.1f) {
        return Vec3(0.0f); // Night
    }
    else if (sun_elevation < 0.0f) {
        // Twilight fade
        float t = (sun_elevation + 0.1f) * 10.0f; 
        return Vec3(0.8f, 0.3f, 0.1f) * t;
    }
    else if (sun_elevation > 0.2f) {
         return Vec3(1.0f); // Day
    }
    else if (sun_elevation > 0.0f) {
        float t = sun_elevation / 0.2f;
        Vec3 orangeColor = Vec3(1.0f, 0.6f, 0.3f);
        Vec3 goldenColor = Vec3(1.0f, 0.85f, 0.6f);
        return orangeColor * (1.0f - t) + goldenColor * t;
    }
    else { // Just around horizon
        float t = std::max(0.0f, 1.0f + sun_elevation * 5.0f);
        Vec3 redColor = Vec3(0.8f, 0.3f, 0.1f);
        Vec3 orangeColor = Vec3(1.0f, 0.6f, 0.3f);
        return redColor * (1.0f - t) + orangeColor * t;
    }
}

static Vec3 to_vec3(const float3& v) { return Vec3(v.x, v.y, v.z); }

// ============================================================================
// GOD RAYS (CPU) - MATCHED TO GPU logic
// ============================================================================
Vec3 VolumetricRenderer::calculateGodRays(const SceneData& scene, const WorldData& world_data, const Ray& ray, float maxDistance, const Hittable* bvh, class AtmosphereLUT* lut) {
    if (world_data.mode != WORLD_MODE_NISHITA || !world_data.nishita.godrays_enabled || world_data.nishita.godrays_intensity <= 0.0f) {
        return Vec3(0.0f);
    }

    const auto& nishita = world_data.nishita;
    Vec3 sunDir = to_vec3(nishita.sun_direction).normalize();
    float sunDot = Vec3::dot(ray.direction, sunDir);

    // Threshold (Relaxed from 0.3 to 0.1 to see more rays at sunset)
    if (sunDot < 0.1f) {
        return Vec3(0.0f);
    }

    // Limits (Matching GPU march distance)
    float marchDistance = std::min(maxDistance, nishita.fog_distance * 0.5f);
    int num_steps = nishita.godrays_samples;
    num_steps = std::max(4, std::min(num_steps, 32)); // Clamped to 32 for CPU performance
    if (num_steps <= 0) return Vec3(0.0f);

    float step_size = marchDistance / (float)num_steps;
    Vec3 total_light(0.0f);
    float transmittance = 1.0f;

    // Jitter start
    float jitter = ((float)rand() / RAND_MAX) * step_size;

    // Elevation factor (Boosted for low angles)
    float sunElevation = sunDir.y;
    float elevationFactor = 1.0f - std::min(0.9f, std::max(0.0f, sunElevation * 2.5f));

    // Media density (Matching GPU)
    float mediaDensity = nishita.godrays_density * 0.01f;

    // --- OPTIMIZATION: HOIST IN VARIANTS OUT OF LOOP ---
    // Phase function is constant for the ray
    float mu = sunDot;
    float g = nishita.mie_anisotropy;
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * mu;
    if (denom < 0.0001f) denom = 0.0001f;
    float phase = (1.0f - g2) / (4.0f * 3.14159265f * pow(denom, 1.5f));

    // Sun Color from LUT (Physical Parity)
    Vec3 sun_color(1.0f);
    if (lut) {
        float3 sunTrans = lut->sampleTransmittance(std::max(0.01f, sunDir.y), nishita.altitude, nishita.atmosphere_height);
        sun_color = Vec3(sunTrans.x, sunTrans.y, sunTrans.z);
    } else {
        sun_color = get_sun_color(sunElevation);
    }

    for (int i = 0; i < num_steps; ++i) {
        float t = jitter + step_size * (float)i + step_size * 0.5f;
        if (t > marchDistance) break;

        Vec3 p = ray.at(t);
        
        // FIX: Distance-based attenuation to prevent near-camera halo (matches GPU)
        float distanceFromCamera = t;
        // VERY AGGRESSIVE FADE: 0.05 = full effect at ~20m (was 0.01 = 100m, still too slow)
        float distanceFade = 1.0f - std::exp(-distanceFromCamera * 0.05f);
        distanceFade = std::max(0.01f, distanceFade); // Minimum 1% to nearly eliminate near-camera
        
        float height = p.y;
        float heightFactor = exp(-std::max(0.0f, height) * 0.0005f); // Matching GPU height falloff

        float density = mediaDensity * heightFactor * distanceFade;

        if (density > 0.0001f) {
            float sunTrans = determineSunTransmittance(p, sunDir, 10000.0f, bvh, scene, world_data);

            // Add Cloud Occlusion Check
            if (sunTrans > 0.001f) {
                float cloudTrans = determine_cloud_transmittance(world_data, p, sunDir, 10000.0f);
                sunTrans *= cloudTrans;
            }

            if (sunTrans > 0.001f) {
                // Decay is distance dependent
              

                // Match GPU Intensity and attenuation
                float global_scale = 0.0002f; // Increased from 0.00005f to better match GPU
                Vec3 contribution = sun_color * (density * phase * nishita.godrays_intensity) * transmittance  * step_size * elevationFactor * sunTrans * nishita.sun_intensity * global_scale;

                if (std::isfinite(contribution.x)) {
                    total_light = total_light + contribution;
                }
            }
        }

        // Absorption through medium (Matching GPU)
        transmittance *= exp(-density * step_size * 0.5f);
        if (transmittance < 0.01f) break;
    }

    return total_light;
}

// ============================================================================
// SUN TRANSMITTANCE (CPU)
// ============================================================================
float VolumetricRenderer::determineSunTransmittance(const Vec3& origin, const Vec3& sunDir, float maxDist, const Hittable* bvh, const SceneData& scene, const WorldData& world_data) {
    // 1. Solid Shadow Test (with null check)
    HitRecord shadow_rec;
    Ray shadow_ray(origin + sunDir * 0.001f, sunDir);
    if (bvh && bvh->hit(shadow_ray, 0.001f, maxDist, shadow_rec, true)) {
        return 0.0f; // Fully blocked by solid
    }

    float transmittance = 1.0f;
    
    // 2. Volumetric Shadow Test (VDB & Gas)
    // Using approximated occlusion for performance in God Rays
    
    // For VDB Volumes
    for (const auto& vdb : scene.vdb_volumes) {
        if (!vdb || !vdb->visible) continue;
        
        float t0, t1;
        if (vdb->intersectTransformedAABB(shadow_ray, 0.001f, maxDist, t0, t1)) {
            // Simplified check: if ray passes through box, assume some attenuation
            // Full ray march is too expensive for CPU God Rays at 32 steps per pixel
            transmittance *= 0.9f; 
        }
    }
    
    return transmittance;
}

// ============================================================================
// GPU SYNC
// ============================================================================
void VolumetricRenderer::syncVolumetricData(SceneData& scene, OptixWrapper* optix_gpu_ptr) {
    if (!optix_gpu_ptr || !g_hasCUDA) return;

    // 1. Prepare VDB Volumes (Unified for VDB and Unified-Path Gas)
    std::vector<GpuVDBVolume> gpu_vdb_volumes;
    auto& mgr = VDBVolumeManager::getInstance();

    // 1a. Scan standard VDB objects
    for (const auto& vdb : scene.vdb_volumes) {
        if (!vdb || !vdb->visible) continue;

        GpuVDBVolume gv = {};
        gv.density_grid = mgr.getGPUGrid(vdb->getVDBVolumeID());
        gv.temperature_grid = mgr.getGPUTemperatureGrid(vdb->getVDBVolumeID());

        if (!gv.density_grid) continue;

        Matrix4x4 m = vdb->getTransform();
        Matrix4x4 inv = vdb->getInverseTransform();
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 4; ++j) {
            gv.transform[i * 4 + j] = m.m[i][j];
            gv.inv_transform[i * 4 + j] = inv.m[i][j];
        }

        gv.local_bbox_min = make_float3(vdb->getLocalBoundsMin().x, vdb->getLocalBoundsMin().y, vdb->getLocalBoundsMin().z);
        gv.local_bbox_max = make_float3(vdb->getLocalBoundsMax().x, vdb->getLocalBoundsMax().y, vdb->getLocalBoundsMax().z);
        
        AABB wb = vdb->getWorldBounds();
        gv.world_bbox_min = make_float3(wb.min.x, wb.min.y, wb.min.z);
        gv.world_bbox_max = make_float3(wb.max.x, wb.max.y, wb.max.z);

        // Pivot offset
        Vec3 po = vdb->getPivotOffset();
        gv.pivot_offset[0] = po.x; gv.pivot_offset[1] = po.y; gv.pivot_offset[2] = po.z;

        // Shader
        auto shader = vdb->volume_shader; 
        if (shader) {
            GpuVolumeShaderData gs = shader->toGPU();
            gv.density_multiplier = gs.density_multiplier * vdb->density_scale;
            gv.density_remap_low = gs.density_remap_low;
            gv.density_remap_high = gs.density_remap_high;
            gv.scatter_color = make_float3(gs.scatter_color_r, gs.scatter_color_g, gs.scatter_color_b);
            gv.scatter_coefficient = gs.scatter_coefficient;
            gv.scatter_anisotropy = gs.scatter_anisotropy;
            gv.scatter_anisotropy_back = gs.scatter_anisotropy_back;
            gv.scatter_lobe_mix = gs.scatter_lobe_mix;
            gv.scatter_multi = gs.scatter_multi;
            gv.absorption_color = make_float3(gs.absorption_color_r, gs.absorption_color_g, gs.absorption_color_b);
            gv.absorption_coefficient = gs.absorption_coefficient;
            gv.emission_mode = gs.emission_mode;
            gv.emission_color = make_float3(gs.emission_color_r, gs.emission_color_g, gs.emission_color_b);
            gv.emission_intensity = gs.emission_intensity;
            gv.temperature_scale = gs.temperature_scale;
            gv.blackbody_intensity = gs.blackbody_intensity;
            gv.step_size = gs.step_size;
            gv.max_steps = gs.max_steps;
            gv.shadow_steps = gs.shadow_steps;
            gv.shadow_strength = gs.shadow_strength;
            gv.voxel_size = vdb->getVoxelSize();
            gv.max_temperature = 6000.0f; // Default for standard VDBs
            
            // Color Ramp
            gv.color_ramp_enabled = gs.color_ramp_enabled;
            gv.ramp_stop_count = gs.ramp_stop_count;
            for (int i = 0; i < gv.ramp_stop_count; ++i) {
                gv.ramp_positions[i] = gs.ramp_positions[i];
                gv.ramp_colors[i] = make_float3(gs.ramp_colors_r[i], gs.ramp_colors_g[i], gs.ramp_colors_b[i]);
            }

            // Temperature fallback
            if (!gv.temperature_grid && gv.density_grid && gs.emission_mode == 2) { // 2 = Blackbody
                gv.temperature_grid = gv.density_grid;
            }
        } else {
            // Default VDB fallback
            gv.density_multiplier = vdb->density_scale;
            gv.scatter_color = make_float3(1.0f);
            gv.scatter_coefficient = 1.0f;
            gv.absorption_coefficient = 0.1f;
            gv.step_size = 0.1f;
            gv.max_steps = 128;
            gv.voxel_size = vdb->getVoxelSize();
        }
        gpu_vdb_volumes.push_back(gv);
    }

    // 1b. Scan Gas Volumes using Unified VDB path
    for (const auto& gas : scene.gas_volumes) {
        if (!gas || !gas->visible || gas->render_path != GasVolume::VolumeRenderPath::VDBUnified) continue;
        if (gas->live_vdb_id < 0) continue;

        GpuVDBVolume gv = {};
        gv.density_grid = mgr.getGPUGrid(gas->live_vdb_id);
        gv.temperature_grid = mgr.getGPUTemperatureGrid(gas->live_vdb_id);

        if (!gv.density_grid) continue;

        auto trans = gas->getTransformHandle();
        Matrix4x4 m = trans ? trans->getFinal() : Matrix4x4::identity();
        
        // PHYSICAL SIZE SYNC
        Vec3 gsize = gas->getSettings().grid_size;
        if (gsize.x > 0.001f && gsize.y > 0.001f && gsize.z > 0.001f) {
            m = m * Matrix4x4::scaling(Vec3(1.0f / gsize.x, 1.0f / gsize.y, 1.0f / gsize.z));
        }

        Matrix4x4 inv = m.inverse();
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 4; ++j) {
            gv.transform[i * 4 + j] = m.m[i][j];
            gv.inv_transform[i * 4 + j] = inv.m[i][j];
        }
        
        gv.pivot_offset[0] = 0; gv.pivot_offset[1] = 0; gv.pivot_offset[2] = 0;

        AABB wb; gas->bounding_box(0, 0, wb);
        gv.world_bbox_min = make_float3(wb.min.x, wb.min.y, wb.min.z);
        gv.world_bbox_max = make_float3(wb.max.x, wb.max.y, wb.max.z);
        gv.local_bbox_min = make_float3(0, 0, 0);
        gv.local_bbox_max = make_float3(gsize.x, gsize.y, gsize.z);

        auto shader = gas->getOrCreateShader(); 
        if (shader) {
            gv.density_multiplier = shader->density.multiplier;
            gv.density_remap_low = shader->density.remap_low;
            gv.density_remap_high = shader->density.remap_high;
            gv.scatter_color = make_float3(shader->scattering.color.x, shader->scattering.color.y, shader->scattering.color.z);
            gv.scatter_coefficient = shader->scattering.coefficient;
            gv.scatter_anisotropy = shader->scattering.anisotropy;
            gv.scatter_anisotropy_back = shader->scattering.anisotropy_back;
            gv.scatter_lobe_mix = shader->scattering.lobe_mix;
            gv.scatter_multi = shader->scattering.multi_scatter;
            gv.absorption_color = make_float3(shader->absorption.color.x, shader->absorption.color.y, shader->absorption.color.z);
            gv.absorption_coefficient = shader->absorption.coefficient;
            gv.emission_mode = static_cast<int>(shader->emission.mode);
            gv.emission_color = make_float3(shader->emission.color.x, shader->emission.color.y, shader->emission.color.z);
            gv.emission_intensity = shader->emission.intensity;
            gv.temperature_scale = shader->emission.temperature_scale;
            gv.blackbody_intensity = shader->emission.blackbody_intensity;
            gv.step_size = shader->quality.step_size;
            gv.max_steps = shader->quality.max_steps;
            gv.shadow_steps = shader->quality.shadow_steps;
            gv.shadow_strength = shader->quality.shadow_strength;
            gv.voxel_size = gas->getSettings().voxel_size;
            gv.max_temperature = gas->getSettings().max_temperature;

            // Color Ramp
            gv.color_ramp_enabled = shader->emission.color_ramp.enabled ? 1 : 0;
            gv.ramp_stop_count = static_cast<int>(std::min(shader->emission.color_ramp.stops.size(), static_cast<size_t>(8)));
            for (int i = 0; i < gv.ramp_stop_count; ++i) {
                gv.ramp_positions[i] = shader->emission.color_ramp.stops[i].position;
                gv.ramp_colors[i] = make_float3(
                    static_cast<float>(shader->emission.color_ramp.stops[i].color.x),
                    static_cast<float>(shader->emission.color_ramp.stops[i].color.y),
                    static_cast<float>(shader->emission.color_ramp.stops[i].color.z)
                );
            }

            if (!gv.temperature_grid && gv.density_grid && shader->emission.mode == VolumeEmissionMode::Blackbody) {
                gv.temperature_grid = gv.density_grid;
            }
        }
        gpu_vdb_volumes.push_back(gv);
    }
    optix_gpu_ptr->updateVDBVolumeBuffer(gpu_vdb_volumes);

    // 2. Prepare Texture-based Gas Volumes (Legacy Path)
    std::vector<GpuGasVolume> gpu_gas_volumes;
    for (const auto& gas : scene.gas_volumes) {
        if (!gas || !gas->visible || gas->render_path != GasVolume::VolumeRenderPath::Legacy) continue;

        GpuGasVolume gv = {};
        gv.density_texture = (cudaTextureObject_t)gas->getDensityTexture();
        gv.temperature_texture = (cudaTextureObject_t)gas->getTemperatureTexture();
        gv.has_texture = (gv.density_texture != 0) ? 1 : 0;

        auto trans = gas->getTransformHandle();
        if (trans) {
            Matrix4x4 mat = trans->getFinal();
            Matrix4x4 inv = mat.inverse();
            std::memcpy(gv.transform, &mat.m[0][0], 12 * sizeof(float));
            std::memcpy(gv.inv_transform, &inv.m[0][0], 12 * sizeof(float));
        }

        AABB bbox; gas->bounding_box(0, 0, bbox);
        gv.world_bbox_min = make_float3(bbox.min.x, bbox.min.y, bbox.min.z);
        gv.world_bbox_max = make_float3(bbox.max.x, bbox.max.y, bbox.max.z);
        gv.local_bbox_min = make_float3(0, 0, 0);
        gv.local_bbox_max = make_float3(gas->getSettings().grid_size.x, gas->getSettings().grid_size.y, gas->getSettings().grid_size.z);

        auto shader = gas->getOrCreateShader();
        if (shader) {
            gv.density_multiplier = shader->density.multiplier;
            gv.density_remap_low = shader->density.remap_low;
            gv.density_remap_high = shader->density.remap_high;
            gv.scatter_color = make_float3(shader->scattering.color.x, shader->scattering.color.y, shader->scattering.color.z);
            gv.scatter_coefficient = shader->scattering.coefficient;
            gv.absorption_color = make_float3(shader->absorption.color.x, shader->absorption.color.y, shader->absorption.color.z);
            gv.absorption_coefficient = shader->absorption.coefficient;
            gv.scatter_anisotropy = shader->scattering.anisotropy;
            gv.scatter_anisotropy_back = shader->scattering.anisotropy_back;
            gv.scatter_lobe_mix = shader->scattering.lobe_mix;
            gv.emission_mode = (int)shader->emission.mode;
            gv.emission_color = make_float3(shader->emission.color.x, shader->emission.color.y, shader->emission.color.z);
            gv.emission_intensity = shader->emission.intensity;
            gv.temperature_scale = shader->emission.temperature_scale;
            gv.blackbody_intensity = shader->emission.blackbody_intensity;
            gv.step_size = shader->quality.step_size;
            gv.max_steps = shader->quality.max_steps;
            gv.shadow_strength = shader->quality.shadow_strength;
            gv.shadow_steps = shader->quality.shadow_steps;
            gv.max_temperature = gas->getSettings().max_temperature;

            // Color Ramp (Legacy)
            gv.color_ramp_enabled = shader->emission.color_ramp.enabled ? 1 : 0;
            gv.ramp_stop_count = static_cast<int>(std::min(shader->emission.color_ramp.stops.size(), static_cast<size_t>(8)));
            for (int i = 0; i < gv.ramp_stop_count; ++i) {
                gv.ramp_positions[i] = shader->emission.color_ramp.stops[i].position;
                gv.ramp_colors[i] = make_float3(
                    static_cast<float>(shader->emission.color_ramp.stops[i].color.x),
                    static_cast<float>(shader->emission.color_ramp.stops[i].color.y),
                    static_cast<float>(shader->emission.color_ramp.stops[i].color.z)
                );
            }
        }
        gpu_gas_volumes.push_back(gv);
    }
    optix_gpu_ptr->updateGasVolumeBuffer(gpu_gas_volumes);
}

// ════════════════════════════════════════════════════════════════════════════
// AERIAL PERSPECTIVE (CPU) - Matches GPU gpu_get_aerial_perspective
// ════════════════════════════════════════════════════════════════════════════
static float calculate_height_fog_factor_cpu(Vec3 rayOrigin, Vec3 rayDir, float distance, float fogDensity, float fogHeight, float fogFalloff) {
    float a = fogDensity * expf(-fogFalloff * rayOrigin.y);
    float b = fogFalloff * rayDir.y;
    if (fabsf(b) < 1e-5f) return 1.0f - expf(-a * distance);
    float fogAmount = a * (1.0f - expf(-b * distance)) / b;
    return 1.0f - expf(-fogAmount);
}

Vec3 VolumetricRenderer::applyAerialPerspective(const SceneData& scene, const WorldData& world_data, const Vec3& origin, const Vec3& dir, float dist, const Vec3& color, AtmosphereLUT* lut) {
    if (world_data.mode != WORLD_MODE_NISHITA || !lut || !lut->is_initialized() || dist > 1e6f) return color;
    if (!world_data.advanced.aerial_perspective) return color;

    float Rg = world_data.nishita.planet_radius;
    if (Rg < 1000.0f) Rg = 6360000.0f; 
    
    // Accurate altitude based on origin (matches GPU)
    Vec3 p = origin + Vec3(0, Rg, 0);
    float current_altitude = p.length() - Rg;
    Vec3 up = p / (Rg + current_altitude);
    
    float cosTheta = std::max(0.01f, Vec3::dot(up, dir));
    float3 trans3 = lut->sampleTransmittance(cosTheta, current_altitude, world_data.nishita.atmosphere_height);
    Vec3 transmittance(trans3.x, trans3.y, trans3.z);
    
    // Matched to GPU density scaling
    float densityFactor = 1.0f + world_data.nishita.fog_density * 300.0f;
    float effectiveDist = dist * densityFactor;
    
    const float min_dist = world_data.advanced.aerial_min_distance;
    const float max_dist = world_data.advanced.aerial_max_distance;
    float ramp = (dist < min_dist) ? 0.0f : std::min(1.0f, (dist - min_dist) / std::max(1.0f, max_dist - min_dist));
    
    // 20km -> 10km scale matched to GPU
    float distFactor = std::min(1.0f, effectiveDist / 10000.0f);
    distFactor *= (ramp * ramp);

    Vec3 finalTrans(
        powf(transmittance.x, distFactor),
        powf(transmittance.y, distFactor),
        powf(transmittance.z, distFactor)
    );
    
    float3 lookupDir = make_float3(dir.x, std::max(0.0f, dir.y), dir.z);
    float3 skyRadiance3 = lut->sampleSkyView(lookupDir, world_data.nishita.sun_direction, Rg, Rg + world_data.nishita.atmosphere_height);
    Vec3 skyRadiance(skyRadiance3.x, skyRadiance3.y, skyRadiance3.z);
    
    // Blend with scene
    Vec3 res = color * finalTrans + skyRadiance * (Vec3(1,1,1) - finalTrans);

    // DYNAMIC HEIGHT FOG OVERLAY (Nishita active)
    if (world_data.nishita.fog_enabled && world_data.nishita.fog_density > 0.01f) {
        float fAmount = calculate_height_fog_factor_cpu(
            origin, dir, dist,
            world_data.nishita.fog_density * 0.5f,
            world_data.nishita.fog_height,
            world_data.nishita.fog_falloff
        );
        Vec3 fogCol(world_data.nishita.fog_color.x, world_data.nishita.fog_color.y, world_data.nishita.fog_color.z);
        res = res * (1.0f - fAmount) + fogCol * fAmount;
    }

    return res;
}
