/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          VolumetricRenderer.cpp
* =========================================================================
*/
#include "VolumetricRenderer.h"
#include "Backend/IBackend.h"
#include "Backend/VulkanBackend.h"
#include "VDBVolume.h"
#include "VDBVolumeManager.h"
#include "GasVolume.h"
#include "VolumeShader.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "Volumetric.h"
#include "renderer.h" 
#include "globals.h"
#include <cmath>
#include <algorithm> // For std::min, std::max
#include "AtmosphereLUT.h"
#include "GodRaysModel.h"

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
        float t = (std::max)(0.0f, 1.0f + sun_elevation * 5.0f);
        Vec3 redColor = Vec3(0.8f, 0.3f, 0.1f);
        Vec3 orangeColor = Vec3(1.0f, 0.6f, 0.3f);
        return redColor * (1.0f - t) + orangeColor * t;
    }
}

static Vec3 to_vec3(const float3& v) { return Vec3(v.x, v.y, v.z); }

namespace {
constexpr const char* kInternalSkyCloudVolumeName = "__RayTrophi_Internal_SkyCloudVolume";

float clamp01(float v) {
    return (std::max)(0.0f, (std::min)(1.0f, v));
}

// A single material graph asset can provide both Surface and Volume outputs.
// VDB/Gas/Fluid consume Volume through VolumeShader::material_program; an SDF
// boundary consumes the same asset's Principled surface closure here.
void applySharedSurfaceMaterial(VDBVolume& volume, const std::shared_ptr<VolumeShader>& shader) {
    if (!volume.render_as_isosurface || !shader || shader->material_graph.empty()) return;
    auto& materials = MaterialManager::getInstance();
    const uint16_t id = materials.getMaterialID(shader->material_graph);
    if (id == MaterialManager::INVALID_MATERIAL_ID) return;
    auto* surface = dynamic_cast<PrincipledBSDF*>(materials.getMaterial(id));
    if (!surface) return;

    volume.render_isosurface_ior = (std::max)(1.0f, surface->getIOR());
    volume.render_isosurface_roughness =
        (std::max)(0.0f, (std::min)(1.0f, surface->getRoughnessValue(Vec2(0.5f, 0.5f))));
}

// A VDB only needs the material VM when the compiled graph actually writes at
// least one Volume closure slot. Surface-only programs are valid for an SDF
// boundary, but interpreting them at every VDB march sample is both ineffective
// (volumeWritten stays zero) and extremely expensive.
int resolveVolumeMaterialProgramIndex(const std::shared_ptr<VolumeShader>& shader) {
    if (!shader || shader->material_graph.empty()) return -1;

    auto& materials = MaterialManager::getInstance();
    const uint16_t id = materials.getMaterialID(shader->material_graph);
    if (id == MaterialManager::INVALID_MATERIAL_ID) return -1;

    const MaterialNodesV2::MaterialProgram* program = nullptr;
    if (auto* surface = dynamic_cast<PrincipledBSDF*>(materials.getMaterial(id))) {
        program = surface->proceduralProgram.get();
    } else if (auto* volume = dynamic_cast<Volumetric*>(materials.getMaterial(id))) {
        program = volume->proceduralProgram.get();
    }

    constexpr uint32_t kVolumeSlotMask = 0x01FF0000u; // closure slots 16..24
    if (!program || !program->active || (program->drivenSlots & kVolumeSlotMask) == 0u)
        return -1;
    // Must match MP_REGISTER_COUNT in volume_closesthit.rchit. Surface graphs
    // retain the full 32-register VM; oversized volume graphs safely keep their
    // already-folded material values instead of overflowing shader scratch.
    constexpr int kVolumeVmRegisterLimit = 12;
    if (program->regCount > kVolumeVmRegisterLimit) return -1;
    return static_cast<int>(id);
}

std::shared_ptr<VDBVolume> findInternalSkyCloudVolume(SceneData& scene) {
    for (const auto& vdb : scene.vdb_volumes) {
        if (vdb && vdb->name == kInternalSkyCloudVolumeName) {
            return vdb;
        }
    }
    return nullptr;
}

void removeInternalSkyCloudVolume(SceneData& scene, const std::shared_ptr<VDBVolume>& volume) {
    if (!volume) return;
    scene.vdb_volumes.erase(
        std::remove(scene.vdb_volumes.begin(), scene.vdb_volumes.end(), volume),
        scene.vdb_volumes.end());
    scene.world.objects.erase(
        std::remove(scene.world.objects.begin(), scene.world.objects.end(), volume),
        scene.world.objects.end());
}

bool ensureInternalSkyCloudVolume(SceneData& scene, const WorldData& worldData) {
    const auto& n = worldData.nishita;
    const bool enabled = (worldData.mode == WORLD_MODE_NISHITA) &&
                         ((n.clouds_enabled != 0 && n.cloud_coverage > 0.001f && n.cloud_density > 0.001f) ||
                          (n.cloud_layer2_enabled != 0 && n.cloud2_coverage > 0.001f && n.cloud2_density > 0.001f));

    auto volume = findInternalSkyCloudVolume(scene);
    if (!enabled) {
        if (volume) {
            removeInternalSkyCloudVolume(scene, volume);
            return true;
        }
        return false;
    }

    const bool created = !volume;
    if (!volume) {
        volume = std::make_shared<VDBVolume>();
        volume->name = kInternalSkyCloudVolumeName;
        volume->visible = true;
        scene.vdb_volumes.push_back(volume);
        scene.world.objects.push_back(volume);
    } else if (std::find(scene.world.objects.begin(), scene.world.objects.end(), volume) == scene.world.objects.end()) {
        scene.world.objects.push_back(volume);
    }

    float minH = n.cloud_height_min;
    float maxH = n.cloud_height_max;
    float coverage = n.cloud_coverage;
    float density = n.cloud_density;
    float scale = n.cloud_scale;
    int maxSteps = n.cloud_base_steps;
    int shadowSteps = n.cloud_light_steps;
    float shadowStrength = n.cloud_shadow_strength;
    float absorption = n.cloud_absorption;

    if (n.cloud_layer2_enabled && (!n.clouds_enabled || n.cloud2_density > density)) {
        minH = n.cloud2_height_min;
        maxH = n.cloud2_height_max;
        coverage = n.cloud2_coverage;
        density = n.cloud2_density;
        scale = n.cloud2_scale;
    }

    if (maxH <= minH + 1.0f) maxH = minH + 100.0f;
    const float thickness = (std::max)(1.0f, maxH - minH);
    const float extent = (std::max)(50000.0f, (std::min)(500000.0f, maxH * 20.0f + thickness * 40.0f));

    Vec3 boundsMin(-extent, minH, -extent);
    Vec3 boundsMax( extent, maxH,  extent);
    const Vec3 oldMin = volume->getLocalBoundsMin();
    const Vec3 oldMax = volume->getLocalBoundsMax();
    const bool boundsChanged =
        std::abs(oldMin.x - boundsMin.x) > 0.01f || std::abs(oldMin.y - boundsMin.y) > 0.01f ||
        std::abs(oldMin.z - boundsMin.z) > 0.01f || std::abs(oldMax.x - boundsMax.x) > 0.01f ||
        std::abs(oldMax.y - boundsMax.y) > 0.01f || std::abs(oldMax.z - boundsMax.z) > 0.01f;

    volume->setProceduralVolumeBounds(boundsMin, boundsMax);
    volume->setTransform(Matrix4x4::identity());
    // The procedural sky cloud already applies the UI density through the
    // volume shader below. Keep the VDB instance scale neutral so Vulkan/OptiX
    // upload does not square the density multiplier.
    volume->density_scale = 1.0f;
    volume->voxel_size = (std::max)(1.0f, 24.0f / (std::max)(0.1f, scale));

    auto shader = volume->getOrCreateShader();
    shader->name = "World Sky Cloud Volume";
    shader->density.multiplier = (std::max)(0.0f, density);
    // Coverage is baked into the procedural cloud density source. Keep the
    // material remap neutral so all backends treat it like a normal volume.
    shader->density.remap_low = 0.0f;
    shader->density.remap_high = 1.0f;
    shader->scattering.color = Vec3(1.0f);
    const float ambientStrength = (std::max)(0.55f, n.cloud_ambient_strength);
    const float densityClass = clamp01((density - 0.25f) / 1.0f);
    shader->scattering.coefficient = 0.042f * (0.75f + ambientStrength * 0.25f);
    shader->scattering.anisotropy = clamp01(n.cloud_anisotropy);
    shader->scattering.anisotropy_back = (std::max)(-0.99f, (std::min)(0.0f, n.cloud_anisotropy_back));
    shader->scattering.lobe_mix = clamp01(n.cloud_lobe_mix);
    shader->scattering.multi_scatter = clamp01(0.45f + n.cloud_silver_intensity * 0.35f + densityClass * 0.15f);
    shader->absorption.color = Vec3(1.0f);
    shader->absorption.coefficient = 0.000075f * (std::max)(0.2f, absorption) * (1.0f - densityClass * 0.25f);
    shader->emission.mode = n.cloud_emissive_intensity > 0.001f ? VolumeEmissionMode::Constant : VolumeEmissionMode::None;
    shader->emission.color = Vec3(n.cloud_emissive_color.x, n.cloud_emissive_color.y, n.cloud_emissive_color.z);
    shader->emission.intensity = n.cloud_emissive_intensity * 4.0f;
    shader->quality.step_size = (std::max)(18.0f, thickness / (std::max)(6, (int)(maxSteps * 0.75f)));
    shader->quality.max_steps = (std::max)(6, (int)(maxSteps * 0.45f));
    shader->quality.shadow_steps = (std::max)(0, (int)(shadowSteps * 0.5f));
    shader->quality.shadow_strength = (std::max)(0.0f, shadowStrength);

    return created || boundsChanged;
}
}

// ============================================================================
// GOD RAYS (CPU) — OptiX GPU calculate_volumetric_god_rays ile birebir eşleşme
// ============================================================================
Vec3 VolumetricRenderer::calculateGodRays(const SceneData& scene, const WorldData& world_data, const Ray& ray, float maxDistance, const Hittable* bvh, class AtmosphereLUT* lut) {
    // Allow god rays even when the world mode is not Nishita; use Nishita fields
    // for parameters but fall back gracefully when LUTs aren't present.
    if (!GodRaysModel::isEnabled(
            world_data.nishita.godrays_enabled,
            world_data.nishita.godrays_intensity,
            world_data.nishita.godrays_density)) {
        return Vec3(0.0f);
    }

    const auto& nishita = world_data.nishita;
    Vec3 sunDir = to_vec3(nishita.sun_direction).normalize();
    float sunDot = Vec3::dot(ray.direction, sunDir);

    // === MATCH GPU: anisotropy-based early exit ===
    float g = nishita.mie_anisotropy;
    float anisotropyFade = GodRaysModel::anisotropyFade(sunDot, g);
    if (anisotropyFade < 0.001f) return Vec3(0.0f);

    // === MATCH GPU: sun below horizon early exit ===
    if (GodRaysModel::isSunBelowHorizon(sunDir.y)) return Vec3(0.0f);

    // === MATCH GPU: march distance cap (5000m) ===
    float marchDistance = GodRaysModel::computeMarchDistance(maxDistance, nishita.fog_distance);

    int numSteps = GodRaysModel::computeStepCount(sunDot, nishita.godrays_samples);

    float stepSize = GodRaysModel::computeStepSize(marchDistance, numSteps);

    // === MATCH GPU: Mie phase ===
    float phase = GodRaysModel::computeMiePhase(sunDot, g);

    float mediaDensity = GodRaysModel::computeMediaDensity(nishita.godrays_density);

    // Use a sun color guess based on elevation as a sensible fallback when no LUT
    // is available, but still modulate with the LUT if present for physical reddening.
    Vec3 sunRadianceBase = get_sun_color(sunDir.y) * nishita.sun_intensity * GodRaysModel::kSunRadianceScale;
    if (lut) {
        float3 st = lut->sampleTransmittance((std::max)(0.01f, sunDir.y), nishita.altitude, nishita.atmosphere_height);
        sunRadianceBase = sunRadianceBase * Vec3(st.x, st.y, st.z);
    }

    Vec3  godRayColor(0.0f);
    float transmittance = 1.0f;

    // === MATCH GPU: jitter starting position ===
    float jitter = ((float)rand() / (float)RAND_MAX);
    float t = jitter * stepSize;

    for (int i = 0; i < numSteps; ++i) {
        if (t > marchDistance) break;

        float nearFade = GodRaysModel::computeNearFade(t);
        if (nearFade > 0.001f) {
            Vec3 samplePos = ray.at(t);

            float heightFactor = GodRaysModel::computeHeightFactor(samplePos.y, nishita.altitude);

            float sigma_s = mediaDensity * heightFactor;
            float sigma_t = sigma_s;
            float stepTrans = GodRaysModel::computeStepTransmittance(sigma_t, stepSize);

            if (sigma_t > 1e-6f) {
                // Solid shadow test
                float occlusion = determineSunTransmittance(samplePos, sunDir, 100000.0f, bvh, scene, world_data);

                if (occlusion > 0.001f) {
                    // === MATCH GPU inscatter formula exactly ===
                    Vec3 inscatter = sunRadianceBase * phase * occlusion * (sigma_s / sigma_t) * nearFade;
                    godRayColor = godRayColor + inscatter * transmittance * (1.0f - stepTrans);
                }
            }

            transmittance *= stepTrans;
        }

        if (transmittance < GodRaysModel::kTransmittanceCutoff) break;
        t += stepSize;
    }

    // === MATCH GPU: multiply by godrays_intensity at end ===
    godRayColor = godRayColor * nishita.godrays_intensity;

    if (!std::isfinite(godRayColor.x) || !std::isfinite(godRayColor.y) || !std::isfinite(godRayColor.z))
        return Vec3(0.0f);
    return godRayColor;
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
void VolumetricRenderer::syncVolumetricData(SceneData& scene, Backend::IBackend* backend, const WorldData* world_data) {
    // Allow sync for both OptiX (CUDA) and Vulkan backends.
    // Legacy CUDA texture gas volumes are still guarded by g_hasCUDA below.
    const bool skyCloudVolumeChanged = world_data && ensureInternalSkyCloudVolume(scene, *world_data);
    if (skyCloudVolumeChanged) {
        g_geometry_dirty = true;
        g_vulkan_rebuild_pending = true;
        g_optix_rebuild_pending = true;
        g_gas_volumes_dirty = true;
    }
    if (!backend) return;

    // 1. Prepare VDB Volumes (Unified for VDB and Unified-Path Gas)
    std::vector<GpuVDBVolume> gpu_vdb_volumes;
    auto& mgr = VDBVolumeManager::getInstance();
    const bool vulkanRT = dynamic_cast<Backend::VulkanBackendAdapter*>(backend) != nullptr;

    // 1a. Scan standard VDB objects
    for (const auto& vdb : scene.vdb_volumes) {
        if (!vdb || !vdb->visible) continue;

        GpuVDBVolume gv = {};
        gv.vdb_id = vdb->getVDBVolumeID();
        const bool proceduralVolume = vdb->isProceduralVolume();
        
        // PROACTIVE CUDA UPLOAD ON BACKEND SWITCH:
        // If we are in OptiX/CUDA path, and the CUDA grid is not yet uploaded,
        // but the host grid exists (e.g. loaded under Vulkan mode), trigger CUDA upload.
        if (render_settings.use_optix && g_hasCUDA && !proceduralVolume) {
            if (!mgr.getGPUGrid(gv.vdb_id) && mgr.getHostGrid(gv.vdb_id)) {
                mgr.uploadToGPU(gv.vdb_id, true);
            }
        }

        gv.density_grid = proceduralVolume ? nullptr : mgr.getGPUGrid(gv.vdb_id);
        gv.temperature_grid = proceduralVolume ? nullptr : mgr.getGPUTemperatureGrid(gv.vdb_id);

        // Allow Vulkan path: density_grid is null (no CUDA), but host grid exists for Vulkan upload
        if (!proceduralVolume && !gv.density_grid && !mgr.getHostGrid(gv.vdb_id)) continue;

        Matrix4x4 m = vdb->getTransform();
        Matrix4x4 inv = vdb->getInverseTransform();
        // VDB transforms frequently contain scene-unit conversion or artist
        // scaling. getVoxelSize() is native-grid space; ray marching happens in
        // world space. Use the smallest transformed basis length so anisotropic
        // scaling never undersamples the most compressed voxel axis.
        const float scaleX = std::sqrt(m.m[0][0] * m.m[0][0] + m.m[1][0] * m.m[1][0] + m.m[2][0] * m.m[2][0]);
        const float scaleY = std::sqrt(m.m[0][1] * m.m[0][1] + m.m[1][1] * m.m[1][1] + m.m[2][1] * m.m[2][1]);
        const float scaleZ = std::sqrt(m.m[0][2] * m.m[0][2] + m.m[1][2] * m.m[1][2] + m.m[2][2] * m.m[2][2]);
        const float minWorldScale = (std::max)(1e-6f, (std::min)(scaleX, (std::min)(scaleY, scaleZ)));
        const float worldVoxelSize = (std::max)(1e-6f, vdb->getVoxelSize() * minWorldScale);
        const float renderVoxelSize = vulkanRT ? worldVoxelSize : vdb->getVoxelSize();
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
        gv.pivot_offset[0] = 0.0f;
        gv.pivot_offset[1] = 0.0f;
        gv.pivot_offset[2] = 0.0f;
        // source_type encoding: 0 = NanoVDB/default, 3 = procedural cloud,
        // 4 = fluid surface SDF (isosurface raymarch + refraction in shaders).
        // The isosurface route also rides the NanoVDB density path (its
        // density channel is the SDF proxy band), so vdb_grid_address etc.
        // remain valid — the shader picks the branch from source_type.
        if (vdb->render_as_isosurface) {
            gv.source_type = 4;
        } else {
            gv.source_type = proceduralVolume ? 3 : 0;
        }
        gv.ior = (vdb->render_isosurface_ior > 1.0f) ? vdb->render_isosurface_ior : 1.33f;
        gv.surface_roughness = (vdb->render_isosurface_roughness > 0.0f)
            ? ((vdb->render_isosurface_roughness < 1.0f) ? vdb->render_isosurface_roughness : 1.0f)
            : 0.0f;
        gv.surface_foam = (vdb->render_isosurface_foam > 0.0f)
            ? ((vdb->render_isosurface_foam < 1.0f) ? vdb->render_isosurface_foam : 1.0f)
            : 0.0f;
        gv.foam_color = make_float3(vdb->render_isosurface_foam_color.x,
                                    vdb->render_isosurface_foam_color.y,
                                    vdb->render_isosurface_foam_color.z);
        gv.foam_opacity = vdb->render_isosurface_foam_opacity;
        gv.cloud_coverage = 1.0f;
        gv.cloud_detail = 1.0f;
        gv.cloud_erosion = 0.5f;
        gv.cloud_base_scale = (std::max)(1.0f, vdb->getVoxelSize());
        gv.cloud_edge_fade = 0.08f;
        gv.cloud_offset_x = 0.0f;
        gv.cloud_offset_z = 0.0f;
        gv.cloud_seed = 0.0f;

        if (proceduralVolume && world_data) {
            const auto& n = world_data->nishita;
            float coverage = n.cloud_coverage;
            float density = n.cloud_density;
            float scale = n.cloud_scale;
            if (n.cloud_layer2_enabled && (!n.clouds_enabled || n.cloud2_density > density)) {
                coverage = n.cloud2_coverage;
                density = n.cloud2_density;
                scale = n.cloud2_scale;
            }
            gv.cloud_coverage = clamp01(coverage);
            gv.cloud_detail = clamp01(n.cloud_detail);
            gv.cloud_erosion = clamp01(1.0f - coverage);
            gv.cloud_base_scale = (std::max)(8.0f, (std::min)(72.0f, 10.0f / (std::max)(0.1f, scale)));
            gv.cloud_edge_fade = 0.08f;
            gv.cloud_offset_x = n.cloud_offset_x * 0.00002f;
            gv.cloud_offset_z = n.cloud_offset_z * 0.00002f;
            gv.cloud_seed = static_cast<float>(n.cloud_seed);
        }

        // Shader. SDF volumes also resolve the Surface output from this same
        // material asset, keeping water/glass boundaries and their interior
        // volume under one graph selection.
        auto shader = vdb->volume_shader;
        applySharedSurfaceMaterial(*vdb, shader);
        gv.material_program_index = resolveVolumeMaterialProgramIndex(shader);
        gv.ior = (vdb->render_isosurface_ior > 1.0f) ? vdb->render_isosurface_ior : 1.33f;
        gv.surface_roughness = clamp01(vdb->render_isosurface_roughness);
        if (shader) {
            GpuVolumeShaderData gs = shader->toGPU();
            gv.density_multiplier = gs.density_multiplier * vdb->density_scale;
            gv.density_remap_low = gs.density_remap_low;
            gv.density_remap_high = gs.density_remap_high;
            gv.density_pad = gs.density_pad;
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
            gv.step_size = shader->quality.primaryStep(renderVoxelSize);
            gv.max_steps = gs.max_steps;
            gv.shadow_steps = gs.shadow_steps;
            gv.shadow_stride = gs.shadow_stride;
            gv.shadow_strength = gs.shadow_strength;
            gv.density_noise_enabled = gs.density_noise_enabled;
            gv.density_noise_scale = gs.density_noise_scale;
            gv.density_noise_strength = gs.density_noise_strength;
            gv.density_noise_detail = gs.density_noise_detail;
            gv.density_noise_seed = gs.density_noise_seed;
            gv.voxel_size = renderVoxelSize;
            gv.max_temperature = 6000.0f; // Default for standard VDBs
            
            // Color Ramp
            gv.color_ramp_enabled = gs.color_ramp_enabled;
            gv.ramp_stop_count = gs.ramp_stop_count;
            for (int i = 0; i < gv.ramp_stop_count; ++i) {
                gv.ramp_positions[i] = gs.ramp_positions[i];
                gv.ramp_colors[i] = make_float3(gs.ramp_colors_r[i], gs.ramp_colors_g[i], gs.ramp_colors_b[i]);
            }

            // Temperature fallback for live/domain VDBs that only carry density.
            if (!gv.temperature_grid && gv.density_grid && gs.emission_mode >= 2) {
                gv.temperature_grid = gv.density_grid;
            }
        } else {
            // Default VDB fallback
            gv.density_multiplier = vdb->density_scale;
            gv.scatter_color = make_float3(1.0f);
            gv.scatter_coefficient = 1.0f;
            gv.absorption_coefficient = 0.1f;
            gv.step_size = (std::max)(renderVoxelSize, 1e-4f);
            gv.max_steps = 128;
            gv.voxel_size = renderVoxelSize;
        }
        gpu_vdb_volumes.push_back(gv);
    }

    // 1b. Scan Gas Volumes using Unified VDB path
    for (const auto& gas : scene.gas_volumes) {
        if (!gas || !gas->visible || gas->render_path != GasVolume::VolumeRenderPath::VDBUnified) continue;
        if (gas->live_vdb_id < 0) continue;

        GpuVDBVolume gv = {};
        gv.vdb_id = gas->live_vdb_id;

        // PROACTIVE CUDA UPLOAD ON BACKEND SWITCH:
        // If we are in OptiX/CUDA path, and the CUDA grid is not yet uploaded,
        // but the host grid exists (e.g. loaded under Vulkan mode), trigger CUDA upload.
        if (render_settings.use_optix && g_hasCUDA) {
            if (!mgr.getGPUGrid(gv.vdb_id) && mgr.getHostGrid(gv.vdb_id)) {
                mgr.uploadToGPU(gv.vdb_id, true);
            }
        }

        gv.density_grid = mgr.getGPUGrid(gv.vdb_id);
        gv.temperature_grid = mgr.getGPUTemperatureGrid(gv.vdb_id);

        // Allow Vulkan path: density_grid is null (no CUDA), but host grid exists for Vulkan upload
        if (!gv.density_grid && !mgr.getHostGrid(gv.vdb_id)) continue;

        Transform* trans = gas->getTransformPtr();
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
        gv.material_program_index = resolveVolumeMaterialProgramIndex(shader);
        if (shader) {
            const GpuVolumeShaderData gs = shader->toGPU();
            gv.density_multiplier = gs.density_multiplier;
            gv.density_remap_low = gs.density_remap_low;
            gv.density_remap_high = gs.density_remap_high;
            gv.density_pad = gs.density_pad;
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
            gv.step_size = shader->quality.primaryStep(gas->getSettings().voxel_size);
            gv.max_steps = gs.max_steps;
            gv.shadow_steps = gs.shadow_steps;
            gv.shadow_stride = gs.shadow_stride;
            gv.shadow_strength = gs.shadow_strength;
            gv.density_noise_enabled = gs.density_noise_enabled;
            gv.density_noise_scale = gs.density_noise_scale;
            gv.density_noise_strength = gs.density_noise_strength;
            gv.density_noise_detail = gs.density_noise_detail;
            gv.density_noise_seed = gs.density_noise_seed;
            gv.voxel_size = gas->getSettings().voxel_size;
            const float ambientKelvin = gas->getSettings().ambient_temperature;
            gv.emission_pad = ambientKelvin + shader->emission.temperature_min;
            gv.max_temperature = ambientKelvin + (std::max)(shader->emission.temperature_max, shader->emission.temperature_min + 1.0f);

            // Color Ramp
            gv.color_ramp_enabled = shader->emission.color_ramp.enabled ? 1 : 0;
            gv.ramp_stop_count = static_cast<int>((std::min)(shader->emission.color_ramp.stops.size(), static_cast<size_t>(8)));
            for (int i = 0; i < gv.ramp_stop_count; ++i) {
                gv.ramp_positions[i] = shader->emission.color_ramp.stops[i].position;
                gv.ramp_colors[i] = make_float3(
                    static_cast<float>(shader->emission.color_ramp.stops[i].color.x),
                    static_cast<float>(shader->emission.color_ramp.stops[i].color.y),
                    static_cast<float>(shader->emission.color_ramp.stops[i].color.z)
                );
            }

            if (!gv.temperature_grid && gv.density_grid &&
                (shader->emission.mode == VolumeEmissionMode::Blackbody ||
                 shader->emission.mode == VolumeEmissionMode::ChannelDriven ||
                 shader->emission.color_ramp.enabled)) {
                gv.temperature_grid = gv.density_grid;
            }
        }
        gpu_vdb_volumes.push_back(gv);
    }
    backend->updateVDBVolumes(gpu_vdb_volumes);

    // 2. Prepare Texture-based Gas Volumes (Legacy Path - CUDA/OptiX only)
    if (!g_hasCUDA) return; // Legacy gas volumes use CUDA textures; skip for Vulkan
    std::vector<GpuGasVolume> gpu_gas_volumes;
    for (const auto& gas : scene.gas_volumes) {
        if (!gas || !gas->visible || gas->render_path != GasVolume::VolumeRenderPath::Legacy) continue;

        GpuGasVolume gv = {};
        gv.density_texture = (cudaTextureObject_t)gas->getDensityTexture();
        gv.temperature_texture = (cudaTextureObject_t)gas->getTemperatureTexture();
        gv.has_texture = (gv.density_texture != 0) ? 1 : 0;

        Transform* trans = gas->getTransformPtr();
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
            const GpuVolumeShaderData gs = shader->toGPU();
            gv.density_multiplier = gs.density_multiplier;
            gv.density_remap_low = gs.density_remap_low;
            gv.density_remap_high = gs.density_remap_high;
            gv.density_pad = gs.density_pad;
            gv.scatter_color = make_float3(gs.scatter_color_r, gs.scatter_color_g, gs.scatter_color_b);
            gv.scatter_coefficient = gs.scatter_coefficient;
            gv.absorption_color = make_float3(gs.absorption_color_r, gs.absorption_color_g, gs.absorption_color_b);
            gv.absorption_coefficient = gs.absorption_coefficient;
            gv.scatter_anisotropy = gs.scatter_anisotropy;
            gv.scatter_anisotropy_back = gs.scatter_anisotropy_back;
            gv.scatter_lobe_mix = gs.scatter_lobe_mix;
            gv.emission_mode = gs.emission_mode;
            gv.emission_color = make_float3(gs.emission_color_r, gs.emission_color_g, gs.emission_color_b);
            gv.emission_intensity = gs.emission_intensity;
            gv.temperature_scale = gs.temperature_scale;
            gv.blackbody_intensity = gs.blackbody_intensity;
            gv.step_size = shader->quality.primaryStep(gas->getSettings().voxel_size);
            gv.max_steps = gs.max_steps;
            gv.shadow_strength = gs.shadow_strength;
            gv.shadow_steps = gs.shadow_steps;
            const float ambientKelvin = gas->getSettings().ambient_temperature;
            gv.emission_pad = ambientKelvin + shader->emission.temperature_min;
            gv.max_temperature = ambientKelvin + (std::max)(shader->emission.temperature_max, shader->emission.temperature_min + 1.0f);

            // Color Ramp (Legacy)
            gv.color_ramp_enabled = shader->emission.color_ramp.enabled ? 1 : 0;
            gv.ramp_stop_count = static_cast<int>((std::min)(shader->emission.color_ramp.stops.size(), static_cast<size_t>(8)));
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
    backend->updateGasVolumes(gpu_gas_volumes);
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
    if (world_data.mode != WORLD_MODE_NISHITA || !lut || !lut->is_initialized()) return color;
    if (!world_data.advanced.aerial_perspective) return color;
    if (!(dist > 0.0f) || !std::isfinite(dist)) return color;

    float Rg = world_data.nishita.planet_radius;
    if (Rg < 1000.0f) Rg = 6360000.0f; 
    
    // Accurate altitude based on origin (matches GPU)
    Vec3 p = origin + Vec3(0, Rg, 0);
    float current_altitude = p.length() - Rg;
    Vec3 up = p / (Rg + current_altitude);
    
    float cosTheta = (std::max)(0.01f, Vec3::dot(up, dir));
    float3 trans3 = lut->sampleTransmittance(cosTheta, current_altitude, world_data.nishita.atmosphere_height);
    Vec3 transmittance(trans3.x, trans3.y, trans3.z);
    
    float clampedDist = (std::min)(dist, 1000000.0f);
    
    const float min_dist = world_data.advanced.aerial_min_distance;
    const float max_dist = world_data.advanced.aerial_max_distance;
    float ramp = (clampedDist < min_dist) ? 0.0f : (std::min)(1.0f, (clampedDist - min_dist) / (std::max)(1.0f, max_dist - min_dist));
    
    float aerialDensity = (std::max)(0.0f, world_data.advanced.aerial_density);
    float atmosphereDensity = (std::max)(0.001f, world_data.nishita.air_density * 0.60f + world_data.nishita.dust_density * 0.40f);
    float densityFactor = aerialDensity * atmosphereDensity * (1.0f + world_data.nishita.fog_density * 120.0f);
    float distFactor = (1.0f - expf(-(clampedDist / 10000.0f) * densityFactor)) * (ramp * ramp);

    Vec3 finalTrans(
        powf(transmittance.x, distFactor),
        powf(transmittance.y, distFactor),
        powf(transmittance.z, distFactor)
    );
    
    float3 lookupDir = make_float3(dir.x, (std::max)(0.0f, dir.y), dir.z);
    float3 skyRadiance3 = lut->sampleSkyView(lookupDir, world_data.nishita.sun_direction, Rg, Rg + world_data.nishita.atmosphere_height);
    Vec3 skyRadiance(skyRadiance3.x, skyRadiance3.y, skyRadiance3.z);
    
    // Blend with scene
    Vec3 res = color * finalTrans + skyRadiance * (Vec3(1,1,1) - finalTrans);

    // DYNAMIC HEIGHT FOG OVERLAY (Nishita active)
    if (world_data.nishita.fog_enabled && world_data.nishita.fog_density > 0.01f) {
        float fAmount = calculate_height_fog_factor_cpu(
            origin, dir, clampedDist,
            world_data.nishita.fog_density * 0.5f,
            world_data.nishita.fog_height,
            world_data.nishita.fog_falloff
        );
        Vec3 fogCol(world_data.nishita.fog_color.x, world_data.nishita.fog_color.y, world_data.nishita.fog_color.z);
        res = res * (1.0f - fAmount) + fogCol * fAmount;
    }

    return res;
}
