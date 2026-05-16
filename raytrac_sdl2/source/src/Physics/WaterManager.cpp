#include "WaterSystem.h"
#include "scene_data.h"
#include "Renderer.h"
#include "OptixWrapper.h"
#include "Backend/IBackend.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "WaterMaterialSync.h"
#include "globals.h"
#include "fft_ocean.cuh"
#include "perlin.h" // For geometric waves
#include "KeyframeSystem.h" // For WaterKeyframe
#include <map>      // For smooth normal calculation
#include <cfloat>

// CUDA Library Linking
#pragma comment(lib, "cufft.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "delayimp.lib") // Required for delay load in MSVC

// #include "GeometryUtils.h" // Removed: Not needed for manual mesh generation

WaterSurface* WaterManager::getWaterSurface(int id) {
    for (auto& surf : water_surfaces) {
        if (surf.id == id) return &surf;
    }
    return nullptr;
}

void WaterManager::removeWaterSurface(SceneData& scene, int id) {
    // 1. Check if surface exists
    auto it = std::find_if(water_surfaces.begin(), water_surfaces.end(), 
        [id](const WaterSurface& ws) { return ws.id == id; });
        
    if (it == water_surfaces.end()) return;
    
    // Cleanup FFT resources
    if (it->fft_state) {
        FFTOceanState* state = static_cast<FFTOceanState*>(it->fft_state);
        cleanupFFTOcean(state);
        delete state;
        it->fft_state = nullptr;
    }
    
    // Cleanup GPU Geometric Wave resources
    if (it->gpu_geo_state && g_hasCUDA) {
        GPUGeoWaveState* gpu_state = static_cast<GPUGeoWaveState*>(it->gpu_geo_state);
        cleanupGPUGeometricWaves(gpu_state);
        delete gpu_state;
        it->gpu_geo_state = nullptr;
    }
    
    // 2. Remove triangles from scene
    for (auto& tri : it->mesh_triangles) {
        auto obj_it = std::find(scene.world.objects.begin(), scene.world.objects.end(), tri);
        if (obj_it != scene.world.objects.end()) {
            scene.world.objects.erase(obj_it);
        }
    }
    
    // 3. Remove from manager
    water_surfaces.erase(it);
}

void WaterManager::clear() {
    // CRITICAL: Clear GPU material FFT handles BEFORE destroying FFT state
    // This prevents OptiX from trying to sample destroyed textures
    for (auto& surf : water_surfaces) {
         // Clear GPU material FFT texture handles first
         if (surf.material_id > 0) {
             auto mat = MaterialManager::getInstance().getMaterial(surf.material_id);
             if (mat && mat->gpuMaterial) {
                 mat->gpuMaterial->fft_height_tex = 0;
                 mat->gpuMaterial->fft_normal_tex = 0;
             }
         }
         
         // Now cleanup FFT state
         if (surf.fft_state) {
             FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
             if (g_hasCUDA) {
                 cleanupFFTOcean(state);
             }
             delete state;
             surf.fft_state = nullptr;
         }
         // Cleanup GPU Geometric Wave resources
         if (surf.gpu_geo_state) {
             GPUGeoWaveState* gpu_state = static_cast<GPUGeoWaveState*>(surf.gpu_geo_state);
             if (g_hasCUDA) {
                 cleanupGPUGeometricWaves(gpu_state);
             }
             delete gpu_state;
             surf.gpu_geo_state = nullptr;
         }
    }
    water_surfaces.clear();
    next_id = 1;
    last_resolved_preview_time = 0.0f;
    static_preview_time = 0.0f;
    last_simulation_time = 0.0f;
    has_last_resolved_preview_time = false;
    has_last_simulation_time = false;
}

WaterUpdateResult WaterManager::update(float waterTime) {
    WaterUpdateResult result;
    const bool time_changed = !has_last_simulation_time || fabsf(waterTime - last_simulation_time) > 1e-6f;
    bool has_animated_water = false;

    for (auto& surf : water_surfaces) {
        // ════════════════════════════════════════════════════════════════════════
        // FFT OCEAN UPDATE (GPU-side animation - shader based)
        // ════════════════════════════════════════════════════════════════════════
        if (surf.params.use_fft_ocean && g_hasCUDA) {
            // Manage FFT State
            if (!surf.fft_state) {
                 FFTOceanState* state = new FFTOceanState();
                 surf.fft_state = (void*)state;
            }
            
            FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
            
            // Map parameters
            FFTOceanParams fft_params;
            fft_params.resolution = surf.params.fft_resolution;
            fft_params.ocean_size = resolveWaveDomainSize(&surf);
            fft_params.wind_speed = surf.params.fft_wind_speed;
            fft_params.wind_direction = surf.params.fft_wind_direction;
            fft_params.choppiness = surf.params.fft_choppiness;
            fft_params.amplitude = surf.params.fft_amplitude;
            fft_params.time_scale = resolveSharedAnimationSpeed(&surf);
            
            // Check initialization
            if (!state->initialized || state->current_resolution != fft_params.resolution) {
                if (initFFTOcean(state, &fft_params)) {
                    result.material_changed = true;
                }
            }
            
            // CRITICAL: Only proceed if state is properly initialized
            // If initFFTOcean failed, skip FFT processing to avoid access violation
            if (!state->initialized) {
                continue; // Skip to next water surface
            }

            // Check if FFT parameters changed (wind/amplitude etc)
            // If they did, updateFFTOcean will regenerate the spectrum
            bool fft_params_changed = (
                fft_params.ocean_size != state->cached_params.ocean_size ||
                fft_params.wind_speed != state->cached_params.wind_speed ||
                fft_params.wind_direction != state->cached_params.wind_direction ||
                fft_params.amplitude != state->cached_params.amplitude ||
                fft_params.choppiness != state->cached_params.choppiness ||
                fft_params.time_scale != state->cached_params.time_scale
            );
            
            // Run simulation
            updateFFTOcean(state, &fft_params, waterTime);
            
            // If FFT params changed, we need to signal for accumulation reset
            if (fft_params_changed) {
                result.material_changed = true;
            }

            // Vulkan cannot sample CUDA texture objects directly; its FFT textures are re-uploaded
            // from CPU-side FFT downloads when materials sync. Mark animated FFT surfaces dirty so
            // backend material upload can refresh those sampled textures without crashing on no-CUDA.
            if (time_changed) {
                result.material_changed = true;
            }
            
            // Connect to Material
            if (surf.material_id > 0) {
                auto mat = MaterialManager::getInstance().getMaterial(surf.material_id);
                if (mat && mat->gpuMaterial) {
                    if (mat->gpuMaterial->fft_height_tex != state->tex_height ||
                        mat->gpuMaterial->fft_normal_tex != state->tex_normal) {
                        
                        mat->gpuMaterial->fft_height_tex = state->tex_height;
                        mat->gpuMaterial->fft_normal_tex = state->tex_normal;
                        result.material_changed = true;
                    }
                }
            }
        } else {
             // Cleanup if disabled but state exists
             if (surf.fft_state) {
                 FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
                 // Only cleanup if we actually have CUDA, or just delete the state pointer if CUDA is gone/lost
                 if (g_hasCUDA) {
                     cleanupFFTOcean(state);
                 }
                 delete state;
                 surf.fft_state = nullptr;
                 result.material_changed = true;
                 
                 if (surf.material_id > 0) {
                     auto mat = MaterialManager::getInstance().getMaterial(surf.material_id);
                     if (mat && mat->gpuMaterial) {
                         mat->gpuMaterial->fft_height_tex = 0;
                         mat->gpuMaterial->fft_normal_tex = 0;
                     }
                 }
             }
        }
        
        // ════════════════════════════════════════════════════════════════════════
        // FFT-DRIVEN MESH DISPLACEMENT (Highest Quality - New!)
        // ════════════════════════════════════════════════════════════════════════
        // Uses FFT ocean data to displace mesh vertices - combines the best of both:
        // Film-quality FFT waves + Physical mesh for raytracing/shadows
        if (surf.params.use_fft_ocean && surf.params.use_fft_mesh_displacement) {
            has_animated_water = true;
            float target_animation_time = waterTime * resolveSharedAnimationSpeed(&surf);
            if (!has_last_simulation_time || fabsf(target_animation_time - surf.animation_time) > 1e-6f) {
                surf.animation_time = target_animation_time;
                surf.animate_mesh = true;
                result.mesh_changed = updateFFTDrivenMesh(&surf, surf.animation_time) || result.mesh_changed;
            }
        }
        // ════════════════════════════════════════════════════════════════════════
        // GEOMETRIC WAVES - MESH ANIMATION (GPU or CPU) - Legacy/Alternative
        // ════════════════════════════════════════════════════════════════════════
        // This runs independently of FFT - uses procedural noise or Gerstner waves
        // Skip if FFT mesh displacement is active (avoid double displacement)
        else if (surf.params.use_geometric_waves && !surf.params.use_fft_mesh_displacement) {
            has_animated_water = true;
            float target_animation_time = waterTime * resolveSharedAnimationSpeed(&surf);
            if (!has_last_simulation_time || fabsf(target_animation_time - surf.animation_time) > 1e-6f) {
                surf.animation_time = target_animation_time;
                surf.animate_mesh = true;
                if (surf.use_gpu_animation && g_hasCUDA) {
                    // GPU Path - much faster for large meshes
                    result.mesh_changed = updateGPUAnimatedWaterMesh(&surf, surf.animation_time) || result.mesh_changed;
                } else {
                    // CPU Path - fallback
                    result.mesh_changed = updateAnimatedWaterMesh(&surf, surf.animation_time) || result.mesh_changed;
                }
            }
        }

        if (surf.params.use_fft_ocean || surf.animate_mesh || surf.params.wave_strength > 0.0001f) {
            has_animated_water = true;
        }
    }

    result.time_changed = time_changed && has_animated_water;
    last_simulation_time = waterTime;
    has_last_simulation_time = true;
    return result;
}

void WaterManager::setPreviewTimeMode(WaterPreviewTimeMode mode) {
    if (preview_time_mode == mode) return;

    if (mode == WaterPreviewTimeMode::Static) {
        static_preview_time = has_last_resolved_preview_time ? last_resolved_preview_time : 0.0f;
    }

    preview_time_mode = mode;
}

float WaterManager::resolvePreviewWaterTime(float realtimeSeconds, int timelineFrame, float fps) {
    float safeFps = fps > 0.0f ? fps : 24.0f;
    float timelineSeconds = static_cast<float>(timelineFrame) / safeFps;
    float resolved = realtimeSeconds;

    switch (preview_time_mode) {
        case WaterPreviewTimeMode::Realtime:
            resolved = realtimeSeconds;
            break;
        case WaterPreviewTimeMode::Timeline:
            resolved = timelineSeconds;
            break;
        case WaterPreviewTimeMode::Static:
            if (!has_last_resolved_preview_time) {
                static_preview_time = timelineSeconds;
            }
            resolved = static_preview_time;
            break;
        default:
            break;
    }

    last_resolved_preview_time = resolved;
    has_last_resolved_preview_time = true;
    return resolved;
}

float WaterManager::getLegacyDomainReferenceSize() const {
    return 20.0f;
}

float WaterManager::getSurfaceWorldExtent(const WaterSurface* surf) const {
    if (!surf) {
        return getLegacyDomainReferenceSize();
    }

    float min_x = FLT_MAX;
    float max_x = -FLT_MAX;
    float min_z = FLT_MAX;
    float max_z = -FLT_MAX;
    bool has_positions = false;

    auto accumulate = [&](const Vec3& p) {
        min_x = fminf(min_x, p.x);
        max_x = fmaxf(max_x, p.x);
        min_z = fminf(min_z, p.z);
        max_z = fmaxf(max_z, p.z);
        has_positions = true;
    };

    for (const Vec3& p : surf->original_positions) {
        accumulate(p);
    }

    if (!has_positions) {
        for (const auto& tri : surf->mesh_triangles) {
            if (!tri) continue;
            for (int v = 0; v < 3; ++v) {
                accumulate(tri->getOriginalVertexPosition(v));
            }
        }
    }

    float extent_x = getLegacyDomainReferenceSize();
    float extent_z = getLegacyDomainReferenceSize();
    if (has_positions) {
        extent_x = fmaxf(max_x - min_x, 0.001f);
        extent_z = fmaxf(max_z - min_z, 0.001f);
    }

    Vec3 world_scale(1.0f, 1.0f, 1.0f);
    if (surf->reference_triangle) {
        if (Transform* t = surf->reference_triangle->getTransformPtr()) {
            world_scale = t->scale;
        }
    }

    extent_x *= fmaxf(fabsf(world_scale.x), 0.001f);
    extent_z *= fmaxf(fabsf(world_scale.z), 0.001f);
    return fmaxf(fmaxf(extent_x, extent_z), 0.001f);
}

float WaterManager::resolveWaveDomainSize(const WaterSurface* surf) const {
    if (!surf) {
        return getLegacyDomainReferenceSize();
    }

    // Keep water shading and FFT tiling on the authored domain. Coupling this
    // to mesh/world extent makes OptiX and Vulkan-RT flatten micro ripples on
    // large scaled water planes.
    return fmaxf(surf->params.fft_ocean_size, 0.001f);
}

float WaterManager::resolveSharedAnimationSpeed(const WaterSurface* surf) const {
    if (!surf) {
        return 1.0f;
    }

    const float fft_speed = surf->params.fft_time_scale;
    const float geo_speed = surf->params.geo_wave_speed;
    const bool fft_valid = fabsf(fft_speed) > 1e-6f;
    const bool geo_valid = fabsf(geo_speed) > 1e-6f;

    if (fft_valid && geo_valid) {
        return fft_speed;
    }
    if (fft_valid) {
        return fft_speed;
    }
    if (geo_valid) {
        return geo_speed;
    }
    return 1.0f;
}

void WaterManager::syncSurfaceMaterial(WaterSurface* surf) {
    if (!surf || surf->material_id == 0) return;

    auto mat = MaterialManager::getInstance().getMaterial(surf->material_id);
    if (!mat) return;

    const float resolved_domain_size = resolveWaveDomainSize(surf);
    const float resolved_animation_speed = resolveSharedAnimationSpeed(surf);
    WaterShader::SurfaceParams shader_params = surf->params.toShaderParams(0.0f, resolved_domain_size);
    shader_params.animation_speed = resolved_animation_speed;

    auto pbsdf = dynamic_cast<PrincipledBSDF*>(mat);
    if (pbsdf) {
        WaterShader::applySurfaceParamsToPrincipledBSDF(shader_params, *pbsdf);
    }

    if (mat->gpuMaterial) {
        auto& gpu = mat->gpuMaterial;
        cudaTextureObject_t fft_height_tex = 0;
        cudaTextureObject_t fft_normal_tex = 0;
        if (g_hasCUDA && surf->params.use_fft_ocean && surf->fft_state) {
            FFTOceanState* state = static_cast<FFTOceanState*>(surf->fft_state);
            if (state && state->initialized && state->tex_height != 0 && state->tex_normal != 0) {
                fft_height_tex = state->tex_height;
                fft_normal_tex = state->tex_normal;
            }
        }
        WaterShader::applySurfaceParamsToGpuMaterial(shader_params, *gpu, fft_height_tex, fft_normal_tex);
    }
}

bool WaterManager::syncVulkanFFTTexturesForMaterial(uint16_t material_id, Backend::IBackend* backend, int64_t& outHeightTexture, int64_t& outNormalTexture) {
    outHeightTexture = 0;
    outNormalTexture = 0;

    if (!backend || !g_hasCUDA) {
        return false;
    }

    for (auto& surf : water_surfaces) {
        if (surf.material_id != material_id || !surf.params.use_fft_ocean || !surf.fft_state) {
            continue;
        }

        FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
        if (!state || !state->initialized) {
            return false;
        }

        int resolution = 0;
        if (!downloadFFTOceanData(state, nullptr, nullptr, nullptr, nullptr, nullptr, &resolution) || resolution <= 0) {
            return false;
        }

        const size_t texelCount = static_cast<size_t>(resolution) * static_cast<size_t>(resolution);
        std::vector<float> heightData(texelCount);
        std::vector<float> normalX(texelCount);
        std::vector<float> normalZ(texelCount);
        if (!downloadFFTOceanData(state, heightData.data(), nullptr, nullptr, normalX.data(), normalZ.data(), &resolution) || resolution <= 0) {
            return false;
        }

        // FFT images are live descriptor resources in Vulkan. Complete in-flight work
        // before replacing them so fallback stays safe on slower or discrete GPUs.
        backend->waitForCompletion();

        std::vector<float4> normalData(texelCount);
        for (size_t i = 0; i < texelCount; ++i) {
            const float nx = normalX[i];
            const float nz = normalZ[i];
            normalData[i] = make_float4(nx, nz, 0.0f, 1.0f);
        }

        const int64_t newHeightTexture = backend->uploadTexture2D(heightData.data(), static_cast<uint32_t>(resolution), static_cast<uint32_t>(resolution), 1, false, true);
        if (!newHeightTexture) {
            return false;
        }

        const int64_t newNormalTexture = backend->uploadTexture2D(normalData.data(), static_cast<uint32_t>(resolution), static_cast<uint32_t>(resolution), 4, false, true);
        if (!newNormalTexture) {
            backend->destroyTexture(newHeightTexture);
            return false;
        }

        if (surf.vulkan_fft_height_texture) {
            backend->destroyTexture(surf.vulkan_fft_height_texture);
        }
        if (surf.vulkan_fft_normal_texture) {
            backend->destroyTexture(surf.vulkan_fft_normal_texture);
        }

        surf.vulkan_fft_height_texture = newHeightTexture;
        surf.vulkan_fft_normal_texture = newNormalTexture;
        outHeightTexture = newHeightTexture;
        outNormalTexture = newNormalTexture;
        return true;
    }

    return false;
}

cudaTextureObject_t WaterManager::getFirstFFTHeightMap() {
    for (const auto& surf : water_surfaces) {
        if (g_hasCUDA && surf.params.use_fft_ocean && surf.fft_state) {
            FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
            if (state && state->initialized && state->tex_height != 0) {
                return state->tex_height;
            }
        }
    }
    return 0;
}

WaterSurface* WaterManager::createWaterPlane(SceneData& scene, const Vec3& pos, float size, float density) {
    WaterSurface surf;
    surf.id = next_id++;
    surf.name = "Water_Plane_" + std::to_string(surf.id);
    
    // 1. Create unique Water Material
    auto water_mat = std::make_shared<PrincipledBSDF>();
    auto gpu = std::make_shared<GpuMaterial>();
    
    // === BASE WATER MATERIAL ===
    // Albedo controls transmission tint - use deep_color for Beer's law
    gpu->albedo = make_float3(
        surf.params.deep_color.x, 
        surf.params.deep_color.y, 
        surf.params.deep_color.z
    );
    gpu->transmission = 1.0f;
    gpu->opacity = 1.0f;
    gpu->roughness = surf.params.roughness;
    gpu->ior = surf.params.ior;
    gpu->metallic = 0.0f;
    gpu->flags |= GPU_MAT_FLAG_WATER;
    
    // === WAVE PARAMS (original packing) ===
    // anisotropic -> Wave Speed
    // sheen -> Wave Strength (Serves as IS_WATER flag if > 0)
    // sheen_tint -> Wave Frequency
    gpu->anisotropic = surf.params.wave_speed;
    gpu->sheen = fmaxf(0.001f, surf.params.wave_strength);  // >0 = IS_WATER flag (always needed)
    gpu->sheen_tint = surf.params.wave_frequency;
    
    // === ADVANCED WATER PARAMS (new packing) ===
    // clearcoat -> Shore Foam Intensity
    // clearcoat_roughness -> Caustic Intensity
    gpu->clearcoat = surf.params.shore_foam_intensity;
    gpu->clearcoat_roughness = surf.params.caustic_intensity;
    
    // subsurface -> Depth Max (scaled: divide by 100 to fit 0-1 range)
    // subsurface_scale -> Absorption Density
    gpu->subsurface = surf.params.depth_max / 100.0f;
    gpu->subsurface_scale = surf.params.absorption_density;
    
    // subsurface_color -> Absorption Color
    gpu->subsurface_color = make_float3(
        surf.params.absorption_color.x,
        surf.params.absorption_color.y,
        surf.params.absorption_color.z
    );
    
    // subsurface_radius -> (shore_foam_distance, caustic_scale, sss_intensity)
    gpu->subsurface_radius = make_float3(
        surf.params.shore_foam_distance,
        surf.params.caustic_scale,
        surf.params.sss_intensity
    );
    
    // emission -> Shallow Color (repurposed for water)
    gpu->emission = make_float3(
        surf.params.shallow_color.x,
        surf.params.shallow_color.y,
        surf.params.shallow_color.z
    );
    
    // translucent -> Foam Level
    gpu->translucent = surf.params.foam_level;
    
    // subsurface_anisotropy -> Caustic Speed
    gpu->subsurface_anisotropy = surf.params.caustic_speed;

    // Water Details (New)
    gpu->micro_detail_strength = surf.params.micro_detail_strength;
    gpu->micro_detail_scale = surf.params.micro_detail_scale;
    gpu->micro_anim_speed = surf.params.micro_anim_speed;
    gpu->micro_morph_speed = surf.params.micro_morph_speed;
    gpu->foam_noise_scale = surf.params.foam_noise_scale;
    gpu->foam_threshold = surf.params.foam_threshold;
    
    // FFT
    gpu->fft_ocean_size = resolveWaveDomainSize(&surf);
    gpu->fft_choppiness = surf.params.fft_choppiness;
    
    // Sync pbsdf properties so Renderer/Vulkan path reads correct values
    water_mat->albedoProperty.color = Vec3(surf.params.deep_color.x, surf.params.deep_color.y, surf.params.deep_color.z);
    water_mat->emissionProperty.color = Vec3(surf.params.shallow_color.x, surf.params.shallow_color.y, surf.params.shallow_color.z);
    water_mat->emissionProperty.intensity = 1.0f;

    water_mat->gpuMaterial = gpu;
    
    // Register material
    std::string mat_name = "Water_Mat_" + std::to_string(surf.id);
    surf.material_id = MaterialManager::getInstance().getOrCreateMaterialID(mat_name, water_mat);
    syncSurfaceMaterial(&surf);
    
    // 2. Generate Grid Mesh (NxN triangles for waves)
    // Resolution based on density
    int segments = static_cast<int>(size * density);
    if (segments < 2) segments = 2;
    if (segments > 256) segments = 256; // Limit for safety
    
    float step = size / segments;
    // Create vertices around origin (local space) - pivot will be at center
    float half_size = size * 0.5f;
    
    // Transform stores the actual world position
    std::shared_ptr<Transform> shared_transform = std::make_shared<Transform>();
    Matrix4x4 world_transform = Matrix4x4::translation(pos);
    shared_transform->setBase(world_transform);
    
    for (int z = 0; z < segments; z++) {
        for (int x = 0; x < segments; x++) {
            // Local space coordinates (centered around origin)
            float x0 = -half_size + (x * step);
            float z0 = -half_size + (z * step);
            float x1 = x0 + step;
            float z1 = z0 + step;
            
            // Grid cell vertices in local space (y=0 at local origin)
            Vec3 v0(x0, 0, z0);
            Vec3 v1(x1, 0, z0);
            Vec3 v2(x1, 0, z1);
            Vec3 v3(x0, 0, z1);
            
            // UVs
            float u0 = (float)x / segments;
            float v_0 = (float)z / segments; // v_0 to avoid variable name conflict
            float u1 = (float)(x + 1) / segments;
            float v_1 = (float)(z + 1) / segments;
            
            Vec3 n(0, 1, 0); // Up normal
            
            // Triangle 1
            auto tri1 = std::make_shared<Triangle>(v0, v1, v2, n, n, n, Vec2(u0, v_0), Vec2(u1, v_0), Vec2(u1, v_1), surf.material_id);
            tri1->setTransformHandle(shared_transform);
            tri1->setNodeName(surf.name);
            
            // Triangle 2
            auto tri2 = std::make_shared<Triangle>(v0, v2, v3, n, n, n, Vec2(u0, v_0), Vec2(u1, v_1), Vec2(u0, v_1), surf.material_id);
            tri2->setTransformHandle(shared_transform);
            tri2->setNodeName(surf.name);
            
            surf.mesh_triangles.push_back(tri1);
            surf.mesh_triangles.push_back(tri2);
            
            // Add to scene
            scene.world.objects.push_back(tri1);
            scene.world.objects.push_back(tri2);
        }
    }
    
    if (!surf.mesh_triangles.empty()) {
        surf.reference_triangle = surf.mesh_triangles[0];
    }
    
    water_surfaces.push_back(surf);
    return &water_surfaces.back();
}

void WaterManager::updateWaterMesh(WaterSurface* surf) {
    if (!surf) return;

    invalidateGeometricAnimationState(surf);
    if (surf->type != WaterSurface::Type::Plane) return;

    bool use_geo = surf->params.use_geometric_waves;
    float max_height = surf->params.geo_wave_height;
    float scale = surf->params.geo_wave_scale;
    float chop = surf->params.geo_wave_choppiness;

    // Get world-space scale from transform so noise coords are scale-independent
    Vec3 world_scale(1.0f, 1.0f, 1.0f);
    if (surf->reference_triangle) {
        if (Transform* t = surf->reference_triangle->getTransformPtr()) {
            world_scale = t->scale;
            if (fabsf(world_scale.x) < 1e-6f) world_scale.x = 1.0f;
            if (fabsf(world_scale.z) < 1e-6f) world_scale.z = 1.0f;
        }
    }
    const float domain_size = resolveWaveDomainSize(surf);
    const float domain_coord_scale = getLegacyDomainReferenceSize() / fmaxf(domain_size, 0.001f);
    world_scale.x *= domain_coord_scale;
    world_scale.z *= domain_coord_scale;

    if (scale < 0.001f) scale = 0.001f;
    
    // Static perlin instance
    static Perlin perlin;
    
    // Noise Parameters
    int octaves = surf->params.geo_octaves;
    float persistence = surf->params.geo_persistence;
    float lacunarity = surf->params.geo_lacunarity;
    float ridge_offset = surf->params.geo_ridge_offset;
    auto noise_type = surf->params.geo_noise_type;
    
    // Ocean parameters
    float damping = surf->params.geo_damping;
    float alignment = surf->params.geo_alignment;
    float depth = surf->params.geo_depth;
    float swell_dir = surf->params.geo_swell_direction * 3.14159265f / 180.0f; // to radians
    float swell_amp = surf->params.geo_swell_amplitude;
    float sharpening = surf->params.geo_sharpening;
    float detail_scale = surf->params.geo_detail_scale;
    float detail_strength = surf->params.geo_detail_strength;
    bool smooth_normals = surf->params.geo_smooth_normals;

    // ════════════════════════════════════════════════════════════════════════
    // GERSTNER WAVE HELPER (Tessendorf-inspired, physically-based circular wave)
    // ════════════════════════════════════════════════════════════════════════
    struct GerstnerWave {
        float amplitude;
        float k;        // wavenumber = 2*PI/wavelength
        float speed;
        float Q;        // steepness (pre-divided by k*amp*numWaves)
        float dx, dz;   // direction unit vector
    };

    // Build wave array ONCE outside the per-vertex lambda (avoid per-vertex heap allocation)
    const int NUM_WAVES = 6 + (swell_amp > 0.001f ? 1 : 0);
    std::vector<GerstnerWave> gerstner_waves;
    if (use_geo && noise_type == WaterWaveParams::NoiseType::Gerstner) {
        gerstner_waves.reserve(NUM_WAVES);
        float dirSpread = (1.0f - alignment) * 3.14159265f * 0.5f;
        for (int i = 0; i < 6; ++i) {
            float freqMult = powf(lacunarity, (float)i);
            float ampMult  = powf(persistence, (float)i);
            float wavelength = scale / freqMult;
            float amplitude  = max_height * ampMult * 0.25f;
            float dirOffset  = ((float)i - 2.5f) / 2.5f * dirSpread;
            float dampFactor = 1.0f - damping * sinf(fabsf(dirOffset));
            amplitude *= fmaxf(0.1f, dampFactor);
            float dir = swell_dir + dirOffset;
            float kk  = 2.0f * 3.14159265f / wavelength;
            float steepness = fminf(1.0f, chop * 0.5f);
            GerstnerWave w;
            w.k     = kk;
            w.speed = sqrtf(9.81f * kk);
            w.amplitude = amplitude;
            w.Q     = fminf(1.0f, steepness / (kk * amplitude * 6.0f));
            w.dx    = cosf(dir);
            w.dz    = sinf(dir);
            gerstner_waves.push_back(w);
        }
        if (swell_amp > 0.001f) {
            float wavelength = scale * 3.0f;
            float amplitude  = max_height * swell_amp * 0.5f;
            float kk         = 2.0f * 3.14159265f / wavelength;
            GerstnerWave w;
            w.k        = kk;
            w.speed    = sqrtf(9.81f * kk);
            w.amplitude = amplitude;
            w.Q        = fminf(1.0f, 0.2f / (kk * amplitude * (float)gerstner_waves.size() + 1.0f));
            w.dx       = cosf(swell_dir + 0.5f);
            w.dz       = sinf(swell_dir + 0.5f);
            gerstner_waves.push_back(w);
        }
    }

    auto getGerstnerDisplacement = [&](float x, float z, float time = 0.0f) -> Vec3 {
        if (!use_geo || noise_type != WaterWaveParams::NoiseType::Gerstner)
            return Vec3(x, 0.0f, z);

        // Convert to world space for wave evaluation
        float wx = x * world_scale.x;
        float wz = z * world_scale.z;

        Vec3 result(0.0f, 0.0f, 0.0f);
        for (const auto& w : gerstner_waves) {
            float phase = w.k * (wx * w.dx + wz * w.dz) - w.speed * time;
            result.x += w.Q * w.amplitude * w.dx * cosf(phase);
            result.y += w.amplitude * sinf(phase);
            result.z += w.Q * w.amplitude * w.dz * cosf(phase);
        }
        return Vec3(x + result.x * chop, result.y, z + result.z * chop);
    };
    
    // ════════════════════════════════════════════════════════════════════════
    // TESSENDORF SIMPLIFIED (Predictable procedural ocean without FFT)
    // ════════════════════════════════════════════════════════════════════════
    auto getTessendorfSimplified = [&](float x, float z) -> float {
        if (!use_geo || noise_type != WaterWaveParams::NoiseType::TessendorfSimple) 
            return 0.0f;
        
        float height = 0.0f;
        float amp = max_height;
        float freq = 1.0f / scale;
        
        // Base direction from swell
        float dirX = cosf(swell_dir);
        float dirZ = sinf(swell_dir);
        
        for (int i = 0; i < octaves; ++i) {
            // Rotate direction slightly per octave
            float angle = (float)i * 0.3f * (1.0f - alignment);
            float dx = cosf(swell_dir + angle);
            float dz = sinf(swell_dir + angle);
            
            float phase = (x * world_scale.x * dx + z * world_scale.z * dz) * freq;
            float wave = sinf(phase * 2.0f * 3.14159265f);
            
            // Apply sharpening (sharper peaks)
            if (sharpening > 0.001f) {
                wave = powf(fabsf(wave), 1.0f - sharpening * 0.5f) * (wave >= 0 ? 1.0f : -1.0f);
            }
            
            // Damping for perpendicular
            float angleDiff = fabsf(angle);
            float dampFactor = 1.0f - damping * sinf(angleDiff);
            
            height += wave * amp * fmaxf(0.1f, dampFactor);
            
            amp *= persistence;
            freq *= lacunarity;
        }
        
        return height;
    };

    // ════════════════════════════════════════════════════════════════════════
    // ADVANCED NOISE GENERATION (Original noise types with detail layer)
    // ════════════════════════════════════════════════════════════════════════
    auto getNoiseValue = [&](float x, float z) -> float {
        if (!use_geo) return 0.0f;
        
        // Handle special wave types
        if (noise_type == WaterWaveParams::NoiseType::TessendorfSimple) {
            return getTessendorfSimplified(x, z);
        }
        
        float nx = (x * world_scale.x) / scale;
        float nz = (z * world_scale.z) / scale;
        
        float value = 0.0f;
        float amp = 1.0f;
        float freq = 1.0f;
        float maxAmp = 0.0f;
        float weight = 1.0f;
        
        for (int i = 0; i < octaves; i++) {
            Vec3 p(nx * freq, 0.0f, nz * freq); 
            float n = perlin.noise(p);
            
            if (noise_type == WaterWaveParams::NoiseType::Ridge) {
                // Ridged Multifractal
                n = ridge_offset - fabsf(n);
                n = n * n;
                if (chop > 0.0f) n = powf(fmaxf(0.0f, n), chop);
                n *= weight;
                weight = fmaxf(0.0f, fminf(1.0f, n * 2.0f));
            } 
            else if (noise_type == WaterWaveParams::NoiseType::Billow) {
                n = fabsf(n);
                n = 2.0f * n - 1.0f;
                n = fabsf(n); 
            }
            else if (noise_type == WaterWaveParams::NoiseType::FBM) {
                // Standard FBM
            }
            else if (noise_type == WaterWaveParams::NoiseType::Perlin) {
                if (i > 0) {
                     maxAmp = 1.0f;
                     continue; 
                }
            }
            else if (noise_type == WaterWaveParams::NoiseType::Voronoi) {
                // Simple worley-like approximation
                float vx = floorf(nx * freq);
                float vz = floorf(nz * freq);
                float minDist = 1.0f;
                for (int ox = -1; ox <= 1; ++ox) {
                    for (int oz = -1; oz <= 1; ++oz) {
                        Vec3 cellP(vx + ox + 0.5f, 0, vz + oz + 0.5f);
                        float hash = perlin.noise(cellP * 0.1f) * 0.5f + 0.5f;
                        float cellX = vx + ox + hash;
                        float cellZ = vz + oz + hash * 0.7f;
                        float dist = sqrtf((nx * freq - cellX) * (nx * freq - cellX) + 
                                          (nz * freq - cellZ) * (nz * freq - cellZ));
                        minDist = fminf(minDist, dist);
                    }
                }
                n = minDist * 2.0f - 1.0f;
            }
            
            value += n * amp;
            maxAmp += amp;
            amp *= persistence;
            freq *= lacunarity;
        }
        
        // Normalization
        if (noise_type != WaterWaveParams::NoiseType::Ridge && maxAmp > 0.001f) {
            value /= maxAmp;
        }
        
        // Add detail layer (high-frequency ripples)
        if (detail_strength > 0.001f) {
            float dx = (x * world_scale.x) / (scale / detail_scale);
            float dz = (z * world_scale.z) / (scale / detail_scale);
            float detail = perlin.noise(Vec3(dx, 0, dz)) * detail_strength;
            value += detail;
        }
        
        // Apply sharpening (sharper wave peaks)
        if (sharpening > 0.001f && value > 0.0f) {
            value = powf(value, 1.0f + sharpening);
        }
        
        return value * max_height;
    };
    
    // ════════════════════════════════════════════════════════════════════════
    // PHASE 1: Apply height displacement to all vertices
    // ════════════════════════════════════════════════════════════════════════
    
    // First, collect unique vertex positions and their heights
    // Using a map to store vertex position -> height
    struct VertexKey {
        int ix, iz; // Grid indices based on position
        bool operator<(const VertexKey& o) const {
            if (ix != o.ix) return ix < o.ix;
            return iz < o.iz;
        }
    };
    
    std::map<VertexKey, Vec3> vertexPositions; // Displaced positions
    std::map<VertexKey, Vec3> vertexNormals;   // Accumulated normals
    std::map<VertexKey, int> vertexCounts;     // Count for averaging
    
    float epsilon = 0.001f;
    auto makeKey = [epsilon](const Vec3& p) -> VertexKey {
        // Quantize to grid to handle floating point precision
        return { (int)roundf(p.x * 100.0f), (int)roundf(p.z * 100.0f) };
    };
    
    // First pass: Displace vertices and store positions
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        
        for (int v = 0; v < 3; ++v) {
            Vec3 p = tri->getOriginalVertexPosition(v);
            VertexKey key = makeKey(p);
            
            if (vertexPositions.find(key) == vertexPositions.end()) {
                Vec3 displaced = p;
                
                if (noise_type == WaterWaveParams::NoiseType::Gerstner) {
                    displaced = getGerstnerDisplacement(p.x, p.z, 0.0f);
                } else {
                    displaced.y = getNoiseValue(p.x, p.z);
                }
                
                vertexPositions[key] = displaced;
                vertexNormals[key] = Vec3(0, 0, 0);
                vertexCounts[key] = 0;
            }
        }
    }
    
    // ════════════════════════════════════════════════════════════════════════
    // PHASE 2: Calculate face normals and accumulate for smooth shading
    // ════════════════════════════════════════════════════════════════════════
    
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        
        // Get displaced positions for this triangle
        Vec3 orig0 = tri->getOriginalVertexPosition(0);
        Vec3 orig1 = tri->getOriginalVertexPosition(1);
        Vec3 orig2 = tri->getOriginalVertexPosition(2);
        
        VertexKey k0 = makeKey(orig0);
        VertexKey k1 = makeKey(orig1);
        VertexKey k2 = makeKey(orig2);
        
        Vec3 p0 = vertexPositions[k0];
        Vec3 p1 = vertexPositions[k1];
        Vec3 p2 = vertexPositions[k2];
        
        // Calculate face normal
        Vec3 edge1 = p1 - p0;
        Vec3 edge2 = p2 - p0;
        Vec3 faceNormal = edge1.cross(edge2);
        float len = faceNormal.length();
        if (len > 0.0001f) {
            faceNormal = faceNormal / len;
        } else {
            faceNormal = Vec3(0, 1, 0);
        }
        
        // Weight by face area (larger faces contribute more to smooth normal)
        float area = len * 0.5f;
        
        // Accumulate normals (for smooth shading)
        if (smooth_normals) {
            vertexNormals[k0] = vertexNormals[k0] + faceNormal * area;
            vertexNormals[k1] = vertexNormals[k1] + faceNormal * area;
            vertexNormals[k2] = vertexNormals[k2] + faceNormal * area;
            vertexCounts[k0]++;
            vertexCounts[k1]++;
            vertexCounts[k2]++;
        } else {
            // Flat shading: each vertex gets face normal
            vertexNormals[k0] = faceNormal;
            vertexNormals[k1] = faceNormal;
            vertexNormals[k2] = faceNormal;
        }
    }
    
    // Normalize accumulated normals
    for (auto& [key, normal] : vertexNormals) {
        float len = normal.length();
        if (len > 0.0001f) {
            normal = normal / len;
        } else {
            normal = Vec3(0, 1, 0);
        }
    }
    
    // ════════════════════════════════════════════════════════════════════════
    // PHASE 3: Apply displaced positions and smooth normals to triangles
    // ════════════════════════════════════════════════════════════════════════
    
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        
        for (int v = 0; v < 3; ++v) {
            Vec3 orig = tri->getOriginalVertexPosition(v);
            VertexKey key = makeKey(orig);
            
            Vec3 newPos = vertexPositions[key];
            Vec3 newNormal = vertexNormals[key];
            
            // Update position
            tri->setVertexPosition(v, newPos);
            tri->setOriginalVertexPosition(v, newPos);
            
            // Update normal (smooth or flat)
            tri->setVertexNormal(v, newNormal);
            tri->setOriginalVertexNormal(v, newNormal);
        }
        
        tri->markAABBDirty();
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// CACHE ORIGINAL POSITIONS (for animation base)
// ════════════════════════════════════════════════════════════════════════════════
void WaterManager::cacheOriginalPositions(WaterSurface* surf) {
    if (!surf) return;
    
    surf->original_positions.clear();
    
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        for (int v = 0; v < 3; ++v) {
            surf->original_positions.push_back(tri->getOriginalVertexPosition(v));
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// ANIMATED MESH UPDATE (time-based wave animation)
// ════════════════════════════════════════════════════════════════════════════════
bool WaterManager::updateAnimatedWaterMesh(WaterSurface* surf, float time) {
    if (!surf) return false;
    if (!surf->params.use_geometric_waves) return false;
    
    // Cache original positions if not done yet
    if (surf->original_positions.empty()) {
        // First time - need flat grid positions, not displaced ones
        // This is a fallback - ideally call cacheOriginalPositions after initial creation
        size_t idx = 0;
        for (auto& tri : surf->mesh_triangles) {
            if (!tri) continue;
            for (int v = 0; v < 3; ++v) {
                surf->original_positions.push_back(tri->getOriginalVertexPosition(v));
            }
        }
    }
    if (surf->original_positions.empty()) return false;
    
    static Perlin perlin;
    
    float max_height = surf->params.geo_wave_height;
    float scale = surf->params.geo_wave_scale;
    float chop = surf->params.geo_wave_choppiness;
    int octaves = surf->params.geo_octaves;
    float persistence = surf->params.geo_persistence;
    float lacunarity = surf->params.geo_lacunarity;
    float swell_dir = surf->params.geo_swell_direction * 3.14159265f / 180.0f;
    float swell_amp = surf->params.geo_swell_amplitude;
    float alignment = surf->params.geo_alignment;
    float damping = surf->params.geo_damping;
    auto noise_type = surf->params.geo_noise_type;

    if (scale < 0.001f) scale = 0.001f;

    // Get world-space scale from transform so noise coords are scale-independent
    Vec3 world_scale(1.0f, 1.0f, 1.0f);
    if (surf->reference_triangle) {
        Transform* rt = surf->reference_triangle->getTransformPtr();
        if (rt) {
            world_scale = rt->scale;
            if (fabsf(world_scale.x) < 1e-6f) world_scale.x = 1.0f;
            if (fabsf(world_scale.z) < 1e-6f) world_scale.z = 1.0f;
        }
    }

    // Pre-compute Gerstner wave parameters ONCE (not per-vertex)
    struct AnimGerstnerWave { float k, speed, amplitude, Q, dx, dz; };
    std::vector<AnimGerstnerWave> anim_waves;
    if (noise_type == WaterWaveParams::NoiseType::Gerstner ||
        noise_type == WaterWaveParams::NoiseType::TessendorfSimple) {
        anim_waves.reserve(7);
        float dirSpread = (1.0f - alignment) * 3.14159265f * 0.5f;
        for (int i = 0; i < 6; ++i) {
            float freqMult  = powf(lacunarity, (float)i);
            float ampMult   = powf(persistence, (float)i);
            float wavelength = scale / freqMult;
            float amplitude  = max_height * ampMult * 0.25f;
            float dirOffset  = ((float)i - 2.5f) / 2.5f * dirSpread;
            amplitude *= fmaxf(0.1f, 1.0f - damping * sinf(fabsf(dirOffset)));
            float dir  = swell_dir + dirOffset;
            float kk   = 2.0f * 3.14159265f / wavelength;
            float steepness = fminf(1.0f, chop * 0.5f);
            AnimGerstnerWave w;
            w.k         = kk;
            w.speed     = sqrtf(9.81f * kk);
            w.amplitude = amplitude;
            w.Q         = fminf(1.0f, steepness / (kk * amplitude * 6.0f));
            w.dx        = cosf(dir);
            w.dz        = sinf(dir);
            anim_waves.push_back(w);
        }
        if (swell_amp > 0.001f) {
            float wavelength = scale * 3.0f;
            float amplitude  = max_height * swell_amp * 0.5f;
            float kk         = 2.0f * 3.14159265f / wavelength;
            AnimGerstnerWave w;
            w.k         = kk;
            w.speed     = sqrtf(9.81f * kk);
            w.amplitude = amplitude;
            w.Q         = 0.0f; // swell has no horizontal displacement
            w.dx        = cosf(swell_dir + 0.5f);
            w.dz        = sinf(swell_dir + 0.5f);
            anim_waves.push_back(w);
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // ANIMATED GERSTNER WAVES
    // ════════════════════════════════════════════════════════════════════════
    auto getAnimatedHeight = [&](float x, float z) -> Vec3 {
        Vec3 result(0, 0, 0);

        if (noise_type == WaterWaveParams::NoiseType::Gerstner ||
            noise_type == WaterWaveParams::NoiseType::TessendorfSimple) {
            float wx = x * world_scale.x;
            float wz = z * world_scale.z;
            for (const auto& w : anim_waves) {
                float phase = w.k * (wx * w.dx + wz * w.dz) - w.speed * time;
                result.x += w.Q * w.amplitude * w.dx * cosf(phase);
                result.y += w.amplitude * sinf(phase);
                result.z += w.Q * w.amplitude * w.dz * cosf(phase);
            }
        } else {
            // Simple animated noise (FBM, Ridge, etc.)
            float value = 0.0f;
            float amp = 1.0f;
            float freq = 1.0f;
            float maxAmp = 0.0f;
            float wind_dx = cosf(swell_dir);
            float wind_dz = sinf(swell_dir);
            float cross_dx = -wind_dz;
            float cross_dz = wind_dx;
            float drift_speed = 0.25f + alignment * 0.75f;
            float morph = 0.85f + lacunarity * 0.05f;

            for (int i = 0; i < octaves; i++) {
                float nx = (x * world_scale.x) / scale * freq;
                float nz = (z * world_scale.z) / scale * freq;
                float octaveMix = static_cast<float>(i + 1);
                float off_x =
                    wind_dx * time * drift_speed * (0.45f / octaveMix) +
                    cross_dx * sinf(time * 0.17f * octaveMix) * 0.12f +
                    sinf(time * 0.31f * morph + octaveMix) * 0.22f;
                float off_z =
                    wind_dz * time * drift_speed * (0.45f / octaveMix) +
                    cross_dz * cosf(time * 0.13f * octaveMix) * 0.12f +
                    cosf(time * 0.23f * morph + octaveMix * 0.7f) * 0.22f;
                Vec3 p(nx + off_x, time * 0.05f * morph, nz + off_z);
                float n = perlin.noise(p);
                
                if (noise_type == WaterWaveParams::NoiseType::Ridge) {
                    n = surf->params.geo_ridge_offset - fabsf(n);
                    n = n * n;
                }
                
                value += n * amp;
                maxAmp += amp;
                amp *= persistence;
                freq *= lacunarity;
            }
            
            if (maxAmp > 0.001f) value /= maxAmp;
            result.y = value * max_height;
        }
        
        return result;
    };
    
    struct VertexKey {
        int ix, iz;
        bool operator<(const VertexKey& o) const {
            if (ix != o.ix) return ix < o.ix;
            return iz < o.iz;
        }
    };
    auto makeKey = [](const Vec3& p) -> VertexKey {
        return { (int)roundf(p.x * 100.0f), (int)roundf(p.z * 100.0f) };
    };

    std::map<VertexKey, Vec3> vertex_positions;
    std::map<VertexKey, Vec3> vertex_normals;

    size_t idx = 0;
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        for (int v = 0; v < 3; ++v) {
            if (idx >= surf->original_positions.size()) break;
            const Vec3& orig = surf->original_positions[idx++];
            VertexKey key = makeKey(orig);
            if (vertex_positions.find(key) == vertex_positions.end()) {
                Vec3 disp = getAnimatedHeight(orig.x, orig.z);
                vertex_positions[key] = Vec3(orig.x + disp.x * chop, disp.y, orig.z + disp.z * chop);
                vertex_normals[key] = Vec3(0, 0, 0);
            }
        }
    }

    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        Vec3 orig0 = tri->getOriginalVertexPosition(0);
        Vec3 orig1 = tri->getOriginalVertexPosition(1);
        Vec3 orig2 = tri->getOriginalVertexPosition(2);
        Vec3 p0 = vertex_positions[makeKey(orig0)];
        Vec3 p1 = vertex_positions[makeKey(orig1)];
        Vec3 p2 = vertex_positions[makeKey(orig2)];
        Vec3 face_normal = (p1 - p0).cross(p2 - p0);
        float len = face_normal.length();
        if (len > 0.0001f) face_normal = face_normal / len;
        else face_normal = Vec3(0, 1, 0);

        if (surf->params.geo_smooth_normals) {
            vertex_normals[makeKey(orig0)] = vertex_normals[makeKey(orig0)] + face_normal;
            vertex_normals[makeKey(orig1)] = vertex_normals[makeKey(orig1)] + face_normal;
            vertex_normals[makeKey(orig2)] = vertex_normals[makeKey(orig2)] + face_normal;
        } else {
            vertex_normals[makeKey(orig0)] = face_normal;
            vertex_normals[makeKey(orig1)] = face_normal;
            vertex_normals[makeKey(orig2)] = face_normal;
        }
    }

    for (auto& it : vertex_normals) {
        float len = it.second.length();
        it.second = len > 0.0001f ? (it.second / len) : Vec3(0, 1, 0);
    }

    idx = 0;
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        for (int v = 0; v < 3; ++v) {
            if (idx >= surf->original_positions.size()) break;
            const Vec3& orig = surf->original_positions[idx++];
            VertexKey key = makeKey(orig);
            tri->setVertexPosition(v, vertex_positions[key]);
            tri->setVertexNormal(v, vertex_normals[key]);
        }
        tri->markAABBDirty();
    }
    return idx > 0;
}

// ════════════════════════════════════════════════════════════════════════════════
// GPU ANIMATED MESH UPDATE
// ════════════════════════════════════════════════════════════════════════════════
bool WaterManager::updateGPUAnimatedWaterMesh(WaterSurface* surf, float time) {
    if (!surf) return false;
    if (!surf->params.use_geometric_waves) return false;
    if (!g_hasCUDA) {
        // Fallback to CPU
        return updateAnimatedWaterMesh(surf, time);
    }
    
    // Cache original positions if not done
    if (surf->original_positions.empty()) {
        cacheOriginalPositions(surf);
    }
    
    int vertex_count = static_cast<int>(surf->original_positions.size());
    if (vertex_count == 0) return false;
    
    // Initialize GPU state if needed
    GPUGeoWaveState* gpu_state = static_cast<GPUGeoWaveState*>(surf->gpu_geo_state);
    if (!gpu_state || !gpu_state->initialized) {
        gpu_state = new GPUGeoWaveState();
        surf->gpu_geo_state = gpu_state;
        
        // Prepare original positions as float array
        std::vector<float> h_original(vertex_count * 3);
        for (int i = 0; i < vertex_count; ++i) {
            h_original[i * 3 + 0] = surf->original_positions[i].x;
            h_original[i * 3 + 1] = surf->original_positions[i].y;
            h_original[i * 3 + 2] = surf->original_positions[i].z;
        }
        
        if (!initGPUGeometricWaves(gpu_state, h_original.data(), vertex_count)) {
            // Init failed, fallback to CPU
            delete gpu_state;
            surf->gpu_geo_state = nullptr;
            return updateAnimatedWaterMesh(surf, time);
        }
    }
    
    // Prepare params
    GeoWaveParams params;
    params.time = time;
    params.wave_height = surf->params.geo_wave_height;
    params.wave_scale = surf->params.geo_wave_scale;
    params.choppiness = surf->params.geo_wave_choppiness;
    params.octaves = surf->params.geo_octaves;
    params.persistence = surf->params.geo_persistence;
    params.lacunarity = surf->params.geo_lacunarity;
    params.swell_direction = surf->params.geo_swell_direction * 3.14159265f / 180.0f;
    params.swell_amplitude = surf->params.geo_swell_amplitude;
    params.alignment = surf->params.geo_alignment;
    params.damping = surf->params.geo_damping;
    params.ridge_offset = surf->params.geo_ridge_offset;
    params.domain_coord_scale = getLegacyDomainReferenceSize() / fmaxf(resolveWaveDomainSize(surf), 0.001f);
    params.noise_type = static_cast<int>(surf->params.geo_noise_type);
    
    // Allocate output buffer
    std::vector<float> h_output(vertex_count * 3);
    
    // Run GPU kernel
    updateGPUGeometricWaves(gpu_state, &params, h_output.data());
    
    struct VertexKey {
        int ix, iz;
        bool operator<(const VertexKey& o) const {
            if (ix != o.ix) return ix < o.ix;
            return iz < o.iz;
        }
    };
    auto makeKey = [](const Vec3& p) -> VertexKey {
        return { (int)roundf(p.x * 100.0f), (int)roundf(p.z * 100.0f) };
    };

    std::map<VertexKey, Vec3> vertex_positions;
    std::map<VertexKey, Vec3> vertex_normals;
    for (int i = 0; i < vertex_count; ++i) {
        const Vec3& orig = surf->original_positions[i];
        VertexKey key = makeKey(orig);
        vertex_positions[key] = Vec3(h_output[i * 3 + 0], h_output[i * 3 + 1], h_output[i * 3 + 2]);
        vertex_normals[key] = Vec3(0, 0, 0);
    }

    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        Vec3 orig0 = tri->getOriginalVertexPosition(0);
        Vec3 orig1 = tri->getOriginalVertexPosition(1);
        Vec3 orig2 = tri->getOriginalVertexPosition(2);
        Vec3 p0 = vertex_positions[makeKey(orig0)];
        Vec3 p1 = vertex_positions[makeKey(orig1)];
        Vec3 p2 = vertex_positions[makeKey(orig2)];
        Vec3 face_normal = (p1 - p0).cross(p2 - p0);
        float len = face_normal.length();
        if (len > 0.0001f) face_normal = face_normal / len;
        else face_normal = Vec3(0, 1, 0);

        if (surf->params.geo_smooth_normals) {
            vertex_normals[makeKey(orig0)] = vertex_normals[makeKey(orig0)] + face_normal;
            vertex_normals[makeKey(orig1)] = vertex_normals[makeKey(orig1)] + face_normal;
            vertex_normals[makeKey(orig2)] = vertex_normals[makeKey(orig2)] + face_normal;
        } else {
            vertex_normals[makeKey(orig0)] = face_normal;
            vertex_normals[makeKey(orig1)] = face_normal;
            vertex_normals[makeKey(orig2)] = face_normal;
        }
    }

    for (auto& it : vertex_normals) {
        float len = it.second.length();
        it.second = len > 0.0001f ? (it.second / len) : Vec3(0, 1, 0);
    }

    size_t idx = 0;
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        for (int v = 0; v < 3; ++v) {
            if (idx >= (size_t)vertex_count) break;
            const Vec3& orig = surf->original_positions[idx++];
            VertexKey key = makeKey(orig);
            tri->setVertexPosition(v, vertex_positions[key]);
            tri->setVertexNormal(v, vertex_normals[key]);
        }
        tri->markAABBDirty();
    }
    return idx > 0;
}

// ════════════════════════════════════════════════════════════════════════════════
// FFT-DRIVEN MESH DISPLACEMENT (Highest Quality - CPU Hybrid Version)
// ════════════════════════════════════════════════════════════════════════════════
// Downloads FFT data once, then processes all vertices on CPU with bilinear interpolation
// More reliable than GPU kernel approach, compatible with geometric waves pattern
bool WaterManager::updateFFTDrivenMesh(WaterSurface* surf, float time) {
    if (!surf) return false;
    if (!surf->params.use_fft_ocean || !surf->params.use_fft_mesh_displacement) return false;
    if (!surf->fft_state || !g_hasCUDA) return false;
    
    FFTOceanState* fft_state = static_cast<FFTOceanState*>(surf->fft_state);
    if (!fft_state->initialized) return false;
    
    // Cache original positions if needed
    if (surf->original_positions.empty()) {
        cacheOriginalPositions(surf);
    }
    
    int vertex_count = static_cast<int>(surf->original_positions.size());
    if (vertex_count == 0) {
        SCENE_LOG_WARN("[FFT Mesh] No vertices to displace!");
        return false;
    }
    
    // Get FFT parameters
    float ocean_size = resolveWaveDomainSize(surf);
    float height_scale = surf->params.fft_mesh_height_scale;
    float choppiness = surf->params.fft_mesh_choppiness;
    
    // Download FFT data to CPU (single batch copy)
    int N = 0;
    std::vector<float> h_height, h_disp_x, h_disp_z, h_normal_x, h_normal_z;
    
    if (!downloadFFTOceanData(fft_state, nullptr, nullptr, nullptr, nullptr, nullptr, &N) || N == 0) {
        SCENE_LOG_WARN("[FFT Mesh] FFT data not ready (N=0)");
        return false;
    }
    
    int N2 = N * N;
    h_height.resize(N2);
    h_disp_x.resize(N2);
    h_disp_z.resize(N2);
    h_normal_x.resize(N2);
    h_normal_z.resize(N2);
    
    if (!downloadFFTOceanData(fft_state, h_height.data(), h_disp_x.data(), h_disp_z.data(), 
                              h_normal_x.data(), h_normal_z.data(), &N)) {
        SCENE_LOG_WARN("[FFT Mesh] Failed to download FFT data");
        return false;
    }
    
    // DEBUG: Check FFT data range
    static int debug_counter = 0;
    if (debug_counter++ % 100 == 0) {
        float minH = h_height[0], maxH = h_height[0];
        for (int i = 1; i < N2; ++i) {
            minH = fminf(minH, h_height[i]);
            maxH = fmaxf(maxH, h_height[i]);
        }
        SCENE_LOG_INFO("[FFT Mesh] N=%d, Height range: [%.4f, %.4f], scale=%.2f, vertices=%d", 
                       N, minH, maxH, height_scale, vertex_count);
    }
    
    // Bilinear sampling lambda
    auto sampleBilinear = [&](const std::vector<float>& data, float u, float v) -> float {
        u = u - floorf(u);
        v = v - floorf(v);
        
        float fx = u * (float)N;
        float fz = v * (float)N;
        
        int ix0 = (int)floorf(fx) % N;
        int iz0 = (int)floorf(fz) % N;
        int ix1 = (ix0 + 1) % N;
        int iz1 = (iz0 + 1) % N;
        
        // Ensure non-negative
        if (ix0 < 0) ix0 += N;
        if (iz0 < 0) iz0 += N;
        
        float tx = fx - floorf(fx);
        float tz = fz - floorf(fz);
        
        float v00 = data[iz0 * N + ix0];
        float v10 = data[iz0 * N + ix1];
        float v01 = data[iz1 * N + ix0];
        float v11 = data[iz1 * N + ix1];
        
        float v0 = v00 * (1.0f - tx) + v10 * tx;
        float v1 = v01 * (1.0f - tx) + v11 * tx;
        return v0 * (1.0f - tz) + v1 * tz;
    };
    
    // Apply displacement to all vertices
    size_t idx = 0;
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        
        for (int v = 0; v < 3; ++v) {
            if (idx >= (size_t)vertex_count) break;
            
            const Vec3& orig = surf->original_positions[idx];
            
            // World position to UV
            float u = orig.x / ocean_size;
            float vv = orig.z / ocean_size;
            
            // Sample FFT data
            float height = sampleBilinear(h_height, u, vv);
            float disp_x = sampleBilinear(h_disp_x, u, vv);
            float disp_z = sampleBilinear(h_disp_z, u, vv);
            float nx = sampleBilinear(h_normal_x, u, vv);
            float nz = sampleBilinear(h_normal_z, u, vv);
            
            // Apply displacement with scaling
            Vec3 newPos(
                orig.x + disp_x * choppiness,
                orig.y + height * height_scale,
                orig.z + disp_z * choppiness
            );
            
            // Calculate normal
            float ny = sqrtf(fmaxf(0.001f, 1.0f - nx * nx - nz * nz));
            Vec3 newNormal = Vec3(nx, ny, nz).normalize();
            
            tri->setVertexPosition(v, newPos);
            tri->setVertexNormal(v, newNormal);
            idx++;
        }
        tri->markAABBDirty();
    }
    return idx > 0;
}

// ============================================================================
// APPLY KEYFRAME (for timeline animation)
// ============================================================================
void WaterManager::applyKeyframe(WaterSurface* surf, const WaterKeyframe& kf) {
    if (!surf) return;
    
    bool geo_changed = false;
    bool fft_changed = false;
    
    // ═══════════════════════════════════════════════════════════════════════
    // GEOMETRIC WAVES - Apply only keyed properties
    // ═══════════════════════════════════════════════════════════════════════
    if (kf.has_wave_height) {
        surf->params.geo_wave_height = kf.wave_height;
        surf->animate_mesh = true;
        geo_changed = true;
    }
    if (kf.has_wave_scale) {
        surf->params.geo_wave_scale = kf.wave_scale;
        geo_changed = true;
    }
    if (kf.has_wind_direction) {
        surf->params.geo_swell_direction = kf.wind_direction;
        geo_changed = true;
    }
    if (kf.has_choppiness) {
        surf->params.geo_wave_choppiness = kf.choppiness;
        geo_changed = true;
    }
    float resolved_animation_speed = surf->params.fft_time_scale;
    bool animation_speed_changed = false;
    if (kf.has_fft_time_scale) {
        resolved_animation_speed = kf.fft_time_scale;
        animation_speed_changed = true;
    } else if (kf.has_geo_speed) {
        resolved_animation_speed = kf.geo_speed;
        animation_speed_changed = true;
    }
    if (animation_speed_changed) {
        surf->params.fft_time_scale = resolved_animation_speed;
        surf->params.geo_wave_speed = resolved_animation_speed;
        surf->animate_mesh = true;
        fft_changed = true;
        geo_changed = true;
    }
    if (kf.has_alignment) {
        surf->params.geo_alignment = kf.alignment;
        geo_changed = true;
    }
    if (kf.has_damping) {
        surf->params.geo_damping = kf.damping;
        geo_changed = true;
    }
    if (kf.has_swell_amplitude) {
        surf->params.geo_swell_amplitude = kf.swell_amplitude;
        geo_changed = true;
    }
    if (kf.has_sharpening) {
        surf->params.geo_sharpening = kf.sharpening;
        geo_changed = true;
    }
    if (kf.has_detail_strength) {
        surf->params.geo_detail_strength = kf.detail_strength;
        geo_changed = true;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // FFT OCEAN - Apply keyed properties (real-time parameter animation!)
    // ═══════════════════════════════════════════════════════════════════════
    if (kf.has_fft_wind_speed) {
        surf->params.fft_wind_speed = kf.fft_wind_speed;
        fft_changed = true;
    }
    if (kf.has_fft_wind_direction) {
        // Convert degrees to radians for FFT system
        surf->params.fft_wind_direction = kf.fft_wind_direction * 3.14159265f / 180.0f;
        fft_changed = true;
    }
    if (kf.has_fft_amplitude) {
        surf->params.fft_amplitude = kf.fft_amplitude;
        fft_changed = true;
    }
    if (kf.has_fft_choppiness) {
        surf->params.fft_choppiness = kf.fft_choppiness;
        fft_changed = true;
    }
    if (kf.has_fft_time_scale) {
        // handled above as the shared water animation speed
    }
    if (kf.has_fft_ocean_size) {
        surf->params.fft_ocean_size = kf.fft_ocean_size;
        fft_changed = true;
    }
    
    // Rebuild mesh if geometric parameters changed
    if (geo_changed && surf->params.use_geometric_waves) {
        updateWaterMesh(surf);
        cacheOriginalPositions(surf);
        
        // CPU BVH needs refresh for picking / CPU rendering, but GPU backends should
        // consume water deformation through their normal per-frame update paths.
        // For timeline playback, forcing full OptiX/Vulkan rebuilds here causes the
        // water shading/state to get stomped every frame.
        extern bool g_bvh_rebuild_pending;
        g_bvh_rebuild_pending = true;
    }
    
    // FFT changes are picked up automatically by update() on next frame
    // No need for explicit rebuild - the FFT system reads params each frame
    if (geo_changed || fft_changed) {
        syncSurfaceMaterial(surf);
    }
}

void WaterManager::invalidateGeometricAnimationState(WaterSurface* surf) {
    if (!surf) return;

    surf->animation_time = 0.0f;
    surf->original_positions.clear();

    if (surf->gpu_geo_state) {
        GPUGeoWaveState* gpu_state = static_cast<GPUGeoWaveState*>(surf->gpu_geo_state);
        if (g_hasCUDA && gpu_state) {
            cleanupGPUGeometricWaves(gpu_state);
        }
        delete gpu_state;
        surf->gpu_geo_state = nullptr;
    }
}

// ============================================================================
// CAPTURE KEYFRAME TO TRACK (for timeline recording)
// ============================================================================
void WaterManager::captureKeyframeToTrack(WaterSurface* surf, ObjectAnimationTrack& track, int frame) {
    if (!surf) return;
    
    Keyframe kf(frame);
    
    // Capture current water parameters
    kf.water.water_surface_id = surf->id;
    
    kf.water.wave_height = surf->params.geo_wave_height;
    kf.water.has_wave_height = true;
    
    kf.water.wave_scale = surf->params.geo_wave_scale;
    kf.water.has_wave_scale = true;
    
    kf.water.wind_direction = surf->params.geo_swell_direction;
    kf.water.has_wind_direction = true;
    
    kf.water.choppiness = surf->params.geo_wave_choppiness;
    kf.water.has_choppiness = true;
    
    kf.water.geo_speed = surf->params.geo_wave_speed;
    kf.water.has_geo_speed = true;
    
    kf.water.alignment = surf->params.geo_alignment;
    kf.water.has_alignment = true;
    
    kf.water.damping = surf->params.geo_damping;
    kf.water.has_damping = true;
    
    kf.water.swell_amplitude = surf->params.geo_swell_amplitude;
    kf.water.has_swell_amplitude = true;
    
    kf.water.sharpening = surf->params.geo_sharpening;
    kf.water.has_sharpening = true;
    
    kf.water.detail_strength = surf->params.geo_detail_strength;
    kf.water.has_detail_strength = true;
    
    // ═══════════════════════════════════════════════════════════════════════
    // FFT OCEAN - Capture if FFT is enabled
    // ═══════════════════════════════════════════════════════════════════════
    if (surf->params.use_fft_ocean) {
        kf.water.fft_wind_speed = surf->params.fft_wind_speed;
        kf.water.has_fft_wind_speed = true;
        
        // Convert radians to degrees for keyframe storage
        kf.water.fft_wind_direction = surf->params.fft_wind_direction * 180.0f / 3.14159265f;
        kf.water.has_fft_wind_direction = true;
        
        kf.water.fft_amplitude = surf->params.fft_amplitude;
        kf.water.has_fft_amplitude = true;
        
        kf.water.fft_choppiness = surf->params.fft_choppiness;
        kf.water.has_fft_choppiness = true;
        
        kf.water.fft_time_scale = surf->params.fft_time_scale;
        kf.water.has_fft_time_scale = true;
        
        kf.water.fft_ocean_size = surf->params.fft_ocean_size;
        kf.water.has_fft_ocean_size = true;
    }
    
    kf.has_water = true;
    
    track.addKeyframe(kf);
}

// ============================================================================
// UPDATE FROM ANIMATION TRACK (called each frame during playback)
// ============================================================================
void WaterManager::updateFromTrack(WaterSurface* surf, const ObjectAnimationTrack& track, int currentFrame) {
    if (!surf) return;
    
    // Evaluate track at current frame (does interpolation)
    Keyframe kf = track.evaluate(currentFrame);
    
    // Apply if has water keyframe data
    // Note: Track is already scoped to this water surface via track name "Water_X"
    if (kf.has_water) {
        applyKeyframe(surf, kf.water);
    }
}

// ============================================================================
// SERIALIZATION
// ============================================================================

nlohmann::json WaterManager::serialize() const {
    nlohmann::json arr = nlohmann::json::array();
    
    for (const auto& surf : water_surfaces) {
        nlohmann::json ws;
        ws["id"] = surf.id;
        ws["name"] = surf.name;
        ws["material_id"] = surf.material_id;
        
        // Wave params
        ws["wave_speed"] = surf.params.wave_speed;
        ws["wave_strength"] = surf.params.wave_strength;
        ws["wave_frequency"] = surf.params.wave_frequency;
        
        // Colors
        ws["deep_color"] = {surf.params.deep_color.x, surf.params.deep_color.y, surf.params.deep_color.z};
        ws["shallow_color"] = {surf.params.shallow_color.x, surf.params.shallow_color.y, surf.params.shallow_color.z};
        
        // Physics
        ws["clarity"] = surf.params.clarity;
        ws["foam_level"] = surf.params.foam_level;
        ws["ior"] = surf.params.ior;
        ws["roughness"] = surf.params.roughness;
        
        // Advanced: Depth & Absorption
        ws["depth_max"] = surf.params.depth_max;
        ws["absorption_color"] = {surf.params.absorption_color.x, surf.params.absorption_color.y, surf.params.absorption_color.z};
        ws["absorption_density"] = surf.params.absorption_density;
        
        // Advanced: Shore Foam
        ws["shore_foam_distance"] = surf.params.shore_foam_distance;
        ws["shore_foam_intensity"] = surf.params.shore_foam_intensity;
        
        // Advanced: Caustics
        ws["caustic_intensity"] = surf.params.caustic_intensity;
        ws["caustic_scale"] = surf.params.caustic_scale;
        ws["caustic_speed"] = surf.params.caustic_speed;
        
        // Advanced: SSS
        ws["sss_intensity"] = surf.params.sss_intensity;
        ws["sss_color"] = {surf.params.sss_color.x, surf.params.sss_color.y, surf.params.sss_color.z};
        
        // Advanced: FFT Ocean
        ws["use_fft_ocean"] = surf.params.use_fft_ocean;
        ws["fft_resolution"] = surf.params.fft_resolution;
        ws["fft_ocean_size"] = surf.params.fft_ocean_size;
        ws["auto_domain_from_mesh"] = surf.params.auto_domain_from_mesh;
        ws["domain_size_multiplier"] = surf.params.domain_size_multiplier;
        ws["fft_wind_speed"] = surf.params.fft_wind_speed;
        ws["fft_wind_direction"] = surf.params.fft_wind_direction;
        ws["fft_choppiness"] = surf.params.fft_choppiness;
        ws["fft_amplitude"] = surf.params.fft_amplitude;
        ws["fft_time_scale"] = surf.params.fft_time_scale;
        
        // Advanced: Water Details
        ws["micro_detail_strength"] = surf.params.micro_detail_strength;
        ws["micro_detail_scale"] = surf.params.micro_detail_scale;
        ws["micro_anim_speed"] = surf.params.micro_anim_speed;
        ws["micro_morph_speed"] = surf.params.micro_morph_speed;
        ws["foam_noise_scale"] = surf.params.foam_noise_scale;
        ws["foam_threshold"] = surf.params.foam_threshold;
        
        // Geometric Displacement
        ws["use_geometric_waves"] = surf.params.use_geometric_waves;
        ws["geo_wave_height"] = surf.params.geo_wave_height;
        ws["geo_wave_scale"] = surf.params.geo_wave_scale;
        ws["geo_wave_choppiness"] = surf.params.geo_wave_choppiness;
        ws["geo_wave_speed"] = surf.params.geo_wave_speed;
        
        // Detailed Noise Params
        ws["geo_noise_type"] = (int)surf.params.geo_noise_type;
        ws["geo_octaves"] = surf.params.geo_octaves;
        ws["geo_persistence"] = surf.params.geo_persistence;
        ws["geo_lacunarity"] = surf.params.geo_lacunarity;
        ws["geo_ridge_offset"] = surf.params.geo_ridge_offset;
        
        // Blender-style Ocean Params
        ws["geo_damping"] = surf.params.geo_damping;
        ws["geo_alignment"] = surf.params.geo_alignment;
        ws["geo_depth"] = surf.params.geo_depth;
        ws["geo_swell_direction"] = surf.params.geo_swell_direction;
        ws["geo_swell_amplitude"] = surf.params.geo_swell_amplitude;
        ws["geo_sharpening"] = surf.params.geo_sharpening;
        ws["geo_detail_scale"] = surf.params.geo_detail_scale;
        ws["geo_detail_strength"] = surf.params.geo_detail_strength;
        ws["geo_smooth_normals"] = surf.params.geo_smooth_normals;
        
        // FFT-Driven Mesh Displacement
        ws["use_fft_mesh_displacement"] = surf.params.use_fft_mesh_displacement;
        ws["fft_mesh_height_scale"] = surf.params.fft_mesh_height_scale;
        ws["fft_mesh_choppiness"] = surf.params.fft_mesh_choppiness;
        
        // Water Preset
        ws["current_preset"] = static_cast<int>(surf.params.current_preset);
        
        // Animation state
        ws["animate_mesh"] = surf.animate_mesh;
        ws["use_gpu_animation"] = surf.use_gpu_animation;
        
        ws["type"] = (int)surf.type;

        // Position (from reference triangle or first triangle)
        if (surf.reference_triangle) {
            Vec3 v0 = surf.reference_triangle->getOriginalVertexPosition(0);
            ws["position"] = {v0.x, v0.y, v0.z};
        }

        
        // Grid info (calculate from triangles count)
        // triangles = segments * segments * 2
        size_t tri_count = surf.mesh_triangles.size();
        int segments = static_cast<int>(sqrt(tri_count / 2));
        ws["segments"] = segments;
        
        // Calculate size from triangle vertices
        if (surf.mesh_triangles.size() >= 2) {
            // First vertex of first triangle and last vertex of last triangle
            Vec3 v_first = surf.mesh_triangles[0]->getOriginalVertexPosition(0);
            Vec3 v_last = surf.mesh_triangles.back()->getOriginalVertexPosition(2);
            float size_x = std::abs(v_last.x - v_first.x);
            float size_z = std::abs(v_last.z - v_first.z);
            ws["size"] = std::max(size_x, size_z);
        }
        
        arr.push_back(ws);
    }
    
    nlohmann::json result;
    result["water_surfaces"] = arr;
    result["next_id"] = next_id;
    result["preview_time_mode"] = static_cast<int>(preview_time_mode);
    
    return result;
}

void WaterManager::deserialize(const nlohmann::json& j, SceneData& scene) {
    // Clear existing water surface metadata (but don't remove triangles - they're loaded from scene geometry)
    water_surfaces.clear();
    
    if (!j.contains("water_surfaces")) return;
    
    next_id = j.value("next_id", 1);
    preview_time_mode = static_cast<WaterPreviewTimeMode>(j.value("preview_time_mode", static_cast<int>(WaterPreviewTimeMode::Static)));
    last_resolved_preview_time = 0.0f;
    static_preview_time = 0.0f;
    last_simulation_time = 0.0f;
    has_last_resolved_preview_time = false;
    has_last_simulation_time = false;
    
    for (const auto& ws : j["water_surfaces"]) {
        WaterSurface surf;
        surf.id = ws.value("id", next_id++);
        surf.name = ws.value("name", "Water_Plane_" + std::to_string(surf.id));
        surf.material_id = ws.value("material_id", 0);
        
        // Restore wave params
        surf.params.wave_speed = ws.value("wave_speed", 1.0f);
        surf.params.wave_strength = ws.value("wave_strength", 0.5f);
        surf.params.wave_frequency = ws.value("wave_frequency", 1.0f);
        surf.params.clarity = ws.value("clarity", 0.8f);
        surf.params.foam_level = ws.value("foam_level", 0.2f);
        surf.params.ior = ws.value("ior", 1.333f);
        surf.params.roughness = ws.value("roughness", 0.02f);
        
        // Colors
        if (ws.contains("deep_color")) {
            surf.params.deep_color = Vec3(ws["deep_color"][0], ws["deep_color"][1], ws["deep_color"][2]);
        }
        if (ws.contains("shallow_color")) {
            surf.params.shallow_color = Vec3(ws["shallow_color"][0], ws["shallow_color"][1], ws["shallow_color"][2]);
        }
        
        // Advanced: Depth & Absorption
        surf.params.depth_max = ws.value("depth_max", 15.0f);
        surf.params.absorption_density = ws.value("absorption_density", 0.5f);
        if (ws.contains("absorption_color")) {
            surf.params.absorption_color = Vec3(ws["absorption_color"][0], ws["absorption_color"][1], ws["absorption_color"][2]);
        }
        
        // Advanced: Shore Foam
        surf.params.shore_foam_distance = ws.value("shore_foam_distance", 1.5f);
        surf.params.shore_foam_intensity = ws.value("shore_foam_intensity", 0.6f);
        
        // Advanced: Caustics
        surf.params.caustic_intensity = ws.value("caustic_intensity", 0.4f);
        surf.params.caustic_scale = ws.value("caustic_scale", 2.0f);
        surf.params.caustic_speed = ws.value("caustic_speed", 1.0f);
        
        // Advanced: SSS
        surf.params.sss_intensity = ws.value("sss_intensity", 0.15f);
        if (ws.contains("sss_color")) {
            surf.params.sss_color = Vec3(ws["sss_color"][0], ws["sss_color"][1], ws["sss_color"][2]);
        }
        
        // Advanced: FFT Ocean
        surf.params.use_fft_ocean = ws.value("use_fft_ocean", false);
        surf.params.fft_resolution = ws.value("fft_resolution", 256);
        surf.params.fft_ocean_size = ws.value("fft_ocean_size", 100.0f);
        surf.params.auto_domain_from_mesh = false;
        surf.params.domain_size_multiplier = 1.0f;
        surf.params.fft_wind_speed = ws.value("fft_wind_speed", 10.0f);
        surf.params.fft_wind_direction = ws.value("fft_wind_direction", 0.0f);
        surf.params.fft_choppiness = ws.value("fft_choppiness", 1.0f);
        surf.params.fft_amplitude = ws.value("fft_amplitude", 0.0002f);
        surf.params.fft_time_scale = ws.value("fft_time_scale", 1.0f);
        
        // Advanced: Water Details
        surf.params.micro_detail_strength = ws.value("micro_detail_strength", 0.05f);
        surf.params.micro_detail_scale = ws.value("micro_detail_scale", 20.0f);
        surf.params.micro_anim_speed = ws.value("micro_anim_speed", 0.1f);
        surf.params.micro_morph_speed = ws.value("micro_morph_speed", 1.0f);
        surf.params.foam_noise_scale = ws.value("foam_noise_scale", 4.0f);
        surf.params.foam_threshold = ws.value("foam_threshold", 0.4f);
        
        // Geometric Displacement
        if (ws.contains("type")) {
            surf.type = (WaterSurface::Type)ws.value("type", (int)WaterSurface::Type::Plane);
        } else {
            surf.type = WaterSurface::Type::Plane;
        }
        
        surf.params.use_geometric_waves = ws.value("use_geometric_waves", false);
        surf.params.geo_wave_height = ws.value("geo_wave_height", 2.0f);
        surf.params.geo_wave_scale = ws.value("geo_wave_scale", 50.0f);
        surf.params.geo_wave_choppiness = ws.value("geo_wave_choppiness", 1.0f);
        surf.params.geo_wave_speed = ws.value("geo_wave_speed", 0.5f);
        const float resolved_speed = resolveSharedAnimationSpeed(&surf);
        surf.params.fft_time_scale = resolved_speed;
        surf.params.geo_wave_speed = resolved_speed;
        
        // Detailed Noise Params
        surf.params.geo_noise_type = (WaterWaveParams::NoiseType)ws.value("geo_noise_type", (int)WaterWaveParams::NoiseType::Ridge);
        surf.params.geo_octaves = ws.value("geo_octaves", 4);
        surf.params.geo_persistence = ws.value("geo_persistence", 0.5f);
        surf.params.geo_lacunarity = ws.value("geo_lacunarity", 2.0f);
        surf.params.geo_ridge_offset = ws.value("geo_ridge_offset", 1.0f);
        
        // Geometric Ocean Params
        surf.params.geo_damping = ws.value("geo_damping", 0.0f);
        surf.params.geo_alignment = ws.value("geo_alignment", 0.5f);
        surf.params.geo_depth = ws.value("geo_depth", 200.0f);
        surf.params.geo_swell_direction = ws.value("geo_swell_direction", 0.0f);
        surf.params.geo_swell_amplitude = ws.value("geo_swell_amplitude", 0.2f);
        surf.params.geo_sharpening = ws.value("geo_sharpening", 0.0f);
        surf.params.geo_detail_scale = ws.value("geo_detail_scale", 3.0f);
        surf.params.geo_detail_strength = ws.value("geo_detail_strength", 0.15f);
        surf.params.geo_smooth_normals = ws.value("geo_smooth_normals", true);
        
        // FFT-Driven Mesh Displacement
        surf.params.use_fft_mesh_displacement = ws.value("use_fft_mesh_displacement", false);
        surf.params.fft_mesh_height_scale = ws.value("fft_mesh_height_scale", 1.0f);
        surf.params.fft_mesh_choppiness = ws.value("fft_mesh_choppiness", 1.0f);
        
        // Water Preset (load as int, cast to enum)
        int preset_val = ws.value("current_preset", 0);
        surf.params.current_preset = static_cast<WaterWaveParams::WaterPreset>(preset_val);
        
        // Animation state
        surf.animate_mesh = ws.value("animate_mesh", false);
        surf.use_gpu_animation = ws.value("use_gpu_animation", true);
        if (surf.params.use_geometric_waves || (surf.params.use_fft_ocean && surf.params.use_fft_mesh_displacement)) {
            surf.animate_mesh = true;
        }

        // Find existing triangles in scene by nodeName (don't create new ones!)
        for (auto& obj : scene.world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->nodeName == surf.name) {
                surf.mesh_triangles.push_back(tri);
                if (!surf.reference_triangle) {
                    surf.reference_triangle = tri;
                }
            }
        }
        
        // Update material with restored params (material should already exist from scene load)
        if (surf.material_id > 0) {
            syncSurfaceMaterial(&surf);
        }
        
        water_surfaces.push_back(std::move(surf));
    }
    
    SCENE_LOG_INFO("[WaterManager] Loaded " + std::to_string(water_surfaces.size()) + " water surfaces (using existing geometry).");
}
