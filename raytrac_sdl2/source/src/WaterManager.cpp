#include "WaterSystem.h"
#include "scene_data.h"
#include "Renderer.h"
#include "OptixWrapper.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "PrincipledBSDF.h"
#include "globals.h"
#include "fft_ocean.cuh"

// CUDA Library Linking
#pragma comment(lib, "cufft.lib")
#pragma comment(lib, "cudart.lib")

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
    for (auto& surf : water_surfaces) {
         if (surf.fft_state) {
             FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
             cleanupFFTOcean(state);
             delete state;
             surf.fft_state = nullptr;
         }
    }
    water_surfaces.clear();
    next_id = 1;
}

bool WaterManager::update(float dt) {
    static float global_time = 0.0f;
    global_time += dt;
    bool needs_gpu_sync = false;

    for (auto& surf : water_surfaces) {
        if (surf.params.use_fft_ocean) {
            // Manage FFT State
            if (!surf.fft_state) {
                 FFTOceanState* state = new FFTOceanState();
                 surf.fft_state = (void*)state;
            }
            
            FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
            
            // Map parameters
            FFTOceanParams fft_params;
            fft_params.resolution = surf.params.fft_resolution;
            fft_params.ocean_size = surf.params.fft_ocean_size;
            fft_params.wind_speed = surf.params.fft_wind_speed;
            fft_params.wind_direction = surf.params.fft_wind_direction;
            fft_params.choppiness = surf.params.fft_choppiness;
            fft_params.amplitude = surf.params.fft_amplitude;
            fft_params.time_scale = surf.params.fft_time_scale;
            
            // Check initialization
            if (!state->initialized || state->current_resolution != fft_params.resolution) {
                if (initFFTOcean(state, &fft_params)) {
                    needs_gpu_sync = true; // Texture handles might have changed (recreated)
                }
            }

            // Run simulation
            updateFFTOcean(state, &fft_params, global_time);
            
            // Connect to Material
            if (surf.material_id > 0) {
                auto mat = MaterialManager::getInstance().getMaterial(surf.material_id);
                if (mat && mat->gpuMaterial) {
                    // Update texture handles in material
                    // Note: If handles haven't changed, this is lightweight. 
                    // The actual texture data is updated on GPU by updateFFTOcean.
                    if (mat->gpuMaterial->fft_height_tex != state->tex_height ||
                        mat->gpuMaterial->fft_normal_tex != state->tex_normal) {
                        
                        mat->gpuMaterial->fft_height_tex = state->tex_height;
                        mat->gpuMaterial->fft_normal_tex = state->tex_normal;
                        needs_gpu_sync = true;
                    }
                }
            }
        } else {
             // Cleanup if disabled but state exists
             if (surf.fft_state) {
                 FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
                 cleanupFFTOcean(state);
                 delete state;
                 surf.fft_state = nullptr;
                 needs_gpu_sync = true; // Texture handles removed
                 
                 // Also reset material handles
                 if (surf.material_id > 0) {
                     auto mat = MaterialManager::getInstance().getMaterial(surf.material_id);
                     if (mat && mat->gpuMaterial) {
                         mat->gpuMaterial->fft_height_tex = 0;
                         mat->gpuMaterial->fft_normal_tex = 0;
                     }
                 }
             }
        }
    }
    return needs_gpu_sync;
}

cudaTextureObject_t WaterManager::getFirstFFTHeightMap() {
    for (const auto& surf : water_surfaces) {
        if (surf.params.use_fft_ocean && surf.fft_state) {
            FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
            return state->tex_height;
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
    
    // === WAVE PARAMS (original packing) ===
    // anisotropic -> Wave Speed
    // sheen -> Wave Strength (Serves as IS_WATER flag if > 0)
    // sheen_tint -> Wave Frequency
    gpu->anisotropic = surf.params.wave_speed;
    gpu->sheen = fmaxf(0.001f, surf.params.wave_strength);  // >0 = IS_WATER flag
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
    gpu->foam_noise_scale = surf.params.foam_noise_scale;
    gpu->foam_threshold = surf.params.foam_threshold;
    
    // FFT
    gpu->fft_ocean_size = surf.params.fft_ocean_size;
    gpu->fft_choppiness = surf.params.fft_choppiness;
    
    water_mat->gpuMaterial = gpu;
    
    // Register material
    std::string mat_name = "Water_Mat_" + std::to_string(surf.id);
    surf.material_id = MaterialManager::getInstance().getOrCreateMaterialID(mat_name, water_mat);
    
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
        ws["fft_wind_speed"] = surf.params.fft_wind_speed;
        ws["fft_wind_direction"] = surf.params.fft_wind_direction;
        ws["fft_choppiness"] = surf.params.fft_choppiness;
        ws["fft_amplitude"] = surf.params.fft_amplitude;
        ws["fft_time_scale"] = surf.params.fft_time_scale;
        
        // Advanced: Water Details
        ws["micro_detail_strength"] = surf.params.micro_detail_strength;
        ws["micro_detail_scale"] = surf.params.micro_detail_scale;
        ws["foam_noise_scale"] = surf.params.foam_noise_scale;
        ws["foam_threshold"] = surf.params.foam_threshold;
        
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
    
    return result;
}

void WaterManager::deserialize(const nlohmann::json& j, SceneData& scene) {
    // Clear existing water surface metadata (but don't remove triangles - they're loaded from scene geometry)
    water_surfaces.clear();
    
    if (!j.contains("water_surfaces")) return;
    
    next_id = j.value("next_id", 1);
    
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
        surf.params.fft_wind_speed = ws.value("fft_wind_speed", 10.0f);
        surf.params.fft_wind_direction = ws.value("fft_wind_direction", 0.0f);
        surf.params.fft_choppiness = ws.value("fft_choppiness", 1.0f);
        surf.params.fft_amplitude = ws.value("fft_amplitude", 0.0002f);
        surf.params.fft_time_scale = ws.value("fft_time_scale", 1.0f);
        
        // Advanced: Water Details
        surf.params.micro_detail_strength = ws.value("micro_detail_strength", 0.05f);
        surf.params.micro_detail_scale = ws.value("micro_detail_scale", 20.0f);
        surf.params.foam_noise_scale = ws.value("foam_noise_scale", 4.0f);
        surf.params.foam_threshold = ws.value("foam_threshold", 0.4f);

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
            auto mat = MaterialManager::getInstance().getMaterial(surf.material_id);
            if (mat && mat->gpuMaterial) {
                auto& gpu = mat->gpuMaterial;
                
                // Base material properties
                gpu->albedo = make_float3(surf.params.deep_color.x, surf.params.deep_color.y, surf.params.deep_color.z);
                gpu->transmission = 1.0f;
                gpu->opacity = 1.0f;
                gpu->metallic = 0.0f;
                gpu->roughness = surf.params.roughness;
                gpu->ior = surf.params.ior;
                
                // Wave params
                gpu->anisotropic = surf.params.wave_speed;
                gpu->sheen = std::fmax(0.001f, surf.params.wave_strength);
                gpu->sheen_tint = surf.params.wave_frequency;
                
                // Advanced params
                gpu->clearcoat = surf.params.shore_foam_intensity;
                gpu->clearcoat_roughness = surf.params.caustic_intensity;
                gpu->subsurface = surf.params.depth_max / 100.0f;
                gpu->subsurface_scale = surf.params.absorption_density;
                gpu->subsurface_color = make_float3(surf.params.absorption_color.x, surf.params.absorption_color.y, surf.params.absorption_color.z);
                gpu->subsurface_radius = make_float3(surf.params.shore_foam_distance, surf.params.caustic_scale, surf.params.sss_intensity);
                gpu->emission = make_float3(surf.params.shallow_color.x, surf.params.shallow_color.y, surf.params.shallow_color.z);
                gpu->translucent = surf.params.foam_level;
                gpu->subsurface_anisotropy = surf.params.caustic_speed;
                
                // Water Details (New)
                gpu->micro_detail_strength = surf.params.micro_detail_strength;
                gpu->micro_detail_scale = surf.params.micro_detail_scale;
                gpu->foam_noise_scale = surf.params.foam_noise_scale;
                gpu->foam_threshold = surf.params.foam_threshold;
                
                // FFT
                gpu->fft_ocean_size = surf.params.fft_ocean_size;
                gpu->fft_choppiness = surf.params.fft_choppiness;
            }
        }
        
        water_surfaces.push_back(std::move(surf));
    }
    
    SCENE_LOG_INFO("[WaterManager] Loaded " + std::to_string(water_surfaces.size()) + " water surfaces (using existing geometry).");
}
