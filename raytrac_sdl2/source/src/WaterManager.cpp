#include "WaterSystem.h"
#include "scene_data.h"
#include "Renderer.h"
#include "OptixWrapper.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "PrincipledBSDF.h"
#include "globals.h"
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

void WaterManager::update(float dt) {
    // Will be used for CPU animation later if needed
}

WaterSurface* WaterManager::createWaterPlane(SceneData& scene, const Vec3& pos, float size, float density) {
    WaterSurface surf;
    surf.id = next_id++;
    surf.name = "Water_Plane_" + std::to_string(surf.id);
    
    // 1. Create unique Water Material
    auto water_mat = std::make_shared<PrincipledBSDF>();
    auto gpu = std::make_shared<GpuMaterial>();
    
    // Approved Water Params
    gpu->albedo = make_float3(1.0f, 1.0f, 1.0f); // White for clear transmission (Absorption = 1 - Albedo)
    gpu->transmission = 1.0f;
    gpu->opacity = 1.0f;
    gpu->roughness = surf.params.roughness;
    gpu->ior = surf.params.ior;
    
    // PACKING WAVE PARAMS into unused PBR fields
    // anisotropic -> Wave Speed
    // sheen -> Wave Strength (Serves as IS_WATER flag if > 0)
    // sheen_tint -> Wave Frequency
    gpu->anisotropic = surf.params.wave_speed;
    gpu->sheen = fmaxf(0.001f, surf.params.wave_strength); // Ensure > 0 to act as flag
    gpu->sheen_tint = surf.params.wave_frequency;

    gpu->metallic = 0.0f;
    // gpu->specular = 1.0f; // REMOVED: Not in GpuMaterial
    // gpu->specTrans = 1.0f; // REMOVED: Not in GpuMaterial
    // gpu->scatter_distance = ... // REMOVED
    
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
    float start_x = pos.x - size * 0.5f;
    float start_z = pos.z - size * 0.5f;
    
    std::shared_ptr<Transform> shared_transform = std::make_shared<Transform>();
    shared_transform->setBase(Matrix4x4::identity()); // World space vertices
    
    for (int z = 0; z < segments; z++) {
        for (int x = 0; x < segments; x++) {
            float x0 = start_x + (x * step);
            float z0 = start_z + (z * step);
            float x1 = x0 + step;
            float z1 = z0 + step;
            
            // Grid cell vertices (y is flat initially)
            Vec3 v0(x0, pos.y, z0);
            Vec3 v1(x1, pos.y, z0);
            Vec3 v2(x1, pos.y, z1);
            Vec3 v3(x0, pos.y, z1);
            
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
        ws["deep_color"] = {surf.params.deep_color.x, surf.params.deep_color.y, surf.params.deep_color.z};
        ws["shallow_color"] = {surf.params.shallow_color.x, surf.params.shallow_color.y, surf.params.shallow_color.z};
        ws["clarity"] = surf.params.clarity;
        ws["foam_level"] = surf.params.foam_level;
        ws["ior"] = surf.params.ior;
        ws["roughness"] = surf.params.roughness;
        
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
        surf.params.foam_level = ws.value("foam_level", 0.1f);
        surf.params.ior = ws.value("ior", 1.333f);
        surf.params.roughness = ws.value("roughness", 0.05f);
        
        if (ws.contains("deep_color")) {
            surf.params.deep_color = Vec3(ws["deep_color"][0], ws["deep_color"][1], ws["deep_color"][2]);
        }
        if (ws.contains("shallow_color")) {
            surf.params.shallow_color = Vec3(ws["shallow_color"][0], ws["shallow_color"][1], ws["shallow_color"][2]);
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
            auto mat = MaterialManager::getInstance().getMaterial(surf.material_id);
            if (mat && mat->gpuMaterial) {
                // Restore ESSENTIAL water material properties
                mat->gpuMaterial->albedo = make_float3(1.0f, 1.0f, 1.0f); // White for clear transmission
                mat->gpuMaterial->transmission = 1.0f;
                mat->gpuMaterial->opacity = 1.0f;
                mat->gpuMaterial->metallic = 0.0f;
                
                // Wave params packed into PBR fields
                mat->gpuMaterial->anisotropic = surf.params.wave_speed;
                mat->gpuMaterial->sheen = std::fmax(0.001f, surf.params.wave_strength); // IS_WATER flag
                mat->gpuMaterial->sheen_tint = surf.params.wave_frequency;
                mat->gpuMaterial->roughness = surf.params.roughness;
                mat->gpuMaterial->ior = surf.params.ior;
            }
        }
        
        water_surfaces.push_back(std::move(surf));
    }
    
    SCENE_LOG_INFO("[WaterManager] Loaded " + std::to_string(water_surfaces.size()) + " water surfaces (using existing geometry).");
}
