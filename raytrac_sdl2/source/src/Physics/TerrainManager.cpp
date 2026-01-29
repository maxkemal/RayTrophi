#include "TerrainManager.h"
#include "scene_data.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "globals.h"
#include <algorithm>
#include <random>
#include <queue>
#include <filesystem>
#include <fstream>
#include "Texture.h"
#include "Material.h" // Ensure Material class is fully defined

// CUDA Driver API
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "erosion_ops.cuh"
#include "OptixWrapper.h"
#include "OptixAccelManager.h"

// For linking with CUDA Driver API (nvcuda.dll is loaded by driver usually, but we need cuda.lib for symbols)
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")

// Helper function for autoMask
inline float smoothstep(float edge0, float edge1, float x) {
    x = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return x * x * (3 - 2 * x);
}

// Helper for sigmoid clamping
static float padding_sigmoid(float x, float k) {
    return x / (x + k); 
}

TerrainObject* TerrainManager::getTerrain(int id) {
    for (auto& t : terrains) {
        if (t.id == id) return &t;
    }
    return nullptr;
}

void TerrainManager::removeTerrain(SceneData& scene, int id) {
    auto it = std::find_if(terrains.begin(), terrains.end(),
        [id](const TerrainObject& t) { return t.id == id; });

    if (it == terrains.end()) return;

    // Remove triangles from scene
    for (auto& tri : it->mesh_triangles) {
        auto obj_it = std::find(scene.world.objects.begin(), scene.world.objects.end(), tri);
        if (obj_it != scene.world.objects.end()) {
            scene.world.objects.erase(obj_it);
        }
    }
    
    // Actually remove from terrain list
    terrains.erase(it);
}
// Optimization: Use unordered_set for O(1) lookup
#include <unordered_set>

void TerrainManager::removeAllTerrains(SceneData& scene) {
    if (terrains.empty()) return;

    // 1. Collect all triangle pointers to remove
    std::unordered_set<std::shared_ptr<Hittable>> terrain_triangles;
    terrain_triangles.reserve(terrains.size() * 1000); // Estimate
    
    for (auto& t : terrains) {
        for (auto& tri : t.mesh_triangles) {
            terrain_triangles.insert(tri);
        }
    }
    
    // 2. Remove from scene using remove_if (O(N))
    auto& objs = scene.world.objects;
    objs.erase(
        std::remove_if(objs.begin(), objs.end(), 
            [&](const std::shared_ptr<Hittable>& obj) {
                return terrain_triangles.count(obj) > 0;
            }),
        objs.end()
    );
    
    terrains.clear();
    next_id = 1;
}

TerrainObject* TerrainManager::createTerrain(SceneData& scene, int resolution, float size) {
    TerrainObject terrain;
    terrain.id = next_id++;
    terrain.name = "Terrain_" + std::to_string(terrain.id);
    
    // Init Heightmap (Flat)
    terrain.heightmap.width = resolution;
    terrain.heightmap.height = resolution;
    terrain.heightmap.scale_xz = size;
    terrain.heightmap.scale_y = 10.0f; // Default range
    terrain.heightmap.data.resize(resolution * resolution, 0.0f);
    
    // Create Material
    auto mat = std::make_shared<PrincipledBSDF>();
    mat->materialName = "TerrainMat_" + std::to_string(terrain.id);
    auto gpu = std::make_shared<GpuMaterial>();
    gpu->albedo = make_float3(0.5f, 0.5f, 0.5f); // Grey
    gpu->roughness = 0.9f;
    gpu->metallic = 0.0f;
    gpu->opacity = 1.0f;  // CRITICAL: Must be 1.0 for opaque rendering
    gpu->ior = 1.5f;
    gpu->transmission = 0.0f;
    gpu->emission = make_float3(0.0f, 0.0f, 0.0f);
    gpu->clearcoat = 0.0f;
    gpu->subsurface = 0.0f;
    gpu->subsurface_color = make_float3(0.0f, 0.0f, 0.0f);
    mat->gpuMaterial = gpu;
    
    terrain.material_id = MaterialManager::getInstance().getOrCreateMaterialID(mat->materialName, mat);
    
    terrain.transform = std::make_shared<Transform>();
    // Center terrain: (-size/2, 0, -size/2)
    terrain.transform->position = Vec3(-size * 0.5f, 0.0f, -size * 0.5f);
    terrain.transform->updateMatrix();
    
    terrains.push_back(terrain);
    TerrainObject* ptr = &terrains.back();
    
    // Generate Mesh
    updateTerrainMesh(ptr);
    
    // Initialize Layer System (Splat Map + 4 Layers)
    initLayers(ptr);
    
    // Add to Scene
    for (auto& tri : ptr->mesh_triangles) {
        scene.world.objects.push_back(tri);
    }
    
    return ptr;
}

TerrainObject* TerrainManager::createTerrainFromHeightmap(SceneData& scene, const std::string& filepath, float size, float maxHeight, int max_resolution) {
    // -----------------------------------------------------------------------------
    if (!&scene) return nullptr;

    int w, h, channels;
    // Force 1 channel
    unsigned char* img = stbi_load(filepath.c_str(), &w, &h, &channels, 1);
    if (!img) {
        SCENE_LOG_ERROR("Failed to load heightmap: " + filepath);
        return nullptr;
    }

    if (w > 16384 || h > 16384) { // Absolute sanity limit
         SCENE_LOG_WARN("Heightmap is huge! Resizing/Clamping might be needed.");
    }
    if (w < 2 || h < 2) {
         SCENE_LOG_ERROR("Heightmap too small!");
         stbi_image_free(img);
         return nullptr;
    }
    
    TerrainObject terrain;
    terrain.id = next_id++;
    terrain.name = "Terrain_Imported_" + std::to_string(terrain.id);
    
    // Downsampling logic
    int fit_dim = max_resolution; 
    int stride = 1;
    if (w > fit_dim || h > fit_dim) {
        stride = std::max(w, h) / fit_dim;
        if (stride < 1) stride = 1;
        SCENE_LOG_WARN("Heightmap (" + std::to_string(w) + "x" + std::to_string(h) + 
                       "). Downsampling with stride " + std::to_string(stride) + 
                       " to target " + std::to_string(fit_dim));
    }

    int new_w = w / stride;
    int new_h = h / stride;

    // Init Heightmap
    terrain.heightmap.width = new_w;
    terrain.heightmap.height = new_h;
    terrain.heightmap.scale_xz = size;
    terrain.heightmap.scale_y = maxHeight;
    // Resize data
    terrain.heightmap.data.resize(new_w * new_h);
    
    // Copy data with stride
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            int src_x = x * stride;
            int src_y = y * stride;
            // Clamp
            if (src_x >= w) src_x = w - 1;
            if (src_y >= h) src_y = h - 1;
            
            int src_idx = src_y * w + src_x;
            terrain.heightmap.data[y * new_w + x] = img[src_idx] / 255.0f;
        }
    }
    
    stbi_image_free(img);
    
    // Create Material
    auto mat = std::make_shared<PrincipledBSDF>();
    mat->materialName = "TerrainMat_" + std::to_string(terrain.id);
    auto gpu = std::make_shared<GpuMaterial>();
    gpu->albedo = make_float3(0.4f, 0.3f, 0.2f); // Earthy
    gpu->roughness = 1.0f;
    gpu->metallic = 0.0f;
    gpu->opacity = 1.0f;  // CRITICAL: Must be 1.0 for opaque rendering
    gpu->ior = 1.5f;
    gpu->transmission = 0.0f;
    gpu->emission = make_float3(0.0f, 0.0f, 0.0f);
    gpu->clearcoat = 0.0f;
    gpu->subsurface = 0.0f;
    gpu->subsurface_color = make_float3(0.0f, 0.0f, 0.0f);
    mat->gpuMaterial = gpu;
    
    terrain.material_id = MaterialManager::getInstance().getOrCreateMaterialID(mat->materialName, mat);
    
    terrain.transform = std::make_shared<Transform>();
    terrain.transform->position = Vec3(-size * 0.5f, 0.0f, -size * 0.5f);
    terrain.transform->updateMatrix();
    
    terrains.push_back(terrain);
    TerrainObject* ptr = &terrains.back();
    
    // Generate Mesh (with smoothing for heightmaps)
    smoothTerrain(ptr, 1);
    
    // Initialize Layer System (Splat Map + 4 Layers)
    initLayers(ptr);
    
    // Add to Scene
    for (auto& tri : ptr->mesh_triangles) {
        scene.world.objects.push_back(tri);
    }
    
    return ptr;
}

// ===========================================================================
// HEIGHT SAMPLING & RAYCAST
// ===========================================================================

TerrainObject* TerrainManager::getTerrainByName(const std::string& name) {
    for (auto& t : terrains) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

Vec3 TerrainManager::sampleNormal(float worldX, float worldZ) const {
    if (terrains.empty()) return Vec3(0, 1, 0);
    
    for (const auto& terrain : terrains) {
        const Heightmap& hm = terrain.heightmap;
        if (hm.data.empty() || hm.width <= 0 || hm.height <= 0) continue;
        
        // 1. Transform World position to Local terrain space
        Vec3 localPos(worldX, 0, worldZ);
        if (terrain.transform) {
            Matrix4x4 inv = terrain.transform->getFinal().inverse();
            localPos = inv.multiplyVector(Vec4(worldX, 0, worldZ, 1.0f)).xyz();
        }

        // 2. Check if local position is within terrain bounds [0, scale_xz]
        if (localPos.x < 0 || localPos.x > hm.scale_xz || localPos.z < 0 || localPos.z > hm.scale_xz) {
            continue;
        }

        // 3. Convert local position to heightmap grid coordinates
        float normalizedX = localPos.x / hm.scale_xz;
        float normalizedZ = localPos.z / hm.scale_xz;
        
        int ix = (int)(normalizedX * (hm.width - 1) + 0.5f);
        int iz = (int)(normalizedZ * (hm.height - 1) + 0.5f);
        
        ix = std::clamp(ix, 0, hm.width - 1);
        iz = std::clamp(iz, 0, hm.height - 1);
        
        // 4. Calculate Sobel Normal (Local)
        TerrainManager* nonConstThis = const_cast<TerrainManager*>(this);
        TerrainObject* nonConstTerrain = const_cast<TerrainObject*>(&terrain);
        Vec3 localNormal = nonConstThis->calculateSobelNormal(nonConstTerrain, ix, iz);
        
        // 5. Transform Normal to World Space
        if (terrain.transform) {
             localNormal = terrain.transform->getFinal().multiplyVector(Vec4(localNormal, 0.0f)).xyz().normalize();
        }
        
        return localNormal;
    }
    
    return Vec3(0, 1, 0);
}

bool TerrainManager::intersectRay(TerrainObject* terrain, const Ray& r, float& t_out, Vec3& normal_out, float t_min, float t_max) {
    if (!terrain || terrain->heightmap.data.empty()) return false;
    
    // 1. Transform World Ray to Local Space (simplified AABB check)
    // Terrain transform is usually just translation + scale. Rotation is rare for terrain but possible.
    // For now, assume Axis-Aligned Terrain relative to World, just offset by position.
    // If terrain has full transform matrix, we should multiply ray by inverse matrix.
    
    Vec3 ray_orig = r.origin;
    Vec3 ray_dir = r.direction;
    
    if (terrain->transform) {
        Matrix4x4 inv = terrain->transform->getFinal().inverse();
        ray_orig = inv.multiplyVector(Vec4(r.origin, 1.0f)).xyz();
        ray_dir = inv.multiplyVector(Vec4(r.direction, 0.0f)).xyz();
        
        // Ray direction might need re-normalization for some algorithms,
        // but for returning a parameter 't' that works with the original ray,
        // we MUST preserve the scaling between local and world space.
    }
    
    // Terrain AABB (Local)
    float size = terrain->heightmap.scale_xz;
    float maxY = terrain->heightmap.scale_y;
    // Bounds: [0, 0, 0] to [size, maxY, size]
    
    // AABB intersection test (Slabs)
    float t0 = t_min, t1 = t_max;
    
    // X axis: 0..size
    if (abs(ray_dir.x) > 1e-6) {
        float tx1 = (0.0f - ray_orig.x) / ray_dir.x;
        float tx2 = (size - ray_orig.x) / ray_dir.x;
        t0 = std::max(t0, std::min(tx1, tx2));
        t1 = std::min(t1, std::max(tx1, tx2));
    }
    
    // Z axis: 0..size
    if (abs(ray_dir.z) > 1e-6) {
        float tz1 = (0.0f - ray_orig.z) / ray_dir.z;
        float tz2 = (size - ray_orig.z) / ray_dir.z;
        t0 = std::max(t0, std::min(tz1, tz2));
        t1 = std::min(t1, std::max(tz1, tz2));
    }
    
    // Y axis: 0..maxY (Optimization: Check Y range strictly?)
    // Actually terrain can be anywhere between min and max height.
    if (t0 > t1) return false;
    
    // 2. Stepped Grid Traversal (Ray Marching)
    // Walk along the ray from t0 to t1
    // Step size: roughly 1 pixel diagonal
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    
    float cell_size_x = size / (float)(w-1);
    float step_dist = cell_size_x * 0.5f; // Step half a cell for precision
    
    float t_curr = t0;
    
    // Limit steps
    int max_steps = 1000;
    
    // Start point
    Vec3 p = ray_orig + ray_dir * t_curr;
    
    // If outside Y bounds (above max height), advance to intersection with MaxY plane
    if (p.y > maxY && ray_dir.y < 0.0f && abs(ray_dir.y) > 1e-6) {
        float t_enterY = (maxY - ray_orig.y) / ray_dir.y;
        if (t_enterY > t_curr) {
            t_curr = t_enterY;
            p = ray_orig + ray_dir * t_curr;
        }
    }
    
    for (int i=0; i<max_steps; i++) {
        if (t_curr > t1) break;
        
        // Map to grid
        float gx = (p.x / size) * (w - 1);
        float gz = (p.z / size) * (h - 1);
        
        // Check bounds
        if (gx >= 0 && gx < w - 1 && gz >= 0 && gz < h - 1) {
            int ix = (int)gx;
            int iz = (int)gz;
            
            // Bilinear Height
            float fx = gx - ix;
            float fz = gz - iz;
            float h00 = terrain->heightmap.getHeight(ix, iz);
            float h10 = terrain->heightmap.getHeight(ix+1, iz);
            float h01 = terrain->heightmap.getHeight(ix, iz+1);
            float h11 = terrain->heightmap.getHeight(ix+1, iz+1);
            float h_interp = (h00*(1-fx) + h10*fx)*(1-fz) + (h01*(1-fx) + h11*fx)*fz;
            
            // Intersection Check
            // If ray Y is BELOW terrain height, we hit!
            if (p.y <= h_interp) {
                // Determine Hit Point
                
                // Refinement (Binary Search) - Optional but good for precision
                // We are somewhere between t_curr-step and t_curr.
                // Binary search for exact surface cross
                float t_low = t_curr - step_dist;
                float t_high = t_curr;
                 
                for(int k=0; k<5; k++) {
                   float t_mid = (t_low + t_high) * 0.5f;
                   Vec3 p_mid = ray_orig + ray_dir * t_mid;
                   float gx_mid = (p_mid.x / size) * (w - 1);
                   float gz_mid = (p_mid.z / size) * (h - 1);
                   int ix_mid = (int)gx_mid;
                   int iz_mid = (int)gz_mid;
                   float h_val_mid = terrain->heightmap.getHeight(ix_mid, iz_mid); // Use simple nearest for binary step speed or bilinear again
                   
                   // Bilinear again for precision
                   float fx_m = gx_mid - ix_mid; float fz_m = gz_mid - iz_mid;
                   float hm00 = terrain->heightmap.getHeight(ix_mid, iz_mid);
                   float hm10 = terrain->heightmap.getHeight(ix_mid+1, iz_mid);
                   float hm01 = terrain->heightmap.getHeight(ix_mid, iz_mid+1);
                   float hm11 = terrain->heightmap.getHeight(ix_mid+1, iz_mid+1);
                   float h_interp_mid = (hm00*(1-fx_m) + hm10*fx_m)*(1-fz_m) + (hm01*(1-fx_m) + hm11*fx_m)*fz_m;
                   
                   if (p_mid.y <= h_interp_mid) {
                       t_high = t_mid; // Still underground
                   } else {
                       t_low = t_mid; // Above ground
                   }
                }
                
                t_out = t_high;
                
                // Normal at hit point
                // Convert back from t_out
                Vec3 p_final = ray_orig + ray_dir * t_out;
                float gx_final = (p_final.x / size) * (w - 1);
                float gz_final = (p_final.z / size) * (h - 1);
                
                normal_out = calculateSobelNormal(terrain, (int)gx_final, (int)gz_final);
                
                // Transform Normal to World Space (ignore translation, use rotation part of matrix)
                if (terrain->transform) {
                     normal_out = terrain->transform->getFinal().multiplyVector(Vec4(normal_out, 0.0f)).xyz().normalize();
                }
                
                return true;
            }
        }
        
        t_curr += step_dist;
        p = ray_orig + ray_dir * t_curr;
    }
    
    return false;
}

// ===========================================================================
// NORMAL CALCULATION METHODS
// ===========================================================================

Vec3 TerrainManager::calculateFastNormal(TerrainObject* terrain, int x, int z) {
    if (!terrain) return Vec3(0, 1, 0);
    
    auto& hmap = terrain->heightmap;
    // Use uniform step based on max dimension to preserve aspect ratio
    // STRETCH MODE: Width and Height independent steps to fit scale_xz
    // This ensures terrain is always scale_xz * scale_xz in world space
    float step_x = hmap.scale_xz / (float)(std::max(1, hmap.width - 1));
    float step_z = hmap.scale_xz / (float)(std::max(1, hmap.height - 1));
    
    // 4-neighbor central difference
    float hl = hmap.getHeight(x - 1, z);
    float hr = hmap.getHeight(x + 1, z);
    float hu = hmap.getHeight(x, z - 1);
    float hd = hmap.getHeight(x, z + 1);
    
    Vec3 tanX(2.0f * step_x, (hr - hl), 0.0f);
    Vec3 tanZ(0.0f, (hd - hu), 2.0f * step_z);
    
    Vec3 n = Vec3::cross(tanZ, tanX).normalize();
    
    // CRITICAL: Ensure normal always points upward (positive Y)
    // This prevents black triangles from inverted normals on steep erosion channels
    if (n.y < 0.0f) {
        n = -n;
    }
    
    // Additional safety: if normal is too horizontal, bias toward up
    if (n.y < 0.1f) {
        n.y = 0.1f;
        n = n.normalize();
    }
    
    return n;
}

Vec3 TerrainManager::calculateSobelNormal(TerrainObject* terrain, int x, int z) {
    if (!terrain) return Vec3(0, 1, 0);
    
    auto& hmap = terrain->heightmap;
    // Use uniform step based on max dimension to preserve aspect ratio
    // STRETCH MODE: Width and Height independent steps to fit scale_xz
    // This ensures terrain is always scale_xz * scale_xz in world space
    float step_x = hmap.scale_xz / (float)(std::max(1, hmap.width - 1));
    float step_z = hmap.scale_xz / (float)(std::max(1, hmap.height - 1));
    // For Sobel, still use step_x as the main scale factor
    float strength = terrain->normal_strength;
    
    // 8-neighbor heights for Sobel filter
    float hl  = hmap.getHeight(x - 1, z);       // Left
    float hr  = hmap.getHeight(x + 1, z);       // Right
    float hu  = hmap.getHeight(x, z - 1);       // Up
    float hd  = hmap.getHeight(x, z + 1);       // Down
    float hlu = hmap.getHeight(x - 1, z - 1);   // Upper-Left
    float hru = hmap.getHeight(x + 1, z - 1);   // Upper-Right
    float hld = hmap.getHeight(x - 1, z + 1);   // Lower-Left
    float hrd = hmap.getHeight(x + 1, z + 1);   // Lower-Right
    
    // Sobel gradients (weighted 3x3 kernel)
    // Gx: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    // Gz: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    float gx = (hru + 2.0f * hr + hrd) - (hlu + 2.0f * hl + hld);
    float gz = (hld + 2.0f * hd + hrd) - (hlu + 2.0f * hu + hru);
    
    // Construct normal with adjustable strength
    // Correctly handle non-uniform steps (stretched terrain)
    float scaleX = (strength) / (8.0f * step_x);
    float scaleZ = (strength) / (8.0f * step_z);
    
    // Gradient vector (-dHeight/dx, 1, -dHeight/dz)
    Vec3 n(-gx * scaleX, 1.0f, -gz * scaleZ);
    n = n.normalize();
    
    // CRITICAL: Ensure normal always points upward (positive Y)
    // This prevents black triangles from inverted normals on steep erosion channels
    if (n.y < 0.0f) {
        n = -n;
    }
    
    // Additional safety: if normal is too horizontal, bias toward up
    if (n.y < 0.1f) {
        n.y = 0.1f;
        n = n.normalize();
    }
    
    return n;
}

Vec3 TerrainManager::calculateNormal(TerrainObject* terrain, int x, int z) {
    if (!terrain) return Vec3(0, 1, 0);
    
    switch (terrain->normal_quality) {
        case NormalQuality::Fast:
            return calculateFastNormal(terrain, x, z);
        case NormalQuality::Sobel:
        case NormalQuality::HighQuality:
        default:
            return calculateSobelNormal(terrain, x, z);
    }
}

void TerrainManager::updateTerrainMesh(TerrainObject* terrain) {
    if (!terrain) return;
    
    // Check if we need to CREATE triangles (first time) or UPDATE existing
    bool create_new = terrain->mesh_triangles.empty();
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    float scale = terrain->heightmap.scale_xz;
    float max_h = terrain->heightmap.scale_y;
    
    // Calculate step to preserve aspect ratio
    // Calculate step to Stretch to Fit (Square Terrain)
    // Use independent X and Z steps to ensure terrain fills scale_xz * scale_xz area
    float step_x = scale / (float)(std::max(1, w - 1));
    float step_z = scale / (float)(std::max(1, h - 1));
    
    // Pre-calculate vertices for grid
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    positions.resize(w * h);
    normals.resize(w * h);
    
    // 1. Calculate Positions (Local) - Parallelized
    #pragma omp parallel for
    for (int z = 0; z < h; z++) {
        for (int x = 0; x < w; x++) {
            float height_val = terrain->heightmap.data[z * w + x];
            positions[z * w + x] = Vec3(x * step_x, height_val * max_h, z * step_z);
        }
    }
    
    // 2. Calculate Normals (Quality-based) - Parallelized
    #pragma omp parallel for
    for (int z = 0; z < h; z++) {
        for (int x = 0; x < w; x++) {
            normals[z * w + x] = calculateNormal(terrain, x, z);
        }
    }
    
    // 3. Create/Update Triangles
    int tri_idx = 0;
    
    for (int z = 0; z < h - 1; z++) {
        for (int x = 0; x < w - 1; x++) {
            // Indices
            int i0 = z * w + x;
            int i1 = z * w + (x + 1);
            int i2 = (z + 1) * w + (x + 1);
            int i3 = (z + 1) * w + x;
            
            // Local Vertices for this quad
            Vec3 v0 = positions[i0];
            Vec3 v1 = positions[i1];
            Vec3 v2 = positions[i2];
            Vec3 v3 = positions[i3];
            
            Vec3 n0 = normals[i0];
            Vec3 n1 = normals[i1];
            Vec3 n2 = normals[i2];
            Vec3 n3 = normals[i3];
            
            // UVs - Standard 0..1 Mapping
            // Vertex 0 -> UV 0.0
            // Vertex N -> UV 1.0
            // This aligns perfectly with 0..1 generated mask data.
            float divW = (float)(w > 1 ? w - 1 : 1);
            float divH = (float)(h > 1 ? h - 1 : 1);
            
            Vec2 uv0((float)x / divW, (float)z / divH);
            Vec2 uv1((float)(x + 1) / divW, (float)z / divH);
            Vec2 uv2((float)(x + 1) / divW, (float)(z + 1) / divH);
            Vec2 uv3((float)x / divW, (float)(z + 1) / divH);
            
            if (create_new) {
                // Tri 1 - Corrected Winding Order (CCW for Upward Normal)
                // v0(TL) -> v2(BR) -> v1(TR)
                auto tri1 = std::make_shared<Triangle>(
                    v0, v2, v1, 
                    n0, n2, n1, // Normals match vertices
                    uv0, uv2, uv1, 
                    terrain->material_id
                );
                tri1->setNodeName(terrain->name + "_Chunk");
                tri1->setTransformHandle(terrain->transform);
                tri1->updateTransformedVertices();
                
                // Tri 2 - Corrected Winding Order (CCW for Upward Normal)
                // v0(TL) -> v3(BL) -> v2(BR)
                auto tri2 = std::make_shared<Triangle>(
                    v0, v3, v2, 
                    n0, n3, n2, // Normals match vertices
                    uv0, uv3, uv2, 
                    terrain->material_id
                );
                tri2->setNodeName(terrain->name + "_Chunk");
                tri2->setTransformHandle(terrain->transform);
                tri2->updateTransformedVertices();

                // Add to detailed list
                terrain->mesh_triangles.push_back(tri1);
                terrain->mesh_triangles.push_back(tri2);
            } 
            else {
                // UPDATE existing
                if (tri_idx + 1 < terrain->mesh_triangles.size()) {
                    auto& tri1 = terrain->mesh_triangles[tri_idx];
                    auto& tri2 = terrain->mesh_triangles[tri_idx+1];
                    
                    // Update Original Vertices & Normals (Bind Pose)
                    // Tri 1: v0 -> v2 -> v1
                    tri1->setOriginalVertexPosition(0, v0);
                    tri1->setOriginalVertexPosition(1, v2);
                    tri1->setOriginalVertexPosition(2, v1);
                    tri1->setOriginalVertexNormal(0, n0);
                    tri1->setOriginalVertexNormal(1, n2);
                    tri1->setOriginalVertexNormal(2, n1);
                    tri1->updateTransformedVertices();
                    
                    // Tri 2: v0 -> v3 -> v2
                    tri2->setOriginalVertexPosition(0, v0);
                    tri2->setOriginalVertexPosition(1, v3);
                    tri2->setOriginalVertexPosition(2, v2);
                    tri2->setOriginalVertexNormal(0, n0);
                    tri2->setOriginalVertexNormal(1, n3);
                    tri2->setOriginalVertexNormal(2, n2);
                    tri2->updateTransformedVertices();
                }
            }
            tri_idx += 2;
        }
    }
    
    // Clear dirty regions after full update
    terrain->dirty_region.clear();
}

void TerrainManager::rebuildTerrainMesh(SceneData& scene, TerrainObject* terrain) {
    if (!terrain) return;

    // 1. Remove old triangles from scene
    // This removes pointers from the global object list if they match those in the terrain
    auto& objs = scene.world.objects;
    
    // Optimization: Create a set for O(1) lookups
    std::unordered_set<Hittable*> triSet;
    triSet.reserve(terrain->mesh_triangles.size());
    for(auto& t : terrain->mesh_triangles) triSet.insert(t.get());
    
    if (!triSet.empty()) {
        objs.erase(
            std::remove_if(objs.begin(), objs.end(), [&](const std::shared_ptr<Hittable>& obj){
                return triSet.count(obj.get()) > 0;
            }),
            objs.end()
        );
    }
    
    // 2. Clear internal list
    terrain->mesh_triangles.clear();
    
    // 3. Re-generate mesh (triangles)
    updateTerrainMesh(terrain);
    
    // 4. Add new triangles to scene
    objs.reserve(objs.size() + terrain->mesh_triangles.size());
    for (auto& tri : terrain->mesh_triangles) {
        objs.push_back(tri);
    }
    
    // 5. Flag for rebuild
    extern bool g_bvh_rebuild_pending;
    extern bool g_optix_rebuild_pending;
    g_bvh_rebuild_pending = true;
    g_optix_rebuild_pending = true;
}

// ===========================================================================
// INCREMENTAL SECTOR UPDATE (For performance optimization)
// ===========================================================================
void TerrainManager::updateDirtySectors(TerrainObject* terrain) {
    if (!terrain || !terrain->dirty_region.has_any_dirty) return;
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    float scale = terrain->heightmap.scale_xz;
    float max_h = terrain->heightmap.scale_y;
    float step_x = scale / (float)(w - 1);
    float step_z = scale / (float)(h - 1);
    
    int sector_w = w / DirtyRegion::SECTOR_GRID_SIZE;
    int sector_h = h / DirtyRegion::SECTOR_GRID_SIZE;
    
    // Process only dirty sectors
    for (int sy = 0; sy < DirtyRegion::SECTOR_GRID_SIZE; sy++) {
        for (int sx = 0; sx < DirtyRegion::SECTOR_GRID_SIZE; sx++) {
            if (!terrain->dirty_region.sectors[sx][sy]) continue;
            
            // Calculate sector bounds
            int startX = sx * sector_w;
            int startZ = sy * sector_h;
            int endX = std::min(startX + sector_w + 1, w);
            int endZ = std::min(startZ + sector_h + 1, h);
            
            // Update vertices and normals for this sector
            for (int z = startZ; z < endZ; z++) {
                for (int x = startX; x < endX; x++) {
                    // Update triangle vertices
                    if (x < w - 1 && z < h - 1) {
                        int quad_idx = z * (w - 1) + x;
                        int tri_base = quad_idx * 2;
                        
                        if (tri_base + 1 < terrain->mesh_triangles.size()) {
                            float height_val = terrain->heightmap.data[z * w + x];
                            Vec3 pos(x * step_x, height_val * max_h, z * step_z);
                            Vec3 normal = calculateNormal(terrain, x, z);
                            
                            // Update triangles that use this vertex
                            auto& tri1 = terrain->mesh_triangles[tri_base];
                            auto& tri2 = terrain->mesh_triangles[tri_base + 1];
                            
                            tri1->setOriginalVertexPosition(0, pos);
                            tri1->setOriginalVertexNormal(0, normal);
                            tri1->updateTransformedVertices();
                            
                            tri2->setOriginalVertexPosition(0, pos);
                            tri2->setOriginalVertexNormal(0, normal);
                            tri2->updateTransformedVertices();
                        }
                    }
                }
            }
        }
    }
    
    terrain->dirty_region.clear();
    SCENE_LOG_INFO("Updated " + std::to_string(terrain->dirty_region.countDirtySectors()) + " dirty sectors");
}

// mode: 0=Raise, 1=Lower, 2=Flatten, 3=Smooth, 4=Stamp
void TerrainManager::sculpt(TerrainObject* terrain, const Vec3& hitPoint, int mode, float radius, float strength, float dt, 
                            float targetHeight, std::shared_ptr<Texture> stampTexture, float rotation) {
    if (!terrain) return;
    
    // 1. Transform HitPoint to Local Space
    Vec3 localPos;
    if (terrain->transform) {
        Matrix4x4 inv = terrain->transform->getFinal().inverse();
        localPos = inv.multiplyVector(Vec4(hitPoint, 1.0f)).xyz();
    } else {
        localPos = hitPoint;
    }
    
    float size = terrain->heightmap.scale_xz;
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    
    // Map to grid coordinates
    float gridX = (localPos.x / size) * (w - 1);
    float gridZ = (localPos.z / size) * (h - 1);
    
    int cx = (int)(gridX + 0.5f);
    int cy = (int)(gridZ + 0.5f);
    
    // Radius in pixels
    int r_pixels = (int)((radius / size) * (w - 1));
    if (r_pixels < 1) r_pixels = 1;
    
    float brush_strength_frame = strength * dt * 5.0f; // Adjusted for frame
    
    // Smooth kernel pre-calculation or on-the-fly?
    // Using on-the-fly for dynamic updates
    
    bool changed = false;
    float scaleY = terrain->heightmap.scale_y;
    if (scaleY < 0.001f) scaleY = 1.0f;

    // Normalizing target height to 0-1 range
    float normalizedTarget = targetHeight / scaleY;

    for (int y = cy - r_pixels; y <= cy + r_pixels; y++) {
        for (int x = cx - r_pixels; x <= cx + r_pixels; x++) {
            if (x < 0 || x >= w || y < 0 || y >= h) continue;
            
            float dx = (float)(x - gridX);
            float dy = (float)(y - gridZ);
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist > r_pixels) continue;
            
            // Gauss Falloff
            float falloff = 1.0f - (dist / r_pixels);
            falloff = falloff * falloff;
            
            float val = terrain->heightmap.data[y * w + x];
            
            if (mode == 0) { // Raise
                val += brush_strength_frame * falloff;
            } else if (mode == 1) { // Lower
                val -= brush_strength_frame * falloff;
            } else if (mode == 2) { // Flatten
                // Lerp towards target
                float t = brush_strength_frame * falloff; // Influence
                if (t > 1.0f) t = 1.0f;
                // Move current 'val' towards 'normalizedTarget'
                val = val + (normalizedTarget - val) * t;
            } else if (mode == 3) { // Smooth
                // Local box blur (5x5)
                float sum = 0.0f;
                float wSum = 0.0f;
                int r_blur = 2; // Kernel radius
                for(int by = -r_blur; by <= r_blur; by++) {
                    for(int bx = -r_blur; bx <= r_blur; bx++) {
                        int nx = x + bx; 
                        int ny = y + by;
                        if(nx >= 0 && nx < w && ny >= 0 && ny < h) {
                            sum += terrain->heightmap.data[ny * w + nx];
                            wSum += 1.0f;
                        }
                    }
                }
                float avg = (wSum > 0) ? sum / wSum : val;
                
                // Lerp towards average
                float t = brush_strength_frame * falloff; 
                if (t > 1.0f) t = 1.0f;
                val = val + (avg - val) * t;

            } else if (mode == 4) { // Stamp
                if (stampTexture && stampTexture->is_loaded()) {
                     // Calculate UV relative to brush center
                     // Normalize dx, dy to -1..1 range within brush radius
                     float u_raw = dx / (float)r_pixels;
                     float v_raw = dy / (float)r_pixels;
                     
                     // Rotate
                     // x' = x*cos - y*sin
                     // y' = x*sin + y*cos
                     float rad = rotation * 3.14159f / 180.0f;
                     float c = cosf(rad);
                     float s = sinf(rad);
                     
                     float u_rot = u_raw * c - v_raw * s;
                     float v_rot = u_raw * s + v_raw * c;
                     
                     // Map -1..1 to 0..1
                     float u = u_rot * 0.5f + 0.5f;
                     float v = v_rot * 0.5f + 0.5f;
                     
                     if (u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f) {
                         // Sample texture (Red channel usually)
                         // Simple nearest neighbor or bilinear if possible. 
                         // Accessing raw pixels from texture class might be slow or not exposed.
                         // Assuming texture has getPixel or raw buffer access.
                         // If not, we might need to rely on CPU buffer if available.
                         // For now, assuming Texture has 'getPixelIntensity(u,v)' or similar.
                         // Actually Texture class structure is unknown here, let's assume it has `data` vector if CPU accessible.
                         // If generic Texture class doesn't keep CPU data, we can't stamp easily.
                         // FALLBACK: Use a simple noise function if texture access is hard, OR check Texture.h
                         
                         float texVal = stampTexture->sampleIntensity(u, v); // Hypothetical method
                         
                         // Apply
                         val += texVal * brush_strength_frame * falloff;
                     }
                }
            }
            
            // Clamp
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            
            terrain->heightmap.data[y * w + x] = val;
            changed = true;
        }
    }
    
    if (changed) {
        updateTerrainMesh(terrain);
        terrain->dirty_mesh = true; // Signal for BVH rebuild
    }
}

void TerrainManager::smoothTerrain(TerrainObject* terrain, int iterations) {
    if (!terrain) return;
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    
    // Copy data for reading
    std::vector<float> temp = terrain->heightmap.data;
    
    for(int iter = 0; iter < iterations; iter++) {
        for(int y = 0; y < h; y++) {
            for(int x = 0; x < w; x++) {
                float sum = 0.0f;
                int count = 0;
                
                // 3x3 Box Filter
                for(int dy = -1; dy <= 1; dy++) {
                    for(int dx = -1; dx <= 1; dx++) {
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        if(nx >= 0 && nx < w && ny >= 0 && ny < h) {
                            sum += temp[ny * w + nx];
                            count++;
                        }
                    }
                }
                
                terrain->heightmap.data[y * w + x] = sum / (float)count;
            }
        }
        // Update temp for next iteration if needed
        if (iter < iterations - 1) {
            temp = terrain->heightmap.data;
        }
    }
    
    updateTerrainMesh(terrain);
}

// --------------------------------------------------------------------------------------------
// TERRAIN LAYER SYSTEM
// --------------------------------------------------------------------------------------------

void TerrainManager::initLayers(TerrainObject* terrain) {
    if (!terrain) return;

    // 1. Initialize Splat Map
    if (!terrain->splatMap) {
        // Create Splat Map (RGBA), default Red channel (Layer 0) active
        terrain->splatMap = std::make_shared<Texture>("SplatMap_" + terrain->name, TextureType::Unknown);
        
        // Resolution: Match heightmap but at least 512
        int sW = std::max(512, terrain->heightmap.width);
        int sH = std::max(512, terrain->heightmap.height);
        
        terrain->splatMap->width = sW;
        terrain->splatMap->height = sH;
        terrain->splatMap->pixels.resize(sW * sH);
        
        // Fill with Red (Layer 0)
        for (auto& p : terrain->splatMap->pixels) {
            p.r = 255; p.g = 0; p.b = 0; p.a = 0;
        }
        
        terrain->splatMap->m_is_loaded = true;
        terrain->splatMap->upload_to_gpu(); // First upload
    }

    // 2. Initialize Layers (if empty)
    if (terrain->layers.empty()) {
        terrain->layers.resize(4);
        terrain->layer_uv_scales.resize(4, 50.0f); // Default tiling
        
        // Assign placeholders or leave null
        // Users will assign materials via UI
    }
}

// Resize splatmap to match heightmap dimensions (bilinear interpolation)
void TerrainManager::resizeSplatMap(TerrainObject* terrain) {
    if (!terrain || !terrain->splatMap) return;
    
    int targetW = std::max(512, terrain->heightmap.width);
    int targetH = std::max(512, terrain->heightmap.height);
    
    int srcW = terrain->splatMap->width;
    int srcH = terrain->splatMap->height;
    
    // Skip if already correct size
    if (srcW == targetW && srcH == targetH) return;
    
    SCENE_LOG_INFO("[TerrainManager] Resizing splatmap from " + std::to_string(srcW) + "x" + std::to_string(srcH) + 
                   " to " + std::to_string(targetW) + "x" + std::to_string(targetH));
    
    // Create new pixel buffer
    std::vector<CompactVec4> newPixels(targetW * targetH);
    
    // Bilinear interpolation
    for (int y = 0; y < targetH; ++y) {
        for (int x = 0; x < targetW; ++x) {
            // Normalized coordinates [0, 1]
            float u = (float)x / (float)(targetW > 1 ? targetW - 1 : 1);
            float v = (float)y / (float)(targetH > 1 ? targetH - 1 : 1);
            
            // Source coordinates
            float srcX = u * (srcW - 1);
            float srcY = v * (srcH - 1);
            
            int x0 = (int)srcX;
            int y0 = (int)srcY;
            int x1 = std::min(x0 + 1, srcW - 1);
            int y1 = std::min(y0 + 1, srcH - 1);
            
            float fx = srcX - x0;
            float fy = srcY - y0;
            
            // Sample 4 corners
            auto& p00 = terrain->splatMap->pixels[y0 * srcW + x0];
            auto& p10 = terrain->splatMap->pixels[y0 * srcW + x1];
            auto& p01 = terrain->splatMap->pixels[y1 * srcW + x0];
            auto& p11 = terrain->splatMap->pixels[y1 * srcW + x1];
            
            // Interpolate
            float r = (1-fx)*(1-fy)*p00.r + fx*(1-fy)*p10.r + (1-fx)*fy*p01.r + fx*fy*p11.r;
            float g = (1-fx)*(1-fy)*p00.g + fx*(1-fy)*p10.g + (1-fx)*fy*p01.g + fx*fy*p11.g;
            float b = (1-fx)*(1-fy)*p00.b + fx*(1-fy)*p10.b + (1-fx)*fy*p01.b + fx*fy*p11.b;
            float a = (1-fx)*(1-fy)*p00.a + fx*(1-fy)*p10.a + (1-fx)*fy*p01.a + fx*fy*p11.a;
            
            newPixels[y * targetW + x].r = (uint8_t)std::clamp(r, 0.0f, 255.0f);
            newPixels[y * targetW + x].g = (uint8_t)std::clamp(g, 0.0f, 255.0f);
            newPixels[y * targetW + x].b = (uint8_t)std::clamp(b, 0.0f, 255.0f);
            newPixels[y * targetW + x].a = (uint8_t)std::clamp(a, 0.0f, 255.0f);
        }
    }
    
    // Swap in new data
    terrain->splatMap->pixels = std::move(newPixels);
    terrain->splatMap->width = targetW;
    terrain->splatMap->height = targetH;
    terrain->splatMap->upload_to_gpu();
}

void TerrainManager::updateSplatMapTexture(TerrainObject* terrain) {
    if (terrain && terrain->splatMap) {
        terrain->splatMap->updateGPU();
    }
}

void TerrainManager::paintSplatMap(TerrainObject* terrain, const Vec3& hitPoint, int channel, float radius, float strength, float dt) {
    if (!terrain || !terrain->splatMap) return;

    // Transform HitPoint to Local Space
    Vec3 localPos;
    if (terrain->transform) {
        Matrix4x4 inv = terrain->transform->getFinal().inverse();
        localPos = inv.multiplyVector(Vec4(hitPoint, 1.0f)).xyz();
    } else {
        localPos = hitPoint;
    }
    
    // Convert to Splat Map Coords
    float size = terrain->heightmap.scale_xz;
    int w = terrain->splatMap->width;
    int h = terrain->splatMap->height;
    
    float gridX = (localPos.x / size) * (w - 1);
    float gridZ = (localPos.z / size) * (h - 1);
    
    int cx = (int)(gridX + 0.5f);
    int cy = (int)(gridZ + 0.5f);
    
    // Radius in pixels
    int r_pixels = (int)((radius / size) * (w - 1));
    if (r_pixels < 1) r_pixels = 1;

    float brush_strength_frame = strength * dt * 5.0f;
    bool changed = false;

    for (int y = cy - r_pixels; y <= cy + r_pixels; y++) {
        for (int x = cx - r_pixels; x <= cx + r_pixels; x++) {
            if (x < 0 || x >= w || y < 0 || y >= h) continue;
            
            float dx = (float)(x - gridX);
            float dy = (float)(y - gridZ);
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist > r_pixels) continue;
            
            float falloff = 1.0f - (dist / r_pixels);
            falloff = falloff * falloff * falloff; // Sharper falloff
            
            auto& p = terrain->splatMap->pixels[y * w + x];
            float vals[4] = { p.r / 255.0f, p.g / 255.0f, p.b / 255.0f, p.a / 255.0f };
            
            // Add weight to target channel
            vals[channel] += brush_strength_frame * falloff;
            if (vals[channel] > 1.0f) vals[channel] = 1.0f;
            
            // Normalize
            float sum = vals[0] + vals[1] + vals[2] + vals[3];
            if (sum > 0.001f) {
                vals[0] /= sum; vals[1] /= sum; vals[2] /= sum; vals[3] /= sum;
            } else {
                vals[0] = 1.0f; vals[1] = 0; vals[2] = 0; vals[3] = 0;
            }
            
            p.r = (uint8_t)(vals[0] * 255.0f);
            p.g = (uint8_t)(vals[1] * 255.0f);
            p.b = (uint8_t)(vals[2] * 255.0f);
            p.a = (uint8_t)(vals[3] * 255.0f);
            
            changed = true;
        }
    }

    if (changed) {
        updateSplatMapTexture(terrain);
    }
}

void TerrainManager::autoMask(TerrainObject* terrain, float slopeWeight, float heightWeight, float heightMin, float heightMax, float slopeSteepness) {
    if (!terrain || !terrain->splatMap) return;
    
    int w = terrain->splatMap->width;
    int h = terrain->splatMap->height;
    float max_h = terrain->heightmap.scale_y;
    float scale = terrain->heightmap.scale_xz;
    
    SCENE_LOG_INFO("[autoMask] SplatMap: " + std::to_string(w) + "x" + std::to_string(h) + 
                   ", Heightmap: " + std::to_string(terrain->heightmap.width) + "x" + std::to_string(terrain->heightmap.height) +
                   ", scale_xz=" + std::to_string(scale) + ", scale_y=" + std::to_string(max_h));
    
    // Calculate cell size for correct slope calculation (Rise / Run)
    // Run = Distance between HEIGHTMAP pixels in world space (since we sample neighbors in heightmap)
    float hmCellSizeX = scale / (float)(std::max(1, terrain->heightmap.width - 1));
    float hmCellSizeZ = scale / (float)(std::max(1, terrain->heightmap.height - 1));

    // Get world position Y for global height check
    float worldPosY = 0.0f;
    if (terrain->transform) {
        worldPosY = terrain->transform->position.y;
    }
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // Sample height and slope at this UV
            // Standard 0..1 Mapping (Aligns with Mesh UVs)
            float u = (float)x / (float)(w > 1 ? w - 1 : 1);
            float v = (float)y / (float)(h > 1 ? h - 1 : 1);
            
            // Map to heightmap coords
            // u is 0..1, maps to 0..Width-1
            float hx_f = u * (terrain->heightmap.width - 1);
            float hy_f = v * (terrain->heightmap.height - 1);
            int hx = (int)(hx_f + 0.5f); // Nearest neighbor
            int hy = (int)(hy_f + 0.5f);
            
            hx = std::clamp(hx, 0, terrain->heightmap.width - 1);
            hy = std::clamp(hy, 0, terrain->heightmap.height - 1);
            
            // getHeight already returns height * scale_y
            float local_height = terrain->heightmap.getHeight(hx, hy);
            float global_height = local_height + worldPosY;
            
            // Calculate Slope
            // Neighbors in Heightmap Grid
            float hl = terrain->heightmap.getHeight(hx - 1, hy);
            float hr = terrain->heightmap.getHeight(hx + 1, hy);
            float hu = terrain->heightmap.getHeight(hx, hy - 1);
            float hd = terrain->heightmap.getHeight(hx, hy + 1);
            
            // Slope magnitude (Rise / Run)
            // Distance between hx-1 and hx+1 is 2 * HeightmapCellSize
            float dX = fabsf(hr - hl) / (2.0f * hmCellSizeX); 
            float dZ = fabsf(hd - hu) / (2.0f * hmCellSizeZ);
            
            // Tangent of the angle
            float slopeTan = sqrtf(dX*dX + dZ*dZ); 
            
            // Normalize slope for weight calculation (0..1 range)
            // slopeSteepness now acts as a scaler for "What is considered steep?"
            // e.g. if steepness is 5.0, a slope of 1/5 = 0.2 (approx 11 degrees) starts becoming rock
            float normalizedSlope = slopeTan * slopeSteepness; 
            normalizedSlope = std::min(normalizedSlope, 1.0f);
            
            // Logic:
            // Layer 0 (R): Base/Grass (Flat)
            // Layer 1 (G): Rock (Steep)
            // Layer 2 (B): Snow (High)
            // Layer 3 (A): Dirt (Transition/Noise)
            
            float w_rock = smoothstep(0.2f, 0.8f, normalizedSlope); // Adjusted thresholds for tangent slope
            float w_snow = smoothstep(heightMin, heightMax, global_height);
            float w_rest = 1.0f - w_rock; // What remains for flat
            
            // Base layer gets the rest, masked by snow
            float w_grass = w_rest * (1.0f - w_snow);
            
            // Snow covers everything implies max priority?
            // Usually Key: Rock overrides Grass. Snow overrides everything.
            // Let's blend.
            
            float final_R = w_grass; // Grass
            float final_G = w_rock * (1.0f - w_snow); // Rock (covered by snow)
            float final_B = w_snow; // Snow
            float final_A = 0.0f; // Unused for now
            
            // Normalize
            float sum = final_R + final_G + final_B + final_A;
            if (sum > 0.001f) {
                final_R /= sum; final_G /= sum; final_B /= sum; final_A /= sum;
            }
            
            auto& p = terrain->splatMap->pixels[y * w + x];
            p.r = (uint8_t)(final_R * 255.0f);
            p.g = (uint8_t)(final_G * 255.0f);
            p.b = (uint8_t)(final_B * 255.0f);
            p.a = (uint8_t)(final_A * 255.0f);
        }
    }
    
    updateSplatMapTexture(terrain);
}

void TerrainManager::exportSplatMap(TerrainObject* terrain, const std::string& filepath) {
    if (!terrain || !terrain->splatMap || terrain->splatMap->pixels.empty()) {
        SCENE_LOG_WARN("Cannot export splat map: no data");
        return;
    }
    
    int sw = terrain->splatMap->width;
    int sh = terrain->splatMap->height;
    std::vector<uint8_t> rgba(sw * sh * 4);
    
    for (int y = 0; y < sh; y++) {
        for (int x = 0; x < sw; x++) {
            int src_idx = y * sw + x;
            int dst_idx = ((sh - 1) - y) * sw + x; // Flip Y for PNG (Near -> Bottom)
            const auto& p = terrain->splatMap->pixels[src_idx];
            rgba[dst_idx * 4 + 0] = p.r;
            rgba[dst_idx * 4 + 1] = p.g;
            rgba[dst_idx * 4 + 2] = p.b;
            rgba[dst_idx * 4 + 3] = 255;
        }
    }
    
    stbi_write_png(filepath.c_str(), sw, sh, 4, rgba.data(), sw * 4);
}

// ===========================================================================
// EROSION SYSTEM - Quality Deformations
// ===========================================================================

// ===========================================================================
// EDGE PRESERVATION HELPER
// Saves original edge heights and blends back after terrain modifications
// ===========================================================================
void TerrainManager::preserveEdges(TerrainObject* terrain, const std::vector<float>& originalHeights, int fadeWidth) {
    if (!terrain || originalHeights.empty()) return;
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    auto& data = terrain->heightmap.data;
    
    if ((int)originalHeights.size() != w * h) return;
    
    // Blend current heights back to original at edges using smooth gradient
    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            
            // Calculate distance from nearest edge
            int distFromEdge = std::min({x, y, w - 1 - x, h - 1 - y});
            
            if (distFromEdge < fadeWidth) {
                // Smoothstep blend: 0 at edge, 1 at fadeWidth
                float t = (float)distFromEdge / (float)fadeWidth;
                float blend = t * t * (3.0f - 2.0f * t);  // Smoothstep
                
                // Blend between original (at edge) and modified (interior)
                data[idx] = originalHeights[idx] * (1.0f - blend) + data[idx] * blend;
            }
        }
    }
}

// Helper to get recommended fade width based on terrain size
int TerrainManager::getEdgeFadeWidth(TerrainObject* terrain) {
    if (!terrain) return 5;
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    // ~3-5% of terrain size, minimum 5 cells, maximum 20 cells
    return std::clamp(std::min(w, h) / 25, 5, 20);
}

// ===========================================================================
// EROSION SYSTEM
// ===========================================================================

void TerrainManager::hydraulicErosion(TerrainObject* terrain, const HydraulicErosionParams& p, const std::vector<float>& mask) {
    if (!terrain) return;
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    auto& data = terrain->heightmap.data;
    float cellSize = terrain->heightmap.scale_xz / w;
    bool hasHardness = !terrain->hardnessMap.empty();
    bool hasMask = !mask.empty() && mask.size() == w * h;
    
    // EDGE PRESERVATION: Save original heights before erosion
    std::vector<float> originalHeights = data;
    int fadeWidth = getEdgeFadeWidth(terrain);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);
    
    // Brush weights for erosion distribution (prevents spikes)
    std::vector<float> brushWeights;
    std::vector<int> brushOffsets;
    int brushRadius = p.erosionRadius;
    float weightSum = 0.0f;
    
    for (int dy = -brushRadius; dy <= brushRadius; dy++) {
        for (int dx = -brushRadius; dx <= brushRadius; dx++) {
            float dist = sqrtf((float)(dx*dx + dy*dy));
            if (dist <= brushRadius) {
                float weight = 1.0f - (dist / brushRadius);
                weight = weight * weight * (3 - 2 * weight); // Smoothstep
                brushWeights.push_back(weight);
                brushOffsets.push_back(dy * w + dx);
                weightSum += weight;
            }
        }
    }
    // Normalize weights
    for (float& w : brushWeights) w /= weightSum;
    
    for (int iter = 0; iter < p.iterations; iter++) {
        float posX = distrib(gen) * (w - 3) + 1;
        float posY = distrib(gen) * (h - 3) + 1;
        float dirX = 0, dirY = 0;
        float speed = 1.0f;
        float water = 1.0f;
        float sediment = 0.0f;
        
        for (int lifetime = 0; lifetime < p.dropletLifetime; lifetime++) {
            int nodeX = (int)posX;
            int nodeY = (int)posY;
            if (nodeX < brushRadius || nodeX >= w - brushRadius || 
                nodeY < brushRadius || nodeY >= h - brushRadius) break;
            int idx = nodeY * w + nodeX;
            
            // Calculate gradient (downhill direction)
            float gradX = data[idx + 1] - data[idx - 1];
            float gradY = data[idx + w] - data[idx - w];
            
            // Store previous direction for momentum preservation
            float prevDirX = dirX;
            float prevDirY = dirY;
            
            // Update direction with inertia
            dirX = dirX * p.inertia - gradX * (1 - p.inertia);
            dirY = dirY * p.inertia - gradY * (1 - p.inertia);
            
            float len = sqrtf(dirX*dirX + dirY*dirY);
            if (len < 0.0001f) {
                // No gradient AND no momentum - use previous direction with slight random perturbation
                // This prevents droplets from stopping in pits
                dirX = prevDirX + (distrib(gen) * 0.4f - 0.2f);
                dirY = prevDirY + (distrib(gen) * 0.4f - 0.2f);
                len = sqrtf(dirX*dirX + dirY*dirY);
                if (len < 0.0001f) {
                    // Truly stuck - pick random
                    dirX = distrib(gen) * 2.0f - 1.0f;
                    dirY = distrib(gen) * 2.0f - 1.0f;
                    len = sqrtf(dirX*dirX + dirY*dirY);
                }
            }
            dirX /= len; dirY /= len;
            
            float newPosX = posX + dirX;
            float newPosY = posY + dirY;
            
            if (newPosX < brushRadius || newPosX >= w - brushRadius || 
                newPosY < brushRadius || newPosY >= h - brushRadius) break;
            
            int newIdx = (int)newPosY * w + (int)newPosX;
            float deltaHeight = data[newIdx] - data[idx];
            
            // INDUSTRY STANDARD: Stream Power Law sediment capacity
            // C = Kc * velocity * slope^n * water (n typically 0.5-1.0)
            // Reference: Whipple & Tucker (1999)
            float slope = fabsf(deltaHeight); // Local slope approximation
            float slopeFactor = sqrtf(slope + 0.001f); // slope^0.5 with epsilon
            float capacity = fmaxf(-deltaHeight * speed * water * p.sedimentCapacity * slopeFactor, p.minSlope);
            
            // Hardness factor
            float hardness = 0.0f;
            if (!terrain->hardnessMap.empty() && idx < (int)terrain->hardnessMap.size()) {
                hardness = terrain->hardnessMap[idx];
            }
            float hardnessFactor = 1.0f - hardness * 0.9f;
            
            // EROSION vs DEPOSITION decision
            if (deltaHeight > 0) {
                // Going UPHILL - but with momentum, we should ERODE through obstacles
                // This creates proper drainage channels instead of ponds
                
                if (speed > 0.5f) {
                    // High momentum - erode the obstacle to create channel
                    float erodeAmount = fminf(deltaHeight * 0.3f * hardnessFactor, data[newIdx] * 0.02f);
                    
                    // Erode the blocking cell to carve through
                    for (size_t b = 0; b < brushOffsets.size(); b++) {
                        int brushIdx = newIdx + brushOffsets[b];
                        if (brushIdx >= 0 && brushIdx < w * h) {
                            data[brushIdx] -= erodeAmount * brushWeights[b];
                        }
                    }
                    sediment += erodeAmount;
                    
                    // Slow down but don't stop
                    speed *= 0.7f;
                } else {
                    // Low momentum - deposit some sediment but keep moving
                    float deposit = fminf(sediment * 0.3f, deltaHeight * 0.5f);
                    sediment -= deposit;
                    data[idx] += deposit;  // Deposit at current position, not blocking one
                    
                    // Try to find alternate route (simulate water flowing around obstacle)
                    // Bias direction perpendicular to obstacle
                    float perpX = -gradY;
                    float perpY = gradX;
                    float perpLen = sqrtf(perpX*perpX + perpY*perpY);
                    if (perpLen > 0.001f) {
                        dirX = perpX / perpLen;
                        dirY = perpY / perpLen;
                    }
                }
            }
            else if (sediment > capacity) {
                // Too much sediment - but DON'T deposit in pits/flat areas near slopes
                // Only deposit if we're on a gentle slope away from accumulation areas
                float slope = fabsf(deltaHeight);
                
                // Check if this is a local minimum (pit) - don't deposit here!
                float hL = data[idx - 1];
                float hR = data[idx + 1];
                float hU = data[idx - w];
                float hD = data[idx + w];
                float minNeighbor = fminf(fminf(hL, hR), fminf(hU, hD));
                bool isPit = (data[idx] < minNeighbor - 0.001f);
                
                if (!isPit && slope > p.minSlope * 0.3f) {
                    // On a slope, not in pit - safe to deposit
                    float deposit = (sediment - capacity) * p.depositSpeed * 0.5f;
                    sediment -= deposit;
                    data[newIdx] += deposit;
                }
                // In pits: just carry the sediment forward, it will deposit somewhere better
            }
            else {
                // CAN ERODE - sediment below capacity
                float erode = fminf((capacity - sediment) * p.erodeSpeed, -deltaHeight);
                erode = fminf(erode, data[idx] * 0.05f); // Max 5% of height per step
                erode *= hardnessFactor;
                
                // MASK INFLUENCE
                if (hasMask) {
                    erode *= mask[idx];
                }
                
                // BRUSH EROSION - distribute erosion to prevent spikes
                for (size_t b = 0; b < brushOffsets.size(); b++) {
                    int brushIdx = idx + brushOffsets[b];
                    if (brushIdx >= 0 && brushIdx < w * h) {
                        float erodeHere = erode * brushWeights[b];
                        data[brushIdx] -= erodeHere;
                    }
                }
                sediment += erode;
            }
            
            // Update physics
            speed = sqrtf(fmaxf(0.01f, speed*speed + deltaHeight * p.gravity));
            water *= (1 - p.evaporateSpeed);
            posX = newPosX;
            posY = newPosY;
            
            if (water < 0.01f) break; // Droplet evaporated
        }
    }
    
    // POST-EROSION: Spike removal pass (gentle thermal smoothing)
    // This eliminates any remaining isolated spikes
    std::vector<float> smoothed = data;
    #pragma omp parallel for
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            float hC = data[idx];
            
            // Check if this is a local maximum (spike)
            float hL = data[idx - 1];
            float hR = data[idx + 1];
            float hU = data[idx - w];
            float hD = data[idx + w];
            float avgNeighbor = (hL + hR + hU + hD) * 0.25f;
            
            // If significantly higher than neighbors, smooth it
            if (hC > avgNeighbor + cellSize * 0.1f) {
                smoothed[idx] = avgNeighbor * 0.3f + hC * 0.7f;
            }
        }
    }
    data = smoothed;
    
    // POST-EROSION: Pit filling pass (eliminates micro-ponds that cause black triangles)
    // Same as spike removal but for local minima
    smoothed = data;
    #pragma omp parallel for
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            float hC = data[idx];
            
            // Check if this is a local minimum (pit)
            float hL = data[idx - 1];
            float hR = data[idx + 1];
            float hU = data[idx - w];
            float hD = data[idx + w];
            float minNeighbor = fminf(fminf(hL, hR), fminf(hU, hD));
            float avgNeighbor = (hL + hR + hU + hD) * 0.25f;
            
            // If significantly lower than neighbors (pit), fill it up
            if (hC < minNeighbor - cellSize * 0.05f) {
                // Fill to average of neighbors to eliminate the pit
                smoothed[idx] = avgNeighbor * 0.5f + hC * 0.5f;
            }
        }
    }
    data = smoothed;
    
    // EDGE FADE-OUT: Prevent "wall" effect at terrain boundaries
    int edgeFadeWidth = std::max(3, w / 80);  // Smaller fade for hydraulic
    
    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            int distFromEdge = std::min({x, y, w - 1 - x, h - 1 - y});
            
            if (distFromEdge < edgeFadeWidth) {
                float t = (float)distFromEdge / (float)edgeFadeWidth;
                float smoothT = t * t * (3.0f - 2.0f * t);  // Smoothstep
                
                // Look inward for reference
                int inwardX = std::clamp(x + (x < w/2 ? edgeFadeWidth : -edgeFadeWidth), 0, w-1);
                int inwardY = std::clamp(y + (y < h/2 ? edgeFadeWidth : -edgeFadeWidth), 0, h-1);
                float inwardHeight = data[inwardY * w + inwardX];
                
                data[idx] = data[idx] * smoothT + inwardHeight * 0.4f * (1.0f - smoothT);
            }
        }
    }
    
    // EDGE PRESERVATION: Restore original edge heights with gradient blend
    preserveEdges(terrain, originalHeights, fadeWidth);
    
    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("Hydraulic erosion completed: " + std::to_string(p.iterations) + " droplets");
}

void TerrainManager::thermalErosion(TerrainObject* terrain, const ThermalErosionParams& p, const std::vector<float>& mask) {
    if (!terrain) return;
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    auto& data = terrain->heightmap.data;
    bool hasHardness = !terrain->hardnessMap.empty();
    bool hasMask = !mask.empty() && mask.size() == w * h;
    
    std::vector<float> temp(data.size());
    
    // D8 neighbor offsets and distance weights (diagonals are sqrt(2) apart)
    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const float distWeight[8] = {0.707f, 1.0f, 0.707f, 1.0f, 1.0f, 0.707f, 1.0f, 0.707f}; // 1/sqrt(2) for diagonals
    
    for (int iter = 0; iter < p.iterations; iter++) {
        std::copy(data.begin(), data.end(), temp.begin());
        
        #pragma omp parallel for
        for (int y = 1; y < h-1; y++) {
            for (int x = 1; x < w-1; x++) {
                int idx = y * w + x;
                float center = temp[idx];
                
                // Hardness modifies Talus Angle
                float hardness = hasHardness ? terrain->hardnessMap[idx] : 0.0f;
                float effectiveTalus = p.talusAngle + hardness * 0.3f;
                
                // INDUSTRY STANDARD: Weighted multi-neighbor transfer
                // Calculate all height differences and total excess slope
                float diffs[8];
                int nIdxs[8];
                float totalExcess = 0.0f;
                int neighborCount = 0;
                
                for (int d = 0; d < 8; d++) {
                    int nx = x + dx[d];
                    int ny = y + dy[d];
                    nIdxs[d] = ny * w + nx;
                    
                    // Adjust talus for diagonal distance
                    float adjustedTalus = effectiveTalus * distWeight[d];
                    float diff = center - temp[nIdxs[d]];
                    
                    if (diff > adjustedTalus) {
                        diffs[d] = diff - adjustedTalus;
                        totalExcess += diffs[d];
                        neighborCount++;
                    } else {
                        diffs[d] = 0.0f;
                    }
                }
                
                // Distribute material proportionally to all steep neighbors
                if (totalExcess > 0.0f && neighborCount > 0) {
                    float totalMove = totalExcess * 0.5f * p.erosionAmount;
                    
                    if (hasMask) {
                        totalMove *= mask[idx];
                    }
                    
                    // Hardness limits movement rate
                    totalMove *= (1.0f - hardness * 0.4f);
                    
                    // Remove from center
                    #pragma omp atomic
                    data[idx] -= totalMove;
                    
                    // Distribute to neighbors proportionally
                    for (int d = 0; d < 8; d++) {
                        if (diffs[d] > 0.0f) {
                            float weight = diffs[d] / totalExcess;
                            #pragma omp atomic
                            data[nIdxs[d]] += totalMove * weight;
                        }
                    }
                }
            }
        }
    }
    
    // POST-EROSION: Pit filling pass (eliminates micro-ponds that cause black triangles)
    float cellSize = terrain->heightmap.scale_xz / w;
    std::vector<float> smoothed = data;
    #pragma omp parallel for
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            float hC = data[idx];
            
            float hL = data[idx - 1];
            float hR = data[idx + 1];
            float hU = data[idx - w];
            float hD = data[idx + w];
            float minNeighbor = fminf(fminf(hL, hR), fminf(hU, hD));
            float avgNeighbor = (hL + hR + hU + hD) * 0.25f;
            
            // If significantly lower than neighbors (pit), fill it up
            if (hC < minNeighbor - cellSize * 0.05f) {
                smoothed[idx] = avgNeighbor * 0.5f + hC * 0.5f;
            }
        }
    }
    data = smoothed;
    
    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("Thermal erosion completed");
}

// ===========================================================================
// WIND EROSION - INDUSTRY STANDARD (Shadow Zone + Saltation Model)
// Based on Bagnold's aeolian transport theory with wind shadow detection
// Reference: Bagnold (1941), Werner (1995) dune formation model
// ===========================================================================
void TerrainManager::windErosion(TerrainObject* terrain, float strength, float direction, int iterations, const std::vector<float>& mask) {
    if (!terrain) return;
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    auto& data = terrain->heightmap.data;
    bool hasHardness = !terrain->hardnessMap.empty();
    bool hasMask = !mask.empty() && mask.size() == w * h;
    
    // Normalize wind direction
    float dirRad = direction * 3.14159f / 180.0f;
    float normWindX = cosf(dirRad);
    float normWindY = sinf(dirRad);
    
    // Shadow zone parameters (Bagnold theory)
    const int SHADOW_STEPS = 6;
    const float SHADOW_ANGLE = 0.15f; // ~8.5 degrees shadow angle (typical for sand)
    
    for (int iter = 0; iter < iterations; iter++) {
        // Use temp buffer for stable reads during iteration
        std::vector<float> temp = data;
        
        #pragma omp parallel for
        for (int y = 3; y < h-3; y++) {
            for (int x = 3; x < w-3; x++) {
                int idx = y * w + x;
                float currentH = temp[idx];
                
                // Get hardness at this location
                float hardness = hasHardness ? terrain->hardnessMap[idx] : 0.0f;
                
                // ========================================
                // SHADOW ZONE DETECTION (Industry Standard)
                // ========================================
                // Ray march upwind to detect if we're in a wind shadow
                float shadowFactor = 1.0f;
                
                for (int step = 1; step <= SHADOW_STEPS; step++) {
                    int checkX = x - (int)(normWindX * step);
                    int checkY = y - (int)(normWindY * step);
                    
                    // Clamp to bounds
                    checkX = std::clamp(checkX, 0, w - 1);
                    checkY = std::clamp(checkY, 0, h - 1);
                    
                    int checkIdx = checkY * w + checkX;
                    float upwindH = temp[checkIdx];
                    
                    // Check if upwind point casts shadow over current point
                    float shadowHeight = upwindH - step * SHADOW_ANGLE;
                    if (currentH < shadowHeight) {
                        shadowFactor *= 0.3f; // In shadow - reduce erosion
                    }
                }
                
                // Calculate upwind/downwind positions
                int upwindX = x - (int)(normWindX * 2);
                int upwindY = y - (int)(normWindY * 2);
                upwindX = std::clamp(upwindX, 0, w - 1);
                upwindY = std::clamp(upwindY, 0, h - 1);
                int upwindIdx = upwindY * w + upwindX;
                
                int downwindX = x + (int)(normWindX * 2);
                int downwindY = y + (int)(normWindY * 2);
                downwindX = std::clamp(downwindX, 0, w - 1);
                downwindY = std::clamp(downwindY, 0, h - 1);
                int downwindIdx = downwindY * w + downwindX;
                
                // Far downwind for saltation jump
                int farDownwindX = x + (int)(normWindX * 4);
                int farDownwindY = y + (int)(normWindY * 4);
                farDownwindX = std::clamp(farDownwindX, 0, w - 1);
                farDownwindY = std::clamp(farDownwindY, 0, h - 1);
                int farDownwindIdx = farDownwindY * w + farDownwindX;
                
                float upwindH = temp[upwindIdx];
                float downwindH = temp[downwindIdx];
                
                // Windward slope (facing into wind)
                float windwardSlope = currentH - upwindH;
                
                // Leeward slope (sheltered side)
                float leewardSlope = currentH - downwindH;
                
                // ========================================
                // EROSION: Windward faces (exposed to wind)
                // ========================================
                if (windwardSlope > 0.0f && shadowFactor > 0.5f) {
                    // Saltation erosion - stronger on exposed windward slopes
                    float erosionAmount = windwardSlope * strength * 0.001f * shadowFactor;
                    
                    // Abrasion boost on steep slopes
                    float abrasionBoost = 1.0f + windwardSlope * 2.0f;
                    erosionAmount *= abrasionBoost;
                    
                    // Hardness resistance
                    erosionAmount *= (1.0f - hardness * 0.8f);
                    
                    if (hasMask) {
                        erosionAmount *= mask[idx];
                    }
                    
                    #pragma omp atomic
                    data[idx] -= erosionAmount;
                    
                    // Saltation: material jumps and lands downwind
                    // Split between immediate downwind and further saltation jump
                    #pragma omp atomic
                    data[downwindIdx] += erosionAmount * 0.5f;
                    #pragma omp atomic
                    data[farDownwindIdx] += erosionAmount * 0.3f;
                }
                
                // ========================================
                // DEPOSITION: Shadow zones and leeward faces
                // ========================================
                if (shadowFactor < 0.7f || leewardSlope > 0.0f) {
                    // In shadow zone or on leeward slope - sediment settles
                    float depositionAmount = (1.0f - shadowFactor) * strength * 0.0002f;
                    
                    #pragma omp atomic
                    data[idx] += depositionAmount;
                }
            }
        }
    }
    
    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("Wind erosion completed (shadow zone + saltation model)");
}

void TerrainManager::fluvialErosion(TerrainObject* terrain, const HydraulicErosionParams& p, const std::vector<float>& mask) {
    if (!terrain) return;
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    auto& height = terrain->heightmap.data;
    float cellSize = terrain->heightmap.scale_xz / w;
    float maxTerrainHeight = terrain->heightmap.scale_y;
    bool hasHardness = !terrain->hardnessMap.empty();
    bool hasMask = !mask.empty() && mask.size() == w * h;
    
    SCENE_LOG_INFO("Fluvial Erosion: Starting...");
    
    // ========================================================
    // STEP 1: PIT FILLING (Depression Breaching)
    // ========================================================
    // This ensures water can flow from any point to the edge
    // by raising pits to their "spill level"
    SCENE_LOG_INFO("Fluvial: Filling pits...");
    
    std::vector<float> filledHeight = height;  // Work on a copy
    const float epsilon = 0.0001f;
    
    // Initialize boundary cells (edges always drain)
    std::vector<bool> processed(w * h, false);
    std::priority_queue<std::pair<float, int>, 
                       std::vector<std::pair<float, int>>,
                       std::greater<std::pair<float, int>>> pq;
    
    // Add all boundary cells to priority queue
    for (int x = 0; x < w; x++) {
        pq.push({filledHeight[x], x});  // Top edge
        pq.push({filledHeight[(h-1)*w + x], (h-1)*w + x});  // Bottom edge
        processed[x] = true;
        processed[(h-1)*w + x] = true;
    }
    for (int y = 1; y < h-1; y++) {
        pq.push({filledHeight[y*w], y*w});  // Left edge
        pq.push({filledHeight[y*w + w-1], y*w + w-1});  // Right edge
        processed[y*w] = true;
        processed[y*w + w-1] = true;
    }
    
    // Process cells from lowest to highest (priority flood algorithm)
    int dx8[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int dy8[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    
    while (!pq.empty()) {
        auto [cellHeight, idx] = pq.top();
        pq.pop();
        
        int x = idx % w;
        int y = idx / w;
        
        // Check all 8 neighbors
        for (int d = 0; d < 8; d++) {
            int nx = x + dx8[d];
            int ny = y + dy8[d];
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            
            int nIdx = ny * w + nx;
            if (processed[nIdx]) continue;
            
            // Raise neighbor if it's lower than current cell (pit)
            float spillHeight = fmaxf(filledHeight[nIdx], cellHeight + epsilon);
            filledHeight[nIdx] = spillHeight;
            processed[nIdx] = true;
            pq.push({spillHeight, nIdx});
        }
    }
    
    SCENE_LOG_INFO("Fluvial: Pit filling complete");
    
    // ========================================================
    // STEP 2: FLOW DIRECTION & ACCUMULATION
    // ========================================================
    SCENE_LOG_INFO("Fluvial: Calculating flow accumulation...");
    
    std::vector<float> flowAccum(w * h, 1.0f);  // Each cell starts with 1 unit of rain
    std::vector<int> flowDir(w * h, -1);  // Direction to steepest neighbor
    std::vector<int> indices(w * h);
    
    // Initialize indices and calculate flow direction
    for (int i = 0; i < w * h; i++) {
        indices[i] = i;
        
        int x = i % w;
        int y = i / w;
        
        float maxSlope = 0.0f;
        int steepestIdx = -1;
        
        for (int d = 0; d < 8; d++) {
            int nx = x + dx8[d];
            int ny = y + dy8[d];
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            
            int nIdx = ny * w + nx;
            float dist = (abs(dx8[d]) + abs(dy8[d]) == 2) ? 1.414f : 1.0f;
            float slope = (filledHeight[i] - filledHeight[nIdx]) / (dist * cellSize);
            
            if (slope > maxSlope) {
                maxSlope = slope;
                steepestIdx = nIdx;
            }
        }
        flowDir[i] = steepestIdx;
    }
    
    // Sort by height (descending) for flow accumulation
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return filledHeight[a] > filledHeight[b];
    });
    
    // Accumulate flow from high to low
    for (int i : indices) {
        int target = flowDir[i];
        if (target != -1 && target >= 0 && target < w * h) {
            flowAccum[target] += flowAccum[i];
        }
    }
    
    SCENE_LOG_INFO("Fluvial: Flow accumulation complete");
    
    // ========================================================
    // STEP 3: STREAM POWER EROSION (Multiple passes for deeper channels)
    // ========================================================
    SCENE_LOG_INFO("Fluvial: Eroding river channels...");
    
    // Erosion coefficient - tuned for realistic results
    // UI value of 1.0 gives moderate erosion, 0.1 gives gentle erosion
    float Ks = p.erodeSpeed * 0.01f;  // Reduced from 0.05 for gentler default
    
    // Create erosion delta buffer
    std::vector<float> erosionAmount(w * h, 0.0f);
    
    // Multiple passes for deeper channels
    int numPasses = 3;
    
    for (int pass = 0; pass < numPasses; pass++) {
        std::fill(erosionAmount.begin(), erosionAmount.end(), 0.0f);
        
        // Calculate erosion based on stream power law with hardness-dependent width
        #pragma omp parallel for
        for (int i = 0; i < w * h; i++) {
            int x = i % w;
            int y = i / w;
            
            if (x <= 2 || x >= w-3 || y <= 2 || y >= h-3) continue;
            
            float flow = flowAccum[i];
            
            // Lower threshold - even small streams should erode
            if (flow < 3.0f) continue;
            
            // Calculate local slope
            float slopeX = (height[i+1] - height[i-1]) / (2.0f * cellSize);
            float slopeY = (height[i+w] - height[i-w]) / (2.0f * cellSize);
            float slope = sqrtf(slopeX*slopeX + slopeY*slopeY);
            
            // Add minimum slope to ensure erosion even on flat areas
            slope = fmaxf(slope, 0.01f);
            
            // Stream Power Law: E = K * sqrt(A) * S
            float streamPower = Ks * sqrtf(flow) * slope;
            
            // ========================================
            // HARDNESS-DEPENDENT EROSION WIDTH
            // ========================================
            // Soft rock (low hardness) = wide valley
            // Hard rock (high hardness) = narrow canyon
            float hardness = hasHardness ? terrain->hardnessMap[i] : 0.3f;
            
            // Channel width: 1-3 cells based on hardness
            // Hardness 0.0 (soft) -> radius 2-3 (wide valley)
            // Hardness 1.0 (hard) -> radius 0-1 (narrow canyon)
            int channelRadius = (int)(2.5f * (1.0f - hardness));
            channelRadius = std::clamp(channelRadius, 0, 2);
            
            // Erosion multiplier: soft erodes faster
            float hardnessMultiplier = 1.0f - hardness * 0.7f;
            streamPower *= hardnessMultiplier;
            
            if (hasMask) {
                streamPower *= mask[i];
            }
            
            // Apply erosion to channel area (wider for soft, narrow for hard)
            for (int dy = -channelRadius; dy <= channelRadius; dy++) {
                for (int dx = -channelRadius; dx <= channelRadius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    
                    int nIdx = ny * w + nx;
                    float dist = sqrtf((float)(dx*dx + dy*dy));
                    
                    // Falloff from center
                    float falloff = 1.0f - (dist / (channelRadius + 1.0f));
                    falloff = fmaxf(falloff, 0.0f);
                    falloff *= falloff; // Quadratic falloff
                    
                    float erode = streamPower * falloff;
                    erode = fminf(erode, height[nIdx] * 0.1f);
                    erode = fminf(erode, 0.5f);
                    
                    #pragma omp atomic
                    erosionAmount[nIdx] += erode;
                }
            }
            
            // ========================================
            // DELTA FORMATION (where rivers meet flat areas)
            // ========================================
            // If slope is very low and flow is high, spread sediment
            if (slope < 0.02f && flow > 20.0f) {
                // This is a potential delta area - reduce erosion, increase deposition
                float deltaSpread = 0.3f * (1.0f - slope / 0.02f);
                erosionAmount[i] *= (1.0f - deltaSpread);
            }
        }
        
        // Apply erosion directly
        #pragma omp parallel for
        for (int i = 0; i < w * h; i++) {
            height[i] -= erosionAmount[i];
            if (height[i] < 0.0f) height[i] = 0.0f;
        }
    }
    
    // ========================================================
    // STEP 4: BANK COLLAPSE (Thermal-like smoothing for channel widening)
    // ========================================================
    SCENE_LOG_INFO("Fluvial: Channel widening...");
    
    // Smooth the channel edges to create natural banks
    std::vector<float> smoothed = height;
    #pragma omp parallel for
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            
            // Only smooth where there's significant flow (near channels)
            if (flowAccum[idx] > 2.0f) {
                float hC = height[idx];
                float hL = height[idx-1];
                float hR = height[idx+1];
                float hU = height[idx-w];
                float hD = height[idx+w];
                
                // Average with neighbors if they're lower (bank collapse)
                float avgLower = 0.0f;
                int lowerCount = 0;
                if (hL < hC) { avgLower += hL; lowerCount++; }
                if (hR < hC) { avgLower += hR; lowerCount++; }
                if (hU < hC) { avgLower += hU; lowerCount++; }
                if (hD < hC) { avgLower += hD; lowerCount++; }
                
                if (lowerCount > 0) {
                    avgLower /= lowerCount;
                    // Blend towards lower neighbors (widen channel)
                    smoothed[idx] = hC * 0.7f + avgLower * 0.3f;
                }
            }
        }
    }
    height = smoothed;
    
    // ========================================================
    // STEP 5: EDGE FADE-OUT (Prevent "wall" effect at boundaries)
    // ========================================================
    SCENE_LOG_INFO("Fluvial: Edge fade-out...");
    
    int edgeFadeWidth = std::max(5, w / 50);  // ~2% of terrain width, minimum 5 cells
    
    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            
            // Calculate distance from nearest edge
            int distFromEdge = std::min({x, y, w - 1 - x, h - 1 - y});
            
            if (distFromEdge < edgeFadeWidth) {
                // Smooth falloff: 0 at edge, 1 at edgeFadeWidth
                float t = (float)distFromEdge / (float)edgeFadeWidth;
                float smoothT = t * t * (3.0f - 2.0f * t);  // Smoothstep
                
                // Blend towards lower height at edges
                // Use average of neighbor heights as reference
                float avgNeighbor = 0.0f;
                int neighborCount = 0;
                
                // Look inward for reference height
                int inwardX = std::clamp(x + (x < w/2 ? edgeFadeWidth : -edgeFadeWidth), 0, w-1);
                int inwardY = std::clamp(y + (y < h/2 ? edgeFadeWidth : -edgeFadeWidth), 0, h-1);
                float inwardHeight = height[inwardY * w + inwardX];
                
                // Fade edge cells towards a fraction of the inward height
                // Use stronger blending to eliminate walls
                height[idx] = height[idx] * smoothT + inwardHeight * (1.0f - smoothT);
            }
        }
    }
    
    // ========================================================
    // STEP 6: AGGRESSIVE THERMAL SMOOTHING (for edge blending)
    // ========================================================
    // More iterations to smooth out any remaining sharp edges
    SCENE_LOG_INFO("Fluvial: Thermal smoothing...");
    ThermalErosionParams tp;
    tp.iterations = 15;  // Increased from 5 for better edge smoothing
    tp.talusAngle = 0.4f;  // Lower talus = more smoothing on slopes
    tp.erosionAmount = 0.4f;
    this->thermalErosion(terrain, tp);  // This also calls updateTerrainMesh
    
    SCENE_LOG_INFO("Fluvial Erosion Complete!");
    // Note: thermalErosion already updates mesh, no need to call again
}

void TerrainManager::exportHeightmap(TerrainObject* terrain, const std::string& filepath) {
    if (!terrain) return;
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    
    std::vector<uint16_t> data16(w * h);
    float minH = *std::min_element(terrain->heightmap.data.begin(), terrain->heightmap.data.end());
    float maxH_data = *std::max_element(terrain->heightmap.data.begin(), terrain->heightmap.data.end());
    
    // Safety clamp to prevent infinite values from export
    if (minH < -1000.0f || maxH_data > 10000.0f) {
         SCENE_LOG_WARN("Export: Terrain contains extreme values! Clamping.");
         for(auto& val : terrain->heightmap.data) val = std::clamp(val, 0.0f, terrain->heightmap.scale_y);
         minH = 0.0f; 
         maxH_data = terrain->heightmap.scale_y;
    }

    float range = maxH_data - minH;
    if (range < 0.001f) range = 1.0f;
    
    for (int i = 0; i < w * h; ++i) {
        float normalized = (terrain->heightmap.data[i] - minH) / range;
        data16[i] = static_cast<uint16_t>(std::clamp(normalized, 0.0f, 1.0f) * 65535.0f);
    }
    
    FILE* f = fopen(filepath.c_str(), "wb");
    if (f) {
        fwrite(data16.data(), sizeof(uint16_t), w * h, f);
        fclose(f);
        SCENE_LOG_INFO("Heightmap exported: " + filepath);
    }
}

void TerrainManager::importMaskChannel(TerrainObject* terrain, const std::string& filepath, int channel) {
    if (!terrain || !terrain->splatMap) return;
    
    int imgW, imgH, channels;
    unsigned char* img = stbi_load(filepath.c_str(), &imgW, &imgH, &channels, 4);
    if (!img) return;
    
    int splatW = terrain->splatMap->width;
    int splatH = terrain->splatMap->height;
    
    for (int y = 0; y < splatH; y++) {
        for (int x = 0; x < splatW; x++) {
            int srcX = x * imgW / splatW;
            int srcY = y * imgH / splatH;
            uint8_t value = img[(srcY * imgW + srcX) * 4];
            
            int py_storage = (splatH - 1) - y; // Flip: Image Y=0 (top) -> Storage Y=H-1 (far/v=1)
            auto& p = terrain->splatMap->pixels[py_storage * splatW + x];
            if (channel == 0) p.r = value;
            else if (channel == 1) p.g = value;
            else if (channel == 2) p.b = value;
            else if (channel == 3) p.a = value;
        }
    }
    
    stbi_image_free(img);
    updateSplatMapTexture(terrain);
    SCENE_LOG_INFO("Mask channel " + std::to_string(channel) + " imported");
}

// ===========================================================================
// HARDNESS SYSTEM
// ===========================================================================

void TerrainManager::initHardnessMap(TerrainObject* terrain, float defaultHardness) {
    if (!terrain) return;
    
    int size = terrain->heightmap.width * terrain->heightmap.height;
    terrain->hardnessMap.resize(size, defaultHardness);
    
    SCENE_LOG_INFO("Hardness map initialized: " + std::to_string(size) + " pixels, default=" + std::to_string(defaultHardness));
}

void TerrainManager::autoGenerateHardness(TerrainObject* terrain, float slopeWeight, float noiseAmount) {
    if (!terrain) return;
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    
    // Make sure hardness map exists
    if (terrain->hardnessMap.size() != (size_t)(w * h)) {
        terrain->hardnessMap.resize(w * h, 0.3f);
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise(-noiseAmount, noiseAmount);
    
    float maxH = terrain->heightmap.scale_y;
    
    // Find min/max height for normalization
    float minHeight = *std::min_element(terrain->heightmap.data.begin(), terrain->heightmap.data.end());
    float maxHeight = *std::max_element(terrain->heightmap.data.begin(), terrain->heightmap.data.end());
    float heightRange = maxHeight - minHeight;
    if (heightRange < 0.001f) heightRange = 1.0f;
    
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            
            // Get normalized height (0-1)
            float normalizedHeight = (terrain->heightmap.data[idx] - minHeight) / heightRange;
            
            // Calculate slope from height differences
            float hl = terrain->heightmap.data[idx - 1];
            float hr = terrain->heightmap.data[idx + 1];
            float hu = terrain->heightmap.data[idx - w];
            float hd = terrain->heightmap.data[idx + w];
            
            float dX = fabsf(hr - hl) * maxH;
            float dZ = fabsf(hd - hu) * maxH;
            float slope = sqrtf(dX * dX + dZ * dZ);
            
            // Normalize slope (0-1 range, clamped)
            slope = std::min(slope * 2.0f, 1.0f);
            
            // ============================================
            // GEOLOGICAL LAYERING
            // ============================================
            // Low areas (valleys, plains) = soft sediment/soil
            // Mid areas = mixed rock/soil
            // High areas (mountains) = hard bedrock
            // Steep slopes anywhere = exposed hard rock
            
            float heightHardness = 0.0f;
            
            if (normalizedHeight < 0.25f) {
                // Low areas: soft sediment (alluvial deposits, soil)
                heightHardness = 0.1f + normalizedHeight * 0.4f;  // 0.1 - 0.2
            }
            else if (normalizedHeight < 0.5f) {
                // Lower-mid: mixed soft rock
                heightHardness = 0.2f + (normalizedHeight - 0.25f) * 0.8f;  // 0.2 - 0.4
            }
            else if (normalizedHeight < 0.75f) {
                // Upper-mid: harder rock
                heightHardness = 0.4f + (normalizedHeight - 0.5f) * 1.2f;  // 0.4 - 0.7
            }
            else {
                // High areas: very hard bedrock
                heightHardness = 0.7f + (normalizedHeight - 0.75f) * 1.2f;  // 0.7 - 1.0
            }
            
            // Steep slopes = exposed hard rock
            float slopeHardness = slope * slopeWeight;
            
            // Combine: take the harder of the two (rock exposed by slope or by elevation)
            float hardness = fmaxf(heightHardness, slopeHardness);
            
            // Add some noise for variation
            hardness += noise(gen);
            
            // Clamp to valid range
            terrain->hardnessMap[idx] = std::clamp(hardness, 0.0f, 1.0f);
        }
    }
    
    SCENE_LOG_INFO("Hardness map auto-generated (Geological Layering + Slope)");
}

void TerrainManager::paintHardness(TerrainObject* terrain, const Vec3& hitPoint, float radius, float strength, float dt, bool increase) {
    if (!terrain) return;
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    
    // Ensure hardness map exists
    if (terrain->hardnessMap.size() != w * h) {
        terrain->hardnessMap.resize(w * h, 0.3f);
    }
    
    // Transform to local space
    Vec3 localPos;
    if (terrain->transform) {
        Matrix4x4 inv = terrain->transform->getFinal().inverse();
        localPos = inv.multiplyVector(Vec4(hitPoint, 1.0f)).xyz();
    } else {
        localPos = hitPoint;
    }
    
    float size = terrain->heightmap.scale_xz;
    float gridX = (localPos.x / size) * (w - 1);
    float gridZ = (localPos.z / size) * (h - 1);
    
    int cx = (int)(gridX + 0.5f);
    int cy = (int)(gridZ + 0.5f);
    
    int r_pixels = (int)((radius / size) * (w - 1));
    if (r_pixels < 1) r_pixels = 1;
    
    float brush_strength = strength * dt * 3.0f;
    
    for (int y = cy - r_pixels; y <= cy + r_pixels; y++) {
        for (int x = cx - r_pixels; x <= cx + r_pixels; x++) {
            if (x < 0 || x >= w || y < 0 || y >= h) continue;
            
            float dx = (float)(x - gridX);
            float dy = (float)(y - gridZ);
            float dist = sqrtf(dx * dx + dy * dy);
            
            if (dist > r_pixels) continue;
            
            // Gaussian-like falloff
            float falloff = 1.0f - (dist / r_pixels);
            falloff = falloff * falloff;
            
            int idx = y * w + x;
            float val = terrain->hardnessMap[idx];
            
            if (increase) {
                val += brush_strength * falloff;
            } else {
                val -= brush_strength * falloff;
            }
            
            terrain->hardnessMap[idx] = std::clamp(val, 0.0f, 1.0f);
        }
    }
}

// ===========================================================================
// COMBINED WIZARD PROCESS
// ===========================================================================

void TerrainManager::applyCombinedErosion(TerrainObject* terrain, int iterations, float strength, bool use_gpu) {
    if (!terrain) return;
    
    SCENE_LOG_INFO(std::string("Starting Combined Wizard Process (GPU: ") + (use_gpu ? "ON" : "OFF") + ")...");
    
    // 1. HARDNESS CHECK - Ensure valid map
    if (terrain->hardnessMap.empty() || terrain->hardnessMap.size() != terrain->heightmap.data.size()) {
        autoGenerateHardness(terrain, 0.7f, 0.2f);
    }
    
    // 2. PARAMS SETUP - Balanced for natural terrain (spike-free)
    HydraulicErosionParams hydro;
    hydro.iterations = iterations * 2000;
    hydro.erodeSpeed = 0.4f * strength;
    hydro.depositSpeed = 0.2f;
    hydro.evaporateSpeed = 0.02f;
    hydro.sedimentCapacity = 8.0f;
    hydro.erosionRadius = 2;
    
    HydraulicErosionParams fluvialParams;
    fluvialParams.erodeSpeed = 1.5f * strength;
    fluvialParams.depositSpeed = 0.1f;
    fluvialParams.iterations = 2000;
    
    ThermalErosionParams thermal;
    thermal.iterations = 15;
    thermal.talusAngle = 0.5f;
    thermal.erosionAmount = 0.4f;
    
    // 3. WIZARD LOOP
    
    // A. Initial Thermal
    SCENE_LOG_INFO("Wizard: Phase 1 - Initial Smoothing");
    if (use_gpu) thermalErosionGPU(terrain, thermal);
    else thermalErosion(terrain, thermal);
    
    // B. Hydraulic
    SCENE_LOG_INFO("Wizard: Phase 2 - Surface Hydraulic");
    if (use_gpu) hydraulicErosionGPU(terrain, hydro);
    else hydraulicErosion(terrain, hydro);
    
    // C. Fluvial x2
    SCENE_LOG_INFO("Wizard: Phase 3 - Fluvial Channels");
    for (int i = 0; i < 2; i++) {
        if (use_gpu) fluvialErosionGPU(terrain, fluvialParams);
        else fluvialErosion(terrain, fluvialParams);
    }
    
    // D. Wind
    SCENE_LOG_INFO("Wizard: Phase 4 - Wind Aging");
    if (use_gpu) windErosionGPU(terrain, 0.5f * strength, 45.0f, 10);
    else windErosion(terrain, 0.5f * strength, 45.0f, 10);
    
    // E. Final Thermal
    SCENE_LOG_INFO("Wizard: Phase 5 - Final Polish");
    thermal.iterations = 10;
    thermal.erosionAmount = 0.3f;
    if (use_gpu) thermalErosionGPU(terrain, thermal);
    else thermalErosion(terrain, thermal);
    
    SCENE_LOG_INFO("Wizard Process Completed!");
}

void TerrainManager::applyCombinedErosionWithProgress(TerrainObject* terrain, int iterations, float strength, ProgressCallback callback) {
    if (!terrain) return;
    
    if (callback) callback(0.0f, "Initializing...");
    
    // 1. HARDNESS CHECK
    if (terrain->hardnessMap.empty() || terrain->hardnessMap.size() != terrain->heightmap.data.size()) {
        if (callback) callback(0.05f, "Generating hardness map...");
        autoGenerateHardness(terrain, 0.7f, 0.2f);
    }
    
    // 2. PARAMS SETUP - Balanced (spike-free)
    HydraulicErosionParams hydro;
    hydro.iterations = iterations * 2000;
    hydro.erodeSpeed = 0.4f * strength;
    hydro.depositSpeed = 0.2f;
    hydro.evaporateSpeed = 0.02f;
    hydro.sedimentCapacity = 8.0f;
    hydro.erosionRadius = 2;
    
    HydraulicErosionParams fluvialParams;
    fluvialParams.erodeSpeed = 1.5f * strength;
    fluvialParams.depositSpeed = 0.1f;
    
    ThermalErosionParams thermal;
    thermal.iterations = 15;
    thermal.talusAngle = 0.5f;
    thermal.erosionAmount = 0.4f;
    
    // 3. WIZARD LOOP with progress
    if (callback) callback(0.1f, "Phase 1: Initial Smoothing...");
    thermalErosion(terrain, thermal);
    
    if (callback) callback(0.25f, "Phase 2: Surface Hydraulic...");
    hydraulicErosion(terrain, hydro);
    
    if (callback) callback(0.45f, "Phase 3: Fluvial Channels (1/2)...");
    fluvialErosion(terrain, fluvialParams);
    
    if (callback) callback(0.60f, "Phase 3: Fluvial Channels (2/2)...");
    fluvialErosion(terrain, fluvialParams);
    
    if (callback) callback(0.80f, "Phase 4: Wind Aging...");
    windErosion(terrain, 0.5f * strength, 45.0f, 10);
    
    if (callback) callback(0.90f, "Phase 5: Final Polish...");
    thermal.iterations = 10;
    thermal.erosionAmount = 0.3f;
    thermalErosion(terrain, thermal);
    
    if (callback) callback(1.0f, "Completed!");
    SCENE_LOG_INFO("Wizard Process with Progress Completed!");
}

// ===========================================================================
// KEYFRAME ANIMATION INTEGRATION
// ===========================================================================

void TerrainManager::captureKeyframeToTrack(TerrainObject* terrain, ObjectAnimationTrack& track, int frame) {
    if (!terrain) return;
    
    // Create new Keyframe
    Keyframe kf(frame);
    
    // Populate Terrain Data (height and hardness only)
    // NOTE: Splat map is NOT saved because it's auto-generated from slope/altitude data
    kf.terrain.width = terrain->heightmap.width;
    kf.terrain.height = terrain->heightmap.height;
    kf.terrain.heightData = terrain->heightmap.data;
    kf.terrain.hardnessData = terrain->hardnessMap;
    kf.terrain.has_data = true;
    
    kf.has_terrain = true;
    
    // Add to track (merges if exists)
    track.addKeyframe(kf);
    
    SCENE_LOG_INFO("Captured Terrain Keyframe at Frame " + std::to_string(frame));
}

void TerrainManager::applyKeyframe(TerrainObject* terrain, const TerrainKeyframe& kf) {
    if (!terrain || !kf.has_data) return;
    
    // Check dimensions - if mismatch, resize terrain buffers
    if (terrain->heightmap.data.size() != kf.heightData.size()) {
         terrain->heightmap.width = kf.width;
         terrain->heightmap.height = kf.height;
         terrain->heightmap.data.resize(kf.heightData.size());
         terrain->hardnessMap.resize(kf.heightData.size());
    }
    
    // Copy Height Data
    terrain->heightmap.data = kf.heightData;
    
    // Copy Hardness Data
    if (!kf.hardnessData.empty()) {
        if (terrain->hardnessMap.size() != kf.hardnessData.size()) {
            terrain->hardnessMap.resize(kf.hardnessData.size());
        }
        terrain->hardnessMap = kf.hardnessData;
    }
    
    // NOTE: Splat map is NOT restored from keyframe
    // It will be auto-regenerated from slope/altitude data during mesh update
    
    // Force Mesh & BVH Update
    terrain->dirty_mesh = true;
    updateTerrainMesh(terrain); // Recalculate normals etc.
}

void TerrainManager::updateFromTrack(TerrainObject* terrain, const ObjectAnimationTrack& track, int currentFrame) {
    if (!terrain) return;
    
    // Evaluate track at current frame
    // This uses KeyframeSystem's internal logic (findPrev, findNext, lerp) which we updated
    Keyframe kf = track.evaluate(currentFrame);
    
    if (kf.has_terrain) {
        applyKeyframe(terrain, kf.terrain);
    }
}

// ============================================================================
// SERIALIZATION
// ============================================================================

using json = nlohmann::json;

void TerrainManager::saveHeightmapBinary(const TerrainObject* terrain, const std::string& filepath) const {
    if (!terrain) return;
    
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        SCENE_LOG_ERROR("Failed to save heightmap binary: " + filepath);
        return;
    }
    
    // Header
    const char magic[4] = {'R', 'T', 'H', 'M'}; // RayTrophi HeightMap
    uint32_t version = 1;
    uint32_t width = terrain->heightmap.width;
    uint32_t height = terrain->heightmap.height;
    
    file.write(magic, 4);
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&width), sizeof(width));
    file.write(reinterpret_cast<const char*>(&height), sizeof(height));
    
    // Float data
    file.write(reinterpret_cast<const char*>(terrain->heightmap.data.data()), 
               terrain->heightmap.data.size() * sizeof(float));
    
    file.close();
    SCENE_LOG_INFO("Saved heightmap binary: " + filepath + " (" + std::to_string(width) + "x" + std::to_string(height) + ")");
}

void TerrainManager::loadHeightmapBinary(TerrainObject* terrain, const std::string& filepath) {
    if (!terrain) return;
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        SCENE_LOG_ERROR("Failed to load heightmap binary: " + filepath);
        return;
    }
    
    // Read header
    char magic[4];
    uint32_t version, width, height;
    
    file.read(magic, 4);
    if (magic[0] != 'R' || magic[1] != 'T' || magic[2] != 'H' || magic[3] != 'M') {
        SCENE_LOG_ERROR("Invalid heightmap binary format: " + filepath);
        return;
    }
    
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&width), sizeof(width));
    file.read(reinterpret_cast<char*>(&height), sizeof(height));
    
    // Read float data
    terrain->heightmap.width = width;
    terrain->heightmap.height = height;
    terrain->heightmap.data.resize(width * height);
    
    file.read(reinterpret_cast<char*>(terrain->heightmap.data.data()), 
              terrain->heightmap.data.size() * sizeof(float));
    
    file.close();
    SCENE_LOG_INFO("Loaded heightmap binary: " + filepath + " (" + std::to_string(width) + "x" + std::to_string(height) + ")");
}

json TerrainManager::serialize(const std::string& terrainDir) const {
    json root;
    root["version"] = TERRAIN_SERIALIZATION_VERSION;
    root["terrain_count"] = terrains.size();
    
    json terrainsArray = json::array();
    
    for (size_t i = 0; i < terrains.size(); ++i) {
        const auto& t = terrains[i];
        json tJson;
        
        tJson["id"] = t.id;
        tJson["name"] = t.name;
        tJson["material_id"] = t.material_id;
        
        // Heightmap metadata
        tJson["heightmap"] = {
            {"width", t.heightmap.width},
            {"height", t.heightmap.height},
            {"scale_xz", t.heightmap.scale_xz},
            {"scale_y", t.heightmap.scale_y},
            {"file", "heightmap_" + std::to_string(t.id) + ".raw"}
        };
        
        // Save heightmap binary
        std::string hmPath = terrainDir + "/heightmap_" + std::to_string(t.id) + ".raw";
        saveHeightmapBinary(&t, hmPath);
        
        // Splat map
        if (t.splatMap && t.splatMap->m_is_loaded) {
            std::string splatPath = terrainDir + "/splatmap_" + std::to_string(t.id) + ".png";
            
            // Export splat map as PNG (Flipped Y so Near is at the bottom)
            std::vector<uint8_t> pixels;
            pixels.resize(t.splatMap->width * t.splatMap->height * 4);
            int tw = t.splatMap->width;
            int th = t.splatMap->height;
            for (int y = 0; y < th; ++y) {
                for (int x = 0; x < tw; ++x) {
                    int src_idx = y * tw + x;
                    int dst_idx = ((th - 1) - y) * tw + x; // Flip Y for PNG
                    const auto& p = t.splatMap->pixels[src_idx];
                    pixels[dst_idx * 4 + 0] = p.r;
                    pixels[dst_idx * 4 + 1] = p.g;
                    pixels[dst_idx * 4 + 2] = p.b;
                    pixels[dst_idx * 4 + 3] = p.a;
                }
            }
            stbi_write_png(splatPath.c_str(), tw, th, 4, pixels.data(), tw * 4);
            
            tJson["splatmap_file"] = "splatmap_" + std::to_string(t.id) + ".png";
            tJson["splatmap_width"] = tw;
            tJson["splatmap_height"] = th;
        }
        
        // Layers (material IDs)
        json layersJson = json::array();
        for (size_t li = 0; li < t.layers.size(); ++li) {
            json layerJson;
            if (t.layers[li]) {
                // Find material ID
                uint16_t matId = MaterialManager::getInstance().getMaterialID(t.layers[li]->materialName);
                layerJson["material_id"] = matId;
                layerJson["material_name"] = t.layers[li]->materialName;
            } else {
                layerJson["material_id"] = -1;
            }
            
            if (li < t.layer_uv_scales.size()) {
                layerJson["uv_scale"] = t.layer_uv_scales[li];
            }
            
            layersJson.push_back(layerJson);
        }
        tJson["layers"] = layersJson;
        
        // Hardness map
        if (!t.hardnessMap.empty()) {
            std::string hardnessPath = terrainDir + "/hardness_" + std::to_string(t.id) + ".raw";
            
            std::ofstream hFile(hardnessPath, std::ios::binary);
            if (hFile.is_open()) {
                uint32_t size = static_cast<uint32_t>(t.hardnessMap.size());
                hFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
                hFile.write(reinterpret_cast<const char*>(t.hardnessMap.data()), size * sizeof(float));
                hFile.close();
                
                tJson["hardness_file"] = "hardness_" + std::to_string(t.id) + ".raw";
            }
        }
        
        // Transform
        if (t.transform) {
            tJson["transform"] = {
                {"position", {t.transform->position.x, t.transform->position.y, t.transform->position.z}},
                {"rotation", {t.transform->rotation.x, t.transform->rotation.y, t.transform->rotation.z}},
                {"scale", {t.transform->scale.x, t.transform->scale.y, t.transform->scale.z}}
            };
        }
        
        // Quality settings
        tJson["normal_quality"] = static_cast<int>(t.normal_quality);
        tJson["normal_strength"] = t.normal_strength;

        // Auto-mask settings
        tJson["am_height_min"] = t.am_height_min;
        tJson["am_height_max"] = t.am_height_max;
        tJson["am_slope"] = t.am_slope;

        // Foliage Layers (Legacy/Direct system)
        json foliageArray = json::array();
        for (const auto& fl : t.foliageLayers) {
            json flJson;
            flJson["name"] = fl.name;
            flJson["enabled"] = fl.enabled;
            flJson["meshPath"] = fl.meshPath;
            flJson["density"] = fl.density;
            flJson["targetMaskLayerId"] = fl.targetMaskLayerId;
            flJson["maskThreshold"] = fl.maskThreshold;
            flJson["scaleRange"] = {fl.scaleRange.x, fl.scaleRange.y};
            flJson["rotationRange"] = {fl.rotationRange.x, fl.rotationRange.y};
            flJson["alignToNormal"] = fl.alignToNormal;
            flJson["yOffsetRange"] = {fl.yOffsetRange.x, fl.yOffsetRange.y};
            foliageArray.push_back(flJson);
        }
        tJson["foliage_layers"] = foliageArray;
        
        terrainsArray.push_back(tJson);
    }
    
    root["terrains"] = terrainsArray;
    
    SCENE_LOG_INFO("[TerrainManager] Serialized " + std::to_string(terrains.size()) + " terrains");
    return root;
}

void TerrainManager::deserialize(const json& data, const std::string& terrainDir, SceneData& scene) {
    // Don't clear existing terrains automatically - let caller decide
    
    int version = data.value("version", 1);
    if (version > TERRAIN_SERIALIZATION_VERSION) {
        SCENE_LOG_WARN("[TerrainManager] Scene uses newer terrain format (v" + 
                       std::to_string(version) + "), some features may not load correctly");
    }
    
    if (!data.contains("terrains") || !data["terrains"].is_array()) {
        SCENE_LOG_WARN("[TerrainManager] No terrains found in scene data");
        return;
    }
    
    for (const auto& tJson : data["terrains"]) {
        TerrainObject terrain;
        
        terrain.id = tJson.value("id", next_id++);
        terrain.name = tJson.value("name", "Terrain_" + std::to_string(terrain.id));
        terrain.material_id = tJson.value("material_id", 0);
        
        // Heightmap
        if (tJson.contains("heightmap")) {
            const auto& hm = tJson["heightmap"];
            terrain.heightmap.width = hm.value("width", 256);
            terrain.heightmap.height = hm.value("height", 256);
            terrain.heightmap.scale_xz = hm.value("scale_xz", 100.0f);
            terrain.heightmap.scale_y = hm.value("scale_y", 10.0f);
            
            // Load binary heightmap
            if (hm.contains("file")) {
                std::string hmPath = terrainDir + "/" + hm["file"].get<std::string>();
                if (std::filesystem::exists(hmPath)) {
                    loadHeightmapBinary(&terrain, hmPath);
                } else {
                    // Initialize flat heightmap
                    terrain.heightmap.data.resize(terrain.heightmap.width * terrain.heightmap.height, 0.0f);
                }
            }
        }
        
        // Transform
        terrain.transform = std::make_shared<Transform>();
        if (tJson.contains("transform")) {
            const auto& tr = tJson["transform"];
            if (tr.contains("position") && tr["position"].is_array()) {
                terrain.transform->position = Vec3(tr["position"][0], tr["position"][1], tr["position"][2]);
            }
            if (tr.contains("rotation") && tr["rotation"].is_array()) {
                terrain.transform->rotation = Vec3(tr["rotation"][0], tr["rotation"][1], tr["rotation"][2]);
            }
            if (tr.contains("scale") && tr["scale"].is_array()) {
                terrain.transform->scale = Vec3(tr["scale"][0], tr["scale"][1], tr["scale"][2]);
            }
            terrain.transform->updateMatrix();
        }
        
        // Quality
        terrain.normal_quality = static_cast<NormalQuality>(tJson.value("normal_quality", 1));
        terrain.normal_strength = tJson.value("normal_strength", 1.0f);
        
        terrain.am_height_min = tJson.value("am_height_min", 5.0f);
        terrain.am_height_max = tJson.value("am_height_max", 20.0f);
        terrain.am_slope = tJson.value("am_slope", 5.0f);
        
        // Add to list
        terrains.push_back(terrain);
        TerrainObject* ptr = &terrains.back();
        
        // Update next_id
        if (terrain.id >= next_id) {
            next_id = terrain.id + 1;
        }
        
        // Initialize layers
        initLayers(ptr);
        
        // Load splat map
        if (tJson.contains("splatmap_file")) {
            std::string splatPath = terrainDir + "/" + tJson["splatmap_file"].get<std::string>();
            if (std::filesystem::exists(splatPath) && ptr->splatMap) {
                int w, h, channels;
                unsigned char* img = stbi_load(splatPath.c_str(), &w, &h, &channels, 4);
                if (img) {
                    ptr->splatMap->width = w;
                    ptr->splatMap->height = h;
                    ptr->splatMap->pixels.resize(w * h);
                    for (int y = 0; y < h; ++y) {
                        for (int x = 0; x < w; ++x) {
                            int src_idx = y * w + x;
                            int dst_idx = ((h - 1) - y) * w + x; // Flip Y: PNG Top -> Storage Far/v=1
                            ptr->splatMap->pixels[dst_idx].r = img[src_idx * 4 + 0];
                            ptr->splatMap->pixels[dst_idx].g = img[src_idx * 4 + 1];
                            ptr->splatMap->pixels[dst_idx].b = img[src_idx * 4 + 2];
                            ptr->splatMap->pixels[dst_idx].a = img[src_idx * 4 + 3];
                        }
                    }
                    stbi_image_free(img);
                    ptr->splatMap->m_is_loaded = true;
                    ptr->splatMap->upload_to_gpu();
                    
                    SCENE_LOG_INFO("[TerrainManager] Loaded splatmap: " + std::to_string(w) + "x" + std::to_string(h) + 
                                   " for heightmap: " + std::to_string(ptr->heightmap.width) + "x" + std::to_string(ptr->heightmap.height));
                }
            }
        }
        
        // Resize splatmap if dimensions don't match heightmap
        resizeSplatMap(ptr);
        
        // Load layers
        if (tJson.contains("layers") && tJson["layers"].is_array()) {
            for (size_t li = 0; li < tJson["layers"].size() && li < 4; ++li) {
                const auto& layerJson = tJson["layers"][li];
                
                // Material reference
                if (layerJson.contains("material_name")) {
                    std::string matName = layerJson["material_name"].get<std::string>();
                    auto mat = MaterialManager::getInstance().getMaterialShared(
                        MaterialManager::getInstance().getMaterialID(matName));
                    if (mat) {
                        ptr->layers[li] = mat;
                    }
                }
                
                // UV scale
                if (layerJson.contains("uv_scale") && li < ptr->layer_uv_scales.size()) {
                    ptr->layer_uv_scales[li] = layerJson["uv_scale"].get<float>();
                }
            }
        }
        
        // Load hardness map
        if (tJson.contains("hardness_file")) {
            std::string hardnessPath = terrainDir + "/" + tJson["hardness_file"].get<std::string>();
            if (std::filesystem::exists(hardnessPath)) {
                std::ifstream hFile(hardnessPath, std::ios::binary);
                if (hFile.is_open()) {
                    uint32_t size;
                    hFile.read(reinterpret_cast<char*>(&size), sizeof(size));
                    ptr->hardnessMap.resize(size);
                    hFile.read(reinterpret_cast<char*>(ptr->hardnessMap.data()), size * sizeof(float));
                    hFile.close();
                }
            }
        }

        // Load Foliage Layers (Legacy system)
        if (tJson.contains("foliage_layers") && tJson["foliage_layers"].is_array()) {
            ptr->foliageLayers.clear();
            for (const auto& flJson : tJson["foliage_layers"]) {
                TerrainFoliageLayer fl;
                fl.name = flJson.value("name", "New Foliage");
                fl.enabled = flJson.value("enabled", true);
                fl.meshPath = flJson.value("meshPath", "");
                fl.density = flJson.value("density", 1000);
                fl.targetMaskLayerId = flJson.value("targetMaskLayerId", 0);
                fl.maskThreshold = flJson.value("maskThreshold", 0.2f);
                
                if (flJson.contains("scaleRange") && flJson["scaleRange"].is_array()) {
                    fl.scaleRange = Vec2(flJson["scaleRange"][0], flJson["scaleRange"][1]);
                }
                if (flJson.contains("rotationRange") && flJson["rotationRange"].is_array()) {
                    fl.rotationRange = Vec2(flJson["rotationRange"][0], flJson["rotationRange"][1]);
                }
                
                fl.alignToNormal = flJson.value("alignToNormal", 0.0f);
                
                if (flJson.contains("yOffsetRange") && flJson["yOffsetRange"].is_array()) {
                    fl.yOffsetRange = Vec2(flJson["yOffsetRange"][0], flJson["yOffsetRange"][1]);
                }
                
                ptr->foliageLayers.push_back(fl);
            }
        }
        
        // Generate mesh
        updateTerrainMesh(ptr);
        
        // Add triangles to scene
        for (auto& tri : ptr->mesh_triangles) {
            scene.world.objects.push_back(tri);
        }
    }
    
    SCENE_LOG_INFO("[TerrainManager] Deserialized " + std::to_string(terrains.size()) + " terrains (format v" + std::to_string(version) + ")");
}

// ===========================================================================
// GPU EROSION IMPLEMENTATION
// ===========================================================================

void TerrainManager::initCuda() {
    if (cudaInitialized) return;

    SCENE_LOG_INFO("[GPU Erosion] Initializing CUDA...");

    // Initialize CUDA Driver API
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] cuInit failed: " + std::to_string(res));
        return;
    }

    // Get Device
    CUdevice device;
    res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] cuDeviceGet failed");
        return;
    }

    // Create Context (or use current)
    CUcontext ctx;
    res = cuCtxGetCurrent(&ctx);
    if (res != CUDA_SUCCESS || ctx == nullptr) {
        SCENE_LOG_INFO("[GPU Erosion] Creating new CUDA context");
        res = cuCtxCreate(&ctx, 0, device);
        if (res != CUDA_SUCCESS) {
            SCENE_LOG_ERROR("[GPU Erosion] cuCtxCreate failed");
            return;
        }
    }

    // Load PTX Module
    std::string ptxPath = "erosion_kernels.ptx";
    if (!std::filesystem::exists(ptxPath)) {
         // Try project root or build dir
         ptxPath = "../erosion_kernels.ptx"; 
    }
    
    // Check absolute path based on execution
    // Assuming run from build dir? User said Compile PTX output is E:/.../raytrac_sdl2/erosion_kernels.ptx
    // I'll try that specific path if relative fails, or just assume relative to executable.
    
    // Using simple path first
    res = cuModuleLoad((CUmodule*)&cudaModule, "erosion_kernels.ptx");
    if (res != CUDA_SUCCESS) {
        // Try reading file content manually to debug or just informative error
        SCENE_LOG_ERROR("[GPU Erosion] Failed to load module 'erosion_kernels.ptx' (Error: " + std::to_string(res) + ")");
        SCENE_LOG_WARN("Make sure to run 'compile_ptx.bat' first!");
        return;
    }

    // Get Function
    res = cuModuleGetFunction((CUfunction*)&erosionKernelFunc, (CUmodule)cudaModule, "hydraulicErosionKernel");
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] Failed to get kernel function 'hydraulicErosionKernel'");
        return;
    }

    res = cuModuleGetFunction((CUfunction*)&smoothKernelFunc, (CUmodule)cudaModule, "smoothTerrainKernel");
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] Failed to get kernel function 'smoothTerrainKernel'");
        return;
    }

    res = cuModuleGetFunction((CUfunction*)&thermalKernelFunc, (CUmodule)cudaModule, "thermalErosionKernel");
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] Failed to get kernel function 'thermalErosionKernel'");
        return;
    }

    cuModuleGetFunction((CUfunction*)&fluvRainKernelFunc, (CUmodule)cudaModule, "fluvialRainKernel");
    cuModuleGetFunction((CUfunction*)&fluvFluxKernelFunc, (CUmodule)cudaModule, "fluvialFluxKernel");
    cuModuleGetFunction((CUfunction*)&fluvWaterKernelFunc, (CUmodule)cudaModule, "fluvialWaterKernel");
    cuModuleGetFunction((CUfunction*)&fluvErodeKernelFunc, (CUmodule)cudaModule, "fluvialErosionKernel");
    cuModuleGetFunction((CUfunction*)&windKernelFunc, (CUmodule)cudaModule, "windErosionKernel");

    // Post-processing kernels (for CPU-GPU parity)
    cuModuleGetFunction((CUfunction*)&pitFillKernelFunc, (CUmodule)cudaModule, "pitFillingKernel");
    cuModuleGetFunction((CUfunction*)&spikeRemovalKernelFunc, (CUmodule)cudaModule, "spikeRemovalKernel");
    cuModuleGetFunction((CUfunction*)&edgePreservationKernelFunc, (CUmodule)cudaModule, "edgePreservationKernel");
    cuModuleGetFunction((CUfunction*)&thermalWithHardnessKernelFunc, (CUmodule)cudaModule, "thermalErosionWithHardnessKernel");

    cudaInitialized = true;
    SCENE_LOG_INFO("[GPU Erosion] CUDA Initialized Successfully (with post-processing kernels)");
}

void TerrainManager::hydraulicErosionGPU(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask) {
    if (!terrain) return;
    
    // Lazy Init
    if (!cudaInitialized) {
        initCuda();
        if (!cudaInitialized) {
            SCENE_LOG_ERROR("[GPU Erosion] CUDA not initialized, falling back to CPU or aborting.");
            // Fallback?
            // hydraulicErosion(terrain, params); 
            return;
        }
    }
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    size_t mapSize = w * h * sizeof(float);
    
    // Allocate Device Memory
    CUdeviceptr d_heightmap;
    CUresult res = cuMemAlloc(&d_heightmap, mapSize);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] Memory Allocation Failed");
        return;
    }
    
    // Copy Host to Device
    res = cuMemcpyHtoD(d_heightmap, terrain->heightmap.data.data(), mapSize);
    if (res != CUDA_SUCCESS) {
        cuMemFree(d_heightmap);
        SCENE_LOG_ERROR("[GPU Erosion] HtoD Copy Failed");
        return;
    }
    
    // Prepare Params
    TerrainPhysics::HydraulicErosionParamsGPU gpuParams;
    gpuParams.mapWidth = w;
    gpuParams.mapHeight = h;
    gpuParams.brushRadius = params.erosionRadius;
    gpuParams.dropletLifetime = params.dropletLifetime;
    gpuParams.inertia = params.inertia;
    gpuParams.sedimentCapacity = params.sedimentCapacity;
    gpuParams.minSlope = params.minSlope;
    gpuParams.erodeSpeed = params.erodeSpeed;
    gpuParams.depositSpeed = params.depositSpeed;
    gpuParams.evaporateSpeed = params.evaporateSpeed;
    gpuParams.gravity = params.gravity;
    gpuParams.seed = rand();
    
    unsigned long long seed_offset = 0; // Could accumulate this
    
    // Kernel Args
    void* args[] = { &d_heightmap, &gpuParams, &seed_offset };
    
    // Launch Config
    int blockSize = 128; // 256
    int numDroplets = params.iterations; // 50,000 to 1,000,000
    int numBlocks = (numDroplets + blockSize - 1) / blockSize;
    
    SCENE_LOG_INFO("[GPU Erosion] Launching kernel with " + std::to_string(numDroplets) + " droplets...");
    
    // Launch
    res = cuLaunchKernel((CUfunction)erosionKernelFunc,
                         numBlocks, 1, 1,    // Grid Dim
                         blockSize, 1, 1,    // Block Dim
                         0, nullptr,         // Shared Mem, Stream
                         args, nullptr);
                         
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] Launch Failed: " + std::to_string(res));
        cuMemFree(d_heightmap);
        return;
    }
    
    // Launch Smoothing Pass
    // Grid dim for 2D heightmap
    int tx = 16, ty = 16;
    int bx = (w + tx - 1) / tx;
    int by = (h + ty - 1) / ty;
    
    SCENE_LOG_INFO("[GPU Erosion] Running post-processing (CPU-GPU parity)...");
    
    // Calculate cellSize for threshold computation
    float cellSize = terrain->heightmap.scale_xz / w;
    
    // Prepare PostProcessParams
    TerrainPhysics::PostProcessParamsGPU postParams;
    postParams.mapWidth = w;
    postParams.mapHeight = h;
    postParams.cellSize = cellSize;
    postParams.pitThreshold = cellSize * 0.05f;     // Match CPU
    postParams.spikeThreshold = cellSize * 0.1f;    // Match CPU
    postParams.edgeFadeWidth = std::max(3, w / 40); // Match CPU
    
    // 1. SPIKE REMOVAL PASS (matches CPU post-erosion spike removal)
    if (spikeRemovalKernelFunc) {
        void* spikeArgs[] = { &d_heightmap, &postParams };
        res = cuLaunchKernel((CUfunction)spikeRemovalKernelFunc,
                             bx, by, 1, tx, ty, 1, 0, nullptr, spikeArgs, nullptr);
        if (res != CUDA_SUCCESS) {
            SCENE_LOG_WARN("[GPU Erosion] Spike Removal Failed: " + std::to_string(res));
        }
    }
    
    // 2. PIT FILLING PASS (matches CPU post-erosion pit filling)
    if (pitFillKernelFunc) {
        void* pitArgs[] = { &d_heightmap, &postParams };
        res = cuLaunchKernel((CUfunction)pitFillKernelFunc,
                             bx, by, 1, tx, ty, 1, 0, nullptr, pitArgs, nullptr);
        if (res != CUDA_SUCCESS) {
            SCENE_LOG_WARN("[GPU Erosion] Pit Filling Failed: " + std::to_string(res));
        }
    }
    
    // 3. EDGE PRESERVATION PASS (matches CPU edge fade-out)
    // Note: For edge preservation, we need original heights which we don't have on GPU
    // So we use the simplified version that blends with inward reference
    if (edgePreservationKernelFunc) {
        CUdeviceptr d_original = 0; // nullptr - we don't have original heights saved
        void* edgeArgs[] = { &d_heightmap, &d_original, &postParams };
        res = cuLaunchKernel((CUfunction)edgePreservationKernelFunc,
                             bx, by, 1, tx, ty, 1, 0, nullptr, edgeArgs, nullptr);
        if (res != CUDA_SUCCESS) {
            SCENE_LOG_WARN("[GPU Erosion] Edge Preservation Failed: " + std::to_string(res));
        }
    }
    
    // 4. FINAL SMOOTHING PASS (existing behavior)
    void* smoothArgs[] = { &d_heightmap, &w, &h };
    res = cuLaunchKernel((CUfunction)smoothKernelFunc,
                         bx, by, 1, tx, ty, 1, 0, nullptr, smoothArgs, nullptr);
    if (res != CUDA_SUCCESS) {
         SCENE_LOG_WARN("[GPU Erosion] Smooth Kernel Failed: " + std::to_string(res));
    }
    
    // Synchronize (Wait for completion)
    cuCtxSynchronize();
    
    // Copy Device to Host
    res = cuMemcpyDtoH(terrain->heightmap.data.data(), d_heightmap, mapSize);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] DtoH Copy Failed");
    }
    
    // Cleanup
    cuMemFree(d_heightmap);
    
    // Update Mesh (Visuals)
    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("[GPU Erosion] Complete with post-processing!");
}

void TerrainManager::thermalErosionGPU(TerrainObject* terrain, const ThermalErosionParams& p, const std::vector<float>& mask) {
    if (!terrain) return;
    
    if (!cudaInitialized) {
        initCuda();
        if (!cudaInitialized) return;
    }
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    size_t mapSize = w * h * sizeof(float);
    
    // Device Alloc
    CUdeviceptr d_heightmap;
    CUresult res = cuMemAlloc(&d_heightmap, mapSize);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Thermal] Alloc Failed");
        return;
    }
    
    // Allocate hardness map if available
    CUdeviceptr d_hardness = 0;
    bool hasHardness = !terrain->hardnessMap.empty() && terrain->hardnessMap.size() == (size_t)(w * h);
    if (hasHardness) {
        res = cuMemAlloc(&d_hardness, mapSize);
        if (res == CUDA_SUCCESS) {
            cuMemcpyHtoD(d_hardness, terrain->hardnessMap.data(), mapSize);
        } else {
            hasHardness = false;
        }
    }
    
    // HtoD
    cuMemcpyHtoD(d_heightmap, terrain->heightmap.data.data(), mapSize);
    
    // Params
    TerrainPhysics::ThermalErosionParamsGPU gpuOps;
    gpuOps.mapWidth = w;
    gpuOps.mapHeight = h;
    gpuOps.talusAngle = p.talusAngle;
    gpuOps.erosionAmount = p.erosionAmount;
    gpuOps.useHardness = hasHardness;
    gpuOps.hardnessMap = hasHardness ? (float*)d_hardness : nullptr;
    
    // Launch Loop
    int tx = 16, ty = 16;
    int bx = (w + tx - 1) / tx;
    int by = (h + ty - 1) / ty;
    
    // Use hardness-aware kernel if available and hardness map exists
    if (hasHardness && thermalWithHardnessKernelFunc) {
        void* args[] = { &d_heightmap, &d_hardness, &gpuOps };
        for (int i = 0; i < p.iterations; i++) {
            res = cuLaunchKernel((CUfunction)thermalWithHardnessKernelFunc,
                                 bx, by, 1, tx, ty, 1, 0, nullptr, args, nullptr);
        }
        SCENE_LOG_INFO("[GPU Thermal] Using hardness-aware kernel.");
    } else {
        void* args[] = { &d_heightmap, &gpuOps };
        for (int i = 0; i < p.iterations; i++) {
            res = cuLaunchKernel((CUfunction)thermalKernelFunc,
                                 bx, by, 1, tx, ty, 1, 0, nullptr, args, nullptr);
        }
    }
    
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Thermal] Launch Failed: " + std::to_string(res));
    }
    
    // POST-PROCESSING: Pit filling (matches CPU behavior)
    float cellSize = terrain->heightmap.scale_xz / w;
    TerrainPhysics::PostProcessParamsGPU postParams;
    postParams.mapWidth = w;
    postParams.mapHeight = h;
    postParams.cellSize = cellSize;
    postParams.pitThreshold = cellSize * 0.05f;
    postParams.spikeThreshold = cellSize * 0.1f;
    postParams.edgeFadeWidth = std::max(3, w / 40);
    
    if (pitFillKernelFunc) {
        void* pitArgs[] = { &d_heightmap, &postParams };
        cuLaunchKernel((CUfunction)pitFillKernelFunc,
                       bx, by, 1, tx, ty, 1, 0, nullptr, pitArgs, nullptr);
    }
    
    cuCtxSynchronize();
    
    // DtoH
    cuMemcpyDtoH(terrain->heightmap.data.data(), d_heightmap, mapSize);
    
    // Cleanup
    cuMemFree(d_heightmap);
    if (d_hardness) cuMemFree(d_hardness);
    
    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("[GPU Thermal] Completed " + std::to_string(p.iterations) + " iterations" + 
                   (hasHardness ? " with hardness." : "."));
}

void TerrainManager::fluvialErosionGPU(TerrainObject* terrain, const HydraulicErosionParams& p, const std::vector<float>& mask) {
    if (!terrain) return;
    if (!cudaInitialized) { initCuda(); if (!cudaInitialized) return; }
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    int numPixels = w * h;
    
    // Alloc temporary buffers
    CUdeviceptr d_height, d_water, d_flux, d_vel, d_sed;
    cuMemAlloc(&d_height, numPixels * sizeof(float));
    cuMemAlloc(&d_water, numPixels * sizeof(float));
    cuMemAlloc(&d_flux, numPixels * 4 * sizeof(float)); // L, R, T, B
    cuMemAlloc(&d_vel, numPixels * 2 * sizeof(float));
    cuMemAlloc(&d_sed, numPixels * sizeof(float)); // Sediment Map
    
    // Init Data
    cuMemcpyHtoD(d_height, terrain->heightmap.data.data(), numPixels * sizeof(float));
    cuMemsetD8(d_water, 0, numPixels * sizeof(float));
    cuMemsetD8(d_flux, 0, numPixels * 4 * sizeof(float));
    cuMemsetD8(d_vel, 0, numPixels * 2 * sizeof(float));
    cuMemsetD8(d_sed, 0, numPixels * sizeof(float));
    
    // Params - TUNED to prevent flattening
    TerrainPhysics::FluvialErosionParamsGPU gpuOps;
    gpuOps.mapWidth = w;
    gpuOps.mapHeight = h;
    gpuOps.fixedDeltaTime = 0.02f; // Smaller time step for stability
    gpuOps.pipeLength = 1.0f;
    gpuOps.cellSize = 1.0f;
    gpuOps.gravity = 9.8f;
    
    // Map UI params - SCALE DOWN significantly
    gpuOps.erosionRate = p.erodeSpeed * 0.001f; // Much smaller erosion
    gpuOps.depositionRate = p.depositSpeed * 0.5f; // Higher deposition to balance
    gpuOps.evaporationRate = 0.15f; // Faster evaporation to prevent water buildup
    gpuOps.sedimentCapacityConstant = p.sedimentCapacity * 0.1f; // Lower capacity

    int tx = 16, ty = 16;
    int bx = (w + tx - 1) / tx;
    int by = (h + ty - 1) / ty;
    
    // Simulation Loop - Use user's iterations directly (scaled)
    int steps = p.iterations; // Direct user control
    if (steps < 100) steps = 100;
    if (steps > 5000) steps = 5000; // Cap for safety
    
    float rainAmount = 0.002f; // Much less rain

    SCENE_LOG_INFO("[GPU Fluvial] Simulating " + std::to_string(steps) + " steps...");
    
    void* rainArgs[] = { &d_water, &w, &h, &rainAmount, &gpuOps.fixedDeltaTime };
    void* fluxArgs[] = { &d_height, &d_water, &d_flux, &gpuOps };
    void* waterArgs[] = { &d_water, &d_flux, &d_vel, &gpuOps };
    void* erodeArgs[] = { &d_height, &d_water, &d_vel, &d_sed, &gpuOps };
    
    for (int i = 0; i < steps; i++) {
        // 1. Rain
        cuLaunchKernel((CUfunction)fluvRainKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr, rainArgs, nullptr);
        // 2. Flux
        cuLaunchKernel((CUfunction)fluvFluxKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr, fluxArgs, nullptr);
        // 3. Water Update
        cuLaunchKernel((CUfunction)fluvWaterKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr, waterArgs, nullptr);
        // 4. Erosion
        cuLaunchKernel((CUfunction)fluvErodeKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr, erodeArgs, nullptr);
    }
    
    // POST-PROCESSING: Spike removal and pit filling (matches CPU behavior)
    float cellSize = terrain->heightmap.scale_xz / w;
    TerrainPhysics::PostProcessParamsGPU postParams;
    postParams.mapWidth = w;
    postParams.mapHeight = h;
    postParams.cellSize = cellSize;
    postParams.pitThreshold = cellSize * 0.05f;
    postParams.spikeThreshold = cellSize * 0.1f;
    postParams.edgeFadeWidth = std::max(3, w / 40);
    
    if (spikeRemovalKernelFunc) {
        void* spikeArgs[] = { &d_height, &postParams };
        cuLaunchKernel((CUfunction)spikeRemovalKernelFunc,
                       bx, by, 1, tx, ty, 1, 0, nullptr, spikeArgs, nullptr);
    }
    
    if (pitFillKernelFunc) {
        void* pitArgs[] = { &d_height, &postParams };
        cuLaunchKernel((CUfunction)pitFillKernelFunc,
                       bx, by, 1, tx, ty, 1, 0, nullptr, pitArgs, nullptr);
    }
    
    cuCtxSynchronize();
    
    // Copy Back
    cuMemcpyDtoH(terrain->heightmap.data.data(), d_height, numPixels * sizeof(float));
    
    // Cleanup
    cuMemFree(d_height);
    cuMemFree(d_water);
    cuMemFree(d_flux);
    cuMemFree(d_vel);
    cuMemFree(d_sed);
    
    // ========================================================
    // CPU POST-PROCESSING: Edge fade-out (matching CPU fluvial)
    // ========================================================
    auto& height = terrain->heightmap.data;
    int edgeFadeWidth = std::max(5, w / 50);  // ~2% of terrain width
    
    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            int distFromEdge = std::min({x, y, w - 1 - x, h - 1 - y});
            
            if (distFromEdge < edgeFadeWidth) {
                float t = (float)distFromEdge / (float)edgeFadeWidth;
                float smoothT = t * t * (3.0f - 2.0f * t);  // Smoothstep
                
                // Look inward for reference height
                int inwardX = std::clamp(x + (x < w/2 ? edgeFadeWidth : -edgeFadeWidth), 0, w-1);
                int inwardY = std::clamp(y + (y < h/2 ? edgeFadeWidth : -edgeFadeWidth), 0, h-1);
                float inwardHeight = height[inwardY * w + inwardX];
                
                // Full blend with inward height (eliminates walls)
                height[idx] = height[idx] * smoothT + inwardHeight * (1.0f - smoothT);
            }
        }
    }
    
    // ========================================================
    // AGGRESSIVE THERMAL SMOOTHING (matching CPU fluvial)
    // ========================================================
    ThermalErosionParams tp;
    tp.iterations = 15;
    tp.talusAngle = 0.4f;
    tp.erosionAmount = 0.4f;
    thermalErosion(terrain, tp);  // This also calls updateTerrainMesh
    
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("[GPU Fluvial] Complete with edge smoothing.");
}

void TerrainManager::windErosionGPU(TerrainObject* terrain, float strength, float direction, int iterations, const std::vector<float>& mask) {
    if (!terrain) return;
    if (!cudaInitialized) { initCuda(); if (!cudaInitialized) return; }
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    size_t mapSize = w * h * sizeof(float);
    
    // Device Alloc
    CUdeviceptr d_heightmap;
    CUresult res = cuMemAlloc(&d_heightmap, mapSize);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Wind] Alloc Failed");
        return;
    }
    
    // HtoD
    cuMemcpyHtoD(d_heightmap, terrain->heightmap.data.data(), mapSize);
    
    // Params
    TerrainPhysics::WindErosionParamsGPU gpuOps;
    gpuOps.mapWidth = w;
    gpuOps.mapHeight = h;
    
    // Convert direction (degrees) to unit vector
    float rad = direction * 3.14159265f / 180.0f;
    gpuOps.windDirX = cosf(rad);
    gpuOps.windDirY = sinf(rad);
    gpuOps.strength = strength * 0.01f; // Scale down
    gpuOps.suspensionRate = 0.3f;
    gpuOps.depositionRate = 0.8f;
    
    int tx = 16, ty = 16;
    int bx = (w + tx - 1) / tx;
    int by = (h + ty - 1) / ty;
    
    void* args[] = { &d_heightmap, &gpuOps };
    
    for (int i = 0; i < iterations; i++) {
        res = cuLaunchKernel((CUfunction)windKernelFunc,
                             bx, by, 1,
                             tx, ty, 1,
                             0, nullptr,
                             args, nullptr);
    }
    
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Wind] Launch Failed: " + std::to_string(res));
    }
    
    // POST-PROCESSING: Smoothing pass (wind erosion tends to create small artifacts)
    void* smoothArgs[] = { &d_heightmap, &w, &h };
    if (smoothKernelFunc) {
        cuLaunchKernel((CUfunction)smoothKernelFunc,
                       bx, by, 1, tx, ty, 1, 0, nullptr, smoothArgs, nullptr);
    }
    
    cuCtxSynchronize();
    
    // DtoH
    cuMemcpyDtoH(terrain->heightmap.data.data(), d_heightmap, mapSize);
    
    cuMemFree(d_heightmap);
    
    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("[GPU Wind] Completed " + std::to_string(iterations) + " iterations with smoothing.");
}


// ===========================================================================
// FOLIAGE SYSTEM
// ===========================================================================

void TerrainManager::updateFoliage(TerrainObject* terrain, OptixWrapper* optix) {
    if (!terrain || !optix) {
        SCENE_LOG_WARN("[Foliage] updateFoliage called with null terrain or optix");
        return;
    }

    OptixAccelManager* accel = optix->getAccelManager();
    if (!accel) {
        SCENE_LOG_WARN("[Foliage] No AccelManager available");
        return;
    }

    SCENE_LOG_INFO("[Foliage] Updating foliage for terrain " + std::to_string(terrain->id) + 
                   ", layers: " + std::to_string(terrain->foliageLayers.size()));

    // 1. Clear existing foliage instances for this terrain
    clearFoliage(terrain, optix);

    int totalSpawned = 0;

    // 2. Iterate through each foliage layer
    for (size_t layerIdx = 0; layerIdx < terrain->foliageLayers.size(); ++layerIdx) {
        auto& slayer = terrain->foliageLayers[layerIdx];
        
        if (!slayer.enabled) {
            SCENE_LOG_INFO("[Foliage] Layer " + std::to_string(layerIdx) + " disabled, skipping");
            continue;
        }
        if (slayer.meshPath.empty()) {
            SCENE_LOG_INFO("[Foliage] Layer " + std::to_string(layerIdx) + " has no meshPath, skipping");
            continue;
        }
        
        // Skip if mesh not assigned (meshId -1)
        if (slayer.meshId == -1) {
            SCENE_LOG_WARN("[Foliage] Layer " + std::to_string(layerIdx) + 
                          " meshId is -1 (path: " + slayer.meshPath + "), skipping");
            continue; 
        }

        SCENE_LOG_INFO("[Foliage] Layer " + std::to_string(layerIdx) + 
                       ": meshId=" + std::to_string(slayer.meshId) + 
                       ", density=" + std::to_string(slayer.density) +
                       ", path=" + slayer.meshPath);

        // 3. Scatter Logic (Multi-threaded)
        int targetCount = slayer.density; 
        if (targetCount <= 0) {
            SCENE_LOG_WARN("[Foliage] Layer " + std::to_string(layerIdx) + " density is 0, skipping");
            continue;
        }

        int maxAttempts = targetCount * 5; // Avoid infinite loop

        // Cache splat texture dimensions for faster access
        int splatW = 0, splatH = 0;
        Texture* splatTex = terrain->splatMap.get();
        if (splatTex && splatTex->is_loaded()) {
            splatW = splatTex->width;
            splatH = splatTex->height;
        }
        
        // Structure to hold pre-computed instance data
        struct InstanceData {
            float transform[12];
            std::string name;
            bool valid = false;
        };
        
        // Multi-threaded instance data generation
        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        size_t attempts_per_thread = (maxAttempts + num_threads - 1) / num_threads;
        
        std::vector<std::vector<InstanceData>> thread_results(num_threads);
        std::vector<std::future<void>> futures;
        
        for (size_t t = 0; t < num_threads; ++t) {
            size_t attempt_start = t * attempts_per_thread;
            size_t attempt_end = std::min(attempt_start + attempts_per_thread, (size_t)maxAttempts);
            if (attempt_start >= attempt_end) continue;
            
            futures.push_back(std::async(std::launch::async, 
                [&, t, attempt_start, attempt_end]() {
                    // Thread-local RNG with unique seed
                    std::mt19937 rng(12345 + terrain->id * 7 + slayer.meshId * 3 + (int)t * 1000);
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    
                    // Skip to thread's start position
                    for (size_t skip = 0; skip < attempt_start; ++skip) {
                        dist(rng); dist(rng); dist(rng); dist(rng); // Consume random numbers
                    }
                    
                    std::vector<InstanceData>& results = thread_results[t];
                    results.reserve((attempt_end - attempt_start) / 5); // Estimate
                    
                    for (size_t i = attempt_start; i < attempt_end; ++i) {
                        // Random position on terrain (0..1)
                        float u = dist(rng);
                        float v = dist(rng);

                        // Grid coords for heightmap
                        int gx = (int)(u * (terrain->heightmap.width - 1));
                        int gy = (int)(v * (terrain->heightmap.height - 1));

                        // Check Mask
                        float maskValue = 0.0f;
                        if (splatTex && splatW > 0 && splatH > 0) {
                            Vec3 color = splatTex->get_color(u, v);
                            float alpha = splatTex->get_alpha(u, v);
                            
                            int ch = slayer.targetMaskLayerId;
                            if (ch == 0) maskValue = color.x;
                            else if (ch == 1) maskValue = color.y;
                            else if (ch == 2) maskValue = color.z;
                            else if (ch == 3) maskValue = alpha;
                        } else {
                            maskValue = 1.0f;
                        }

                        if (maskValue < slayer.maskThreshold) continue;

                        // Compute World Position
                        float x = u * terrain->heightmap.scale_xz;
                        float z = v * terrain->heightmap.scale_xz;
                        
                        float h_norm = terrain->heightmap.getHeight(gx, gy) / terrain->heightmap.scale_y;
                        float y = h_norm * terrain->heightmap.scale_y;
                        
                        Vec3 pos(x, y, z);
                        if (terrain->transform) {
                            pos = terrain->transform->getFinal().multiplyVector(Vec4(pos, 1.0f)).xyz();
                        }

                        // Transform Generation
                        float scaleC = slayer.scaleRange.x + (slayer.scaleRange.y - slayer.scaleRange.x) * dist(rng);
                        float rotY = slayer.rotationRange.x + (slayer.rotationRange.y - slayer.rotationRange.x) * dist(rng);
                        
                        float rad = rotY * 3.14159265f / 180.0f;
                        float c = cosf(rad);
                        float s = sinf(rad);
                        
                        InstanceData inst;
                        inst.transform[0] = scaleC * c;  inst.transform[1] = 0.0f;      inst.transform[2] = scaleC * s;  inst.transform[3] = pos.x;
                        inst.transform[4] = 0.0f;        inst.transform[5] = scaleC;    inst.transform[6] = 0.0f;        inst.transform[7] = pos.y;
                        inst.transform[8] = scaleC * -s; inst.transform[9] = 0.0f;      inst.transform[10] = scaleC * c; inst.transform[11] = pos.z;
                        inst.name = "Foliage_" + std::to_string(terrain->id) + "_" + std::to_string(slayer.meshId) + "_" + std::to_string(i);
                        inst.valid = true;
                        
                        results.push_back(inst);
                    }
                }));
        }
        
        // Wait for all threads
        for (auto& f : futures) f.get();
        
        // Merge results and add to GPU (serial - GPU API not thread-safe)
        int spawnedCount = 0;
        for (auto& results : thread_results) {
            for (auto& inst : results) {
                if (spawnedCount >= targetCount) break;
                if (!inst.valid) continue;
                
                int instId = accel->addInstance(slayer.meshId, inst.transform, 0, inst.name);
                if (instId >= 0) {
                    slayer.instanceIds.push_back(instId);
                    spawnedCount++;
                }
            }
            if (spawnedCount >= targetCount) break;
        }
        
        SCENE_LOG_INFO("[Foliage] Layer " + std::to_string(layerIdx) + 
                       " spawned " + std::to_string(spawnedCount) + "/" + std::to_string(targetCount) + " instances");
        totalSpawned += spawnedCount;
    }
    
    SCENE_LOG_INFO("[Foliage] Total spawned: " + std::to_string(totalSpawned) + " instances");
    
    // Trigger TLAS rebuild
    if (totalSpawned > 0) {
        accel->buildTLAS();
        SCENE_LOG_INFO("[Foliage] TLAS rebuild triggered");
    }
}

void TerrainManager::clearFoliage(TerrainObject* terrain, OptixWrapper* optix) {
    if (!terrain || !optix) return;
    OptixAccelManager* accel = optix->getAccelManager();
    if (!accel) return;

    for (auto& slayer : terrain->foliageLayers) {
        for (int id : slayer.instanceIds) {
            accel->removeInstance(id);
        }
        slayer.instanceIds.clear();
    }
}


void TerrainManager::reapplyAllFoliage(OptixWrapper* optix) {
    if (!optix) return;
    for (auto& t : terrains) {
        // Only update if foliage layers exist and are enabled
        bool hasFoliage = false;
        for(const auto& l : t.foliageLayers) if(l.enabled && l.density > 0) hasFoliage = true;
        
        if (hasFoliage) {
            updateFoliage(&t, optix);
        }
    }
}

// ===========================================================================
// RIVER BED CARVING SYSTEM
// ===========================================================================

void TerrainManager::lowerHeightAt(float worldX, float worldZ, float amount, float radius, int terrainId) {
    TerrainObject* terrain = nullptr;
    
    if (terrainId < 0) {
        if (!terrains.empty()) terrain = &terrains[0];
    } else {
        terrain = getTerrain(terrainId);
    }
    
    if (!terrain) return;
    
    auto& hm = terrain->heightmap;
    int w = hm.width;
    int h = hm.height;
    
    // Convert world coords to local grid coords using transform matrix
    Vec3 localPos(worldX, 0, worldZ);
    if (terrain->transform) {
        localPos = terrain->transform->getFinal().inverse().multiplyVector(Vec4(worldX, 0, worldZ, 1.0f)).xyz();
    }
    
    float gx = (localPos.x / hm.scale_xz) * (w - 1);
    float gz = (localPos.z / hm.scale_xz) * (h - 1);
    
    // Radius in grid units
    float gridRadius = (radius / hm.scale_xz) * w;
    int iRadius = (int)std::ceil(gridRadius);
    
    int cx = (int)gx;
    int cz = (int)gz;
    
    for (int dz = -iRadius; dz <= iRadius; ++dz) {
        for (int dx = -iRadius; dx <= iRadius; ++dx) {
            int ix = cx + dx;
            int iz = cz + dz;
            
            if (ix < 0 || ix >= w || iz < 0 || iz >= h) continue;
            
            float dist = sqrtf((float)(dx * dx + dz * dz));
            if (dist > gridRadius) continue;
            
            // Smooth falloff (cosine-based)
            float t = dist / gridRadius;
            float falloff = 0.5f * (1.0f + cosf(t * 3.14159f));
            
            int idx = iz * w + ix;
            hm.data[idx] -= (amount / hm.scale_y) * falloff;
        }
    }
}

void TerrainManager::carveRiverBed(int terrainId, 
                                   const std::vector<Vec3>& points,
                                   const std::vector<float>& widths,
                                   const std::vector<float>& depths,
                                   float smoothness,
                                   SceneData& scene) {
    if (points.size() < 2) return;
    
    TerrainObject* terrain = nullptr;
    if (terrainId < 0) {
        if (!terrains.empty()) terrain = &terrains[0];
    } else {
        terrain = getTerrain(terrainId);
    }
    
    if (!terrain) {
        SCENE_LOG_WARN("[TerrainManager] carveRiverBed: No terrain found");
        return;
    }
    
    auto& hm = terrain->heightmap;
    int w = hm.width;
    int h = hm.height;
    float halfSize = hm.scale_xz * 0.5f;
    
    SCENE_LOG_INFO("[TerrainManager] Carving continuous river bed with " + 
                   std::to_string(points.size()) + " points");
    
    // For each heightmap cell, check distance to the river line
    #pragma omp parallel for
    for (int gz = 0; gz < h; ++gz) {
        for (int gx = 0; gx < w; ++gx) {
            // Convert grid to world coords using transform matrix
            Vec3 localPos((float)gx / (w - 1) * hm.scale_xz, 0, (float)gz / (h - 1) * hm.scale_xz);
            Vec3 worldPos = localPos;
            if (terrain->transform) {
                worldPos = terrain->transform->getFinal().multiplyVector(Vec4(localPos, 1.0f)).xyz();
            }
            float worldX = worldPos.x;
            float worldZ = worldPos.z;
            
            // Find nearest point on river spline segments
            float minDist = 1e9f;
            float nearestWidth = 2.0f;
            float nearestDepth = 0.5f;
            
            for (size_t i = 0; i < points.size() - 1; ++i) {
                const Vec3& p0 = points[i];
                const Vec3& p1 = points[i + 1];
                
                // Project point onto line segment (2D: XZ plane)
                Vec3 lineDir(p1.x - p0.x, 0, p1.z - p0.z);
                float lineLen = sqrtf(lineDir.x * lineDir.x + lineDir.z * lineDir.z);
                if (lineLen < 0.001f) continue;
                
                lineDir.x /= lineLen;
                lineDir.z /= lineLen;
                
                Vec3 toPoint(worldX - p0.x, 0, worldZ - p0.z);
                float projection = toPoint.x * lineDir.x + toPoint.z * lineDir.z;
                
                // Clamp to segment
                float t = projection / lineLen;
                t = (std::max)(0.0f, (std::min)(1.0f, t));
                
                // Nearest point on segment
                float nearX = p0.x + lineDir.x * lineLen * t;
                float nearZ = p0.z + lineDir.z * lineLen * t;
                
                float dx = worldX - nearX;
                float dz = worldZ - nearZ;
                float dist = sqrtf(dx * dx + dz * dz);
                
                if (dist < minDist) {
                    minDist = dist;
                    // Interpolate width and depth
                    float w0 = (i < widths.size()) ? widths[i] : 2.0f;
                    float w1 = (i + 1 < widths.size()) ? widths[i + 1] : 2.0f;
                    float d0 = (i < depths.size()) ? depths[i] : 0.5f;
                    float d1 = (i + 1 < depths.size()) ? depths[i + 1] : 0.5f;
                    
                    nearestWidth = w0 + (w1 - w0) * t;
                    nearestDepth = d0 + (d1 - d0) * t;
                }
            }
            
            // Carve if within river width (with falloff for banks)
            float halfWidth = nearestWidth * 0.5f;
            float bankWidth = halfWidth * (1.0f + smoothness * 0.5f);  // Extended bank area
            
            if (minDist < bankWidth) {
                float normalizedDist = minDist / halfWidth;
                
                float carveAmount;
                if (normalizedDist < 1.0f) {
                    // Inside main channel - full depth with flat bottom
                    // Use smoothstep for gradual transition at edges
                    float edgeFade = 1.0f - normalizedDist * normalizedDist * normalizedDist;
                    carveAmount = nearestDepth * edgeFade;
                } else {
                    // Bank area - gradual slope
                    float bankT = (minDist - halfWidth) / (bankWidth - halfWidth);
                    bankT = (std::min)(1.0f, bankT);
                    // Smooth falloff using cosine
                    float bankFade = 0.5f * (1.0f + cosf(bankT * 3.14159f));
                    carveAmount = nearestDepth * 0.3f * bankFade;
                }
                
                if (carveAmount > 0.001f) {
                    int idx = gz * w + gx;
                    hm.data[idx] -= carveAmount / hm.scale_y;
                }
            }
        }
    }
    
    // Update terrain mesh
    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    
    SCENE_LOG_INFO("[TerrainManager] River bed carved successfully (continuous band)");
}

// ═══════════════════════════════════════════════════════════════════════════════
// NATURAL RIVER BED CARVING - Advanced Algorithm
// ═══════════════════════════════════════════════════════════════════════════════
// Features:
// - Perlin-like noise for edge irregularity
// - Curvature-based asymmetric banks (meander physics)
// - Deep pools at random intervals
// - Shallow riffles (rapids zones)
// - Point bar deposits on inner bends
// ═══════════════════════════════════════════════════════════════════════════════

void TerrainManager::carveRiverBedNatural(int terrainId, 
                                          const std::vector<Vec3>& points,
                                          const std::vector<float>& widths,
                                          const std::vector<float>& depths,
                                          float smoothness,
                                          const NaturalCarveParams& np,
                                          SceneData& scene) {
    if (points.size() < 2) return;
    
    TerrainObject* terrain = nullptr;
    if (terrainId < 0) {
        if (!terrains.empty()) terrain = &terrains[0];
    } else {
        terrain = getTerrain(terrainId);
    }
    
    if (!terrain) {
        SCENE_LOG_WARN("[TerrainManager] carveRiverBedNatural: No terrain found");
        return;
    }
    
    auto& hm = terrain->heightmap;
    int w = hm.width;
    int h = hm.height;
    float halfSize = hm.scale_xz * 0.5f;
    
    // ─────────────────────────────────────────────────────────────────────────
    // TERRAIN RESOLUTION CHECKS
    // ─────────────────────────────────────────────────────────────────────────
    float cellSize = hm.scale_xz / (float)(w - 1);
    
    // Calculate maximum reasonable depth based on terrain scale
    // Don't carve deeper than 20% of terrain height range
    float maxReasonableDepth = hm.scale_y * 0.2f;
    
    // Warn if terrain resolution is too low for river width
    float minRiverWidth = 1e9f;
    for (const auto& width : widths) {
        if (width < minRiverWidth) minRiverWidth = width;
    }
    int pixelsAcrossRiver = (int)(minRiverWidth / cellSize);
    if (pixelsAcrossRiver < 4) {
        SCENE_LOG_WARN("[TerrainManager] Low resolution warning: River is only " + 
                       std::to_string(pixelsAcrossRiver) + " pixels wide. Consider higher terrain resolution.");
    }
    
    SCENE_LOG_INFO("[TerrainManager] Carving NATURAL river bed with " + 
                   std::to_string(points.size()) + " points (cell size: " + 
                   std::to_string(cellSize) + ", max depth: " + std::to_string(maxReasonableDepth) + ")");
    
    // ─────────────────────────────────────────────────────────────────────────
    // PRECOMPUTE: Curvature and tangent data along spline
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<float> curvatures(points.size(), 0.0f);
    std::vector<Vec3> tangents(points.size());
    std::vector<Vec3> normals(points.size());  // Right-hand perpendicular
    std::vector<float> turnDirections(points.size(), 0.0f);  // +1 = right, -1 = left
    
    for (size_t i = 0; i < points.size(); ++i) {
        // Tangent calculation
        Vec3 tangent(0, 0, 1);
        if (i == 0 && points.size() > 1) {
            tangent = (points[1] - points[0]);
        } else if (i == points.size() - 1 && points.size() > 1) {
            tangent = (points[i] - points[i - 1]);
        } else if (i > 0 && i < points.size() - 1) {
            tangent = (points[i + 1] - points[i - 1]) * 0.5f;
        }
        tangent.y = 0;  // Project to XZ plane
        float len = sqrtf(tangent.x * tangent.x + tangent.z * tangent.z);
        if (len > 0.001f) {
            tangent.x /= len;
            tangent.z /= len;
        }
        tangents[i] = tangent;
        
        // Normal (perpendicular, pointing right)
        normals[i] = Vec3(-tangent.z, 0, tangent.x);
        
        // Curvature calculation (change in tangent direction)
        if (i > 0 && i < points.size() - 1) {
            Vec3 t1 = (points[i] - points[i - 1]);
            Vec3 t2 = (points[i + 1] - points[i]);
            t1.y = 0; t2.y = 0;
            float l1 = sqrtf(t1.x * t1.x + t1.z * t1.z);
            float l2 = sqrtf(t2.x * t2.x + t2.z * t2.z);
            if (l1 > 0.001f && l2 > 0.001f) {
                t1.x /= l1; t1.z /= l1;
                t2.x /= l2; t2.z /= l2;
                float dot = t1.x * t2.x + t1.z * t2.z;
                curvatures[i] = 1.0f - std::clamp(dot, -1.0f, 1.0f);
                
                // Turn direction: cross product z component
                float cross = t1.x * t2.z - t1.z * t2.x;
                turnDirections[i] = (cross > 0) ? 1.0f : -1.0f;
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // PRECOMPUTE: Pool and Riffle zones (pseudo-random along river)
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<float> depthModifiers(points.size(), 1.0f);  // 1.0 = normal
    
    if (np.enableDeepPools || np.enableRiffles) {
        // Simple hash-based pseudo-random for deterministic results
        auto hashFloat = [](float x, float y, float seed) -> float {
            int ix = (int)(x * 1000.0f);
            int iy = (int)(y * 1000.0f);
            int h = ix * 374761393 + iy * 668265263 + (int)(seed * 1000.0f);
            h = (h ^ (h >> 13)) * 1274126177;
            return (float)(h & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        };
        
        for (size_t i = 0; i < points.size(); ++i) {
            float randVal = hashFloat(points[i].x, points[i].z, 42.0f);
            
            // Deep pools (less frequent, deeper)
            if (np.enableDeepPools && randVal < np.poolFrequency) {
                depthModifiers[i] = np.poolDepthMult;
            }
            // Riffles (more frequent, shallower) - only if not a pool
            else if (np.enableRiffles && randVal > (1.0f - np.riffleFrequency)) {
                depthModifiers[i] = np.riffleDepthMult;
            }
        }
        
        // Smooth depth modifiers for natural transitions
        std::vector<float> smoothed = depthModifiers;
        for (size_t i = 2; i < points.size() - 2; ++i) {
            smoothed[i] = (depthModifiers[i-2] + depthModifiers[i-1] * 2 + 
                          depthModifiers[i] * 4 + depthModifiers[i+1] * 2 + 
                          depthModifiers[i+2]) / 10.0f;
        }
        depthModifiers = smoothed;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // NOISE FUNCTION: Smooth gradient noise for natural variation
    // Uses smooth hermite interpolation to avoid sharp transitions
    // ─────────────────────────────────────────────────────────────────────────
    
    // Smooth step function (hermite interpolation)
    auto smoothstep = [](float t) -> float {
        t = std::clamp(t, 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    };
    
    // Smoother step (Ken Perlin's improved version)
    auto smootherstep = [](float t) -> float {
        t = std::clamp(t, 0.0f, 1.0f);
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    };
    
    // Hash function for pseudo-random gradient
    auto hash2D = [](int ix, int iy) -> float {
        int n = ix + iy * 57;
        n = (n << 13) ^ n;
        return (1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f);
    };
    
    // Smooth value noise (grid-based interpolation)
    auto getSmoothNoise = [&](float x, float z, float scale) -> float {
        x *= scale;
        z *= scale;
        
        int ix = (int)std::floor(x);
        int iz = (int)std::floor(z);
        
        float fx = x - ix;
        float fz = z - iz;
        
        // Smooth the fractions
        fx = smootherstep(fx);
        fz = smootherstep(fz);
        
        // Get corner values
        float v00 = hash2D(ix, iz);
        float v10 = hash2D(ix + 1, iz);
        float v01 = hash2D(ix, iz + 1);
        float v11 = hash2D(ix + 1, iz + 1);
        
        // Bilinear interpolation with smoothed fractions
        float v0 = v00 * (1.0f - fx) + v10 * fx;
        float v1 = v01 * (1.0f - fx) + v11 * fx;
        
        return v0 * (1.0f - fz) + v1 * fz;
    };
    
    // Multi-octave smooth noise (FBM-like)
    auto getNoiseValue = [&](float x, float z) -> float {
        if (!np.enableNoise) return 0.0f;
        
        float value = 0.0f;
        float amplitude = 1.0f;
        float totalAmp = 0.0f;
        float freq = np.noiseScale;
        
        // 3 octaves of smooth noise
        for (int oct = 0; oct < 3; ++oct) {
            value += getSmoothNoise(x, z, freq) * amplitude;
            totalAmp += amplitude;
            freq *= 2.0f;
            amplitude *= 0.5f;
        }
        
        value /= totalAmp;  // Normalize to [-1, 1]
        return value * np.noiseStrength;
    };
    
    // ─────────────────────────────────────────────────────────────────────────
    // MAIN CARVING LOOP
    // ─────────────────────────────────────────────────────────────────────────
    #pragma omp parallel for
    for (int gz = 0; gz < h; ++gz) {
        for (int gx = 0; gx < w; ++gx) {
            // Convert grid to world coords using transform matrix
            Vec3 localPos((float)gx / (w - 1) * hm.scale_xz, 0, (float)gz / (h - 1) * hm.scale_xz);
            Vec3 worldPos = localPos;
            if (terrain->transform) {
                worldPos = terrain->transform->getFinal().multiplyVector(Vec4(localPos, 1.0f)).xyz();
            }
            float worldX = worldPos.x;
            float worldZ = worldPos.z;
            
            // Find nearest point on river spline
            float minDist = 1e9f;
            float nearestWidth = 2.0f;
            float nearestDepth = 0.5f;
            float nearestCurvature = 0.0f;
            float nearestTurnDir = 0.0f;
            float nearestDepthMod = 1.0f;
            Vec3 nearestNormal(0, 0, 1);
            Vec3 nearestToPoint(0, 0, 0);
            float nearestT = 0.0f;  // Parameter along segment
            
            for (size_t i = 0; i < points.size() - 1; ++i) {
                const Vec3& p0 = points[i];
                const Vec3& p1 = points[i + 1];
                
                // Project point onto line segment (2D: XZ plane)
                Vec3 lineDir(p1.x - p0.x, 0, p1.z - p0.z);
                float lineLen = sqrtf(lineDir.x * lineDir.x + lineDir.z * lineDir.z);
                if (lineLen < 0.001f) continue;
                
                lineDir.x /= lineLen;
                lineDir.z /= lineLen;
                
                Vec3 toPoint(worldX - p0.x, 0, worldZ - p0.z);
                float projection = toPoint.x * lineDir.x + toPoint.z * lineDir.z;
                
                // Clamp to segment
                float t = projection / lineLen;
                t = std::clamp(t, 0.0f, 1.0f);
                
                // Nearest point on segment
                float nearX = p0.x + lineDir.x * lineLen * t;
                float nearZ = p0.z + lineDir.z * lineLen * t;
                
                float dx = worldX - nearX;
                float dz = worldZ - nearZ;
                float dist = sqrtf(dx * dx + dz * dz);
                
                if (dist < minDist) {
                    minDist = dist;
                    nearestToPoint = Vec3(dx, 0, dz);
                    nearestT = t;
                    
                    // Interpolate properties
                    float w0 = (i < widths.size()) ? widths[i] : 2.0f;
                    float w1 = (i + 1 < widths.size()) ? widths[i + 1] : 2.0f;
                    float d0 = (i < depths.size()) ? depths[i] : 0.5f;
                    float d1 = (i + 1 < depths.size()) ? depths[i + 1] : 0.5f;
                    
                    nearestWidth = w0 + (w1 - w0) * t;
                    nearestDepth = d0 + (d1 - d0) * t;
                    
                    // Interpolate precomputed data
                    nearestCurvature = curvatures[i] + (curvatures[i + 1] - curvatures[i]) * t;
                    nearestTurnDir = turnDirections[i] + (turnDirections[i + 1] - turnDirections[i]) * t;
                    nearestNormal = normals[i];  // Use segment start normal
                    nearestDepthMod = depthModifiers[i] + (depthModifiers[i + 1] - depthModifiers[i]) * t;
                }
            }
            
            // ─────────────────────────────────────────────────────────────────
            // WIDTH - NO NOISE ON WIDTH (prevents sawtooth edges)
            // Noise is applied to DEPTH only, not width
            // ─────────────────────────────────────────────────────────────────
            float effectiveWidth = nearestWidth;  // Clean width without noise
            
            // ─────────────────────────────────────────────────────────────────
            // ASYMMETRIC BANKS (Meander Physics)
            // ─────────────────────────────────────────────────────────────────
            float asymmetryFactor = 1.0f;
            if (np.enableAsymmetry && nearestCurvature > 0.01f) {
                // Determine which side of river this point is on
                float sideSign = (nearestToPoint.x * nearestNormal.x + 
                                 nearestToPoint.z * nearestNormal.z);
                sideSign = (sideSign > 0) ? 1.0f : -1.0f;
                
                // Inner bank (same side as turn) is shallower, outer is deeper
                float isInnerBank = sideSign * nearestTurnDir;
                
                // Asymmetry modifies depth: outer bank deeper, inner shallower
                float asymmetryAmount = nearestCurvature * np.asymmetryStrength * isInnerBank;
                asymmetryFactor = 1.0f - asymmetryAmount * 0.4f;  // 0.6 to 1.4 range
            }
            
            // ─────────────────────────────────────────────────────────────────
            // POINT BAR DEPOSITS (Inner bends)
            // ─────────────────────────────────────────────────────────────────
            float pointBarRaise = 0.0f;
            if (np.enablePointBars && nearestCurvature > 0.05f) {
                float sideSign = (nearestToPoint.x * nearestNormal.x + 
                                 nearestToPoint.z * nearestNormal.z);
                sideSign = (sideSign > 0) ? 1.0f : -1.0f;
                float isInnerBank = sideSign * nearestTurnDir;
                
                // Point bar on inner half
                if (isInnerBank > 0) {
                    float halfWidth = effectiveWidth * 0.5f;
                    float normalizedDist = minDist / halfWidth;
                    if (normalizedDist < 1.2f && normalizedDist > 0.3f) {
                        float barStrength = nearestCurvature * np.pointBarStrength;
                        float barProfile = 1.0f - fabsf(normalizedDist - 0.7f) / 0.5f;
                        barProfile = std::clamp(barProfile, 0.0f, 1.0f);
                        barProfile = smoothstep(barProfile);  // Smooth the profile
                        pointBarRaise = barStrength * barProfile * nearestDepth * 0.5f;
                    }
                }
            }
            
            // ─────────────────────────────────────────────────────────────────
            // CARVING CALCULATION
            // ─────────────────────────────────────────────────────────────────
            float halfWidth = effectiveWidth * 0.5f;
            float bankWidth = halfWidth * (1.0f + smoothness * 0.5f);
            
            int idx = gz * w + gx;
            
            if (minDist < bankWidth) {
                float normalizedDist = minDist / halfWidth;
                
                // Apply depth modifier (pools/riffles) and asymmetry
                float finalDepth = nearestDepth * nearestDepthMod * asymmetryFactor;
                
                // CLAMP depth to reasonable maximum (prevents excessive carving)
                finalDepth = std::min(finalDepth, maxReasonableDepth);
                
                // Get smooth noise for this position (used for depth variation only)
                float depthNoise = getNoiseValue(worldX, worldZ);
                
                float carveAmount;
                if (normalizedDist < 1.0f) {
                    // Inside main channel
                    // Smooth parabolic profile for natural cross-section
                    float profileT = normalizedDist;
                    float channelProfile = 1.0f - profileT * profileT;
                    
                    // Apply smooth noise to DEPTH only (not edges)
                    // Noise effect decreases toward edges (center has most variation)
                    float centerWeight = 1.0f - profileT;  // 1 at center, 0 at edge
                    float noiseEffect = depthNoise * 0.15f * centerWeight;
                    
                    carveAmount = finalDepth * channelProfile * (1.0f + noiseEffect);
                } else {
                    // Bank area - smooth gradual slope
                    float bankT = (minDist - halfWidth) / (bankWidth - halfWidth);
                    bankT = std::clamp(bankT, 0.0f, 1.0f);
                    
                    // Smoothstep falloff (no noise on banks to prevent sawtooth)
                    float bankFade = 1.0f - smoothstep(bankT);
                    
                    carveAmount = finalDepth * 0.3f * bankFade;
                }
                
                // Apply point bar (reduces carving / raises terrain)
                carveAmount -= pointBarRaise;
                
                if (carveAmount > 0.001f) {
                    hm.data[idx] -= carveAmount / hm.scale_y;
                } else if (carveAmount < -0.001f) {
                    // Point bar actually raises terrain
                    hm.data[idx] -= carveAmount / hm.scale_y;  // Negative carve = raise
                }
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // POST-CARVE SMOOTHING PASS
    // Eliminates sawtooth artifacts on low-resolution terrains
    // Uses Gaussian-like weighted average for smooth results
    // ─────────────────────────────────────────────────────────────────────────
    
    // Smoothing radius in pixels (at least 1, scale with resolution)
    // Low resolution = more smoothing needed
    int smoothRadius = std::max(1, (int)std::ceil(2.0f / cellSize));
    smoothRadius = std::min(smoothRadius, 3);  // Cap at 3 for performance
    
    // Create a copy of affected heightmap region for smoothing
    std::vector<float> smoothedData = hm.data;
    
    // Apply Gaussian-like smoothing only to carved areas
    #pragma omp parallel for
    for (int gz = smoothRadius; gz < h - smoothRadius; ++gz) {
        for (int gx = smoothRadius; gx < w - smoothRadius; ++gx) {
            // Quick check: only smooth near river path
            float worldX = ((float)gx / (w - 1)) * hm.scale_xz - halfSize;
            float worldZ = ((float)gz / (h - 1)) * hm.scale_xz - halfSize;
            
            // Find approximate distance to river (fast check)
            float minDistApprox = 1e9f;
            float maxRiverWidth = 0.0f;
            for (size_t i = 0; i < points.size(); i += 3) {  // Sample every 3rd point for speed
                float dx = worldX - points[i].x;
                float dz = worldZ - points[i].z;
                float d = sqrtf(dx * dx + dz * dz);
                if (d < minDistApprox) {
                    minDistApprox = d;
                    if (i < widths.size()) maxRiverWidth = widths[i];
                }
            }
            
            // Only smooth within extended river area
            if (minDistApprox > maxRiverWidth * 1.5f) continue;
            
            int idx = gz * w + gx;
            
            // Gaussian-weighted average
            float sum = 0.0f;
            float weightSum = 0.0f;
            
            for (int dz = -smoothRadius; dz <= smoothRadius; ++dz) {
                for (int dx = -smoothRadius; dx <= smoothRadius; ++dx) {
                    int nx = gx + dx;
                    int nz = gz + dz;
                    int nidx = nz * w + nx;
                    
                    // Gaussian weight based on distance
                    float dist = sqrtf((float)(dx * dx + dz * dz));
                    float sigma = (float)smoothRadius * 0.5f;
                    float weight = expf(-(dist * dist) / (2.0f * sigma * sigma));
                    
                    sum += hm.data[nidx] * weight;
                    weightSum += weight;
                }
            }
            
            if (weightSum > 0.0f) {
                // Blend: 70% smoothed, 30% original (preserves some detail)
                float smoothedValue = sum / weightSum;
                smoothedData[idx] = smoothedValue * 0.7f + hm.data[idx] * 0.3f;
            }
        }
    }
    
    // Apply smoothed data back
    hm.data = smoothedData;
    
    // Update terrain mesh
    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    
    SCENE_LOG_INFO("[TerrainManager] Natural river bed carved with smoothing (radius: " + 
                   std::to_string(smoothRadius) + ")!");
}
