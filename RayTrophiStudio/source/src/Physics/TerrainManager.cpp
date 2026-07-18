#include "TerrainManager.h"
#include "scene_data.h"
#include "Triangle.h" // Added for explicit type visibility
#include "Hittable.h" // Added for explicit type visibility
#include "TerrainNodesV2.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "globals.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <queue>
#include <limits>
#include <filesystem>
#include <fstream>
#include <functional>
#include <array>
#include "Texture.h"
#include "Material.h" // Ensure Material class is fully defined
#include "InstanceManager.h"
#include "Backend/IBackend.h"
#include "Backend/VulkanBackend.h"
#include "SimulationCompute.h"   // shared Vulkan compute backend (GPU terrain erosion)

// CUDA Driver API
#include <cuda.h>
#include <cuda_runtime.h>
#include "erosion_ops.cuh"
#include "OptixWrapper.h"
#include "OptixAccelManager.h"
#include "../Utils/image_resample.h"
#include "../Utils/image_filters.h"

// For linking with CUDA Driver API (nvcuda.dll is loaded by driver usually, but we need cuda.lib for symbols)
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "delayimp.lib") // Required for delay load in MSVC

extern std::unique_ptr<Backend::IBackend> g_backend;

namespace {
bool terrainRenderBackendIsVulkan() {
    return dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get()) != nullptr;
}
}

// Helper function for autoMask
inline float smoothstep(float edge0, float edge1, float x) {
    x = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return x * x * (3 - 2 * x);
}

// Helper for sigmoid clamping
static float padding_sigmoid(float x, float k) {
    return x / (x + k); 
}

// Remove single-pixel spikes using a 3x3 median filter applied selectively.
// If a pixel differs from the local median by more than `threshold`,
// replace it with the median (or blend if blend_factor < 1).
static void remove_spikes_3x3(std::vector<float>& data, int w, int h, float threshold = 0.05f, float blend_factor = 1.0f) {
    if (w <= 0 || h <= 0) return;
    std::vector<float> copy = data;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float neigh[9];
            int idx = 0;
            for (int oy = -1; oy <= 1; ++oy) {
                int yy = y + oy;
                if (yy < 0) yy = 0; if (yy >= h) yy = h - 1;
                for (int ox = -1; ox <= 1; ++ox) {
                    int xx = x + ox;
                    if (xx < 0) xx = 0; if (xx >= w) xx = w - 1;
                    neigh[idx++] = copy[yy * w + xx];
                }
            }
            // median of 9
            std::sort(neigh, neigh + 9);
            float med = neigh[4];
            float cur = copy[y * w + x];
            float diff = fabsf(cur - med);
            if (diff > threshold) {
                float out = med * blend_factor + cur * (1.0f - blend_factor);
                data[y * w + x] = out;
            }
        }
    }
}

namespace {
std::vector<std::shared_ptr<Triangle>> findLegacyFoliageSourceTriangles(
    const std::string& meshPathOrNodeName,
    const SceneData& scene) {
    std::vector<std::shared_ptr<Triangle>> triangles;
    if (meshPathOrNodeName.empty()) return triangles;

    const std::string stem = std::filesystem::path(meshPathOrNodeName).stem().string();
    auto nameMatches = [&](const std::string& nodeName) {
        return !nodeName.empty() &&
               (nodeName == meshPathOrNodeName || (!stem.empty() && nodeName == stem));
    };

    // Flat (SoA) objects live in world.objects as TriangleMesh, not per-face Triangle facades —
    // a Triangle-only scan found nothing for them, and multi-material imports split into several
    // sibling TriangleMesh sharing one nodeName. Materialize every face of every matching sibling.
    std::unordered_set<TriangleMesh*> seenMeshes;
    for (const auto& obj : scene.world.objects) {
        if (auto tmesh = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (!tmesh->geometry || !nameMatches(tmesh->nodeName)) continue;
            if (!seenMeshes.insert(tmesh.get()).second) continue;
            const size_t nTris = tmesh->num_triangles();
            triangles.reserve(triangles.size() + nTris);
            for (size_t f = 0; f < nTris; ++f) {
                triangles.push_back(std::make_shared<Triangle>(tmesh, static_cast<uint32_t>(f)));
            }
            continue;
        }
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (tri && nameMatches(tri->getNodeName())) {
            triangles.push_back(tri);
        }
    }

    return triangles;
}

std::string makeUniqueFoliageGroupName(const std::string& baseName, InstanceManager& im) {
    std::string candidate = baseName.empty() ? "Foliage_Legacy" : baseName;
    int suffix = 1;
    while (im.findGroupByName(candidate) != nullptr) {
        candidate = baseName + "_" + std::to_string(suffix++);
    }
    return candidate;
}

Vec3 computeLegacyFoliageNormal(TerrainManager& mgr, TerrainObject* terrain, const Vec3& worldPos) {
    if (!terrain || terrain->heightmap.width <= 1 || terrain->heightmap.height <= 1) {
        return Vec3(0.0f, 1.0f, 0.0f);
    }

    Vec3 localPos = worldPos;
    if (terrain->transform) {
        Matrix4x4 inv = terrain->transform->getFinal().inverse();
        localPos = inv.multiplyVector(Vec4(worldPos, 1.0f)).xyz();
    }

    float normalizedX = std::clamp(localPos.x / terrain->heightmap.scale_xz, 0.0f, 1.0f);
    float normalizedZ = std::clamp(localPos.z / terrain->heightmap.scale_xz, 0.0f, 1.0f);
    int ix = std::clamp((int)(normalizedX * (terrain->heightmap.width - 1) + 0.5f), 0, terrain->heightmap.width - 1);
    int iz = std::clamp((int)(normalizedZ * (terrain->heightmap.height - 1) + 0.5f), 0, terrain->heightmap.height - 1);

    Vec3 localNormal = mgr.calculateSobelNormal(terrain, ix, iz);
    if (terrain->transform) {
        localNormal = terrain->transform->getFinal().multiplyVector(Vec4(localNormal, 0.0f)).xyz().normalize();
    }
    return localNormal;
}

void applyLegacyNormalAlignment(InstanceTransform& inst, const Vec3& normal, float influence) {
    if (influence <= 0.0f || normal.length() <= 0.01f) return;
    Vec3 up(0.0f, 1.0f, 0.0f);
    Vec3 target = (up * (1.0f - influence) + normal.normalize() * influence).normalize();
    inst.rotation.x += asinf(-target.z) * 180.0f / 3.14159f * influence;
    inst.rotation.z += asinf(target.x) * 180.0f / 3.14159f * influence;
}
}

TerrainObject* TerrainManager::getTerrain(int id) {
    for (auto& t : terrains) {
        if (t.id == id) return &t;
    }
    return nullptr;
}

namespace {
// Builds a WELDED flat TriangleMesh from a regular (w x h) vertex grid + explicit
// index buffer. Terrain (and, in a later phase, Water/River) grids are regular
// meshes where adjacent quads legitimately share vertices/normals/UVs — unlike
// the unwelded per-corner soup used by podToFlatMesh (MeshModifiers.cpp).
std::shared_ptr<TriangleMesh> gridToFlatMesh(
    const std::vector<Vec3>& positions,   // local space, w*h
    const std::vector<Vec3>& normals,     // local space, w*h
    const std::vector<Vec2>& uvs,         // w*h
    const std::vector<uint32_t>& indices, // 3 * triCount, welded grid indices
    uint16_t materialID,
    const std::shared_ptr<Transform>& transform,
    const std::string& nodeName) {
    const size_t vCount = positions.size();
    if (vCount == 0 || indices.empty()) return nullptr;

    auto tm = std::make_shared<TriangleMesh>();
    tm->transform = transform;
    tm->nodeName = nodeName;
    tm->geometry->resize_vertices(vCount);

    tm->geometry->add_attribute<Vec3>("P");
    tm->geometry->add_attribute<Vec3>("N");
    tm->geometry->add_attribute<Vec3>("P_orig");
    tm->geometry->add_attribute<Vec3>("N_orig");
    tm->geometry->add_attribute<Vec2>("uv");
    tm->geometry->add_attribute<uint16_t>("materialID");

    Vec3* P  = tm->geometry->get_attribute_data_mut<Vec3>("P");
    Vec3* N  = tm->geometry->get_attribute_data_mut<Vec3>("N");
    Vec3* Po = tm->geometry->get_attribute_data_mut<Vec3>("P_orig");
    Vec3* No = tm->geometry->get_attribute_data_mut<Vec3>("N_orig");
    Vec2* UV = tm->geometry->get_attribute_data_mut<Vec2>("uv");
    uint16_t* M = tm->geometry->get_attribute_data_mut<uint16_t>("materialID");

    Matrix4x4 finalT = Matrix4x4::identity();
    Matrix4x4 normalT = Matrix4x4::identity();
    if (transform) {
        finalT = transform->getFinal();
        normalT = transform->getNormalTransform();
    }

    #pragma omp parallel for schedule(static) if(vCount >= 2048)
    for (int i = 0; i < (int)vCount; ++i) {
        const Vec3& lp = positions[(size_t)i];
        const Vec3& ln = normals[(size_t)i];
        if (Po) Po[i] = lp;
        if (No) No[i] = ln;
        if (P)  P[i]  = finalT.transform_point(lp);
        if (N)  N[i]  = normalT.transform_vector(ln).normalize();
        if (UV && !uvs.empty()) UV[i] = uvs[(size_t)i];
        if (M)  M[i]  = materialID;
    }

    tm->geometry->indices.resize(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) tm->geometry->indices[i] = indices[i];

    return tm;
}

// A TerrainObject owns exactly one scene object.  Keep this invariant centralized:
// graph evaluation may rebuild/register the mesh before a load/create path reaches
// its own finalization step, so a blind push_back can otherwise register the same
// shared_ptr more than once.
static void registerTerrainMeshOnce(SceneData& scene, const std::shared_ptr<TriangleMesh>& mesh) {
    if (!mesh) return;

    auto& objects = scene.world.objects;
    objects.erase(std::remove_if(objects.begin(), objects.end(),
        [&](const std::shared_ptr<Hittable>& object) {
            return object.get() == mesh.get();
        }), objects.end());
    objects.push_back(mesh);
}
} // namespace

void TerrainManager::removeTerrain(SceneData& scene, int id) {
    for (auto it = terrains.begin(); it != terrains.end();) {
        if (it->id != id) {
            ++it;
            continue;
        }

        // Remove every scene occurrence. Older load paths could register the same
        // pointer twice, while malformed state could also contain two owners with
        // the same persistent ID.
        if (it->flatMesh) {
            Hittable* mesh = it->flatMesh.get();
            auto& objects = scene.world.objects;
            objects.erase(std::remove_if(objects.begin(), objects.end(),
                [&](const std::shared_ptr<Hittable>& object) {
                    return object.get() == mesh;
                }), objects.end());
        }

        it = terrains.erase(it);
    }
}
// Optimization: Use unordered_set for O(1) lookup
#include <unordered_set>
#include <ProjectData.h>

void TerrainManager::removeAllTerrains(SceneData& scene) {
    if (terrains.empty()) return;

    // 1. Collect all terrain flat-mesh pointers to remove
    std::unordered_set<Hittable*> terrain_meshes;
    terrain_meshes.reserve(terrains.size());

    for (auto& t : terrains) {
        if (t.flatMesh) terrain_meshes.insert(t.flatMesh.get());
    }

    // 2. Remove from scene using remove_if (O(N))
    auto& objs = scene.world.objects;
    objs.erase(
        std::remove_if(objs.begin(), objs.end(),
            [&](const std::shared_ptr<Hittable>& obj) {
                return terrain_meshes.count(obj.get()) > 0;
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
    
    // Initialize Node Graph
    terrain.nodeGraph = std::make_shared<TerrainNodesV2::TerrainNodeGraphV2>();
    
    // Init Heightmap (Flat)
    terrain.heightmap.width = resolution;
    terrain.heightmap.height = resolution;
    terrain.heightmap.scale_xz = size;
    terrain.heightmap.scale_y = 10.0f; // Default range
    terrain.heightmap.data.resize(resolution * resolution, 0.0f);
    terrain.original_heightmap_data = terrain.heightmap.data;
    
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
    registerTerrainMeshOnce(scene, ptr->flatMesh);

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
    
    // Initialize Node Graph
    terrain.nodeGraph = std::make_shared<TerrainNodesV2::TerrainNodeGraphV2>();

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
    
    // If stride == 1, simple copy; if downsampling needed, use Lanczos resampling for better quality
    bool downsampled = (stride > 1);
    bool high_input = (w >= 3000 || h >= 3000);

    if (stride == 1) {
        // Direct copy with sRGB->linear conversion. Apply light dithering
        // only on small/medium images to break banding; skip for very large inputs.
        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> dither_dist(-0.5f / 255.0f, 0.5f / 255.0f);
        const float gamma = 2.2f;
        bool apply_dither = !high_input; // disable dithering for very large (4k+) inputs
        for (int y = 0; y < new_h; y++) {
            for (int x = 0; x < new_w; x++) {
                int srcIdx = y * w + x;
                float v = (float)img[srcIdx] / 255.0f;
                // Assume common grayscale PNG is gamma-encoded; convert to linear
                v = powf(v, gamma);
                if (apply_dither) {
                    v = v + dither_dist(rng);
                }
                v = std::clamp(v, 0.0f, 1.0f);
                terrain.heightmap.data[y * new_w + x] = v;
            }
        }
    } else {
        // Use Lanczos resampler (single-channel 8-bit)
        std::vector<uint8_t> dstBuf((size_t)new_w * new_h);
        // Use a slightly smaller Lanczos window to reduce ringing on aggressive downsampling
        int lanczos_a = 2;
        ImageResample::lanczos_resample_u8(img, w, h, dstBuf.data(), new_w, new_h, lanczos_a);
        for (int i = 0; i < new_w * new_h; ++i) terrain.heightmap.data[i] = dstBuf[i] / 255.0f;
    }
    
    stbi_image_free(img);
    
    // Apply frequency-separation (high-quality): lowpass + detail restore
    try {
        std::vector<float> fs_out;
        float fs_sigma = 1.0f;
        float fs_detail_strength = 0.85f;
        if (downsampled) {
            // stronger lowpass when we downsample to remove aliasing
            fs_sigma = 1.4f;
            fs_detail_strength = 0.8f;
        } else if (high_input) {
            // for high-res native maps, be conservative
            fs_sigma = 0.6f;
            fs_detail_strength = 0.97f;
        }
        ImageFilters::frequency_separation(terrain.heightmap.data, new_w, new_h, fs_out, fs_sigma, fs_detail_strength);
        terrain.heightmap.data.swap(fs_out);
    } catch (...) {
        // If any failure, leave original data
    }

    // Remove isolated spikes (selective median) to avoid mesh 'peaks' at low-res
    // Compute global stddev and apply median only on low-variance maps to avoid removing real detail
    {
        double mean = 0.0, sq = 0.0;
        int N = new_w * new_h;
        for (int i = 0; i < N; ++i) { mean += terrain.heightmap.data[i]; sq += terrain.heightmap.data[i] * terrain.heightmap.data[i]; }
        mean /= (double)N;
        double var = sq / (double)N - mean * mean;
        double stddev = (var > 0.0) ? sqrt(var) : 0.0;
        if (stddev < 0.03 || downsampled) {
            remove_spikes_3x3(terrain.heightmap.data, new_w, new_h, 0.06f, 1.0f);
        }
    }
    
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
    ptr->original_heightmap_data = ptr->heightmap.data;
    
    // Initialize Layer System (Splat Map + 4 Layers)
    initLayers(ptr);

    // Add to Scene
    registerTerrainMeshOnce(scene, ptr->flatMesh);

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

float TerrainManager::sampleSplatChannel(float worldX, float worldZ, int channel) const {
    if (channel < 0 || channel > 3) return -1.0f;
    if (terrains.empty()) return -1.0f;

    for (const auto& terrain : terrains) {
        if (!terrain.splatMap) continue;
        auto& sp = terrain.splatMap;
        if (sp->pixels.empty() || sp->width <= 0 || sp->height <= 0) continue;

        // Transform world to local terrain space
        Vec3 localPos(worldX, 0.0f, worldZ);
        if (terrain.transform) {
            Matrix4x4 inv = terrain.transform->getFinal().inverse();
            localPos = inv.multiplyVector(Vec4(worldX, 0.0f, worldZ, 1.0f)).xyz();
        }

        float size = terrain.heightmap.scale_xz;
        if (localPos.x < 0.0f || localPos.x > size || localPos.z < 0.0f || localPos.z > size) continue;

        int w = sp->width;
        int h = sp->height;

        // Map local position to splatmap coordinates (note Y-flip used elsewhere)
        float gridX = (localPos.x / size) * (w - 1);
        float gridZ = (1.0f - (localPos.z / size)) * (h - 1);

        // Bilinear sample
        float fx = std::clamp(gridX, 0.0f, (float)(w - 1));
        float fz = std::clamp(gridZ, 0.0f, (float)(h - 1));
        int x0 = (int)std::floor(fx);
        int y0 = (int)std::floor(fz);
        int x1 = std::min(x0 + 1, w - 1);
        int y1 = std::min(y0 + 1, h - 1);
        float sx = fx - x0;
        float sy = fz - y0;

        auto samplePix = [&](int xx, int yy) -> float {
            const auto& p = sp->pixels[yy * w + xx];
            int v = 0;
            switch (channel) {
                case 0: v = p.r; break;
                case 1: v = p.g; break;
                case 2: v = p.b; break;
                case 3: v = p.a; break;
            }
            return (float)v / 255.0f;
        };

        float v00 = samplePix(x0, y0);
        float v10 = samplePix(x1, y0);
        float v01 = samplePix(x0, y1);
        float v11 = samplePix(x1, y1);

        float vx0 = v00 * (1.0f - sx) + v10 * sx;
        float vx1 = v01 * (1.0f - sx) + v11 * sx;
        float v = vx0 * (1.0f - sy) + vx1 * sy;
        return std::clamp(v, 0.0f, 1.0f);
    }

    return -1.0f;
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
    
    const float cell_size_x = size / (float)(std::max(1, w - 1));
    const float cell_size_z = size / (float)(std::max(1, h - 1));

    // Advance in ray-param units based on how quickly the ray traverses terrain cells.
    // The old fixed `cell_size_x * 0.5` step was not in ray-param space, so shallow-angle
    // rays could tunnel across edited regions and produce patchy sculpt hits after the
    // first terrain deformation.
    float step_tx = std::numeric_limits<float>::infinity();
    float step_tz = std::numeric_limits<float>::infinity();
    if (std::abs(ray_dir.x) > 1e-6f) {
        step_tx = cell_size_x / std::abs(ray_dir.x);
    }
    if (std::abs(ray_dir.z) > 1e-6f) {
        step_tz = cell_size_z / std::abs(ray_dir.z);
    }

    float step_dist = std::min(step_tx, step_tz) * 0.5f;
    if (!std::isfinite(step_dist) || step_dist <= 1e-6f) {
        step_dist = std::min(cell_size_x, cell_size_z) * 0.5f;
    }
    
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

void TerrainManager::updateTerrainMesh(TerrainObject* terrain, bool signalRebuild) {
    if (!terrain) return;

    // Node graph erosion operates on the heightmap first and finalizes geometry
    // once, on the main thread.  Do not let an intermediate erosion routine
    // create an orphan flatMesh (or mutate a live mesh from the worker thread).
    if (terrain->defer_mesh_updates) {
        terrain->dirty_mesh = true;
        return;
    }

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    float scale = terrain->heightmap.scale_xz;
    float max_h = terrain->heightmap.scale_y;

    // Rebuild topology whenever terrain resolution changes triangle count.
    const size_t required_triangle_count = (size_t)(std::max(0, w - 1)) * (size_t)(std::max(0, h - 1)) * 2ull;
    bool create_new = !terrain->flatMesh || terrain->flatMesh->num_triangles() != required_triangle_count;

    // Calculate step to preserve aspect ratio

    // Calculate step to Stretch to Fit (Square Terrain)
    // Use independent X and Z steps to ensure terrain fills scale_xz * scale_xz area
    float step_x = scale / (float)(std::max(1, w - 1));
    float step_z = scale / (float)(std::max(1, h - 1));

    // Pre-calculate vertices for grid
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    positions.resize((size_t)w * (size_t)h);
    normals.resize((size_t)w * (size_t)h);

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

    // 3. Build or update the flat (SoA) mesh
    if (create_new) {
        // Standard 0..1 UV mapping, one welded vertex per grid cell.
        float divW = (float)(w > 1 ? w - 1 : 1);
        float divH = (float)(h > 1 ? h - 1 : 1);
        std::vector<Vec2> uvs((size_t)w * (size_t)h);
        #pragma omp parallel for
        for (int z = 0; z < h; z++) {
            for (int x = 0; x < w; x++) {
                uvs[z * w + x] = Vec2((float)x / divW, (float)z / divH);
            }
        }

        std::vector<uint32_t> indices(required_triangle_count * 3ull);
        #pragma omp parallel for collapse(2)
        for (int z = 0; z < (h - 1); z++) {
            for (int x = 0; x < (w - 1); x++) {
                size_t tri_idx = (static_cast<size_t>(z) * (w - 1) + x) * 2;

                uint32_t i0 = z * w + x;
                uint32_t i1 = z * w + (x + 1);
                uint32_t i2 = (z + 1) * w + (x + 1);
                uint32_t i3 = (z + 1) * w + x;

                size_t o1 = tri_idx * 3ull;
                indices[o1 + 0] = i0; indices[o1 + 1] = i2; indices[o1 + 2] = i1;
                size_t o2 = (tri_idx + 1) * 3ull;
                indices[o2 + 0] = i0; indices[o2 + 1] = i3; indices[o2 + 2] = i2;
            }
        }

        terrain->flatMesh = gridToFlatMesh(positions, normals, uvs, indices,
                                            terrain->material_id, terrain->transform,
                                            terrain->name + "_Chunk");
        if (terrain->flatMesh) terrain->flatMesh->terrain_id = terrain->id;
    }
    else {
        // In-place update: same topology, only heights (and derived normals) changed.
        auto& geo = *terrain->flatMesh->geometry;
        Vec3* P  = geo.get_attribute_data_mut<Vec3>("P");
        Vec3* N  = geo.get_attribute_data_mut<Vec3>("N");
        Vec3* Po = geo.get_attribute_data_mut<Vec3>("P_orig");
        Vec3* No = geo.get_attribute_data_mut<Vec3>("N_orig");

        Matrix4x4 finalT = Matrix4x4::identity();
        Matrix4x4 normalT = Matrix4x4::identity();
        if (terrain->transform) {
            finalT = terrain->transform->getFinal();
            normalT = terrain->transform->getNormalTransform();
        }

        #pragma omp parallel for
        for (int z = 0; z < h; z++) {
            for (int x = 0; x < w; x++) {
                size_t vIdx = (size_t)z * w + x;
                const Vec3& lp = positions[vIdx];
                const Vec3& ln = normals[vIdx];
                if (Po) Po[vIdx] = lp;
                if (No) No[vIdx] = ln;
                if (P)  P[vIdx]  = finalT.transform_point(lp);
                if (N)  N[vIdx]  = normalT.transform_vector(ln).normalize();
            }
        }
    }

    // Publish graph-derived named terrain fields only after the canonical mesh
    // exists and on the main-thread finalize path. This keeps worker evaluation
    // height/data-only while giving foliage a zero-conversion vertex-field bridge.
    if (terrain->flatMesh && terrain->flatMesh->geometry) {
        auto& geometry = *terrain->flatMesh->geometry;
        static const std::array<const char*, 22> knownTerrainFields = {
            "terrain.slope", "terrain.concavity", "terrain.convexity",
            "terrain.valley", "terrain.wetness", "biome.forest",
            "biome.grass", "biome.rock", "biome.alpine",
            "hydrology.accumulation", "hydrology.direction", "hydrology.basins",
            "hydrology.channels", "hydrology.stream_order", "hydrology.sources",
            "hydrology.river_bed", "hydrology.lake_mask", "hydrology.lake_depth",
            "hydrology.lake_level", "hydrology.lake_shoreline",
            "hydrology.lake_spill", "hydrology.lake_id"
        };
        const size_t vertexCount = geometry.get_vertex_count();
        for (const char* fieldName : knownTerrainFields) {
            const auto fieldIt = terrain->analysisFields.find(fieldName);
            const bool valid = fieldIt != terrain->analysisFields.end() &&
                fieldIt->second && fieldIt->second->size() == vertexCount;
            if (!valid) {
                geometry.remove_custom_attribute(fieldName);
                continue;
            }
            if (!geometry.has_attribute(fieldName)) geometry.add_attribute<float>(fieldName);
            float* destination = geometry.get_attribute_data_mut<float>(fieldName);
            if (destination) {
                std::copy(fieldIt->second->begin(), fieldIt->second->end(), destination);
            }
        }
    }

    // Clear dirty regions after full update
    terrain->dirty_region.clear();

    // [CPU/GPU REBUILD FIX] Ensure changes are uploaded to backend
    if (signalRebuild) {
        extern bool g_bvh_rebuild_pending;
        extern bool g_optix_rebuild_pending;
        extern bool g_vulkan_rebuild_pending;
        extern bool g_viewport_raster_rebuild_pending;
        g_bvh_rebuild_pending = true;
        g_optix_rebuild_pending = true;
        g_viewport_raster_rebuild_pending = true;
        if (terrainRenderBackendIsVulkan()) g_vulkan_rebuild_pending = true;
    }
}

void TerrainManager::rebuildTerrainMesh(SceneData& scene, TerrainObject* terrain) {
    if (!terrain) return;

    // 1. Remove every old registration. This also repairs scenes loaded by the
    // former deserialize path which could contain the same mesh twice.
    auto& objs = scene.world.objects;
    if (terrain->flatMesh) {
        Hittable* old_mesh = terrain->flatMesh.get();
        objs.erase(std::remove_if(objs.begin(), objs.end(),
            [&](const std::shared_ptr<Hittable>& object) {
                return object.get() == old_mesh;
            }), objs.end());
    }

    // 2. Drop the old mesh so updateTerrainMesh() treats this as a full rebuild
    terrain->flatMesh.reset();

    // 3. Re-generate mesh
    updateTerrainMesh(terrain);

    // 4. Add the new flat mesh to scene
    registerTerrainMeshOnce(scene, terrain->flatMesh);

    // 5. Flag for rebuild
    extern bool g_bvh_rebuild_pending;
    extern bool g_optix_rebuild_pending;
    extern bool g_vulkan_rebuild_pending;
    extern bool g_viewport_raster_rebuild_pending;
    g_bvh_rebuild_pending = true;
    g_optix_rebuild_pending = true;
    g_viewport_raster_rebuild_pending = true;
    if (terrainRenderBackendIsVulkan()) g_vulkan_rebuild_pending = true;
}

// ===========================================================================
// INCREMENTAL SECTOR UPDATE (For performance optimization)
// ===========================================================================
void TerrainManager::updateDirtySectors(TerrainObject* terrain, bool clearRegion) {
    if (!terrain || !terrain->dirty_region.has_any_dirty || !terrain->flatMesh) return;

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    float scale = terrain->heightmap.scale_xz;
    float max_h = terrain->heightmap.scale_y;
    float step_x = scale / (float)(w - 1);
    float step_z = scale / (float)(h - 1);

    int sector_w = w / DirtyRegion::SECTOR_GRID_SIZE;
    int sector_h = h / DirtyRegion::SECTOR_GRID_SIZE;

    auto& geo = *terrain->flatMesh->geometry;
    Vec3* P  = geo.get_attribute_data_mut<Vec3>("P");
    Vec3* N  = geo.get_attribute_data_mut<Vec3>("N");
    Vec3* Po = geo.get_attribute_data_mut<Vec3>("P_orig");
    Vec3* No = geo.get_attribute_data_mut<Vec3>("N_orig");

    Matrix4x4 finalT = Matrix4x4::identity();
    Matrix4x4 normalT = Matrix4x4::identity();
    if (terrain->transform) {
        finalT = terrain->transform->getFinal();
        normalT = terrain->transform->getNormalTransform();
    }

    // Process each dirty sector in parallel, writing directly into the flat SoA
    // vertex arrays (one write per touched vertex — no per-quad corner duplication,
    // and no per-triangle facade lookups).
    #pragma omp parallel for
    for (int combined = 0; combined < DirtyRegion::SECTOR_GRID_SIZE * DirtyRegion::SECTOR_GRID_SIZE; combined++) {
        int sx = combined % DirtyRegion::SECTOR_GRID_SIZE;
        int sy = combined / DirtyRegion::SECTOR_GRID_SIZE;

        if (!terrain->dirty_region.sectors[sx][sy]) continue;

        // Calculate sector bounds (quads); the vertex range touched by these
        // quads is [startX, endX] x [startZ, endZ] inclusive (quad x touches
        // vertices x and x+1).
        int startX = sx * sector_w;
        int startZ = sy * sector_h;
        int endX = std::min(startX + sector_w, w - 1);
        int endZ = std::min(startZ + sector_h, h - 1);

        for (int z = startZ; z <= endZ; z++) {
            for (int x = startX; x <= endX; x++) {
                size_t vIdx = (size_t)z * w + x;
                Vec3 lp(x * step_x, terrain->heightmap.data[vIdx] * max_h, z * step_z);
                Vec3 ln = calculateNormal(terrain, x, z);

                if (Po) Po[vIdx] = lp;
                if (No) No[vIdx] = ln;
                if (P)  P[vIdx]  = finalT.transform_point(lp);
                if (N)  N[vIdx]  = normalT.transform_vector(ln).normalize();
            }
        }
    }

    if (clearRegion) {
        terrain->dirty_region.clear();
    }
}

// mode: 0=Raise, 1=Lower, 2=Flatten, 3=Smooth, 4=Stamp
void TerrainManager::sculpt(TerrainObject* terrain, const Vec3& hitPoint, int mode, float radius, float strength, float dt,
                            float curve, float targetHeight, std::shared_ptr<Texture> stampTexture, float rotation,
                            bool signalHeavyRebuild) {
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
    
    bool changed = false;
    float brush_delta_world = strength * dt * 2.0f;
    float scaleY = terrain->heightmap.scale_y;
    if (scaleY < 0.001f) scaleY = 1.0f;
    float brush_strength_frame = brush_delta_world / scaleY;
    curve = std::clamp(curve, 0.25f, 4.0f);
    
    // Smooth kernel pre-calculation or on-the-fly?
    // Using on-the-fly for dynamic updates

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
            falloff = std::pow(std::clamp(falloff, 0.0f, 1.0f), curve);
            
            float val = terrain->heightmap.data[y * w + x];
            
            if (mode == 0) { // Raise
                val += brush_strength_frame * falloff;
            } else if (mode == 1) { // Lower
                val -= brush_strength_frame * falloff;
            } else if (mode == 2) { // Flatten
                // Lerp towards target
                float maxDelta = brush_strength_frame * falloff;
                float delta = normalizedTarget - val;
                delta = std::clamp(delta, -maxDelta, maxDelta);
                float t = brush_strength_frame * falloff; // Influence
                if (t > 1.0f) t = 1.0f;
                val = val + delta * t;
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
                float maxDelta = brush_strength_frame * falloff;
                float delta = std::clamp(avg - val, -maxDelta, maxDelta);
                val = val + delta * t;

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
                         // Bilinear texture sampling for high-fidelity stamp edges
                         Vec3 color = stampTexture->get_color_bilinear(u, v);
                         float texVal = 0.299f * color.x + 0.587f * color.y + 0.114f * color.z;
                         texVal = std::clamp(texVal, 0.0f, 1.0f);

                         float stampMask = texVal * falloff;
                         float stampDelta = (stampMask * brush_delta_world) / scaleY;
                         val += stampDelta;
                     }
                }
            }
            
            // Clamp
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            
            terrain->heightmap.data[y * w + x] = val;
            terrain->markCellDirty(x, y); // Mark for partial mesh update
            changed = true;
        }
    }
    
    if (changed) {
        // If brush is very large, full update is faster than many sectors
        if (r_pixels > (w / 8)) {
            updateTerrainMesh(terrain, signalHeavyRebuild);
        } else {
            updateDirtySectors(terrain, false);
        }
        terrain->dirty_mesh = signalHeavyRebuild;
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
        // Resolution: Match heightmap but at least 512
        int sW = std::max(512, terrain->heightmap.width);
        int sH = std::max(512, terrain->heightmap.height);

        // Create Splat Map (RGBA) using procedural constructor (no disk load)
        terrain->splatMap = std::make_shared<Texture>("SplatMap_" + terrain->name, sW, sH, TextureType::Unknown);
        
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
        terrain->layers.resize(4, nullptr);
        terrain->layer_uv_scales.resize(4, 50.0f); // Default tiling
        
        static const char* defLayerNames[4] = {"Grass", "Rock", "Snow", "Flow"};
        static const Vec3 defLayerColors[4] = {
            Vec3(0.3f, 0.5f, 0.2f),  // Grass
            Vec3(0.4f, 0.4f, 0.4f),  // Rock
            Vec3(0.9f, 0.9f, 0.95f), // Snow
            Vec3(0.5f, 0.35f, 0.2f)  // Flow
        };

        for (int i = 0; i < 4; ++i) {
            std::string matName = terrain->name + "_" + defLayerNames[i];
            
            // Check if material already exists in manager (e.g. from previous run or load)
            if (MaterialManager::getInstance().hasMaterial(matName)) {
                terrain->layers[i] = MaterialManager::getInstance().getMaterialShared(
                    MaterialManager::getInstance().getMaterialID(matName)
                );
            } else {
                // Create default PBR material
                auto mat = std::make_shared<PrincipledBSDF>(defLayerColors[i], 0.8f, 0.0f);
                mat->materialName = matName;
                MaterialManager::getInstance().addMaterial(matName, mat);
                terrain->layers[i] = mat;
            }
        }
    }
}

// Resize splatmap to match heightmap dimensions (bilinear interpolation)
void TerrainManager::resizeSplatMap(TerrainObject* terrain) {
    if (!terrain || !terrain->splatMap) return;
    
    int targetW = std::max(512, terrain->heightmap.width);
    int targetH = std::max(512, terrain->heightmap.height);
    
    int srcW = terrain->splatMap->width;
    int srcH = terrain->splatMap->height;
    
    const size_t targetPixelCount = static_cast<size_t>(targetW) * static_cast<size_t>(targetH);
    const bool sourceStorageValid =
        srcW > 0 && srcH > 0 &&
        terrain->splatMap->pixels.size() ==
            static_cast<size_t>(srcW) * static_cast<size_t>(srcH);

    // Metadata alone is not sufficient: a resolution transition or interrupted
    // import can leave width/height current while the CPU pixel vector still has
    // the old allocation.
    if (srcW == targetW && srcH == targetH &&
        terrain->splatMap->pixels.size() == targetPixelCount) return;
    
    SCENE_LOG_INFO("[TerrainManager] Resizing splatmap from " + std::to_string(srcW) + "x" + std::to_string(srcH) + 
                   " to " + std::to_string(targetW) + "x" + std::to_string(targetH));
    
    // Create new pixel buffer. If the old metadata/storage pair is inconsistent,
    // it cannot be sampled safely; reset to the base layer instead of indexing an
    // unknown layout.
    std::vector<CompactVec4> newPixels(targetPixelCount);
    if (!sourceStorageValid) {
        for (auto& p : newPixels) {
            p.r = 255;
            p.g = 0;
            p.b = 0;
            p.a = 0;
        }
    }
    
    // Bilinear interpolation
    for (int y = 0; sourceStorageValid && y < targetH; ++y) {
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
    
    // If dimensions changed, we MUST re-allocate
    // Texture class handles updateGPU by memcpy if allocated, but we resized pixels vector.
    // If we just resized, the GPU array size is wrong.
    
    // Check if we need full re-allocation
    // We can assume that if we are here, dimensions likely changed (due to check at top of function).
    // However, importSplatMap manipulates pixels then calls this.
    
    // Force re-allocation on GPU to match new size
    terrain->splatMap->cleanup_gpu();
    terrain->splatMap->upload_to_gpu();
    
    // Rebuild SBT only if dimensions changed significantly enough to require new texture handle
    // Since we destroyed the handle, we MUST update SBT.
    if (g_hasOptix) {
        g_optix_rebuild_pending = true;
    }
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
    float gridZ = (1.0f - (localPos.z / size)) * (h - 1);
    
    int cx = (int)(gridX + 0.5f);
    int cy = (int)(gridZ + 0.5f);
    
    // Radius in pixels
    int r_pixels = (int)((radius / size) * (w - 1));
    if (r_pixels < 1) r_pixels = 1;

    float brush_strength_frame = strength * dt * 5.0f;
    bool changed = false;

    // Track modified region for partial GPU upload
    int minX = w, minY = h, maxX = 0, maxY = 0;

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
            
            minX = std::min(minX, x); minY = std::min(minY, y);
            maxX = std::max(maxX, x); maxY = std::max(maxY, y);
            changed = true;
        }
    }

    if (changed) {
        // Optimization: Partial GPU upload
        int regionW = maxX - minX + 1;
        int regionH = maxY - minY + 1;
        if (regionW > (w / 2) || regionH > (h / 2)) {
             terrain->splatMap->updateGPU(); // Full update if large
        } else {
             terrain->splatMap->upload_region_to_gpu(minX, minY, regionW, regionH);
        }
    }
}

void TerrainManager::autoMask(TerrainObject* terrain, float slopeWeight, float heightWeight, float heightMin, float heightMax, float slopeSteepness) {
    if (!terrain || !terrain->splatMap) return;

    // Height/flow/erosion vectors are replaced by terrain-node evaluation. Do not
    // read them until the worker result has completed its main-thread finalize.
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync()) {
        SCENE_LOG_WARN("[autoMask] Terrain node evaluation is still pending; mask generation skipped.");
        return;
    }

    const int hmW = terrain->heightmap.width;
    const int hmH = terrain->heightmap.height;
    if (hmW < 2 || hmH < 2) return;

    const size_t heightPixelCount = static_cast<size_t>(hmW) * static_cast<size_t>(hmH);
    if (terrain->heightmap.data.size() != heightPixelCount) {
        SCENE_LOG_WARN("[autoMask] Heightmap dimensions do not match its CPU buffer; mask generation skipped.");
        return;
    }

    // Resolution changes must update both texture metadata and its CPU storage
    // before the y*w+x write loop below.
    resizeSplatMap(terrain);
    if (terrain->splatMap->width <= 0 || terrain->splatMap->height <= 0 ||
        terrain->splatMap->pixels.size() !=
            static_cast<size_t>(terrain->splatMap->width) * terrain->splatMap->height) {
        SCENE_LOG_WARN("[autoMask] Splat map storage is invalid after resize; mask generation skipped.");
        return;
    }
    
    // A non-empty map can still belong to the previous terrain resolution.
    if (terrain->flowMap.size() != heightPixelCount) {
        calculateFlowMap(terrain);
    }

    const bool hasFlowMap = terrain->flowMap.size() == heightPixelCount;

    int w = terrain->splatMap->width;
    int h = terrain->splatMap->height;
    float max_h = terrain->heightmap.scale_y;
    float scale = terrain->heightmap.scale_xz;
    
    SCENE_LOG_INFO("[autoMask] SplatMap: " + std::to_string(w) + "x" + std::to_string(h) + 
                   ", Heightmap: " + std::to_string(terrain->heightmap.width) + "x" + std::to_string(terrain->heightmap.height) +
                   ", scale_xz=" + std::to_string(scale) + ", scale_y=" + std::to_string(max_h));

    const bool hasErosionMap =
        terrain->erosionMapRGBA.size() == heightPixelCount * 4;
    
    // Calculate cell size for correct slope calculation (Rise / Run)
    // Run = Distance between HEIGHTMAP pixels in world space (since we sample neighbors in heightmap)
    float hmCellSizeX = scale / (float)(std::max(1, terrain->heightmap.width - 1));
    float hmCellSizeZ = scale / (float)(std::max(1, terrain->heightmap.height - 1));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // Sample height and slope at this UV
            // Standard 0..1 Mapping (Aligns with Mesh UVs)
            float u = (float)x / (float)(w > 1 ? w - 1 : 1);
            float v = 1.0f - (float)y / (float)(h > 1 ? h - 1 : 1);
            
            // Map to heightmap coords (use fractional coords and bilinear sampling)
            float hx_f = u * (terrain->heightmap.width - 1);
            float hy_f = v * (terrain->heightmap.height - 1);
            int w_hm = terrain->heightmap.width;
            int h_hm = terrain->heightmap.height;

            // Bilinear sample helper (samples using world-scaled heights via getHeight)
            auto sampleBilinear = [&](float fx, float fy) -> float {
                fx = std::clamp(fx, 0.0f, (float)(w_hm - 1));
                fy = std::clamp(fy, 0.0f, (float)(h_hm - 1));
                int x0 = (int)floorf(fx);
                int x1 = std::min(x0 + 1, w_hm - 1);
                int y0 = (int)floorf(fy);
                int y1 = std::min(y0 + 1, h_hm - 1);
                float sx = fx - (float)x0;
                float sy = fy - (float)y0;
                float h00 = terrain->heightmap.getHeight(x0, y0);
                float h10 = terrain->heightmap.getHeight(x1, y0);
                float h01 = terrain->heightmap.getHeight(x0, y1);
                float h11 = terrain->heightmap.getHeight(x1, y1);
                float hx0 = h00 * (1.0f - sx) + h10 * sx;
                float hx1 = h01 * (1.0f - sx) + h11 * sx;
                return hx0 * (1.0f - sy) + hx1 * sy;
            };

            // Keep integer indices for erosion map indexing where needed
            int hx = (int)std::clamp((int)std::lround(hx_f), 0, w_hm - 1);
            int hy = (int)std::clamp((int)std::lround(hy_f), 0, h_hm - 1);

            // Sample height using bilinear interpolation for higher detail
            float local_height = sampleBilinear(hx_f, hy_f);

            // Calculate Slope using bilinear samples offset by one texel
            float hl = sampleBilinear(hx_f - 1.0f, hy_f);
            float hr = sampleBilinear(hx_f + 1.0f, hy_f);
            float hu = sampleBilinear(hx_f, hy_f - 1.0f);
            float hd = sampleBilinear(hx_f, hy_f + 1.0f);
            
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
            // Fix: Use local_height so mask moves with terrain
            float w_snow = smoothstep(heightMin, heightMax, local_height); 
            float w_rest = 1.0f - w_rock; // What remains for flat
            
            // Base layer gets the rest, masked by snow
            float w_grass = w_rest * (1.0f - w_snow);
            
            // Snow covers everything implies max priority?
            // Usually Key: Rock overrides Grass. Snow overrides everything.
            // Let's blend.
            
            float final_R = w_grass; // Grass
            float final_G = w_rock * (1.0f - w_snow); // Rock
            float final_B = w_snow; // Snow
            float final_A = 0.0f; // Dirt/Flow Mask

            float erosionWear = 0.0f;
            float erosionDeposit = 0.0f;
            float erosionFlow = 0.0f;
            float erosionInfluence = 1.0f;
            if (hasErosionMap) {
                size_t eIdx = static_cast<size_t>(hy * terrain->heightmap.width + hx) * 4;
                erosionWear = std::clamp(terrain->erosionMapRGBA[eIdx + 0], 0.0f, 1.0f);
                erosionDeposit = std::clamp(terrain->erosionMapRGBA[eIdx + 1], 0.0f, 1.0f);
                erosionFlow = std::clamp(terrain->erosionMapRGBA[eIdx + 2], 0.0f, 1.0f);
                erosionInfluence = std::clamp(terrain->erosionMapRGBA[eIdx + 3], 0.0f, 1.0f);
            }
            
            if (hasFlowMap) {
                float flowVal = terrain->flowMap[hy * terrain->heightmap.width + hx];
                
                // --- GAEA STYLE HIERARCHY ---
                // 1. Threshold
                float flowNorm = fmaxf(0.0f, flowVal - terrain->am_flow_threshold); 
                
                // 2. Linear catch for all flow features
                float tributaries = 1.0f - expf(-flowNorm * 0.4f); 
                
                // 3. Blend: Main trunks get most of the white value
                // Hierarchical blend: Tributaries populate the 'base' flow while main trunks dominate the center.
                // Tributaries (fast catch): 0.6f
                // Main Trunks (slow ramp): 0.005f (Was 0.5f, which was too aggressive)
                float mainTrunks = 1.0f - expf(-flowNorm * 0.005f); 
                final_A = (tributaries * 0.4f + mainTrunks * 0.6f);
                
                // 4. Max weight & Slope mask
                // REDUCED slope influence (0.05 instead of 0.15) so flow map stays bright even on steeper riverbanks.
                final_A = std::clamp(final_A, 0.0f, 1.0f);
                final_A *= (1.0f - w_rock * 0.05f); 
            }

            if (hasErosionMap) {
                final_R *= (1.0f + erosionDeposit * 0.35f);
                final_R *= (1.0f - erosionWear * 0.45f);

                final_G *= (1.0f + erosionWear * 0.40f);
                final_G *= (1.0f - erosionDeposit * 0.20f);

                final_B *= (1.0f - erosionFlow * 0.25f);
                final_B *= (1.0f - erosionWear * 0.15f);

                float erosionAlphaBoost = erosionFlow * (0.65f + erosionWear * 0.35f);
                final_A = std::max(final_A, erosionAlphaBoost);
                final_A *= (0.85f + erosionInfluence * 0.15f);
                final_A = std::clamp(final_A, 0.0f, 1.0f);
            }

    // Normalize with a 'soft' blend for the alpha channel
            // Instead of full normalization which erases other layers, 
            // we let Alpha act more as an overlay weight
            float baseSum = final_R + final_G + final_B;
            if (baseSum > 0.001f) {
                float mult = (1.0f - final_A);
                final_R = (final_R / baseSum) * mult;
                final_G = (final_G / baseSum) * mult;
                final_B = (final_B / baseSum) * mult;
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

void TerrainManager::importSplatMap(TerrainObject* terrain, const std::string& filepath) {
    if (!terrain) return;
    if (!terrain->splatMap) initLayers(terrain);
    if (!terrain->splatMap) return;
    
    int w, h, channels;
    unsigned char* img = stbi_load(filepath.c_str(), &w, &h, &channels, 4);
    if (!img) {
        SCENE_LOG_ERROR("Failed to load splatmap image: " + filepath);
        return;
    }
    
    terrain->splatMap->width = w;
    terrain->splatMap->height = h;
    terrain->splatMap->pixels.resize(static_cast<size_t>(w) * h);
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int srcIdx = y * w + x;
            const int dstIdx = ((h - 1) - y) * w + x;
            terrain->splatMap->pixels[dstIdx].r = img[srcIdx * 4 + 0];
            terrain->splatMap->pixels[dstIdx].g = img[srcIdx * 4 + 1];
            terrain->splatMap->pixels[dstIdx].b = img[srcIdx * 4 + 2];
            terrain->splatMap->pixels[dstIdx].a = img[srcIdx * 4 + 3];
        }
    }
    
    stbi_image_free(img);
    terrain->splatMap->m_is_loaded = true;
    terrain->splatMap->markVulkanDirtyFull();
    
    // Ensure it matches terrain dimensions
    resizeSplatMap(terrain);
    
    // Upload to GPU
    // If resizeSplatMap caused a re-allocation (size change), it already triggered SBT rebuild.
    // If size didn't change (resizeSplatMap returned early), we just need to update content.
    if (terrain->splatMap->is_gpu_uploaded) {
        // Just update content (Fast path)
        terrain->splatMap->updateGPU();
    } else {
        // First upload
        terrain->splatMap->upload_to_gpu();
        if (g_hasOptix) g_optix_rebuild_pending = true;
    }
    
    SCENE_LOG_INFO("[TerrainManager] Imported splat map for terrain: " + terrain->name + " (" + std::to_string(w) + "x" + std::to_string(h) + ")");
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
            rgba[dst_idx * 4 + 3] = p.a;
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

void TerrainManager::hydraulicErosion(TerrainObject* terrain, const HydraulicErosionParams& p, const std::vector<float>& mask, const std::function<void(float)>& progressCallback) {
    if (!terrain) return;
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    auto& data = terrain->heightmap.data;
    float cellSize = terrain->heightmap.scale_xz / w;
    bool hasHardness = !terrain->hardnessMap.empty() && terrain->hardnessMap.size() == (size_t)(w * h);
    bool hasMask = !mask.empty() && mask.size() == (size_t)(w * h);
    
    // EDGE PRESERVATION: Save original heights before erosion
    std::vector<float> originalHeights = data;
    int fadeWidth = getEdgeFadeWidth(terrain);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

    auto sampleHeight = [&](float x, float y) -> float {
        x = std::clamp(x, 0.0f, (float)(w - 1));
        y = std::clamp(y, 0.0f, (float)(h - 1));
        int x0 = (int)x;
        int y0 = (int)y;
        int x1 = std::min(x0 + 1, w - 1);
        int y1 = std::min(y0 + 1, h - 1);
        float tx = x - x0;
        float ty = y - y0;
        float v00 = data[y0 * w + x0];
        float v10 = data[y0 * w + x1];
        float v01 = data[y1 * w + x0];
        float v11 = data[y1 * w + x1];
        float a = v00 + (v10 - v00) * tx;
        float b = v01 + (v11 - v01) * tx;
        return a + (b - a) * ty;
    };

    auto sampleMap = [&](const std::vector<float>& src, float x, float y, float defaultValue) -> float {
        if (src.empty()) return defaultValue;
        x = std::clamp(x, 0.0f, (float)(w - 1));
        y = std::clamp(y, 0.0f, (float)(h - 1));
        int x0 = (int)x;
        int y0 = (int)y;
        int x1 = std::min(x0 + 1, w - 1);
        int y1 = std::min(y0 + 1, h - 1);
        float tx = x - x0;
        float ty = y - y0;
        float v00 = src[y0 * w + x0];
        float v10 = src[y0 * w + x1];
        float v01 = src[y1 * w + x0];
        float v11 = src[y1 * w + x1];
        float a = v00 + (v10 - v00) * tx;
        float b = v01 + (v11 - v01) * tx;
        return a + (b - a) * ty;
    };

    auto updateHeight = [&](float x, float y, float change) {
        int x0 = (int)x;
        int y0 = (int)y;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        if (x0 < 0 || x0 >= w || y0 < 0 || y0 >= h) return;
        if (x1 < 0 || x1 >= w || y1 < 0 || y1 >= h) return;

        float tx = x - x0;
        float ty = y - y0;
        data[y0 * w + x0] += change * (1.0f - tx) * (1.0f - ty);
        data[y0 * w + x1] += change * tx * (1.0f - ty);
        data[y1 * w + x0] += change * (1.0f - tx) * ty;
        data[y1 * w + x1] += change * tx * ty;
    };

    auto erodeWithBrush = [&](float x, float y, float amount) {
        if (amount <= 0.0f) return;
        if (p.erosionRadius <= 1) {
            updateHeight(x, y, -amount);
            return;
        }

        int minX = std::max(0, (int)std::floor(x) - p.erosionRadius);
        int maxX = std::min(w - 1, (int)std::floor(x) + p.erosionRadius);
        int minY = std::max(0, (int)std::floor(y) - p.erosionRadius);
        int maxY = std::min(h - 1, (int)std::floor(y) + p.erosionRadius);

        float weightSum = 0.0f;
        for (int iy = minY; iy <= maxY; ++iy) {
            for (int ix = minX; ix <= maxX; ++ix) {
                float dx = ((float)ix + 0.5f) - x;
                float dy = ((float)iy + 0.5f) - y;
                float dist = std::sqrt(dx * dx + dy * dy);
                if (dist > (float)p.erosionRadius) continue;
                float t = 1.0f - dist / std::max(1.0f, (float)p.erosionRadius);
                weightSum += t * t * (3.0f - 2.0f * t);
            }
        }

        if (weightSum <= 1e-6f) {
            updateHeight(x, y, -amount);
            return;
        }

        for (int iy = minY; iy <= maxY; ++iy) {
            for (int ix = minX; ix <= maxX; ++ix) {
                float dx = ((float)ix + 0.5f) - x;
                float dy = ((float)iy + 0.5f) - y;
                float dist = std::sqrt(dx * dx + dy * dy);
                if (dist > (float)p.erosionRadius) continue;
                float t = 1.0f - dist / std::max(1.0f, (float)p.erosionRadius);
                float weight = t * t * (3.0f - 2.0f * t);
                data[iy * w + ix] -= amount * (weight / weightSum);
            }
        }
    };
    
    // Reported every 1% of iterations rather than every iteration — this loop
    // runs up to hundreds of thousands of times, and a std::function call per
    // droplet would be measurable overhead for no visible UI benefit.
    const int progressStep = std::max(1, p.iterations / 100);
    for (int iter = 0; iter < p.iterations; iter++) {
        if (progressCallback && (iter % progressStep) == 0) {
            progressCallback(static_cast<float>(iter) / static_cast<float>(p.iterations));
        }
        float posX = distrib(gen) * (w - 1);
        float posY = distrib(gen) * (h - 1);
        float dirX = 0.0f, dirY = 0.0f;
        float speed = 1.0f;
        float water = 1.0f;
        float sediment = 0.0f;
        float invCellSize = 1.0f / std::max(0.001f, cellSize);

        for (int lifetime = 0; lifetime < p.dropletLifetime; lifetime++) {
            int nodeX = (int)posX;
            int nodeY = (int)posY;
            if (nodeX <= 1 || nodeX >= w - 2 || nodeY <= 1 || nodeY >= h - 2) break;
            int gridIdx = nodeY * w + nodeX;

            float h00 = data[gridIdx];
            float h10 = data[gridIdx + 1];
            float h01 = data[gridIdx + w];
            float h11 = data[gridIdx + w + 1];

            float u = posX - nodeX;
            float v = posY - nodeY;

            float gradX = ((h10 - h00) * (1.0f - v) + (h11 - h01) * v) * invCellSize;
            float gradY = ((h01 - h00) * (1.0f - u) + (h11 - h10) * u) * invCellSize;

            dirX = dirX * p.inertia - gradX * (1.0f - p.inertia);
            dirY = dirY * p.inertia - gradY * (1.0f - p.inertia);

            float len = std::sqrt(dirX * dirX + dirY * dirY);
            if (len < 1e-6f) {
                dirX = distrib(gen) * 0.2f - 0.1f;
                dirY = distrib(gen) * 0.2f - 0.1f;
            } else {
                dirX /= len;
                dirY /= len;
            }
            
            float newPosX = posX + dirX;
            float newPosY = posY + dirY;
            
            if (newPosX <= 1.0f || newPosX >= w - 2.0f || newPosY <= 1.0f || newPosY >= h - 2.0f) break;

            float oldH = sampleHeight(posX, posY);
            float newH = sampleHeight(newPosX, newPosY);
            float deltaHeight = newH - oldH;
            float localSlope = -deltaHeight * invCellSize;
            float capacity = fmaxf(localSlope * speed * water * p.sedimentCapacity * sqrtf(fmaxf(0.0f, localSlope) + 0.001f), p.minSlope);
            
            float hardness = hasHardness ? sampleMap(terrain->hardnessMap, posX, posY, 0.0f) : 0.0f;
            float hardnessFactor = 1.0f - hardness * 0.9f;
            float maskValue = hasMask ? sampleMap(mask, posX, posY, 1.0f) : 1.0f;
            if (maskValue < 0.001f) {
                if (sediment > 0.0f) {
                    updateHeight(posX, posY, sediment * 0.5f);
                }
                break;
            }
            
            float downhillSlope = std::max(localSlope, 0.0f);
            float flatness = 1.0f - std::min(1.0f, downhillSlope / std::max(p.minSlope * 8.0f, 0.001f));
            float speedBeforePhysics = speed;

            if (deltaHeight > 0) {
                if (speed > 0.5f && sediment < capacity) {
                    float erodeAmount = fminf(deltaHeight * 0.3f, oldH * 0.05f);
                    erodeAmount *= hardnessFactor * maskValue;
                    erodeWithBrush(newPosX, newPosY, erodeAmount);
                    sediment += erodeAmount;
                    speed *= 0.6f;
                } else {
                    float deposit = fminf(sediment * 0.3f, deltaHeight * 0.5f);
                    updateHeight(posX, posY, deposit);
                    sediment -= deposit;
                    speed = 0.0f;
                }
            }
            else if (sediment > capacity) {
                float deposit = (sediment - capacity) * p.depositSpeed;
                float flatDepositCap = sediment * flatness * 0.35f;
                deposit = fminf(deposit, std::max(-deltaHeight * 0.5f, flatDepositCap));
                updateHeight(posX, posY, deposit);
                sediment -= deposit;
            }
            else {
                float erode = fminf((capacity - sediment) * p.erodeSpeed, -deltaHeight);
                erode = fminf(erode, oldH * 0.05f);
                erode *= hardnessFactor * maskValue;
                erodeWithBrush(posX, posY, erode);
                sediment += erode;
            }

            if (deltaHeight <= 0.0f && sediment > 0.0f && flatness > 0.0f) {
                float slowFactor = 1.0f - std::min(speed, 1.0f);
                float settlingRate = (0.04f + p.depositSpeed * 0.25f) * flatness * (0.35f + 0.65f * slowFactor);
                float settlingAmount = std::min(sediment, sediment * settlingRate);
                if (settlingAmount > 1e-6f) {
                    updateHeight(posX, posY, settlingAmount);
                    sediment -= settlingAmount;
                }
            }
            
            speed = sqrtf(fmaxf(0.001f, speed*speed - deltaHeight * p.gravity));
            water *= (1.0f - p.evaporateSpeed);

            if (deltaHeight <= 0.0f && sediment > 0.0f) {
                float speedDrop = std::max(0.0f, speedBeforePhysics - speed);
                float speedDropNorm = std::min(1.0f, speedDrop / std::max(speedBeforePhysics + 1e-4f, 0.25f));
                float transitionSettling = sediment * speedDropNorm * (0.12f + 0.38f * flatness) * (0.4f + 0.6f * (1.0f - water));
                if (transitionSettling > 1e-6f) {
                    updateHeight(posX, posY, transitionSettling);
                    sediment -= transitionSettling;
                }
            }

            posX = newPosX;
            posY = newPosY;
            
            if (water < 0.01f || speed < 0.01f) {
                if (sediment > 0.0f) {
                    updateHeight(posX, posY, sediment);
                }
                break;
            }
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
    int edgeFadeWidth = std::max(3, w / 40);
    
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

void TerrainManager::thermalErosion(TerrainObject* terrain, const ThermalErosionParams& p, const std::vector<float>& mask, const std::function<void(float)>& progressCallback) {
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
        if (progressCallback) {
            progressCallback(static_cast<float>(iter) / static_cast<float>(std::max(1, p.iterations)));
        }
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
                
                // Physical slope = dh * heightScale / (dist * cellSize)
                float safeHeightScale = fmaxf(0.1f, terrain->heightmap.scale_y);
                float invCellSize = 1.0f / fmaxf(0.01f, terrain->heightmap.scale_xz / (float)w);
                float slopeScale = safeHeightScale * invCellSize;
                
                for (int d = 0; d < 8; d++) {
                    int nx = x + dx[d];
                    int ny = y + dy[d];
                    nIdxs[d] = ny * w + nx;
                    
                    // Physical slope = dh * heightScale / (dist * cellSize)
                    float diff = center - temp[nIdxs[d]];
                    float slope = diff * slopeScale / distWeight[d];
                    
                    if (slope > effectiveTalus) {
                        float excess = (slope - effectiveTalus) * distWeight[d] * (terrain->heightmap.scale_xz / (float)w) / safeHeightScale;
                        diffs[d] = excess;
                        totalExcess += excess;
                        neighborCount++;
                    } else {
                        diffs[d] = 0.0f;
                    }
                }
                
                // Distribute material proportionally to all steep neighbors
                if (totalExcess > 0.0f && neighborCount > 0) {
                    float totalMove = totalExcess * 0.5f * p.erosionAmount;
                    
                    // STABILITY CAP: Never move more than 1/4 of total height or small constant
                    totalMove = fminf(totalMove, 0.05f);
                    
                    // Ensure center doesn't go negative
                    if (totalMove > center) totalMove = center * 0.9f;
                    
                    // Hardness reduces erosion rate
                    totalMove *= (1.0f - hardness * 0.7f);
                    
                    if (hasMask) {
                        totalMove *= mask[idx];
                    }
                    
                    // Remove from center
                    data[idx] -= totalMove;
                    
                    // Distribute to neighbors proportionally
                    for (int d = 0; d < 8; d++) {
                        if (diffs[d] > 0.0f) {
                            float weight = diffs[d] / totalExcess;
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
void TerrainManager::windErosion(TerrainObject* terrain, float strength, float direction, int iterations, const std::vector<float>& mask, const std::function<void(float)>& progressCallback) {
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
        if (progressCallback) {
            progressCallback(static_cast<float>(iter) / static_cast<float>(std::max(1, iterations)));
        }
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
                    // SHADOW_ANGLE is tangent. Drop in meters = step * cellSize * tangent
                    // Drop in 0-1 units = (step * cellSize * tangent) / heightScale
                    float cellSize = terrain->heightmap.scale_xz / (float)w;
                    float heightScale = fmaxf(1.0f, terrain->heightmap.scale_y);
                    float unitDrop = (step * cellSize * SHADOW_ANGLE) / heightScale;
                    
                    float shadowHeight = upwindH - unitDrop;
                    if (currentH < shadowHeight) {
                        shadowFactor *= 0.4f; // In shadow - reduce erosion
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
                    // Physical slope correction for CPU
                    float cellSize = terrain->heightmap.scale_xz / (float)w;
                    float physicalSlope = windwardSlope * terrain->heightmap.scale_y / cellSize;
                    
                    // Saltation erosion - stronger on exposed windward slopes
                    float erosionAmount = physicalSlope * strength * 0.002f * shadowFactor;
                    
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
                    // Reduced from 0.0002f to be less aggressive
                    float depositionAmount = (1.0f - shadowFactor) * strength * 0.00005f;
                    
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

static bool applyNormalizedFluvialFlowGuide(TerrainObject* terrain,
                                             const std::vector<float>& flowGuide,
                                             int w, int h) {
    const size_t count = static_cast<size_t>(w) * h;
    if (!terrain || flowGuide.size() != count || count == 0) return false;
    terrain->flowMap.resize(count);
    // Watershed publishes accumulation normalized to its largest basin. Mapping
    // that value back to the full pixel count makes a 2K/4K terrain inject
    // millions of units into sqrt(A), immediately saturating the legacy
    // stream-power carver. A bounded effective catchment keeps the same channel
    // hierarchy and threshold semantics without resolution-dependent incision.
    const float contributingAreaScale = static_cast<float>((std::min)(count, size_t{4096}));
    for (size_t i = 0; i < count; ++i) {
        terrain->flowMap[i] = std::clamp(flowGuide[i], 0.0f, 1.0f) * contributingAreaScale;
    }
    return true;
}

void TerrainManager::fluvialErosion(TerrainObject* terrain, const HydraulicErosionParams& p,
                                    const std::vector<float>& mask,
                                    const std::function<void(float)>& progressCallback,
                                    const std::vector<float>& flowGuide) {
    if (!terrain) return;

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    auto& height = terrain->heightmap.data;
    float cellSize = terrain->heightmap.scale_xz / w;
    bool hasHardness = !terrain->hardnessMap.empty();
    bool hasMask = !mask.empty() && mask.size() == w * h;
    const bool usePrescribedFlow = flowGuide.size() == height.size();

    SCENE_LOG_INFO("[CPU Fluvial] Starting Stream Power River Carver...");

    // Same fixes as fluvialErosionGPU's updateFlowCPU (see
    // project_terrain_erosion_gpu_migration memory): (1) hash-based tie-breaker
    // instead of height-based (which is exactly zero on flat ground, causing a
    // deterministic axis-aligned artifact), (2) coherent low-frequency noise on the
    // flow-routing working copy so flat-ground channels meander organically instead
    // of following the boundary-seeded flood's rectilinear distance field. This CPU
    // path duplicates fluvialErosionGPU's flow lambda, so it had the identical bug.
    auto flowTieBreakHashCPU = [](int idx) -> float {
        uint32_t hh = (uint32_t)idx * 2654435761u;
        hh ^= hh >> 15; hh *= 0x2c1b3c6du; hh ^= hh >> 12; hh *= 0x297a2d39u; hh ^= hh >> 15;
        return (float)(hh & 0xFFFFFFu) / (float)0x1000000;
    };
    auto flowCoherentNoiseCPU = [](int x, int y) -> float {
        auto hash2 = [](int xi, int yi) -> float {
            uint32_t hh = (uint32_t)xi * 374761393u + (uint32_t)yi * 668265263u;
            hh = (hh ^ (hh >> 13u)) * 1274126177u;
            hh ^= hh >> 16u;
            return (float)(hh & 0xFFFFFFu) / (float)0x1000000;
        };
        const float blobSize = 20.0f;
        float fx = (float)x / blobSize;
        float fy = (float)y / blobSize;
        int ix = (int)std::floor(fx);
        int iy = (int)std::floor(fy);
        float tx = fx - (float)ix;
        float ty = fy - (float)iy;
        float a = hash2(ix, iy);
        float b = hash2(ix + 1, iy);
        float c = hash2(ix, iy + 1);
        float d = hash2(ix + 1, iy + 1);
        float sx = tx * tx * (3.0f - 2.0f * tx);
        float sy = ty * ty * (3.0f - 2.0f * ty);
        float top = a + (b - a) * sx;
        float bottom = c + (d - c) * sx;
        return top + (bottom - top) * sy;
    };

    std::vector<float> globalFilledHeight;
    auto updateFlowCPU = [&]() {
        if (applyNormalizedFluvialFlowGuide(terrain, flowGuide, w, h)) return;
        std::vector<float> filledHeight = height;
        const float noiseAmplitude = 0.03f;
        for (int ny = 0; ny < h; ++ny) {
            for (int nx = 0; nx < w; ++nx) {
                filledHeight[ny * w + nx] += flowCoherentNoiseCPU(nx, ny) * noiseAmplitude;
            }
        }
        std::vector<int> drainageParent(w * h, -1);
        std::vector<bool> processed(w * h, false);
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> pq;
        const float eps = 0.0001f;
        int dx8[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
        int dy8[] = { -1, -1, -1, 0, 0, 1, 1, 1 };

        for (int x = 0; x < w; x++) {
            pq.push({ filledHeight[x], x });
            pq.push({ filledHeight[(h - 1) * w + x], (h - 1) * w + x });
            processed[x] = processed[(h - 1) * w + x] = true;
        }
        for (int y = 1; y < h - 1; y++) {
            pq.push({ filledHeight[y * w], y * w });
            pq.push({ filledHeight[y * w + w - 1], y * w + w - 1 });
            processed[y * w] = processed[y * w + w - 1] = true;
        }

        while (!pq.empty()) {
            auto [priority, idx] = pq.top(); pq.pop();
            int x = idx % w, y = idx / w;
            float cH = filledHeight[idx];

            for (int d = 0; d < 8; d++) {
                int nx = x + dx8[d], ny = y + dy8[d];
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                int nIdx = ny * w + nx;
                if (processed[nIdx]) continue;
                
                // Tie-breaker for organic paths — hash-based, not height-based (see
                // flowTieBreakHashCPU comment above).
                float tieBreaker = flowTieBreakHashCPU(nIdx) * eps * 0.5f;
                float tH = fmaxf(filledHeight[nIdx], cH + eps);

                filledHeight[nIdx] = tH;
                drainageParent[nIdx] = idx;
                processed[nIdx] = true;
                pq.push({ tH + tieBreaker, nIdx });
            }
        }
        globalFilledHeight = filledHeight;


        std::vector<int> indices(w * h);
        for (int i = 0; i < w * h; i++) indices[i] = i;
        std::sort(indices.begin(), indices.end(), [&](int a, int b) { return filledHeight[a] > filledHeight[b]; });

        terrain->flowMap.assign(w * h, 1.0f);
        for (int i : indices) {
            int x = i % w, y = i / w;
            float currentH = filledHeight[i];
            float totalSlopePower = 0.0f;
            struct FlowNeighbor { int id; float weight; };
            std::vector<FlowNeighbor> neighbors;

            for (int d = 0; d < 8; d++) {
                int nx = x + dx8[d], ny = y + dy8[d];
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                int nIdx = ny * w + nx;
                float dist = (abs(dx8[d]) + abs(dy8[d]) == 2) ? 1.414f : 1.0f;
                float slope = (currentH - filledHeight[nIdx]) / (dist * cellSize);
                if (slope > 0.0f) {
                    float power = powf(slope, 1.5f); // Slightly lower power for better branching
                    neighbors.push_back({ nIdx, power });
                    totalSlopePower += power;
                }
            }

            if (totalSlopePower > 0.0f) {
                for (const auto& n : neighbors) {
                    terrain->flowMap[n.id] += terrain->flowMap[i] * (n.weight / totalSlopePower);
                }
            }
            else if (drainageParent[i] != -1) {
                terrain->flowMap[drainageParent[i]] += terrain->flowMap[i];
            }
        }

        // Resolution-aware diffusion:
        // low-res maps need extra de-gridding on flats, while high-res maps should keep their sharper channels.
        std::vector<float> diffused = terrain->flowMap;
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                int idx = y * w + x;
                float sum = diffused[idx] * 4.0f;
                for (int d = 0; d < 8; d++) sum += diffused[(y + dy8[d]) * w + (x + dx8[d])];
                terrain->flowMap[idx] = sum / 12.0f;
            }
        }

        const float lowResFactor = std::clamp((1024.0f - (float)std::min(w, h)) / 768.0f, 0.0f, 1.0f);
        const int extraDiffusionPasses = (lowResFactor > 0.35f ? 1 : 0) + (lowResFactor > 0.75f ? 1 : 0);
        const float invCellSize = 1.0f / fmaxf(0.001f, cellSize);
        for (int pass = 0; pass < extraDiffusionPasses; ++pass) {
            std::vector<float> sourceFlow = terrain->flowMap;
            for (int y = 1; y < h - 1; ++y) {
                for (int x = 1; x < w - 1; ++x) {
                    int idx = y * w + x;
                    float sum = sourceFlow[idx] * 4.0f;
                    for (int d = 0; d < 8; ++d) sum += sourceFlow[(y + dy8[d]) * w + (x + dx8[d])];
                    float blurred = sum / 12.0f;

                    float slopeX = (filledHeight[idx + 1] - filledHeight[idx - 1]) * 0.5f * invCellSize;
                    float slopeY = (filledHeight[idx + w] - filledHeight[idx - w]) * 0.5f * invCellSize;
                    float localSlope = sqrtf(slopeX * slopeX + slopeY * slopeY);
                    float flatBlend = lowResFactor * (1.0f - std::clamp(localSlope / 0.03f, 0.0f, 1.0f));
                    terrain->flowMap[idx] = sourceFlow[idx] * (1.0f - flatBlend) + blurred * flatBlend;
                }
            }
        }
    };

    int numPasses = std::max(1, p.iterations / 1000);
    if (numPasses > 10) numPasses = 10;
    if (usePrescribedFlow) numPasses = (std::min)(numPasses, 2);

    std::vector<float> erosionAmount(w * h, 0.0f);
    const float Ks = p.erodeSpeed * 0.02f * (usePrescribedFlow ? 0.12f : 1.0f);
    const float maxPassErosionFraction = usePrescribedFlow ? 0.005f : 0.2f;

    for (int pass = 0; pass < numPasses; ++pass) {
        if (progressCallback) {
            progressCallback(static_cast<float>(pass) / static_cast<float>(numPasses));
        }
        updateFlowCPU();
        std::fill(erosionAmount.begin(), erosionAmount.end(), 0.0f);

#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (x <= 2 || x >= w - 3 || y <= 2 || y >= h - 3) continue;
                int idx = y * w + x;

                if (hasMask && mask[idx] < 0.01f) continue;

                float flow = terrain->flowMap[idx];
                if (flow < 3.0f) continue;

                float invCellSize = 1.0f / fmaxf(0.001f, cellSize);
                float slopeX = (height[idx + 1] - height[idx - 1]) * 0.5f * invCellSize;
                float slopeY = (height[idx + w] - height[idx - w]) * 0.5f * invCellSize;
                float slope = sqrtf(slopeX * slopeX + slopeY * slopeY);

                // Guide slope on flat areas using filledHeight gradient
                if (slope < 0.01f && !globalFilledHeight.empty()) {
                    float fSX = (globalFilledHeight[idx + 1] - globalFilledHeight[idx - 1]) * 0.5f * invCellSize;
                    float fSY = (globalFilledHeight[idx + w] - globalFilledHeight[idx - w]) * 0.5f * invCellSize;
                    slope = fmaxf(slope, sqrtf(fSX * fSX + fSY * fSY) * 0.5f);
                }
                
                slope = fmaxf(slope, p.minSlope);

                float streamPower = Ks * sqrtf(flow) * slope;
                if (slope < 0.05f) {
                    float noise = (float)((idx * 1103515245 + 12345) & 0x7FFFFFFF) / 2147483647.0f;
                    float flatIntensity = 1.0f - (slope / 0.05f);
                    streamPower += Ks * sqrtf(flow) * 0.25f * flatIntensity * (0.7f + noise * 0.6f);
                }

                float hardness = hasHardness ? terrain->hardnessMap[idx] : 0.3f;
                float flowScale = fminf(1.0f, (flow - 3.0f) / 100.0f);
                float baseRadius = (float)p.erosionRadius * 0.5f;
                float dynamicRadius = fmaxf(0.5f, baseRadius * sqrtf(flowScale) * (1.5f - hardness));
                int channelRadius = (int)ceilf(dynamicRadius);
                if (channelRadius > 10) channelRadius = 10;

                float hardnessMultiplier = 1.0f - hardness * 0.7f;
                float flowPower = fminf(1.5f, sqrtf(flow) * 0.2f);
                float slopeDampening = fminf(1.0f, slope * 60.0f);
                streamPower *= hardnessMultiplier * p.sedimentCapacity * flowPower * slopeDampening;
                if (hasMask) streamPower *= mask[idx];

                for (int dy = -channelRadius; dy <= channelRadius; ++dy) {
                    for (int dx = -channelRadius; dx <= channelRadius; ++dx) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;

                        float distSq = (float)(dx * dx + dy * dy);
                        if (distSq > dynamicRadius * dynamicRadius) continue;

                        int nIdx = ny * w + nx;
                        float dist = sqrtf(distSq);
                        float t = dist / (dynamicRadius + 0.001f);
                        float falloff = 1.0f - t * t * (3.0f - 2.0f * t);
                        falloff = fmaxf(falloff, 0.0f);

                        float targetHeight = height[nIdx];
                        if (!std::isfinite(targetHeight) || targetHeight <= 0.0f) continue;

                        float erode = streamPower * falloff;
                        erode = fminf(erode, targetHeight * 0.1f);
                        if (!std::isfinite(erode) || erode <= 0.0f) continue;

#pragma omp atomic
                        erosionAmount[nIdx] += erode;
                    }
                }
            }
        }

#pragma omp parallel for
        for (int i = 0; i < w * h; ++i) {
            float currentHeight = height[i];
            float accumulated = erosionAmount[i];

            if (!std::isfinite(currentHeight) || currentHeight <= 0.0f) {
                height[i] = 0.0f;
                continue;
            }

            if (!std::isfinite(accumulated) || accumulated <= 0.0f) continue;

            float maxPassErode = currentHeight * maxPassErosionFraction;
            accumulated = fminf(accumulated, maxPassErode);
            height[i] = fmaxf(currentHeight - accumulated, 0.0f);
        }
    }

    updateTerrainMesh(terrain, false);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("[CPU Fluvial] Stream Power River Carver complete.");
}

// Helper to save a float map (0-1 range) to 8-bit grayscale PNG
static void saveMapPNG(const std::vector<float>& data, int w, int h, const std::string& filepath) {
    if (data.empty() || w <= 0 || h <= 0) return;
    std::vector<uint8_t> pixels(w * h);
    #pragma omp parallel for
    for (int i = 0; i < w * h; i++) {
        pixels[i] = (uint8_t)(std::clamp(data[i], 0.0f, 1.0f) * 255.0f);
    }
    stbi_write_png(filepath.c_str(), w, h, 1, pixels.data(), w);
}

// Helper to load a float map from PNG
static bool loadMapPNG(std::vector<float>& data, int& w, int& h, const std::string& filepath) {
    int imgW, imgH, channels;
    unsigned char* img = stbi_load(filepath.c_str(), &imgW, &imgH, &channels, 1);
    if (!img) return false;
    
    w = imgW;
    h = imgH;
    data.resize(w * h);
    #pragma omp parallel for
    for (int i = 0; i < w * h; i++) {
        data[i] = (float)img[i] / 255.0f;
    }
    stbi_image_free(img);
    return true;
}

void TerrainManager::exportHeightmap(TerrainObject* terrain, const std::string& filepath) {
    if (!terrain) return;
    
    std::string path = filepath;
    if (path.find(".png") == std::string::npos) {
        // Change extension if user provided something else
        std::filesystem::path p(path);
        path = p.replace_extension(".png").string();
    }

    saveMapPNG(terrain->heightmap.data, terrain->heightmap.width, terrain->heightmap.height, path);
    SCENE_LOG_INFO("Heightmap exported as PNG: " + path);
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
   
    // Find min/max height for normalization
    float minH = FLT_MAX, maxH = -FLT_MAX;
    for (float hv : terrain->heightmap.data) {
        if (hv < minH) minH = hv;
        if (hv > maxH) maxH = hv;
    }
    float heightRange = maxH - minH;
    if (heightRange < 0.001f) heightRange = 1.0f;
    
    std::vector<float> rawHardness(w * h, 0.5f);
    float terrainScaleY = terrain->heightmap.scale_y;
    float cellSize = terrain->heightmap.scale_xz / (float)w;

    #pragma omp parallel for
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            
            // Normalized height (0-1)
            float normH = (terrain->heightmap.data[idx] - minH) / heightRange;
            
            // Slope calculation
            float hl = terrain->heightmap.data[idx - 1] * terrainScaleY;
            float hr = terrain->heightmap.data[idx + 1] * terrainScaleY;
            float hu = terrain->heightmap.data[idx - w] * terrainScaleY;
            float hd = terrain->heightmap.data[idx + w] * terrainScaleY;
            
            float dzdx = (hr - hl) / (2.0f * cellSize);
            float dzdy = (hd - hu) / (2.0f * cellSize);
            float slopeRad = std::atan(sqrtf(dzdx * dzdx + dzdy * dzdy));
            float slopeNorm = std::min(slopeRad * 2.0f / 3.14159f, 1.0f);
            
            // Base hardness from height (Stratification)
            float hHardness = 0.1f + 0.8f * std::pow(normH, 1.5f);
            
            // Combine: take the harder of the two
            float combined = fmaxf(hHardness, slopeNorm * slopeWeight);
            
            // Add noise
            combined += noise(gen);
            
            rawHardness[idx] = std::clamp(combined, 0.05f, 1.0f);
        }
    }
    
    // Blur to remove salt-and-pepper noise and create smooth transitions
    terrain->hardnessMap.resize(w * h);
    #pragma omp parallel for
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    sum += rawHardness[(y + dy) * w + (x + dx)];
                }
            }
            terrain->hardnessMap[idx] = sum / 9.0f;
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
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    float resScale = (float)(w * h) / (512.0f * 512.0f); // Use 512x512 as baseline
    
    HydraulicErosionParams hydro;
    hydro.iterations = (int)(iterations * 1000 * resScale); // Doubled from 500
    if (hydro.iterations < 2000) hydro.iterations = 2000;
    if (use_gpu && hydro.iterations < 100000) hydro.iterations = 100000; // GPU loves work
    hydro.erodeSpeed = 0.5f * strength;
    hydro.depositSpeed = 0.2f;
    hydro.evaporateSpeed = 0.01f;
    hydro.sedimentCapacity = 12.0f; // Sharper results
    hydro.erosionRadius = 3;
    hydro.dropletLifetime = 128;
    
    HydraulicErosionParams fluvialParams;
    fluvialParams.erodeSpeed = 1.8f * strength;
    fluvialParams.depositSpeed = 0.05f;
    fluvialParams.iterations = 3000;
    
    ThermalErosionParams thermal;
    thermal.iterations = 25;
    thermal.talusAngle = 0.45f;
    thermal.erosionAmount = 0.5f;
    
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

namespace {
bool saveHeightmapBinaryData(const std::vector<float>& data, uint32_t width, uint32_t height,
                             const std::string& filepath) {
    if (data.size() != static_cast<size_t>(width) * height) return false;
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    const char magic[4] = {'R', 'T', 'H', 'M'};
    const uint32_t version = 1;
    file.write(magic, 4);
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&width), sizeof(width));
    file.write(reinterpret_cast<const char*>(&height), sizeof(height));
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return file.good();
}

bool loadHeightmapBinaryData(std::vector<float>& data, uint32_t& width, uint32_t& height,
                             const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    char magic[4]{};
    uint32_t version = 0;
    file.read(magic, 4);
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&width), sizeof(width));
    file.read(reinterpret_cast<char*>(&height), sizeof(height));
    if (!file.good() || magic[0] != 'R' || magic[1] != 'T' ||
        magic[2] != 'H' || magic[3] != 'M' || version != 1 || width == 0 || height == 0) {
        return false;
    }
    data.resize(static_cast<size_t>(width) * height);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    return file.good();
}
} // namespace

void TerrainManager::saveHeightmapBinary(const TerrainObject* terrain, const std::string& filepath) const {
    if (!terrain) return;
    const uint32_t width = static_cast<uint32_t>(terrain->heightmap.width);
    const uint32_t height = static_cast<uint32_t>(terrain->heightmap.height);
    if (!saveHeightmapBinaryData(terrain->heightmap.data, width, height, filepath)) {
        SCENE_LOG_ERROR("Failed to save heightmap binary: " + filepath);
        return;
    }
    SCENE_LOG_INFO("Saved heightmap binary: " + filepath + " (" + std::to_string(width) + "x" + std::to_string(height) + ")");
}

void TerrainManager::loadHeightmapBinary(TerrainObject* terrain, const std::string& filepath) {
    if (!terrain) return;
    
    uint32_t width = 0, height = 0;
    std::vector<float> loaded;
    if (!loadHeightmapBinaryData(loaded, width, height, filepath)) {
        SCENE_LOG_ERROR("Invalid heightmap binary format: " + filepath);
        return;
    }
    terrain->heightmap.width = width;
    terrain->heightmap.height = height;
    terrain->heightmap.data = std::move(loaded);
    terrain->original_heightmap_data = terrain->heightmap.data;
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
        // Sanitize project name
        std::string safeProjName = g_project.project_name;
        std::replace_if(safeProjName.begin(), safeProjName.end(), 
            [](char c){ return !isalnum(c) && c != '-' && c != '_' && c != ' '; }, '_');
        // Remove spaces for safer filenames
        std::replace(safeProjName.begin(), safeProjName.end(), ' ', '_');

        std::string hmFilename = safeProjName + "_" + t.name + "_heightmap_" + std::to_string(t.id) + ".png";
        std::string binFilename = safeProjName + "_" + t.name + "_heightmap_" + std::to_string(t.id) + ".rthm";
        std::string sourceBinFilename = safeProjName + "_" + t.name + "_height_source_" + std::to_string(t.id) + ".rthm";

        tJson["heightmap"] = {
            {"width", t.heightmap.width},
            {"height", t.heightmap.height},
            {"scale_xz", t.heightmap.scale_xz},
            {"scale_y", t.heightmap.scale_y},
            {"file", hmFilename},
            {"binary_file", binFilename},
            {"source_binary_file", sourceBinFilename}
        };
        
        // Save binary heightmap (Float32 fidelity)
        std::string binPath = (std::filesystem::path(terrainDir) / binFilename).string();
        saveHeightmapBinary(&t, binPath);

        // Persist the authored Terrain-mode input separately from the evaluated
        // output. Without this, reopening a project would promote the final
        // eroded result to the next evaluation's source and accumulate carving.
        const std::vector<float>& sourceHeight =
            t.original_heightmap_data.size() == t.heightmap.data.size()
                ? t.original_heightmap_data : t.heightmap.data;
        const std::string sourceBinPath =
            (std::filesystem::path(terrainDir) / sourceBinFilename).string();
        if (!saveHeightmapBinaryData(sourceHeight,
                static_cast<uint32_t>(t.heightmap.width),
                static_cast<uint32_t>(t.heightmap.height), sourceBinPath)) {
            SCENE_LOG_ERROR("Failed to save terrain source heightmap: " + sourceBinPath);
        }

        // Save heightmap as PNG (Preview / Legacy)
        std::string hmPath = (std::filesystem::path(terrainDir) / hmFilename).string();
        saveMapPNG(t.heightmap.data, t.heightmap.width, t.heightmap.height, hmPath);
        
        // Splat map
        if (t.splatMap && t.splatMap->m_is_loaded) {
            // Ensure splatmap is appropriate for terrain size before saving
            // We use const_cast here because serialize is const but resizeSplatMap is not
            // and we want to ensure the data being saved is correct.
            const_cast<TerrainManager*>(this)->resizeSplatMap(const_cast<TerrainObject*>(&t));

            // Sanitize project name (re-do for scope or reuse variable if strictly in same scope)
            std::string safeProjName = g_project.project_name;
            std::replace_if(safeProjName.begin(), safeProjName.end(), 
                [](char c){ return !isalnum(c) && c != '-' && c != '_' && c != ' '; }, '_');
            std::replace(safeProjName.begin(), safeProjName.end(), ' ', '_');

            std::string splatFilename = safeProjName + "_" + t.name + "_splatmap_" + std::to_string(t.id) + ".png";
            std::string splatPath = (std::filesystem::path(terrainDir) / splatFilename).string();
            
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
            
            tJson["splatmap_file"] = splatFilename;
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
        
        // Node Graph (Non-destructive editing)
        if (t.nodeGraph) {
            tJson["node_graph"] = t.nodeGraph->toJson();
        }
        
        // Hardness map
        if (!t.hardnessMap.empty()) {
            std::string hardnessFilename = safeProjName + "_" + t.name + "_hardness_" + std::to_string(t.id) + ".png";
            std::string hardnessPath = (std::filesystem::path(terrainDir) / hardnessFilename).string();
            saveMapPNG(t.hardnessMap, t.heightmap.width, t.heightmap.height, hardnessPath);
            tJson["hardness_file"] = hardnessFilename;
        }

        // Flow map
        if (!t.flowMap.empty()) {
            std::string flowFilename = safeProjName + "_" + t.name + "_flow_" + std::to_string(t.id) + ".png";
            std::string flowPath = (std::filesystem::path(terrainDir) / flowFilename).string();
            
            // Flow accumulated values can be large, use log scaling + contrast boost
            float maxF = 1.0f;
            for(float f : t.flowMap) if(f > maxF) maxF = f;
            
            std::vector<float> visibleFlow(t.flowMap.size());
            float logMax = log1pf(maxF);
            #pragma omp parallel for
            for(int i = 0; i < (int)t.flowMap.size(); i++) {
                float v = log1pf(t.flowMap[i]) / logMax;
                visibleFlow[i] = powf(v, 2.5f); // Contrast boost: makes main rivers glow/thicker
            }
            
            saveMapPNG(visibleFlow, t.heightmap.width, t.heightmap.height, flowPath);
            tJson["flow_file"] = flowFilename;
            tJson["flow_max"] = maxF; // Store max for restoration
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
        tJson["am_flow_threshold"] = t.am_flow_threshold;

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
    
    std::unordered_set<int> loaded_ids;
    for (const auto& tJson : data["terrains"]) {
        TerrainObject terrain;

        // Do not use value("id", next_id++): the default argument is evaluated
        // even when an ID exists, causing next_id to drift on every load.
        terrain.id = tJson.contains("id") ? tJson["id"].get<int>() : next_id++;
        if (!loaded_ids.insert(terrain.id).second) {
            SCENE_LOG_WARN("[TerrainManager] Duplicate terrain ID " +
                           std::to_string(terrain.id) + " in project; ignoring duplicate entry");
            continue;
        }

        // deserialize supports callers that intentionally retain manager state.
        // Replace an existing terrain with the same persistent ID instead of
        // allowing getTerrain(id) and the graph editor to bind ambiguously.
        if (getTerrain(terrain.id)) {
            SCENE_LOG_WARN("[TerrainManager] Replacing existing terrain ID " +
                           std::to_string(terrain.id) + " during deserialize");
            removeTerrain(scene, terrain.id);
        }

        terrain.name = tJson.value("name", "Terrain_" + std::to_string(terrain.id));
        terrain.material_id = tJson.value("material_id", 0);
        
        // Heightmap
        if (tJson.contains("heightmap")) {
            const auto& hm = tJson["heightmap"];
            terrain.heightmap.width = hm.value("width", 256);
            terrain.heightmap.height = hm.value("height", 256);
            terrain.heightmap.scale_xz = hm.value("scale_xz", 100.0f);
            terrain.heightmap.scale_y = hm.value("scale_y", 10.0f);
            
            // Load heightmap (Binary for v1, PNG for v2+)
            if (hm.contains("binary_file")) {
                 std::string binFile = hm["binary_file"];
                 std::string binPath = (std::filesystem::path(terrainDir) / binFile).string();
                 if (std::filesystem::exists(binPath)) {
                     SCENE_LOG_INFO("Loading binary heightmap: " + binFile);
                     loadHeightmapBinary(&terrain, binPath);
                 } else {
                     SCENE_LOG_WARN("Binary heightmap missing, falling back to PNG: " + binFile);
                     // Fallback to PNG below
                     if (hm.contains("file")) {
                         std::string hmFile = hm["file"];
                         std::string hmPath = (std::filesystem::path(terrainDir) / hmFile).string();
                         int w, h; // Need to declare w, h for loadMapPNG
                         loadMapPNG(terrain.heightmap.data, w, h, hmPath);
                         terrain.heightmap.width = w;
                         terrain.heightmap.height = h;
                     } else {
                         // Initialize flat heightmap if no file found
                         terrain.heightmap.data.resize(terrain.heightmap.width * terrain.heightmap.height, 0.0f);
                     }
                 }
            }
            else if (hm.contains("file")) {
                std::string hmFile = hm["file"];
                std::string hmPath = (std::filesystem::path(terrainDir) / hmFile).string();
                if (std::filesystem::exists(hmPath)) {
                    if (version < 2 && hmFile.find(".raw") != std::string::npos) { // Original v1 raw check
                        loadHeightmapBinary(&terrain, hmPath);
                    } else {
                        int w, h;
                        loadMapPNG(terrain.heightmap.data, w, h, hmPath);
                        terrain.heightmap.width = w;
                        terrain.heightmap.height = h;
                    }
                } else {
                    // Initialize flat heightmap
                    terrain.heightmap.data.resize(terrain.heightmap.width * terrain.heightmap.height, 0.0f);
                }
            }

            bool loadedAuthoredSource = false;
            if (hm.contains("source_binary_file")) {
                const std::string sourcePath = (std::filesystem::path(terrainDir) /
                    hm["source_binary_file"].get<std::string>()).string();
                uint32_t sourceWidth = 0, sourceHeight = 0;
                std::vector<float> sourceData;
                loadedAuthoredSource = loadHeightmapBinaryData(
                    sourceData, sourceWidth, sourceHeight, sourcePath) &&
                    sourceWidth == static_cast<uint32_t>(terrain.heightmap.width) &&
                    sourceHeight == static_cast<uint32_t>(terrain.heightmap.height);
                if (loadedAuthoredSource) {
                    terrain.original_heightmap_data = std::move(sourceData);
                } else {
                    SCENE_LOG_WARN("Terrain source heightmap missing or invalid: " + sourcePath);
                }
            }
            if (!loadedAuthoredSource) {
                // Legacy project: at least freeze the loaded terrain so repeated
                // Evaluate calls stop compounding from this point onward.
                terrain.original_heightmap_data = terrain.heightmap.data;
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
        terrain.am_flow_threshold = tJson.value("am_flow_threshold", 5.0f);
        
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
            std::string splatFilename = tJson["splatmap_file"].get<std::string>();
            std::string splatPath = (std::filesystem::path(terrainDir) / splatFilename).string();
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
                    
                    // Force re-allocation on GPU to match internal size
                    ptr->splatMap->cleanup_gpu();
                    ptr->splatMap->upload_to_gpu();
                    
                    // Rebuild SBT to pick up new texture handle
                    if (g_hasOptix) {
                        g_optix_rebuild_pending = true;
                    }
                    
                    SCENE_LOG_INFO("[TerrainManager] Loaded splatmap: " + std::to_string(w) + "x" + std::to_string(h) + 
                                   " for " + ptr->name);
                }
            }
        }
        
        // Resize splatmap if dimensions don't match heightmap
        // This ensures consistent behavior if heightmap scale was changed manually in JSON
        resizeSplatMap(ptr);
        
        // Final GPU Sync
        if (ptr->splatMap) {
            ptr->splatMap->cleanup_gpu(); // Clear any partial init
            ptr->splatMap->upload_to_gpu();
        }
        
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
            std::string hFileNM = tJson["hardness_file"].get<std::string>();
            std::string hPath = (std::filesystem::path(terrainDir) / hFileNM).string();
            if (std::filesystem::exists(hPath)) {
                if (version < 2 && hFileNM.find(".raw") != std::string::npos) {
                    std::ifstream hStream(hPath, std::ios::binary);
                    if (hStream.is_open()) {
                        uint32_t size = 0;
                        hStream.read(reinterpret_cast<char*>(&size), sizeof(size));
                        ptr->hardnessMap.resize(size);
                        hStream.read(reinterpret_cast<char*>(ptr->hardnessMap.data()), size * sizeof(float));
                        hStream.close();
                    }
                } else {
                    int w, h;
                    loadMapPNG(ptr->hardnessMap, w, h, hPath);
                }
            }
        }

        // Flow map
        if (tJson.contains("flow_file")) {
            std::string fFile = tJson["flow_file"].get<std::string>();
            std::string fPath = (std::filesystem::path(terrainDir) / fFile).string();
            float maxF = tJson.value("flow_max", 1.0f);
            if (std::filesystem::exists(fPath)) {
                int w, h;
                if (loadMapPNG(ptr->flowMap, w, h, fPath)) {
                    // Restore original magnitudes
                    #pragma omp parallel for
                    for(int i = 0; i < (int)ptr->flowMap.size(); i++) ptr->flowMap[i] *= maxF;
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
        
        // Node Graph (Non-destructive editing)
        ptr->nodeGraph = std::make_shared<TerrainNodesV2::TerrainNodeGraphV2>();
        if (tJson.contains("node_graph")) {
            ptr->nodeGraph->fromJson(tJson["node_graph"], ptr);
            
            // Initial evaluation to sync heightmap with loaded graph
            ptr->nodeGraph->evaluateTerrain(ptr, scene);
        }
        
        // Generate mesh
        updateTerrainMesh(ptr);

        // evaluateTerrain() may already have rebuilt and registered this mesh.
        // Canonicalize to one scene entry instead of blindly appending it again.
        registerTerrainMeshOnce(scene, ptr->flatMesh);
    }
    
    SCENE_LOG_INFO("[TerrainManager] Deserialized " + std::to_string(terrains.size()) + " terrains (format v" + std::to_string(version) + ")");
}

// ===========================================================================
// GPU EROSION IMPLEMENTATION
// ===========================================================================

void TerrainManager::initCuda() {
    if (cudaInitialized) return;

    if (!g_hasCUDA) {
        SCENE_LOG_WARN("[GPU Erosion] CUDA not available, skipping GPU erosion init");
        return;
    }

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

    // Use Primary Context (sharing with OptiX/CUDART)
    CUcontext ctx;
    res = cuDevicePrimaryCtxRetain(&ctx, device);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] cuDevicePrimaryCtxRetain failed");
        return;
    }
    res = cuCtxSetCurrent(ctx);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] cuCtxSetCurrent failed");
        return;
    }
    
    // Force CUDART initialization to ensure interop is ready
    cudaFree(0);

    // Load PTX Module (prefer ptx/ subfolder next to exe, fall back to legacy root / build layout)
    std::string ptxPath = "ptx/erosion_kernels.ptx";
    if (!std::filesystem::exists(ptxPath)) ptxPath = "erosion_kernels.ptx";
    if (!std::filesystem::exists(ptxPath)) ptxPath = "../ptx/erosion_kernels.ptx";
    if (!std::filesystem::exists(ptxPath)) ptxPath = "../erosion_kernels.ptx";

    res = cuModuleLoad((CUmodule*)&cudaModule, ptxPath.c_str());
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
    cuModuleGetFunction((CUfunction*)&streamPowerKernelFunc, (CUmodule)cudaModule, "fluvialStreamPowerKernel");
    cuModuleGetFunction((CUfunction*)&applyStreamPowerKernelFunc, (CUmodule)cudaModule, "applyFluvialStreamPowerKernel");
    cuModuleGetFunction((CUfunction*)&windKernelFunc, (CUmodule)cudaModule, "windErosionKernel");

    // Post-processing kernels (for CPU-GPU parity)
    cuModuleGetFunction((CUfunction*)&pitFillKernelFunc, (CUmodule)cudaModule, "pitFillingKernel");
    cuModuleGetFunction((CUfunction*)&spikeRemovalKernelFunc, (CUmodule)cudaModule, "spikeRemovalKernel");
    cuModuleGetFunction((CUfunction*)&edgePreservationKernelFunc, (CUmodule)cudaModule, "edgePreservationKernel");
    cuModuleGetFunction((CUfunction*)&thermalWithHardnessKernelFunc, (CUmodule)cudaModule, "thermalErosionWithHardnessKernel");

    cudaInitialized = true;
    SCENE_LOG_INFO("[GPU Erosion] CUDA Initialized Successfully (with post-processing kernels)");
}

// Tries the Vulkan compute Monte-Carlo droplet hydraulic model (1:1 port of
// hydraulicErosionKernel in erosion_kernels.cu) — matches the "GPU Hydraulic Droplet
// Simulation" the UI already advertises for this node. Replaces an earlier shallow-
// water "pipe model" port attempt: the user visually compared both and found the
// droplet model's organic, non-homogeneous channel character clearly better (the pipe
// model, being purely gradient-driven, produced very uniform/patterned channels even
// after noise-seeding fixes). See project_terrain_erosion_gpu_migration memory.
//
// Batches droplets like the CUDA host code did (avoids a single dispatch running long
// enough to trip a TDR), synchronizing every batch — with realistic droplet counts
// (tens of thousands to ~1M) that's only a handful of dispatches, nowhere near the
// 512-descriptor-set cap that bit the pipe-model's per-substep loop.
static bool hydraulicErosionGpuVulkan(TerrainObject* terrain, const HydraulicErosionParams& p,
                                      const std::vector<float>& mask) {
    using namespace RayTrophiSim;
    std::lock_guard<std::recursive_mutex> computeLock(sharedMeshComputeMutex());
    ISimulationComputeBackend* backend = acquireSharedMeshComputeBackend();
    if (!backend) return false;

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    int numPixels = w * h;
    size_t mapSize = (size_t)numPixels * sizeof(float);
    bool hasHardness = terrain->hardnessMap.size() == (size_t)numPixels;
    bool hasMask = mask.size() == (size_t)numPixels;

    struct DropletPushConstants {
        int mapWidth, mapHeight;
        int brushRadius, dropletLifetime;
        float inertia, sedimentCapacity, minSlope;
        float erodeSpeed, depositSpeed, evaporateSpeed;
        float gravity, cellSize, heightScale;
        uint32_t seed, seedOffset;
        uint32_t hasHardness, hasMask;
        uint32_t dropletCount;
    } gpuOps{};
    static_assert(sizeof(DropletPushConstants) == 72, "must match terrain_hydraulic_droplet.comp push_constant size");

    gpuOps.mapWidth = w;
    gpuOps.mapHeight = h;
    gpuOps.brushRadius = p.erosionRadius;
    gpuOps.dropletLifetime = p.dropletLifetime;
    gpuOps.inertia = p.inertia;
    gpuOps.sedimentCapacity = p.sedimentCapacity;
    gpuOps.minSlope = p.minSlope;
    gpuOps.erodeSpeed = p.erodeSpeed;
    gpuOps.depositSpeed = p.depositSpeed;
    gpuOps.evaporateSpeed = p.evaporateSpeed;
    gpuOps.gravity = p.gravity;
    gpuOps.cellSize = (float)terrain->heightmap.scale_xz / (float)w;
    gpuOps.heightScale = (float)terrain->heightmap.scale_y;
    gpuOps.seed = (uint32_t)rand();
    gpuOps.hasHardness = hasHardness ? 1u : 0u;
    gpuOps.hasMask = hasMask ? 1u : 0u;

    ComputeBufferDesc d; d.size_bytes = mapSize;
    ComputeBufferHandle hHeight = backend->createBuffer(d);
    ComputeBufferHandle hHardness = backend->createBuffer(d);
    ComputeBufferHandle hMask = backend->createBuffer(d);
    auto cleanup = [&]() {
        if (hHeight.valid())   backend->destroyBuffer(hHeight);
        if (hHardness.valid()) backend->destroyBuffer(hHardness);
        if (hMask.valid())     backend->destroyBuffer(hMask);
    };

    bool ok = hHeight.valid() && hHardness.valid() && hMask.valid();
    if (ok) ok = backend->uploadBuffer(hHeight, terrain->heightmap.data.data(), mapSize);
    if (ok) {
        if (hasHardness) ok = backend->uploadBuffer(hHardness, terrain->hardnessMap.data(), mapSize);
        else { std::vector<float> zeros(numPixels, 0.0f); ok = backend->uploadBuffer(hHardness, zeros.data(), mapSize); }
    }
    if (ok) {
        if (hasMask) ok = backend->uploadBuffer(hMask, mask.data(), mapSize);
        else { std::vector<float> ones(numPixels, 1.0f); ok = backend->uploadBuffer(hMask, ones.data(), mapSize); }
    }
    if (!ok) { cleanup(); return false; }

    // Batch config matches the CUDA host loop (batchSize=256000, blockSize=128 there;
    // 256 here to match this codebase's compute-shader convention).
    constexpr int kBatchSize = 256000;
    int dropletsProcessed = 0;
    uint32_t seedOffset = 0;
    while (ok && dropletsProcessed < p.iterations) {
        int currentBatch = std::min(kBatchSize, p.iterations - dropletsProcessed);
        gpuOps.seedOffset = seedOffset;
        gpuOps.dropletCount = (uint32_t)currentBatch;

        ComputeBufferHandle bufs[3] = { hHeight, hHardness, hMask };
        ComputeDispatch cmd;
        cmd.kernel = "terrain_hydraulic_droplet";
        cmd.groups.groups_x = (uint32_t)((currentBatch + 255) / 256);
        cmd.buffers = bufs;
        cmd.buffer_count = 3;
        cmd.constants = &gpuOps;
        cmd.constants_size = sizeof(gpuOps);
        ok = backend->dispatch(cmd);
        if (ok) backend->synchronize();

        seedOffset += (uint32_t)currentBatch;
        dropletsProcessed += currentBatch;
    }

    // Post-processing chain (1:1 with the CUDA host function): spike removal -> pit
    // filling -> edge preservation (against the pre-erosion heights) -> final smoothing.
    const uint32_t groups = (uint32_t)((numPixels + 255) / 256);
    ComputeBufferHandle hOriginal{};
    if (ok) {
        float cellSize = terrain->heightmap.scale_xz / w;
        TerrainPhysics::PostProcessParamsGPU postParams{};
        postParams.mapWidth = w;
        postParams.mapHeight = h;
        postParams.cellSize = cellSize;
        postParams.pitThreshold = cellSize * 0.05f;
        postParams.spikeThreshold = cellSize * 0.1f;
        postParams.edgeFadeWidth = std::max(3, w / 40);

        ComputeBufferHandle spikeBufs[1] = { hHeight };
        ComputeDispatch spikeCmd;
        spikeCmd.kernel = "terrain_spike_removal";
        spikeCmd.groups.groups_x = groups;
        spikeCmd.buffers = spikeBufs;
        spikeCmd.buffer_count = 1;
        spikeCmd.constants = &postParams;
        spikeCmd.constants_size = sizeof(postParams);
        ok = backend->dispatch(spikeCmd);

        if (ok) {
            ComputeBufferHandle pitBufs[1] = { hHeight };
            ComputeDispatch pitCmd;
            pitCmd.kernel = "terrain_pit_fill";
            pitCmd.groups.groups_x = groups;
            pitCmd.buffers = pitBufs;
            pitCmd.buffer_count = 1;
            pitCmd.constants = &postParams;
            pitCmd.constants_size = sizeof(postParams);
            ok = backend->dispatch(pitCmd);
        }

        if (ok) {
            hOriginal = backend->createBuffer(d);
            ok = hOriginal.valid() && backend->uploadBuffer(hOriginal, terrain->heightmap.data.data(), mapSize);
        }
        if (ok) {
            ComputeBufferHandle edgeBufs[2] = { hHeight, hOriginal };
            ComputeDispatch edgeCmd;
            edgeCmd.kernel = "terrain_edge_preservation";
            edgeCmd.groups.groups_x = groups;
            edgeCmd.buffers = edgeBufs;
            edgeCmd.buffer_count = 2;
            edgeCmd.constants = &postParams;
            edgeCmd.constants_size = sizeof(postParams);
            ok = backend->dispatch(edgeCmd);
        }

        if (ok) {
            struct SmoothPushConstants { int width, height; } smoothPc{ w, h };
            static_assert(sizeof(SmoothPushConstants) == 8, "must match terrain_smooth.comp push_constant size");
            ComputeBufferHandle smoothBufs[1] = { hHeight };
            ComputeDispatch smoothCmd;
            smoothCmd.kernel = "terrain_smooth";
            smoothCmd.groups.groups_x = groups;
            smoothCmd.buffers = smoothBufs;
            smoothCmd.buffer_count = 1;
            smoothCmd.constants = &smoothPc;
            smoothCmd.constants_size = sizeof(smoothPc);
            ok = backend->dispatch(smoothCmd);
        }
    }

    if (ok) {
        backend->synchronize();
        ok = backend->downloadBuffer(hHeight, terrain->heightmap.data.data(), mapSize);
    }

    if (!ok) {
        SCENE_LOG_WARN("[GPU Erosion] Vulkan droplet path failed, falling back to CUDA/CPU.");
    }
    if (hOriginal.valid()) backend->destroyBuffer(hOriginal);
    cleanup();
    return ok;
}

void TerrainManager::hydraulicErosionGPU(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask) {
    if (!terrain) return;

    if (hydraulicErosionGpuVulkan(terrain, params, mask)) {
        updateTerrainMesh(terrain);
        terrain->dirty_mesh = true;
        SCENE_LOG_INFO("[GPU Erosion] Droplet hydraulic erosion complete (Vulkan compute).");
        return;
    }

    // Lazy Init
    if (!cudaInitialized) {
        initCuda();
        if (!cudaInitialized) {
            SCENE_LOG_WARN("[GPU Erosion] CUDA not initialized. Falling back to CPU hydraulic erosion.");
            hydraulicErosion(terrain, params, mask);
            return;
        }
    }

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    size_t mapSize = w * h * sizeof(float);
    bool hasHardness = !terrain->hardnessMap.empty() && terrain->hardnessMap.size() == (size_t)(w * h);
    bool hasMask = !mask.empty() && mask.size() == (size_t)(w * h);
    
    // Allocate Device Memory
    CUdeviceptr d_heightmap = 0;
    CUdeviceptr d_original = 0;
    CUdeviceptr d_hardness = 0;
    CUdeviceptr d_mask = 0;
    auto freeGpuBuffers = [&]() {
        if (d_mask) cuMemFree(d_mask);
        if (d_hardness) cuMemFree(d_hardness);
        if (d_original) cuMemFree(d_original);
        if (d_heightmap) cuMemFree(d_heightmap);
    };

    CUresult res = cuMemAlloc(&d_heightmap, mapSize);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_WARN("[GPU Erosion] Memory allocation failed. Falling back to CPU.");
        hydraulicErosion(terrain, params, mask);
        return;
    }
    
    // Copy Host to Device
    res = cuMemcpyHtoD(d_heightmap, terrain->heightmap.data.data(), mapSize);
    if (res != CUDA_SUCCESS) {
        freeGpuBuffers();
        SCENE_LOG_WARN("[GPU Erosion] HtoD copy failed. Falling back to CPU.");
        hydraulicErosion(terrain, params, mask);
        return;
    }

    res = cuMemAlloc(&d_original, mapSize);
    if (res == CUDA_SUCCESS) {
        res = cuMemcpyHtoD(d_original, terrain->heightmap.data.data(), mapSize);
    }
    if (res != CUDA_SUCCESS) {
        freeGpuBuffers();
        SCENE_LOG_WARN("[GPU Erosion] Failed to preserve original heights on GPU. Falling back to CPU.");
        hydraulicErosion(terrain, params, mask);
        return;
    }

    if (hasHardness) {
        res = cuMemAlloc(&d_hardness, mapSize);
        if (res == CUDA_SUCCESS) {
            res = cuMemcpyHtoD(d_hardness, terrain->hardnessMap.data(), mapSize);
        }
        if (res != CUDA_SUCCESS) {
            freeGpuBuffers();
            SCENE_LOG_WARN("[GPU Erosion] Hardness upload failed. Falling back to CPU.");
            hydraulicErosion(terrain, params, mask);
            return;
        }
    }

    if (hasMask) {
        res = cuMemAlloc(&d_mask, mapSize);
        if (res == CUDA_SUCCESS) {
            res = cuMemcpyHtoD(d_mask, mask.data(), mapSize);
        }
        if (res != CUDA_SUCCESS) {
            freeGpuBuffers();
            SCENE_LOG_WARN("[GPU Erosion] Mask upload failed. Falling back to CPU.");
            hydraulicErosion(terrain, params, mask);
            return;
        }
    }
    
    // Prepare Params
    TerrainPhysics::HydraulicErosionParamsGPU gpuParams;
    memset(&gpuParams, 0, sizeof(gpuParams)); // Ensure no garbage pointers
    
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
    gpuParams.cellSize = (float)terrain->heightmap.scale_xz / (float)w;
    gpuParams.heightScale = (float)terrain->heightmap.scale_y;
    gpuParams.seed = (unsigned int)rand();
    gpuParams.hardnessMap = hasHardness ? reinterpret_cast<float*>(d_hardness) : nullptr;
    gpuParams.maskMap = hasMask ? reinterpret_cast<float*>(d_mask) : nullptr;
    
    unsigned long long seed_offset = 0;
    
    // Kernel Args
    void* args[] = { &d_heightmap, &gpuParams, &seed_offset };
    
    // Launch Config
    int blockSize = 128;
    int totalDroplets = params.iterations;
    int batchSize = 256000; // Process in smaller batches to avoid TDR timeout
    
    SCENE_LOG_INFO("[GPU Erosion] Launching kernel with " + std::to_string(totalDroplets) + " total droplets in batches...");
    
    int dropletsProcessed = 0;
    while (dropletsProcessed < totalDroplets) {
        int currentBatch = std::min(batchSize, totalDroplets - dropletsProcessed);
        int numBlocks = (currentBatch + blockSize - 1) / blockSize;
        
        // Kernel Args
        void* args[] = { &d_heightmap, &gpuParams, &seed_offset };
        
        res = cuLaunchKernel((CUfunction)erosionKernelFunc,
                             numBlocks, 1, 1,    // Grid Dim
                             blockSize, 1, 1,    // Block Dim
                             0, nullptr,         // Shared Mem, Stream
                             args, nullptr);
                             
        if (res != CUDA_SUCCESS) {
            SCENE_LOG_ERROR("[GPU Erosion] Launch Failed at droplet " + std::to_string(dropletsProcessed) + ": Error " + std::to_string(res));
            freeGpuBuffers();
            hydraulicErosion(terrain, params, mask);
            return;
        }
        
        // Synchronize batch and update offset
        cuCtxSynchronize(); 
        seed_offset += currentBatch;
        dropletsProcessed += currentBatch;
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
    // We now preserve a copy of the original heights on device for a closer match.
    if (edgePreservationKernelFunc) {
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
        SCENE_LOG_ERROR("[GPU Erosion] DtoH Copy Failed! CUDA Result: " + std::to_string(res));
        if (res == CUDA_ERROR_LAUNCH_FAILED) {
            SCENE_LOG_ERROR("[GPU Erosion] Kernel crash detected (likely OOB or TDR).");
        }
    }
    
    // Cleanup
    freeGpuBuffers();
    
    // Update Mesh (Visuals)
    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("[GPU Erosion] Complete with post-processing!");
}

// Tries the Vulkan compute path for thermal erosion (works on any GPU vendor,
// no CUDA required). Returns false if the shared Vulkan compute backend is
// unavailable or any step fails, leaving `terrain` untouched so the caller can
// fall back to the CUDA/CPU path.
static bool thermalErosionGpuVulkan(TerrainObject* terrain, const ThermalErosionParams& p) {
    using namespace RayTrophiSim;
    std::lock_guard<std::recursive_mutex> computeLock(sharedMeshComputeMutex());
    ISimulationComputeBackend* backend = acquireSharedMeshComputeBackend();
    if (!backend) return false;

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    int numPixels = w * h;
    size_t mapSize = (size_t)numPixels * sizeof(float);
    bool hasHardness = terrain->hardnessMap.size() == (size_t)numPixels;

    // NOTE: intentionally NOT TerrainPhysics::ThermalErosionParamsGPU — that struct
    // carries a trailing bool + pointer (for the CUDA hardness map arg) whose size/
    // padding doesn't match the shader's 24-byte push-constant range. This trimmed
    // POD mirrors the .comp layout (mapWidth,mapHeight,talusAngle,erosionAmount,
    // cellSize,heightScale) exactly — all 4-byte fields, no padding.
    struct ThermalPushConstants {
        int mapWidth, mapHeight;
        float talusAngle, erosionAmount;
        float cellSize, heightScale;
    } gpuOps{};
    gpuOps.mapWidth = w;
    gpuOps.mapHeight = h;
    gpuOps.talusAngle = p.talusAngle;
    gpuOps.erosionAmount = p.erosionAmount;
    gpuOps.cellSize = (float)terrain->heightmap.scale_xz / (float)w;
    gpuOps.heightScale = (float)terrain->heightmap.scale_y;
    static_assert(sizeof(ThermalPushConstants) == 24, "must match terrain_thermal[_hardness].comp push_constant size");

    ComputeBufferDesc d; d.size_bytes = mapSize;
    ComputeBufferHandle hHeight = backend->createBuffer(d);
    ComputeBufferHandle hHardness = hasHardness ? backend->createBuffer(d) : ComputeBufferHandle{};
    auto cleanup = [&]() {
        if (hHeight.valid()) backend->destroyBuffer(hHeight);
        if (hHardness.valid()) backend->destroyBuffer(hHardness);
    };

    bool ok = hHeight.valid() && (!hasHardness || hHardness.valid());
    if (ok) ok = backend->uploadBuffer(hHeight, terrain->heightmap.data.data(), mapSize);
    if (ok && hasHardness) ok = backend->uploadBuffer(hHardness, terrain->hardnessMap.data(), mapSize);
    if (!ok) { cleanup(); return false; }

    const uint32_t groups = (uint32_t)((numPixels + 255) / 256);
    // Batch dispatches between synchronize() calls: the descriptor pool backing
    // dispatch() only holds MAX_DESC_SETS (512) sets and is reset by synchronize().
    // Without periodic synchronize() calls, iteration counts above ~512 silently
    // exhaust the pool, dispatch() starts returning false, and this whole function
    // bails out to the CUDA/CPU fallback with no visible error. See
    // project_terrain_erosion_gpu_migration memory.
    constexpr int kSyncBatch = 128;
    for (int i = 0; i < p.iterations && ok; ++i) {
        ComputeDispatch cmd;
        cmd.groups.groups_x = groups;
        cmd.constants = &gpuOps;
        cmd.constants_size = sizeof(gpuOps);
        if (hasHardness) {
            ComputeBufferHandle bufs[2] = { hHeight, hHardness };
            cmd.kernel = "terrain_thermal_hardness";
            cmd.buffers = bufs;
            cmd.buffer_count = 2;
            ok = backend->dispatch(cmd);
        } else {
            ComputeBufferHandle bufs[1] = { hHeight };
            cmd.kernel = "terrain_thermal";
            cmd.buffers = bufs;
            cmd.buffer_count = 1;
            ok = backend->dispatch(cmd);
        }
        if (ok && (i % kSyncBatch) == (kSyncBatch - 1)) backend->synchronize();
    }

    // Post-processing: pit filling (matches CUDA/CPU behavior).
    if (ok) {
        float cellSize = terrain->heightmap.scale_xz / w;
        TerrainPhysics::PostProcessParamsGPU postParams{};
        postParams.mapWidth = w;
        postParams.mapHeight = h;
        postParams.cellSize = cellSize;
        postParams.pitThreshold = cellSize * 0.05f;
        postParams.spikeThreshold = cellSize * 0.1f;
        postParams.edgeFadeWidth = std::max(3, w / 40);

        ComputeBufferHandle bufs[1] = { hHeight };
        ComputeDispatch cmd;
        cmd.kernel = "terrain_pit_fill";
        cmd.groups.groups_x = groups;
        cmd.buffers = bufs;
        cmd.buffer_count = 1;
        cmd.constants = &postParams;
        cmd.constants_size = sizeof(postParams);
        ok = backend->dispatch(cmd);
    }

    if (ok) {
        backend->synchronize();
        ok = backend->downloadBuffer(hHeight, terrain->heightmap.data.data(), mapSize);
    }

    if (!ok) {
        SCENE_LOG_WARN("[GPU Thermal] Vulkan compute path failed, falling back to CUDA/CPU.");
    }
    cleanup();
    return ok;
}

void TerrainManager::thermalErosionGPU(TerrainObject* terrain, const ThermalErosionParams& p, const std::vector<float>& mask) {
    if (!terrain) return;

    if (thermalErosionGpuVulkan(terrain, p)) {
        updateTerrainMesh(terrain);
        terrain->dirty_mesh = true;
        SCENE_LOG_INFO("[GPU Thermal] Completed " + std::to_string(p.iterations) + " iterations (Vulkan compute).");
        return;
    }

    if (!cudaInitialized) {
        initCuda();
        if (!cudaInitialized) {
            SCENE_LOG_WARN("[GPU Thermal] CUDA not initialized. Falling back to CPU thermal erosion.");
            thermalErosion(terrain, p, mask);
            return;
        }
    }
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    size_t mapSize = w * h * sizeof(float);
    
    // Device Alloc
    CUdeviceptr d_heightmap;
    CUresult res = cuMemAlloc(&d_heightmap, mapSize);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_WARN("[GPU Thermal] Allocation failed. Falling back to CPU.");
        thermalErosion(terrain, p, mask);
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
    gpuOps.cellSize = (float)terrain->heightmap.scale_xz / (float)w;
    gpuOps.heightScale = (float)terrain->heightmap.scale_y;
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

// Fast device-resident fluvial runoff.  Unlike the legacy stream-power path below,
// this uses particle transport, keeps discharge on the GPU between batches, and
// carries sediment until it is deposited.  The CPU fluvial implementation remains
// untouched; the legacy Vulkan/CUDA stream-power path is retained as a fallback.
static bool fluvialRunoffGpuVulkan(TerrainObject* terrain,
                                   const HydraulicErosionParams& p,
                                   const std::vector<float>& mask) {
    using namespace RayTrophiSim;
    std::lock_guard<std::recursive_mutex> computeLock(sharedMeshComputeMutex());
    ISimulationComputeBackend* backend = acquireSharedMeshComputeBackend();
    if (!backend) return false;

    const int w = terrain->heightmap.width;
    const int h = terrain->heightmap.height;
    const int numPixels = w * h;
    if (w < 8 || h < 8 || numPixels <= 0) return false;

    const size_t mapSize = (size_t)numPixels * sizeof(float);
    const bool hasHardness = terrain->hardnessMap.size() == (size_t)numPixels;
    const bool hasMask = mask.size() == (size_t)numPixels;

    struct RunoffPushConstants {
        int mapWidth, mapHeight;
        int brushRadius, dropletLifetime;
        float inertia, sedimentCapacity, minSlope, erodeSpeed;
        float depositSpeed, evaporateSpeed, gravity, cellSize;
        float heightScale, flowBoost, flowNormalization, flatSlope;
        float meanderStrength, maxErosionFraction, flowDecay, depositSpread;
        uint32_t seed, seedOffset, hasHardness, hasMask;
        uint32_t dropletCount, batchIndex;
        float minWater;
        uint32_t pad;
    } pc{};
    static_assert(sizeof(RunoffPushConstants) == 112,
                  "must match terrain_fluvial_*.comp push constants");

    pc.mapWidth = w;
    pc.mapHeight = h;
    pc.brushRadius = std::clamp(p.erosionRadius, 1, 12);
    pc.dropletLifetime = std::clamp(p.dropletLifetime, 32, 1024);
    pc.inertia = std::clamp(p.inertia, 0.12f, 0.98f);
    pc.sedimentCapacity = std::max(p.sedimentCapacity, 0.05f);
    pc.minSlope = std::max(p.minSlope, 0.0001f);
    pc.erodeSpeed = std::clamp(p.erodeSpeed, 0.001f, 2.0f);
    pc.depositSpeed = std::clamp(p.depositSpeed, 0.0f, 1.0f);
    pc.evaporateSpeed = std::clamp(p.evaporateSpeed, 0.0001f, 0.02f);
    pc.gravity = std::max(p.gravity, 0.1f);
    pc.cellSize = std::max(0.001f, terrain->heightmap.scale_xz / (float)w);
    pc.heightScale = std::max(0.001f, terrain->heightmap.scale_y);
    pc.flowBoost = 0.65f;
    pc.flowNormalization = 16.0f;
    pc.flatSlope = std::max(0.01f, pc.minSlope * 4.0f);
    pc.meanderStrength = pc.flatSlope * 1.5f;
    pc.maxErosionFraction = 0.010f;
    // Applied after every 32-cell continuation segment.  Slow decay preserves the
    // accumulated catchment field across the full authored travel distance.
    pc.flowDecay = 0.9995f;
    pc.depositSpread = std::clamp((float)pc.brushRadius * 0.75f + 1.0f, 2.0f, 8.0f);
    pc.seed = (uint32_t)rand();
    pc.hasHardness = hasHardness ? 1u : 0u;
    pc.hasMask = hasMask ? 1u : 0u;
    pc.minWater = 0.01f;

    constexpr int kBatchSize = 16384;
    constexpr int kTravelSegmentLength = 32;
    const int totalTravelLength = pc.dropletLifetime;
    const uint32_t continuationSegments = (uint32_t)(
        (totalTravelLength + kTravelSegmentLength - 1) / kTravelSegmentLength);
    // Low 16 bits: continuation count. High 16 bits: exact authored travel.
    // The exact value lets the shader distribute final flat settling correctly when
    // Travel Length is not an integer multiple of the 32-cell continuation size.
    pc.pad = ((uint32_t)totalTravelLength << 16u) |
             (continuationSegments & 0xffffu);

    ComputeBufferDesc d;
    d.size_bytes = mapSize;
    ComputeBufferDesc parcelStateDesc;
    parcelStateDesc.size_bytes = (size_t)kBatchSize * 12u * sizeof(float);
    ComputeBufferHandle hHeight = backend->createBuffer(d);
    ComputeBufferHandle hHardness = backend->createBuffer(d);
    ComputeBufferHandle hMask = backend->createBuffer(d);
    ComputeBufferHandle hFlow = backend->createBuffer(d);
    ComputeBufferHandle hErosion = backend->createBuffer(d);
    ComputeBufferHandle hDeposition = backend->createBuffer(d);
    ComputeBufferHandle hFlowDirectionX = backend->createBuffer(d);
    ComputeBufferHandle hFlowDirectionY = backend->createBuffer(d);
    ComputeBufferHandle hParcelStates = backend->createBuffer(parcelStateDesc);
    auto cleanup = [&]() {
        if (hHeight.valid()) backend->destroyBuffer(hHeight);
        if (hHardness.valid()) backend->destroyBuffer(hHardness);
        if (hMask.valid()) backend->destroyBuffer(hMask);
        if (hFlow.valid()) backend->destroyBuffer(hFlow);
        if (hErosion.valid()) backend->destroyBuffer(hErosion);
        if (hDeposition.valid()) backend->destroyBuffer(hDeposition);
        if (hFlowDirectionX.valid()) backend->destroyBuffer(hFlowDirectionX);
        if (hFlowDirectionY.valid()) backend->destroyBuffer(hFlowDirectionY);
        if (hParcelStates.valid()) backend->destroyBuffer(hParcelStates);
    };

    bool ok = hHeight.valid() && hHardness.valid() && hMask.valid() && hFlow.valid() &&
              hErosion.valid() && hDeposition.valid() && hFlowDirectionX.valid() &&
              hFlowDirectionY.valid() && hParcelStates.valid();
    std::vector<float> zeros((size_t)numPixels, 0.0f);
    std::vector<float> ones;
    if (ok) ok = backend->uploadBuffer(hHeight, terrain->heightmap.data.data(), mapSize);
    if (ok) ok = backend->uploadBuffer(hHardness,
        hasHardness ? terrain->hardnessMap.data() : zeros.data(), mapSize);
    if (ok) {
        if (hasMask) ok = backend->uploadBuffer(hMask, mask.data(), mapSize);
        else {
            ones.assign((size_t)numPixels, 1.0f);
            ok = backend->uploadBuffer(hMask, ones.data(), mapSize);
        }
    }
    if (ok) ok = backend->uploadBuffer(hFlow, zeros.data(), mapSize);
    if (ok) ok = backend->uploadBuffer(hErosion, zeros.data(), mapSize);
    if (ok) ok = backend->uploadBuffer(hDeposition, zeros.data(), mapSize);
    if (ok) ok = backend->uploadBuffer(hFlowDirectionX, zeros.data(), mapSize);
    if (ok) ok = backend->uploadBuffer(hFlowDirectionY, zeros.data(), mapSize);
    if (!ok) {
        cleanup();
        return false;
    }

    // Each parcel's authored reach is split into 32-cell GPU continuation segments.
    // Height/delta application between segments preserves the visually correct short-
    // step erosion character while parcel state carries water and sediment to plains.
    const uint32_t mapGroups = (uint32_t)((numPixels + 255) / 256);
    int processed = 0;
    int recordedDispatches = 0;
    while (ok && processed < p.iterations) {
        const int count = std::min(kBatchSize, p.iterations - processed);
        pc.seedOffset = (uint32_t)processed;
        pc.dropletCount = (uint32_t)count;

        int traveled = 0;
        uint32_t segmentIndex = 0;
        while (ok && traveled < totalTravelLength) {
            const int segmentSteps = std::min(kTravelSegmentLength, totalTravelLength - traveled);
            pc.dropletLifetime = segmentSteps;
            pc.batchIndex = segmentIndex; // zero initializes, later values continue state

            ComputeBufferHandle runoffBuffers[9] = {
                hHeight, hHardness, hMask, hFlow, hErosion, hDeposition,
                hFlowDirectionX, hFlowDirectionY, hParcelStates
            };
            ComputeDispatch runoff;
            runoff.kernel = "terrain_fluvial_runoff";
            runoff.groups.groups_x = (uint32_t)((count + 255) / 256);
            runoff.buffers = runoffBuffers;
            runoff.buffer_count = 9;
            runoff.constants = &pc;
            runoff.constants_size = sizeof(pc);
            ok = backend->dispatch(runoff);
            ++recordedDispatches;
            if (!ok) break;

            ComputeBufferHandle applyBuffers[6] = {
                hHeight, hFlow, hErosion, hDeposition, hFlowDirectionX, hFlowDirectionY
            };
            ComputeDispatch apply;
            apply.kernel = "terrain_fluvial_apply";
            apply.groups.groups_x = mapGroups;
            apply.buffers = applyBuffers;
            apply.buffer_count = 6;
            apply.constants = &pc;
            apply.constants_size = sizeof(pc);
            ok = backend->dispatch(apply);
            ++recordedDispatches;

            traveled += segmentSteps;
            ++segmentIndex;

            // Talus every second segment (and at the final segment) is enough to
            // prevent spikes without turning the continuation chain into a blur.
            if (ok && (((segmentIndex & 1u) == 0u) || traveled >= totalTravelLength)) {
                ComputeBufferHandle talusBuffers[2] = { hHeight, hFlow };
                ComputeDispatch talus;
                talus.kernel = "terrain_fluvial_talus";
                talus.groups.groups_x = mapGroups;
                talus.buffers = talusBuffers;
                talus.buffer_count = 2;
                talus.constants = &pc;
                talus.constants_size = sizeof(pc);
                ok = backend->dispatch(talus);
                ++recordedDispatches;
            }

            // Vulkan's shared descriptor pool has 512 sets.  Submit before reaching
            // the limit; buffers and parcel continuation state remain device-resident.
            if (ok && recordedDispatches >= 400) {
                backend->synchronize();
                recordedDispatches = 0;
            }
        }

        processed += count;
    }

    pc.dropletLifetime = totalTravelLength;

    // A few cheap settling passes after the last rainfall wave remove any residual
    // one-cell peaks while retaining channel banks below the physical talus limit.
    for (int settlePass = 0; ok && settlePass < 5; ++settlePass) {
        ComputeBufferHandle talusBuffers[2] = { hHeight, hFlow };
        ComputeDispatch talus;
        talus.kernel = "terrain_fluvial_talus";
        talus.groups.groups_x = mapGroups;
        talus.buffers = talusBuffers;
        talus.buffer_count = 2;
        talus.constants = &pc;
        talus.constants_size = sizeof(pc);
        ok = backend->dispatch(talus);
    }

    if (ok) {
        backend->synchronize();
        ok = backend->downloadBuffer(hHeight, terrain->heightmap.data.data(), mapSize);
    }
    if (ok) {
        terrain->flowMap.assign((size_t)numPixels, 0.0f);
        ok = backend->downloadBuffer(hFlow, terrain->flowMap.data(), mapSize);
    }

    cleanup();
    if (!ok) {
        SCENE_LOG_WARN("[GPU Fluvial] Particle runoff failed; trying legacy stream-power fallback.");
    }
    return ok;
}

float TerrainManager::sampleAnalysisField(float worldX, float worldZ, const std::string& fieldName) const {
    if (fieldName.empty()) return 1.0f;

    for (const auto& terrain : terrains) {
        const auto fieldIt = terrain.analysisFields.find(fieldName);
        if (fieldIt == terrain.analysisFields.end() || !fieldIt->second) continue;

        const Heightmap& hm = terrain.heightmap;
        const auto& field = *fieldIt->second;
        const size_t expected = static_cast<size_t>(hm.width) * hm.height;
        if (hm.width < 2 || hm.height < 2 || hm.scale_xz <= 0.0f || field.size() != expected) continue;

        Vec3 localPos(worldX, 0.0f, worldZ);
        if (terrain.transform) {
            const Matrix4x4 inv = terrain.transform->getFinal().inverse();
            localPos = inv.multiplyVector(Vec4(worldX, 0.0f, worldZ, 1.0f)).xyz();
        }
        if (localPos.x < 0.0f || localPos.x > hm.scale_xz ||
            localPos.z < 0.0f || localPos.z > hm.scale_xz) continue;

        const float gridX = std::clamp(localPos.x / hm.scale_xz, 0.0f, 1.0f) * (hm.width - 1);
        const float gridZ = std::clamp(localPos.z / hm.scale_xz, 0.0f, 1.0f) * (hm.height - 1);
        const int x0 = static_cast<int>(std::floor(gridX));
        const int z0 = static_cast<int>(std::floor(gridZ));
        const int x1 = std::min(x0 + 1, hm.width - 1);
        const int z1 = std::min(z0 + 1, hm.height - 1);
        const float fx = gridX - x0;
        const float fz = gridZ - z0;
        auto at = [&](int x, int z) { return field[static_cast<size_t>(z) * hm.width + x]; };
        const float a = at(x0, z0) * (1.0f - fx) + at(x1, z0) * fx;
        const float b = at(x0, z1) * (1.0f - fx) + at(x1, z1) * fx;
        return std::clamp(a * (1.0f - fz) + b * fz, 0.0f, 1.0f);
    }
    return -1.0f;
}

// Tries GPU flow-accumulation: depression-fill (terrain_flow_fill.comp, ping-pong
// Jacobi relaxation) -> outflow weights (terrain_flow_weights.comp, one-time) -> flow
// accumulation (terrain_flow_accumulate.comp, ping-pong relaxation). Approximates the
// CPU priority-flood + topological-order accumulation in updateFlowCPU — a true
// parallel equivalent is a much bigger undertaking (see project_terrain_erosion_gpu_migration
// memory). Downloads the result into terrain->flowMap on success, matching
// updateFlowCPU's output contract, so callers can try this first and fall back to the
// CPU lambda on failure.
//
// Pass counts (kFillPasses=150, kAccumPasses=200) are a first-cut budget, not a
// verified convergence bound — very large flat regions or very long single rivers may
// need more passes than this to fully settle. Expect retuning after visual testing,
// same as every other pass-count constant introduced this migration.
static bool computeFlowMapGpuVulkan(TerrainObject* terrain, float cellSize) {
    using namespace RayTrophiSim;
    std::lock_guard<std::recursive_mutex> computeLock(sharedMeshComputeMutex());
    ISimulationComputeBackend* backend = acquireSharedMeshComputeBackend();
    if (!backend) return false;

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    int numPixels = w * h;
    size_t mapSize = (size_t)numPixels * sizeof(float);
    size_t weightsSize = mapSize * 8;

    ComputeBufferDesc d1; d1.size_bytes = mapSize;
    ComputeBufferDesc d8; d8.size_bytes = weightsSize;

    ComputeBufferHandle hOrig    = backend->createBuffer(d1);
    ComputeBufferHandle hFillA   = backend->createBuffer(d1);
    ComputeBufferHandle hFillB   = backend->createBuffer(d1);
    ComputeBufferHandle hWeights = backend->createBuffer(d8);
    ComputeBufferHandle hFlowA   = backend->createBuffer(d1);
    ComputeBufferHandle hFlowB   = backend->createBuffer(d1);
    auto cleanup = [&]() {
        if (hOrig.valid())    backend->destroyBuffer(hOrig);
        if (hFillA.valid())   backend->destroyBuffer(hFillA);
        if (hFillB.valid())   backend->destroyBuffer(hFillB);
        if (hWeights.valid()) backend->destroyBuffer(hWeights);
        if (hFlowA.valid())   backend->destroyBuffer(hFlowA);
        if (hFlowB.valid())   backend->destroyBuffer(hFlowB);
    };

    bool ok = hOrig.valid() && hFillA.valid() && hFillB.valid() && hWeights.valid() && hFlowA.valid() && hFlowB.valid();
    if (ok) ok = backend->uploadBuffer(hOrig, terrain->heightmap.data.data(), mapSize);
    if (ok) { std::vector<float> big(numPixels, 1e18f); ok = backend->uploadBuffer(hFillA, big.data(), mapSize); }
    if (ok) { std::vector<float> ones(numPixels, 1.0f); ok = backend->uploadBuffer(hFlowA, ones.data(), mapSize); }
    if (!ok) { cleanup(); return false; }

    const uint32_t groups = (uint32_t)((numPixels + 255) / 256);
    constexpr int kSyncBatch = 100;

    // Jacobi relaxation only propagates information ~1 cell per pass from the
    // boundary seed. A fixed small pass count (the original 150/200 here) only
    // converges within a *band* that many cells wide from the map edge — confirmed
    // by the user: "GPU yapı arazinin çevresinde sadece belli bir genişlikte
    // uygulandı" (only a band around the perimeter got processed, the interior
    // stayed at stale/sentinel values). Fix: scale pass count to the map's diagonal
    // (in cells) so both the fill and accumulation relaxations can fully reach the
    // center of the terrain regardless of size. This trades performance for
    // correctness — large terrains now run proportionally more passes. See
    // project_terrain_erosion_gpu_migration memory.
    const int convergencePasses = (int)std::ceil(std::sqrt((double)w * w + (double)h * h)) + 10;

    // ---- Step 1: depression fill (ping-pong Jacobi relaxation) ----
    struct FillPushConstants { int mapWidth, mapHeight; float eps, noiseAmplitude; }
        fillPc{ w, h, 0.0001f, 0.03f };
    static_assert(sizeof(FillPushConstants) == 16, "must match terrain_flow_fill.comp push_constant size");

    const int kFillPasses = convergencePasses;
    ComputeBufferHandle fillSrc = hFillA, fillDst = hFillB;
    for (int i = 0; ok && i < kFillPasses; ++i) {
        ComputeBufferHandle bufs[3] = { hOrig, fillSrc, fillDst };
        ComputeDispatch cmd;
        cmd.kernel = "terrain_flow_fill";
        cmd.groups.groups_x = groups;
        cmd.buffers = bufs;
        cmd.buffer_count = 3;
        cmd.constants = &fillPc;
        cmd.constants_size = sizeof(fillPc);
        ok = backend->dispatch(cmd);
        std::swap(fillSrc, fillDst); // latest result now lives in fillSrc
        if (ok && (i % kSyncBatch) == (kSyncBatch - 1)) backend->synchronize();
    }

    // ---- Step 2: outflow weights (one-time) ----
    struct WeightsPushConstants { int mapWidth, mapHeight; float cellSizeVal; }
        weightsPc{ w, h, cellSize };
    static_assert(sizeof(WeightsPushConstants) == 12, "must match terrain_flow_weights.comp push_constant size");
    if (ok) {
        ComputeBufferHandle bufs[2] = { fillSrc, hWeights };
        ComputeDispatch cmd;
        cmd.kernel = "terrain_flow_weights";
        cmd.groups.groups_x = groups;
        cmd.buffers = bufs;
        cmd.buffer_count = 2;
        cmd.constants = &weightsPc;
        cmd.constants_size = sizeof(weightsPc);
        ok = backend->dispatch(cmd);
        if (ok) backend->synchronize();
    }

    // ---- Step 3: flow accumulation (ping-pong relaxation) ----
    struct AccumPushConstants { int mapWidth, mapHeight; } accumPc{ w, h };
    static_assert(sizeof(AccumPushConstants) == 8, "must match terrain_flow_accumulate.comp push_constant size");
    const int kAccumPasses = convergencePasses;
    ComputeBufferHandle flowSrc = hFlowA, flowDst = hFlowB;
    for (int i = 0; ok && i < kAccumPasses; ++i) {
        ComputeBufferHandle bufs[3] = { hWeights, flowSrc, flowDst };
        ComputeDispatch cmd;
        cmd.kernel = "terrain_flow_accumulate";
        cmd.groups.groups_x = groups;
        cmd.buffers = bufs;
        cmd.buffer_count = 3;
        cmd.constants = &accumPc;
        cmd.constants_size = sizeof(accumPc);
        ok = backend->dispatch(cmd);
        std::swap(flowSrc, flowDst); // latest result now lives in flowSrc
        if (ok && (i % kSyncBatch) == (kSyncBatch - 1)) backend->synchronize();
    }

    if (ok) {
        backend->synchronize();
        terrain->flowMap.assign((size_t)numPixels, 1.0f);
        ok = backend->downloadBuffer(flowSrc, terrain->flowMap.data(), mapSize);
    }

    if (!ok) {
        SCENE_LOG_WARN("[GPU Fluvial] GPU flow-accumulation failed, falling back to CPU flow routing.");
    }
    cleanup();
    return ok;
}

// Tries the Vulkan compute path for stream-power fluvial erosion (works on any
// GPU vendor, no CUDA required). Flow accumulation tries GPU first (see
// computeFlowMapGpuVulkan above), falling back to the CPU updateFlowCPU lambda passed
// in by the caller — only the per-pass stream-power + apply kernels are GPU-only with
// no CPU-in-the-loop fallback path of their own. Returns false if the shared Vulkan
// compute backend is unavailable or any step fails, leaving `terrain` untouched so the
// caller can fall back to CUDA/CPU.
static bool fluvialStreamPowerGpuVulkan(TerrainObject* terrain,
                                        const TerrainPhysics::StreamPowerParamsGPU& sp,
                                        int numPasses,
                                        const std::vector<float>& mask,
                                        const std::function<void()>& updateFlowCPU,
                                        bool usePrescribedFlow) {
    using namespace RayTrophiSim;
    std::lock_guard<std::recursive_mutex> computeLock(sharedMeshComputeMutex());
    ISimulationComputeBackend* backend = acquireSharedMeshComputeBackend();
    if (!backend) return false;

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    int numPixels = w * h;
    size_t mapSize = (size_t)numPixels * sizeof(float);
    auto& height = terrain->heightmap.data;

    std::vector<float> hardnessHost = (terrain->hardnessMap.size() == (size_t)numPixels)
        ? terrain->hardnessMap : std::vector<float>(numPixels, 0.0f);
    std::vector<float> maskHost = (mask.size() == (size_t)numPixels)
        ? mask : std::vector<float>(numPixels, 1.0f);

    ComputeBufferDesc d; d.size_bytes = mapSize;
    ComputeBufferHandle hHeight = backend->createBuffer(d);
    ComputeBufferHandle hFlow   = backend->createBuffer(d);
    ComputeBufferHandle hHard   = backend->createBuffer(d);
    ComputeBufferHandle hMask   = backend->createBuffer(d);
    ComputeBufferHandle hAccum  = backend->createBuffer(d);
    auto cleanup = [&]() {
        if (hHeight.valid()) backend->destroyBuffer(hHeight);
        if (hFlow.valid())   backend->destroyBuffer(hFlow);
        if (hHard.valid())   backend->destroyBuffer(hHard);
        if (hMask.valid())   backend->destroyBuffer(hMask);
        if (hAccum.valid())  backend->destroyBuffer(hAccum);
    };

    bool ok = hHeight.valid() && hFlow.valid() && hHard.valid() && hMask.valid() && hAccum.valid();
    if (ok) ok = backend->uploadBuffer(hHard, hardnessHost.data(), mapSize);
    if (ok) ok = backend->uploadBuffer(hMask, maskHost.data(), mapSize);
    if (!ok) { cleanup(); return false; }

    const uint32_t groups = (uint32_t)((numPixels + 255) / 256);
    std::vector<float> zeroAccum(numPixels, 0.0f);

    for (int pass = 0; ok && pass < numPasses; ++pass) {
        if (usePrescribedFlow) updateFlowCPU();
        else if (!computeFlowMapGpuVulkan(terrain, sp.cellSize)) updateFlowCPU();

        ok = backend->uploadBuffer(hHeight, height.data(), mapSize);
        if (ok) ok = backend->uploadBuffer(hFlow, terrain->flowMap.data(), mapSize);
        if (ok) ok = backend->uploadBuffer(hAccum, zeroAccum.data(), mapSize);
        if (!ok) break;

        ComputeBufferHandle spBufs[5] = { hHeight, hAccum, hFlow, hHard, hMask };
        ComputeDispatch cmd;
        cmd.kernel = "terrain_stream_power";
        cmd.groups.groups_x = groups;
        cmd.buffers = spBufs;
        cmd.buffer_count = 5;
        cmd.constants = &sp;
        cmd.constants_size = sizeof(sp);
        ok = backend->dispatch(cmd);
        if (!ok) break;

        ComputeBufferHandle applyBufs[2] = { hHeight, hAccum };
        ComputeDispatch applyCmd;
        applyCmd.kernel = "terrain_apply_stream_power";
        applyCmd.groups.groups_x = groups;
        applyCmd.buffers = applyBufs;
        applyCmd.buffer_count = 2;
        applyCmd.constants = &sp;
        applyCmd.constants_size = sizeof(sp);
        ok = backend->dispatch(applyCmd);
        if (!ok) break;

        backend->synchronize();
        ok = backend->downloadBuffer(hHeight, height.data(), mapSize);
    }

    if (!ok) {
        SCENE_LOG_WARN("[GPU Fluvial] Vulkan compute path failed, falling back to CUDA/CPU.");
    }
    cleanup();
    return ok;
}

void TerrainManager::fluvialErosionGPU(TerrainObject* terrain, const HydraulicErosionParams& p,
                                       const std::vector<float>& mask,
                                       const std::vector<float>& flowGuide) {
    if (!terrain) return;

    const bool usePrescribedFlow = flowGuide.size() == terrain->heightmap.data.size();
    SCENE_LOG_INFO(usePrescribedFlow
        ? "[GPU Fluvial] Starting watershed-guided stream power..."
        : "[GPU Fluvial] Starting device-resident particle runoff...");
    if (!usePrescribedFlow && fluvialRunoffGpuVulkan(terrain, p, mask)) {
        updateTerrainMesh(terrain);
        terrain->dirty_mesh = true;
        SCENE_LOG_INFO("[GPU Fluvial] Particle runoff complete (transport + deposition).");
        return;
    }

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    int numPixels = w * h;
    auto& height = terrain->heightmap.data;
    float cellSize = terrain->heightmap.scale_xz / (float)w;

    TerrainPhysics::StreamPowerParamsGPU sp{};
    sp.mapWidth = w; sp.mapHeight = h;
    sp.cellSize = cellSize;
    sp.heightScale = terrain->heightmap.scale_y;
    // Particle-runoff defaults use very large iteration counts. When a fixed
    // watershed guide selects the stream-power fallback, use a deliberately
    // conservative coefficient/cap so the same channel is not excavated to
    // bedrock over repeated passes.
    sp.erodeSpeed = p.erodeSpeed * (usePrescribedFlow ? 0.12f : 1.0f);
    sp.sedimentCapacity = p.sedimentCapacity;
    sp.minSlope = p.minSlope;
    sp.erosionRadius = p.erosionRadius;
    sp.maxPassErosionFraction = usePrescribedFlow ? 0.005f : 0.2f;

    int numPasses = std::max(1, p.iterations / 1000); // 1 pass per 1000 iterations for carving
    if (numPasses > 10) numPasses = 10;
    if (usePrescribedFlow) numPasses = (std::min)(numPasses, 2);

    // --------------------------------------------------------
    // Step 1: Flow Accumulation (CPU for accuracy, then HtoD) — shared by both backends
    // --------------------------------------------------------
    // Hash-based per-cell tie-breaker for the priority-flood below. Physics fix
    // (2026-07-01): the original tie-breaker was `height[nIdx] * 1e-6f`, which is
    // EXACTLY ZERO on flat ground (height==0 everywhere pre-sculpt, or any locally
    // flat valley floor). With no tie-breaking, the priority_queue resolves ties by
    // the fixed dx8/dy8 neighbor-scan order every time, so the flood — and the flow
    // directions derived from its filledHeight gradient — follows a deterministic,
    // axis-aligned "distance transform" pattern radiating from the map boundary,
    // which is exactly the geometric/angular riverbed the user reported on flat
    // terrain. A height-independent hash breaks ties randomly instead. See
    // project_terrain_erosion_gpu_migration memory.
    auto flowTieBreakHash = [](int idx) -> float {
        uint32_t h = (uint32_t)idx * 2654435761u;
        h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
        return (float)(h & 0xFFFFFFu) / (float)0x1000000;
    };

    // Coherent low-frequency (~20-cell) value noise, applied to a WORKING COPY of the
    // heightmap used only for flow routing (never written back to the real terrain).
    // Physics fix (2026-07-01): the per-cell tie-break hash above only reorders which
    // neighbor wins among *equal-priority* ties — it doesn't change the fact that the
    // priority-flood still seeds from the map boundary and grows via a uniform `+eps`
    // per hop, so on flat ground the whole flow-direction field is still fundamentally
    // a rectilinear/radial "distance from boundary" gradient. That's why the user still
    // saw wide, dead-straight bands where a high-flow channel reached flat ground (it
    // was following that boundary-distance gradient almost exactly, then carving at its
    // saturated max channel width). A coherent noise perturbation big enough to dominate
    // the eps-accumulated trend on flat ground — but tiny next to real mountain relief,
    // since its gradient (amplitude/blobSize) stays small relative to steep slopes —
    // lets flow meander organically only where the real terrain has (almost) no opinion.
    // See project_terrain_erosion_gpu_migration memory.
    auto flowCoherentNoise = [](int x, int y) -> float {
        auto hash2 = [](int xi, int yi) -> float {
            uint32_t hh = (uint32_t)xi * 374761393u + (uint32_t)yi * 668265263u;
            hh = (hh ^ (hh >> 13u)) * 1274126177u;
            hh ^= hh >> 16u;
            return (float)(hh & 0xFFFFFFu) / (float)0x1000000;
        };
        const float blobSize = 20.0f;
        float fx = (float)x / blobSize;
        float fy = (float)y / blobSize;
        int ix = (int)std::floor(fx);
        int iy = (int)std::floor(fy);
        float tx = fx - (float)ix;
        float ty = fy - (float)iy;
        float a = hash2(ix, iy);
        float b = hash2(ix + 1, iy);
        float c = hash2(ix, iy + 1);
        float d = hash2(ix + 1, iy + 1);
        float sx = tx * tx * (3.0f - 2.0f * tx);
        float sy = ty * ty * (3.0f - 2.0f * ty);
        float top = a + (b - a) * sx;
        float bottom = c + (d - c) * sx;
        return top + (bottom - top) * sy;
    };

    auto updateFlowCPU = [&]() {
        if (applyNormalizedFluvialFlowGuide(terrain, flowGuide, w, h)) return;
        std::vector<float> filledHeight = height;
        const float noiseAmplitude = 0.03f;
        for (int ny = 0; ny < h; ++ny) {
            for (int nx = 0; nx < w; ++nx) {
                filledHeight[ny * w + nx] += flowCoherentNoise(nx, ny) * noiseAmplitude;
            }
        }
        std::vector<int> drainageParent(numPixels, -1);
        std::vector<bool> processed(numPixels, false);
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> pq;
        const float eps = 0.0001f;
        int dx8[] = {-1, 0, 1, -1, 1, -1, 0, 1};
        int dy8[] = {-1, -1, -1, 0, 0, 1, 1, 1};

        for (int x = 0; x < w; x++) {
            pq.push({filledHeight[x], x}); pq.push({filledHeight[(h-1)*w + x], (h-1)*w + x});
            processed[x] = processed[(h-1)*w + x] = true;
        }
        for (int y = 1; y < h-1; y++) {
            pq.push({filledHeight[y*w], y*w}); pq.push({filledHeight[y*w + w-1], y*w + w-1});
            processed[y*w] = processed[y*w + w-1] = true;
        }
        while (!pq.empty()) {
            auto [priority, idx] = pq.top(); pq.pop();
            int x = idx % w, y = idx / w;
            float cH = filledHeight[idx];
            for (int d = 0; d < 8; d++) {
                int nx = x + dx8[d], ny = y + dy8[d];
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                int nIdx = ny * w + nx;
                if (processed[nIdx]) continue;
                
                // Tie-breaker for organic paths on flats — hash-based, not height-based
                // (see flowTieBreakHash comment above: height-based was zero on flat
                // ground, producing a deterministic axis-aligned artifact).
                float tieBreaker = flowTieBreakHash(nIdx) * eps * 0.5f;
                float tH = fmaxf(filledHeight[nIdx], cH + eps);
                
                filledHeight[nIdx] = tH; 
                drainageParent[nIdx] = idx;
                processed[nIdx] = true; 
                pq.push({tH + tieBreaker, nIdx});
            }
        }

        std::vector<int> indices(numPixels);
        for (int i = 0; i < numPixels; i++) indices[i] = i;
        std::sort(indices.begin(), indices.end(), [&](int a, int b) { return filledHeight[a] > filledHeight[b]; });
        
        terrain->flowMap.assign(numPixels, 1.0f);
        for (int i : indices) {
            int x = i % w, y = i / w;
            float currentH = filledHeight[i];
            float totalSlopePower = 0.0f;
            struct FlowNeighbor { int id; float weight; };
            std::vector<FlowNeighbor> neighbors;
            for (int d = 0; d < 8; d++) {
                int nx = x + dx8[d], ny = y + dy8[d];
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                int nIdx = ny * w + nx;
                float dist = (abs(dx8[d]) + abs(dy8[d]) == 2) ? 1.414f : 1.0f;
                float slope = (currentH - filledHeight[nIdx]) / (dist * cellSize);
                if (slope > 0.0f) {
                    float power = powf(slope, 1.5f);
                    neighbors.push_back({ nIdx, power });
                    totalSlopePower += power;
                }
            }

            if (totalSlopePower > 0.0f) {
                for (const auto& n : neighbors) {
                    terrain->flowMap[n.id] += terrain->flowMap[i] * (n.weight / totalSlopePower);
                }
            } else if (drainageParent[i] != -1) {
                terrain->flowMap[drainageParent[i]] += terrain->flowMap[i];
            }
        }

        // Resolution-aware diffusion:
        // low-res maps need extra de-gridding on flats, while high-res maps should keep their sharper channels.
        std::vector<float> diffused = terrain->flowMap;
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                int idx = y * w + x;
                float sum = diffused[idx] * 4.0f;
                for (int d = 0; d < 8; d++) sum += diffused[(y + dy8[d]) * w + (x + dx8[d])];
                terrain->flowMap[idx] = sum / 12.0f;
            }
        }

        const float lowResFactor = std::clamp((1024.0f - (float)std::min(w, h)) / 768.0f, 0.0f, 1.0f);
        const int extraDiffusionPasses = (lowResFactor > 0.35f ? 1 : 0) + (lowResFactor > 0.75f ? 1 : 0);
        const float invCellSize = 1.0f / fmaxf(0.001f, cellSize);
        for (int pass = 0; pass < extraDiffusionPasses; ++pass) {
            std::vector<float> sourceFlow = terrain->flowMap;
            for (int y = 1; y < h - 1; ++y) {
                for (int x = 1; x < w - 1; ++x) {
                    int idx = y * w + x;
                    float sum = sourceFlow[idx] * 4.0f;
                    for (int d = 0; d < 8; ++d) sum += sourceFlow[(y + dy8[d]) * w + (x + dx8[d])];
                    float blurred = sum / 12.0f;

                    float slopeX = (filledHeight[idx + 1] - filledHeight[idx - 1]) * 0.5f * invCellSize;
                    float slopeY = (filledHeight[idx + w] - filledHeight[idx - w]) * 0.5f * invCellSize;
                    float localSlope = sqrtf(slopeX * slopeX + slopeY * slopeY);
                    float flatBlend = lowResFactor * (1.0f - std::clamp(localSlope / 0.03f, 0.0f, 1.0f));
                    terrain->flowMap[idx] = sourceFlow[idx] * (1.0f - flatBlend) + blurred * flatBlend;
                }
            }
        }
    };

    // --------------------------------------------------------
    // Step 2: Multi-Pass Carving — try Vulkan compute first (any GPU vendor).
    // --------------------------------------------------------
    SCENE_LOG_INFO("[GPU Fluvial] Starting High-Power River Carver...");

    if (fluvialStreamPowerGpuVulkan(
            terrain, sp, numPasses, mask, updateFlowCPU, usePrescribedFlow)) {
        updateTerrainMesh(terrain);
        terrain->dirty_mesh = true;
        SCENE_LOG_INFO("[GPU Fluvial] River Carving Complete (Vulkan compute).");
        return;
    }

    if (!cudaInitialized) {
        initCuda();
        if (!cudaInitialized) {
            SCENE_LOG_WARN("[GPU Fluvial] CUDA not initialized. Falling back to CPU fluvial erosion.");
            fluvialErosion(terrain, p, mask, std::function<void(float)>{}, flowGuide);
            return;
        }
    }

    // Alloc temporary buffers
    CUdeviceptr d_height, d_flow, d_mask, d_hardness, d_erosionAccum;
    cuMemAlloc(&d_height, numPixels * sizeof(float));
    cuMemAlloc(&d_flow, numPixels * sizeof(float));
    cuMemAlloc(&d_mask, numPixels * sizeof(float));
    cuMemAlloc(&d_hardness, numPixels * sizeof(float));
    cuMemAlloc(&d_erosionAccum, numPixels * sizeof(float));

    if (!terrain->hardnessMap.empty()) {
        cuMemcpyHtoD(d_hardness, terrain->hardnessMap.data(), numPixels * sizeof(float));
    } else {
        cuMemsetD8(d_hardness, 0, numPixels * sizeof(float));
    }

    if (!mask.empty()) {
        cuMemcpyHtoD(d_mask, mask.data(), numPixels * sizeof(float));
    } else {
        std::vector<float> fullMask(numPixels, 1.0f);
        cuMemcpyHtoD(d_mask, fullMask.data(), numPixels * sizeof(float));
    }

    int tx = 16, ty = 16;
    int bx = (w + tx - 1) / tx;
    int by = (h + ty - 1) / ty;

    for (int pass = 0; pass < numPasses; pass++) {
        updateFlowCPU();
        cuMemcpyHtoD(d_height, height.data(), numPixels * sizeof(float));
        cuMemcpyHtoD(d_flow, terrain->flowMap.data(), numPixels * sizeof(float));
        cuMemsetD8(d_erosionAccum, 0, numPixels * sizeof(float));
        
        void* args[] = { &d_height, &d_erosionAccum, &d_flow, &d_hardness, &d_mask, &sp };
        cuLaunchKernel((CUfunction)streamPowerKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr, args, nullptr);
        if (applyStreamPowerKernelFunc) {
            void* applyArgs[] = { &d_height, &d_erosionAccum, &sp };
            cuLaunchKernel((CUfunction)applyStreamPowerKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr, applyArgs, nullptr);
        }
        cuCtxSynchronize();
        
        cuMemcpyDtoH(height.data(), d_height, numPixels * sizeof(float));
    }

    // Cleanup
    cuMemFree(d_height);
    cuMemFree(d_flow);
    cuMemFree(d_mask);
    cuMemFree(d_hardness);
    cuMemFree(d_erosionAccum);

    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("[GPU Fluvial] River Carving Complete.");
}


// Tries the Vulkan compute path for wind erosion (works on any GPU vendor, no CUDA
// required). Physics fix vs the CUDA original: curand_init(1337, idx, 0, &state) used
// a FIXED seed every iteration, so the wind-direction jitter never varied between
// iterations. Fixed here with a hash seeded by (idx, iterationSeed), incremented per
// dispatch. See project_terrain_erosion_gpu_migration memory.
static bool windErosionGpuVulkan(TerrainObject* terrain, float strength, float direction, int iterations) {
    using namespace RayTrophiSim;
    std::lock_guard<std::recursive_mutex> computeLock(sharedMeshComputeMutex());
    ISimulationComputeBackend* backend = acquireSharedMeshComputeBackend();
    if (!backend) return false;

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    int numPixels = w * h;
    size_t mapSize = (size_t)numPixels * sizeof(float);

    struct WindPushConstants {
        int mapWidth, mapHeight;
        float windDirX, windDirY;
        float strength, suspensionRate, depositionRate;
        float cellSize, heightScale;
        uint32_t iterationSeed;
    } gpuOps{};
    static_assert(sizeof(WindPushConstants) == 40, "must match terrain_wind.comp push_constant size");

    float rad = direction * 3.14159265f / 180.0f;
    gpuOps.mapWidth = w;
    gpuOps.mapHeight = h;
    gpuOps.windDirX = cosf(rad);
    gpuOps.windDirY = sinf(rad);
    gpuOps.strength = strength * 0.1f;
    gpuOps.suspensionRate = 0.3f;
    gpuOps.depositionRate = 0.8f;
    gpuOps.cellSize = (float)terrain->heightmap.scale_xz / (float)w;
    gpuOps.heightScale = (float)terrain->heightmap.scale_y;

    ComputeBufferDesc d; d.size_bytes = mapSize;
    ComputeBufferHandle hHeight = backend->createBuffer(d);
    auto cleanup = [&]() { if (hHeight.valid()) backend->destroyBuffer(hHeight); };

    bool ok = hHeight.valid();
    if (ok) ok = backend->uploadBuffer(hHeight, terrain->heightmap.data.data(), mapSize);
    if (!ok) { cleanup(); return false; }

    const uint32_t groups = (uint32_t)((numPixels + 255) / 256);
    // See kSyncBatch note in thermalErosionGpuVulkan — the dispatch() descriptor pool
    // only holds 512 sets and is reset by synchronize(); without periodic
    // synchronize() calls, iteration counts above ~512 silently exhaust it and this
    // function bails out to CUDA/CPU with no visible error.
    constexpr int kSyncBatch = 128;
    for (int i = 0; i < iterations && ok; ++i) {
        gpuOps.iterationSeed = (uint32_t)i;
        ComputeBufferHandle bufs[1] = { hHeight };
        ComputeDispatch cmd;
        cmd.kernel = "terrain_wind";
        cmd.groups.groups_x = groups;
        cmd.buffers = bufs;
        cmd.buffer_count = 1;
        cmd.constants = &gpuOps;
        cmd.constants_size = sizeof(gpuOps);
        ok = backend->dispatch(cmd);
        if (ok && (i % kSyncBatch) == (kSyncBatch - 1)) backend->synchronize();
    }

    // Post-processing: smoothing pass (matches CUDA behavior — wind erosion tends to
    // create small artifacts).
    if (ok) {
        struct SmoothPushConstants { int width, height; } smoothPc{ w, h };
        static_assert(sizeof(SmoothPushConstants) == 8, "must match terrain_smooth.comp push_constant size");
        ComputeBufferHandle bufs[1] = { hHeight };
        ComputeDispatch cmd;
        cmd.kernel = "terrain_smooth";
        cmd.groups.groups_x = groups;
        cmd.buffers = bufs;
        cmd.buffer_count = 1;
        cmd.constants = &smoothPc;
        cmd.constants_size = sizeof(smoothPc);
        ok = backend->dispatch(cmd);
    }

    if (ok) {
        backend->synchronize();
        ok = backend->downloadBuffer(hHeight, terrain->heightmap.data.data(), mapSize);
    }

    if (!ok) {
        SCENE_LOG_WARN("[GPU Wind] Vulkan compute path failed, falling back to CUDA/CPU.");
    }
    cleanup();
    return ok;
}

void TerrainManager::windErosionGPU(TerrainObject* terrain, float strength, float direction, int iterations, const std::vector<float>& mask) {
    if (!terrain) return;

    if (windErosionGpuVulkan(terrain, strength, direction, iterations)) {
        updateTerrainMesh(terrain);
        terrain->dirty_mesh = true;
        SCENE_LOG_INFO("[GPU Wind] Completed " + std::to_string(iterations) + " iterations with smoothing (Vulkan compute).");
        return;
    }

    if (!cudaInitialized) {
        initCuda();
        if (!cudaInitialized) {
            SCENE_LOG_WARN("[GPU Wind] CUDA not initialized. Falling back to CPU wind erosion.");
            windErosion(terrain, strength, direction, iterations, mask);
            return;
        }
    }

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    size_t mapSize = w * h * sizeof(float);

    // Device Alloc
    CUdeviceptr d_heightmap;
    CUresult res = cuMemAlloc(&d_heightmap, mapSize);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_WARN("[GPU Wind] Allocation failed. Falling back to CPU.");
        windErosion(terrain, strength, direction, iterations, mask);
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
    gpuOps.strength = strength * 0.1f; // Increased from 0.01f
    gpuOps.suspensionRate = 0.3f;
    gpuOps.depositionRate = 0.8f;
    gpuOps.cellSize = (float)terrain->heightmap.scale_xz / (float)w;
    gpuOps.heightScale = (float)terrain->heightmap.scale_y;
    
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

void TerrainManager::calculateFlowMap(TerrainObject* terrain) {
    if (!terrain) return;
    
    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    if (w < 2 || h < 2) return;

    auto& height = terrain->heightmap.data;
    float heightScale = terrain->heightmap.scale_y;

    SCENE_LOG_INFO("[TerrainManager] Calculating Organic Flow Map (MFD + Diffusion)...");

    // 1. Sort indices by height (Descending)
    std::vector<int> indices(w * h);
    for (int i = 0; i < w * h; i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) { 
        return height[a] > height[b]; 
    });

    // 2. Accumulate Flow (Multiple Flow Direction)
    // Initial flow is 1.0 (Rain)
    std::vector<float> flow(w * h, 1.0f);
    
    const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const float dist[] = {1.414f, 1.0f, 1.414f, 1.0f, 1.0f, 1.414f, 1.0f, 1.414f};

    for (int idx : indices) {
        int x = idx % w;
        int y = idx / w;
        if (x == 0 || x == w - 1 || y == 0 || y == h - 1) continue;

        float centerH = height[idx] * heightScale;
        float currentFlow = flow[idx];

        float totalSlopePower = 0.0f;
        float power[8];
        int downhillCount = 0;

        for (int d = 0; d < 8; d++) {
            int nidx = (y + dy[d]) * w + (x + dx[d]);
            float slope = (centerH - height[nidx] * heightScale) / dist[d];
            if (slope > 0.001f) {
                float p = powf(slope, 1.1f); // Exponent for channelization
                power[d] = p;
                totalSlopePower += p;
                downhillCount++;
            } else {
                power[d] = 0.0f;
            }
        }

        if (totalSlopePower > 0.0f) {
            // Distribute flow to all downhill neighbors (MFD)
            for (int d = 0; d < 8; d++) {
                if (power[d] > 0.0f) {
                    int nidx = (y + dy[d]) * w + (x + dx[d]);
                    flow[nidx] += currentFlow * (power[d] / totalSlopePower);
                }
            }
        } else {
            // SOFT FLAT AREA HANDLING: Spread flow to all neighbors
            // This avoids the "sharp geometric line" artifacts caused by sink-filling.
            // On a plateau, water spreads out like a pool.
            for (int d = 0; d < 8; d++) {
                int nidx = (y + dy[d]) * w + (x + dx[d]);
                flow[nidx] += currentFlow * 0.125f; 
            }
        }
    }

    // 3. Post-Processing: Flow Diffusion Pass
    // Softens the resulting mask to make it look organic and handle micro-noise
    std::vector<float> diffuseFlow = flow;
    for (int dPass = 0; dPass < 1; dPass++) {
        std::vector<float> temp = diffuseFlow;
        #pragma omp parallel for collapse(2)
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int idx = y * w + x;
                float sum = diffuseFlow[idx] * 2.0f;
                for (int d = 0; d < 8; d++) {
                    sum += diffuseFlow[(y + dy[d]) * w + (x + dx[d])];
                }
                temp[idx] = sum / 10.0f;
            }
        }
        diffuseFlow = temp;
    }

    terrain->flowMap = diffuseFlow;
}

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
                        if (splatTex && splatW > 0 && splatH > 0 && !splatTex->pixels.empty()) {
                            int sampleX = std::clamp((int)(u * (float)(splatW - 1)), 0, splatW - 1);
                            int sampleY = std::clamp((int)((1.0f - v) * (float)(splatH - 1)), 0, splatH - 1);
                            int pIdx = sampleY * splatW + sampleX;
                            if (pIdx >= 0 && pIdx < (int)splatTex->pixels.size()) {
                                const auto& p = splatTex->pixels[pIdx];
                                int ch = slayer.targetMaskLayerId;
                                if (ch == 0) maskValue = p.r / 255.0f;
                                else if (ch == 1) maskValue = p.g / 255.0f;
                                else if (ch == 2) maskValue = p.b / 255.0f;
                                else if (ch == 3) maskValue = p.a / 255.0f;
                            }
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
                
                int instId = accel->addInstance(slayer.meshId, inst.transform, 0, InstanceType::Mesh, inst.name);
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

bool TerrainManager::hasLegacyFoliage() const {
    for (const auto& terrain : terrains) {
        if (!terrain.foliageLayers.empty()) {
            return true;
        }
    }
    return false;
}

int TerrainManager::migrateLegacyFoliageToInstanceGroups(SceneData& scene, bool clearLegacy) {
    InstanceManager& im = InstanceManager::getInstance();
    int migratedGroups = 0;

    for (auto& terrain : terrains) {
        for (const auto& layer : terrain.foliageLayers) {
            if (!layer.enabled || layer.density <= 0 || layer.meshPath.empty()) continue;

            std::vector<std::shared_ptr<Triangle>> sourceTriangles = layer.sourceTriangles;
            if (sourceTriangles.empty()) {
                sourceTriangles = findLegacyFoliageSourceTriangles(layer.meshPath, scene);
            }
            if (sourceTriangles.empty()) {
                SCENE_LOG_WARN("[Foliage Migration] Could not resolve source mesh for legacy layer '" + layer.name +
                               "' (meshPath=" + layer.meshPath + ")");
                continue;
            }

            std::string groupBase = "Foliage_" + terrain.name + "_" + layer.name;
            std::string groupName = makeUniqueFoliageGroupName(groupBase, im);
            int groupId = im.createGroup(groupName, layer.meshPath, {});
            InstanceGroup* group = im.getGroup(groupId);
            if (!group) continue;

            group->sources.clear();
            group->sources.emplace_back(layer.meshPath, sourceTriangles);
            group->brush_settings.use_global_settings = true;
            group->brush_settings.target_count = layer.density;
            group->brush_settings.scale_min = layer.scaleRange.x;
            group->brush_settings.scale_max = layer.scaleRange.y;
            group->brush_settings.rotation_random_y = std::max(0.0f, layer.rotationRange.y - layer.rotationRange.x);
            group->brush_settings.y_offset_min = layer.yOffsetRange.x;
            group->brush_settings.y_offset_max = layer.yOffsetRange.y;
            group->brush_settings.align_to_normal = layer.alignToNormal > 0.001f;
            group->brush_settings.normal_influence = std::clamp(layer.alignToNormal, 0.0f, 1.0f);
            group->brush_settings.splat_map_channel = layer.targetMaskLayerId;

            Texture* splatTex = terrain.splatMap.get();
            const int targetCount = layer.density;
            const int maxAttempts = targetCount * 5;
            std::mt19937 rng(12345 + terrain.id * 7 + (int)migratedGroups * 101);
            std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
            std::uniform_real_distribution<float> distScale(layer.scaleRange.x, layer.scaleRange.y);
            std::uniform_real_distribution<float> distRot(layer.rotationRange.x, layer.rotationRange.y);
            std::uniform_real_distribution<float> distYOffset(layer.yOffsetRange.x, layer.yOffsetRange.y);

            int spawnedCount = 0;
            for (int attempt = 0; attempt < maxAttempts && spawnedCount < targetCount; ++attempt) {
                float u = dist01(rng);
                float v = dist01(rng);

                float maskValue = 1.0f;
                if (splatTex && splatTex->is_loaded() && !splatTex->pixels.empty()) {
                    int sampleX = std::clamp((int)(u * (float)(splatTex->width - 1)), 0, splatTex->width - 1);
                    int sampleY = std::clamp((int)((1.0f - v) * (float)(splatTex->height - 1)), 0, splatTex->height - 1);
                    int pIdx = sampleY * splatTex->width + sampleX;
                    if (pIdx >= 0 && pIdx < (int)splatTex->pixels.size()) {
                        const auto& p = splatTex->pixels[pIdx];
                        const int ch = layer.targetMaskLayerId;
                        if (ch == 0) maskValue = p.r / 255.0f;
                        else if (ch == 1) maskValue = p.g / 255.0f;
                        else if (ch == 2) maskValue = p.b / 255.0f;
                        else if (ch == 3) maskValue = p.a / 255.0f;
                    }
                }
                if (maskValue < layer.maskThreshold) continue;

                int gx = std::clamp((int)(u * (terrain.heightmap.width - 1)), 0, terrain.heightmap.width - 1);
                int gy = std::clamp((int)(v * (terrain.heightmap.height - 1)), 0, terrain.heightmap.height - 1);

                float localX = u * terrain.heightmap.scale_xz;
                float localZ = v * terrain.heightmap.scale_xz;
                float localY = terrain.heightmap.getHeight(gx, gy);
                Vec3 worldPos(localX, localY, localZ);
                if (terrain.transform) {
                    worldPos = terrain.transform->getFinal().multiplyVector(Vec4(worldPos, 1.0f)).xyz();
                }

                InstanceTransform inst;
                inst.position = worldPos;
                inst.position.y += distYOffset(rng);
                float uniformScale = distScale(rng);
                inst.scale = Vec3(uniformScale, uniformScale, uniformScale);
                inst.rotation = Vec3(0.0f, distRot(rng), 0.0f);
                inst.source_index = 0;

                if (layer.alignToNormal > 0.001f) {
                    Vec3 normal = computeLegacyFoliageNormal(*this, &terrain, worldPos);
                    applyLegacyNormalAlignment(inst, normal, std::clamp(layer.alignToNormal, 0.0f, 1.0f));
                }

                group->addInstance(inst);
                spawnedCount++;
            }

            SCENE_LOG_INFO("[Foliage Migration] Migrated legacy layer '" + layer.name + "' to group '" +
                           groupName + "' with " + std::to_string(group->instances.size()) + " instances");
            migratedGroups++;
        }

        if (clearLegacy) {
            terrain.foliageLayers.clear();
        }
    }

    return migratedGroups;
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
