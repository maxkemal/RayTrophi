// ═══════════════════════════════════════════════════════════════════════════════
// RIVER MANAGER IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════
// Rivers are registered as WaterSurface objects for full water system integration
// ═══════════════════════════════════════════════════════════════════════════════

#include "RiverSpline.h"
#include "scene_data.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "TerrainManager.h"
#include "WaterSystem.h"
#include "globals.h"
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════════
// TERRAIN HEIGHT SAMPLING
// ═══════════════════════════════════════════════════════════════════════════════
float RiverManager::sampleTerrainHeight(const Vec3& position) const {
    auto& terrainManager = TerrainManager::getInstance();
    
    if (!terrainManager.hasActiveTerrain()) {
        return position.y;
    }
    
    return terrainManager.sampleHeight(position.x, position.z);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: Sample terrain normal at world position
// ═══════════════════════════════════════════════════════════════════════════════
static Vec3 sampleTerrainNormal(float worldX, float worldZ) {
    auto& tm = TerrainManager::getInstance();
    if (!tm.hasActiveTerrain()) {
        return Vec3(0, 1, 0);
    }
    
    float delta = 0.5f;
    float hC = tm.sampleHeight(worldX, worldZ);
    float hL = tm.sampleHeight(worldX - delta, worldZ);
    float hR = tm.sampleHeight(worldX + delta, worldZ);
    float hB = tm.sampleHeight(worldX, worldZ - delta);
    float hF = tm.sampleHeight(worldX, worldZ + delta);
    
    Vec3 tangentX(2.0f * delta, hR - hL, 0.0f);
    Vec3 tangentZ(0.0f, hF - hB, 2.0f * delta);
    Vec3 normal = Vec3::cross(tangentZ, tangentX).normalize();
    
    if (normal.y < 0) normal = normal * -1.0f;
    return normal;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MESH GENERATION - Creates WaterSurface in WaterManager
// ═══════════════════════════════════════════════════════════════════════════════
void RiverManager::generateMesh(RiverSpline* river, SceneData& scene) {
    if (!river || river->spline.pointCount() < 2) return;
    
    auto& waterMgr = WaterManager::getInstance();
    auto& matMgr = MaterialManager::getInstance();
    
    // ─────────────────────────────────────────────────────────────────────────
    // STEP 1: Remove old mesh from scene and WaterManager
    // ─────────────────────────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────────────────────────
    // STEP 1: Remove old mesh from scene
    // ─────────────────────────────────────────────────────────────────────────
    if (!river->meshTriangles.empty()) {
        for (auto& tri : river->meshTriangles) {
            auto it = std::find(scene.world.objects.begin(), scene.world.objects.end(), 
                               std::static_pointer_cast<Hittable>(tri));
            if (it != scene.world.objects.end()) {
                scene.world.objects.erase(it);
            }
        }
        river->meshTriangles.clear();
    }
    else {
        // Fallback: Check for existing objects with this river's name (e.g. from load)
        // to prevent duplicate geometry if meshTriangles wasn't populated yet.
        auto it = scene.world.objects.begin();
        while (it != scene.world.objects.end()) {
            if (auto tri = std::dynamic_pointer_cast<Triangle>(*it)) {
                if (tri->getNodeName() == river->name) {
                    it = scene.world.objects.erase(it);
                    continue;
                }
            }
            ++it;
        }
    }
    
    // Determine the expected Water Surface ID for this river
    int expectedWaterID = 10000 + river->id;

    // We no longer blindly remove the water surface here.
    // We will check for its existence in Step 6 and reuse/update it.
    
    // ─────────────────────────────────────────────────────────────────────────
    // STEP 2: Create water material (WaterManager style)
    // ─────────────────────────────────────────────────────────────────────────
    std::string matName = "RiverWater_" + river->name;
    uint16_t waterMatID = matMgr.getMaterialID(matName);
    
    if (waterMatID == MaterialManager::INVALID_MATERIAL_ID) {
        auto water_mat = std::make_shared<PrincipledBSDF>();
        auto gpu = std::make_shared<GpuMaterial>();
        
        auto& wp = river->waterParams;
        
        // Full water material setup (same as WaterManager)
        gpu->albedo = make_float3(wp.deep_color.x, wp.deep_color.y, wp.deep_color.z);
        gpu->transmission = 1.0f;
        gpu->opacity = 1.0f;
        gpu->roughness = wp.roughness;
        gpu->ior = wp.ior;
        gpu->metallic = 0.0f;
        
        gpu->anisotropic = wp.wave_speed;
        gpu->sheen = fmaxf(0.001f, wp.wave_strength);  // > 0 = IS_WATER
        gpu->sheen_tint = wp.wave_frequency;
        
        gpu->clearcoat = wp.shore_foam_intensity;
        gpu->clearcoat_roughness = wp.caustic_intensity;
        gpu->subsurface = wp.depth_max / 100.0f;
        gpu->subsurface_scale = wp.absorption_density;
        gpu->subsurface_color = make_float3(
            wp.absorption_color.x, wp.absorption_color.y, wp.absorption_color.z);
        gpu->subsurface_radius = make_float3(
            wp.shore_foam_distance, wp.caustic_scale, wp.sss_intensity);
        gpu->emission = make_float3(
            wp.shallow_color.x, wp.shallow_color.y, wp.shallow_color.z);
        gpu->translucent = wp.foam_level;
        gpu->subsurface_anisotropy = wp.caustic_speed;
        
        gpu->micro_detail_strength = wp.micro_detail_strength;
        gpu->micro_detail_scale = wp.micro_detail_scale;
        gpu->foam_noise_scale = wp.foam_noise_scale;
        gpu->foam_threshold = wp.foam_threshold;
        gpu->fft_ocean_size = wp.fft_ocean_size;
        gpu->fft_choppiness = wp.fft_choppiness;
        
        water_mat->gpuMaterial = gpu;
        water_mat->materialName = matName;
        
        waterMatID = matMgr.addMaterial(matName, water_mat);
    }
    river->material_id = waterMatID;
    
    // ─────────────────────────────────────────────────────────────────────────
    // STEP 3: Generate vertex grid
    // ─────────────────────────────────────────────────────────────────────────
    int lengthSegs = river->lengthSubdivisions;
    int widthSegs = river->widthSegments;
    
    struct VertexInfo {
        Vec3 position;
        Vec3 normal;
        Vec2 uv;
    };
    
    std::vector<std::vector<VertexInfo>> grid(lengthSegs + 1, std::vector<VertexInfo>(widthSegs + 1));
    
    float accumulatedLength = 0.0f;
    Vec3 prevCenter = river->spline.samplePosition(0);
    
    for (int i = 0; i <= lengthSegs; ++i) {
        float t = (float)i / (float)lengthSegs;
        
        // Sample main properties
        Vec3 centerPos = river->spline.samplePosition(t);
        Vec3 tangent = river->spline.sampleTangent(t);
        Vec3 rightVec = river->spline.sampleRight(t);
        float width = river->spline.sampleUserData1(t);
        float depth = river->spline.sampleUserData2(t);
        
        // Calculate Slope (Vertical descent)
        // Look ahead and behind to smooth slope calculation
        float t_prev = (std::max)(0.0f, t - 0.05f);
        float t_next = (std::min)(1.0f, t + 0.05f);
        Vec3 p_prev = river->spline.samplePosition(t_prev);
        Vec3 p_next = river->spline.samplePosition(t_next);
        
        // Terrain height sampling for slope calculation
        if (river->followTerrain && TerrainManager::getInstance().hasActiveTerrain()) {
            p_prev.y = sampleTerrainHeight(p_prev);
            p_next.y = sampleTerrainHeight(p_next);
        }
        
        float segmentDist = (p_next - p_prev).length();
        float drop = p_prev.y - p_next.y;
        float slope = (segmentDist > 0.001f) ? (drop / segmentDist) : 0.0f;
        
        // Calculate Curvature (Change in direction)
        Vec3 t1 = river->spline.sampleTangent(t_prev);
        Vec3 t2 = river->spline.sampleTangent(t_next);
        float curvature = (1.0f - t1.dot(t2)) * 10.0f; // 0 = straight, higher = curvy
        
        // Determine turning direction (Left or Right) for banking
        // Cross product of tangents gives up vector. If Y is positive/negative tells direction.
        float turnDirection = (t1.x * t2.z - t1.z * t2.x); // Simple 2D cross product component
        
        if (i > 0) {
            accumulatedLength += (centerPos - prevCenter).length();
        }
        prevCenter = centerPos;
        
        float uCoord = accumulatedLength * 0.5f;
        
        // Base Height
        float centerY = centerPos.y;
        if (river->followTerrain) {
            float terrainY = sampleTerrainHeight(centerPos);
            centerY = terrainY + river->bankHeight;
        }
        
        for (int j = 0; j <= widthSegs; ++j) {
            float widthT = (float)j / (float)widthSegs; // 0.0 to 1.0 (Left to Right)
            float offsetMult = (widthT - 0.5f) * 2.0f;  // -1.0 to 1.0
            float offset = offsetMult * (width * 0.5f);
            
            // Calculate XZ position along width
            Vec3 vertPos = centerPos + rightVec * offset;
            
            // Start with base center height
            float finalY = centerY;
            
            // -----------------------------------------------------------------
            // PHYSICAL GEOMETRIC DISPLACEMENT
            // -----------------------------------------------------------------
            
            // -----------------------------------------------------------------
            // PHYSICAL GEOMETRIC DISPLACEMENT
            // -----------------------------------------------------------------
            
            RiverSpline::PhysicsParams& pp = river->physics;
            
            // Clamp Slope to prevent extreme values on short segments
            slope = (std::max)(-1.0f, (std::min)(1.0f, slope)); // Max 45 degrees
            
            // Helper: Simple Pseudo-Noise (Fractal alike) - INCREASED FREQUENCY for ripples instead of waves
            auto getNoise = [&](float x, float z) -> float {
                float v = 0.0f;
                float s = pp.noiseScale * 3.0f; // Scale up base frequency (smaller ripples)
                // Layer 1
                v += sinf(x * 1.0f * s + z * 1.3f * s);
                v += cosf(x * 0.7f * s - z * 1.9f * s);
                // Layer 2 (Detail)
                v += sinf(x * 3.2f * s + z * 2.1f * s) * 0.5f;
                v += cosf(x * 4.5f * s - z * 3.8f * s) * 0.5f;
                // Layer 3 (High freq)
                v += sinf(x * 8.0f * s + z * 9.0f * s) * 0.25f;
                return v;
            };

            // 1. TURBULENCE (Slope Driven - Rapids/Waterfalls)
            if (pp.enableTurbulence && slope > pp.turbulenceThreshold) {
                // Slower ramp up, softer transition
                float rapidIntensity = (slope - pp.turbulenceThreshold) * 2.0f; 
                rapidIntensity = (std::min)(rapidIntensity, 1.0f); // Hard cap at 1.0
                
                rapidIntensity *= pp.turbulenceStrength;
                
                float noiseVal = getNoise(vertPos.x, vertPos.z);
                
                // Add vertical displacement
                finalY += noiseVal * 0.02f * rapidIntensity;
            }
            
            // 2. BANKING (Curvature Driven - Superelevation)
            if (pp.enableBanking) {
                // Reduced multiplier further 2.5 -> 1.5
                float bankingAmount = curvature * offsetMult * (turnDirection * 1.5f); 
                
                // Apply user strength
                bankingAmount *= pp.bankingStrength;
                
                // Clamp banking severely (max 0.2m difference)
                bankingAmount = (std::max)(-0.2f, (std::min)(0.2f, bankingAmount));
                finalY += bankingAmount;
            }
            
            // 3. CENTER BULGE (Flow Driven)
            if (pp.enableFlowBulge) {
                float flowBulge = (1.0f - abs(offsetMult));
                // Cap bulge amount (max 0.1m)
                float bulgeAmount = 0.01f + (std::min)(0.1f, slope * 0.1f); 
                finalY += flowBulge * bulgeAmount * pp.flowBulgeStrength;
            }
            
            // -----------------------------------------------------------------
            
            vertPos.y = finalY;
            
            // Normal will be recalculated in smoothing step, so placeholder is fine
            Vec3 vertNormal(0, 1, 0);
            
            Vec2 uv(uCoord, widthT);
            grid[i][j] = { vertPos, vertNormal, uv };
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // STEP 4: Smooth normals
    // ─────────────────────────────────────────────────────────────────────────
    for (int i = 0; i <= lengthSegs; ++i) {
        for (int j = 0; j <= widthSegs; ++j) {
            Vec3 sumNormal = grid[i][j].normal;
            int count = 1;
            
            if (i > 0) { sumNormal = sumNormal + grid[i-1][j].normal; count++; }
            if (i < lengthSegs) { sumNormal = sumNormal + grid[i+1][j].normal; count++; }
            if (j > 0) { sumNormal = sumNormal + grid[i][j-1].normal; count++; }
            if (j < widthSegs) { sumNormal = sumNormal + grid[i][j+1].normal; count++; }
            
            grid[i][j].normal = (sumNormal * (1.0f / count)).normalize();
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // STEP 5: Create shared transform and triangles
    // ─────────────────────────────────────────────────────────────────────────
    auto sharedTransform = std::make_shared<Transform>();
    sharedTransform->setBase(Matrix4x4::identity());
    
    for (int i = 0; i < lengthSegs; ++i) {
        for (int j = 0; j < widthSegs; ++j) {
            auto& v00 = grid[i][j];
            auto& v10 = grid[i + 1][j];
            auto& v01 = grid[i][j + 1];
            auto& v11 = grid[i + 1][j + 1];
            
            auto tri1 = std::make_shared<Triangle>(
                v00.position, v10.position, v11.position,
                v00.normal, v10.normal, v11.normal,
                v00.uv, v10.uv, v11.uv,
                waterMatID
            );
            tri1->setNodeName(river->name);
            tri1->setTransformHandle(sharedTransform);
            
            auto tri2 = std::make_shared<Triangle>(
                v00.position, v11.position, v01.position,
                v00.normal, v11.normal, v01.normal,
                v00.uv, v11.uv, v01.uv,
                waterMatID
            );
            tri2->setNodeName(river->name);
            tri2->setTransformHandle(sharedTransform);
            
            river->meshTriangles.push_back(tri1);
            river->meshTriangles.push_back(tri2);
            
            scene.world.objects.push_back(tri1);
            scene.world.objects.push_back(tri2);
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // STEP 6: Register or Update WaterSurface in WaterManager
    // ─────────────────────────────────────────────────────────────────────────
    
    // Check if surface already exists (e.g. loaded from project)
    WaterSurface* existingSurf = waterMgr.getWaterSurface(expectedWaterID);
    
    if (existingSurf) {
        // UPDATE Existing Surface
        existingSurf->name = river->name;
        existingSurf->params = river->waterParams;
        existingSurf->material_id = waterMatID;
        existingSurf->mesh_triangles = river->meshTriangles;
        
        if (!river->meshTriangles.empty()) {
            existingSurf->reference_triangle = river->meshTriangles[0];
        }
        
        river->waterSurfaceId = existingSurf->id;
        
        SCENE_LOG_INFO("[RiverManager] Updated existing WaterSurface '" + river->name + 
                       "' (ID: " + std::to_string(existingSurf->id) + ")");
    } 
    else {
        // CREATE New Surface
        WaterSurface waterSurf;
        waterSurf.id = expectedWaterID;
        waterSurf.name = river->name;
        waterSurf.params = river->waterParams;
        waterSurf.material_id = waterMatID;
        waterSurf.mesh_triangles = river->meshTriangles;
        waterSurf.fft_state = nullptr;
        
        if (!river->meshTriangles.empty()) {
            waterSurf.reference_triangle = river->meshTriangles[0];
        }
        
        // Add to WaterManager's list
        waterMgr.getWaterSurfaces().push_back(waterSurf);
        river->waterSurfaceId = waterSurf.id;
        
        SCENE_LOG_INFO("[RiverManager] Created new WaterSurface '" + river->name + 
                       "' (ID: " + std::to_string(waterSurf.id) + ")");
    }
    
    river->needsRebuild = false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SYNC WATER PARAMS TO WATERSURFACE (Call when params change without rebuild)
// ═══════════════════════════════════════════════════════════════════════════════
void RiverManager::syncWaterParams(RiverSpline* river) {
    if (!river || river->waterSurfaceId < 0) return;
    
    auto* waterSurf = WaterManager::getInstance().getWaterSurface(river->waterSurfaceId);
    if (waterSurf) {
        waterSurf->params = river->waterParams;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UPDATE ALL RIVERS
// ═══════════════════════════════════════════════════════════════════════════════
void RiverManager::updateAllRivers(SceneData& scene) {
    for (auto& river : rivers) {
        if (river.needsRebuild) {
            generateMesh(&river, scene);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REMOVE RIVER
// ═══════════════════════════════════════════════════════════════════════════════
void RiverManager::removeRiver(SceneData& scene, int id) {
    auto it = std::find_if(rivers.begin(), rivers.end(),
                           [id](const RiverSpline& r) { return r.id == id; });
    
    if (it == rivers.end()) return;
    
    // Remove WaterSurface from WaterManager
    if (it->waterSurfaceId >= 0) {
        WaterManager::getInstance().removeWaterSurface(scene, it->waterSurfaceId);
    }
    
    // Remove triangles from scene (in case WaterManager didn't)
    for (auto& tri : it->meshTriangles) {
        auto objIt = std::find(scene.world.objects.begin(), scene.world.objects.end(),
                               std::static_pointer_cast<Hittable>(tri));
        if (objIt != scene.world.objects.end()) {
            scene.world.objects.erase(objIt);
        }
    }
    it->meshTriangles.clear();
    
    // Remove from list
    rivers.erase(it);
    
    SCENE_LOG_INFO("[RiverManager] Removed river ID: " + std::to_string(id));
}
