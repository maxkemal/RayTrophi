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
#include "TriangleMesh.h"
#include "globals.h"

namespace {
std::shared_ptr<TriangleMesh> gridToFlatMesh(
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>& normals,
    const std::vector<Vec2>& uvs,
    const std::vector<uint32_t>& indices,
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
}
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════════
// CLEAR ALL RIVERS (Also removes WaterSurfaces from WaterManager)
// ═══════════════════════════════════════════════════════════════════════════════
void RiverManager::clear(SceneData* scene) {
    // Remove associated WaterSurfaces from WaterManager BEFORE clearing rivers
    // This ensures FFT handles are properly cleaned up
    if (scene) {
        for (auto& river : rivers) {
            if (river.waterSurfaceId >= 0) {
                WaterManager::getInstance().removeWaterSurface(*scene, river.waterSurfaceId);
            }
        }
    }
    
    rivers.clear();
    next_id = 1;
}

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
    // STEP 1: Remove old flatMesh from scene
    // ─────────────────────────────────────────────────────────────────────────
    if (river->flatMesh) {
        auto it = std::find(scene.world.objects.begin(), scene.world.objects.end(), 
                           river->flatMesh);
        if (it != scene.world.objects.end()) {
            scene.world.objects.erase(it);
        }
        river->flatMesh = nullptr;
    }
    else {
        // Fallback: Check for existing TriangleMesh with this river's name
        auto it = scene.world.objects.begin();
        while (it != scene.world.objects.end()) {
            if (auto tmesh = std::dynamic_pointer_cast<TriangleMesh>(*it)) {
                if (tmesh->nodeName == river->name) {
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
        
        // Explicit Vulkan/CPU water contract. River shading follows ribbon UVs;
        // it must not masquerade as an ocean merely through the legacy sheen bit.
        gpu->flags |= GPU_MAT_FLAG_WATER | GPU_MAT_FLAG_WATER_RIVER;
        gpu->sheen = (std::max)(wp.wave_strength, 0.001f);
        gpu->anisotropic = wp.wave_speed;
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
        gpu->micro_anim_speed = wp.micro_anim_speed;
        gpu->micro_morph_speed = wp.micro_morph_speed;
        gpu->foam_noise_scale = wp.foam_noise_scale;
        gpu->foam_threshold = wp.foam_threshold;
        gpu->fft_ocean_size = wp.fft_ocean_size;
        gpu->fft_choppiness = wp.fft_choppiness;
        gpu->fft_wind_speed = wp.fft_wind_speed;
        gpu->fft_wind_direction = wp.fft_wind_direction;
        gpu->fft_time_scale = wp.fft_time_scale;
        
        // Sync pbsdf properties so Renderer doesn't override with incorrect values
        water_mat->albedoProperty.color = Vec3(wp.deep_color.x, wp.deep_color.y, wp.deep_color.z);
        water_mat->emissionProperty.color = Vec3(wp.shallow_color.x, wp.shallow_color.y, wp.shallow_color.z);
        water_mat->emissionProperty.intensity = 1.0f;

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
        Vec3 flowDirection;
        float depth = 0.0f;
        float flowSpeed = 0.0f;
        float discharge = 0.0f;
        float froude = 0.0f;
        float foamPotential = 0.0f;
        float riverS = 0.0f;
        float riverT = 0.0f;
        float riverWidth = 1.0f;
    };
    
    std::vector<std::vector<VertexInfo>> grid(lengthSegs + 1, std::vector<VertexInfo>(widthSegs + 1));

    // Build a stable 2D ribbon frame before emitting cross sections. Sampling
    // sampleRight() independently at every point makes the frame rotate abruptly
    // around tight Bezier control points; wide adjacent sections then intersect
    // and appear as repeated knots. A capped miter join preserves the apparent
    // width while keeping neighboring cross sections ordered.
    std::vector<Vec3> centerline(static_cast<size_t>(lengthSegs) + 1u);
    std::vector<float> sampledWidths(static_cast<size_t>(lengthSegs) + 1u, 1.0f);
    for (int i = 0; i <= lengthSegs; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(lengthSegs);
        centerline[static_cast<size_t>(i)] = river->spline.samplePosition(t);
        sampledWidths[static_cast<size_t>(i)] = (std::max)(river->spline.sampleUserData1(t), 0.02f);
    }
    for (int i = 1; i < lengthSegs; ++i) {
        Vec3 incoming = centerline[static_cast<size_t>(i)] - centerline[static_cast<size_t>(i - 1)];
        Vec3 outgoing = centerline[static_cast<size_t>(i + 1)] - centerline[static_cast<size_t>(i)];
        incoming.y = 0.0f;
        outgoing.y = 0.0f;
        const float incomingLength = incoming.length();
        const float outgoingLength = outgoing.length();
        if (incomingLength < 1e-5f || outgoingLength < 1e-5f) continue;
        incoming = incoming * (1.0f / incomingLength);
        outgoing = outgoing * (1.0f / outgoingLength);
        const float turnAngle = std::acos(std::clamp(incoming.dot(outgoing), -1.0f, 1.0f));
        if (turnAngle > 0.05f) {
            const float localRadius = (std::min)(incomingLength, outgoingLength) /
                (std::max)(2.0f * std::sin(turnAngle * 0.5f), 0.05f);
            sampledWidths[static_cast<size_t>(i)] = (std::min)(
                sampledWidths[static_cast<size_t>(i)], localRadius * 1.6f);
        }
    }
    // Width changes are hydrologically meaningful, but a one-sample spike is
    // not. Smooth only the sampling noise; endpoints retain authored values.
    for (int pass = 0; pass < 2 && lengthSegs > 2; ++pass) {
        std::vector<float> sourceWidths = sampledWidths;
        for (int i = 1; i < lengthSegs; ++i) {
            sampledWidths[static_cast<size_t>(i)] =
                sourceWidths[static_cast<size_t>(i - 1)] * 0.25f +
                sourceWidths[static_cast<size_t>(i)] * 0.50f +
                sourceWidths[static_cast<size_t>(i + 1)] * 0.25f;
        }
    }
    
    float accumulatedLength = 0.0f;
    Vec3 prevCenter = river->spline.samplePosition(0);
    Vec3 previousRight(0.0f);
    
    for (int i = 0; i <= lengthSegs; ++i) {
        float t = (float)i / (float)lengthSegs;
        
        // Sample main properties
        Vec3 centerPos = centerline[static_cast<size_t>(i)];
        float width = sampledWidths[static_cast<size_t>(i)];
        const RiverSpline::HydraulicPoint hydraulic = river->sampleHydraulics(t);
        const float waterDepth = (std::max)(river->sampleDepth(t), 0.0f);

        const Vec3 previousCenter = centerline[static_cast<size_t>((std::max)(i - 1, 0))];
        const Vec3 nextCenter = centerline[static_cast<size_t>((std::min)(i + 1, lengthSegs))];
        Vec3 incoming = centerPos - previousCenter;
        Vec3 outgoing = nextCenter - centerPos;
        incoming.y = 0.0f;
        outgoing.y = 0.0f;
        if (incoming.length() < 1e-5f) incoming = outgoing;
        if (outgoing.length() < 1e-5f) outgoing = incoming;
        incoming = incoming.length() > 1e-5f ? incoming.normalize() : Vec3(1, 0, 0);
        outgoing = outgoing.length() > 1e-5f ? outgoing.normalize() : incoming;
        Vec3 flowDirection = incoming + outgoing;
        flowDirection = flowDirection.length() > 1e-5f ? flowDirection.normalize() : outgoing;
        Vec3 rightIn = incoming.cross(Vec3(0, 1, 0)).normalize();
        Vec3 rightOut = outgoing.cross(Vec3(0, 1, 0)).normalize();
        if (rightIn.dot(rightOut) < 0.0f) rightOut = rightOut * -1.0f;
        Vec3 rightVec = rightIn + rightOut;
        rightVec = rightVec.length() > 1e-5f ? rightVec.normalize() : rightOut;
        if (i > 0 && previousRight.length() > 1e-5f && rightVec.dot(previousRight) < 0.0f) {
            rightVec = rightVec * -1.0f;
        }
        previousRight = rightVec;
        const float miterDenominator = (std::max)(std::fabs(rightVec.dot(rightOut)), 0.65f);
        const float joinScale = (std::min)(1.0f / miterDenominator, 1.45f);
        
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

        // A section counts as carved when the bed sits meaningfully below the
        // water level; only then is terrain distance the true water column and
        // the shader can trust the depth attribute to hit zero at the waterline.
        const bool terrainActive = TerrainManager::getInstance().hasActiveTerrain();
        const float centerTerrainY = terrainActive ? sampleTerrainHeight(centerPos) : centerY;
        const bool carvedSection = terrainActive && waterDepth > 0.001f &&
            (centerY - centerTerrainY) > 0.35f * waterDepth;

        for (int j = 0; j <= widthSegs; ++j) {
            float widthT = (float)j / (float)widthSegs; // 0.0 to 1.0 (Left to Right)
            float offsetMult = (widthT - 0.5f) * 2.0f;  // -1.0 to 1.0
            float offset = offsetMult * (width * 0.5f) * joinScale;
            
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

            // ── Shore skirt ──────────────────────────────────────────────────
            const float vertexTerrainY = terrainActive
                ? sampleTerrainHeight(vertPos) : finalY;
            if (terrainActive && (j == 0 || j == widthSegs)) {
                // Bury the ribbon edge under the bank so the visible waterline
                // is the terrain intersection curve, not the mesh border. When
                // the bank sits below the surface (edge floating in air), drop
                // the edge just beneath the terrain instead, capped so the
                // surface never folds into a deep curtain.
                const float bankEmbed = std::clamp(waterDepth * 0.4f, 0.04f, 0.35f);
                const float maxDrop = (std::max)(waterDepth, 0.3f);
                const float buried = vertexTerrainY - bankEmbed;
                if (buried < finalY) finalY = (std::max)(buried, centerY - maxDrop);
            }

            vertPos.y = finalY;

            // Normal will be recalculated in smoothing step, so placeholder is fine
            Vec3 vertNormal(0, 1, 0);

            Vec2 uv(uCoord, widthT);
            float crossSectionDepth = waterDepth *
                (1.0f - std::pow((std::min)(std::fabs(offsetMult), 1.0f), 4.0f));
            if (carvedSection) {
                // True water column above the carved bed: reaches zero exactly
                // at the waterline so the shader can fade the interface out.
                crossSectionDepth = (std::max)(centerY - vertexTerrainY, 0.0f);
            }
            grid[i][j] = {vertPos, vertNormal, uv, flowDirection, crossSectionDepth,
                          hydraulic.flowSpeed, hydraulic.discharge, hydraulic.froude,
                          hydraulic.foamPotential, accumulatedLength, widthT,
                          width * joinScale};
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
    // STEP 5: Create shared transform and flat TriangleMesh
    // ─────────────────────────────────────────────────────────────────────────
    auto sharedTransform = std::make_shared<Transform>();
    sharedTransform->setBase(Matrix4x4::identity());
    
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;
    std::vector<Vec3> flowDirections;
    std::vector<float> waterDepths;
    std::vector<float> flowSpeeds;
    std::vector<float> discharges;
    std::vector<float> froudes;
    std::vector<float> foamPotentials;
    std::vector<float> riverS;
    std::vector<float> riverT;
    std::vector<float> riverWidths;
    positions.reserve((lengthSegs + 1) * (widthSegs + 1));
    normals.reserve((lengthSegs + 1) * (widthSegs + 1));
    uvs.reserve((lengthSegs + 1) * (widthSegs + 1));
    flowDirections.reserve((lengthSegs + 1) * (widthSegs + 1));
    waterDepths.reserve((lengthSegs + 1) * (widthSegs + 1));
    flowSpeeds.reserve((lengthSegs + 1) * (widthSegs + 1));
    discharges.reserve((lengthSegs + 1) * (widthSegs + 1));
    froudes.reserve((lengthSegs + 1) * (widthSegs + 1));
    foamPotentials.reserve((lengthSegs + 1) * (widthSegs + 1));
    riverS.reserve((lengthSegs + 1) * (widthSegs + 1));
    riverT.reserve((lengthSegs + 1) * (widthSegs + 1));
    riverWidths.reserve((lengthSegs + 1) * (widthSegs + 1));

    for (int i = 0; i <= lengthSegs; ++i) {
        for (int j = 0; j <= widthSegs; ++j) {
            positions.push_back(grid[i][j].position);
            normals.push_back(grid[i][j].normal);
            uvs.push_back(grid[i][j].uv);
            flowDirections.push_back(grid[i][j].flowDirection);
            waterDepths.push_back(grid[i][j].depth);
            flowSpeeds.push_back(grid[i][j].flowSpeed);
            discharges.push_back(grid[i][j].discharge);
            froudes.push_back(grid[i][j].froude);
            foamPotentials.push_back(grid[i][j].foamPotential);
            riverS.push_back(grid[i][j].riverS);
            riverT.push_back(grid[i][j].riverT);
            riverWidths.push_back(grid[i][j].riverWidth);
        }
    }

    std::vector<uint32_t> indices;
    indices.reserve(lengthSegs * widthSegs * 6);
    int gridW = widthSegs + 1;
    for (int i = 0; i < lengthSegs; ++i) {
        for (int j = 0; j < widthSegs; ++j) {
            uint32_t i00 = i * gridW + j;
            uint32_t i10 = (i + 1) * gridW + j;
            uint32_t i01 = i * gridW + (j + 1);
            uint32_t i11 = (i + 1) * gridW + (j + 1);

            // Triangle 1 (v00, v10, v11)
            indices.push_back(i00);
            indices.push_back(i10);
            indices.push_back(i11);

            // Triangle 2 (v00, v11, v01)
            indices.push_back(i00);
            indices.push_back(i11);
            indices.push_back(i01);
        }
    }

    river->flatMesh = gridToFlatMesh(
        positions,
        normals,
        uvs,
        indices,
        waterMatID,
        sharedTransform,
        river->name
    );

    if (river->flatMesh && river->flatMesh->geometry) {
        auto& geometry = *river->flatMesh->geometry;
        if (!geometry.has_attribute("river_flow_direction")) geometry.add_attribute<Vec3>("river_flow_direction");
        if (!geometry.has_attribute("river_flow_velocity")) geometry.add_attribute<Vec3>("river_flow_velocity");
        if (!geometry.has_attribute("water_depth")) geometry.add_attribute<float>("water_depth");
        if (!geometry.has_attribute("shore_factor")) geometry.add_attribute<float>("shore_factor");
        if (!geometry.has_attribute("river_water_depth")) geometry.add_attribute<float>("river_water_depth");
        if (!geometry.has_attribute("river_flow_speed")) geometry.add_attribute<float>("river_flow_speed");
        if (!geometry.has_attribute("river_discharge")) geometry.add_attribute<float>("river_discharge");
        if (!geometry.has_attribute("river_froude")) geometry.add_attribute<float>("river_froude");
        if (!geometry.has_attribute("river_foam_potential")) geometry.add_attribute<float>("river_foam_potential");
        if (!geometry.has_attribute("river_s")) geometry.add_attribute<float>("river_s");
        if (!geometry.has_attribute("river_t")) geometry.add_attribute<float>("river_t");
        if (!geometry.has_attribute("river_width")) geometry.add_attribute<float>("river_width");
        Vec3* flowAttribute = geometry.get_attribute_data_mut<Vec3>("river_flow_direction");
        Vec3* velocityAttribute = geometry.get_attribute_data_mut<Vec3>("river_flow_velocity");
        float* standardDepthAttribute = geometry.get_attribute_data_mut<float>("water_depth");
        float* shoreAttribute = geometry.get_attribute_data_mut<float>("shore_factor");
        float* depthAttribute = geometry.get_attribute_data_mut<float>("river_water_depth");
        float* speedAttribute = geometry.get_attribute_data_mut<float>("river_flow_speed");
        float* dischargeAttribute = geometry.get_attribute_data_mut<float>("river_discharge");
        float* froudeAttribute = geometry.get_attribute_data_mut<float>("river_froude");
        float* foamAttribute = geometry.get_attribute_data_mut<float>("river_foam_potential");
        float* sAttribute = geometry.get_attribute_data_mut<float>("river_s");
        float* tAttribute = geometry.get_attribute_data_mut<float>("river_t");
        float* widthAttribute = geometry.get_attribute_data_mut<float>("river_width");
        for (size_t vertex = 0; vertex < positions.size(); ++vertex) {
            if (flowAttribute) flowAttribute[vertex] = flowDirections[vertex];
            if (velocityAttribute) velocityAttribute[vertex] = flowDirections[vertex] * flowSpeeds[vertex];
            if (standardDepthAttribute) standardDepthAttribute[vertex] = waterDepths[vertex];
            if (shoreAttribute) shoreAttribute[vertex] = std::fabs(riverT[vertex] * 2.0f - 1.0f);
            if (depthAttribute) depthAttribute[vertex] = waterDepths[vertex];
            if (speedAttribute) speedAttribute[vertex] = flowSpeeds[vertex];
            if (dischargeAttribute) dischargeAttribute[vertex] = discharges[vertex];
            if (froudeAttribute) froudeAttribute[vertex] = froudes[vertex];
            if (foamAttribute) foamAttribute[vertex] = foamPotentials[vertex];
            if (sAttribute) sAttribute[vertex] = riverS[vertex];
            if (tAttribute) tAttribute[vertex] = riverT[vertex];
            if (widthAttribute) widthAttribute[vertex] = riverWidths[vertex];
        }
    }

    if (river->flatMesh) {
        scene.world.objects.push_back(river->flatMesh);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // STEP 6: Register or Update WaterSurface in WaterManager
    // ─────────────────────────────────────────────────────────────────────────
    
    // Check if surface already exists (e.g. loaded from project)
    WaterSurface* existingSurf = waterMgr.getWaterSurface(expectedWaterID);
    
    if (existingSurf) {
        // UPDATE Existing Surface
        existingSurf->name = river->name;
        // The linked WaterSurface is the live material authority. Its complete
        // preset/custom payload may have been edited from Water UI and is also
        // restored before rivers during project load. Pull it into the spline
        // instead of overwriting it with legacy/incomplete river defaults.
        river->waterParams = existingSurf->params;
        existingSurf->material_id = waterMatID;
        existingSurf->flatMesh = river->flatMesh;
        existingSurf->type = WaterSurface::Type::River;
        waterMgr.syncSurfaceMaterial(existingSurf);
        
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
        waterSurf.flatMesh = river->flatMesh;
        waterSurf.type = WaterSurface::Type::River;
        waterSurf.fft_state = nullptr;
        
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
    
    // Remove flatMesh from scene (in case WaterManager didn't)
    if (it->flatMesh) {
        auto objIt = std::find(scene.world.objects.begin(), scene.world.objects.end(),
                               it->flatMesh);
        if (objIt != scene.world.objects.end()) {
            scene.world.objects.erase(objIt);
        }
        it->flatMesh = nullptr;
    }
    
    // Remove from list
    rivers.erase(it);
    
    SCENE_LOG_INFO("[RiverManager] Removed river ID: " + std::to_string(id));
}
