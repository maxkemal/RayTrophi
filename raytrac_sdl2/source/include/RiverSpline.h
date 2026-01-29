/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          RiverSpline.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
// ═══════════════════════════════════════════════════════════════════════════════
// RIVER SYSTEM - Bezier Spline Based Rivers and Streams
// ═══════════════════════════════════════════════════════════════════════════════
// Uses the generic BezierSpline system for path definition
// ═══════════════════════════════════════════════════════════════════════════════
#pragma once

#include "Vec3.h"
#include "BezierSpline.h"
#include "WaterSystem.h"
#include "Triangle.h"
#include "json.hpp"
#include <vector>
#include <memory>
#include <string>

// ═══════════════════════════════════════════════════════════════════════════════
// RIVER SPLINE
// Uses BezierSpline with userData1 = width, userData2 = depth
// ═══════════════════════════════════════════════════════════════════════════════
struct RiverSpline {
    struct PhysicsParams {
        bool enableTurbulence = false;       // Enable rapid waves (Default: OFF for flat surface)
        bool enableBanking = false;          // Enable curve banking
        bool enableFlowBulge = false;        // Enable center bulge
        
        float turbulenceStrength = 1.0f;     // Intensify/dampen rapids
        float turbulenceThreshold = 0.05f;   // Slope needed to trigger rapids
        float noiseScale = 1.0f;             // Frequency of the turbulence noise
        float bankingStrength = 1.0f;        // Multiplier for curve banking (superelevation)
        float flowBulgeStrength = 1.0f;      // Multiplier for center bulge
    };

    int id = -1;
    std::string name = "River";
    
    // Mesh generation settings
    int lengthSubdivisions = 32;
    int widthSegments = 4;
    float bankHeight = 0.05f;
    bool followTerrain = true;
    
    // Physics
    PhysicsParams physics;
    
    // The underlying bezier spline
    // userData1 = river width at each point
    // userData2 = water depth below terrain
    BezierSpline spline;
    
    // Water visual parameters (shared with WaterSystem)
    WaterWaveParams waterParams;
    
    // Generated mesh
    std::vector<std::shared_ptr<Triangle>> meshTriangles;
    uint16_t material_id = 0;
    bool needsRebuild = true;
    
    // WaterManager integration (river is registered as WaterSurface)
    int waterSurfaceId = -1;  // ID in WaterManager's list
    
    // Editor state
    bool showControlPoints = true;
    bool showSpline = true;
    int selectedPointIndex = -1;
    
  
    
    // ─────────────────────────────────────────────────────────────────────────
    // Convenience methods for river-specific access
    // ─────────────────────────────────────────────────────────────────────────
    
    void addControlPoint(const Vec3& position, float width = 2.0f, float depth = 0.5f) {
        spline.points.emplace_back(position, width);
        spline.points.back().userData2 = depth;
        if (spline.points.size() > 1) {
            spline.calculateAutoTangents();
        }
        needsRebuild = true;
    }
    
    void removeControlPoint(int index) {
        spline.removePoint(index);
        needsRebuild = true;
    }
    
    size_t controlPointCount() const { return spline.pointCount(); }
    
    BezierControlPoint* getControlPoint(int index) {
        if (index >= 0 && index < (int)spline.points.size()) {
            return &spline.points[index];
        }
        return nullptr;
    }
    
    // Sample river properties
    Vec3 samplePosition(float t) const { return spline.samplePosition(t); }
    Vec3 sampleTangent(float t) const { return spline.sampleTangent(t); }
    Vec3 sampleRight(float t) const { return spline.sampleRight(t); }
    float sampleWidth(float t) const { return spline.sampleUserData1(t); }
    float sampleDepth(float t) const { return spline.sampleUserData2(t); }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Serialization helpers
    // ─────────────────────────────────────────────────────────────────────────
    nlohmann::json serializeSpline() const {
        nlohmann::json j;
        j["id"] = id;
        j["name"] = name;
        j["lengthSubdivisions"] = lengthSubdivisions;
        j["widthSegments"] = widthSegments;
        j["bankHeight"] = bankHeight;
        j["followTerrain"] = followTerrain;
        j["isClosed"] = spline.isClosed;
        
        // Control points
        nlohmann::json pointsJson = nlohmann::json::array();
        for (const auto& pt : spline.points) {
            nlohmann::json pj;
            pj["position"] = { pt.position.x, pt.position.y, pt.position.z };
            pj["tangentIn"] = { pt.tangentIn.x, pt.tangentIn.y, pt.tangentIn.z };
            pj["tangentOut"] = { pt.tangentOut.x, pt.tangentOut.y, pt.tangentOut.z };
            pj["width"] = pt.userData1;
            pj["depth"] = pt.userData2;
            pj["autoTangent"] = pt.autoTangent;
            pj["handleMode"] = static_cast<int>(pt.handleMode);
            pointsJson.push_back(pj);
        }
        j["controlPoints"] = pointsJson;
        
        // Water params (basic for now)
        j["waterParams"] = {
            {"wave_speed", waterParams.wave_speed},
            {"wave_strength", waterParams.wave_strength},
            {"clarity", waterParams.clarity},
            {"ior", waterParams.ior}
        };
        
        return j;
    }
    
    void deserializeSpline(const nlohmann::json& j) {
        id = j.value("id", -1);
        name = j.value("name", "River");
        lengthSubdivisions = j.value("lengthSubdivisions", 32);
        widthSegments = j.value("widthSegments", 4);
        bankHeight = j.value("bankHeight", 0.05f);
        followTerrain = j.value("followTerrain", true);
        spline.isClosed = j.value("isClosed", false);
        
        spline.clear();
        if (j.contains("controlPoints")) {
            for (const auto& pj : j["controlPoints"]) {
                BezierControlPoint pt;
                auto pos = pj["position"];
                pt.position = Vec3(pos[0], pos[1], pos[2]);
                
                if (pj.contains("tangentIn")) {
                    auto ti = pj["tangentIn"];
                    pt.tangentIn = Vec3(ti[0], ti[1], ti[2]);
                }
                if (pj.contains("tangentOut")) {
                    auto to = pj["tangentOut"];
                    pt.tangentOut = Vec3(to[0], to[1], to[2]);
                }
                
                pt.userData1 = pj.value("width", 2.0f);
                pt.userData2 = pj.value("depth", 0.5f);
                pt.autoTangent = pj.value("autoTangent", true);
                pt.handleMode = static_cast<BezierControlPoint::HandleMode>(pj.value("handleMode", 2));
                
                spline.points.push_back(pt);
            }
        }
        
        if (j.contains("waterParams")) {
            auto& wp = j["waterParams"];
            waterParams.wave_speed = wp.value("wave_speed", 1.0f);
            waterParams.wave_strength = wp.value("wave_strength", 0.5f);
            waterParams.clarity = wp.value("clarity", 0.8f);
            waterParams.ior = wp.value("ior", 1.333f);
        }
        
        needsRebuild = true;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// RIVER MANAGER
// ═══════════════════════════════════════════════════════════════════════════════
struct SceneData;

class RiverManager {
public:
    static RiverManager& getInstance() {
        static RiverManager instance;
        return instance;
    }
    
    // Create a new river
    RiverSpline* createRiver(const std::string& name = "River") {
        RiverSpline river;
        river.id = next_id++;
        river.name = name + "_" + std::to_string(river.id);
        rivers.push_back(river);
        return &rivers.back();
    }
    
    // Get all rivers
    std::vector<RiverSpline>& getRivers() { return rivers; }
    const std::vector<RiverSpline>& getRivers() const { return rivers; }
    
    // Get river by ID
    RiverSpline* getRiver(int id) {
        for (auto& r : rivers) {
            if (r.id == id) return &r;
        }
        return nullptr;
    }
    
    // Remove river
    void removeRiver(SceneData& scene, int id);
    
    // Generate mesh for river (implemented in .cpp)
    void generateMesh(RiverSpline* river, SceneData& scene);
    
    // Sync water params to WaterManager (call when params change without rebuild)
    void syncWaterParams(RiverSpline* river);
    
    // Update all rivers that need rebuild
    void updateAllRivers(SceneData& scene);
    
    // Sample terrain height at position (interface to TerrainSystem)
    float sampleTerrainHeight(const Vec3& position) const;
    
    // Clear all rivers
    void clear() {
        rivers.clear();
        next_id = 1;
    }
    
    // Serialization
    nlohmann::json serialize() const {
        nlohmann::json j = nlohmann::json::array();
        for (const auto& r : rivers) {
            j.push_back(r.serializeSpline());
        }
        return j;
    }
    
    void deserialize(const nlohmann::json& j, SceneData& scene) {
        clear();
        for (const auto& rj : j) {
            RiverSpline river;
            river.deserializeSpline(rj);
            if (river.id >= next_id) next_id = river.id + 1;
            rivers.push_back(river);
        }
        // Rebuild meshes after deserialization
        updateAllRivers(scene);
    }
    
    // Editor state
    bool isEditing = false;
    int editingRiverId = -1;
    int selectedControlPoint = -1;
    float defaultWidth = 2.0f;
    float defaultDepth = 0.5f;
    bool isDraggingPoint = false;          // Track if currently dragging a point
    bool showGizmosWhenInactive = false;   // Show river gizmos even when panel not focused
    int lastActiveFrame = 0;               // Frame number when panel was last drawn
    
    // Carve Settings
    float carveDepthMult = 1.0f;
    float carveSmoothness = 0.5f;
    bool carveAutoPostErosion = true;
    int carveErosionIterations = 10;
    bool autoCarveOnMove = false;          // Automatically carve on move end
    
    // Natural Cave Settings (Doğal Nehir Yatağı)
    bool carveEnableNoise = true;          // Add noise-based edge irregularity
    float carveNoiseScale = 0.08f;         // Noise frequency (lower = larger features)
    float carveNoiseStrength = 0.15f;      // How much noise affects depth (0-1)
    
    bool carveEnableDeepPools = true;      // Random deep pools along river
    float carvePoolFrequency = 0.1f;       // How often pools occur (0-1)
    float carvePoolDepthMult = 1.3f;       // Pool depth multiplier (reduced from 1.8)
    
    bool carveEnableRiffles = true;        // Shallow riffle zones
    float carveRiffleFrequency = 0.15f;    // How often riffles occur (0-1)
    float carveRiffleDepthMult = 0.5f;     // Riffle depth multiplier
    
    bool carveEnableAsymmetry = true;      // Asymmetric bank profiles (meander physics)
    float carveAsymmetryStrength = 0.4f;   // How much inner/outer banks differ (reduced)
    
    bool carveEnablePointBars = true;      // Point bar deposits on inner bends
    float carvePointBarStrength = 0.3f;    // How much point bars raise terrain (reduced)
    
    // Terrain Backup for Undo/Reset
    std::vector<float> terrainBackupData;
    int terrainBackupWidth = 0;
    int terrainBackupHeight = 0;
    int terrainBackupId = -1;
    bool hasTerrainBackup = false;
    
private:
    RiverManager() = default;
    
    std::vector<RiverSpline> rivers;
    int next_id = 1;
};

