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
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════════
// RIVER SPLINE
// Uses BezierSpline with userData1 = width, userData2 = depth
// ═══════════════════════════════════════════════════════════════════════════════
struct RiverSpline {
    struct HydraulicPoint {
        float discharge = 0.0f;       // m3/s
        float flowSpeed = 0.0f;       // m/s
        float froude = 0.0f;
        float foamPotential = 0.0f;
        float surfaceElevation = 0.0f;
    };

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
    // Kept parallel to spline.points so manual rivers remain compatible while
    // generated rivers can carry physically meaningful shader/simulation data.
    std::vector<HydraulicPoint> hydraulics;
    
    // Water visual parameters (shared with WaterSystem)
    WaterWaveParams waterParams;
    
    // Generated mesh
    std::shared_ptr<TriangleMesh> flatMesh;
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
        hydraulics.emplace_back();
        if (spline.points.size() > 1) {
            spline.calculateAutoTangents();
        }
        needsRebuild = true;
    }
    
    void removeControlPoint(int index) {
        spline.removePoint(index);
        if (index >= 0 && index < static_cast<int>(hydraulics.size())) {
            hydraulics.erase(hydraulics.begin() + index);
        }
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

    void setHydraulicPoint(int index, const HydraulicPoint& value) {
        if (index < 0 || index >= static_cast<int>(spline.points.size())) return;
        if (hydraulics.size() < spline.points.size()) hydraulics.resize(spline.points.size());
        hydraulics[static_cast<size_t>(index)] = value;
    }

    HydraulicPoint sampleHydraulics(float t) const {
        if (hydraulics.empty()) return {};
        if (hydraulics.size() == 1 || spline.points.size() <= 1) return hydraulics.front();
        const float scaled = (std::min)((std::max)(t, 0.0f), 1.0f) *
            static_cast<float>(hydraulics.size() - 1);
        const size_t first = (std::min)(static_cast<size_t>(scaled), hydraulics.size() - 1);
        const size_t second = (std::min)(first + 1, hydraulics.size() - 1);
        const float blend = scaled - static_cast<float>(first);
        const auto lerp = [blend](float a, float b) { return a + (b - a) * blend; };
        HydraulicPoint value;
        value.discharge = lerp(hydraulics[first].discharge, hydraulics[second].discharge);
        value.flowSpeed = lerp(hydraulics[first].flowSpeed, hydraulics[second].flowSpeed);
        value.froude = lerp(hydraulics[first].froude, hydraulics[second].froude);
        value.foamPotential = lerp(hydraulics[first].foamPotential, hydraulics[second].foamPotential);
        value.surfaceElevation = lerp(hydraulics[first].surfaceElevation, hydraulics[second].surfaceElevation);
        return value;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Serialization helpers
    // ─────────────────────────────────────────────────────────────────────────
    nlohmann::json serializeSpline(const WaterWaveParams* currentWaterParams = nullptr) const {
        nlohmann::json j;
        j["id"] = id;
        j["name"] = name;
        j["lengthSubdivisions"] = lengthSubdivisions;
        j["widthSegments"] = widthSegments;
        j["bankHeight"] = bankHeight;
        j["followTerrain"] = followTerrain;
        j["isClosed"] = spline.isClosed;
        j["showControlPoints"] = showControlPoints;
        j["showSpline"] = showSpline;

        j["physics"] = {
            {"enableTurbulence", physics.enableTurbulence},
            {"enableBanking", physics.enableBanking},
            {"enableFlowBulge", physics.enableFlowBulge},
            {"turbulenceStrength", physics.turbulenceStrength},
            {"turbulenceThreshold", physics.turbulenceThreshold},
            {"noiseScale", physics.noiseScale},
            {"bankingStrength", physics.bankingStrength},
            {"flowBulgeStrength", physics.flowBulgeStrength}
        };
        
        // Control points
        nlohmann::json pointsJson = nlohmann::json::array();
        for (size_t pointIndex = 0; pointIndex < spline.points.size(); ++pointIndex) {
            const auto& pt = spline.points[pointIndex];
            const HydraulicPoint hydraulic = pointIndex < hydraulics.size()
                ? hydraulics[pointIndex] : HydraulicPoint{};
            nlohmann::json pj;
            pj["position"] = { pt.position.x, pt.position.y, pt.position.z };
            pj["tangentIn"] = { pt.tangentIn.x, pt.tangentIn.y, pt.tangentIn.z };
            pj["tangentOut"] = { pt.tangentOut.x, pt.tangentOut.y, pt.tangentOut.z };
            pj["width"] = pt.userData1;
            pj["depth"] = pt.userData2;
            pj["discharge"] = hydraulic.discharge;
            pj["flowSpeed"] = hydraulic.flowSpeed;
            pj["froude"] = hydraulic.froude;
            pj["foamPotential"] = hydraulic.foamPotential;
            pj["surfaceElevation"] = hydraulic.surfaceElevation;
            pj["autoTangent"] = pt.autoTangent;
            pj["handleMode"] = static_cast<int>(pt.handleMode);
            pointsJson.push_back(pj);
        }
        j["controlPoints"] = pointsJson;
        
        const WaterWaveParams& savedWaterParams = currentWaterParams ? *currentWaterParams : waterParams;
        j["waterParams"] = savedWaterParams.serializeParams();
        
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
        showControlPoints = j.value("showControlPoints", showControlPoints);
        showSpline = j.value("showSpline", showSpline);

        if (j.contains("physics") && j["physics"].is_object()) {
            const auto& p = j["physics"];
            physics.enableTurbulence = p.value("enableTurbulence", physics.enableTurbulence);
            physics.enableBanking = p.value("enableBanking", physics.enableBanking);
            physics.enableFlowBulge = p.value("enableFlowBulge", physics.enableFlowBulge);
            physics.turbulenceStrength = p.value("turbulenceStrength", physics.turbulenceStrength);
            physics.turbulenceThreshold = p.value("turbulenceThreshold", physics.turbulenceThreshold);
            physics.noiseScale = p.value("noiseScale", physics.noiseScale);
            physics.bankingStrength = p.value("bankingStrength", physics.bankingStrength);
            physics.flowBulgeStrength = p.value("flowBulgeStrength", physics.flowBulgeStrength);
        }
        
        spline.clear();
        hydraulics.clear();
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
                HydraulicPoint hydraulic;
                hydraulic.discharge = pj.value("discharge", 0.0f);
                hydraulic.flowSpeed = pj.value("flowSpeed", 0.0f);
                hydraulic.froude = pj.value("froude", 0.0f);
                hydraulic.foamPotential = pj.value("foamPotential", 0.0f);
                hydraulic.surfaceElevation = pj.value("surfaceElevation", pt.position.y);
                hydraulics.push_back(hydraulic);
            }
        }
        
        if (j.contains("waterParams")) {
            auto& wp = j["waterParams"];
            waterParams.deserializeParams(wp);
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
    
    // Clear all rivers (implemented in .cpp - also clears WaterManager surfaces)
    void clear(SceneData* scene = nullptr);
    
    // Serialization
    nlohmann::json serialize() const {
        nlohmann::json riverItems = nlohmann::json::array();
        for (const auto& r : rivers) {
            const WaterSurface* linkedSurface = r.waterSurfaceId >= 0
                ? WaterManager::getInstance().getWaterSurface(r.waterSurfaceId) : nullptr;
            riverItems.push_back(r.serializeSpline(linkedSurface ? &linkedSurface->params : nullptr));
        }
        return {
            {"version", 2},
            {"items", riverItems},
            {"authoring", {
                {"defaultWidth", defaultWidth}, {"defaultDepth", defaultDepth},
                {"showGizmosWhenInactive", showGizmosWhenInactive},
                {"autoCarveOnMove", autoCarveOnMove},
                {"carveDepthMult", carveDepthMult}, {"carveSmoothness", carveSmoothness},
                {"carveAutoPostErosion", carveAutoPostErosion},
                {"carveErosionIterations", carveErosionIterations},
                {"carveEnableNoise", carveEnableNoise}, {"carveNoiseScale", carveNoiseScale},
                {"carveNoiseStrength", carveNoiseStrength},
                {"carveEnableDeepPools", carveEnableDeepPools},
                {"carvePoolFrequency", carvePoolFrequency}, {"carvePoolDepthMult", carvePoolDepthMult},
                {"carveEnableRiffles", carveEnableRiffles},
                {"carveRiffleFrequency", carveRiffleFrequency}, {"carveRiffleDepthMult", carveRiffleDepthMult},
                {"carveEnableAsymmetry", carveEnableAsymmetry},
                {"carveAsymmetryStrength", carveAsymmetryStrength},
                {"carveEnablePointBars", carveEnablePointBars},
                {"carvePointBarStrength", carvePointBarStrength}
            }}
        };
    }
    
    void deserialize(const nlohmann::json& j, SceneData& scene) {
        clear();
        // Deterministic defaults for legacy array-only projects; do not leak
        // authoring state from the previously opened project.
        defaultWidth = 2.0f; defaultDepth = 0.5f; showGizmosWhenInactive = false;
        autoCarveOnMove = false; carveDepthMult = 1.0f; carveSmoothness = 0.5f;
        carveAutoPostErosion = true; carveErosionIterations = 10;
        carveEnableNoise = true; carveNoiseScale = 0.08f; carveNoiseStrength = 0.15f;
        carveEnableDeepPools = true; carvePoolFrequency = 0.1f; carvePoolDepthMult = 1.3f;
        carveEnableRiffles = true; carveRiffleFrequency = 0.15f; carveRiffleDepthMult = 0.5f;
        carveEnableAsymmetry = true; carveAsymmetryStrength = 0.4f;
        carveEnablePointBars = true; carvePointBarStrength = 0.3f;
        const nlohmann::json* riverItems = &j;
        if (j.is_object()) {
            if (j.contains("items") && j["items"].is_array()) riverItems = &j["items"];
            if (j.contains("authoring") && j["authoring"].is_object()) {
                const auto& a = j["authoring"];
                defaultWidth = a.value("defaultWidth", defaultWidth);
                defaultDepth = a.value("defaultDepth", defaultDepth);
                showGizmosWhenInactive = a.value("showGizmosWhenInactive", showGizmosWhenInactive);
                autoCarveOnMove = a.value("autoCarveOnMove", autoCarveOnMove);
                carveDepthMult = a.value("carveDepthMult", carveDepthMult);
                carveSmoothness = a.value("carveSmoothness", carveSmoothness);
                carveAutoPostErosion = a.value("carveAutoPostErosion", carveAutoPostErosion);
                carveErosionIterations = a.value("carveErosionIterations", carveErosionIterations);
                carveEnableNoise = a.value("carveEnableNoise", carveEnableNoise);
                carveNoiseScale = a.value("carveNoiseScale", carveNoiseScale);
                carveNoiseStrength = a.value("carveNoiseStrength", carveNoiseStrength);
                carveEnableDeepPools = a.value("carveEnableDeepPools", carveEnableDeepPools);
                carvePoolFrequency = a.value("carvePoolFrequency", carvePoolFrequency);
                carvePoolDepthMult = a.value("carvePoolDepthMult", carvePoolDepthMult);
                carveEnableRiffles = a.value("carveEnableRiffles", carveEnableRiffles);
                carveRiffleFrequency = a.value("carveRiffleFrequency", carveRiffleFrequency);
                carveRiffleDepthMult = a.value("carveRiffleDepthMult", carveRiffleDepthMult);
                carveEnableAsymmetry = a.value("carveEnableAsymmetry", carveEnableAsymmetry);
                carveAsymmetryStrength = a.value("carveAsymmetryStrength", carveAsymmetryStrength);
                carveEnablePointBars = a.value("carveEnablePointBars", carveEnablePointBars);
                carvePointBarStrength = a.value("carvePointBarStrength", carvePointBarStrength);
            }
        }
        if (!riverItems->is_array()) return;
        for (const auto& rj : *riverItems) {
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

