/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          HairStrand.cpp
 * Author:        Kemal Demirtaş
 * Description:   Hair strand data structure implementations
 * =========================================================================
 */

#include "Hair/HairStrand.h"
#include <algorithm>
#include <cmath>

namespace Hair {

// ============================================================================
// HairStrand Implementation
// ============================================================================

Vec3 HairStrand::evaluatePosition(float t) const {
    if (points.empty()) return Vec3(0, 0, 0);
    if (points.size() == 1) return points[0].position;
    
    // Clamp t to [0, 1]
    t = std::max(0.0f, std::min(1.0f, t));
    
    // Find segment
    float segmentFloat = t * (points.size() - 1);
    size_t segmentIndex = static_cast<size_t>(segmentFloat);
    float localT = segmentFloat - segmentIndex;
    
    // Clamp segment index
    if (segmentIndex >= points.size() - 1) {
        segmentIndex = points.size() - 2;
        localT = 1.0f;
    }
    
    // Linear interpolation between control points
    // For smoother results, could use Catmull-Rom spline
    const Vec3& p0 = points[segmentIndex].position;
    const Vec3& p1 = points[segmentIndex + 1].position;
    
    return p0 * (1.0f - localT) + p1 * localT;
}

float HairStrand::evaluateRadius(float t) const {
    if (points.empty()) return 0.0f;
    if (points.size() == 1) return points[0].radius;
    
    t = std::max(0.0f, std::min(1.0f, t));
    
    // Simple linear interpolation from root to tip
    return rootRadius * (1.0f - t) + tipRadius * t;
}

// ============================================================================
// HairGPUData Implementation
// ============================================================================

void HairGPUData::buildFromStrands(const std::vector<HairStrand>& strands) {
    clear();
    
    if (strands.empty()) return;
    
    // Count total points
    totalStrands = strands.size();
    totalPoints = 0;
    for (const auto& strand : strands) {
        totalPoints += strand.points.size();
    }
    
    // Reserve memory
    positions.reserve(totalPoints * 3);
    radii.reserve(totalPoints);
    offsets.reserve(totalStrands + 1);
    materials.reserve(totalStrands);
    rootUVs.reserve(totalStrands);
    
    // Build flat arrays
    uint32_t currentOffset = 0;
    for (const auto& strand : strands) {
        offsets.push_back(currentOffset);
        materials.push_back(strand.materialID);
        rootUVs.push_back(strand.rootUV);
        
        for (const auto& point : strand.points) {
            positions.push_back(point.position.x);
            positions.push_back(point.position.y);
            positions.push_back(point.position.z);
            radii.push_back(point.radius);
            currentOffset++;
        }
    }
    
    // Final offset marks end
    offsets.push_back(currentOffset);
}

void HairGPUData::clear() {
    positions.clear();
    radii.clear();
    offsets.clear();
    materials.clear();
    rootUVs.clear();
    totalPoints = 0;
    totalStrands = 0;
}

void from_json(const nlohmann::json& j, HairStrand& s) {
    if (j.contains("points") && j["points"].is_array()) {
        s.points = j["points"].get<std::vector<HairPoint>>();
    }
    s.strandID = j.value("strandID", 0u);
    s.materialID = j.value("materialID", (uint16_t)0);
    s.type = static_cast<StrandType>(j.value("type", 0));
    s.rootRadius = j.value("rootRadius", 0.001f);
    s.tipRadius = j.value("tipRadius", 0.0001f);
    s.randomSeed = j.value("randomSeed", 0.0f);
    
    if (j.contains("rootUV") && j["rootUV"].is_array() && j["rootUV"].size() >= 2) {
        const auto& uv = j["rootUV"];
        s.rootUV = Vec2(uv[0].get<float>(), uv[1].get<float>());
    }
    if (j.contains("rootNormal") && j["rootNormal"].is_array() && j["rootNormal"].size() >= 3) {
        const auto& n = j["rootNormal"];
        s.rootNormal = Vec3(n[0].get<float>(), n[1].get<float>(), n[2].get<float>());
    }
    if (j.contains("baseRootPos") && j["baseRootPos"].is_array() && j["baseRootPos"].size() >= 3) {
        const auto& p = j["baseRootPos"];
        s.baseRootPos = Vec3(p[0].get<float>(), p[1].get<float>(), p[2].get<float>());
    }
    s.baseLength = j.value("baseLength", 0.1f);
    s.clumpScale = j.value("clumpScale", 1.0f);
    s.meshMaterialID = j.value("meshMaterialID", (uint16_t)0xFFFF);
    
    if (j.contains("groomedPositions") && j["groomedPositions"].is_array()) {
        s.groomedPositions.clear();
        for (const auto& pos : j["groomedPositions"]) {
            if (pos.is_array() && pos.size() >= 3) {
                s.groomedPositions.push_back(Vec3(pos[0].get<float>(), pos[1].get<float>(), pos[2].get<float>()));
            }
        }
    }
}



void to_json(nlohmann::json& j, const HairPoint& p) {
    j = nlohmann::json{
        {"pos", {p.position.x, p.position.y, p.position.z}},
        {"rad", p.radius},
        {"v", p.v_coord}
    };
}

void from_json(const nlohmann::json& j, HairPoint& p) {
    if (j.contains("pos") && j["pos"].is_array() && j["pos"].size() >= 3) {
        const auto& arr = j["pos"];
        p.position = Vec3(arr[0].get<float>(), arr[1].get<float>(), arr[2].get<float>());
    }
    p.radius = j.value("rad", 0.001f);
    p.v_coord = j.value("v", 0.0f);
}


void to_json(nlohmann::json& j, const HairStrand& s) {
    j = nlohmann::json{
        {"points", s.points},
        {"strandID", s.strandID},
        {"materialID", s.materialID},
        {"type", static_cast<int>(s.type)},
        {"rootRadius", s.rootRadius},
        {"tipRadius", s.tipRadius},
        {"randomSeed", s.randomSeed},
        {"rootUV", {s.rootUV.u, s.rootUV.v}},
        {"rootNormal", {s.rootNormal.x, s.rootNormal.y, s.rootNormal.z}},
        {"baseRootPos", {s.baseRootPos.x, s.baseRootPos.y, s.baseRootPos.z}},
        {"baseLength", s.baseLength},
        {"clumpScale", s.clumpScale},
        {"meshMaterialID", s.meshMaterialID},
        {"groomedPositions", nlohmann::json::array()}
    };
    
    for (const auto& pos : s.groomedPositions) {
        j["groomedPositions"].push_back({pos.x, pos.y, pos.z});
    }
}


} // namespace Hair

