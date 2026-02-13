/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          HairStrand.h
 * Author:        Kemal Demirtaş
 * Description:   Hair/Fur strand data structures
 *                Optimized for GPU ray tracing with curve primitives
 * =========================================================================
 */
#ifndef HAIR_STRAND_H
#define HAIR_STRAND_H

#include "Vec3.h"
#include "Vec2.h"
#include <vector>
#include <cstdint>
#include "json.hpp"


namespace Hair {

/**
 * @brief Single control point of a hair strand
 * Memory: 20 bytes per point
 */
struct HairPoint {
    Vec3 position;      // 12 bytes - World position
    float radius;       // 4 bytes  - Strand thickness at this point
    float v_coord;      // 4 bytes  - Parametric V coordinate (0=root, 1=tip)
};

void to_json(nlohmann::json& j, const HairPoint& p);
void from_json(const nlohmann::json& j, HairPoint& p);


/**
 * @brief Hair strand types for different rendering strategies
 */
enum class StrandType : uint8_t {
    GUIDE,          // Master strand for interpolation
    INTERPOLATED,   // Generated from guides
    CLUMP_CENTER,   // Clump attraction center
    FUR_UNDERCOAT,  // Short fuzzy layer
    FUR_GUARD       // Long protective hairs
};

/**
 * @brief Single hair strand (typically 4-16 control points)
 * Uses cubic Bezier or Catmull-Rom for smooth curves
 * 
 * GPU Layout: Stores as flat buffer for coalesced access
 * [P0, P1, P2, P3] [P0, P1, P2, P3] ... 
 */
struct HairStrand {
    std::vector<HairPoint> points;  // Control points (root to tip)
    
    uint32_t strandID;              // Unique identifier
    uint16_t materialID;            // Hair material (uses existing MaterialManager)
    StrandType type;                // Guide/interpolated/fur
    
    float rootRadius;               // Radius at scalp
    float tipRadius;                // Radius at tip (usually thinner)
    float randomSeed;               // Per-strand variation seed
    uint16_t meshMaterialID;        // Inherited material from the source mesh (for texture sampling)
    
    // Optional: UV on scalp surface for texture lookup
    Vec2 rootUV;
    
    // Root normal for brush operations
    Vec3 rootNormal;
    
    // Base geometry for non-destructive styling
    Vec3 baseRootPos;               // Anchored position on mesh surface
    float baseLength;               // Original strand length
    std::vector<Vec3> groomedPositions; // Current groomed positions (World Space - Animated)
    std::vector<Vec3> restGroomedPositions; // Bind Pose groomed positions (Relative to Bind Triangle)
    float clumpScale = 1.0f;        // Per-strand clumpiness multiplier

    // Skinning / Binding Data
    uint32_t triangleIndex = 0xFFFFFFFF; // Index of the triangle this strand is bound to
    Vec2 barycentricUV;             // Barycentric coordinates (u, v) on the bound triangle


    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    inline size_t numPoints() const { return points.size(); }
    inline size_t numSegments() const { return points.size() > 1 ? points.size() - 1 : 0; }
    
    inline const Vec3& rootPosition() const { return points.front().position; }
    inline const Vec3& tipPosition() const { return points.back().position; }
    
    inline float length() const {
        float len = 0.0f;
        for (size_t i = 1; i < points.size(); ++i) {
            len += (points[i].position - points[i-1].position).length();
        }
        return len;
    }
    
    // Linear interpolation along strand
    Vec3 evaluatePosition(float t) const;
    float evaluateRadius(float t) const;
};

void to_json(nlohmann::json& j, const HairStrand& s);
void from_json(const nlohmann::json& j, HairStrand& s);


/**
 * @brief GPU-friendly flat hair data for upload
 * Optimized for CUDA/Vulkan buffer binding
 */
struct HairGPUData {
    // Flat arrays for GPU
    std::vector<float> positions;    // [x,y,z, x,y,z, ...] all points
    std::vector<float> radii;        // Per-point radius
    std::vector<uint32_t> offsets;   // Strand start indices
    std::vector<uint16_t> materials; // Per-strand material ID
    std::vector<Vec2> rootUVs;       // Per-strand root UV
    
    size_t totalPoints;
    size_t totalStrands;
    
    void buildFromStrands(const std::vector<HairStrand>& strands);
    void clear();
};

} // namespace Hair

#endif // HAIR_STRAND_H
