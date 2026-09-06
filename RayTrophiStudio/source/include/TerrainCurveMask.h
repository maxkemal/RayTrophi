#pragma once

#include "MeshEdit/CurveNodeData.h"
#include "NodeSystem/NodeCore.h"

#include <string>
#include <vector>

namespace TerrainNodesV2 {

enum class CurveMaskMode : int {
    Stroke = 0,
    ClosedFill = 1
};

struct CurveMaskSettings {
    CurveMaskMode mode = CurveMaskMode::Stroke;
    float widthMeters = 4.0f;
    float falloffMeters = 2.0f;
    bool usePointWidth = true;
    bool invert = false;
};

struct TerrainCurveSample {
    float x = 0.0f;
    float z = 0.0f;
    float widthMultiplier = 1.0f;
    float distanceMeters = 0.0f;
};

// Shared metric sampling contract for every curve-driven terrain consumer.
bool sampleTerrainCurveXZ(const MeshEdit::CurveNodeData& curve,
                          const Matrix4x4& terrainWorldToLocal,
                          float cellSizeMeters,
                          bool usePointWidth,
                          std::vector<TerrainCurveSample>& samples,
                          std::string* error = nullptr);

// Pure CPU conversion. The curve and transform are immutable main-thread
// snapshots, so this function is safe to call from terrain worker evaluation.
bool rasterizeCurveMask(const MeshEdit::CurveNodeData& curve,
                        const Matrix4x4& terrainWorldToLocal,
                        int width,
                        int height,
                        float terrainScaleXZ,
                        const CurveMaskSettings& settings,
                        NodeSystem::Image2DData& output,
                        std::string* error = nullptr);

} // namespace TerrainNodesV2
