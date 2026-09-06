#pragma once

// ═══════════════════════════════════════════════════════════════════════════════
// ROAD SURFACE MESH - the optional render ribbon, built from the SOLVED route
// ═══════════════════════════════════════════════════════════════════════════════
// The mesh is generated from the very samples that carved the terrain, not from
// a second pass over the curve. Re-sampling would drift from the published
// fields exactly where alignment is visible - bends and side slopes - and the
// drift would look like a modelling problem rather than a sampling one.
//
// Output is canonical flat geometry (DNA::GeometryDetail). No Triangle facade is
// produced anywhere on this path.

#include "TerrainRoadCarve.h"

#include <memory>
#include <string>

class Matrix4x4;
namespace DNA { class GeometryDetail; }

namespace TerrainNodesV2 {

struct RoadMeshSettings {
    // Shoulders are part of the surface an author usually wants to see; the
    // ditch is not - a ditch is terrain, and giving it a second, floating
    // representation would put two surfaces where the ground already is.
    bool includeShoulder = true;
    // Lifted off the graded surface. The terrain is a discretised heightfield
    // and the ribbon is continuous, so at zero offset the two interpenetrate on
    // every cell boundary and the road reads as torn.
    float surfaceOffsetMeters = 0.05f;
    // Metres of road per V tile.
    float uvMetersPerTile = 4.0f;
    // A bore carries no deck. Skipping tunnel samples splits the ribbon into
    // spans instead of drawing a road through the inside of a mountain.
    bool skipTunnels = true;
};

struct RoadMeshStats {
    size_t vertexCount = 0;
    size_t triangleCount = 0;
    int spanCount = 0;
    float lengthMeters = 0.0f;
    int bridgeSamples = 0;
    int tunnelSamples = 0;
};

// Builds one ribbon for one solved road. Returns false with an explicit error
// when the route is degenerate; it never returns an empty mesh as success.
bool buildRoadRibbonGeometry(const RoadSolvedRoute& route,
                             float terrainScaleY,
                             const Matrix4x4& terrainLocalToWorld,
                             const RoadMeshSettings& settings,
                             std::shared_ptr<DNA::GeometryDetail>& out,
                             RoadMeshStats& stats,
                             std::string* error = nullptr);

} // namespace TerrainNodesV2
