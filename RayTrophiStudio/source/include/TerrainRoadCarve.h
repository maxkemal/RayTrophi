#pragma once

#include "TerrainCurveMask.h"

#include <cstdint>
#include <string>
#include <vector>

namespace TerrainNodesV2 {

// Where a road stops being carried by the ground. Declared per assignment and
// resolved PER ROUTE SAMPLE by the solver, because a road does not cross a
// valley the same way along its whole length - only the span that needs it.
//
// This enum lives here rather than in TerrainRoadProfile.h because crossing is a
// solver concept: the profile says how wide the road is, the crossing says
// whether the ground under it is touched at all.
enum class RoadCrossingMode : uint8_t {
    Auto = 0,
    Terrain,
    Bridge,
    Ford,
    Tunnel
};

struct RoadCarveSettings {
    float roadWidthMeters = 4.0f;
    float shoulderWidthMeters = 1.5f;
    float gradingFalloffMeters = 3.0f;
    float foliageExclusionMarginMeters = 2.0f;
    float maxGradePercent = 12.0f;
    float elevationOffsetMeters = 0.0f;
    // How far the road may deviate from the ground it crosses. WITHOUT these the
    // grade limiter is unbounded: a 12% road crossing a mountain cannot climb
    // fast enough, so the profile stays near its entry height and the rasterizer
    // faithfully builds what that implies - a canyon through the peak and a
    // mountain-high embankment across the valley. The maths was right; nothing
    // told it that a road is not a viaduct.
    //
    // Defaults are road-engineering scale, not solver scale: an 8 m cut is
    // already a serious excavation and a 6 m fill a serious embankment. An
    // author who wants a viaduct raises them deliberately.
    float maxCutMeters = 8.0f;
    float maxFillMeters = 6.0f;
    // ── Cross-section ────────────────────────────────────────────────────────
    // Crown and ditch are not cosmetic. A ditchless, crownless road is a flat
    // linear TRENCH, which is a perfect channel from a flow solver's point of
    // view - which is exactly why carved roads were being read as river beds.
    // A crowned surface sheds water sideways and the ditch carries it downhill:
    // the road pushes water, the ditch moves it. That is the authority
    // declaration, made as GEOMETRY rather than as a flag, and it is also what
    // real road engineering does for the same reason.
    float crownMeters = 0.12f;        // centreline rise above the shoulder edge
    float ditchWidthMeters = 1.2f;    // 0 disables the ditch entirely
    float ditchDepthMeters = 0.5f;    // invert depth below the shoulder edge
    bool usePointWidth = true;
};

// One solved route sample, in the same terrain-local metric frame the mask
// rasterizer works in. This LEAVES the solver so the optional surface mesh is
// built from the very geometry that carved the terrain - a second sampling pass
// would drift from the fields on exactly the bends where it matters most.
struct RoadRouteSample {
    float x = 0.0f;                // terrain-local metres
    float z = 0.0f;                // terrain-local metres
    float height = 0.0f;           // field units; metres = height * scale_y
    float groundHeight = 0.0f;     // field units, before the road
    float widthMultiplier = 1.0f;
    float distanceMeters = 0.0f;
    RoadCrossingMode crossing = RoadCrossingMode::Terrain;
};

struct RoadSolvedRoute {
    std::string label;
    RoadCarveSettings settings;
    RoadCrossingMode declaredCrossing = RoadCrossingMode::Terrain;
    std::vector<RoadRouteSample> samples;
};

struct RoadCarveResult {
    uint64_t revision = 0;
    // What the solve actually produced. A limit that silently binds is a limit
    // nobody can calibrate, so the node reports the extremes it reached and how
    // much of the route hit a constraint.
    float peakCutMeters = 0.0f;
    float peakFillMeters = 0.0f;
    int envelopeClampedSamples = 0;   // route samples pinned by maxCut/maxFill
    int gradeExceededSamples = 0;     // samples where the envelope beat the grade
    int bridgeSamples = 0;
    int tunnelSamples = 0;
    int fordSamples = 0;
    int routeSampleCount = 0;
    int roadCount = 0;
    // Set when a crossing mode was declared that this solve could not act on -
    // a Ford with no Water field, for instance. Reported rather than silently
    // downgraded to Terrain, because a ford that quietly became a normal road
    // looks entirely plausible in the viewport.
    std::string crossingDiagnostic;
    NodeSystem::Image2DData height;
    NodeSystem::Image2DData roadCore;
    NodeSystem::Image2DData shoulder;
    NodeSystem::Image2DData ditch;
    NodeSystem::Image2DData cut;
    NodeSystem::Image2DData fill;
    NodeSystem::Image2DData foliageExclusion;
    // One entry per solved road, in solve order.
    std::vector<RoadSolvedRoute> routes;
};

// One road in a multi-road solve: a curve plus the profile it carves with.
struct RoadCarveInput {
    const MeshEdit::CurveNodeData* curve = nullptr;
    RoadCarveSettings settings;
    RoadCrossingMode crossing = RoadCrossingMode::Terrain;
    // Reported back in diagnostics so a message names the road the author knows.
    std::string label;
};

// Multi-road solve. All roads share ONE nearest-distance pass, so overlapping
// roads resolve by proximity - the closest road's profile wins a pixel - and a
// junction does not double-carve. This is also what removes the per-curve node
// explosion: N roads are one node, one solve and one snapshot, not N chains.
//
// `waterMask` is optional. When absent, Auto behaves exactly as Terrain and a
// Ford cannot be resolved: water is the only thing that tells the solver where
// a crossing IS, and inventing one from height alone would put bridges over dry
// gullies.
bool solveRoadNetworkCarve(const NodeSystem::Image2DData& sourceHeight,
                           float terrainScaleXZ,
                           float terrainScaleY,
                           const std::vector<RoadCarveInput>& roads,
                           const Matrix4x4& terrainWorldToLocal,
                           const NodeSystem::Image2DData* waterMask,
                           uint64_t revision,
                           RoadCarveResult& result,
                           std::string* error = nullptr);

// Pure, non-destructive solver. Every product is created from the same route
// samples and carries the caller-provided immutable revision as one snapshot.
bool solveRoadCarve(const NodeSystem::Image2DData& sourceHeight,
                    float terrainScaleXZ,
                    float terrainScaleY,
                    const MeshEdit::CurveNodeData& curve,
                    const Matrix4x4& terrainWorldToLocal,
                    const RoadCarveSettings& settings,
                    uint64_t revision,
                    RoadCarveResult& result,
                    std::string* error = nullptr);

} // namespace TerrainNodesV2
