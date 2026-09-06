/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiRoadMesh.cpp
 * License:       MIT
 * =========================================================================
 * Road route measurement and the optional road surface mesh.
 *
 * Both read the SOLVED route out of the terrain graph's Road Network node.
 * There is deliberately no second sampling of the curve here: a mesh built from
 * its own pass would drift from the published fields on exactly the bends and
 * side slopes where the misalignment is visible, and it would look like a
 * modelling problem rather than the sampling problem it is.
 */

#include "Api/RtApi.h"
#include "Api/RtApiInternal.h"
#include "MeshEdit/ProfileAuthoringService.h"
#include "NodeSystem/Graph.h"
#include "ProjectManager.h"
#include "TerrainManager.h"
#include "TerrainRoadMesh.h"
#include "TerrainRoadNetwork.h"
#include "TerrainRoadNetworkNode.h"
#include "Transform.h"
#include "TriangleMesh.h"

#include <algorithm>
#include <cmath>

namespace rtapi {
namespace {

struct SolvedRouteLookup {
    TerrainObject* terrain = nullptr;
    std::shared_ptr<const TerrainNodesV2::RoadCarveResult> result;
    const TerrainNodesV2::RoadSolvedRoute* route = nullptr;
};

// A road is addressable by the spline it was assigned to, so the lookup matches
// on the route LABEL the network solver stamped. The single-curve Road Carve
// node carries no label - it has no assignment - so it is not addressable here,
// and saying so is better than silently returning some other road's route.
bool findSolvedRoute(const std::string& splineObject, SolvedRouteLookup& out,
                     std::string& error) {
    bool sawNetworkNode = false;
    for (auto& terrain : TerrainManager::getInstance().getTerrains()) {
        if (!terrain.nodeGraph) continue;
        // The solve runs on a worker. Reading a node's cached result while that
        // worker is replacing it is a race, and the shape of the failure would
        // be a mesh built from half of two different solves - geometry that
        // looks merely wrong rather than stale.
        if (terrain.nodeGraph->isEvaluatingAsync()) {
            error = "terrain '" + terrain.name + "' is still evaluating; "
                    "wait for terrain.evaluation_status before reading a road route";
            return false;
        }
        for (const auto& node : terrain.nodeGraph->nodes) {
            auto* network = dynamic_cast<TerrainNodesV2::TerrainRoadNetworkNode*>(node.get());
            if (!network) continue;
            sawNetworkNode = true;
            auto solved = network->latestResult();
            if (!solved) continue;
            for (const auto& route : solved->routes) {
                if (route.label != splineObject) continue;
                out.terrain = &terrain;
                out.result = solved;
                out.route = &route;
                return true;
            }
        }
    }
    error = sawNetworkNode
        ? "no solved route for '" + splineObject +
          "': the Road Network node has not carved this assignment yet "
          "(check that it is enabled and that its curve exists, then evaluate the graph)"
        : "no Road Network node in any terrain graph; add TerrainV2.RoadNetwork and "
          "evaluate before asking for a road route";
    return false;
}

TriangleMesh* findMeshObject(const std::string& name) {
    if (!g_ctx || name.empty()) return nullptr;
    for (const auto& object : g_ctx->scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (mesh && mesh->nodeName == name) return mesh.get();
    }
    return nullptr;
}

} // namespace

Result getRoadRoute(const std::string& spline_object, int max_samples, RoadRouteInfo& out) {
    out = RoadRouteInfo{};
    if (!g_ctx) return notBound();
    SolvedRouteLookup lookup;
    std::string error;
    if (!findSolvedRoute(spline_object, lookup, error)) return Result::fail(error);

    out.spline_object = spline_object;
    out.terrain = lookup.terrain ? lookup.terrain->name : std::string();
    out.revision = lookup.result->revision;
    out.peak_cut_meters = lookup.result->peakCutMeters;
    out.peak_fill_meters = lookup.result->peakFillMeters;
    out.crossing_diagnostic = lookup.result->crossingDiagnostic;

    const auto& samples = lookup.route->samples;
    out.sample_count = static_cast<int>(samples.size());
    const float scaleY = lookup.terrain ? lookup.terrain->heightmap.scale_y : 1.0f;
    for (const auto& sample : samples) {
        switch (sample.crossing) {
            case TerrainNodesV2::RoadCrossingMode::Bridge: ++out.bridge_samples; break;
            case TerrainNodesV2::RoadCrossingMode::Ford:   ++out.ford_samples;   break;
            case TerrainNodesV2::RoadCrossingMode::Tunnel: ++out.tunnel_samples; break;
            default: break;
        }
    }
    if (!samples.empty()) out.length_meters = samples.back().distanceMeters;
    if (max_samples <= 0) return Result::success();

    // Decimated, with both endpoints kept. A truncated head would make "the
    // route ends here" and "the listing ends here" the same observation.
    const size_t wanted = (std::min)(static_cast<size_t>(max_samples), samples.size());
    out.samples.reserve(wanted);
    for (size_t i = 0; i < wanted; ++i) {
        const size_t index = wanted <= 1
            ? 0
            : static_cast<size_t>(std::llround(static_cast<double>(i) *
                  static_cast<double>(samples.size() - 1) /
                  static_cast<double>(wanted - 1)));
        const auto& sample = samples[index];
        RoadRouteSampleInfo info;
        info.x = sample.x;
        info.z = sample.z;
        info.height_meters = sample.height * scaleY;
        info.ground_meters = sample.groundHeight * scaleY;
        info.distance_meters = sample.distanceMeters;
        info.crossing = TerrainNodesV2::roadCrossingModeName(sample.crossing);
        out.samples.push_back(std::move(info));
    }
    return Result::success();
}

Result buildRoadMesh(const std::string& spline_object, const RoadMeshOptions& options,
                     RoadMeshInfo& out) {
    out = RoadMeshInfo{};
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!std::isfinite(options.surface_offset) || options.surface_offset < 0.0f)
        return Result::fail("surface_offset must be finite and non-negative");
    if (!std::isfinite(options.uv_meters_per_tile) || options.uv_meters_per_tile <= 0.0f)
        return Result::fail("uv_meters_per_tile must be greater than zero");

    auto& registry = TerrainNodesV2::RoadNetworkRegistry::getInstance();
    const auto* assignment = registry.find(spline_object);
    if (!assignment)
        return Result::fail("no road assignment for '" + spline_object + "'");

    SolvedRouteLookup lookup;
    std::string error;
    if (!findSolvedRoute(spline_object, lookup, error)) return Result::fail(error);

    TerrainNodesV2::RoadMeshSettings settings;
    settings.includeShoulder = options.include_shoulder;
    settings.surfaceOffsetMeters = options.surface_offset;
    settings.uvMetersPerTile = options.uv_meters_per_tile;
    settings.skipTunnels = options.skip_tunnels;

    const Matrix4x4 localToWorld = (lookup.terrain && lookup.terrain->transform)
        ? lookup.terrain->transform->getFinal() : Matrix4x4::identity();
    const float scaleY = lookup.terrain ? lookup.terrain->heightmap.scale_y : 1.0f;

    std::shared_ptr<DNA::GeometryDetail> geometry;
    TerrainNodesV2::RoadMeshStats stats;
    std::string buildError;
    if (!TerrainNodesV2::buildRoadRibbonGeometry(*lookup.route, scaleY, localToWorld,
                                                 settings, geometry, stats, &buildError)) {
        return Result::fail(buildError);
    }

    // Rebuilding replaces the geometry of the object this assignment already
    // owns. Publishing a new object every time is how repeated generation leaves
    // a stack of stale roads in the scene, each one looking correct.
    const std::string desired = !options.object.empty()
        ? options.object
        : (!assignment->meshObject.empty() ? assignment->meshObject
                                           : spline_object + "_RoadSurface");
    if (TriangleMesh* existing = findMeshObject(desired)) {
        existing->geometry = std::move(geometry);
        existing->build_local_bvh();
        scheduleSceneMutationRebuilds(*g_ctx, true);
        out.object_name = desired;
        out.replaced_existing = true;
    } else {
        if (!g_history) return Result::fail("scene history is unavailable");
        const MeshEdit::ProfilePublishResult published = MeshEdit::publishGeneratedProfile(
            *g_ctx, *g_history, std::move(geometry), desired, "terrain.road.mesh");
        if (!published.report.ok) {
            return Result::fail(published.report.diagnostics.empty()
                ? std::string("road surface mesh could not be published")
                : published.report.diagnostics.front().code + ": " +
                  published.report.diagnostics.front().message);
        }
        out.object_name = published.object_name;
    }

    registry.setMeshObject(spline_object, out.object_name);
    out.vertex_count = stats.vertexCount;
    out.triangle_count = stats.triangleCount;
    out.span_count = stats.spanCount;
    out.length_meters = stats.lengthMeters;
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result clearRoadMesh(const std::string& spline_object) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& registry = TerrainNodesV2::RoadNetworkRegistry::getInstance();
    const auto* assignment = registry.find(spline_object);
    if (!assignment)
        return Result::fail("no road assignment for '" + spline_object + "'");
    if (assignment->meshObject.empty())
        return Result::fail("'" + spline_object + "' owns no generated road surface");
    const std::string meshObject = assignment->meshObject;
    // The record is cleared even when the object is already gone: an ownership
    // entry pointing at nothing would make the next build refuse to replace a
    // mesh that does not exist.
    registry.setMeshObject(spline_object, std::string());
    if (findMeshObject(meshObject)) {
        if (Result deleted = deleteObject(meshObject); !deleted) return deleted;
    }
    ProjectManager::getInstance().markModified();
    return Result::success();
}

} // namespace rtapi
