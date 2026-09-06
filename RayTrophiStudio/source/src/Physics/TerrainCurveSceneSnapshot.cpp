#include "TerrainCurveSceneSnapshot.h"

#include "TerrainNodesV2.h"
#include "MeshEdit/CurveNodeData.h"
#include "MeshEdit/SplineObject.h"
#include "scene_data.h"

#include <cmath>
#include <functional>
#include <memory>
#include <string>

namespace TerrainNodesV2 {
namespace {

size_t hashSnapshot(const MeshEdit::CurveNodeData& curve) {
    size_t seed = std::hash<std::string>{}(curve.source_name);
    const auto combine = [&seed](float value) {
        seed ^= std::hash<float>{}(value) + static_cast<size_t>(0x9e3779b9u) +
            (seed << 6) + (seed >> 2);
    };
    seed ^= std::hash<size_t>{}(curve.spline.points.size()) + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>{}(static_cast<int>(curve.spline.curveType)) + (seed << 6) + (seed >> 2);
    seed ^= std::hash<bool>{}(curve.spline.isClosed) + (seed << 6) + (seed >> 2);
    for (const auto& point : curve.spline.points) {
        for (float value : {point.position.x, point.position.y, point.position.z,
                            point.tangentIn.x, point.tangentIn.y, point.tangentIn.z,
                            point.tangentOut.x, point.tangentOut.y, point.tangentOut.z,
                            point.userData1}) combine(value);
    }
    for (float knot : curve.spline.knots) combine(knot);
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            combine(curve.local_to_world.m[row][column]);
    return seed;
}

} // namespace

void captureTerrainCurveSnapshots(TerrainContext& terrainContext,
                                  const SceneData& scene) {
    terrainContext.curveSnapshots.clear();
    terrainContext.terrainWorldToLocal =
        terrainContext.terrain && terrainContext.terrain->transform
        ? terrainContext.terrain->transform->getFinal().inverse()
        : Matrix4x4::identity();

    for (const auto& object : scene.world.objects) {
        const auto spline = std::dynamic_pointer_cast<MeshEdit::SplineObject>(object);
        if (!spline || spline->nodeName.empty()) continue;
        auto snapshot = std::make_shared<MeshEdit::CurveNodeData>();
        snapshot->spline = spline->spline;
        snapshot->plane = spline->plane;
        snapshot->source_name = spline->nodeName;
        if (spline->transform) {
            snapshot->local_to_world = spline->transform->getFinal();
            snapshot->object_scale = spline->transform->scale;
            snapshot->pivot_offset = spline->transform->pivot_offset;
        }
        snapshot->source_signature = hashSnapshot(*snapshot);
        // Scene names are canonical identifiers. Keep the first match so a
        // duplicate cannot make evaluation depend on container traversal order.
        terrainContext.curveSnapshots.emplace(snapshot->source_name, std::move(snapshot));
    }
}

bool terrainCurveSnapshotsMatch(const TerrainContext& a,
                                const TerrainContext& b) {
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            if (std::abs(a.terrainWorldToLocal.m[row][column] -
                         b.terrainWorldToLocal.m[row][column]) > 1.0e-6f) return false;
    if (a.curveSnapshots.size() != b.curveSnapshots.size()) return false;
    for (const auto& [name, curve] : a.curveSnapshots) {
        const auto found = b.curveSnapshots.find(name);
        if (found == b.curveSnapshots.end() || !curve || !found->second ||
            curve->source_signature != found->second->source_signature) return false;
    }
    return true;
}

} // namespace TerrainNodesV2
