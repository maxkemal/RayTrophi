#pragma once

struct SceneData;

namespace TerrainNodesV2 {

struct TerrainContext;

// Main-thread bridge between scene authoring and worker-safe terrain data.
void captureTerrainCurveSnapshots(TerrainContext& terrainContext,
                                  const SceneData& scene);
bool terrainCurveSnapshotsMatch(const TerrainContext& a,
                                const TerrainContext& b);

} // namespace TerrainNodesV2
