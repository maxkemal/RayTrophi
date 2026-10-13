#pragma once

struct SceneData;

namespace atmosphere {

// Installs the scene query behind resolveLightningStrike (LightningStrike.h):
// meshes through the scene BVH, terrains through TerrainManager, scattered instances
// through InstanceManager. `scene` must outlive the process main loop.
void installLightningStrikeResolver(SceneData& scene);

} // namespace atmosphere
