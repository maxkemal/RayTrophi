#pragma once

#include <string>

struct SceneData;

namespace RigAuthoring {

// Rebuild the model-owned animation runtime from the canonical scene clip list.
// Rig authoring replaces AnimationData snapshots for undo/redo, so retaining the
// previous controller/Ozz pointers would make Anim Graph sample stale clips.
bool synchronizeClipRuntime(SceneData& scene, const std::string& character);

} // namespace RigAuthoring
