#include "HiddenGameObjectsInternal.h"
#include "Paint/HiddenPaintSafety.h"
#include "ParticleSimulation.h"
#include "scene_data.h"
#include <string>

namespace Paint {
HiddenPaintSpawnGuard::HiddenPaintSpawnGuard(SceneData& scene) {
    HiddenDetail::discover(scene);
    for (const auto& item : HiddenDetail::state().entries) {
        if (item.second.visibility != "paint") {
            continue;
        }
        const auto box = HiddenDetail::objectBox(item.first);
        // Covers the contact skin as well as the interior. Conservative for an
        // open / concave toy, intentionally preferring no paint to an inside hit.
        boxes_.emplace_back(box.low - Vec3(0.025f), box.high + Vec3(0.025f));
    }
}
bool HiddenPaintSpawnGuard::allows(const Vec3& position) const {
    for (const auto& box : boxes_) {
        if (position.x >= box.first.x && position.x <= box.second.x &&
            position.y >= box.first.y && position.y <= box.second.y &&
            position.z >= box.first.z && position.z <= box.second.z) {
            return false;
        }
    }
    return true;
}
void clearBalloonPaintInFlight(SceneData& scene) {
    for (auto& entry : scene.particle_systems) {
        const auto& name = entry.name;
        if (entry.runtime && (name == "BalloonPaintRed" || name == "BalloonPaintBlue" ||
            name == "BalloonPaintYellow" || name == "BalloonPaintGreen" ||
            name == "BalloonDropGhosts" || name == "BalloonSparkleTrails")) {
            entry.runtime->clear();
        }
    }
    scene.preserveScriptSimulationPreview();
}
} // namespace Paint
