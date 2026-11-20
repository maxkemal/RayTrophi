#pragma once
#include "Vec3.h"
#include <utility>
#include <vector>
struct SceneData;
namespace Paint {
// Snapshot once per burst: conservative exterior-only spawn policy for reveal
// geometry, without per-particle mesh queries or GPU downloads.
class HiddenPaintSpawnGuard {
public:
    explicit HiddenPaintSpawnGuard(SceneData& scene);
    bool allows(const Vec3& position) const;
private:
    std::vector<std::pair<Vec3, Vec3>> boxes_;
};
// A clean room must discard live demo drops / ghosts before clearing the atlas.
void clearBalloonPaintInFlight(SceneData& scene);
} // namespace Paint
