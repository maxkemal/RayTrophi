#pragma once
#include "json.hpp"
#include <string>
#include <unordered_map>
#include <vector>

struct AnimationData;
namespace DNA {
class GeometryDetail;
}
namespace RayTrophi {
class NodeHierarchy;
}

namespace RigAuthoring {
struct RigAnatomy;

// Range-of-motion test: bends every limb and spine joint through a fixed set of motions and
// measures how the bound skin deforms at each peak, so weight quality can be judged by
// numbers instead of by scrubbing a clip.
struct RomTestSettings {
    float scale = 1.f;        // Multiplies every test angle, 0.1..1.5.
    int framesPerMotion = 12; // Clip frames from rest to peak and back, 4..120.
    float fps = 24.f;
};

struct RomTestMesh {
    std::string name;
    const DNA::GeometryDetail *geometry = nullptr;
    size_t vertices = 0;
};

// boneIndex maps a hierarchy bone name to the skin index used in skin_weights. The skin is
// evaluated as pose * inverse(rest), the authored-binding contract (offset = inverse rest).
// clip, when given, receives rest/peak/rest keys for every motion.
bool runRomTest(const RayTrophi::NodeHierarchy &hierarchy, const RigAnatomy &anatomy,
                const std::unordered_map<std::string, int> &boneIndex,
                const std::vector<RomTestMesh> &meshes, const RomTestSettings &settings,
                nlohmann::json &report, AnimationData *clip, std::string &error);
} // namespace RigAuthoring
