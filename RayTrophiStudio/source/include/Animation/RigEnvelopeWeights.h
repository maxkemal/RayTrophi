#pragma once

#include "Vec3.h"
#include "json.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class SceneData;
class TriangleMesh;
namespace DNA {
class GeometryDetail;
}

namespace RigAuthoring {
struct EnvelopeWeightSettings {
    float torsoRadius = .16f;
    float limbRadius = .065f;
    float extremityRadius = .05f;
    float falloff = 2.f;
};

struct EnvelopeBoneProfile {
    std::string bone;
    float startRadius = .065f;
    float endRadius = .065f;
    float startExtension = 0.f;
    float endExtension = 0.f;
    float falloff = 2.f;
};

struct EnvelopeWeightPart {
    std::shared_ptr<TriangleMesh> mesh;
    std::shared_ptr<DNA::GeometryDetail> geometry;
};

struct EnvelopeWeightState {
    std::string character;
    std::string algorithm = "anatomical_capsule_v1";
    EnvelopeWeightSettings settings;
    std::vector<EnvelopeBoneProfile> profiles;
    uint64_t revision = 0;
    std::vector<EnvelopeWeightPart> parts;
};

struct EnvelopeSegmentView {
    std::string bone;
    std::string category;
    Vec3 start;
    Vec3 end;
    float startRadius = 0.f;
    float endRadius = 0.f;
};

bool validEnvelopeWeightSettings(const EnvelopeWeightSettings &settings);
bool envelopeSegmentViews(const SceneData &scene, const std::string &character,
                          const EnvelopeWeightSettings &settings,
                          std::vector<EnvelopeSegmentView> &segments, std::string &error);
bool getEnvelopeBoneProfile(const SceneData &scene, const std::string &character,
                            const std::string &bone, EnvelopeBoneProfile &profile,
                            uint64_t &revision, bool &overridden, std::string &error);
bool stageEnvelopeBoneProfile(const SceneData &scene, const std::string &character,
                              const EnvelopeBoneProfile &profile, uint64_t expectedRevision,
                              EnvelopeWeightState &state, nlohmann::json &report,
                              std::string &error);
bool previewEnvelopeWeights(const SceneData &scene, const std::string &character,
                            const EnvelopeWeightSettings &settings, nlohmann::json &report,
                            std::string &error);
bool stageEnvelopeWeights(const SceneData &scene, const std::string &character,
                          const EnvelopeWeightSettings &settings, uint64_t expectedRevision,
                          EnvelopeWeightState &state, nlohmann::json &report, std::string &error);
} // namespace RigAuthoring
