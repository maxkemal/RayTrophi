#pragma once
#include "json.hpp"
#include <array>
#include <cstddef>
#include <string>
#include <vector>

struct UIContext;
namespace audio {
struct AcousticPoint {
    double x = 0, y = 0, z = 0;
};
struct AcousticFace {
    AcousticPoint a, b, c;
    std::size_t object = 0;
};
struct AcousticSceneSnapshot {
    std::vector<AcousticFace> faces;
    nlohmann::json objects = nlohmann::json::array();
    std::size_t unsupported = 0;
    std::size_t transient = 0;
    std::size_t hidden = 0;
};
std::vector<std::string> acousticTraceParameters();
nlohmann::json querySceneAcoustics(UIContext& ctx, const nlohmann::json& params);
nlohmann::json traceAcousticSegment(const AcousticSceneSnapshot& scene,
    AcousticPoint source, AcousticPoint listener, double metersPerUnit, double soundSpeed,
    unsigned maxHits, bool thinSurfaces, bool objectBarriers = false);
nlohmann::json queryAcousticDebug(UIContext& ctx, const nlohmann::json& params);
void tickSceneAcoustics(UIContext& ctx);
void drawAcousticDiagnostics(UIContext& ctx, const nlohmann::json& emitter);
} // namespace audio
