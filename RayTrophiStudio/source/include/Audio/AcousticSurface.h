#pragma once
#include "Audio/AcousticGpu.h"
#include "Audio/AcousticRenderDebug.h"
#include <memory>
#include "json.hpp"

struct UIContext;
namespace audio {
struct alignas(16) AcousticSurfaceInstance {
    AcousticGpuInstance geometry;
    float reflection[8]{}; // diffuse share only; total reflection is split by scattering
    float specular[8]{};
};
static_assert(sizeof(AcousticSurfaceInstance) == 128, "Acoustic surface shader ABI");
struct AcousticSurfaceBatch {
    std::shared_ptr<const std::vector<AcousticSurfaceInstance>> instances;
    std::array<AcousticRenderSource, 4> sources{};
    unsigned sourceCount = 0;
    int band = -1;
    unsigned reflectionSamples = 1;
    std::string stamp;
    std::string error;
    std::function<bool()> stillCurrent;
};
void resolveAcousticGpuResponses(UIContext& ctx, AcousticGpuGeometry& geometry);
nlohmann::json acousticSurfaceStatus(UIContext& ctx);
} // namespace audio
