#pragma once

#include <array>
#include "json.hpp"

struct UIContext;

namespace audio {
inline constexpr int kAcousticSurfaceDebugView = 15;
// First 14 floats of a padded 64-byte surface-compute source record.
// xyz/gain, min/max/rolloff/enabled, direction/inner radians, outer radians/gain.
using AcousticRenderSource = std::array<float, 14>;

AcousticRenderSource acousticRenderSource(UIContext& ctx);
AcousticRenderSource acousticRenderSource(UIContext& ctx, const nlohmann::json& emitter);
void updateAcousticRenderDebug(UIContext& ctx);
void drawAcousticRenderDebugControls(UIContext& ctx);
} // namespace audio
