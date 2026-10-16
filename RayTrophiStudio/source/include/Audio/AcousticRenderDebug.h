#pragma once

#include <array>

struct UIContext;

namespace audio {
inline constexpr int kAcousticSurfaceDebugView = 15;
// Scalar push-constant ABI, appended at byte 196 of CameraPC (total 252 bytes).
// xyz/gain, min/max/rolloff/enabled, direction/inner radians, outer radians/gain.
using AcousticRenderSource = std::array<float, 14>;

AcousticRenderSource acousticRenderSource(UIContext& ctx);
void updateAcousticRenderDebug(UIContext& ctx);
void drawAcousticRenderDebugControls(UIContext& ctx);
} // namespace audio
