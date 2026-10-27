#pragma once

#include "Backend/RealtimeDenoiserTypes.h"
#include <optional>

namespace Backend { class IBackend; }
namespace rtapi {
struct Result;
VulkanRT::RealtimeDenoiserSettings realtimeDenoiserSettings();
Result setRealtimeDenoiser(const std::optional<std::string>& mode,
    std::optional<int> atrous_iterations, std::optional<int> history_max,
    std::optional<float> firefly_clip, std::optional<float> still_noise = std::nullopt,
    std::optional<bool> debug_history = std::nullopt,
    std::optional<bool> scene_motion = std::nullopt);
VulkanRT::RealtimeDenoiserStats realtimeDenoiserStats(bool synchronize = true);
void updateRealtimeDenoiserPolicy(bool cameraMoving, bool finalRender);
bool realtimeDenoiserOwnsViewport(Backend::IBackend* backend);
void drawRealtimeDenoiserControls();
} // namespace rtapi
