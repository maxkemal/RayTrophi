#pragma once

#include "Backend/UpscaleTypes.h"
#include <optional>

namespace rtapi {
struct Result;
VulkanRT::UpscaleSettings upscaleSettings();
Result setUpscale(const std::optional<std::string>& mode,
                  const std::optional<std::string>& quality,
                  std::optional<float> sharpness,
                  std::optional<bool> rejection_reactive = std::nullopt);
VulkanRT::UpscaleStats upscaleStats(bool synchronize = true);
void drawUpscaleControls();
} // namespace rtapi
