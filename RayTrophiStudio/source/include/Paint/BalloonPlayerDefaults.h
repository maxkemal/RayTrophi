#pragma once

#include "json.hpp"

namespace Paint {

inline constexpr int defaultBalloonCount = 5;
#ifdef __ANDROID__
// Android Player caps authored wet-paint surfaces to 512 on project load
// (matches Android 512 texture staging; 4x less GPU memory and Mali paint cost).
// F2 / game settings resize can still override per session.
inline constexpr int defaultMaxPaintResolution = 512;
#else
inline constexpr int defaultMaxPaintResolution = 4096;
#endif

inline void applyBalloonPlayerDefaults(nlohmann::json& config) {
    if (!config.is_object()) {
        return;
    }
    // Keep the authored mesh roster available to the Player settings.
    if (config.contains("balloons") && config["balloons"].is_number_integer()) {
        const auto count = config["balloons"].get<double>();
        if (count > defaultBalloonCount && count <= 24) {
            config["balloons"] = defaultBalloonCount;
        }
    }
    if (config.contains("visual_nodes") &&
        (!config.contains("jolt_motion") || config["jolt_motion"].is_boolean())) {
        config["jolt_motion"] = false;
    }
}

} // namespace Paint
