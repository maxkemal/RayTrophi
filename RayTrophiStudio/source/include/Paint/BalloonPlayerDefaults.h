#pragma once

#include "json.hpp"

namespace Paint {

inline constexpr int defaultBalloonCount = 5;

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
