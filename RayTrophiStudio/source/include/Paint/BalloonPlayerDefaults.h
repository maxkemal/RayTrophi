#pragma once

#include <cstdint>

#include "Paint/WetPaintMobileBudget.h"
#include "json.hpp"

namespace Paint {

inline constexpr int defaultBalloonCount = 5;
#ifdef __ANDROID__
// Android Player caps authored wet-paint surfaces to 512 on project load
// (matches Android 512 texture staging; 4x less GPU memory and Mali paint cost).
// F2 / game settings resize can still override per session.
inline constexpr int defaultMaxPaintResolution = 512;
inline constexpr uint32_t defaultMaxEventsPerTick = 512u;
inline constexpr int defaultMaxParticlesPerBalloon = 700;
#else
inline constexpr int defaultMaxPaintResolution = 4096;
inline constexpr uint32_t defaultMaxEventsPerTick = 4096u;
inline constexpr int defaultMaxParticlesPerBalloon = 8192;
#endif

// Runtime-effective caps: integrated GPUs (e.g. AMD A10/R7, Intel UHD) get the
// same 512px / 700-droplet budget as Android so shared-RAM APUs stay near 60 FPS
// during multi-balloon pops, while discrete GPUs keep full 1024px / 1800 drops.
inline int effectiveMaxPaintResolution() {
    return WetPaintMobileBudget::isLowPowerGpu() ? 512 : defaultMaxPaintResolution;
}

inline int effectiveMaxParticlesPerBalloon() {
    return WetPaintMobileBudget::isLowPowerGpu() ? 700 : defaultMaxParticlesPerBalloon;
}

// Integrated GPUs and mobile GPUs drop steeply when many wet-paint tiles stay
// active for 12s across rapid pops; cap authored drying time to 3.5s on load.
inline float effectiveMaxPaintLifetimeSeconds() {
    return WetPaintMobileBudget::isLowPowerGpu() ? 3.5f : 600.0f;
}

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
    if (config.contains("particles_per_balloon") &&
        config["particles_per_balloon"].is_number_integer()) {
        const auto count = config["particles_per_balloon"].get<int>();
        const int cap = effectiveMaxParticlesPerBalloon();
        if (count > cap) {
            config["particles_per_balloon"] = cap;
        }
    }
    if (config.contains("visual_nodes") &&
        (!config.contains("jolt_motion") || config["jolt_motion"].is_boolean())) {
        config["jolt_motion"] = false;
    }
}

} // namespace Paint
