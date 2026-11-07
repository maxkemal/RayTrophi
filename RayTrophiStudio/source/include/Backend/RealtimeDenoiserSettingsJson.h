#pragma once

#include "Backend/RealtimeDenoiserTypes.h"
#include "json.hpp"
#include <cmath>
#include <stdexcept>

namespace VulkanRT {

inline nlohmann::json serializeRealtimeDenoiserSettings(
    const RealtimeDenoiserSettings& settings) {
    return {{"enabled", settings.enabled},
            {"atrous_iterations", settings.atrous_iterations},
            {"history_max", settings.history_max},
            {"firefly_clip", settings.firefly_clip},
            {"still_noise", settings.still_noise},
            {"scene_motion", settings.scene_motion}};
}

inline RealtimeDenoiserSettings deserializeRealtimeDenoiserSettings(
    const nlohmann::json& renderSettings) {
    RealtimeDenoiserSettings settings;
    const auto entry = renderSettings.find("realtime_denoiser");
    if (entry == renderSettings.end()) {
        return settings; // Old projects and New Project start disabled.
    }
    if (!entry->is_object()) {
        throw std::invalid_argument("render_settings.realtime_denoiser must be an object");
    }
    const auto readBool = [&](const char* key, bool fallback) {
        const auto field = entry->find(key);
        if (field == entry->end()) {
            return fallback;
        }
        if (!field->is_boolean()) {
            throw std::invalid_argument(std::string("invalid realtime_denoiser.") + key);
        }
        return field->get<bool>();
    };
    const auto readInt = [&](const char* key, int fallback, int minimum, int maximum) {
        const auto field = entry->find(key);
        if (field == entry->end()) {
            return fallback;
        }
        if (!field->is_number_integer() || *field < minimum || *field > maximum) {
            throw std::invalid_argument(std::string("invalid realtime_denoiser.") + key);
        }
        return field->get<int>();
    };
    const auto readFloat = [&](const char* key, float fallback, float minimum, float maximum) {
        const auto field = entry->find(key);
        if (field == entry->end()) {
            return fallback;
        }
        if (!field->is_number()) {
            throw std::invalid_argument(std::string("invalid realtime_denoiser.") + key);
        }
        const float value = field->get<float>();
        if (!std::isfinite(value) || value < minimum || value > maximum) {
            throw std::invalid_argument(std::string("invalid realtime_denoiser.") + key);
        }
        return value;
    };
    settings.enabled = readBool("enabled", settings.enabled);
    settings.atrous_iterations = readInt("atrous_iterations", settings.atrous_iterations, 1, 5);
    settings.history_max = readInt("history_max", settings.history_max, 1, 64);
    settings.firefly_clip = readFloat("firefly_clip", settings.firefly_clip, 0.1f, 10.0f);
    settings.still_noise = readFloat("still_noise", settings.still_noise, 0.0f, 0.5f);
    settings.scene_motion = readBool("scene_motion", settings.scene_motion);
    // debug_history, GPU resources, temporal counters and camera state are not saved.
    return settings;
}

} // namespace VulkanRT
