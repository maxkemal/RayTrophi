#pragma once

#include "Backend/UpscaleTypes.h"
#include "json.hpp"

#include <cmath>
#include <stdexcept>

namespace VulkanRT {

// Persist authored preferences only; GPU resources, eligibility and history are runtime state.
inline nlohmann::json serializeUpscaleSettings(const UpscaleSettings& settings) {
    return {{"mode", static_cast<int>(settings.mode)},
            {"quality", static_cast<int>(settings.quality)},
            {"sharpness", settings.sharpness},
            {"rejection_reactive", settings.rejection_reactive}};
}

inline UpscaleSettings deserializeUpscaleSettings(const nlohmann::json& renderSettings) {
    UpscaleSettings settings;
    const auto entry = renderSettings.find("upscale");
    if (entry == renderSettings.end()) {
        return settings;
    }
    if (!entry->is_object()) {
        throw std::invalid_argument("render_settings.upscale must be an object");
    }
    const auto readEnum = [&](const char* key, int fallback, int minimum, int maximum) {
        const auto field = entry->find(key);
        if (field == entry->end()) {
            return fallback;
        }
        if (!field->is_number_integer() || *field < minimum || *field > maximum) {
            throw std::invalid_argument(std::string("invalid render_settings.upscale.") + key);
        }
        return field->get<int>();
    };
    settings.mode = static_cast<UpscaleMode>(readEnum(
        "mode", static_cast<int>(settings.mode), static_cast<int>(UpscaleMode::Off),
        static_cast<int>(UpscaleMode::Auto)));
    settings.quality = static_cast<UpscaleQuality>(readEnum(
        "quality", static_cast<int>(settings.quality), static_cast<int>(UpscaleQuality::Quality),
        static_cast<int>(UpscaleQuality::UltraPerformance)));
    const auto sharpness = entry->find("sharpness");
    if (sharpness != entry->end()) {
        if (!sharpness->is_number()) {
            throw std::invalid_argument("render_settings.upscale.sharpness must be numeric");
        }
        settings.sharpness = sharpness->get<float>();
    }
    if (!std::isfinite(settings.sharpness) || settings.sharpness < 0.0f ||
        settings.sharpness > 1.0f) {
        throw std::invalid_argument("render_settings.upscale.sharpness must be finite in [0, 1]");
    }
    settings.rejection_reactive = entry->value("rejection_reactive", settings.rejection_reactive);
    return settings;
}

} // namespace VulkanRT
