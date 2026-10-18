#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/AudioEmitterSchema.h"
#include "Audio/AudioAcousticSettings.h"
#include "Audio/AudioRequestValidation.h"
#include "Audio/AudioMixer.h"
#include "Audio/AudioReaction.h"
#include "Audio/PhysicalAudioSource.h"
#include "Audio/PhysicalAudioTimeline.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace audio {
using json = nlohmann::json;
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

float number(const json& p, const char* key, float fallback, float low, float high) {
    const float value = p.value(key, fallback);
    require(std::isfinite(value) && value >= low && value <= high,
            std::string(key) + " must be finite and in range");
    return value;
}

}

json validateAudioVector(const json& p, const char* key, const json& fallback) {
    const json value = p.value(key, fallback);
    require(value.is_array() && value.size() == 3, std::string(key) + " needs three numbers");
    for (const auto& component : value) {
        require(component.is_number() && std::isfinite(component.get<float>()),
                std::string(key) + " must be finite");
    }
    return value;
}

std::vector<std::string> audioEmitterParameters(bool includeId) {
    std::vector<std::string> keys = {
        "id", "name", "asset_id", "position", "direction", "gain", "pitch", "loop",
        "spatial", "min_distance", "max_distance", "rolloff", "cone_inner", "cone_outer",
        "cone_gain", "bus", "parent", "reaction", "environment_role", "generator",
        "generator_history", "acoustics"};
    if (!includeId) {
        keys.erase(keys.begin());
    }
    return keys;
}

json normalizeAudioEmitter(const json& p) {
    validateAudioRequest("create", p);
    const auto keys = audioEmitterParameters(true);
    for (auto it = p.begin(); it != p.end(); ++it) {
        require(std::find(keys.begin(), keys.end(), it.key()) != keys.end(),
                "Unknown emitter property: " + it.key());
    }
    json e = p;
    e["acoustics"] = validateSourceAcoustics(p.value("acoustics", json::object()));
    e["name"] = p.value("name", std::string("Audio Source"));
    e["asset_id"] = p.value("asset_id", std::string());
    e["parent"] = p.value("parent", std::string());
    e["reaction"] = AudioReaction::validate(p.value("reaction", json::object()));
    e["environment_role"] = p.value("environment_role", std::string());
    require(e["environment_role"] == "" || e["environment_role"] == "wind" ||
            e["environment_role"] == "rain", "Unknown environment role");
    e["position"] = validateAudioVector(p, "position", json::array({0, 0, 0}));
    e["direction"] = validateAudioVector(p, "direction", json::array({0, 0, -1}));
    float length = 0;
    for (const auto& component : e["direction"]) {
        length += component.get<float>() * component.get<float>();
    }
    require(length > 1e-12f && std::isfinite(length), "direction must be nonzero");
    e["gain"] = number(p, "gain", 1, 0, 16);
    e["pitch"] = number(p, "pitch", 1, 0.125f, 8);
    e["min_distance"] = number(p, "min_distance", 1, 0.001f, 1000000);
    e["max_distance"] = number(p, "max_distance", 100, 0.001f, 1000000);
    require(e["max_distance"].get<float>() >= e["min_distance"].get<float>(),
            "max_distance must be >= min_distance");
    e["rolloff"] = number(p, "rolloff", 1, 0, 10);
    e["cone_inner"] = number(p, "cone_inner", 360, 0, 360);
    e["cone_outer"] = number(p, "cone_outer", 360, 0, 360);
    require(e["cone_outer"].get<float>() >= e["cone_inner"].get<float>(),
            "cone_outer must be >= cone_inner");
    e["cone_gain"] = number(p, "cone_gain", 1, 0, 1);
    e["loop"] = p.value("loop", false);
    require(!e.at("reaction").at("automatic").get<bool>() || e.at("loop").get<bool>(),
            "Automatic continuous audio requires loop=true");
    e["spatial"] = p.value("spatial", true);
    e["bus"] = p.value("bus", std::string("Effects"));
    const auto& buses = AudioMixer::names();
    require(std::find(buses.begin(), buses.end(), e["bus"].get<std::string>()) != buses.end(),
            "Unknown audio bus");
    validatePhysicalAudioSource(e);
    validateStoredPhysicalTimeline(e);
    return e;
}
}
