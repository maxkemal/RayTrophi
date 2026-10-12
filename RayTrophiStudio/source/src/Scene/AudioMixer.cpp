#include "Audio/AudioMixer.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace audio {
namespace {
using json = nlohmann::json;
void require(bool valid, const std::string& message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}
bool anySolo(const json& config) {
    for (const auto& name : AudioMixer::names()) {
        if (name != "Master" && config.at(name).at("solo").get<bool>()) {
            return true;
        }
    }
    return false;
}
}

const std::array<std::string, 5>& AudioMixer::names() {
    static const std::array<std::string, 5> names = {
        "Master", "Weather", "Effects", "Ambience", "Music"};
    return names;
}

nlohmann::json AudioMixer::validate(const json& input) {
    require(input.is_object(), "Mixer settings must be an object");
    for (auto it = input.begin(); it != input.end(); ++it) {
        require(std::find(names().begin(), names().end(), it.key()) != names().end(),
                "Unknown mixer bus: " + it.key());
    }
    json result = json::object();
    for (const auto& name : names()) {
        const auto settings = input.value(name, json::object());
        require(settings.is_object(), "Mixer bus settings must be an object");
        for (auto it = settings.begin(); it != settings.end(); ++it) {
            require(it.key() == "gain" || it.key() == "muted" || it.key() == "solo",
                    "Unknown mixer property: " + it.key());
        }
        const float gain = settings.value("gain", 1.0f);
        require(std::isfinite(gain) && gain >= 0 && gain <= 4,
                "Mixer gain must be finite and in 0..4");
        const auto muted = settings.value("muted", json(false));
        const auto solo = settings.value("solo", json(false));
        require(muted.is_boolean() && solo.is_boolean(), "Mixer mute/solo must be boolean");
        require(name != "Master" || !solo.get<bool>(), "Master has no solo; solo child buses");
        result[name] = {{"gain", gain}, {"muted", muted}, {"solo", solo}};
    }
    return result;
}

nlohmann::json AudioMixer::update(const json& current, const json& patch) {
    require(patch.is_object() && patch.contains("bus"), "Mixer update requires bus");
    const std::string bus = patch.at("bus");
    json candidate = validate(current);
    require(candidate.contains(bus), "Unknown mixer bus: " + bus);
    for (auto it = patch.begin(); it != patch.end(); ++it) {
        if (it.key() != "bus") {
            candidate[bus][it.key()] = it.value();
        }
    }
    return validate(candidate);
}

float AudioMixer::effectiveGain(const json& config, const std::string& bus) {
    const auto& settings = config.at(bus);
    if (settings.at("muted").get<bool>() ||
        (bus != "Master" && anySolo(config) && !settings.at("solo").get<bool>())) {
        return 0;
    }
    return settings.at("gain").get<float>();
}

float AudioMixer::directMasterGain(const json& config) {
    return anySolo(config) ? 0.0f : 1.0f;
}
}
