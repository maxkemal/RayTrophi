#include "Audio/AudioThunder.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace audio {
namespace {
using json = nlohmann::json;
void check(bool valid, const std::string& message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}
float numeric(const json& p, const char* key, float fallback, float low, float high) {
    const float value = p.value(key, fallback);
    check(std::isfinite(value) && value >= low && value <= high,
          std::string(key) + " must be finite and in range");
    return value;
}
void position(const json& p) {
    check(p.is_array() && p.size() == 3, "Thunder position requires three numbers");
    for (const auto& v : p) {
        check(v.is_number() && std::isfinite(v.get<float>()), "Thunder position must be finite");
    }
}
std::uint32_t hash(const std::string& key) {
    std::uint32_t value = 2166136261u;
    for (const unsigned char c : key) {
        value = (value ^ c) * 16777619u;
    }
    return value;
}
}

nlohmann::json AudioThunder::validate(const json& input) {
    check(input.is_object(), "Thunder config must be an object");
    const json defaults = {{"enabled", false}, {"gain", 1.0f}, {"sound_speed_mps", 343.0f},
        {"reference_distance_m", 500.0f}, {"max_distance_m", 30000.0f},
        {"crossover_m", 3000.0f}, {"near_assets", json::array()}, {"far_assets", json::array()}};
    for (auto it = input.begin(); it != input.end(); ++it) {
        check(defaults.contains(it.key()), "Unknown thunder property: " + it.key());
    }
    json result = defaults;
    result.update(input);
    check(result.at("enabled").is_boolean(), "Thunder enabled must be boolean");
    result["gain"] = numeric(result, "gain", 1, 0, 4);
    result["sound_speed_mps"] = numeric(result, "sound_speed_mps", 343, 250, 450);
    result["reference_distance_m"] = numeric(result, "reference_distance_m", 500, 1, 100000);
    result["max_distance_m"] = numeric(result, "max_distance_m", 30000, 1, 100000);
    result["crossover_m"] = numeric(result, "crossover_m", 3000, 1, 100000);
    check(result.at("max_distance_m").get<float>() >=
          result.at("reference_distance_m").get<float>(), "Thunder distance range is reversed");
    for (const char* key : {"near_assets", "far_assets"}) {
        const auto& assets = result.at(key);
        check(assets.is_array() && assets.size() <= 16, "Thunder needs at most 16 assets per range");
        for (const auto& asset : assets) {
            check(asset.is_string() && !asset.get<std::string>().empty(),
                  "Thunder asset ids must be nonempty strings");
        }
        check(!result.at("enabled").get<bool>() || !assets.empty(),
              "Enabled thunder requires both near_assets and far_assets");
    }
    return result;
}

void AudioThunder::configure(const json& config) {
    config_ = validate(config);
    reset();
}

void AudioThunder::reset(bool suspend) {
    pending_.clear();
    seen_.clear();
    lastAsset_.clear();
    suspended_ = suspend;
    dropped_ = 0;
}

json AudioThunder::publish(const json& input, const json& listener) {
    const std::string key = input.at("key");
    check(!key.empty() && key.size() <= 256, "Thunder key must have 1..256 bytes");
    position(input.at("position"));
    position(listener);
    const float intensity = numeric(input, "intensity", 30, 0, 100000);
    const float age = numeric(input, "age_seconds", 0, 0, 2);
    if (!config_.at("enabled").get<bool>() || suspended_) {
        return {{"queued", false}, {"reason", "disabled_or_suspended"}};
    }
    if (std::find(seen_.begin(), seen_.end(), key) != seen_.end()) {
        return {{"queued", false}, {"reason", "duplicate"}};
    }
    double squared = 0;
    for (int i = 0; i < 3; ++i) {
        const double delta = input.at("position")[i].get<double>() - listener[i].get<double>();
        squared += delta * delta;
    }
    const double distance = std::sqrt(squared);
    seen_.push_back(key);
    if (seen_.size() > 128) {
        seen_.pop_front();
    }
    if (distance > config_.at("max_distance_m").get<float>() || intensity == 0) {
        return {{"queued", false}, {"reason", "inaudible"}};
    }
    if (pending_.size() >= 64) {
        ++dropped_;
        return {{"queued", false}, {"reason", "queue_full"}};
    }
    const auto& assets = config_.at(distance < config_.at("crossover_m").get<float>() ?
                                    "near_assets" : "far_assets");
    const auto seed = hash(key);
    std::size_t index = seed % assets.size();
    if (assets.size() > 1 && assets[index] == lastAsset_) {
        index = (index + 1) % assets.size();
    }
    lastAsset_ = assets[index];
    const double delay = distance / config_.at("sound_speed_mps").get<float>();
    json event = input;
    event["asset_id"] = lastAsset_;
    event["distance_m"] = distance;
    event["delay_seconds"] = delay;
    event["gain"] = config_.at("gain").get<float>() * std::min(4.0f, std::sqrt(intensity / 30));
    event["pitch"] = 0.94f + float((seed >> 16) % 101) * 0.0012f;
    event["min_distance"] = config_.at("reference_distance_m");
    event["max_distance"] = config_.at("max_distance_m");
    pending_.push_back({event, now() + std::max(0.0, delay - age)});
    return {{"queued", true}, {"event", event}};
}

void AudioThunder::resume() {
    suspended_ = false;
}

void AudioThunder::clock(bool timeline, double seconds) {
    timeline_ = timeline;
    sceneTime_ = seconds;
}

double AudioThunder::now() const {
    return timeline_ ? sceneTime_ :
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

json AudioThunder::due() {
    json result = json::array();
    const double current = now();
    for (auto it = pending_.begin(); it != pending_.end();) {
        if (it->arrival <= current && result.size() < 2) {
            result.push_back(it->event);
            it = pending_.erase(it);
        } else {
            ++it;
        }
    }
    return result;
}

json AudioThunder::status() const {
    json pending = json::array();
    const double current = now();
    for (const auto& p : pending_) {
        auto event = p.event;
        event["remaining_seconds"] = std::max(0.0, p.arrival - current);
        pending.push_back(event);
    }
    return {{"pending", pending}, {"suspended", suspended_}, {"seen_count", seen_.size()},
            {"dropped_queue_full", dropped_}};
}
}
