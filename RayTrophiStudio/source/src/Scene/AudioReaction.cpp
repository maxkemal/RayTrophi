#include "Audio/AudioReaction.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace audio {
namespace {
using json = nlohmann::json;
void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}
float field(const json& p, const char* key, float fallback, float low, float high) {
    const float value = p.value(key, fallback);
    require(std::isfinite(value) && value >= low && value <= high,
            std::string(key) + " must be finite and in range");
    return value;
}
void keys(const json& data, const std::vector<std::string>& allowed) {
    require(data.is_object(), "Reaction/signal must be an object");
    for (auto it = data.begin(); it != data.end(); ++it) {
        require(std::find(allowed.begin(), allowed.end(), it.key()) != allowed.end(),
                "Unknown reaction/signal field: " + it.key());
    }
}
}

nlohmann::json AudioReaction::validate(const json& p) {
    keys(p, {"driver", "input_min", "input_max", "gain_min", "gain_max", "pitch_min",
             "pitch_max", "response_seconds", "signal_key", "follow_position",
             "source_material", "target_material", "automatic", "start_threshold",
             "stop_threshold", "system_id", "domain_name", "input_metric", "domain_channel"});
    json result;
    const std::string driver = p.value("driver", std::string("manual"));
    require(driver == "manual" || driver == "wind" || driver == "rain" || driver == "signal" ||
            driver == "domain_contact", "Unknown reaction driver");
    result["driver"] = driver;
    const std::string metric = p.value("input_metric", std::string("energy_j"));
    require(metric == "energy_j" || metric == "power_w", "Unknown domain input_metric");
    require(driver == "domain_contact" || metric == "energy_j",
            "power_w input_metric requires domain_contact driver");
    result["input_metric"] = metric;
    const std::string channel = p.value("domain_channel", std::string("mixed_contact"));
    require(channel == "mixed_contact" || channel == "dem_contact" ||
            channel == "fluid_flow" || channel == "gas_flow" || channel == "mpm_deformation",
            "Unknown domain_channel");
    result["domain_channel"] = channel;
    result["input_min"] = field(p, "input_min", 0, 0, 1000000000);
    result["input_max"] = field(p, "input_max", driver == "wind" ? 30 : 1, 0, 1000000000);
    require(result["input_max"].get<float>() > result["input_min"].get<float>(),
            "Reaction input_max must exceed input_min");
    result["gain_min"] = field(p, "gain_min", 0, 0, 1);
    result["gain_max"] = field(p, "gain_max", 1, 0, 1);
    result["pitch_min"] = field(p, "pitch_min", 1, 0.5f, 2);
    result["pitch_max"] = field(p, "pitch_max", 1, 0.5f, 2);
    result["response_seconds"] = field(p, "response_seconds", 0.25f, 0, 10);
    result["signal_key"] = p.value("signal_key", std::string());
    result["source_material"] = p.value("source_material", std::string());
    result["target_material"] = p.value("target_material", std::string());
    require(result["source_material"].get<std::string>().size() <= 128 &&
            result["target_material"].get<std::string>().size() <= 128,
            "Material labels must not exceed 128 characters");
    result["follow_position"] = p.value("follow_position", false);
    result["automatic"] = p.value("automatic", false);
    result["start_threshold"] = field(p, "start_threshold", 0.02f, 0.0001f, 1);
    result["stop_threshold"] = field(p, "stop_threshold", 0.005f, 0, 1);
    require(result["stop_threshold"].get<float>() < result["start_threshold"].get<float>(),
            "stop_threshold must be below start_threshold");
    require(!result["automatic"].get<bool>() || driver != "manual",
            "Automatic playback needs an environment/signal/domain driver");
    if (p.contains("system_id")) {
        require(p.at("system_id").is_number_integer() &&
                p.at("system_id").get<double>() >= 0 && p.at("system_id").get<double>() <= 2147483647,
                "system_id must be a nonnegative 32-bit integer");
    }
    result["system_id"] = p.value("system_id", 0);
    require(result["system_id"].get<int>() >= 0, "system_id must be nonnegative");
    result["domain_name"] = p.value("domain_name", std::string());
    require(result["domain_name"].get<std::string>().size() <= 128, "domain_name is too long");
    if (driver == "domain_contact") {
        require(result["system_id"].get<int>() > 0 && !result["domain_name"].get<std::string>().empty(),
                "Domain contact driver needs a system_id and domain_name");
        require(result["source_material"] == "" && result["target_material"] == "",
                "Aggregate domain contacts cannot classify material pairs");
        result["signal_key"] = "@domain/" + std::to_string(result["system_id"].get<int>()) +
                               "/" + result["domain_name"].get<std::string>();
        if (channel != "mixed_contact") {
            result["signal_key"] = "@domain/" + channel + "/" +
                std::to_string(result["system_id"].get<int>()) + "/" +
                result["domain_name"].get<std::string>();
        }
    }
    require(driver != "signal" || !result["signal_key"].get<std::string>().empty(),
            "Signal reaction requires signal_key");
    require(result["signal_key"].get<std::string>().size() <= 256, "Signal key is too long");
    return result;
}

json AudioReaction::publish(const json& p) {
    keys(p, {"key", "value", "flow_kg_s", "speed_mps", "position", "source_material",
             "target_material", "ttl_seconds", "power_w"});
    const std::string key = p.at("key");
    require(!key.empty() && key.size() <= 256, "Signal key must contain 1..256 characters");
    require(p.contains("value") != (p.contains("flow_kg_s") || p.contains("speed_mps")),
            "Supply either value or both flow_kg_s and speed_mps");
    json data = p;
    float value = 0;
    if (p.contains("value")) {
        value = field(p, "value", 0, 0, 1000000000);
    } else {
        require(p.contains("flow_kg_s") && p.contains("speed_mps"),
                "Physical signal needs both flow_kg_s and speed_mps");
        const float flow = field(p, "flow_kg_s", 0, 0, 1000000);
        const float speed = field(p, "speed_mps", 0, 0, 10000);
        value = 0.5f * flow * speed * speed;
        require(std::isfinite(value) && value <= 1000000000, "Signal power exceeds limit");
    }
    data["value"] = value;
    if (p.contains("power_w")) {
        data["power_w"] = field(p, "power_w", 0, 0, 1000000000);
    }
    data["source_material"] = p.value("source_material", std::string());
    data["target_material"] = p.value("target_material", std::string());
    require(data["source_material"].get<std::string>().size() <= 128 &&
            data["target_material"].get<std::string>().size() <= 128,
            "Material labels must not exceed 128 characters");
    if (p.contains("position")) {
        const auto& position = p.at("position");
        require(position.is_array() && position.size() == 3, "Signal position needs three numbers");
        for (const auto& component : position) {
            require(component.is_number() && std::isfinite(component.get<float>()),
                    "Signal position must be finite");
        }
    }
    const float ttl = field(p, "ttl_seconds", 0.5f, 0.01f, 10);
    const double current = now();
    for (auto it = signals_.begin(); it != signals_.end();) {
        if (current >= it->second.expires) {
            it = signals_.erase(it);
        } else {
            ++it;
        }
    }
    require(signals_.contains(key) || signals_.size() < 256, "Audio signal budget reached (256)");
    signals_[key] = {data, current + ttl};
    return data;
}

ReactionOutput AudioReaction::evaluate(const std::string& id, const json& config,
                                      float windMps, float rain, float dt) {
    ReactionOutput result;
    const auto driver = config.at("driver").get<std::string>();
    if (driver == "manual") {
        forget(id);
        return result;
    }
    float input = driver == "wind" ? windMps : (driver == "rain" ? rain : 0);
    if (driver == "signal" || driver == "domain_contact") {
        const auto signal = signals_.find(config.at("signal_key").get<std::string>());
        if (signal != signals_.end() && now() < signal->second.expires) {
            const auto& data = signal->second.data;
            const auto source = config.at("source_material").get<std::string>();
            const auto target = config.at("target_material").get<std::string>();
            if ((source.empty() || data.at("source_material") == source) &&
                (target.empty() || data.at("target_material") == target)) {
                input = data.at("value");
                if (driver == "domain_contact" && config.value("input_metric", "energy_j") ==
                    "power_w") {
                    input = data.value("power_w", 0.0f);
                }
                if (config.at("follow_position").get<bool>() && data.contains("position")) {
                    result.position = data.at("position");
                    lastPositions_[id] = result.position;
                }
            }
        }
        if (config.at("follow_position").get<bool>() && result.position.is_null()) {
            const auto position = lastPositions_.find(id);
            if (position != lastPositions_.end()) {
                result.position = position->second;
            }
        }
    }
    if (!std::isfinite(input) || input < 0) {
        input = 0;
    }
    const float low = config.at("input_min");
    const float high = config.at("input_max");
    const float target = std::clamp((input - low) / (high - low), 0.0f, 1.0f);
    float& envelope = envelopes_[id];
    const float response = config.at("response_seconds");
    const float blend = response == 0 ? 1 : -std::expm1(-dt / response);
    envelope += (target - envelope) * blend;
    const float perceptual = std::sqrt(envelope);
    result.activity = envelope;
    result.gain = config.at("gain_min").get<float>() + perceptual *
        (config.at("gain_max").get<float>() - config.at("gain_min").get<float>());
    result.pitch = config.at("pitch_min").get<float>() + envelope *
        (config.at("pitch_max").get<float>() - config.at("pitch_min").get<float>());
    outputs_[id] = {{"input", input}, {"envelope", envelope}, {"gain_multiplier", result.gain},
                    {"pitch_multiplier", result.pitch}, {"position", result.position}};
    return result;
}

json AudioReaction::status() const {
    json signals = json::object();
    const double current = now();
    for (const auto& [key, signal] : signals_) {
        if (current < signal.expires) {
            signals[key] = signal.data;
        }
    }
    return {{"signals", signals}, {"outputs", outputs_}};
}
void AudioReaction::clear() {
    signals_.clear();
    envelopes_.clear();
    lastPositions_.clear();
    outputs_.clear();
}

void AudioReaction::clock(bool timeline, double seconds) {
    timeline_ = timeline;
    sceneTime_ = seconds;
}

double AudioReaction::now() const {
    return timeline_ ? sceneTime_ :
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void AudioReaction::forget(const std::string& id) {
    envelopes_.erase(id);
    lastPositions_.erase(id);
    outputs_.erase(id);
}
void AudioReaction::invalidateSignal(const std::string& key) {
    signals_.erase(key);
}
}
