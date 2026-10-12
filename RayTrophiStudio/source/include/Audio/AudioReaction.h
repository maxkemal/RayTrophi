#pragma once
#include "json.hpp"
#include <chrono>
#include <map>
#include <string>

namespace audio {
struct ReactionOutput {
    float activity = 1;
    float gain = 1;
    float pitch = 1;
    nlohmann::json position;
};

// Main-thread continuous control. This never decodes audio or accesses scene geometry.
class AudioReaction {
public:
    static nlohmann::json validate(const nlohmann::json& config);
    nlohmann::json publish(const nlohmann::json& signal);
    ReactionOutput evaluate(const std::string& id, const nlohmann::json& config,
                            float windMps, float rain, float dt);
    nlohmann::json status() const;
    void clear();
    void forget(const std::string& id);
    void invalidateSignal(const std::string& key);
    void clock(bool timeline, double seconds);
private:
    struct Signal {
        nlohmann::json data;
        double expires = 0;
    };
    std::map<std::string, Signal> signals_;
    std::map<std::string, float> envelopes_;
    std::map<std::string, nlohmann::json> lastPositions_;
    std::map<std::string, nlohmann::json> outputs_;
    bool timeline_ = false;
    double sceneTime_ = 0;
    double now() const;
};
}
