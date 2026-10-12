#pragma once
#include "json.hpp"
#include <chrono>
#include <cstdint>
#include <deque>
#include <string>

namespace audio {
// Main-thread event scheduling. The queue outlives the visual flash; no renderer/device access.
class AudioThunder {
public:
    static nlohmann::json validate(const nlohmann::json& config);
    void configure(const nlohmann::json& config);
    void reset(bool suspend = false);
    void resume();
    void clock(bool timeline, double seconds);
    nlohmann::json publish(const nlohmann::json& event, const nlohmann::json& listener);
    nlohmann::json due();
    nlohmann::json status() const;
private:
    struct Pending {
        nlohmann::json event;
        double arrival = 0;
    };
    nlohmann::json config_ = validate(nlohmann::json::object());
    std::deque<Pending> pending_;
    std::deque<std::string> seen_;
    std::string lastAsset_;
    bool suspended_ = false;
    std::uint64_t dropped_ = 0;
    bool timeline_ = false;
    double sceneTime_ = 0;
    double now() const;
};
}
