#pragma once
#include "json.hpp"
#include <chrono>
#include <cstdint>
#include <string>

namespace audio {
struct TransportStep {
    bool seek = false;
    bool resumed = false;
};
// Scene-time transport policy; device voices are managed by AudioService.
class AudioTransport {
public:
    static std::string validateMode(const nlohmann::json& value);
    void configure(const std::string& mode);
    TransportStep advance(double seconds, bool playing);
    bool timeline() const {
        return mode_ == "timeline";
    }
    bool paused() const {
        return timeline() && !playing_;
    }
    double time() const {
        return time_;
    }
    float dt() const {
        return dt_;
    }
    nlohmann::json status() const;
private:
    std::string mode_ = "continuous";
    bool initialized_ = false;
    bool playing_ = false;
    double time_ = 0;
    float dt_ = 0;
    std::uint64_t seeks_ = 0;
    std::chrono::steady_clock::time_point lastWall_ = std::chrono::steady_clock::now();
};
}
