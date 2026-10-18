#pragma once
#include "json.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace audio {
class AudioService;
// One bounded sampled-control take per service. Never accesses simulation state.
class PhysicalAudioHistory {
public:
    static std::vector<std::string> parameters(const std::string& operation);
    nlohmann::json invoke(AudioService& service, const std::string& operation,
                          const nlohmann::json& params, bool timeline, double seconds);
    void sample(AudioService& service, bool timeline, double seconds);
    void event(AudioService& service, const std::string& id, const nlohmann::json& event,
               bool timeline, double seconds);
    void cancel();

private:
    std::string state_ = "idle";
    std::string error_;
    nlohmann::json source_ = nlohmann::json::object();
    nlohmann::json controls_ = nlohmann::json::array();
    nlohmann::json events_ = nlohmann::json::array();
    nlohmann::json pending_ = nlohmann::json::array();
    std::uint64_t generation_ = 0;
    double start_ = 0;
    double elapsed_ = 0;
    double maximum_ = 60;
    bool timeline_ = true;
    bool wasResident_ = false;
    double clock(bool timeline, double seconds) const;
    nlohmann::json status() const;
    void appendEvent(const nlohmann::json& event, std::uint64_t frame);
};
}
