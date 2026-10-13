#pragma once
#include "json.hpp"
#include <memory>

namespace audio {
// Records scene-clock control envelopes/events, never speaker/callback PCM.
class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();
    nlohmann::json start(double time, const nlohmann::json& params);
    void sample(double time, const nlohmann::json& voices);
    nlohmann::json finish(double time, nlohmann::json& scene);
    void cancel();
    nlohmann::json status() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace audio
