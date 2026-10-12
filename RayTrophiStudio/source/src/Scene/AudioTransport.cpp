#include "Audio/AudioTransport.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace audio {
std::string AudioTransport::validateMode(const nlohmann::json& value) {
    if (!value.is_string() || (value != "continuous" && value != "timeline")) {
        throw std::invalid_argument("playback mode must be continuous or timeline");
    }
    return value.get<std::string>();
}

void AudioTransport::configure(const std::string& mode) {
    mode_ = validateMode(mode);
    initialized_ = false;
    playing_ = false;
    time_ = 0;
    dt_ = 0;
    seeks_ = 0;
    lastWall_ = std::chrono::steady_clock::now();
}

TransportStep AudioTransport::advance(double seconds, bool playing) {
    if (!std::isfinite(seconds) || seconds < 0) {
        throw std::invalid_argument("Timeline audio time must be finite and nonnegative");
    }
    const auto now = std::chrono::steady_clock::now();
    const double delta = seconds - time_;
    TransportStep step;
    step.resumed = playing && !playing_;
    step.seek = initialized_ && (delta < 0 || delta > 2 ||
                                (!playing && !playing_ && delta != 0));
    if (step.seek) {
        ++seeks_;
    }
    const double elapsed = timeline() ? (playing && !step.seek && initialized_ ? delta : 0) :
        std::chrono::duration<double>(now - lastWall_).count();
    dt_ = static_cast<float>(std::clamp(elapsed, 0.0, 0.25));
    time_ = seconds;
    playing_ = playing;
    initialized_ = true;
    lastWall_ = now;
    return step;
}

nlohmann::json AudioTransport::status() const {
    return {{"mode", mode_}, {"playing", playing_}, {"paused", paused()},
            {"time_seconds", time_}, {"dt_seconds", dt_}, {"seek_count", seeks_}};
}
}
