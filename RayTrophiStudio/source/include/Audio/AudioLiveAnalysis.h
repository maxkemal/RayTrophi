#pragma once

#include "json.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>

namespace audio {

// One producer (audio callback), one consumer (main-thread queries). Slot ownership
// protects plain PCM without locks, allocation, FFT work or unbounded retries in audio.
class AudioLiveAnalysis {
public:
    static constexpr unsigned frames = 2048;
    void reset(unsigned sampleRate = 48000); // Only with the callback detached.
    void push(const float* stereo, unsigned count);
    nlohmann::json read(bool details = false);

private:
    struct Slot {
        std::atomic<unsigned> state{0}; // free=0, writing=1, ready=2, reading=3
        std::array<float, frames * 2> pcm{};
        std::uint64_t endFrame = 0;
    };
    std::array<Slot, 3> slots_;
    std::array<float, frames * 2> pending_{};
    unsigned pendingFrames_ = 0;
    unsigned sampleRate_ = 48000;
    std::uint64_t writtenFrames_ = 0;
    std::uint64_t analyzedFrame_ = 0;
    nlohmann::json cached_ = nlohmann::json::object();
    std::chrono::steady_clock::time_point received_{};
    std::deque<nlohmann::json> history_; // Main-thread sampled snapshots, never callback work.
    void publish();
    nlohmann::json analyze(const std::array<float, frames * 2>& pcm) const;
};

} // namespace audio
