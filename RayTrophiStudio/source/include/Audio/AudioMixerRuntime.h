#pragma once
#include "Audio/AudioMixer.h"
#include "Audio/AudioLiveAnalysis.h"
#include "../../../external/miniaudio/miniaudio.h"
#include <atomic>
#include <cstdint>

namespace audio {
// Stable nodes: callback touches only PCM and lock-free meter scalars.
class AudioMixerRuntime {
public:
    ~AudioMixerRuntime();
    void initialize(ma_engine& engine);
    void shutdown();
    void apply(const nlohmann::json& config);
    ma_sound_group* group(const std::string& bus);
    void route(ma_sound& sound, const std::string& bus);
    nlohmann::json meters() const;
    nlohmann::json analysis(const std::string& bus, bool details = false);
    void resetPeaks();
private:
    struct Meter {
        ma_node_base node{};
        std::atomic<std::uint32_t> peak{0};
        std::atomic<std::uint32_t> rms{0};
        std::atomic<std::uint32_t> hold{0};
        std::atomic<std::uint32_t> clips{0};
        AudioLiveAnalysis* analysis = nullptr;
    };
    struct Bus {
        ma_sound_group group{};
        Meter meter;
        bool groupReady = false;
        bool meterReady = false;
    };
    std::array<Bus, 5> buses_;
    std::array<AudioLiveAnalysis, 5> analysis_;
    bool ready_ = false;
    static void process(ma_node* node, const float** input, ma_uint32* countIn,
                        float** output, ma_uint32* countOut);
};
}
