#pragma once

#include "../../../external/miniaudio/miniaudio.h"
#include "json.hpp"
#include "Audio/AcousticGpu.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace audio {
// Stable per-voice node. Only lock-free coefficients cross into the PCM callback.
class AcousticPlayback {
public:
    AcousticPlayback() = default;
    ~AcousticPlayback();
    void initialize(ma_engine& engine, ma_sound& sound, ma_sound_group* bus);
    void route(ma_sound_group* bus);
    void response(const nlohmann::json& energy);
    void reflections(const nlohmann::json& paths, const nlohmann::json& source,
                     const nlohmann::json& listener, const nlohmann::json& position);

private:
    struct Node {
        ma_node_base base{};
        AcousticPlayback* owner = nullptr;
    };
    Node node_{};
    bool ready_ = false;
    std::array<std::atomic<std::uint32_t>, 6> target_{};
    std::array<float, 6> gain_{1, 1, 1, 1, 1, 1};
    float low_[2][5]{};
    struct Tap {
        std::array<std::atomic<std::uint32_t>, 6> target{};
        std::atomic<std::uint32_t> delay{0};
        std::array<float, 6> gain{};
        float frames = 0;
        float low[2][5]{};
    };
    std::array<Tap, kAcousticPlaybackTaps> taps_{};
    std::vector<float> history_;
    std::size_t cursor_ = 0;
    static void process(ma_node* node, const float** input, ma_uint32* countIn,
                        float** output, ma_uint32* countOut);
};
}
