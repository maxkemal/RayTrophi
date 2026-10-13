#pragma once
#include "json.hpp"
#include <memory>
#include <cstddef>

struct ma_engine;
namespace audio {
class AudioMixerRuntime;
class AudioClipRuntime {
public:
    AudioClipRuntime();
    ~AudioClipRuntime();
    void tick(const nlohmann::json& scene, ma_engine& engine, AudioMixerRuntime& mixer,
              double time, bool playing, std::size_t voiceSlots, std::size_t byteBudget);
    void clear();
    std::size_t voices() const;
    std::size_t bytes() const;
    nlohmann::json status() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace audio
