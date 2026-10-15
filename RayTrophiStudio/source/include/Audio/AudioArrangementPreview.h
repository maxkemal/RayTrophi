#pragma once
#include "Audio/AudioPcm.h"
#include <memory>
struct ma_engine;
namespace audio {
class AudioArrangementPreview {
public:
    AudioArrangementPreview();
    ~AudioArrangementPreview();
    nlohmann::json play(AudioPcm pcm, ma_engine& engine, const nlohmann::json& params);
    nlohmann::json start(const nlohmann::json& scene, const nlohmann::json& params);
    void tick(const nlohmann::json& scene, ma_engine& engine);
    nlohmann::json status() const;
    bool playing() const;
    void stop();
    void clear();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace audio
