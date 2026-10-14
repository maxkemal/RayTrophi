#pragma once

#include "Audio/AudioPcm.h"
#include <memory>

struct ma_engine;
struct ma_sound;

namespace audio {
// One bounded, transient design per scene service. PCM never references scene/UI objects.
class AudioSoundDesign {
public:
    AudioSoundDesign();
    ~AudioSoundDesign();
    nlohmann::json render(const nlohmann::json& recipe, const nlohmann::json& assets);
    nlohmann::json status() const;
    nlohmann::json preview(ma_engine& engine, ma_sound* group,
                           const nlohmann::json& params);
    nlohmann::json save(const nlohmann::json& params) const;
    void stop();
    void clear();
    bool playing() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

nlohmann::json soundDesignDestination(const nlohmann::json& params);
} // namespace audio
