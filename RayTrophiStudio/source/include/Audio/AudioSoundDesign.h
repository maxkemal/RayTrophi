#pragma once

#include "Audio/AudioPcm.h"
#include <memory>

struct ma_engine;
struct ma_sound;

namespace audio {
struct AudioDesignResult;
// One bounded, transient design per scene service. PCM never references scene/UI objects.
class AudioSoundDesign {
public:
    AudioSoundDesign();
    ~AudioSoundDesign();
    nlohmann::json render(const nlohmann::json& recipe, const nlohmann::json& assets);
    nlohmann::json status(bool details = true) const;
    nlohmann::json analyze(const nlohmann::json& params) const;
    nlohmann::json waveform(const nlohmann::json& params, const nlohmann::json& assets);
    nlohmann::json undo(const nlohmann::json& assets, bool redo, bool background = false);
    nlohmann::json preview(ma_engine& engine, ma_sound* group,
                           const nlohmann::json& params);
    nlohmann::json save(const nlohmann::json& params) const;
    nlohmann::json jobStart(const nlohmann::json& params, const nlohmann::json& assets);
    nlohmann::json jobStatus() const;
    nlohmann::json jobCancel();
    nlohmann::json jobApply(const nlohmann::json& assets);
    void stop();
    void clear();
    bool playing() const;
private:
    nlohmann::json commit(AudioDesignResult prepared);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

nlohmann::json soundDesignDestination(const nlohmann::json& params);
} // namespace audio
