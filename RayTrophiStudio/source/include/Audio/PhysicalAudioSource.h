#pragma once

#include "../../../external/miniaudio/miniaudio.h"
#include "json.hpp"
#include "Audio/AudioDesignJobs.h"
#include <map>
#include <cstdint>
#include <memory>
#include <string>

namespace audio {
class AudioService;
class PhysicalAudioSchedule;
bool isPhysicalAudioSource(const nlohmann::json& source);
void validatePhysicalAudioSource(nlohmann::json& source);

// Main-thread ownership; sources outlive their attached ma_sound instances.
// DSP and lock-free control transfer are implemented outside AudioService.
class PhysicalAudioRuntime {
public:
    PhysicalAudioRuntime();
    ~PhysicalAudioRuntime();
    std::unique_ptr<ma_sound> prepare(ma_engine& engine, ma_sound_group* bus,
                                    const nlohmann::json& source);
    void activity(const std::string& id, float value);
    float outputGain(const std::string& id, float value);
    bool retire(const std::string& id, bool inactive, float dt);
    nlohmann::json status(const std::string& id) const;
    static nlohmann::json validateImpact(const nlohmann::json& source,
                                         const nlohmann::json& params);
    nlohmann::json impact(const std::string& id, const nlohmann::json& event);
    bool duplicate(const std::string& id, std::uint64_t sequence) const;
    bool isReplay(const std::string& id) const;
    void validateReplaySource(const nlohmann::json& source) const;
    static std::vector<std::string> replayParameters(const std::string& operation);
    nlohmann::json replay(AudioService& service, const std::string& operation,
                          const nlohmann::json& params);
    void release(const std::string& id);
    void forget(const std::string& id);
    void clear();

private:
    struct Source;
    std::map<std::string, std::unique_ptr<Source>> sources_;
    std::map<std::string, std::uint64_t> sequences_;
    std::map<std::string, std::shared_ptr<PhysicalAudioSchedule>> preparedReplays_;
    AudioDesignJobs replayJobs_;
};
}
