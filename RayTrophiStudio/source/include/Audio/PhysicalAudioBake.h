#pragma once
#include "Audio/AudioDesignJobs.h"
#include "Audio/AudioService.h"

namespace audio {
std::filesystem::path physicalBakeDestination(const nlohmann::json& params);
class PhysicalAudioBake {
public:
    nlohmann::json invoke(AudioService& service, const std::string& operation,
                          const nlohmann::json& params);
    void cancel();

private:
    AudioDesignJobs jobs_;
};
}
