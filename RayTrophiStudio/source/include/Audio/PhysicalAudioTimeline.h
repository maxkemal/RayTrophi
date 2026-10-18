#pragma once
#include "Audio/AudioPcm.h"
#include "Audio/AudioDesignJobs.h"
#include <cstdint>

namespace audio {
void validateStoredPhysicalTimeline(nlohmann::json& source);
nlohmann::json physicalTimelineInput(const nlohmann::json& source,
                                      const nlohmann::json& params);
nlohmann::json validatePhysicalTimeline(const nlohmann::json& source,
                                         const nlohmann::json& params);
AudioPcm renderPhysicalTimeline(const nlohmann::json& timeline,
                               const AudioCancelCheck& check = {});
AudioPcm renderPhysicalMix(const nlohmann::json& timelines, std::uint64_t start,
                          std::uint64_t end, const AudioCancelCheck& check = {});
}
