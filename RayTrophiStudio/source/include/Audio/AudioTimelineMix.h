#pragma once
#include "Audio/AudioPcm.h"
#include "Audio/AudioTracks.h"
#include "Audio/AudioClips.h"
#include <functional>
namespace audio {
struct TimelineClipMix {
    AudioTrackMix track;
    float left = 1, right = 1;
};
TimelineClipMix timelineClipMix(const nlohmann::json& clip, const AudioTrackMix& track,
                                const nlohmann::json& mixer);
void accumulateTimelineClip(float& left, float& right, const AudioPcm& pcm,
                            const nlohmann::json& clip, AudioClipSample control,
                            const TimelineClipMix& mix);
void validateTimelinePreview(const nlohmann::json& scene, const nlohmann::json& params);
AudioPcm renderTimelinePreview(const nlohmann::json& scene, const nlohmann::json& params,
                             const std::function<void()>& check = {});
} // namespace audio
