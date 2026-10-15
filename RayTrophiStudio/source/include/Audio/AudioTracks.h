#pragma once
#include "json.hpp"
#include <string>
#include <vector>

namespace audio {
struct AudioClipSample;
struct AudioTrackMix {
    float gain = 1;
    float pan = 0;
    bool muted = false;
};
class AudioTracks {
public:
    static nlohmann::json validate(const nlohmann::json& tracks);
    static void bindClips(nlohmann::json& scene);
    static nlohmann::json mutate(nlohmann::json& scene, const std::string& operation,
                                 const nlohmann::json& params);
    static std::vector<std::string> parameters(const std::string& operation);
    static AudioTrackMix mix(const nlohmann::json& clip, const nlohmann::json& tracks);
    static void apply(AudioClipSample& sample, const AudioTrackMix& mix);
};
} // namespace audio
