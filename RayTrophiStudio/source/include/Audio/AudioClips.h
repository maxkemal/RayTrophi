#pragma once
#include "json.hpp"
#include <string>
#include <vector>

namespace audio {
struct AudioClipSample {
    bool active = false;
    double source_seconds = 0;
    float gain = 0;
    float pan = 0;
    float pitch = 1;
};

class AudioClips {
public:
    static nlohmann::json validate(const nlohmann::json& input,
                                   const nlohmann::json& assets);
    static nlohmann::json validateCollection(const nlohmann::json& clips,
                                             const nlohmann::json& assets);
    static AudioClipSample evaluate(const nlohmann::json& clip, double seconds);
    static double sourceTravel(const nlohmann::json& validatedClip, double localSeconds);
    static nlohmann::json mutate(nlohmann::json& scene, const std::string& operation,
                                 const nlohmann::json& params);
    static std::vector<std::string> parameters(const std::string& operation);
};
} // namespace audio
