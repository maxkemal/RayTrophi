#pragma once
#include "json.hpp"
namespace audio {
class AudioClipBatch {
public:
    static nlohmann::json mutate(nlohmann::json& scene, const nlohmann::json& params);
};
} // namespace audio
