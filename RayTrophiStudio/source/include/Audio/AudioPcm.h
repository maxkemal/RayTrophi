#pragma once
#include "json.hpp"
#include <filesystem>
#include <vector>
#include <cstdint>
#include <map>
#include <string>

namespace audio {
struct AudioPcm {
    unsigned channels = 0;
    unsigned sample_rate = 48000;
    std::vector<float> samples;
    std::uint64_t frames() const {
        return channels ? samples.size() / channels : 0;
    }
};

AudioPcm decodeAudioPcm(const std::filesystem::path& path);
nlohmann::json audioWaveform(const nlohmann::json& asset, unsigned points);
class AudioWaveformCache {
public:
    nlohmann::json get(const nlohmann::json& asset, unsigned points);
    void clear();
private:
    struct Entry {
        std::filesystem::file_time_type modified;
        std::uintmax_t size = 0;
        std::uint64_t used = 0;
        unsigned points = 0;
        nlohmann::json envelope;
    };
    std::map<std::string, Entry> entries_;
    std::uint64_t clock_ = 0;
};
} // namespace audio
