#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "Audio/AudioPcm.h"
#include "Audio/AudioService.h"
#include "../../../external/miniaudio/miniaudio.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace audio {
AudioPcm decodeAudioPcm(const std::filesystem::path& path) {
    struct Decoder {
        ma_decoder value{};
        bool initialized = false;
        ~Decoder() {
            if (initialized) {
                ma_decoder_uninit(&value);
            }
        }
    } decoder;
    auto config = ma_decoder_config_init(ma_format_f32, 0, 48000);
#ifdef _WIN32
    const auto initialized = ma_decoder_init_file_w(path.c_str(), &config, &decoder.value);
#else
    const auto initialized = ma_decoder_init_file(path.c_str(), &config, &decoder.value);
#endif
    if (initialized != MA_SUCCESS) {
        throw std::runtime_error("Audio decode failed: " + pathUtf8(path));
    }
    decoder.initialized = true;
    AudioPcm result;
    ma_format format;
    ma_uint32 channels = 0, sampleRate = 0;
    ma_uint64 frames = 0;
    if (ma_decoder_get_data_format(&decoder.value, &format, &channels, &sampleRate, nullptr, 0) !=
            MA_SUCCESS || (channels != 1 && channels != 2) || sampleRate != 48000 ||
        ma_decoder_get_length_in_pcm_frames(&decoder.value, &frames) != MA_SUCCESS || frames == 0 ||
        frames > 600ULL * 48000 || frames * channels * sizeof(float) > 128ULL * 1024 * 1024) {
        throw std::runtime_error("Invalid or oversized decoded audio");
    }
    result.channels = channels;
    result.samples.resize(static_cast<std::size_t>(frames) * channels);
    ma_uint64 read = 0;
    const auto status = ma_decoder_read_pcm_frames(&decoder.value, result.samples.data(), frames, &read);
    if ((status != MA_SUCCESS && status != MA_AT_END) || read == 0) {
        throw std::runtime_error("Incomplete audio decode");
    }
    result.samples.resize(static_cast<std::size_t>(read) * channels);
    for (const float value : result.samples) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Nonfinite decoded audio sample");
        }
    }
    return result;
}

nlohmann::json audioWaveform(const nlohmann::json& asset, unsigned points) {
    if (points < 16 || points > 4096) {
        throw std::invalid_argument("Waveform points must be in 16..4096");
    }
    const auto pathText = asset.at("path").get<std::string>();
    const std::filesystem::path path(std::u8string(pathText.begin(), pathText.end()));
    const auto pcm = decodeAudioPcm(path);
    const auto frames = pcm.frames();
    nlohmann::json envelope = nlohmann::json::array();
    points = static_cast<unsigned>(std::min<std::uint64_t>(points, frames));
    for (unsigned point = 0; point < points; ++point) {
        const auto first = frames * point / points;
        const auto last = frames * (point + 1) / points;
        nlohmann::json minimum = nlohmann::json::array();
        nlohmann::json maximum = nlohmann::json::array();
        nlohmann::json rms = nlohmann::json::array();
        for (unsigned channel = 0; channel < pcm.channels; ++channel) {
            float low = 1e30f, high = -1e30f;
            double squares = 0;
            for (auto frame = first; frame < last; ++frame) {
                const float value = pcm.samples[frame * pcm.channels + channel];
                low = std::min(low, value);
                high = std::max(high, value);
                squares += static_cast<double>(value) * value;
            }
            minimum.push_back(low);
            maximum.push_back(high);
            rms.push_back(std::sqrt(squares / (last - first)));
        }
        envelope.push_back({{"first_frame", first}, {"end_frame_exclusive", last},
            {"min", minimum}, {"max", maximum}, {"rms", rms}});
    }
    return {{"asset_id", asset.at("id")}, {"sample_rate", pcm.sample_rate},
        {"channels", pcm.channels}, {"frames", frames}, {"points", envelope}};
}
void AudioWaveformCache::clear() {
    entries_.clear();
}

nlohmann::json AudioWaveformCache::get(const nlohmann::json& asset, unsigned points) {
    const auto text = asset.at("path").get<std::string>();
    const std::filesystem::path path(std::u8string(text.begin(), text.end()));
    const auto size = std::filesystem::file_size(path);
    const auto modified = std::filesystem::last_write_time(path);
    const std::string key = asset.at("id").get<std::string>() + "/" + text + "/" +
                            std::to_string(points);
    if (const auto found = entries_.find(key); found != entries_.end() &&
        found->second.size == size && found->second.modified == modified) {
        found->second.used = ++clock_;
        return found->second.envelope;
    }
    auto result = audioWaveform(asset, points);
    entries_.erase(key);
    auto usage = [&] {
        unsigned total = 0;
        for (const auto& [entryKey, entry] : entries_) {
            total += entry.points;
        }
        return total;
    };
    while (!entries_.empty() && (entries_.size() >= 32 || usage() + points > 32768)) {
        auto oldest = entries_.begin();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->second.used < oldest->second.used) {
                oldest = it;
            }
        }
        entries_.erase(oldest);
    }
    entries_[key] = {modified, size, ++clock_, points, result};
    return result;
}
} // namespace audio
