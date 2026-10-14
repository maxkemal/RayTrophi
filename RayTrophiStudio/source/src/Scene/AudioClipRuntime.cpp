#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "Audio/AudioClipRuntime.h"
#include "Audio/AudioClips.h"
#include "Audio/AudioPcm.h"
#include "Audio/AudioMixerRuntime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace audio {
using json = nlohmann::json;
struct AudioClipRuntime::Impl {
    struct Voice {
        AudioPcm pcm;
        json configuration;
        ma_audio_buffer_ref buffer{};
        ma_sound sound{};
        bool bufferReady = false, soundReady = false;
        ~Voice() {
            if (soundReady) {
                ma_sound_uninit(&sound);
            }
            if (bufferReady) {
                ma_audio_buffer_ref_uninit(&buffer);
            }
        }
    };
    std::map<std::string, std::unique_ptr<Voice>> voices;
    std::map<std::string, std::string> errors;
    std::map<std::string, std::chrono::steady_clock::time_point> failed;
};

AudioClipRuntime::AudioClipRuntime() : impl_(std::make_unique<Impl>()) {}
AudioClipRuntime::~AudioClipRuntime() = default;

void AudioClipRuntime::clear() {
    impl_->voices.clear();
    impl_->errors.clear();
    impl_->failed.clear();
}

std::size_t AudioClipRuntime::voices() const {
    return impl_->voices.size();
}

std::size_t AudioClipRuntime::bytes() const {
    std::size_t result = 0;
    for (const auto& [id, voice] : impl_->voices) {
        result += voice->pcm.samples.size() * sizeof(float);
    }
    return result;
}

void AudioClipRuntime::tick(const json& scene, ma_engine& engine, AudioMixerRuntime& mixer,
                           double time, bool playing, std::size_t voiceSlots,
                           std::size_t byteBudget) {
    const auto now = std::chrono::steady_clock::now();
    std::set<std::string> active;
    unsigned starts = 0;
    for (const auto& clip : scene.at("clips")) {
        const auto sample = AudioClips::evaluate(clip, time);
        const std::string id = clip.at("id");
        if (!sample.active) {
            continue;
        }
        active.insert(id);
        const json* asset = nullptr;
        for (const auto& entry : scene.at("assets")) {
            if (entry.at("id") == clip.at("asset_id")) {
                asset = &entry;
                break;
            }
        }
        const json configuration = {
            {"asset_id", clip.at("asset_id")}, {"path", asset ? asset->at("path") : json()},
            {"source_in", clip.at("source_in_seconds")},
            {"source_out", clip.at("source_out_seconds")},
            {"bus", clip.at("bus")}, {"loop", clip.at("loop")}};
        auto found = impl_->voices.find(id);
        if (found != impl_->voices.end() && found->second->configuration != configuration) {
            impl_->voices.erase(found);
            impl_->failed.erase(id);
            found = impl_->voices.end();
        }
        if (found == impl_->voices.end() && playing && starts < 2 &&
            (!impl_->failed.contains(id) || now - impl_->failed.at(id) >= std::chrono::seconds(5))) {
            try {
                if (impl_->voices.size() >= voiceSlots) {
                    throw std::runtime_error("Shared scene voice budget reached");
                }
                if (!asset) {
                    throw std::runtime_error("Clip asset not found");
                }
                const auto pathText = asset->at("path").get<std::string>();
                auto voice = std::make_unique<Impl::Voice>();
                voice->configuration = configuration;
                voice->pcm = decodeAudioPcm(
                    std::filesystem::path(std::u8string(pathText.begin(), pathText.end())));
                if (voice->pcm.samples.size() * sizeof(float) + bytes() > byteBudget) {
                    throw std::runtime_error("Shared scene decoded-audio budget reached");
                }
                const auto first = static_cast<ma_uint64>(std::llround(
                    clip.at("source_in_seconds").get<double>() * 48000));
                const auto end = static_cast<ma_uint64>(std::llround(
                    clip.at("source_out_seconds").get<double>() * 48000));
                if (end <= first || end > voice->pcm.frames()) {
                    throw std::runtime_error("Clip range is outside decoded PCM");
                }
                if (ma_audio_buffer_ref_init(ma_format_f32, voice->pcm.channels,
                        voice->pcm.samples.data() + first * voice->pcm.channels,
                        end - first, &voice->buffer) != MA_SUCCESS) {
                    throw std::runtime_error("Clip PCM buffer initialization failed");
                }
                voice->bufferReady = true;
                voice->buffer.sampleRate = 48000;
                if (ma_sound_init_from_data_source(&engine,
                        reinterpret_cast<ma_data_source*>(&voice->buffer), 0,
                        mixer.group(clip.at("bus")), &voice->sound) != MA_SUCCESS) {
                    throw std::runtime_error("Clip voice initialization failed");
                }
                voice->soundReady = true;
                ma_sound_set_spatialization_enabled(&voice->sound, false);
                ma_sound_set_pan_mode(&voice->sound, ma_pan_mode_balance);
                ma_sound_set_looping(&voice->sound, clip.at("loop").get<bool>());
                ma_sound_seek_to_pcm_frame(&voice->sound, static_cast<ma_uint64>(std::llround(
                    (sample.source_seconds - clip.at("source_in_seconds").get<double>()) * 48000)));
                impl_->voices[id] = std::move(voice);
                impl_->errors.erase(id);
                impl_->failed.erase(id);
                found = impl_->voices.find(id);
                ++starts;
            } catch (const std::exception& error) {
                impl_->errors[id] = error.what();
                impl_->failed[id] = now;
            }
        }
        if (found != impl_->voices.end()) {
            auto& sound = found->second->sound;
            if (!playing) {
                ma_sound_stop(&sound);
                continue;
            }
            ma_uint64 cursor = 0;
            ma_sound_get_cursor_in_pcm_frames(&sound, &cursor);
            const auto wanted = static_cast<ma_uint64>(std::llround(
                (sample.source_seconds - clip.at("source_in_seconds").get<double>()) * 48000));
            if (std::abs(static_cast<double>(cursor) - wanted) > 960 ||
                !ma_sound_is_playing(&sound)) {
                ma_sound_seek_to_pcm_frame(&sound, wanted);
            }
            const float direct = clip.at("bus") == "Master"
                ? AudioMixer::directMasterGain(scene.at("mixer")) : 1;
            ma_sound_set_volume(&sound, sample.gain * direct);
            ma_sound_set_pitch(&sound, sample.pitch);
            ma_sound_set_pan(&sound, sample.pan);
            ma_sound_set_start_time_in_pcm_frames(&sound, ma_engine_get_time_in_pcm_frames(&engine));
            const double remaining = clip.at("start_seconds").get<double>() +
                                     clip.at("duration_seconds").get<double>() - time;
            ma_sound_set_stop_time_in_pcm_frames(&sound, ma_engine_get_time_in_pcm_frames(&engine) +
                static_cast<ma_uint64>(std::llround(std::max(0.0, remaining) * 48000)));
            if (!ma_sound_is_playing(&sound) && ma_sound_start(&sound) != MA_SUCCESS) {
                impl_->errors[id] = "Clip voice start failed";
            }
        }
    }
    for (auto it = impl_->voices.begin(); it != impl_->voices.end();) {
        if (!active.contains(it->first)) {
            it = impl_->voices.erase(it);
        } else {
            ++it;
        }
    }
}

json AudioClipRuntime::status() const {
    json voices = json::array();
    for (const auto& [id, voice] : impl_->voices) {
        ma_uint64 cursor = 0;
        ma_sound_get_cursor_in_pcm_frames(&voice->sound, &cursor);
        voices.push_back({{"id", id}, {"playing", bool(ma_sound_is_playing(&voice->sound))},
                          {"cursor_frames", cursor}});
    }
    return {{"voices", voices}, {"errors", impl_->errors}, {"decoded_bytes", bytes()}};
}
} // namespace audio
