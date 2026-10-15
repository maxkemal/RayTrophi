#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/AudioArrangementPreview.h"
#include "Audio/AudioDesignJobs.h"
#include "Audio/AudioTimelineMix.h"
#include "../../../external/miniaudio/miniaudio.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace audio {
using json = nlohmann::json;
namespace {
json mixSnapshot(const json& scene) {
    return {{"assets", scene.at("assets")}, {"clips", scene.at("clips")},
        {"tracks", scene.at("tracks")}, {"mixer", scene.at("mixer")}};
}

json sourceVersions(const json& scene) {
    auto result = json::array();
    for (const auto& asset : scene.at("assets")) {
        const auto text = asset.at("path").get<std::string>();
        const std::filesystem::path path(std::u8string(text.begin(), text.end()));
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error) {
            result.push_back({{"path", text}, {"unavailable", true}});
            continue;
        }
        const auto time = std::filesystem::last_write_time(path, error);
        if (error) {
            throw std::runtime_error("Cannot inspect arrangement source: " + text);
        }
        result.push_back({{"path", text}, {"size", size},
            {"modified", time.time_since_epoch().count()}});
    }
    return result;
}
}
struct AudioArrangementPreview::Impl {
    AudioPcm pcm;
    ma_audio_buffer buffer{};
    ma_sound voice{};
    bool bufferReady = false, voiceReady = false;
    json report = {{"ready", false}};
    json request;
    json jobReport;
    AudioDesignJobs jobs;
};
AudioArrangementPreview::AudioArrangementPreview() : impl_(std::make_unique<Impl>()) {}
AudioArrangementPreview::~AudioArrangementPreview() {
    stop();
}
void AudioArrangementPreview::stop() {
    impl_->jobs.cancel();
    impl_->jobReport = impl_->jobs.status();
    if (impl_->voiceReady) {
        ma_sound_uninit(&impl_->voice);
        impl_->voiceReady = false;
    }
    if (impl_->bufferReady) {
        ma_audio_buffer_uninit(&impl_->buffer);
        impl_->bufferReady = false;
    }
}
void AudioArrangementPreview::clear() {
    stop();
    impl_->pcm = {};
    impl_->report = {{"ready", false}};
}
bool AudioArrangementPreview::playing() const {
    return impl_->voiceReady && ma_sound_is_playing(&impl_->voice);
}
json AudioArrangementPreview::status() const {
    auto result = impl_->report;
    const auto job = impl_->jobs.status();
    result["job"] = impl_->jobReport.is_null() ? job : impl_->jobReport;
    result["rendering"] = job.value("state", "idle") == "queued" ||
        job.value("state", "idle") == "running";
    result["playing"] = playing();
    result["loop"] = impl_->voiceReady && ma_sound_is_looping(&impl_->voice);
    ma_uint64 cursor = 0;
    if (impl_->voiceReady) {
        ma_sound_get_cursor_in_pcm_frames(&impl_->voice, &cursor);
    }
    result["cursor_seconds"] = cursor / 48000.0;
    return result;
}

json AudioArrangementPreview::start(const json& scene, const json& params) {
    // Validate before replacing a valid audition or pending request.
    validateTimelinePreview(scene, params);
    const auto snapshot = mixSnapshot(scene);
    const auto files = sourceVersions(snapshot);
    stop();
    impl_->request = params;
    impl_->jobs.start("arrangement", 0,
        [snapshot, files, params](const AudioCancelCheck& check) {
            check();
            if (sourceVersions(snapshot) != files) {
                throw std::runtime_error("Arrangement source changed before rendering");
            }
            AudioDesignJobs::Result result;
            result.assets = snapshot;
            result.files = files;
            result.design.processed = renderTimelinePreview(snapshot, params, check);
            check();
            if (sourceVersions(snapshot) != files) {
                throw std::runtime_error("Arrangement source changed during rendering");
            }
            return result;
        });
    impl_->jobReport = nullptr;
    return status();
}

void AudioArrangementPreview::tick(const json& scene, ma_engine& engine) {
    auto result = impl_->jobs.take();
    if (!result) {
        return;
    }
    const auto id = result->id;
    try {
        if (result->assets != mixSnapshot(scene) || result->files != sourceVersions(scene)) {
            impl_->jobReport = {{"id", id}, {"state", "stale"}};
            return;
        }
        play(std::move(result->design.processed), engine, impl_->request);
        impl_->jobReport = {{"id", id}, {"state", "applied"}};
    } catch (const std::exception& error) {
        impl_->jobReport = {{"id", id}, {"state", "failed"}, {"error", error.what()}};
    }
}
json AudioArrangementPreview::play(AudioPcm pcm, ma_engine& engine, const json& params) {
    if (pcm.channels != 2 || pcm.sample_rate != 48000 || pcm.frames() == 0) {
        throw std::invalid_argument("Arrangement preview requires nonempty stereo 48 kHz PCM");
    }
    stop();
    impl_->report = {{"ready", false}};
    impl_->pcm = std::move(pcm);
    auto config = ma_audio_buffer_config_init(ma_format_f32, impl_->pcm.channels,
        impl_->pcm.frames(), impl_->pcm.samples.data(), nullptr);
    config.sampleRate = 48000;
    if (ma_audio_buffer_init(&config, &impl_->buffer) != MA_SUCCESS) {
        throw std::runtime_error("Cannot initialize arrangement PCM");
    }
    impl_->bufferReady = true;
    if (ma_sound_init_from_data_source(&engine, &impl_->buffer, MA_SOUND_FLAG_NO_SPATIALIZATION,
        nullptr, &impl_->voice) != MA_SUCCESS) {
        stop();
        throw std::runtime_error("Cannot initialize arrangement voice");
    }
    impl_->voiceReady = true;
    ma_sound_set_looping(&impl_->voice, params.value("loop", false));
    double energy = 0;
    float peak = 0;
    for (const auto value : impl_->pcm.samples) {
        energy += double(value) * value;
        peak = std::max(peak, std::abs(value));
    }
    impl_->report = {{"ready", true}, {"start_seconds", std::llround(params.value("start_seconds", 0.0) * 48000) / 48000.0},
        {"duration_seconds", impl_->pcm.frames() / 48000.0}, {"peak", peak},
        {"rms", std::sqrt(energy / impl_->pcm.samples.size())},
        {"ids", params.value("ids", json::array())}};
    if (ma_sound_start(&impl_->voice) != MA_SUCCESS) {
        stop();
        throw std::runtime_error("Cannot start arrangement preview");
    }
    return status();
}
} // namespace audio
