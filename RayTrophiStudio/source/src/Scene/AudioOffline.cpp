#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "Audio/AudioOffline.h"
#include "Audio/AudioClips.h"
#include "Audio/AudioTracks.h"
#include "Audio/AudioTimelineMix.h"
#include "Audio/AudioPcm.h"
#include "Audio/AudioMixer.h"
#include "Audio/AudioService.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace audio {
namespace {
using json = nlohmann::json;
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

std::uint64_t integer(const json& params, const char* key, std::uint64_t fallback,
                      std::uint64_t low, std::uint64_t high) {
    if (params.contains(key)) {
        require(params.at(key).is_number_integer(), "Export frame/FPS fields must be integers");
        require(params.at(key).get<double>() >= 0, "Export integer must be nonnegative");
    }
    const auto result = params.value(key, fallback);
    require(result >= low && result <= high, "Export integer is outside its range");
    return result;
}

std::uint64_t boundary(std::uint64_t frame, std::uint64_t numerator, std::uint64_t denominator) {
    return (frame * 48000ULL * denominator + numerator / 2) / numerator;
}

void word(std::ofstream& output, std::uint32_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i) {
        output.put(static_cast<char>((value >> (8 * i)) & 255));
    }
}

void header(std::ofstream& output, std::uint32_t frames) {
    output.write("RIFF", 4);
    word(output, 48 + frames * 8, 4);
    output.write("WAVEfmt ", 8);
    word(output, 16, 4);
    word(output, 3, 2); // IEEE float32, stereo, 48kHz.
    word(output, 2, 2);
    word(output, 48000, 4);
    word(output, 48000 * 8, 4);
    word(output, 8, 2);
    word(output, 32, 2);
    output.write("fact", 4);
    word(output, 4, 4);
    word(output, frames, 4);
    output.write("data", 4);
    word(output, frames * 8, 4);
}


}

struct AudioOfflineJob::Impl {
    json report = {{"state", "idle"}};
    json clips = json::array(), mixer;
    std::map<std::string, AudioTrackMix> trackMix;
    std::map<std::string, AudioPcm> assets;
    std::filesystem::path target, temporary;
    std::ofstream output;
    std::uint64_t first = 0, last = 0, cursor = 0, overload = 0;
    double peak = 0, squares = 0;
    bool overwrite = false;
    ~Impl() {
        discard();
    }

    void discard() {
        if (output.is_open()) {
            output.close();
        }
        if (!temporary.empty()) {
            std::error_code error;
            std::filesystem::remove(temporary, error);
        }
        temporary.clear();
        assets.clear();
    }
};

AudioOfflineJob::AudioOfflineJob() : impl_(std::make_unique<Impl>()) {}
AudioOfflineJob::~AudioOfflineJob() {
    impl_->discard();
}

std::vector<std::string> AudioOfflineJob::parameters() {
    return {"path", "start_frame", "end_frame_exclusive", "fps_num", "fps_den", "overwrite"};
}

json AudioOfflineJob::start(const json& scene, const json& params) {
    require(impl_->report.value("state", "") != "running", "An audio export is already running");
    const auto begin = integer(params, "start_frame", 0, 0, 10000000);
    const auto end = integer(params, "end_frame_exclusive", 0, 1, 10000000);
    const auto numerator = integer(params, "fps_num", 24, 1, 1000000);
    const auto denominator = integer(params, "fps_den", 1, 1, 1000000);
    require(end > begin && double(numerator) / denominator >= 1 &&
            double(numerator) / denominator <= 240, "Invalid export range/FPS");
    const auto first = boundary(begin, numerator, denominator);
    const auto last = boundary(end, numerator, denominator);
    require(last > first && last - first <= (0xffffffffULL - 48) / 8,
            "Export exceeds the RIFF WAV size limit");
    const auto text = params.at("path").get<std::string>();
    require(!text.empty(), "Export requires an output path");
    const auto target = std::filesystem::absolute(
        std::filesystem::path(std::u8string(text.begin(), text.end())));
    require(target.extension() == ".wav", "Offline export currently writes .wav float32");
    const auto overwrite = params.value("overwrite", json(false));
    require(overwrite.is_boolean(), "overwrite must be boolean");
    require(!std::filesystem::exists(target) || overwrite.get<bool>(), "Output already exists");
    require(std::filesystem::is_directory(target.parent_path()), "Output folder does not exist");
    auto candidate = std::make_unique<Impl>();
    candidate->clips = AudioClips::validateCollection(scene.value("clips", json::array()),
                                                      scene.at("assets"));
    candidate->mixer = AudioMixer::validate(scene.at("mixer"));
    const auto tracks = AudioTracks::validate(scene.value("tracks", json::array()));
    for (const auto& clip : candidate->clips) {
        candidate->trackMix[clip.at("id").get<std::string>()] = AudioTracks::mix(clip, tracks);
    }
    candidate->first = candidate->cursor = first;
    candidate->last = last;
    candidate->target = target;
    candidate->overwrite = overwrite.get<bool>();
    std::size_t decodedBytes = 0;
    for (const auto& clip : candidate->clips) {
        const double start = clip.at("start_seconds"), duration = clip.at("duration_seconds");
        if (start >= double(last) / 48000 || start + duration <= double(first) / 48000 ||
            clip.at("muted").get<bool>() ||
            candidate->trackMix.at(clip.at("id").get<std::string>()).muted) {
            continue;
        }
        const std::string id = clip.at("asset_id");
        if (candidate->assets.contains(id)) {
            continue;
        }
        for (const auto& asset : scene.at("assets")) {
            if (asset.at("id") != id) {
                continue;
            }
            const auto pathText = asset.at("path").get<std::string>();
            const std::filesystem::path input(std::u8string(pathText.begin(), pathText.end()));
            require(std::filesystem::weakly_canonical(input) != std::filesystem::weakly_canonical(target),
                    "Export cannot overwrite a source asset");
            auto pcm = decodeAudioPcm(input);
            decodedBytes += pcm.samples.size() * sizeof(float);
            require(decodedBytes <= 512ULL * 1024 * 1024, "Offline decoded-audio budget exceeded");
            candidate->assets.emplace(id, std::move(pcm));
        }
    }
    static std::atomic<std::uint64_t> sequence{0};
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    candidate->temporary = target;
    candidate->temporary += ".rt_audio_" + std::to_string(unique) + "_" +
                            std::to_string(++sequence) + ".part";
    require(!std::filesystem::exists(candidate->temporary), "Temporary export path collision");
    candidate->output.open(candidate->temporary, std::ios::binary | std::ios::trunc);
    require(bool(candidate->output), "Could not open temporary audio output");
    header(candidate->output, static_cast<std::uint32_t>(last - first));
    candidate->report = {{"state", "running"}, {"path", pathUtf8(target)},
        {"start_frame", begin}, {"end_frame_exclusive", end}, {"fps_num", numerator},
        {"fps_den", denominator}, {"first_sample", first}, {"end_sample_exclusive", last},
        {"frames", last - first}, {"written_frames", 0}, {"sample_rate", 48000},
        {"channels", 2}, {"format", "wav_float32"}, {"scope", "authored_timeline_clips"},
        {"preview_mute_ignored", true}};
    impl_->discard();
    impl_ = std::move(candidate);
    return status();
}

void AudioOfflineJob::tick() {
    auto& job = *impl_;
    if (job.report.value("state", "") != "running") {
        return;
    }
    try {
        static_assert(std::endian::native == std::endian::little);
        const auto count = std::min<std::uint64_t>(4096, job.last - job.cursor);
        std::vector<float> block(static_cast<std::size_t>(count) * 2, 0);
        for (const auto& clip : job.clips) {
            const double start = clip.at("start_seconds"), duration = clip.at("duration_seconds");
            if (start >= double(job.cursor + count) / 48000 || start + duration <= double(job.cursor) / 48000) {
                continue;
            }
            const auto source = job.assets.find(clip.at("asset_id").get<std::string>());
            if (source == job.assets.end()) {
                continue;
            }
            const auto mix = timelineClipMix(clip,
                job.trackMix.at(clip.at("id").get<std::string>()), job.mixer);
            for (std::uint64_t frame = 0; frame < count; ++frame) {
                auto control = AudioClips::evaluate(clip, double(job.cursor + frame) / 48000);
                accumulateTimelineClip(block[2 * frame], block[2 * frame + 1],
                    source->second, clip, control, mix);
            }
        }
        for (const auto value : block) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Nonfinite offline mix");
            }
            job.peak = std::max(job.peak, std::abs(static_cast<double>(value)));
            job.squares += double(value) * value;
            job.overload += std::abs(value) > 1;
        }
        job.output.write(reinterpret_cast<const char*>(block.data()),
                         static_cast<std::streamsize>(block.size() * sizeof(float)));
        if (!job.output) {
            throw std::runtime_error("Offline audio write failed");
        }
        job.cursor += count;
        job.report["written_frames"] = job.cursor - job.first;
        if (job.cursor == job.last) {
            job.output.close();
            if (!job.output) {
                throw std::runtime_error("Offline audio final flush failed");
            }
#ifdef _WIN32
            const DWORD flags = MOVEFILE_WRITE_THROUGH |
                (job.overwrite ? MOVEFILE_REPLACE_EXISTING : 0);
            if (!MoveFileExW(job.temporary.c_str(), job.target.c_str(), flags)) {
                throw std::runtime_error("Could not publish completed audio output");
            }
#else
            if (job.overwrite) {
                std::filesystem::rename(job.temporary, job.target);
            } else {
                std::filesystem::create_hard_link(job.temporary, job.target);
                std::filesystem::remove(job.temporary);
            }
#endif
            job.temporary.clear();
            job.report["state"] = "completed";
            job.report["peak"] = job.peak;
            job.report["rms"] = std::sqrt(job.squares / (2 * (job.last - job.first)));
            job.report["overload_samples"] = job.overload;
            job.assets.clear();
        }
    } catch (const std::exception& error) {
        job.report["state"] = "failed";
        job.report["error"] = error.what();
        job.discard();
    }
}

void AudioOfflineJob::cancel() {
    if (impl_->report.value("state", "") == "running") {
        impl_->report["state"] = "cancelled";
        impl_->discard();
    }
}

json AudioOfflineJob::status() const {
    return impl_->report;
}
} // namespace audio
