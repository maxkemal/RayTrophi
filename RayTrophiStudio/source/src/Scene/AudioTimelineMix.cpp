#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/AudioTimelineMix.h"
#include "Audio/AudioMixer.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace audio {
namespace {
using json = nlohmann::json;
float sample(const AudioPcm& pcm, const json& clip, double seconds, unsigned channel) {
    const auto start = static_cast<std::uint64_t>(std::llround(
        clip.at("source_in_seconds").get<double>() * 48000));
    const auto end = std::min(pcm.frames(), static_cast<std::uint64_t>(std::llround(
        clip.at("source_out_seconds").get<double>() * 48000)));
    const double frame = (seconds - clip.at("source_in_seconds").get<double>()) * 48000;
    if (frame < 0 || end <= start || frame >= end - start) {
        return 0;
    }
    const auto offset = static_cast<std::uint64_t>(frame);
    const auto first = start + offset;
    const auto second = first + 1 < end ? first + 1 :
        (clip.at("loop").get<bool>() ? start : end - 1);
    const unsigned sourceChannel = pcm.channels == 1 ? 0 : channel;
    const double blend = frame - offset;
    return static_cast<float>(pcm.samples[first * pcm.channels + sourceChannel] * (1 - blend) +
                              pcm.samples[second * pcm.channels + sourceChannel] * blend);
}

} // namespace
TimelineClipMix timelineClipMix(const json& clip, const AudioTrackMix& track, const json& mixer) {
    TimelineClipMix result;
    result.track = track;
    const auto bus = clip.at("bus").get<std::string>();
    const float master = AudioMixer::effectiveGain(mixer, "Master");
    const float gain = bus == "Master" ? master * AudioMixer::directMasterGain(mixer) :
        AudioMixer::effectiveGain(mixer, bus) * master;
    const float masterPan = mixer.at("Master").value("pan", 0.0f);
    const float busPan = bus == "Master" ? 0.0f : mixer.at(bus).value("pan", 0.0f);
    result.left = gain * std::min(1.0f, 1.0f - masterPan) * std::min(1.0f, 1.0f - busPan);
    result.right = gain * std::min(1.0f, 1.0f + masterPan) * std::min(1.0f, 1.0f + busPan);
    return result;
}

void accumulateTimelineClip(float& left, float& right, const AudioPcm& pcm,
                            const json& clip, AudioClipSample control, const TimelineClipMix& mix) {
    AudioTracks::apply(control, mix.track);
    if (!control.active) {
        return;
    }
    left += sample(pcm, clip, control.source_seconds, 0) * control.gain *
            std::min(1.0f, 1.0f - control.pan) * mix.left;
    right += sample(pcm, clip, control.source_seconds, 1) * control.gain *
             std::min(1.0f, 1.0f + control.pan) * mix.right;
}

void validateTimelinePreview(const json& scene, const json& params) {
    const auto require = [](bool valid, const char* message) {
        if (!valid) {
            throw std::invalid_argument(message);
        }
    };
    const auto startValue = params.value("start_seconds", json(0));
    const auto endValue = params.value("end_seconds", json(5));
    require(startValue.is_number() && endValue.is_number(), "Preview range must be numeric");
    const double start = startValue.get<double>(), end = endValue.get<double>();
    require(std::isfinite(start) && std::isfinite(end) && start >= 0 && end <= 86400 &&
        end - start >= 0.05 - 1e-7 && end - start <= 30 + 1e-7, "Preview range must be 0.05..30 seconds within the day");
    const auto ids = params.value("ids", json::array());
    require(ids.is_array() && ids.size() <= 1024, "Preview IDs must be an array of at most 1024 IDs");
    std::set<std::string> wanted;
    for (const auto& id : ids) {
        require(id.is_string() && wanted.insert(id.get<std::string>()).second,
                "Preview IDs must be unique strings");
    }
    for (const auto& id : wanted) {
        require(std::any_of(scene.at("clips").begin(), scene.at("clips").end(), [&](const json& clip) {
            return clip.at("id").get<std::string>() == id;
        }), "Unknown preview clip ID");
    }
}

AudioPcm renderTimelinePreview(const json& scene, const json& params,
                             const std::function<void()>& check) {
    validateTimelinePreview(scene, params);
    const auto require = [](bool valid, const char* message) {
        if (!valid) {
            throw std::invalid_argument(message);
        }
    };
    const double start = params.value("start_seconds", 0.0);
    const double end = params.value("end_seconds", 5.0);
    std::set<std::string> wanted;
    for (const auto& id : params.value("ids", json::array())) {
        wanted.insert(id.get<std::string>());
    }
    AudioPcm result;
    result.channels = 2;
    const auto first = static_cast<std::uint64_t>(std::llround(start * 48000));
    const auto last = static_cast<std::uint64_t>(std::llround(end * 48000));
    result.samples.resize((last - first) * 2, 0);
    std::map<std::string, AudioPcm> sources;
    std::size_t bytes = 0, count = 0;
    for (const auto& clip : scene.at("clips")) {
        if (check) {
            check();
        }
        if ((!wanted.empty() && !wanted.contains(clip.at("id").get<std::string>())) ||
            clip.at("start_seconds").get<double>() >= end ||
            clip.at("start_seconds").get<double>() + clip.at("duration_seconds").get<double>() <= start) {
            continue;
        }
        const auto track = AudioTracks::mix(clip, scene.at("tracks"));
        if (track.muted || clip.at("muted").get<bool>()) {
            continue;
        }
        require(++count <= 32, "Preview supports at most 32 clips intersecting the range");
        const auto assetId = clip.at("asset_id").get<std::string>();
        if (!sources.contains(assetId)) {
            const auto asset = std::find_if(scene.at("assets").begin(), scene.at("assets").end(),
                [&](const json& value) { return value.at("id").get<std::string>() == assetId; });
            require(asset != scene.at("assets").end(), "Preview asset is unavailable");
            const auto text = asset->at("path").get<std::string>();
            auto pcm = decodeAudioPcm(std::filesystem::path(std::u8string(text.begin(), text.end())));
            if (check) {
                check();
            }
            bytes += pcm.samples.size() * sizeof(float);
            require(bytes <= 256 * 1024 * 1024, "Preview decoded-audio budget exceeded");
            sources[assetId] = std::move(pcm);
        }
        const auto mix = timelineClipMix(clip, track, scene.at("mixer"));
        const auto& pcm = sources.at(assetId);
        for (std::uint64_t frame = 0; frame < last - first; ++frame) {
            if (check && frame % 4096 == 0) {
                check();
            }
            accumulateTimelineClip(result.samples[2 * frame], result.samples[2 * frame + 1], pcm,
                clip, AudioClips::evaluate(clip, double(first + frame) / 48000), mix);
        }
    }
    for (auto& value : result.samples) {
        require(std::isfinite(value), "Nonfinite arrangement preview sample");
        value = std::clamp(value, -1.0f, 1.0f);
    }
    return result;
}
} // namespace audio
