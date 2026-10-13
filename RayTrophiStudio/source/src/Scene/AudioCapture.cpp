#include "Audio/AudioCapture.h"
#include "Audio/AudioClips.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
#include <utility>

namespace audio {
using json = nlohmann::json;
struct AudioCapture::Impl {
    struct Track {
        json voice;
        double start = 0, end = 0;
        bool closed = false;
        json gain = json::array(), pitch = json::array(), pan = json::array();
    };
    std::string state = "idle", error, name = "Captured";
    double begin = 0, previous = 0, stop = 0, maximum = 120;
    std::vector<Track> tracks;
    std::map<std::string, std::size_t> current;
};

AudioCapture::AudioCapture() : impl_(std::make_unique<Impl>()) {}
AudioCapture::~AudioCapture() = default;

json AudioCapture::start(double time, const json& params) {
    if (impl_->state == "recording" || impl_->state == "ready") {
        throw std::invalid_argument("Finish or cancel the existing audio capture first");
    }
    const double maximum = params.value("max_seconds", 120.0);
    const auto name = params.value("name", std::string("Captured"));
    if (!std::isfinite(time) || time < 0 || !std::isfinite(maximum) ||
        maximum < 1 || maximum > 600 || name.size() > 128) {
        throw std::invalid_argument("Capture requires finite max_seconds 1..600 and a short name");
    }
    impl_ = std::make_unique<Impl>();
    impl_->begin = impl_->previous = time;
    impl_->maximum = maximum;
    impl_->name = name;
    impl_->state = "recording";
    return status();
}

void AudioCapture::sample(double time, const json& voices) {
    auto& capture = *impl_;
    if (capture.state != "recording") {
        return;
    }
    if (!std::isfinite(time) || time < capture.previous) {
        capture.state = "failed";
        capture.error = "Capture clock moved backwards; restart after seeking";
        return;
    }
    bool topologyChanged = voices.size() != capture.current.size();
    for (const auto& voice : voices) {
        topologyChanged = topologyChanged || !capture.current.contains(voice.at("key").get<std::string>());
    }
    if (!topologyChanged && time - capture.previous < 1.0 / 30.0) {
        return;
    }
    if (time > capture.begin + capture.maximum) {
        capture.stop = capture.begin + capture.maximum;
        capture.state = "ready";
        return;
    }
    capture.previous = time;
    try {
        std::set<std::string> seen;
        for (const auto& voice : voices) {
            const std::string key = voice.at("key");
            seen.insert(key);
            auto current = capture.current.find(key);
            if (current == capture.current.end()) {
                if (capture.tracks.size() >= 512) {
                    throw std::runtime_error("Capture track budget reached (512)");
                }
                Impl::Track track;
                track.voice = voice;
                track.start = track.end = time;
                capture.tracks.push_back(std::move(track));
                capture.current[key] = capture.tracks.size() - 1;
                current = capture.current.find(key);
            }
            auto& track = capture.tracks[current->second];
            const double local = time - track.start;
            for (auto pair : {std::pair{&track.gain, "gain"}, std::pair{&track.pitch, "pitch"},
                              std::pair{&track.pan, "pan"}}) {
                auto& curve = *pair.first;
                if (curve.size() >= 4095) {
                    capture.state = "ready";
                    capture.stop = time;
                    capture.error = "Automation budget reached; finish this capture segment";
                    return;
                }
                const double value = voice.at(pair.second);
                if (!std::isfinite(value)) {
                    throw std::runtime_error("Capture received nonfinite control");
                }
                if (!curve.empty() && curve.back()[0].get<double>() == local) {
                    curve.back()[1] = value;
                } else {
                    curve.push_back({local, value});
                }
            }
            track.end = time;
        }
        for (auto it = capture.current.begin(); it != capture.current.end();) {
            if (!seen.contains(it->first)) {
                auto& track = capture.tracks[it->second];
                track.end = time;
                track.closed = true;
                it = capture.current.erase(it);
            } else {
                ++it;
            }
        }
    } catch (const std::exception& exception) {
        capture.state = "failed";
        capture.error = exception.what();
    }
}

json AudioCapture::finish(double time, json& scene) {
    auto& capture = *impl_;
    if (capture.state != "recording" && capture.state != "ready") {
        throw std::invalid_argument("No capture is ready to finish");
    }
    const double stop = capture.state == "ready" ? capture.stop : time;
    if (!std::isfinite(stop) || stop < capture.previous || stop <= capture.begin) {
        throw std::invalid_argument("Capture needs a forward scene-clock interval");
    }
    json candidate = scene;
    json added = json::array();
    for (auto& track : capture.tracks) {
        const double duration = (track.closed ? track.end : stop) - track.start;
        if (duration < 1.0 / 48000) {
            continue;
        }
        for (auto* curve : {&track.gain, &track.pitch, &track.pan}) {
            if (curve->back()[0].get<double>() < duration) {
                curve->push_back({duration, curve->back()[1]});
            }
        }
        const auto trackName = (capture.name + " " +
            track.voice.at("key").get<std::string>()).substr(0, 256);
        json params = {{"name", trackName},
            {"asset_id", track.voice.at("asset_id")}, {"start_seconds", track.start},
            {"duration_seconds", duration}, {"loop", true}, {"pitch", 1}, {"gain", 1},
            {"pan", 0}, {"phase_seconds", track.voice.at("phase_seconds")}, {"bus", track.voice.at("bus")},
            {"gain_curve", track.gain}, {"pitch_curve", track.pitch}, {"pan_curve", track.pan}};
        auto clip = AudioClips::mutate(candidate, "clip_create", params);
        if (!track.voice.value("loop", false)) {
            const double available = clip.at("source_out_seconds").get<double>() -
                clip.at("source_in_seconds").get<double>() - clip.at("phase_seconds").get<double>();
            double limited = duration;
            if (AudioClips::sourceTravel(clip, limited) > available) {
                double low = 0, high = limited;
                for (int iteration = 0; iteration < 60; ++iteration) {
                    const double middle = (low + high) * 0.5;
                    if (AudioClips::sourceTravel(clip, middle) <= available) {
                        low = middle;
                    } else {
                        high = middle;
                    }
                }
                limited = low;
            }
            if (limited < 1.0 / 48000) {
                AudioClips::mutate(candidate, "clip_remove", {{"id", clip.at("id")}});
                continue;
            }
            clip = AudioClips::mutate(candidate, "clip_update", {{"id", clip.at("id")},
                {"duration_seconds", limited}, {"envelope_duration_seconds", duration}, {"loop", false}});
        }
        added.push_back(clip);
    }
    if (added.empty()) {
        throw std::invalid_argument("Capture contains no played voices");
    }
    scene = std::move(candidate);
    capture.state = "completed";
    capture.stop = stop;
    capture.current.clear();
    return {{"state", "completed"}, {"clips", added}, {"start_seconds", capture.begin},
            {"end_seconds", stop}, {"scope", "scene_clock_control_envelopes"}};
}

void AudioCapture::cancel() {
    impl_->state = "cancelled";
    impl_->tracks.clear();
    impl_->current.clear();
}

json AudioCapture::status() const {
    return {{"state", impl_->state}, {"error", impl_->error}, {"name", impl_->name},
        {"start_seconds", impl_->begin}, {"last_sample_seconds", impl_->previous},
        {"tracks", impl_->tracks.size()}, {"max_seconds", impl_->maximum}};
}
} // namespace audio
