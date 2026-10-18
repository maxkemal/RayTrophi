#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/PhysicalAudioSchedule.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace audio {
PhysicalAudioSchedule::PhysicalAudioSchedule(const nlohmann::json& timeline)
    : initial_(timeline.at("generator")), dsp_(initial_) {
    frames_ = timeline.at("frames").get<std::uint64_t>();
    controls_.reserve(timeline.at("controls").size());
    events_.reserve(timeline.at("events").size());
    for (const auto& control : timeline.at("controls")) {
        controls_.push_back({control.at("frame").get<std::uint64_t>(),
            control.at("activity").get<float>(), control.at("gain").get<float>(),
            control.at("paused").get<bool>(), control.at("reset").get<bool>()});
    }
    const double reference = initial_.impactMode() ?
        timeline.at("generator").at("reference_energy_j").get<double>() : 1;
    for (const auto& event : timeline.at("events")) {
        events_.push_back({event.at("frame").get<std::uint64_t>(),
            static_cast<float>(std::sqrt(std::min(1.0,
                event.at("energy_j").get<double>() / reference)))});
    }
}

float PhysicalAudioSchedule::render(float* output, std::size_t count) {
    float peak = 0;
    std::size_t produced = 0;
    while (produced < count && cursor_ < frames_) {
        while (control_ < controls_.size() && controls_[control_].frame == cursor_) {
            const auto& control = controls_[control_++];
            activity_ = control.activity;
            gain_ = control.gain;
            paused_ = control.paused;
            if (control.reset) {
                dsp_ = initial_;
            }
        }
        float impulse = 0;
        while (event_ < events_.size() && events_[event_].frame == cursor_) {
            impulse += events_[event_++].impulse;
        }
        auto end = std::min(frames_, cursor_ + static_cast<std::uint64_t>(count - produced));
        if (control_ < controls_.size()) {
            end = std::min(end, controls_[control_].frame);
        }
        if (event_ < events_.size()) {
            end = std::min(end, events_[event_].frame);
        }
        const auto span = static_cast<std::size_t>(end - cursor_);
        auto* destination = output ? output + produced : nullptr;
        if (paused_) {
            if (destination) {
                std::fill_n(destination, span, 0.0f);
            }
        } else {
            const float localPeak = dsp_.render(destination, span, activity_, impulse);
            peak = std::max(peak, localPeak * gain_);
            if (destination) {
                for (std::size_t index = 0; index < span; ++index) {
                    destination[index] *= gain_;
                }
            }
        }
        produced += span;
        cursor_ = end;
    }
    if (output && produced < count) {
        std::fill(output + produced, output + count, 0.0f);
    }
    return peak;
}

void PhysicalAudioSchedule::seek(std::uint64_t frame, const AudioCancelCheck& check) {
    if (frame > frames_) {
        throw std::invalid_argument("Physical schedule seek exceeds its frame range");
    }
    dsp_ = initial_;
    cursor_ = 0;
    control_ = 0;
    event_ = 0;
    activity_ = 0;
    gain_ = 1;
    paused_ = false;
    while (cursor_ < frame) {
        if (check) {
            check();
        }
        render(nullptr, static_cast<std::size_t>(std::min<std::uint64_t>(1024, frame - cursor_)));
    }
}
}
