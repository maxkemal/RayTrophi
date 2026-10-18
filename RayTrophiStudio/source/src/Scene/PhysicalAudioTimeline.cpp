#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/PhysicalAudioTimeline.h"
#include "Audio/PhysicalAudioSource.h"
#include "Audio/PhysicalAudioSchedule.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace audio {
namespace {
using json = nlohmann::json;
void require(bool valid, const char* message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}
void keys(const json& value, const std::vector<std::string>& allowed) {
    require(value.is_object(), "Physical timeline row must be an object");
    for (auto field = value.begin(); field != value.end(); ++field) {
        require(std::find(allowed.begin(), allowed.end(), field.key()) != allowed.end(),
            "Unknown physical timeline row field");
    }
}
std::uint64_t frame(const json& row, std::uint64_t frames) {
    const auto value = row.value("frame", json());
    require(value.is_number_integer() && value.get<double>() >= 0 &&
        value.get<double>() < static_cast<double>(frames),
        "Physical timeline frame must be an integer in the half-open output range");
    return value.get<std::uint64_t>();
}
double number(const json& row, const char* key, double fallback, double low, double high) {
    const auto value = row.value(key, json(fallback));
    require(value.is_number() && std::isfinite(value.get<double>()) &&
        value.get<double>() >= low && value.get<double>() <= high,
        "Physical timeline numeric control is outside its finite range");
    return value.get<double>();
}
}

json validatePhysicalTimeline(const json& input, const json& params) {
    auto source = input;
    validatePhysicalAudioSource(source);
    require(isPhysicalAudioSource(source), "Physical bake requires a procedural source");
    require(source.at("pitch") == 1 && source.at("reaction").at("pitch_min") == 1 &&
        source.at("reaction").at("pitch_max") == 1,
        "Physical history bake currently requires unit source/reaction pitch");
    const auto duration = number(params, "duration_seconds", 2, 0.05, 60);
    const auto frames = static_cast<std::uint64_t>(std::llround(duration * 48000));
    const auto controls = params.value("controls", json::array());
    const auto events = params.value("events", json::array());
    require(controls.is_array() && controls.size() <= 4096 &&
        events.is_array() && events.size() <= 4096,
        "Physical history allows at most 4096 controls and 4096 events");
    const bool impact = source.at("generator").at("model") == "impact";
    require(impact || events.empty(), "Energy events require an impact generator");
    require(impact || !controls.empty(), "Continuous bake requires activity controls from frame zero");
    auto normalizedControls = json::array();
    auto normalizedEvents = json::array();
    std::uint64_t previous = 0;
    for (const auto& control : controls) {
        keys(control, {"frame", "activity", "gain", "paused", "reset"});
        const auto at = frame(control, frames);
        require(normalizedControls.empty() ? at == 0 : at > previous,
            "Controls must begin at frame zero and have strictly increasing frames");
        const auto activity = number(control, "activity", 0, 0, 1);
        require(!impact || activity == 0, "Impact history has no continuous activity input");
        const auto paused = control.value("paused", json(false));
        const auto reset = control.value("reset", json(false));
        require(paused.is_boolean() && reset.is_boolean(), "paused/reset must be boolean");
        normalizedControls.push_back({{"frame", at}, {"activity", activity},
            {"gain", number(control, "gain", 1, 0, 16)}, {"paused", paused}, {"reset", reset}});
        previous = at;
    }
    previous = 0;
    std::uint64_t previousSequence = 0;
    unsigned coincident = 0;
    std::size_t controlIndex = 0;
    bool paused = false;
    for (const auto& event : events) {
        keys(event, {"frame", "sequence", "energy_j"});
        const auto at = frame(event, frames);
        const auto valid = PhysicalAudioRuntime::validateImpact(source, event);
        const auto sequence = valid.at("sequence").get<std::uint64_t>();
        while (controlIndex < normalizedControls.size() &&
            normalizedControls[controlIndex].at("frame").get<std::uint64_t>() <= at) {
            paused = normalizedControls[controlIndex].at("paused").get<bool>();
            ++controlIndex;
        }
        require(!paused, "An impact cannot be scheduled while its history is paused");
        require(normalizedEvents.empty() || (at >= previous && sequence > previousSequence),
            "Events require nondecreasing frames and strictly increasing sequences");
        coincident = !normalizedEvents.empty() && at == previous ? coincident + 1 : 1;
        require(coincident <= 32, "At most 32 impact events may share a sample frame");
        normalizedEvents.push_back({{"frame", at}, {"sequence", sequence},
            {"energy_j", valid.at("energy_j")}});
        previous = at;
        previousSequence = sequence;
    }
    return {{"version", 1}, {"sample_rate", 48000}, {"frames", frames},
        {"generator", source.at("generator")}, {"source_gain", source.at("gain")},
        {"controls", normalizedControls}, {"events", normalizedEvents},
        {"scope", "dry_mono_generator_history"}, {"acoustic_processing", false}};
}

void validateStoredPhysicalTimeline(json& source) {
    const auto stored = source.value("generator_history", json());
    if (stored.is_null()) {
        source["generator_history"] = nullptr;
        return;
    }
    keys(stored, {"version", "duration_seconds", "controls", "events"});
    const auto version = stored.value("version", json(1));
    require(version.is_number_integer() && version == 1, "Unsupported physical history version");
    require(stored.contains("duration_seconds"), "Stored history requires duration_seconds");
    const auto timeline = validatePhysicalTimeline(source, stored);
    source["generator_history"] = {{"version", 1},
        {"duration_seconds", timeline.at("frames").get<double>() / 48000},
        {"controls", timeline.at("controls")}, {"events", timeline.at("events")}};
}

json physicalTimelineInput(const json& source, const json& params) {
    const bool explicitHistory = params.contains("duration_seconds") ||
        params.contains("controls") || params.contains("events");
    if (!explicitHistory && source.contains("generator_history") &&
        !source.at("generator_history").is_null()) {
        return source.at("generator_history");
    }
    return params;
}

AudioPcm renderPhysicalTimeline(const json& timeline, const AudioCancelCheck& check) {
    return renderPhysicalMix(json::array({timeline}), 0,
        timeline.at("frames").get<std::uint64_t>(), check);
}

AudioPcm renderPhysicalMix(const json& timelines, std::uint64_t start,
                           std::uint64_t end, const AudioCancelCheck& check) {
    require(timelines.is_array() && !timelines.empty() && timelines.size() <= 32,
        "Physical mix requires 1..32 validated histories");
    std::uint64_t horizon = 0;
    for (const auto& timeline : timelines) {
        horizon = std::max(horizon, timeline.at("frames").get<std::uint64_t>());
    }
    require(start < end && end <= horizon && horizon <= 2880000,
        "Physical mix range must belong to its bounded half-open history range");
    AudioPcm pcm;
    pcm.channels = 1;
    const auto frames = static_cast<std::size_t>(end - start);
    pcm.samples.assign(frames, 0);
    std::array<float, 1024> block{};
    // Stable array order preserves floating-point summation order.
    // Only one schedule and one scratch block are resident beside the output.
    for (const auto& timeline : timelines) {
        if (check) {
            check();
        }
        PhysicalAudioSchedule schedule(timeline);
        schedule.seek(std::min(start, schedule.frames()), check);
        const float amplitude = timeline.at("source_gain").get<float>();
        std::size_t cursor = 0;
        while (cursor < frames) {
            if (check) {
                check();
            }
            const auto count = std::min<std::size_t>(block.size(), frames - cursor);
            schedule.render(block.data(), count);
            for (std::size_t index = 0; index < count; ++index) {
                pcm.samples[cursor + index] += block[index] * amplitude;
            }
            cursor += count;
        }
    }
    return pcm;
}
}
