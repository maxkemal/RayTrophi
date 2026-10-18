#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/PhysicalAudioBake.h"
#include "Audio/PhysicalAudioTimeline.h"
#include "Audio/AudioScope.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace audio {
namespace {
using json = nlohmann::json;
void require(bool valid, const char* message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}
std::filesystem::path fromUtf8(const std::string& text) {
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
}
struct TemporaryBake {
    std::filesystem::path temporary;
    std::filesystem::path destination;
    bool ownsDestination = false;
    ~TemporaryBake() {
        std::error_code ignored;
        if (!temporary.empty()) {
            std::filesystem::remove(temporary, ignored);
        }
        if (ownsDestination) {
            std::filesystem::remove(destination, ignored);
        }
    }
};
void word(std::ofstream& stream, std::uint32_t value, unsigned count) {
    for (unsigned byte = 0; byte < count; ++byte) {
        stream.put(static_cast<char>((value >> (8 * byte)) & 255));
    }
}
json writeBake(const AudioPcm& pcm, const std::shared_ptr<TemporaryBake>& artifact,
               const AudioCancelCheck& check) {
#ifdef _WIN32
    wchar_t reserved[MAX_PATH];
    require(GetTempFileNameW(artifact->destination.parent_path().c_str(), L"rtp", 0, reserved) != 0,
        "Cannot reserve physical bake temporary output");
    artifact->temporary = reserved;
#else
    throw std::runtime_error("Physical bake publication currently requires Windows");
#endif
    std::ofstream stream(artifact->temporary, std::ios::binary | std::ios::trunc);
    require(stream.good(), "Cannot open physical bake temporary output");
    const auto frames = static_cast<std::uint32_t>(pcm.frames());
    stream.write("RIFF", 4);
    word(stream, 48 + frames * 4, 4);
    stream.write("WAVEfmt ", 8);
    word(stream, 16, 4);
    word(stream, 3, 2);
    word(stream, 1, 2);
    word(stream, 48000, 4);
    word(stream, 192000, 4);
    word(stream, 4, 2);
    word(stream, 32, 2);
    stream.write("fact", 4);
    word(stream, 4, 4);
    word(stream, frames, 4);
    stream.write("data", 4);
    word(stream, frames * 4, 4);
    double square = 0;
    float peak = 0;
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t index = 0; index < pcm.samples.size(); ++index) {
        if (index % 1024 == 0 && check) {
            check();
        }
        const float sample = pcm.samples[index];
        require(std::isfinite(sample), "Physical bake generated nonfinite PCM");
        const auto bits = std::bit_cast<std::uint32_t>(sample);
        word(stream, bits, 4);
        for (unsigned byte = 0; byte < 4; ++byte) {
            hash ^= (bits >> (8 * byte)) & 255;
            hash *= 1099511628211ULL;
        }
        peak = std::max(peak, std::abs(sample));
        square += static_cast<double>(sample) * sample;
    }
    stream.flush();
    require(stream.good(), "Physical bake WAV write failed");
    stream.close();
    require(!stream.fail(), "Physical bake WAV close failed");
    std::ostringstream digest;
    digest << std::hex << std::setw(16) << std::setfill('0') << hash;
    return {{"frames", frames}, {"sample_rate", 48000}, {"channels", 1},
        {"duration_seconds", static_cast<double>(frames) / 48000},
        {"peak", peak}, {"rms", std::sqrt(square / frames)},
        {"pcm_fnv1a64", digest.str()}, {"limited", false}, {"acoustic_processing", false}};
}
json bakeSources(AudioService& service, const json& params) {
    json sources = json::array();
    if (params.contains("id")) {
        require(!params.contains("scope") && !params.contains("ids"),
            "Physical bake accepts id or scope/ids, not both");
        sources.push_back(service.invoke("get", {{"id", params.at("id")}}));
    } else {
        const auto emitters = service.invoke("list", json::object()).at("emitters");
        const auto ids = resolveAudioScope(emitters, params);
        for (const auto& id : ids) {
            sources.push_back(service.invoke("get", {{"id", id}}));
        }
    }
    require(!sources.empty() && sources.size() <= 32,
        "Physical bake requires 1..32 procedural sources");
    std::sort(sources.begin(), sources.end(), [](const json& left, const json& right) {
        return left.at("id").get<std::string>() < right.at("id").get<std::string>();
    });
    return sources;
}
}

std::filesystem::path physicalBakeDestination(const json& params) {
    require(params.contains("path") && params.at("path").is_string(),
        "Physical bake requires an absolute new .wav path");
    const auto path = fromUtf8(params.at("path").get<std::string>());
    require(path.is_absolute() && (path.extension() == ".wav" || path.extension() == ".WAV"),
        "Physical bake requires an absolute new .wav path");
    require(std::filesystem::is_directory(path.parent_path()) && !std::filesystem::exists(path),
        "Physical bake needs an existing directory and an unused output path");
    return path;
}

json PhysicalAudioBake::invoke(AudioService& service, const std::string& operation,
                              const json& params) {
    if (operation == "physical_bake_status") {
        return jobs_.status();
    }
    if (operation == "physical_bake_cancel") {
        cancel();
        return jobs_.status();
    }
    if (operation == "physical_bake_start") {
        const auto sources = bakeSources(service, params);
        auto timelines = json::array();
        std::uint64_t horizon = 0;
        for (const auto& source : sources) {
            auto timeline = validatePhysicalTimeline(source, physicalTimelineInput(source, params));
            timeline["source_id"] = source.at("id");
            horizon = std::max(horizon, timeline.at("frames").get<std::uint64_t>());
            timelines.push_back(std::move(timeline));
        }
        const auto startValue = params.value("start_frame", json(0));
        const auto endValue = params.value("end_frame_exclusive", json(horizon));
        require(startValue.is_number_integer() && endValue.is_number_integer() &&
            startValue.get<double>() >= 0 && endValue.get<double>() <= horizon &&
            startValue.get<double>() < endValue.get<double>(),
            "Physical bake requires a nonempty half-open sample range within its histories");
        const auto start = startValue.get<std::uint64_t>();
        const auto end = endValue.get<std::uint64_t>();
        const auto path = physicalBakeDestination(params);
        jobs_.start("physical_bake", service.generation(),
            [sources, timelines, path, start, end](const AudioCancelCheck& check) {
                AudioDesignJobs::Result result;
                auto artifact = std::make_shared<TemporaryBake>();
                artifact->destination = path;
                result.lifetime = artifact;
                const auto pcm = renderPhysicalMix(timelines, start, end, check);
                auto report = writeBake(pcm, artifact, check);
                report["timelines"] = timelines;
                if (timelines.size() == 1) {
                    report["timeline"] = timelines.front();
                }
                report["source_count"] = sources.size();
                report["start_frame"] = start;
                report["end_frame_exclusive"] = end;
                report["mixer_processing"] = false;
                report["path"] = pathUtf8(path);
                result.assets = sources;
                result.design.report = std::move(report);
                return result;
            });
        return jobs_.status();
    }
    require(operation == "physical_bake_apply", "Unknown physical bake operation");
    auto result = jobs_.take();
    if (!result) {
        return jobs_.status();
    }
    if (result->revision != service.generation()) {
        return {{"state", "stale"}, {"reason", "scene_generation_changed"}};
    }
    for (const auto& snapshot : result->assets) {
        json source;
        try {
            source = service.invoke("get", {{"id", snapshot.at("id")}});
        } catch (const std::invalid_argument&) {
            return {{"state", "stale"}, {"reason", "source_removed"}};
        }
        if (source != snapshot) {
            return {{"state", "stale"}, {"reason", "source_changed"}};
        }
    }
    const auto artifact = std::static_pointer_cast<TemporaryBake>(result->lifetime);
#ifdef _WIN32
    require(MoveFileExW(artifact->temporary.c_str(), artifact->destination.c_str(),
        MOVEFILE_WRITE_THROUGH) != 0, "Cannot publish physical bake; output may already exist");
#else
    throw std::runtime_error("Physical bake publication currently requires Windows");
#endif
    artifact->ownsDestination = true;
    const auto& report = result->design.report;
    const auto asset = service.invoke("asset_import", {{"path", report.at("path")},
        {"source", std::string("RayTrophi physical bake v2 ") +
            json({{"timelines", report.at("timelines")}, {"start_frame", report.at("start_frame")},
                  {"end_frame_exclusive", report.at("end_frame_exclusive")}}).dump()},
        {"tags", json::array({"synthetic", "physical_bake"})}});
    artifact->ownsDestination = false;
    return {{"state", "applied"}, {"asset", asset}, {"bake", report}};
}

void PhysicalAudioBake::cancel() {
    jobs_.cancel();
}
}
