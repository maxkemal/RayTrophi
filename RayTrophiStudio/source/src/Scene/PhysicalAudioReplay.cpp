#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/PhysicalAudioSource.h"
#include "Audio/PhysicalAudioSchedule.h"
#include "Audio/PhysicalAudioTimeline.h"
#include "Audio/AudioService.h"
#include <stdexcept>

namespace audio {
using json = nlohmann::json;
namespace {
void require(bool valid, const char* message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}
}
std::vector<std::string> PhysicalAudioRuntime::replayParameters(const std::string& operation) {
    if (operation == "physical_replay_start") {
        return {"id", "duration_seconds", "controls", "events", "start_frame"};
    }
    return {};
}

json PhysicalAudioRuntime::replay(AudioService& service, const std::string& operation,
                                  const json& params) {
    if (operation == "physical_replay_status") {
        auto report = replayJobs_.status();
        report["sources"] = json::array();
        for (const auto& [id, source] : sources_) {
            if (isReplay(id)) {
                auto entry = status(id);
                entry["id"] = id;
                report["sources"].push_back(std::move(entry));
            }
        }
        return report;
    }
    if (operation == "physical_replay_cancel") {
        replayJobs_.cancel();
        preparedReplays_.clear();
        return replayJobs_.status();
    }
    if (operation == "physical_replay_start") {
        const auto source = service.invoke("get", {{"id", params.at("id")}});
        const std::string id = source.at("id");
        require(status(id).at("state") == "not_resident", "Stop the source before preparing replay");
        const auto timeline = validatePhysicalTimeline(source, physicalTimelineInput(source, params));
        const auto start = params.value("start_frame", json(0));
        require(start.is_number_integer() && start.get<double>() >= 0 &&
            start.get<double>() < timeline.at("frames").get<double>(),
            "Replay start_frame must belong to its half-open sample range");
        const auto frame = start.get<std::uint64_t>();
        replayJobs_.start("physical_replay", service.generation(),
            [source, timeline, frame](const AudioCancelCheck& check) {
                AudioDesignJobs::Result result;
                auto schedule = std::make_shared<PhysicalAudioSchedule>(timeline);
                schedule->seek(frame, check);
                result.lifetime = std::move(schedule);
                result.assets = source;
                result.design.report = {{"start_frame", frame}, {"frames", timeline.at("frames")},
                    {"scope", "scheduled_dry_generator"}, {"acoustic_export_parity", false}};
                return result;
            });
        return replayJobs_.status();
    }
    require(operation == "physical_replay_apply", "Unknown physical replay operation");
    auto result = replayJobs_.take();
    if (!result) {
        return replayJobs_.status();
    }
    if (result->revision != service.generation()) {
        return {{"state", "stale"}, {"reason", "scene_generation_changed"}};
    }
    json current;
    try {
        current = service.invoke("get", {{"id", result->assets.at("id")}});
    } catch (const std::invalid_argument&) {
        return {{"state", "stale"}, {"reason", "source_removed"}};
    }
    const std::string id = result->assets.at("id");
    if (current != result->assets || status(id).at("state") != "not_resident") {
        return {{"state", "stale"}, {"reason", "source_changed_or_became_resident"}};
    }
    preparedReplays_[id] = std::static_pointer_cast<PhysicalAudioSchedule>(result->lifetime);
    try {
        service.invoke("play", {{"id", id}});
    } catch (...) {
        preparedReplays_.erase(id);
        service.invoke("stop", {{"id", id}});
        throw;
    }
    auto report = result->design.report;
    report["state"] = "applied";
    report["id"] = id;
    return report;
}
}
