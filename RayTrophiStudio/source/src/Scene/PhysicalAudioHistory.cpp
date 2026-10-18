#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/PhysicalAudioHistory.h"
#include "Audio/AudioService.h"
#include "Audio/PhysicalAudioSource.h"
#include "Audio/PhysicalAudioBake.h"
#include <algorithm>
#include <chrono>
#include <cmath>
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
double PhysicalAudioHistory::clock(bool timeline, double seconds) const {
    return timeline ? seconds : std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::vector<std::string> PhysicalAudioHistory::parameters(const std::string& operation) {
    if (operation == "physical_history_start") {
        return {"id", "duration_seconds"};
    }
    if (operation == "physical_history_finish") {
        return {"path", "store"};
    }
    return {};
}
json PhysicalAudioHistory::status() const {
    return {{"state", state_}, {"error", error_}, {"source_id", source_.value("id", "")},
        {"elapsed_seconds", elapsed_}, {"maximum_seconds", maximum_},
        {"control_count", controls_.size()}, {"event_count", events_.size()},
        {"held_events", pending_.size()},
        {"scope", "sampled_dry_generator_controls"}, {"sample_exact_capture", false}};
}
void PhysicalAudioHistory::cancel() {
    state_ = "cancelled";
    error_.clear();
    controls_.clear();
    events_.clear();
    pending_.clear();
    source_ = json::object();
}
void PhysicalAudioHistory::sample(AudioService& service, bool timeline, double seconds) {
    if (state_ != "recording") {
        return;
    }
    try {
        require(timeline == timeline_ && generation_ == service.generation(),
            "History clock or scene generation changed");
        const double elapsed = clock(timeline, seconds) - start_;
        require(std::isfinite(elapsed) && elapsed >= elapsed_, "History clock moved backwards");
        require(service.invoke("get", {{"id", source_.at("id")}}) == source_,
            "History source changed");
        elapsed_ = std::min(elapsed, maximum_);
        if (elapsed >= maximum_) {
            state_ = "limit_reached";
            return;
        }
        const auto runtime = service.invoke("source_runtime", {{"id", source_.at("id")}});
        const auto generator = runtime.at("generator_runtime");
        require(generator.value("playback_input", "live_controls") != "scheduled_history",
            "Bake the original scheduled history instead of sampling an active replay");
        const bool resident = generator.at("state") != "not_resident";
        const bool draining = generator.at("state") == "draining";
        const bool paused = !runtime.at("playing").get<bool>();
        const bool impact = source_.at("generator").at("model") == "impact";
        const auto frame = static_cast<std::uint64_t>(std::floor(elapsed_ * 48000));
        if (!paused) {
            for (const auto& event : pending_) {
                appendEvent(event, frame);
            }
            pending_.clear();
        } else if (!pending_.empty() && !resident) {
            throw std::invalid_argument("Held impact source was stopped before consumption");
        }
        json control = {{"frame", frame}, {"activity", impact || draining ? 0 :
            runtime.at("activity").get<float>()}, {"gain", draining ?
            generator.at("held_reaction_gain") : runtime.at("reaction_gain")},
            {"paused", paused}, {"reset", resident && !wasResident_}};
        wasResident_ = resident;
        if (!controls_.empty()) {
            auto comparable = control;
            auto last = controls_.back();
            comparable.erase("frame");
            last.erase("frame");
            // reset is an edge, not an enduring state on subsequent samples.
            if (!control.at("reset").get<bool>()) {
                comparable.erase("reset");
                last.erase("reset");
            }
            if (comparable == last) {
                return;
            }
            if (controls_.back().at("frame") == frame) {
                control["reset"] = control.at("reset").get<bool>() ||
                    controls_.back().at("reset").get<bool>();
                controls_.back() = std::move(control);
                return;
            }
        }
        require(controls_.size() < 4096, "Physical history control capacity reached (4096)");
        controls_.push_back(std::move(control));
    } catch (const std::exception& error) {
        state_ = "failed";
        error_ = error.what();
    }
}
void PhysicalAudioHistory::event(AudioService& service, const std::string& id, const json& event,
                                  bool timeline, double seconds) {
    if (state_ != "recording" || source_.at("id") != id) {
        return;
    }
    sample(service, timeline, seconds);
    if (state_ != "recording") {
        return;
    }
    try {
        const auto last = !pending_.empty() ? pending_.back() :
            !events_.empty() ? events_.back() : json();
        require(last.is_null() || event.at("sequence") > last.at("sequence"),
            "Impact sequence epoch changed during history recording");
        const auto runtime = service.invoke("source_runtime", {{"id", id}});
        if (!runtime.at("playing").get<bool>()) {
            require(pending_.size() < 32, "Too many held history events (32)");
            pending_.push_back(event);
        } else {
            appendEvent(event, static_cast<std::uint64_t>(std::floor(elapsed_ * 48000)));
        }
    } catch (const std::exception& error) {
        state_ = "failed";
        error_ = error.what();
    }
}
void PhysicalAudioHistory::appendEvent(const json& event, std::uint64_t frame) {
    require(events_.size() < 4096, "Physical history event capacity reached (4096)");
    unsigned coincident = 0;
    for (auto index = events_.size(); index > 0 && events_[index - 1].at("frame") == frame; --index) {
        ++coincident;
    }
    require(coincident < 32, "Too many history impacts at one frame (32)");
    events_.push_back({{"frame", frame}, {"sequence", event.at("sequence")},
        {"energy_j", event.at("energy_j")}});
}
json PhysicalAudioHistory::invoke(AudioService& service, const std::string& operation,
                                  const json& params, bool timeline, double seconds) {
    if (operation == "physical_history_cancel") {
        cancel();
        return status();
    }
    if (operation == "physical_history_status") {
        return status();
    }
    if (operation == "physical_history_start") {
        const auto source = service.invoke("get", {{"id", params.at("id")}});
        require(isPhysicalAudioSource(source), "History requires a procedural source");
        const auto runtime = service.invoke("source_runtime", {{"id", source.at("id")}});
        require(runtime.at("generator_runtime").at("state") == "not_resident",
            "Stop the procedural source before starting a history take");
        require(source.at("pitch") == 1 && source.at("reaction").at("pitch_min") == 1 &&
            source.at("reaction").at("pitch_max") == 1, "History currently requires unit pitch");
        const auto maximum = params.value("duration_seconds", json(60));
        require(maximum.is_number() && std::isfinite(maximum.get<double>()) &&
            maximum.get<double>() >= 0.05 && maximum.get<double>() <= 60,
            "History duration must be finite within 0.05..60 seconds");
        const auto begin = clock(timeline, seconds);
        require(std::isfinite(begin), "History clock is unavailable");
        source_ = source;
        controls_ = json::array();
        events_ = json::array();
        pending_ = json::array();
        maximum_ = maximum.get<double>();
        start_ = begin;
        elapsed_ = 0;
        timeline_ = timeline;
        generation_ = service.generation();
        wasResident_ = false;
        error_.clear();
        state_ = "recording";
        sample(service, timeline, seconds);
        // The worker's history always begins at zero, independent of wall-clock call overhead.
        if (!controls_.empty()) {
            controls_[0]["frame"] = 0;
        }
        return status();
    }
    require(operation == "physical_history_finish", "Unknown physical history operation");
    const auto store = params.value("store", json(false));
    require(store.is_boolean(), "History store must be boolean");
    if (params.contains("path")) {
        physicalBakeDestination(params);
    }
    sample(service, timeline, seconds);
    require(state_ == "recording" || state_ == "limit_reached", "No valid history take to finish");
    require(pending_.empty(), "Resume held impacts before finishing their history");
    require(timeline == timeline_ && generation_ == service.generation() &&
        service.invoke("get", {{"id", source_.at("id")}}) == source_, "History source became stale");
    const auto frames = std::max<std::uint64_t>(2400,
        static_cast<std::uint64_t>(std::ceil(elapsed_ * 48000)));
    auto controls = controls_;
    auto events = events_;
    // End-boundary samples do not belong to the half-open output interval.
    while (!controls.empty() && controls.back().at("frame").get<std::uint64_t>() >= frames) {
        controls.erase(controls.end() - 1);
    }
    while (!events.empty() && events.back().at("frame").get<std::uint64_t>() >= frames) {
        events.erase(events.end() - 1);
    }
    json history = {{"id", source_.at("id")}, {"duration_seconds", frames / 48000.0},
        {"controls", controls}, {"events", events}};
    json result = {{"state", "completed"}, {"history", history},
        {"sample_exact_capture", false}, {"acoustic_processing", false}};
    if (store.get<bool>()) {
        auto stored = history;
        stored.erase("id");
        stored["version"] = 1;
        source_ = service.invoke("update", {{"id", source_.at("id")},
            {"generator_history", stored}});
        result["stored"] = true;
    }
    if (params.contains("path")) {
        history["path"] = params.at("path");
        try {
            result["bake_job"] = service.invoke("physical_bake_start", history);
        } catch (const std::exception& error) {
            // Preserve completed storage if later job creation or output ownership fails.
            result["bake_error"] = error.what();
        }
    }
    state_ = "completed";
    return result;
}
}
