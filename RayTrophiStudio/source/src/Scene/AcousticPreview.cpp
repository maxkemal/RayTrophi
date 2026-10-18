#include "Audio/AcousticPreview.h"
#include "Audio/AcousticGpu.h"
#include "Audio/AudioScope.h"
#include <atomic>
#include <cstdint>
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
bool eligible(const json& emitter) {
    return emitter.value("spatial", false);
}
}

bool supportsDiffuseSource(const json& emitter) {
    return emitter.at("cone_inner").get<float>() >= 360 &&
        emitter.at("cone_outer").get<float>() >= 360;
}

json AcousticPreview::configure(const json& emitter, const json& params) {
    require(params.contains("enabled") && params.at("enabled").is_boolean(),
        "acoustics_preview requires boolean enabled");
    const std::string id = emitter.at("id");
    const auto spectral = params.value("spectral", json(true));
    require(spectral.is_boolean(), "spectral must be boolean");
    const auto reflections = params.value("reflections", json(false));
    require(reflections.is_boolean(), "reflections must be boolean");
    const auto diffuse = params.value("diffuse", json(false));
    require(diffuse.is_boolean(), "diffuse must be boolean");
    const auto units = params.value("meters_per_unit", json(1.0));
    require(units.is_number() && std::isfinite(units.get<double>()) &&
        units.get<double>() >= 1e-6 && units.get<double>() <= 1e6,
        "meters_per_unit must be finite within 1e-6..1e6");
    const auto mode = params.value("boundary_mode", json("object_barrier"));
    require(mode.is_string() && (mode == "object_barrier" || mode == "thin_surface"),
        "Preview boundary_mode must be object_barrier or thin_surface");
    if (!params.at("enabled").get<bool>()) {
        entries_.erase(id);
        return {{"id", id}, {"enabled", false}, {"amplitude_gain", 1}};
    }
    require(eligible(emitter), "Acoustic preview requires a spatial source");
    require(!diffuse.get<bool>() || supportsDiffuseSource(emitter),
        "Diffuse transport requires an omnidirectional source");
    require(entries_.contains(id) || entries_.size() < kAcousticGpuMaxSources,
        "Acoustic preview supports at most 32 sources; disable another first");
    static std::atomic<std::uint64_t> revision{0};
    entries_[id] = {{"id", id}, {"enabled", true}, {"meters_per_unit", units},
        {"preview_revision", ++revision},
        {"spectral", spectral}, {"band_energy", json::array({1, 1, 1, 1, 1, 1})},
        {"reflections", reflections}, {"reflection_paths", json::array()},
        {"diffuse", diffuse},
        {"boundary_mode", mode}, {"amplitude_gain", 1}, {"status", "pending"}};
    return entries_.at(id);
}

json AcousticPreview::status() const {
    auto result = json::array();
    for (const auto& [id, entry] : entries_) {
        result.push_back(entry);
    }
    return result;
}

json AcousticPreview::batch(const json& emitters, const json& params) {
    require(params.contains("action"), "Acoustic batch requires action");
    const auto ids = resolveAudioScope(emitters, params);
    const auto action = params.value("action", json("configure"));
    require(action.is_string() && (action == "configure" || action == "dry" ||
        action == "processed"), "Acoustic batch action must be configure, dry or processed");
    const auto config = params.value("config", json::object());
    require(config.is_object(), "config must be an object");
    require(action == "configure" || !params.contains("config"),
        "dry/processed must omit config; they preserve the current recipe");
    for (auto field = config.begin(); field != config.end(); ++field) {
        require(field.key() == "enabled" || field.key() == "spectral" ||
            field.key() == "reflections" || field.key() == "diffuse" ||
            field.key() == "meters_per_unit" || field.key() == "boundary_mode",
            "Unknown acoustic batch config field");
    }
    // Stage every validation and budget change before replacing live entries.
    if (action == "configure") {
        AcousticPreview validator;
        validator.configure({{"id", "validation"}, {"spatial", true},
            {"cone_inner", 360}, {"cone_outer", 360}}, config);
    }
    // Revision tokens may have gaps after rejection; published entries do not change.
    auto staged = *this;
    auto results = json::array();
    for (const auto& id : ids) {
        if (action != "configure") {
            const auto found = staged.entries_.find(id.get<std::string>());
            require(found != staged.entries_.end(),
                "Configure every scoped source before dry/processed comparison");
            found->second["bypassed"] = action == "dry";
            results.push_back(found->second);
            continue;
        }
        for (const auto& emitter : emitters) {
            if (emitter.at("id") == id) {
                results.push_back(staged.configure(emitter, config));
                break;
            }
        }
    }
    entries_.swap(staged.entries_);
    return {{"ok", true}, {"scope", params.value("scope", "selected")},
        {"ids", ids}, {"action", action}, {"entries", results}};
}

void AcousticPreview::update(const std::string& id, const json& report) {
    auto found = entries_.find(id);
    if (found == entries_.end()) {
        return;
    }
    auto& entry = found->second;
    if (report.contains("solver")) {
        entry["solver"] = report.at("solver");
    }
    entry["status"] = report.value("transmission_status", "error");
    // An asynchronous pose refresh has no new transmission estimate yet.
    // Keep the last gain until completion instead of briefly opening the barrier.
    if (entry.at("status") == "pending") {
        entry["pending_report"] = report;
        return;
    }
    entry["report"] = report;
    entry.erase("pending_report");
    entry["amplitude_gain"] = 1;
    entry["band_energy"] = json::array({1, 1, 1, 1, 1, 1});
    entry["reflection_paths"] = json::array();
    // A reflection can remain valid when the direct path is blocked/unknown.
    for (const auto& path : report.value("reflections", json::array())) {
        const bool diffuse = path.value("kind", "specular") == "diffuse";
        if (entry.value(diffuse ? "diffuse" : "reflections", false)) {
            entry["reflection_paths"].push_back(path);
        }
    }
    const auto energy = report.value("transmission_energy", json());
    if (!energy.is_array() || energy.size() != 6) {
        return;
    }
    double mean = 0;
    for (const auto& band : energy) {
        if (!band.is_number() || !std::isfinite(band.get<double>()) ||
            band.get<double>() < 0 || band.get<double>() > 1) {
            entry["status"] = "invalid_energy";
            return;
        }
        mean += band.get<double>() / 6;
    }
    // Transmission coefficients are energy fractions; sound volume is amplitude.
    entry["amplitude_gain"] = std::sqrt(mean);
    entry["band_energy"] = energy;
}

void AcousticPreview::forget(const std::string& id) {
    entries_.erase(id);
}

float AcousticPreview::gain(const json& emitter) const {
    const auto found = entries_.find(emitter.at("id").get<std::string>());
    return found == entries_.end() || !eligible(emitter) ||
        found->second.value("bypassed", false) ? 1.0f :
        found->second.at("amplitude_gain").get<float>();
}

json AcousticPreview::energy(const json& emitter) const {
    const auto found = entries_.find(emitter.at("id").get<std::string>());
    if (found == entries_.end() || !eligible(emitter) || found->second.value("bypassed", false)) {
        return json::array({1, 1, 1, 1, 1, 1});
    }
    const auto& entry = found->second;
    if (entry.value("spectral", true)) {
        return entry.at("band_energy");
    }
    const float amplitude = entry.at("amplitude_gain");
    return json::array({amplitude * amplitude, amplitude * amplitude,
        amplitude * amplitude, amplitude * amplitude, amplitude * amplitude,
        amplitude * amplitude});
}

json AcousticPreview::reflections(const json& emitter) const {
    const auto found = entries_.find(emitter.at("id").get<std::string>());
    if (found == entries_.end() || !eligible(emitter) || found->second.value("bypassed", false)) {
        return json::array();
    }
    return found->second.at("reflection_paths");
}

}
