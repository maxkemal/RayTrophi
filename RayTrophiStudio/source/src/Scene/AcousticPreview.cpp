#include "Audio/AcousticPreview.h"
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
    return emitter.value("spatial", false) &&
        emitter.at("reaction").value("driver", "manual") == "manual";
}
}

json AcousticPreview::configure(const json& emitter, const json& params) {
    require(params.contains("enabled") && params.at("enabled").is_boolean(),
        "acoustics_preview requires boolean enabled");
    const std::string id = emitter.at("id");
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
    require(eligible(emitter), "Acoustic preview requires a spatial source with manual reaction");
    require(entries_.contains(id) || entries_.size() < 4,
        "Acoustic preview supports at most four sources; disable another first");
    static std::atomic<std::uint64_t> revision{0};
    entries_[id] = {{"id", id}, {"enabled", true}, {"meters_per_unit", units},
        {"preview_revision", ++revision},
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

void AcousticPreview::update(const std::string& id, const json& report) {
    auto found = entries_.find(id);
    if (found == entries_.end()) {
        return;
    }
    auto& entry = found->second;
    entry["report"] = report;
    if (report.contains("solver")) {
        entry["solver"] = report.at("solver");
    }
    entry["amplitude_gain"] = 1;
    entry["status"] = report.value("transmission_status", "error");
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
}

void AcousticPreview::forget(const std::string& id) {
    entries_.erase(id);
}

float AcousticPreview::gain(const json& emitter) const {
    const auto found = entries_.find(emitter.at("id").get<std::string>());
    return found == entries_.end() || !eligible(emitter) ? 1.0f :
        found->second.at("amplitude_gain").get<float>();
}

}
