#include "Audio/AudioDomainBridge.h"
#include "Audio/AudioService.h"
#include "scene_ui.h"
#include "scene_data.h"
#include <cmath>
#include <exception>
#include <map>
#include <memory>
#include <set>

namespace audio {
namespace {
using json = nlohmann::json;
std::map<std::string, std::string> publicationErrors;
std::string keyFor(unsigned system, const std::string& domain) {
    return "@domain/" + std::to_string(system) + "/" + domain;
}
}

json domainAudioInputs(UIContext& ctx) {
    json result = json::array();
    for (const auto& system : ctx.scene.particle_systems) {
        if (!system.runtime) {
            continue;
        }
        const auto& domains = system.runtime->gridDomains();
        const auto& states = system.runtime->gridDomainStates();
        for (std::size_t i = 0; i < domains.size(); ++i) {
            const auto* state = i < states.size() && states[i].valid ? &states[i] : nullptr;
            const bool enabled = system.enabled && system.runtime->enabled() && domains[i].enabled;
            const auto* stats = state ? &state->fluid_stats : nullptr;
            const double duration = stats ? stats->mixed_contact_duration_seconds : 0;
            const bool measured = enabled && state && state->fluid_stats.mixed_model_step &&
                !state->fluid_stats.mixed_step_held && std::isfinite(duration) && duration > 0;
            const double energy = measured ? state->fluid_stats.mixed_contact_energy_loss : 0;
            const bool finite = std::isfinite(energy) && energy >= 0 && energy <= 1e9;
            const double power = finite && std::isfinite(duration) && duration > 0
                ? energy / duration : 0;
            const bool powerMeasured = measured && finite && std::isfinite(duration) &&
                duration > 0 && std::isfinite(power) && power >= 0 && power <= 1e9;
            const bool localized = measured && stats &&
                std::isfinite(stats->mixed_contact_position_weight) &&
                stats->mixed_contact_position_weight > 0 &&
                std::isfinite(stats->mixed_contact_position.x) &&
                std::isfinite(stats->mixed_contact_position.y) &&
                std::isfinite(stats->mixed_contact_position.z);
            const Vec3 center = localized ? stats->mixed_contact_position :
                (state ? (state->bounds_min + state->bounds_max) * 0.5f : Vec3(0));
            const bool positionValid = std::isfinite(center.x) && std::isfinite(center.y) &&
                                       std::isfinite(center.z);
            const std::string key = keyFor(system.id, domains[i].name);
            const char* reason = !enabled ? "disabled" : (!state ? "not_initialized" :
                (!state->fluid_stats.mixed_model_step ? "solver_contact_metric_unavailable" :
                 (state->fluid_stats.mixed_step_held ? "mixed_step_held" :
                  (!std::isfinite(duration) || duration <= 0 ? "contact_energy_not_published" :
                   (!finite ? "invalid_contact_energy" : "available")))));
            result.push_back({{"system_id", system.id}, {"system_name", system.name},
                {"domain_name", domains[i].name}, {"signal_key", key},
                {"enabled", enabled}, {"valid", state != nullptr},
                {"version", state ? state->version : 0}, {"measured", measured && finite},
                {"contact_energy_j", finite ? energy : 0},
                {"contact_power_w", powerMeasured ? power : 0},
                {"power_measured", powerMeasured},
                {"measurement_duration_seconds", std::isfinite(duration) ? duration : 0},
                {"reason", reason},
                {"audio_error", publicationErrors.contains(key) ? publicationErrors.at(key) : ""},
                {"contact_pairs", measured ? state->fluid_stats.mixed_contact_pairs : 0},
                {"position", positionValid ? json::array({center.x, center.y, center.z}) : json()},
                {"position_semantics", localized && positionValid ?
                    "impulse_weighted_contact_cell_center" : "domain_center"},
                {"material_pairs_available", false},
                {"measurement", "mixed_contact_energy_loss_per_published_solver_step"}});
        }
    }
    return result;
}

void tickDomainAudioInputs(UIContext& ctx) {
    static std::weak_ptr<AudioService> owner;
    static std::uint64_t generation = 0;
    static std::map<std::string, std::uint64_t> versions;
    static float lastTime = 0;
    const auto& clock = ctx.scene.getSimulationWorld().getStats();
    const float time = clock.simulated_time_seconds;
    if (owner.lock() != ctx.scene.audio || generation != ctx.scene.audio->generation() ||
        !std::isfinite(time) || time < lastTime ||
        (time != lastTime && clock.substeps_last_advance == 0)) {
        // Reset on scene replacement, undo/load and backwards seek; never replay an old report.
        for (const auto& entry : versions) {
            ctx.scene.audio->invalidateSignal(entry.first);
        }
        owner = ctx.scene.audio;
        generation = ctx.scene.audio->generation();
        versions.clear();
        publicationErrors.clear();
        lastTime = std::isfinite(time) ? time : 0;
        return;
    }
    const auto data = ctx.scene.audio->snapshot();
    std::set<std::string> watched;
    for (const auto& e : data.at("emitters")) {
        if (e.at("reaction").at("driver") == "domain_contact") {
            watched.insert(e.at("reaction").at("signal_key").get<std::string>());
        }
    }
    if (watched.empty()) {
        lastTime = time;
        return;
    }
    const auto inputs = domainAudioInputs(ctx);
    std::set<std::string> available;
    for (const auto& input : inputs) {
        const std::string key = input.at("signal_key");
        if (!watched.contains(key)) {
            continue;
        }
        available.insert(key);
        if (!input.at("measured").get<bool>()) {
            ctx.scene.audio->invalidateSignal(key);
            continue;
        }
        const auto version = input.at("version").get<std::uint64_t>();
        if (clock.substeps_last_advance <= 0 || time <= lastTime ||
            (versions.contains(key) && versions.at(key) >= version)) {
            if (versions.contains(key) && versions.at(key) > version) {
                ctx.scene.audio->invalidateSignal(key);
                versions[key] = version;
            }
            continue;
        }
        versions[key] = version;
        json signal = {{"key", key}, {"value", input.at("contact_energy_j")}, {"ttl_seconds", 0.3f}};
        signal["power_w"] = input.at("contact_power_w");
        if (!input.at("position").is_null()) {
            signal["position"] = input.at("position");
        }
        try {
            ctx.scene.audio->invoke("signal_set", signal);
            publicationErrors.erase(key);
        } catch (const std::exception& error) {
            // Budget/invalid report fails quiet; the old value must not remain latched.
            ctx.scene.audio->invalidateSignal(key);
            publicationErrors[key] = error.what();
        }
    }
    for (const auto& key : watched) {
        if (!available.contains(key)) {
            ctx.scene.audio->invalidateSignal(key);
        }
    }
    for (auto it = versions.begin(); it != versions.end();) {
        if (!watched.contains(it->first)) {
            ctx.scene.audio->invalidateSignal(it->first);
            publicationErrors.erase(it->first);
            it = versions.erase(it);
        } else {
            ++it;
        }
    }
    lastTime = time;
}
}
