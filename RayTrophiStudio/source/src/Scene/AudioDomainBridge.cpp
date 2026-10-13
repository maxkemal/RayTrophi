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
std::string keyFor(unsigned system, const std::string& domain, const std::string& channel) {
    const std::string base = "@domain/" + std::to_string(system) + "/" + domain;
    return channel == "mixed_contact" ? base :
        "@domain/" + channel + "/" + std::to_string(system) + "/" + domain;
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
            const auto* grain = stats ? &stats->grain_report : nullptr;
            for (const std::string channel : {"mixed_contact", "dem_contact", "fluid_flow",
                                               "gas_flow", "mpm_deformation"}) {
                const bool grainMetric = channel == "dem_contact";
                const bool producerAvailable = grainMetric ? grain && grain->contact_damping_measured :
                    channel == "mixed_contact";
                const double duration = grainMetric ?
                    (grain ? grain->contact_damping_duration_seconds : 0) :
                    (channel == "mixed_contact" && stats ? stats->mixed_contact_duration_seconds : 0);
                const bool measured = producerAvailable && enabled && state &&
                    state->fluid_stats.mixed_model_step &&
                    !state->fluid_stats.mixed_step_held && std::isfinite(duration) && duration > 0;
                const double energy = measured ? (grainMetric ? grain->contact_damping_energy_j :
                    stats->mixed_contact_energy_loss) : 0;
                const bool finite = std::isfinite(energy) && energy >= 0 && energy <= 1e9;
                const double power = finite && std::isfinite(duration) && duration > 0
                    ? energy / duration : 0;
                const bool powerMeasured = measured && finite && std::isfinite(duration) &&
                    duration > 0 && std::isfinite(power) && power >= 0 && power <= 1e9;
                const Vec3 contactPosition = grainMetric ?
                    (grain ? grain->contact_damping_position : Vec3(0)) :
                    (stats ? stats->mixed_contact_position : Vec3(0));
                const double positionWeight = grainMetric ? energy :
                    (stats ? stats->mixed_contact_position_weight : 0);
                const bool localized = measured && std::isfinite(positionWeight) &&
                    positionWeight > 0 && std::isfinite(contactPosition.x) &&
                    std::isfinite(contactPosition.y) && std::isfinite(contactPosition.z);
                const Vec3 center = localized ? contactPosition :
                    (state ? (state->bounds_min + state->bounds_max) * 0.5f : Vec3(0));
                const bool positionValid = std::isfinite(center.x) && std::isfinite(center.y) &&
                                           std::isfinite(center.z);
                const std::string key = keyFor(system.id, domains[i].name, channel);
                const char* reason = !enabled ? "disabled" : (!state ? "not_initialized" :
                    (!producerAvailable ? "channel_metric_not_published" :
                     (!state->fluid_stats.mixed_model_step ? "solver_contact_metric_unavailable" :
                     (state->fluid_stats.mixed_step_held ? "mixed_step_held" :
                      (!std::isfinite(duration) || duration <= 0 ? "contact_energy_not_published" :
                       (!finite ? "invalid_contact_energy" : "available"))))));
                result.push_back({{"system_id", system.id}, {"system_name", system.name},
                    {"domain_name", domains[i].name}, {"signal_key", key},
                    {"domain_channel", channel},
                    {"enabled", enabled}, {"valid", state != nullptr},
                    {"version", state ? state->version : 0}, {"measured", measured && finite},
                    {"contact_energy_j", finite ? energy : 0},
                    {"contact_power_w", powerMeasured ? power : 0},
                    {"power_measured", powerMeasured},
                    {"measurement_duration_seconds", std::isfinite(duration) ? duration : 0},
                    {"reason", reason},
                    {"audio_error", publicationErrors.contains(key) ? publicationErrors.at(key) : ""},
                    {"contact_pairs", measured ? (grainMetric ? grain->contacts :
                        state->fluid_stats.mixed_contact_pairs) : 0},
                    {"position", positionValid ? json::array({center.x, center.y, center.z}) : json()},
                    {"position_semantics", localized && positionValid ?
                        (grainMetric ? "energy_weighted_dem_contact_center" :
                         "impulse_weighted_contact_cell_center") : "domain_center"},
                    {"material_pairs_available", false},
                    {"measurement", grainMetric ? "dem_normal_damping_work_per_published_step" :
                        (channel == "mixed_contact" ?
                         "mixed_contact_energy_loss_per_published_solver_step" : "unavailable")}});
            }
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
        !std::isfinite(time) || time < lastTime) {
        // Reset on scene replacement, undo/load and backwards seek.
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
        // RAM cache retains fluid_stats and restore publishes a fresh version.
        // A forward cached frame is a valid report even without a solver substep.
        // Require both time and version progress so idle redraws/restores cannot
        // refresh TTL with the same frame's contact indefinitely.
        if (time <= lastTime ||
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
