#include "Audio/AudioDomainBridge.h"
#include "Audio/AudioService.h"
#include "scene_ui.h"
#include "scene_data.h"
#include <algorithm>
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
            for (const std::string channel : {"mixed_contact", "dem_contact", "dem_friction",
                    "dem_rolling", "fluid_flow", "gas_flow", "mpm_deformation", "mpm_fracture"}) {
                RayTrophiSim::Fluid::DomainAudioMeasurement measurement;
                std::string semantics = "activity_weighted_center";
                std::string label = "unavailable";
                std::string units = "W";
                if (state) {
                    const auto& stats = state->fluid_stats;
                    const auto& grain = stats.grain_report;
                    if (channel == "mixed_contact") {
                        measurement.energy_measured = true;
                        measurement.measured = stats.mixed_model_step && !stats.mixed_step_held &&
                            stats.mixed_contact_duration_seconds > 0;
                        measurement.energy_j = stats.mixed_contact_energy_loss;
                        measurement.duration_seconds = stats.mixed_contact_duration_seconds;
                        measurement.position_weight = stats.mixed_contact_position_weight;
                        measurement.position = stats.mixed_contact_position;
                        measurement.carriers = stats.mixed_contact_pairs;
                        semantics = "impulse_weighted_contact_cell_center";
                        label = "mixed_contact_energy_loss_per_published_solver_step";
                    } else if (channel == "dem_contact" || channel == "dem_friction" ||
                               channel == "dem_rolling") {
                        measurement.energy_measured = true;
                        measurement.measured = channel == "dem_contact" ? grain.contact_damping_measured :
                            (channel == "dem_friction" ? grain.friction_measured : grain.rolling_measured);
                        measurement.measured = measurement.measured && !stats.mixed_step_held;
                        measurement.energy_j = channel == "dem_contact" ? grain.contact_damping_energy_j :
                            (channel == "dem_friction" ? grain.friction_energy_j : grain.rolling_energy_j);
                        measurement.position = channel == "dem_contact" ? grain.contact_damping_position :
                            (channel == "dem_friction" ? grain.friction_position : grain.rolling_position);
                        measurement.duration_seconds = grain.contact_damping_duration_seconds;
                        measurement.position_weight = measurement.energy_j;
                        measurement.carriers = grain.contacts;
                        semantics = "energy_weighted_dem_contact_center";
                        label = channel == "dem_contact" ? "dem_normal_damping_work_per_published_step" :
                            (channel == "dem_friction" ? "dem_sliding_damping_and_plastic_return_work" :
                             "dem_rolling_damping_plastic_return_and_twist_work");
                    } else if (channel == "fluid_flow") {
                        measurement = state->audio_stats.fluid_flow;
                        label = "fluid_kinetic_transport_activity";
                    } else if (channel == "gas_flow") {
                        measurement = state->audio_stats.gas_flow;
                        label = "physical_gas_kinetic_transport_activity";
                    } else if (channel == "mpm_deformation") {
                        measurement = state->audio_stats.mpm_deformation;
                        label = "mpm_end_state_positive_stress_power";
                    } else {
                        measurement = state->audio_stats.mpm_fracture;
                        label = "mpm_mass_weighted_damage_growth_rate";
                        units = "1/s";
                    }
                }
                const double duration = measurement.duration_seconds;
                const double energy = measurement.energy_j;
                const bool finite = std::isfinite(duration) && duration > 0 &&
                    std::isfinite(energy) && energy >= 0 && energy <= 1e9;
                const double power = finite && measurement.physical_power
                    ? (measurement.energy_measured ? energy / duration : measurement.value) : 0;
                const double value = measurement.physical_power ? power : measurement.value;
                const bool measured = enabled && measurement.measured && finite &&
                    std::isfinite(value) && value >= 0 && value <= 1e9;
                const auto& contact = measurement.position;
                const bool localized = measured && measurement.position_weight > 0 &&
                    std::isfinite(contact.x) && std::isfinite(contact.y) && std::isfinite(contact.z);
                const Vec3 position = localized ? contact :
                    (state ? (state->bounds_min + state->bounds_max) * 0.5f : Vec3(0));
                const std::string key = keyFor(system.id, domains[i].name, channel);
                const std::string reason = !enabled ? "disabled" : (!state ? "not_initialized" :
                    (measured ? "available" : (measurement.measured ? "invalid_measurement" :
                     measurement.reason)));
                result.push_back({{"system_id", system.id}, {"system_name", system.name},
                    {"domain_name", domains[i].name}, {"signal_key", key}, {"domain_channel", channel},
                    {"enabled", enabled}, {"valid", state != nullptr},
                    {"version", state ? std::max(state->version,
                        state->audio_stats.publication_version) : 0}, {"measured", measured},
                    {"contact_energy_j", measured ? energy : 0},
                    {"contact_power_w", measured ? power : 0},
                    {"power_measured", measured && measurement.physical_power},
                    {"energy_measured", measured && measurement.energy_measured},
                    {"value", measured ? value : 0}, {"value_units", units},
                    {"measurement_duration_seconds", finite ? duration : 0}, {"reason", reason},
                    {"audio_error", publicationErrors.contains(key) ? publicationErrors.at(key) : ""},
                    {"contact_pairs", measured ? measurement.carriers : 0},
                    {"carrier_count", measured ? measurement.carriers : 0},
                    {"position", json::array({position.x, position.y, position.z})},
                    {"position_semantics", localized ? semantics : "domain_center"},
                    {"material_pairs_available", false}, {"measurement", label}});
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
        signal["activity_value"] = input.at("value");
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
