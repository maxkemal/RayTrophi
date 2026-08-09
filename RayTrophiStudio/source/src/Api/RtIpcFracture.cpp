#include "RtIpcFracture.h"

#include "Api/RtApi.h"

#include <vector>

using json = nlohmann::json;

namespace {

json fractureInfoJson(const rtapi::FractureGroupInfo& info) {
    return json{{"group", info.group}, {"shard_count", info.shard_count},
                {"broken_count", info.broken_count},
                {"base_break_impulse", info.base_break_impulse},
                {"effective_break_impulse", info.effective_break_impulse},
                {"integrity_weakening", info.integrity_weakening},
                {"integrity_exponent", info.integrity_exponent},
                {"minimum_threshold_scale", info.minimum_threshold_scale},
                {"mean_integrity", info.mean_integrity},
                {"minimum_integrity", info.minimum_integrity},
                {"remaining_support_ratio", info.remaining_support_ratio}};
}

} // namespace

bool dispatchFractureIpc(const std::string& method, const json& params,
                         const RtIpcFractureEnqueue& enqueue,
                         json& out_result) {
    if (method == "physics.make_fracture_group") {
        const std::string group = params.at("group").get<std::string>();
        const auto shards = params.value("shard_objects", std::vector<std::string>{});
        const float threshold = params.value("break_impulse", 5.0f);
        const bool weakening = params.value("integrity_weakening", true);
        const float exponent = params.value("integrity_exponent", 1.5f);
        const float minimum = params.value("minimum_threshold_scale", 0.15f);
        out_result = enqueue([=](UIContext&) {
            rtapi::FractureGroupInfo info;
            const auto result = rtapi::makePhysicsFractureGroup(
                group, shards, threshold, weakening, exponent, minimum, info);
            return result.ok ? fractureInfoJson(info)
                             : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "physics.get_fracture_group") {
        const std::string group = params.at("group").get<std::string>();
        out_result = enqueue([group](UIContext&) {
            rtapi::FractureGroupInfo info;
            const auto result = rtapi::getPhysicsFractureGroup(group, info);
            return result.ok ? fractureInfoJson(info)
                             : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "physics.break_fracture_group") {
        const std::string group = params.at("group").get<std::string>();
        const float strength = params.value("strength", 6.0f);
        out_result = enqueue([group, strength](UIContext&) {
            const auto result = rtapi::breakPhysicsFractureGroup(group, strength);
            return result.ok ? json(true) : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "physics.apply_fracture_impulse") {
        const std::string group = params.at("group").get<std::string>();
        const auto point = params.value("point", std::vector<float>{0.0f, 0.0f, 0.0f});
        const auto direction = params.value("direction", std::vector<float>{0.0f, 1.0f, 0.0f});
        const float impulse = params.value("impulse", 1.0f);
        if (point.size() != 3u || direction.size() != 3u) {
            out_result = json{{"__error", "point and direction must contain three values"}};
            return true;
        }
        out_result = enqueue([group, point, direction, impulse](UIContext&) {
            bool triggered = false;
            const auto result = rtapi::applyPhysicsFractureImpulse(
                group, Vec3(point[0], point[1], point[2]),
                Vec3(direction[0], direction[1], direction[2]),
                impulse, triggered);
            return result.ok ? json{{"triggered", triggered}}
                             : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "gas.pressure_pulse") {
        const std::string domain = params.at("domain").get<std::string>();
        const auto center = params.value("center", std::vector<float>{});
        const float radius = params.value("radius", 1.0f);
        const float peak = params.value("peak_pressure_kpa", 0.0f);
        const float duration = params.value("duration_seconds", 0.02f);
        const float coupling = params.value("coupling", 1.0f);
        if (center.size() != 3u) {
            out_result = json{{"__error", "center must contain three values"}};
            return true;
        }
        out_result = enqueue([=](UIContext&) {
            uint64_t sequence = 0;
            const auto result = rtapi::emitGasPressurePulse(
                domain, Vec3(center[0], center[1], center[2]), radius,
                peak, duration, coupling, sequence);
            return result.ok ? json{{"sequence", sequence}}
                             : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "gas.structural_impulse_stats") {
        out_result = enqueue([](UIContext&) {
            rtapi::StructuralImpulseInfo info;
            const auto result = rtapi::getStructuralImpulseInfo(info);
            return result.ok
                ? json{{"queued", info.queued}, {"consumed", info.consumed},
                       {"affected_groups", info.affected_groups},
                       {"fractured_groups", info.fractured_groups},
                       {"last_peak_pressure_kpa", info.last_peak_pressure_kpa},
                       {"last_max_impulse", info.last_max_impulse}}
                : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "debris.configure") {
        out_result = enqueue([params](UIContext&) {
            const auto result = rtapi::configureAshDebris(
                params.value("enabled", true), params.value("max_particles", 4096ull),
                params.value("particles_per_kg", 120.0f),
                params.value("near_distance", 12.0f),
                params.value("far_lod_scale", 0.25f),
                params.value("lifetime_seconds", 5.0f));
            return result.ok ? json(true) : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "debris.emit_ash") {
        const auto center = params.value("center", std::vector<float>{});
        const auto velocity = params.value("velocity", std::vector<float>{0, 0, 0});
        if (center.size() != 3u || velocity.size() != 3u) {
            out_result = json{{"__error", "center and velocity must contain three values"}};
            return true;
        }
        out_result = enqueue([params, center, velocity](UIContext&) {
            uint64_t spawned = 0;
            const auto result = rtapi::emitAshDebris(
                Vec3(center[0], center[1], center[2]),
                Vec3(velocity[0], velocity[1], velocity[2]),
                params.value("mass_kg", 0.0f),
                params.value("camera_distance", 0.0f),
                params.value("seed", 1u), spawned);
            return result.ok ? json{{"spawned", spawned}}
                             : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "debris.stats") {
        out_result = enqueue([](UIContext&) {
            rtapi::AshDebrisInfo i;
            const auto result = rtapi::getAshDebrisInfo(i);
            return result.ok
                ? json{{"enabled", i.enabled}, {"max_particles", i.max_particles},
                       {"particles_per_kg", i.particles_per_kg},
                       {"near_distance", i.near_distance},
                       {"far_lod_scale", i.far_lod_scale},
                       {"lifetime_seconds", i.lifetime_seconds},
                       {"alive_particles", i.alive_particles}, {"events", i.events},
                       {"requested_particles", i.requested_particles},
                       {"spawned_particles", i.spawned_particles},
                       {"lod_reduced_particles", i.lod_reduced_particles},
                       {"budget_rejected_particles", i.budget_rejected_particles},
                       {"accepted_mass_kg", i.accepted_mass_kg}}
                : json{{"__error", result.error}};
        });
        return true;
    }
    return false;
}
