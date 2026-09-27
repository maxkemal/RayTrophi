/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtIpcParticle.cpp
* License:       MIT
* =========================================================================
*
* particle.* IPC dispatch, split out of RtIpc.cpp for particle roadmap Phase 1.
*
* Every per-system method accepts two optional parameters:
*   system_id  (int)     the system's stable id, from particle.list_systems
*   system     (string)  panel index or name
* Neither given = the ACTIVE system, i.e. the particle panel's selection, which
* is what every method did before Phase 1. An explicit reference never falls
* back to the active system; it fails.
*
* JSON keys match the rt.particle Python dicts one-to-one, so a script ports
* between the two verbatim.
*/

#include "RtIpcParticle.h"

#include "Api/RtApi.h"

#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

// Same names and error text as RtIpc.cpp's helpers: gen_ipc_descriptors.py
// reads parameter names and types from these call sites.
std::string requireString(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_string())
        throw std::runtime_error(std::string("missing or invalid string param: ") + key);
    return params[key].get<std::string>();
}

int requireInt(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_number_integer())
        throw std::runtime_error(std::string("missing or invalid int param: ") + key);
    return params[key].get<int>();
}

bool requireBool(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_boolean())
        throw std::runtime_error(std::string("missing or invalid boolean param: ") + key);
    return params[key].get<bool>();
}

std::string optionalString(const json& params, const char* key, const std::string& default_val = "") {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_string()) throw std::runtime_error(std::string("invalid string param: ") + key);
    return params[key].get<std::string>();
}

int optionalInt(const json& params, const char* key, int default_val = 0) {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_number_integer()) throw std::runtime_error(std::string("invalid int param: ") + key);
    return params[key].get<int>();
}

float optionalFloat(const json& params, const char* key, float default_val = 0.0f) {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_number()) throw std::runtime_error(std::string("invalid number param: ") + key);
    return params[key].get<float>();
}

Vec3 requireVec3(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_array() || params[key].size() != 3)
        throw std::runtime_error(std::string("missing or invalid Vec3 param: ") + key);
    const auto& a = params[key];
    return Vec3(a[0].get<float>(), a[1].get<float>(), a[2].get<float>());
}

Vec3 optionalVec3(const json& params, const char* key, const Vec3& default_val = Vec3(0.0f, 0.0f, 0.0f)) {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_array() || params[key].size() != 3)
        throw std::runtime_error(std::string("invalid Vec3 param: ") + key);
    return Vec3(params[key][0].get<float>(), params[key][1].get<float>(), params[key][2].get<float>());
}

json vec3ToJson(const Vec3& v) {
    return json::array({v.x, v.y, v.z});
}

json resultToJson(const rtapi::Result& r) {
    return r.ok ? json(true) : json{{"__error", r.error}};
}

// Reads the two optional addressing keys every per-system method takes.
rtapi::ParticleSystemRef readSystemRef(const json& params) {
    rtapi::ParticleSystemRef ref;
    ref.id = optionalInt(params, "system_id", -1);
    ref.index_or_name = optionalString(params, "system", "");
    return ref;
}

// An emitter is named by `emitter` (index, name or "uid:<n>") or by
// `emitter_uid` (the stable uid as a number).
std::string readEmitterRef(const json& params) {
    if (params.contains("emitter_uid")) {
        if (!params["emitter_uid"].is_number_unsigned() && !params["emitter_uid"].is_number_integer())
            throw std::runtime_error("invalid int param: emitter_uid");
        return "uid:" + std::to_string(params["emitter_uid"].get<uint64_t>());
    }
    return requireString(params, "emitter");
}

json particleEmitterToJson(const rtapi::ParticleEmitterInfo& info) {
    return json{
        {"index", info.index}, {"uid", info.uid}, {"system_id", info.system_id},
        {"name", info.name},
        {"source_mode", info.source_mode}, {"spawn_mode", info.spawn_mode},
        {"source_name", info.source_name}, {"enabled", info.enabled},
        {"point", vec3ToJson(info.point)},
        {"local_offset", vec3ToJson(info.local_offset)},
        {"direction", vec3ToJson(info.direction)},
        {"surface_offset", info.surface_offset},
        {"rate_per_second", info.rate_per_second}, {"burst_count", info.burst_count},
        {"speed", info.speed}, {"spread", info.spread},
        {"lifetime_seconds", info.lifetime_seconds}, {"mass", info.mass},
        {"appearance_profile_id", info.appearance_profile_id},
        {"size_jitter", info.size_jitter},
        {"angular_velocity", info.angular_velocity},
        {"angular_jitter", info.angular_jitter}, {"seed", info.seed},
        // These five were Python-only until Phase 1: an IPC client could not
        // parent an emitter or give it its own gas deposit.
        {"parent_object", info.parent_object},
        {"velocity_space", info.velocity_space},
        {"inherit_velocity", info.inherit_velocity},
        {"override_grid_deposit", info.override_grid_deposit},
        {"grid_density_deposit", info.grid_density_deposit},
        {"grid_temperature_deposit", info.grid_temperature_deposit},
        {"grid_fuel_deposit", info.grid_fuel_deposit}};
}

void applyParticleEmitterPatch(const json& patch, rtapi::ParticleEmitterInfo& info) {
    auto str = [&](const char* key, std::string& target) {
        if (patch.contains(key)) target = patch[key].get<std::string>();
    };
    auto flt = [&](const char* key, float& target) {
        if (patch.contains(key)) target = patch[key].get<float>();
    };
    auto boolean = [&](const char* key, bool& target) {
        if (patch.contains(key)) target = patch[key].get<bool>();
    };
    auto vector = [&](const char* key, Vec3& target) {
        if (patch.contains(key)) target = requireVec3(patch, key);
    };
    str("name", info.name); str("source_mode", info.source_mode);
    str("spawn_mode", info.spawn_mode); str("source_name", info.source_name);
    boolean("enabled", info.enabled);
    vector("point", info.point); vector("local_offset", info.local_offset);
    vector("direction", info.direction);
    flt("surface_offset", info.surface_offset);
    flt("rate_per_second", info.rate_per_second);
    if (patch.contains("burst_count")) info.burst_count = patch["burst_count"].get<int>();
    flt("speed", info.speed); flt("spread", info.spread);
    flt("lifetime_seconds", info.lifetime_seconds); flt("mass", info.mass);
    if (patch.contains("appearance_profile_id"))
        info.appearance_profile_id = patch["appearance_profile_id"].get<uint32_t>();
    flt("size_jitter", info.size_jitter);
    flt("angular_velocity", info.angular_velocity);
    flt("angular_jitter", info.angular_jitter);
    if (patch.contains("seed")) info.seed = patch["seed"].get<unsigned int>();
    str("parent_object", info.parent_object);
    str("velocity_space", info.velocity_space);
    flt("inherit_velocity", info.inherit_velocity);
    boolean("override_grid_deposit", info.override_grid_deposit);
    flt("grid_density_deposit", info.grid_density_deposit);
    flt("grid_temperature_deposit", info.grid_temperature_deposit);
    flt("grid_fuel_deposit", info.grid_fuel_deposit);
}

json particleSystemToJson(const rtapi::ParticleSystemInfo& s) {
    return json{
        {"index", s.index}, {"id", s.id}, {"name", s.name},
        {"active", s.active}, {"enabled", s.enabled}, {"visible", s.visible},
        {"emitter_only", s.emitter_only},
        {"render_in_raytrace", s.render_in_raytrace},
        {"domain_count", s.domain_count},
        {"flow_source_count", s.flow_source_count},
        {"emitter_count", s.emitter_count},
        {"collider_count", s.collider_count},
        {"appearance_profile_count", s.appearance_profile_count}};
}

void applyParticleSystemPatch(const json& patch, rtapi::ParticleSystemInfo& info) {
    if (patch.contains("name")) info.name = patch["name"].get<std::string>();
    if (patch.contains("visible")) info.visible = patch["visible"].get<bool>();
}

// Keys that Phase 1.5 moved to appearance profiles. Accepting and ignoring
// them would turn an old script into a silent no-op, so they are refused.
void rejectMovedAppearanceKeys(const json& params) {
    static const char* kMoved[] = {"start_size", "end_size", "start_opacity",
                                   "end_opacity", "start_color", "end_color", "blend_mode"};
    for (const char* key : kMoved) {
        if (params.contains(key)) {
            throw std::runtime_error(
                std::string(key) + " moved to appearance profiles: set it with "
                "particle.set_appearance (the emitter's appearance_profile_id)");
        }
    }
}

json curveToJson(const std::vector<rtapi::ParticleCurveKeyInfo>& keys) {
    json out = json::array();
    for (const auto& k : keys) out.push_back(json::array({k.t, k.value}));
    return out;
}

json particleAppearanceToJson(const rtapi::ParticleAppearanceInfo& a) {
    json ramp = json::array();
    for (const auto& stop : a.color_ramp)
        ramp.push_back(json::array({stop.t, stop.color.x, stop.color.y, stop.color.z}));
    return json{
        {"id", a.id}, {"system_id", a.system_id}, {"name", a.name}, {"blend", a.blend},
        {"color_ramp", ramp},
        {"opacity_curve", curveToJson(a.opacity_curve)},
        {"size_curve", curveToJson(a.size_curve)},
        {"emission_curve", curveToJson(a.emission_curve)},
        {"used_by_emitter_uids", a.used_by_emitter_uids}};
}

// Validates the SHAPE before anything is enqueued; ranges are checked by rtapi.
void applyParticleAppearancePatch(const json& patch, rtapi::ParticleAppearanceInfo& info) {
    if (patch.contains("name")) info.name = patch["name"].get<std::string>();
    if (patch.contains("blend")) info.blend = patch["blend"].get<std::string>();
    if (patch.contains("color_ramp")) {
        const json& arr = patch["color_ramp"];
        if (!arr.is_array()) throw std::runtime_error("color_ramp must be [[t, r, g, b], ...]");
        info.color_ramp.clear();
        for (const auto& item : arr) {
            if (!item.is_array() || item.size() != 4)
                throw std::runtime_error("color_ramp must be [[t, r, g, b], ...]");
            info.color_ramp.push_back({item[0].get<float>(),
                                       Vec3(item[1].get<float>(), item[2].get<float>(),
                                            item[3].get<float>())});
        }
    }
    auto curve = [&](const char* key, std::vector<rtapi::ParticleCurveKeyInfo>& target) {
        if (!patch.contains(key)) return;
        const json& arr = patch[key];
        if (!arr.is_array())
            throw std::runtime_error(std::string(key) + " must be [[t, value], ...]");
        target.clear();
        for (const auto& item : arr) {
            if (!item.is_array() || item.size() != 2)
                throw std::runtime_error(std::string(key) + " must be [[t, value], ...]");
            target.push_back({item[0].get<float>(), item[1].get<float>()});
        }
    };
    curve("opacity_curve", info.opacity_curve);
    curve("size_curve", info.size_curve);
    curve("emission_curve", info.emission_curve);
}

uint32_t requireProfileId(const json& params) {
    if (!params.contains("profile_id") || !params["profile_id"].is_number_integer() ||
        params["profile_id"].get<int64_t>() <= 0)
        throw std::runtime_error("missing or invalid int param: profile_id");
    return params["profile_id"].get<uint32_t>();
}

json particleRenderToJson(const rtapi::ParticleRenderInfo& r) {
    json sources = json::array();
    for (const auto& s : r.mesh_sources) {
        sources.push_back(json{{"node_name", s.node_name}, {"weight", s.weight},
                               {"resolved", s.resolved}});
    }
    return json{
        {"emitter_only", r.emitter_only},
        {"render_in_raytrace", r.render_in_raytrace},
        {"shape", r.shape},
        {"size_multiplier", r.size_multiplier},
        {"sphere_subdivisions", r.sphere_subdivisions},
        {"emissive", r.emissive},
        {"inherit_color_from_emitter", r.inherit_color_from_emitter},
        {"base_color", vec3ToJson(r.base_color)},
        {"emission_strength", r.emission_strength},
        {"roughness", r.roughness},
        {"mesh_sources", sources}};
}

void applyParticleRenderPatch(const json& patch, rtapi::ParticleRenderInfo& info) {
    auto flt = [&](const char* key, float& target) {
        if (patch.contains(key)) target = patch[key].get<float>();
    };
    auto boolean = [&](const char* key, bool& target) {
        if (patch.contains(key)) target = patch[key].get<bool>();
    };
    boolean("emitter_only", info.emitter_only);
    boolean("render_in_raytrace", info.render_in_raytrace);
    if (patch.contains("shape")) info.shape = patch["shape"].get<std::string>();
    flt("size_multiplier", info.size_multiplier);
    if (patch.contains("sphere_subdivisions"))
        info.sphere_subdivisions = patch["sphere_subdivisions"].get<int>();
    boolean("emissive", info.emissive);
    boolean("inherit_color_from_emitter", info.inherit_color_from_emitter);
    if (patch.contains("base_color")) info.base_color = requireVec3(patch, "base_color");
    flt("emission_strength", info.emission_strength);
    flt("roughness", info.roughness);
    // Replaces the whole list; `resolved` is read-only and ignored here.
    if (patch.contains("mesh_sources")) {
        const json& list = patch["mesh_sources"];
        if (!list.is_array()) throw std::runtime_error("invalid array param: mesh_sources");
        info.mesh_sources.clear();
        for (const json& item : list) {
            rtapi::ParticleRenderMeshSourceInfo entry;
            entry.node_name = requireString(item, "node_name");
            entry.weight = optionalFloat(item, "weight", 1.0f);
            info.mesh_sources.push_back(std::move(entry));
        }
    }
}

json particlePhysicsToJson(const rtapi::ParticlePhysicsInfo& info) {
    return json{
        {"mode", info.mode}, {"quality", info.quality},
        {"execution_policy", info.execution_policy},
        {"particle_radius", info.particle_radius},
        {"self_collision_enabled", info.self_collision_enabled},
        {"solver_iterations", info.solver_iterations},
        {"max_neighbors_per_particle", info.max_neighbors_per_particle},
        {"viscosity", info.viscosity}, {"cohesion", info.cohesion},
        {"pressure_stiffness", info.pressure_stiffness},
        {"rest_density", info.rest_density}, {"buoyancy", info.buoyancy},
        {"gravity_scale", info.gravity_scale}, {"vorticity", info.vorticity},
        {"grid_density_deposit", info.grid_density_deposit},
        {"grid_temperature_deposit", info.grid_temperature_deposit},
        {"grid_fuel_deposit", info.grid_fuel_deposit},
        {"grid_deposit_fade_with_age", info.grid_deposit_fade_with_age}};
}

void applyParticlePhysicsPatch(const json& patch, rtapi::ParticlePhysicsInfo& info) {
    auto str = [&](const char* key, std::string& target) {
        if (patch.contains(key)) target = patch[key].get<std::string>();
    };
    auto flt = [&](const char* key, float& target) {
        if (patch.contains(key)) target = patch[key].get<float>();
    };
    auto integer = [&](const char* key, int& target) {
        if (patch.contains(key)) target = patch[key].get<int>();
    };
    auto boolean = [&](const char* key, bool& target) {
        if (patch.contains(key)) target = patch[key].get<bool>();
    };
    str("mode", info.mode); str("quality", info.quality);
    str("execution_policy", info.execution_policy);
    flt("particle_radius", info.particle_radius);
    boolean("self_collision_enabled", info.self_collision_enabled);
    integer("solver_iterations", info.solver_iterations);
    integer("max_neighbors_per_particle", info.max_neighbors_per_particle);
    flt("viscosity", info.viscosity); flt("cohesion", info.cohesion);
    flt("pressure_stiffness", info.pressure_stiffness);
    flt("rest_density", info.rest_density); flt("buoyancy", info.buoyancy);
    flt("gravity_scale", info.gravity_scale); flt("vorticity", info.vorticity);
    flt("grid_density_deposit", info.grid_density_deposit);
    flt("grid_temperature_deposit", info.grid_temperature_deposit);
    flt("grid_fuel_deposit", info.grid_fuel_deposit);
    boolean("grid_deposit_fade_with_age", info.grid_deposit_fade_with_age);
}

json particleStatsToJson(const rtapi::ParticleStatsInfo& info) {
    return json{{"system_id", info.system_id},
                {"alive_count", info.alive_count}, {"capacity", info.capacity},
                {"emitter_count", info.emitter_count},
                {"collider_count", info.collider_count},
                {"domain_count", info.domain_count},
                {"total_ms", info.total_ms}, {"emit_ms", info.emit_ms},
                {"integrate_ms", info.integrate_ms},
                {"self_collision_ms", info.self_collision_ms},
                {"grid_domain_ms", info.grid_domain_ms},
                {"execution_policy", info.execution_policy},
                {"compute_backend", info.compute_backend},
                {"gpu_status", info.gpu_status},
                {"device_resident", info.device_resident},
                {"step_blocked", info.step_blocked},
                {"stage_backends", {
                    {"emit", info.emit_backend},
                    {"forces", info.forces_backend},
                    {"integrate", info.integrate_backend},
                    {"scene_collision", info.scene_collision_backend},
                    {"self_collision", info.self_collision_backend}}},
                {"gpu_step_ms", info.gpu_step_ms},
                {"residency", info.residency},
                {"resident_capacity", info.resident_capacity},
                {"slot_records", info.slot_records},
                {"step_upload_bytes", info.step_upload_bytes},
                {"step_download_bytes", info.step_download_bytes},
                {"step_dispatch_calls", info.step_dispatch_calls},
                {"step_synchronize_calls", info.step_synchronize_calls},
                {"step_upload_call_ms", info.step_upload_call_ms},
                {"step_download_call_ms", info.step_download_call_ms},
                {"step_synchronize_ms", info.step_synchronize_ms},
                {"snapshot_sync_count", info.snapshot_sync_count},
                {"snapshot_mirrored_steps", info.snapshot_mirrored_steps},
                {"snapshot_download_bytes", info.snapshot_download_bytes},
                {"snapshot_last_reason", info.snapshot_last_reason},
                {"nonfinite_particles", info.nonfinite_particles},
                {"nonfinite_measured", info.nonfinite_measured},
                {"grid_deposit_landed", info.grid_deposit_landed},
                {"grid_deposit_dropped_no_domain", info.grid_deposit_dropped_no_domain},
                {"grid_deposit_dropped_no_channel", info.grid_deposit_dropped_no_channel}};
}

} // namespace

bool dispatchParticleIpc(const std::string& method, const json& params,
                         const RtIpcParticleEnqueue& enqueue,
                         json& out_result) {
    // ── Emitters ─────────────────────────────────────────────────────────
    if (method == "particle.emitters") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([ref](UIContext&) {
            std::vector<rtapi::ParticleEmitterInfo> emitters;
            const rtapi::Result r = rtapi::listParticleEmitters(ref, emitters);
            if (!r.ok) return json{{"__error", r.error}};
            json result = json::array();
            for (const auto& info : emitters) result.push_back(particleEmitterToJson(info));
            return result;
        });
        return true;
    }
    if (method == "particle.get_emitter") {
        const std::string emitter = readEmitterRef(params);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([emitter, ref](UIContext&) {
            rtapi::ParticleEmitterInfo info;
            const rtapi::Result r = rtapi::getParticleEmitter(emitter, info, ref);
            if (!r.ok) return json{{"__error", r.error}};
            return particleEmitterToJson(info);
        });
        return true;
    }
    if (method == "particle.add_emitter") {
        rejectMovedAppearanceKeys(params);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        const json patch = params;
        out_result = enqueue([patch, ref](UIContext&) {
            rtapi::ParticleEmitterInfo info;   // facade defaults
            applyParticleEmitterPatch(patch, info);
            rtapi::ParticleEmitterInfo created;
            const rtapi::Result r = rtapi::addParticleEmitter(info, created, ref);
            if (!r.ok) return json{{"__error", r.error}};
            return particleEmitterToJson(created);
        });
        return true;
    }
    if (method == "particle.set_emitter") {
        rejectMovedAppearanceKeys(params);
        const std::string emitter = readEmitterRef(params);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        json patch = params;
        patch.erase("emitter");
        out_result = enqueue([emitter, ref, patch](UIContext&) {
            rtapi::ParticleEmitterInfo info;
            const rtapi::Result read = rtapi::getParticleEmitter(emitter, info, ref);
            if (!read.ok) return resultToJson(read);
            applyParticleEmitterPatch(patch, info);
            return resultToJson(rtapi::updateParticleEmitter(emitter, info, ref));
        });
        return true;
    }
    if (method == "particle.remove_emitter") {
        const std::string emitter = readEmitterRef(params);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([emitter, ref](UIContext&) {
            return resultToJson(rtapi::removeParticleEmitter(emitter, ref));
        });
        return true;
    }
    if (method == "particle.clear_emitters") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([ref](UIContext&) {
            return resultToJson(rtapi::clearParticleEmitters(ref));
        });
        return true;
    }
    // Timeline keys. Only the channels present are keyed, so two calls can key
    // different channels on the same frame. Was Python-only before Phase 1.
    if (method == "particle.key_emitter") {
        const std::string emitter = readEmitterRef(params);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        rtapi::ParticleEmitterKey key;
        key.frame = requireInt(params, "frame");
        if (params.contains("enabled")) { key.has_enabled = true; key.enabled = requireBool(params, "enabled"); }
        if (params.contains("rate_per_second")) { key.has_rate = true; key.rate_per_second = optionalFloat(params, "rate_per_second", 0.0f); }
        if (params.contains("speed")) { key.has_speed = true; key.speed = optionalFloat(params, "speed", 0.0f); }
        if (params.contains("spread")) { key.has_spread = true; key.spread = optionalFloat(params, "spread", 0.0f); }
        if (params.contains("point")) { key.has_point = true; key.point = optionalVec3(params, "point"); }
        if (params.contains("direction")) { key.has_direction = true; key.direction = optionalVec3(params, "direction"); }
        out_result = enqueue([emitter, ref, key](UIContext&) {
            return resultToJson(rtapi::keyParticleEmitter(emitter, key, ref));
        });
        return true;
    }
    if (method == "particle.clear_emitter_key") {
        const std::string emitter = readEmitterRef(params);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        const int frame = requireInt(params, "frame");
        out_result = enqueue([emitter, ref, frame](UIContext&) {
            return resultToJson(rtapi::clearParticleEmitterKey(emitter, frame, ref));
        });
        return true;
    }

    // ── Systems ──────────────────────────────────────────────────────────
    // -- Appearance profiles (Phase 1.5) ------------------------------------
    if (method == "particle.list_appearances") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([ref](UIContext&) {
            std::vector<rtapi::ParticleAppearanceInfo> list;
            const rtapi::Result r = rtapi::listParticleAppearances(ref, list);
            if (!r.ok) return json{{"__error", r.error}};
            json arr = json::array();
            for (const auto& a : list) arr.push_back(particleAppearanceToJson(a));
            return json{{"appearances", arr}};
        });
        return true;
    }
    if (method == "particle.get_appearance") {
        const uint32_t id = requireProfileId(params);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([id, ref](UIContext&) {
            rtapi::ParticleAppearanceInfo info;
            const rtapi::Result r = rtapi::getParticleAppearance(id, info, ref);
            if (!r.ok) return json{{"__error", r.error}};
            return particleAppearanceToJson(info);
        });
        return true;
    }
    if (method == "particle.add_appearance") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        rtapi::ParticleAppearanceInfo info;
        applyParticleAppearancePatch(params, info);
        out_result = enqueue([info, ref](UIContext&) {
            rtapi::ParticleAppearanceInfo created;
            const rtapi::Result r = rtapi::addParticleAppearance(info, created, ref);
            if (!r.ok) return json{{"__error", r.error}};
            return particleAppearanceToJson(created);
        });
        return true;
    }
    // Read-modify-write: keys not given keep their current value.
    if (method == "particle.set_appearance") {
        const uint32_t id = requireProfileId(params);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        rtapi::ParticleAppearanceInfo shape_check;
        applyParticleAppearancePatch(params, shape_check);
        const json patch = params;
        out_result = enqueue([id, ref, patch](UIContext&) {
            rtapi::ParticleAppearanceInfo info;
            const rtapi::Result read = rtapi::getParticleAppearance(id, info, ref);
            if (!read.ok) return resultToJson(read);
            applyParticleAppearancePatch(patch, info);
            return resultToJson(rtapi::updateParticleAppearance(id, info, ref));
        });
        return true;
    }
    if (method == "particle.remove_appearance") {
        const uint32_t id = requireProfileId(params);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([id, ref](UIContext&) {
            return resultToJson(rtapi::removeParticleAppearance(id, ref));
        });
        return true;
    }
    // ".list" / ".get" in the name is what classifies these as Read in
    // RtIpcSecurity.
    if (method == "particle.list_systems") {
        out_result = enqueue([](UIContext&) {
            std::vector<rtapi::ParticleSystemInfo> systems;
            const rtapi::Result r = rtapi::listParticleSystems(systems);
            if (!r.ok) return json{{"__error", r.error}};
            json arr = json::array();
            for (const auto& s : systems) arr.push_back(particleSystemToJson(s));
            return json{{"systems", arr}};
        });
        return true;
    }
    if (method == "particle.get_system") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([ref](UIContext&) {
            rtapi::ParticleSystemInfo info;
            const rtapi::Result r = rtapi::getParticleSystem(ref, info);
            if (!r.ok) return json{{"__error", r.error}};
            return particleSystemToJson(info);
        });
        return true;
    }
    // Patches name / visible (enabled is read-only, see RtApi.h).
    if (method == "particle.set_system") {
        rejectMovedAppearanceKeys(params);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        const json patch = params;
        out_result = enqueue([ref, patch](UIContext&) {
            rtapi::ParticleSystemInfo info;
            const rtapi::Result read = rtapi::getParticleSystem(ref, info);
            if (!read.ok) return resultToJson(read);
            applyParticleSystemPatch(patch, info);
            return resultToJson(rtapi::updateParticleSystem(ref, info));
        });
        return true;
    }
    if (method == "particle.remove_system") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([ref](UIContext&) {
            return resultToJson(rtapi::removeParticleSystem(ref));
        });
        return true;
    }
    if (method == "particle.set_active_system") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([ref](UIContext&) {
            return resultToJson(rtapi::setActiveParticleSystem(ref));
        });
        return true;
    }
    if (method == "particle.set_system_emitter_only") {
        const std::string system = requireString(params, "system");
        const bool emitter_only = params.value("emitter_only", true);
        out_result = enqueue([system, emitter_only](UIContext&) {
            return resultToJson(rtapi::setParticleSystemEmitterOnly(system, emitter_only));
        });
        return true;
    }
    if (method == "particle.add_system") {
        const std::string name = params.value("name", "Particle System");
        out_result = enqueue([name](UIContext&) {
            rtapi::ParticleSystemInfo info;
            const rtapi::Result r = rtapi::addParticleSystem(name, info);
            if (!r.ok) return json{{"__error", r.error}};
            return particleSystemToJson(info);
        });
        return true;
    }
    if (method == "particle.add_preset") {
        const std::string preset = requireString(params, "preset");
        out_result = enqueue([preset](UIContext&) {
            rtapi::ParticleSystemInfo info;
            const rtapi::Result r = rtapi::addParticleSystemPreset(preset, info);
            if (!r.ok) return json{{"__error", r.error}};
            return particleSystemToJson(info);
        });
        return true;
    }
    if (method == "particle.clear_systems") {
        out_result = enqueue([](UIContext&) {
            return resultToJson(rtapi::clearParticleSystems());
        });
        return true;
    }

    // ── Render settings ──────────────────────────────────────────────────
    if (method == "particle.get_render") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([ref](UIContext&) {
            rtapi::ParticleRenderInfo info;
            const rtapi::Result r = rtapi::getParticleRender(ref, info);
            if (!r.ok) return json{{"__error", r.error}};
            return particleRenderToJson(info);
        });
        return true;
    }
    if (method == "particle.set_render") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        const json patch = params;
        out_result = enqueue([ref, patch](UIContext&) {
            rtapi::ParticleRenderInfo info;
            const rtapi::Result read = rtapi::getParticleRender(ref, info);
            if (!read.ok) return resultToJson(read);
            applyParticleRenderPatch(patch, info);
            return resultToJson(rtapi::updateParticleRender(ref, info));
        });
        return true;
    }

    // ── Solver settings ──────────────────────────────────────────────────
    if (method == "particle.get_physics") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([ref](UIContext&) {
            rtapi::ParticlePhysicsInfo info;
            const rtapi::Result r = rtapi::getParticlePhysics(info, ref);
            if (!r.ok) return json{{"__error", r.error}};
            return particlePhysicsToJson(info);
        });
        return true;
    }
    if (method == "particle.set_physics") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        const json patch = params;
        out_result = enqueue([ref, patch](UIContext&) {
            rtapi::ParticlePhysicsInfo info;
            const rtapi::Result read = rtapi::getParticlePhysics(info, ref);
            if (!read.ok) return resultToJson(read);
            applyParticlePhysicsPatch(patch, info);
            return resultToJson(rtapi::updateParticlePhysics(info, ref));
        });
        return true;
    }

    // ── Statistics and state ─────────────────────────────────────────────
    if (method == "particle.stats") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([ref](UIContext&) {
            rtapi::ParticleStatsInfo info;
            const rtapi::Result r = rtapi::getParticleStats(info, ref);
            if (!r.ok) return json{{"__error", r.error}};
            return particleStatsToJson(info);
        });
        return true;
    }
    if (method == "particle.get_state_sample") {
        const int max_count = params.value("max_count", 256);
        const int offset = params.value("offset", 0);
        const int stride = params.value("stride", 1);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([max_count, offset, stride, ref](UIContext&) {
            rtapi::ParticleStateSample sample;
            const rtapi::Result r =
                rtapi::getParticleStateSample(max_count, offset, stride, sample, ref);
            if (!r.ok) return json{{"__error", r.error}};
            json positions = json::array();
            json velocities = json::array();
            for (int i = 0; i < sample.returned; ++i) {
                positions.push_back(vec3ToJson(sample.positions[static_cast<std::size_t>(i)]));
                velocities.push_back(vec3ToJson(sample.velocities[static_cast<std::size_t>(i)]));
            }
            return json{{"alive_count", sample.alive_count},
                        {"capacity", sample.capacity},
                        {"returned", sample.returned},
                        {"indices", sample.indices},
                        {"positions", positions},
                        {"velocities", velocities},
                        {"ages", sample.ages},
                        {"centroid", vec3ToJson(sample.centroid)},
                        {"mean_velocity", vec3ToJson(sample.mean_velocity)},
                        {"bounds_min", vec3ToJson(sample.bounds_min)},
                        {"bounds_max", vec3ToJson(sample.bounds_max)},
                        {"nonfinite", sample.nonfinite}};
        });
        return true;
    }

    // ── Direct control ───────────────────────────────────────────────────
    if (method == "particle.spawn") {
        const Vec3 position = requireVec3(params, "position");
        const Vec3 velocity = optionalVec3(params, "velocity");
        const float lifetime = params.value("lifetime_seconds", 5.0f);
        const float mass = params.value("mass", 1.0f);
        const float size = params.value("size", 0.05f);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([position, velocity, lifetime, mass, size, ref](UIContext&) {
            int index = -1;
            const rtapi::Result r =
                rtapi::spawnParticle(position, velocity, lifetime, mass, size, index, ref);
            if (!r.ok) return json{{"__error", r.error}};
            return json(index);
        });
        return true;
    }
    if (method == "particle.clear") {
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([ref](UIContext&) {
            return resultToJson(rtapi::clearParticles(ref));
        });
        return true;
    }
    if (method == "particle.step") {
        const float dt = params.value("dt", 0.0166667f);
        const rtapi::ParticleSystemRef ref = readSystemRef(params);
        out_result = enqueue([dt, ref](UIContext&) {
            return resultToJson(rtapi::stepParticleSimulation(dt, ref));
        });
        return true;
    }
    return false;
}
