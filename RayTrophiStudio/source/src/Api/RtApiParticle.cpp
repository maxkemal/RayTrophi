/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApiParticle.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* Particle emitter / solver facade (Faz 5.6b).
*
* Scope note: emitters, solver settings, live stats and direct spawn/step only.
* Particle COLLIDERS and grid domains hang off the SAME runtime and are already
* scripted from RtApiFluid.cpp (simulation colliders / fluid domains); adding a
* second spelling here would have meant two facades mutating one runtime.
*
* Both files reach the runtime through scriptSimulationRuntime() and end every
* mutation with invalidateScriptSimulation() (RtApiInternal.h) — dropping that
* leaves the cached simulation frames and the timeline resync stale, so a
* scripted edit silently does nothing on an already-simulated timeline.
*
* ★burst_count is one-shot but is NEVER zeroed to consume it: the runtime keeps
* `burst_consumed` separately so the burst survives serialization and replays
* on rewind. Zeroing the count directly is what once made the Explosion preset
* fire once and stay dead forever, including on disk.
*/

#include "RtApiInternal.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

#include "ParticleSimulation.h"

namespace rtapi {
namespace {

using RayTrophiSim::ParticleEmitterDesc;
using RayTrophiSim::ParticleEmitterSourceMode;
using RayTrophiSim::ParticleEmitterSpawnMode;
using RayTrophiSim::ParticlePhysicsMode;
using RayTrophiSim::ParticlePhysicsSettings;
using RayTrophiSim::ParticleQualityMode;

// Separators are dropped entirely, so "Object Origin", "object-origin" and
// "object_origin" all reach the same enum: the panel labels these with spaces
// and the API spells them with underscores, and a script should not have to
// know which spelling it is holding. Both sides of a comparison go through
// this, so the underscore in the canonical NAME is dropped too.
std::string canonical(const std::string& text) {
    std::string out;
    for (char c : text) {
        if (c == ' ' || c == '-' || c == '_') continue;
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

struct SourceModeName { ParticleEmitterSourceMode mode; const char* name; };
const SourceModeName kSourceModes[] = {
    { ParticleEmitterSourceMode::Point,            "point" },
    { ParticleEmitterSourceMode::ObjectOrigin,     "object_origin" },
    { ParticleEmitterSourceMode::ForceFieldOrigin, "force_field_origin" },
};

struct SpawnModeName { ParticleEmitterSpawnMode mode; const char* name; };
const SpawnModeName kSpawnModes[] = {
    { ParticleEmitterSpawnMode::Center,            "center" },
    { ParticleEmitterSpawnMode::ObjectAABBSurface, "object_aabb_surface" },
    { ParticleEmitterSpawnMode::MeshSurface,       "mesh_surface" },
};

struct PhysicsModeName { ParticlePhysicsMode mode; const char* name; };
const PhysicsModeName kPhysicsModes[] = {
    { ParticlePhysicsMode::Spark,    "spark" },
    { ParticlePhysicsMode::Granular, "granular" },
    { ParticlePhysicsMode::Fluid,    "fluid" },
    { ParticlePhysicsMode::Gas,      "gas" },
};

struct QualityName { ParticleQualityMode mode; const char* name; };
const QualityName kQualities[] = {
    { ParticleQualityMode::Realtime, "realtime" },
    { ParticleQualityMode::Preview,  "preview" },
    { ParticleQualityMode::Offline,  "offline" },
};

template <typename Table>
const char* nameOf(const Table& table, decltype(table[0].mode) mode, const char* fallback) {
    for (const auto& entry : table)
        if (entry.mode == mode) return entry.name;
    return fallback;
}

template <typename Table, typename Enum>
bool parseMode(const Table& table, const std::string& text, Enum& out) {
    const std::string key = canonical(text);
    for (const auto& entry : table) {
        if (key == canonical(entry.name)) { out = entry.mode; return true; }
    }
    return false;
}

template <typename Table>
std::string optionList(const Table& table) {
    std::string out;
    for (const auto& entry : table) {
        if (!out.empty()) out += "|";
        out += entry.name;
    }
    return out;
}

// Emitters have no stable id, so they are addressed the way the panel lists
// them: by index. An all-digit token is an index; anything else is matched
// against the name, first hit wins (the runtime does not unique names).
Result resolveEmitterIndex(const std::string& index_or_name, std::size_t& out_index) {
    auto& emitters = scriptSimulationRuntime().emitters();
    if (emitters.empty()) return Result::fail("no particle emitters in the scene");
    const bool numeric = !index_or_name.empty() &&
        std::all_of(index_or_name.begin(), index_or_name.end(),
                    [](unsigned char c) { return std::isdigit(c) != 0; });
    if (numeric) {
        const long value = std::atol(index_or_name.c_str());
        if (value < 0 || static_cast<std::size_t>(value) >= emitters.size())
            return Result::fail("particle emitter index out of range: " + index_or_name);
        out_index = static_cast<std::size_t>(value);
        return Result::success();
    }
    for (std::size_t i = 0; i < emitters.size(); ++i) {
        if (emitters[i].name == index_or_name) { out_index = i; return Result::success(); }
    }
    return Result::fail("particle emitter not found: " + index_or_name);
}

ParticleEmitterInfo infoFromEmitter(const ParticleEmitterDesc& desc, int index) {
    ParticleEmitterInfo info;
    info.index = index;
    info.name = desc.name;
    info.source_mode = nameOf(kSourceModes, desc.source_mode, "point");
    info.spawn_mode = nameOf(kSpawnModes, desc.spawn_mode, "center");
    info.source_name = desc.source_name;
    info.enabled = desc.enabled;
    info.point = desc.point;
    info.local_offset = desc.local_offset;
    info.direction = desc.direction;
    info.surface_offset = desc.surface_offset;
    info.rate_per_second = desc.rate_per_second;
    info.burst_count = desc.burst_count;
    info.speed = desc.speed;
    info.spread = desc.spread;
    info.lifetime_seconds = desc.lifetime_seconds;
    info.mass = desc.mass;
    info.start_size = desc.start_size;
    info.end_size = desc.end_size;
    info.size_jitter = desc.size_jitter;
    info.start_opacity = desc.start_opacity;
    info.end_opacity = desc.end_opacity;
    info.start_color = desc.start_color;
    info.end_color = desc.end_color;
    info.angular_velocity = desc.angular_velocity;
    info.angular_jitter = desc.angular_jitter;
    info.seed = desc.seed;
    return info;
}

// Enums and ranges are validated BEFORE anything is written, so a rejected
// update leaves the emitter exactly as it was instead of half-applied.
// `accumulator` and `burst_consumed` are runtime bookkeeping and are never
// touched here — see the burst note in the file header.
Result applyInfoToEmitter(const ParticleEmitterInfo& info, ParticleEmitterDesc& desc) {
    ParticleEmitterSourceMode source = desc.source_mode;
    if (!info.source_mode.empty() && !parseMode(kSourceModes, info.source_mode, source))
        return Result::fail("unknown emitter source mode: " + info.source_mode +
                            " (" + optionList(kSourceModes) + ")");
    ParticleEmitterSpawnMode spawn = desc.spawn_mode;
    if (!info.spawn_mode.empty() && !parseMode(kSpawnModes, info.spawn_mode, spawn))
        return Result::fail("unknown emitter spawn mode: " + info.spawn_mode +
                            " (" + optionList(kSpawnModes) + ")");
    // ★An ObjectOrigin emitter whose source object does not exist is ERASED by
    // scene.pruneInvalidParticleObjectBindings(), which runs as ordinary scene
    // maintenance — and an empty source_name counts as "does not exist" (the
    // collider branch of that same prune guards against empty, the emitter
    // branch does not). Without this check a script could set the mode, get a
    // success back, and find the emitter gone on the next frame. Rejecting it
    // here turns a silent disappearance into an explicit error.
    if (source == ParticleEmitterSourceMode::ObjectOrigin) {
        if (info.source_name.empty())
            return Result::fail("source_mode 'object_origin' requires source_name: "
                                "an object-bound emitter with no object is pruned by the scene");
        if (!objectExists(info.source_name))
            return Result::fail("emitter source object not found: " + info.source_name +
                                " (an object-bound emitter with a missing object is pruned)");
    }
    // ForceFieldOrigin is not pruned, so the field may legitimately be created
    // after the emitter; only the obviously-empty binding is rejected.
    if (source == ParticleEmitterSourceMode::ForceFieldOrigin && info.source_name.empty())
        return Result::fail("source_mode 'force_field_origin' requires source_name");
    if (info.rate_per_second < 0.0f)
        return Result::fail("rate_per_second must not be negative");
    if (info.burst_count < 0)
        return Result::fail("burst_count must not be negative");
    if (info.lifetime_seconds <= 0.0f)
        return Result::fail("lifetime_seconds must be positive");
    if (info.mass <= 0.0f)
        return Result::fail("mass must be positive");

    desc.source_mode = source;
    desc.spawn_mode = spawn;
    if (!info.name.empty()) desc.name = info.name;
    desc.source_name = info.source_name;
    desc.enabled = info.enabled;
    desc.point = info.point;
    desc.local_offset = info.local_offset;
    desc.direction = info.direction;
    desc.surface_offset = info.surface_offset;
    desc.rate_per_second = info.rate_per_second;
    desc.burst_count = info.burst_count;
    desc.speed = info.speed;
    desc.spread = info.spread;
    desc.lifetime_seconds = info.lifetime_seconds;
    desc.mass = info.mass;
    desc.start_size = info.start_size;
    desc.end_size = info.end_size;
    desc.size_jitter = info.size_jitter;
    desc.start_opacity = info.start_opacity;
    desc.end_opacity = info.end_opacity;
    desc.start_color = info.start_color;
    desc.end_color = info.end_color;
    desc.angular_velocity = info.angular_velocity;
    desc.angular_jitter = info.angular_jitter;
    desc.seed = info.seed;
    return Result::success();
}

} // namespace

std::vector<ParticleEmitterInfo> listParticleEmitters() {
    std::vector<ParticleEmitterInfo> out;
    if (!g_ctx) return out;
    const auto& emitters = scriptSimulationRuntime().emitters();
    out.reserve(emitters.size());
    for (std::size_t i = 0; i < emitters.size(); ++i)
        out.push_back(infoFromEmitter(emitters[i], static_cast<int>(i)));
    return out;
}

Result getParticleEmitter(const std::string& index_or_name, ParticleEmitterInfo& out) {
    if (!g_ctx) return notBound();
    std::size_t index = 0;
    if (Result r = resolveEmitterIndex(index_or_name, index); !r) return r;
    out = infoFromEmitter(scriptSimulationRuntime().emitters()[index], static_cast<int>(index));
    return Result::success();
}

Result addParticleEmitter(const ParticleEmitterInfo& info, ParticleEmitterInfo& out) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    ParticleEmitterDesc desc;
    if (Result r = applyInfoToEmitter(info, desc); !r) return r;
    // Goes through the scene wrapper, not runtime.addEmitter(), so the active
    // particle-system object is created when the scene has none yet.
    ParticleEmitterDesc& created = g_ctx->scene.addParticleEmitter(desc);
    const auto& emitters = scriptSimulationRuntime().emitters();
    out = infoFromEmitter(created, static_cast<int>(emitters.size()) - 1);
    invalidateScriptSimulation();
    return Result::success();
}

Result removeParticleEmitter(const std::string& index_or_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    std::size_t index = 0;
    if (Result r = resolveEmitterIndex(index_or_name, index); !r) return r;
    if (!scriptSimulationRuntime().removeEmitter(index))
        return Result::fail("could not remove particle emitter: " + index_or_name);
    invalidateScriptSimulation();
    return Result::success();
}

Result updateParticleEmitter(const std::string& index_or_name, const ParticleEmitterInfo& info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    std::size_t index = 0;
    if (Result r = resolveEmitterIndex(index_or_name, index); !r) return r;
    if (Result r = applyInfoToEmitter(info, scriptSimulationRuntime().emitters()[index]); !r)
        return r;
    invalidateScriptSimulation();
    return Result::success();
}

Result clearParticleEmitters() {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    g_ctx->scene.clearParticleEmitters();
    invalidateScriptSimulation();
    return Result::success();
}

Result getParticlePhysics(ParticlePhysicsInfo& out) {
    if (!g_ctx) return notBound();
    const ParticlePhysicsSettings& s = scriptSimulationRuntime().physicsSettings();
    out.mode = nameOf(kPhysicsModes, s.mode, "spark");
    out.quality = nameOf(kQualities, s.quality, "realtime");
    out.particle_radius = s.particle_radius;
    out.self_collision_enabled = s.self_collision_enabled;
    out.solver_iterations = s.solver_iterations;
    out.max_neighbors_per_particle = s.max_neighbors_per_particle;
    out.viscosity = s.viscosity;
    out.cohesion = s.cohesion;
    out.pressure_stiffness = s.pressure_stiffness;
    out.rest_density = s.rest_density;
    out.buoyancy = s.buoyancy;
    out.gravity_scale = s.gravity_scale;
    out.vorticity = s.vorticity;
    out.grid_density_deposit = s.grid_density_deposit;
    out.grid_temperature_deposit = s.grid_temperature_deposit;
    out.grid_fuel_deposit = s.grid_fuel_deposit;
    out.grid_deposit_fade_with_age = s.grid_deposit_fade_with_age;
    return Result::success();
}

Result updateParticlePhysics(const ParticlePhysicsInfo& info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    ParticlePhysicsSettings& s = scriptSimulationRuntime().physicsSettings();

    ParticlePhysicsMode mode = s.mode;
    if (!info.mode.empty() && !parseMode(kPhysicsModes, info.mode, mode))
        return Result::fail("unknown particle physics mode: " + info.mode +
                            " (" + optionList(kPhysicsModes) + ")");
    ParticleQualityMode quality = s.quality;
    if (!info.quality.empty() && !parseMode(kQualities, info.quality, quality))
        return Result::fail("unknown particle quality mode: " + info.quality +
                            " (" + optionList(kQualities) + ")");
    if (info.particle_radius <= 0.0f)
        return Result::fail("particle_radius must be positive");
    if (info.solver_iterations < 1)
        return Result::fail("solver_iterations must be at least 1");
    if (info.max_neighbors_per_particle < 1)
        return Result::fail("max_neighbors_per_particle must be at least 1");
    if (info.rest_density <= 0.0f)
        return Result::fail("rest_density must be positive");

    s.mode = mode;
    s.quality = quality;
    s.particle_radius = info.particle_radius;
    s.self_collision_enabled = info.self_collision_enabled;
    s.solver_iterations = info.solver_iterations;
    s.max_neighbors_per_particle = info.max_neighbors_per_particle;
    s.viscosity = info.viscosity;
    s.cohesion = info.cohesion;
    s.pressure_stiffness = info.pressure_stiffness;
    s.rest_density = info.rest_density;
    s.buoyancy = info.buoyancy;
    s.gravity_scale = info.gravity_scale;
    s.vorticity = info.vorticity;
    s.grid_density_deposit = info.grid_density_deposit;
    s.grid_temperature_deposit = info.grid_temperature_deposit;
    s.grid_fuel_deposit = info.grid_fuel_deposit;
    s.grid_deposit_fade_with_age = info.grid_deposit_fade_with_age;
    invalidateScriptSimulation();
    return Result::success();
}

Result getParticleStats(ParticleStatsInfo& out) {
    if (!g_ctx) return notBound();
    auto& runtime = scriptSimulationRuntime();
    const RayTrophiSim::ParticleSimulationStats& stats = runtime.stats();
    // ★The counts come from the LIVE containers, not from stats_: the runtime
    // only refreshes those fields inside step(), so a script that adds an
    // emitter and immediately reads stats() would see 0 and conclude the add
    // failed. The timings below are genuinely per-step measurements and stay
    // as they are (all zero until the first step, which is honest).
    out.alive_count = static_cast<int>(runtime.aliveCount());
    out.capacity = static_cast<int>(runtime.capacity());
    out.emitter_count = static_cast<int>(runtime.emitters().size());
    out.collider_count = static_cast<int>(runtime.colliders().size());
    out.domain_count = static_cast<int>(runtime.gridDomains().size());
    out.total_ms = stats.total_ms;
    out.emit_ms = stats.emit_ms;
    out.integrate_ms = stats.integrate_ms;
    out.self_collision_ms = stats.self_collision_ms;
    out.grid_domain_ms = stats.grid_domain_ms;
    return Result::success();
}

Result spawnParticle(Vec3 position, Vec3 velocity, float lifetime_seconds, float mass,
                     float size, int& out_index) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (lifetime_seconds <= 0.0f) return Result::fail("lifetime_seconds must be positive");
    if (mass <= 0.0f) return Result::fail("mass must be positive");
    RayTrophiSim::ParticleSpawnDesc desc;
    desc.position = position;
    desc.velocity = velocity;
    desc.lifetime_seconds = lifetime_seconds;
    desc.mass = mass;
    desc.start_size = size;
    desc.end_size = size;
    out_index = static_cast<int>(g_ctx->scene.spawnParticle(desc));
    invalidateScriptSimulation();
    return Result::success();
}

Result clearParticles() {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    scriptSimulationRuntime().clear();
    invalidateScriptSimulation();
    return Result::success();
}

Result stepParticleSimulation(float dt) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (dt <= 0.0f) dt = 0.0166667f;
    RayTrophiSim::SimulationContext context =
        g_ctx->scene.simulation_world.makeContext(dt, 0, 1);
    context.dt = dt;
    scriptSimulationRuntime().step(context);
    return Result::success();
}

} // namespace rtapi
