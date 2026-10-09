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
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "ParticleSimulation.h"
#include "Fluid/MatterPhaseGrid.h"
#include "ParticleSystemUsage.h"
#include "ProjectManager.h"
#include "TriangleMesh.h"

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

struct ExecutionPolicyName { RayTrophiSim::ParticleExecutionPolicy mode; const char* name; };
const ExecutionPolicyName kExecutionPolicies[] = {
    { RayTrophiSim::ParticleExecutionPolicy::Auto,        "auto" },
    { RayTrophiSim::ParticleExecutionPolicy::GPURequired, "gpu_required" },
    { RayTrophiSim::ParticleExecutionPolicy::CPU,         "cpu" },
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

// Emitters are addressed the way the panel lists them -- by index -- or by
// name (first hit wins; the runtime does not unique names), or by their stable
// timeline uid spelled "uid:<n>". Only the uid survives removal of an earlier
// emitter and a save/load round trip.
Result resolveEmitterIndex(const RayTrophiSim::ParticleSimulationSystem& runtime,
                           const std::string& index_or_name, std::size_t& out_index) {
    const auto& emitters = runtime.emitters();
    if (emitters.empty()) return Result::fail("no particle emitters in the scene");
    static const std::string kUidPrefix = "uid:";
    if (index_or_name.compare(0, kUidPrefix.size(), kUidPrefix) == 0) {
        const std::string digits = index_or_name.substr(kUidPrefix.size());
        if (digits.empty() ||
            !std::all_of(digits.begin(), digits.end(),
                         [](unsigned char c) { return std::isdigit(c) != 0; })) {
            return Result::fail("malformed particle emitter uid: " + index_or_name);
        }
        const uint64_t uid = std::strtoull(digits.c_str(), nullptr, 10);
        for (std::size_t i = 0; i < emitters.size(); ++i) {
            if (emitters[i].timeline_uid == uid) { out_index = i; return Result::success(); }
        }
        return Result::fail("particle emitter not found: " + index_or_name);
    }
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

Result resolveSystemIndex(const std::string& index_or_name, std::size_t& out_index) {
    const auto& systems = g_ctx->scene.particle_systems;
    if (systems.empty()) {
        return Result::fail("no particle systems in the scene");
    }
    const bool numeric = !index_or_name.empty() &&
        std::all_of(index_or_name.begin(), index_or_name.end(),
                    [](unsigned char c) { return std::isdigit(c) != 0; });
    if (numeric) {
        const long value = std::atol(index_or_name.c_str());
        if (value < 0 || static_cast<std::size_t>(value) >= systems.size()) {
            return Result::fail(
                "particle system index out of range: " + index_or_name);
        }
        out_index = static_cast<std::size_t>(value);
        return Result::success();
    }
    for (std::size_t i = 0; i < systems.size(); ++i) {
        if (systems[i].name == index_or_name) {
            out_index = i;
            return Result::success();
        }
    }
    return Result::fail("particle system not found: " + index_or_name);
}

// The system a call targets. The default ref is the ACTIVE system and creates
// one when the scene has none -- what every pre-Phase-1 script relies on. An
// explicit ref never falls back to the active system: a typo must fail, not
// quietly edit whatever the panel happens to have selected.
Result resolveSystemObject(const ParticleSystemRef& ref, std::size_t& out_index) {
    auto& scene = g_ctx->scene;
    if (ref.isDefault()) {
        scene.ensureActiveParticleSystemObject();
        if (scene.active_particle_system_index < 0 ||
            static_cast<std::size_t>(scene.active_particle_system_index) >=
                scene.particle_systems.size()) {
            return Result::fail("no active particle system");
        }
        out_index = static_cast<std::size_t>(scene.active_particle_system_index);
        return Result::success();
    }
    if (ref.id >= 0) {
        for (std::size_t i = 0; i < scene.particle_systems.size(); ++i) {
            if (scene.particle_systems[i].id == static_cast<uint32_t>(ref.id)) {
                out_index = i;
                return Result::success();
            }
        }
        return Result::fail("particle system not found: id " + std::to_string(ref.id));
    }
    return resolveSystemIndex(ref.index_or_name, out_index);
}

Result resolveRuntime(const ParticleSystemRef& ref,
                      RayTrophiSim::ParticleSimulationSystem*& out_runtime,
                      uint32_t* out_system_id = nullptr) {
    std::size_t index = 0;
    if (Result r = resolveSystemObject(ref, index); !r) return r;
    auto& system = g_ctx->scene.particle_systems[index];
    if (!system.runtime) {
        return Result::fail("particle system has no runtime: " + system.name);
    }
    out_runtime = system.runtime.get();
    if (out_system_id) *out_system_id = system.id;
    return Result::success();
}

ParticleSystemInfo infoFromSystem(const SceneData::ParticleSystemObject& sys, std::size_t index) {
    ParticleSystemInfo info;
    info.index = static_cast<int>(index);
    info.id = sys.id;
    info.name = sys.name;
    info.active = (static_cast<int>(index) == g_ctx->scene.active_particle_system_index);
    info.enabled = sys.enabled;
    info.visible = sys.visible;
    info.emitter_only = sys.render.emitter_only;
    info.render_in_raytrace = sys.render.render_in_raytrace;
    if (sys.runtime) {
        info.domain_count = static_cast<int>(sys.runtime->gridDomains().size());
        info.flow_source_count = static_cast<int>(sys.runtime->flowSources().size());
        info.emitter_count = static_cast<int>(sys.runtime->emitters().size());
        info.collider_count = static_cast<int>(sys.runtime->colliders().size());
        info.appearance_profile_count =
            static_cast<int>(sys.runtime->appearanceProfiles().size());
    }
    return info;
}

// Same test ParticleRenderBridge's gatherSceneMeshSource applies: a flat
// TriangleMesh with that node name and geometry. objectExists() is NOT that
// test -- it also accepts splines and Triangle facades, which the bridge never
// reads, so "resolved" would be true for a source that renders nothing.
bool particleMeshSourceResolves(const std::string& node_name) {
    if (node_name.empty() || g_ctx->scene.isEditorPendingDeleteObjectName(node_name))
        return false;
    for (const auto& obj : g_ctx->scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj);
        if (mesh && mesh->nodeName == node_name && mesh->geometry) return true;
    }
    return false;
}

struct RenderShapeName { SceneData::ParticleRenderShape mode; const char* name; };
const RenderShapeName kRenderShapes[] = {
    { SceneData::ParticleRenderShape::Sphere,      "sphere" },
    { SceneData::ParticleRenderShape::Cube,        "cube" },
    { SceneData::ParticleRenderShape::Tetra,       "tetra" },
    { SceneData::ParticleRenderShape::Quad,        "quad" },
    { SceneData::ParticleRenderShape::SceneMeshes, "scene_meshes" },
};

ParticleEmitterInfo infoFromEmitter(const ParticleEmitterDesc& desc, int index) {
    ParticleEmitterInfo info;
    info.index = index;
    info.uid = desc.timeline_uid;
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
    info.appearance_profile_id = desc.appearance_profile_id;
    info.size_jitter = desc.size_jitter;
    info.angular_velocity = desc.angular_velocity;
    info.angular_jitter = desc.angular_jitter;
    info.seed = desc.seed;
    info.parent_object = desc.parent_object;
    info.velocity_space =
        (desc.velocity_space == RayTrophiSim::SimulationEmissionVelocitySpace::World)
            ? "world" : "local";
    info.inherit_velocity = desc.inherit_velocity;
    info.override_grid_deposit = desc.override_grid_deposit;
    info.grid_density_deposit = desc.grid_density_deposit;
    info.grid_temperature_deposit = desc.grid_temperature_deposit;
    info.grid_fuel_deposit = desc.grid_fuel_deposit;
    return info;
}

// Enums and ranges are validated BEFORE anything is written, so a rejected
// update leaves the emitter exactly as it was instead of half-applied.
// `accumulator` and `burst_consumed` are runtime bookkeeping and are never
// touched here — see the burst note in the file header.
Result applyInfoToEmitter(const ParticleEmitterInfo& info, ParticleEmitterDesc& desc,
                          const RayTrophiSim::ParticleSimulationSystem& runtime) {
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
    // 0 is only meaningful on add (addEmitter then creates a default profile);
    // an update must keep pointing at a real profile of THIS system.
    if (info.appearance_profile_id != 0 &&
        runtime.findAppearanceProfile(info.appearance_profile_id) == nullptr)
        return Result::fail("appearance profile " + std::to_string(info.appearance_profile_id) +
                            " does not exist in this particle system");
    if (info.appearance_profile_id == 0 && desc.appearance_profile_id != 0)
        return Result::fail("appearance_profile_id must name a profile of this system");

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
    desc.appearance_profile_id = info.appearance_profile_id;
    desc.size_jitter = info.size_jitter;
    desc.angular_velocity = info.angular_velocity;
    desc.angular_jitter = info.angular_jitter;
    desc.seed = info.seed;
    desc.parent_object = info.parent_object;
    {
        // Validated before anything is written, like the enums above, so a bad
        // value cannot leave the emitter half-applied.
        std::string space = info.velocity_space;
        std::transform(space.begin(), space.end(), space.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (space == "world") {
            desc.velocity_space = RayTrophiSim::SimulationEmissionVelocitySpace::World;
        } else if (space.empty() || space == "local") {
            desc.velocity_space = RayTrophiSim::SimulationEmissionVelocitySpace::Local;
        } else {
            return Result::fail("unknown emitter velocity_space: " + info.velocity_space);
        }
    }
    desc.inherit_velocity = info.inherit_velocity;
    desc.override_grid_deposit = info.override_grid_deposit;
    desc.grid_density_deposit = std::max(0.0f, info.grid_density_deposit);
    desc.grid_temperature_deposit = std::max(0.0f, info.grid_temperature_deposit);
    desc.grid_fuel_deposit = std::max(0.0f, info.grid_fuel_deposit);
    return Result::success();
}

} // namespace

Result listParticleEmitters(const ParticleSystemRef& system,
                            std::vector<ParticleEmitterInfo>& out) {
    out.clear();
    if (!g_ctx) return notBound();
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    uint32_t system_id = 0;
    if (Result r = resolveRuntime(system, runtime, &system_id); !r) return r;
    const auto& emitters = runtime->emitters();
    out.reserve(emitters.size());
    for (std::size_t i = 0; i < emitters.size(); ++i) {
        out.push_back(infoFromEmitter(emitters[i], static_cast<int>(i)));
        out.back().system_id = system_id;
    }
    return Result::success();
}

std::vector<ParticleEmitterInfo> listParticleEmitters(const ParticleSystemRef& system) {
    std::vector<ParticleEmitterInfo> out;
    (void)listParticleEmitters(system, out);
    return out;
}

Result getParticleEmitter(const std::string& index_or_name, ParticleEmitterInfo& out,
                          const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    uint32_t system_id = 0;
    if (Result r = resolveRuntime(system, runtime, &system_id); !r) return r;
    std::size_t index = 0;
    if (Result r = resolveEmitterIndex(*runtime, index_or_name, index); !r) return r;
    out = infoFromEmitter(runtime->emitters()[index], static_cast<int>(index));
    out.system_id = system_id;
    return Result::success();
}

Result addParticleEmitter(const ParticleEmitterInfo& info, ParticleEmitterInfo& out,
                          const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    // The default ref creates the active system when the scene has none, so
    // runtime->addEmitter() here is what scene.addParticleEmitter() used to do.
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    uint32_t system_id = 0;
    if (Result r = resolveRuntime(system, runtime, &system_id); !r) return r;
    ParticleEmitterDesc desc;
    if (Result r = applyInfoToEmitter(info, desc, *runtime); !r) return r;
    ParticleEmitterDesc& created = runtime->addEmitter(desc);
    out = infoFromEmitter(created, static_cast<int>(runtime->emitters().size()) - 1);
    out.system_id = system_id;
    invalidateScriptSimulation();
    return Result::success();
}

Result removeParticleEmitter(const std::string& index_or_name, const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    if (Result r = resolveRuntime(system, runtime); !r) return r;
    std::size_t index = 0;
    if (Result r = resolveEmitterIndex(*runtime, index_or_name, index); !r) return r;
    if (!runtime->removeEmitter(index))
        return Result::fail("could not remove particle emitter: " + index_or_name);
    invalidateScriptSimulation();
    return Result::success();
}

Result updateParticleEmitter(const std::string& index_or_name, const ParticleEmitterInfo& info,
                             const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    if (Result r = resolveRuntime(system, runtime); !r) return r;
    std::size_t index = 0;
    if (Result r = resolveEmitterIndex(*runtime, index_or_name, index); !r) return r;
    if (Result r = applyInfoToEmitter(info, runtime->emitters()[index], *runtime); !r)
        return r;
    invalidateScriptSimulation();
    return Result::success();
}

Result keyParticleEmitter(const std::string& index_or_name, const ParticleEmitterKey& key,
                          const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    if (Result r = resolveRuntime(system, runtime); !r) return r;
    std::size_t index = 0;
    if (Result r = resolveEmitterIndex(*runtime, index_or_name, index); !r) return r;
    auto& emitter = runtime->emitters()[index];
    // Merge, so several calls can key different channels on one frame.
    auto& stored = emitter.keyframes[key.frame];
    if (key.has_enabled)   { stored.has_enabled = true;   stored.enabled = key.enabled; }
    if (key.has_rate)      { stored.has_rate = true;      stored.rate_per_second = std::max(0.0f, key.rate_per_second); }
    if (key.has_speed)     { stored.has_speed = true;     stored.speed = std::max(0.0f, key.speed); }
    if (key.has_spread)    { stored.has_spread = true;    stored.spread = std::max(0.0f, key.spread); }
    if (key.has_point)     { stored.has_point = true;     stored.point = key.point; }
    if (key.has_direction) { stored.has_direction = true; stored.direction = key.direction; }
    invalidateScriptSimulation();
    return Result::success();
}

Result clearParticleEmitterKey(const std::string& index_or_name, int frame,
                               const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    if (Result r = resolveRuntime(system, runtime); !r) return r;
    std::size_t index = 0;
    if (Result r = resolveEmitterIndex(*runtime, index_or_name, index); !r) return r;
    runtime->emitters()[index].keyframes.erase(frame);
    invalidateScriptSimulation();
    return Result::success();
}

Result clearParticleEmitters(const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    if (Result r = resolveRuntime(system, runtime); !r) return r;
    runtime->clearEmitters();
    invalidateScriptSimulation();
    return Result::success();
}

Result listParticleSystems(std::vector<ParticleSystemInfo>& out) {
    if (!g_ctx) return notBound();
    out.clear();
    for (std::size_t i = 0; i < g_ctx->scene.particle_systems.size(); ++i) {
        out.push_back(infoFromSystem(g_ctx->scene.particle_systems[i], i));
    }
    return Result::success();
}

Result getParticleSystem(const ParticleSystemRef& system, ParticleSystemInfo& out) {
    if (!g_ctx) return notBound();
    // A read must not create a system as a side effect, so the default ref is
    // resolved without ensureActiveParticleSystemObject().
    if (system.isDefault() && !g_ctx->scene.activeParticleSystemObject())
        return Result::fail("no active particle system");
    std::size_t index = 0;
    if (Result r = resolveSystemObject(system, index); !r) return r;
    out = infoFromSystem(g_ctx->scene.particle_systems[index], index);
    return Result::success();
}

Result addParticleSystem(const std::string& name, ParticleSystemInfo& out) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    SceneData::ParticleSystemObject& system =
        g_ctx->scene.addParticleSystemObject(name);
    out = infoFromSystem(system, g_ctx->scene.particle_systems.size() - 1u);
    invalidateScriptSimulation();
    return Result::success();
}

Result updateParticleSystem(const ParticleSystemRef& system, const ParticleSystemInfo& info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    std::size_t index = 0;
    if (Result r = resolveSystemObject(system, index); !r) return r;
    auto& systems = g_ctx->scene.particle_systems;

    // Validate everything before writing anything.
    if (info.name.empty()) return Result::fail("particle system name must not be empty");
    for (std::size_t i = 0; i < systems.size(); ++i) {
        if (i != index && systems[i].name == info.name)
            return Result::fail("particle system name already in use: " + info.name);
    }
    auto& target = systems[index];
    target.name = info.name;
    target.visible = info.visible;
    SceneData::applyParticleSystemEnabledState(target);
    g_ctx->renderer.resetCPUAccumulation();
    ProjectManager::getInstance().markModified();
    invalidateScriptSimulation();
    return Result::success();
}

Result removeParticleSystem(const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (system.isDefault())
        return Result::fail("removeParticleSystem needs an explicit system (id, index or name)");
    std::size_t index = 0;
    if (Result r = resolveSystemObject(system, index); !r) return r;
    if (!g_ctx->scene.removeParticleSystemObject(index))
        return Result::fail("could not remove particle system");
    g_ctx->renderer.resetCPUAccumulation();
    ProjectManager::getInstance().markModified();
    invalidateScriptSimulation();
    return Result::success();
}

Result setActiveParticleSystem(const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (system.isDefault())
        return Result::fail("setActiveParticleSystem needs an explicit system (id, index or name)");
    std::size_t index = 0;
    if (Result r = resolveSystemObject(system, index); !r) return r;
    g_ctx->scene.setActiveParticleSystemObject(index);
    return Result::success();
}

Result getParticleRender(const ParticleSystemRef& system, ParticleRenderInfo& out) {
    if (!g_ctx) return notBound();
    std::size_t index = 0;
    if (Result r = resolveSystemObject(system, index); !r) return r;
    const auto& rs = g_ctx->scene.particle_systems[index].render;
    out = ParticleRenderInfo{};
    out.emitter_only = rs.emitter_only;
    out.render_in_raytrace = rs.render_in_raytrace;
    out.shape = nameOf(kRenderShapes, rs.shape, "sphere");
    out.size_multiplier = rs.size_multiplier;
    out.sphere_subdivisions = rs.sphere_subdivisions;
    out.emissive = rs.emissive;
    out.inherit_color_from_emitter = rs.inherit_color_from_emitter;
    out.base_color = rs.base_color;
    out.emission_strength = rs.emission_strength;
    out.roughness = rs.roughness;
    for (const auto& source : rs.mesh_sources) {
        ParticleRenderMeshSourceInfo entry;
        entry.node_name = source.node_name;
        entry.weight = source.weight;
        entry.resolved = particleMeshSourceResolves(source.node_name);
        out.mesh_sources.push_back(std::move(entry));
    }
    return Result::success();
}

Result updateParticleRender(const ParticleSystemRef& system, const ParticleRenderInfo& info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    std::size_t index = 0;
    if (Result r = resolveSystemObject(system, index); !r) return r;

    SceneData::ParticleRenderShape shape = SceneData::ParticleRenderShape::Sphere;
    if (!parseMode(kRenderShapes, info.shape, shape))
        return Result::fail("unknown particle render shape: " + info.shape +
                            " (" + optionList(kRenderShapes) + ")");
    if (!(info.size_multiplier > 0.0f))
        return Result::fail("size_multiplier must be positive");
    if (info.sphere_subdivisions < 0 || info.sphere_subdivisions > 3)
        return Result::fail("sphere_subdivisions must be in 0..3");
    if (info.emission_strength < 0.0f)
        return Result::fail("emission_strength must be >= 0");
    if (info.roughness < 0.0f || info.roughness > 1.0f)
        return Result::fail("roughness must be in 0..1");
    std::vector<SceneData::ParticleRenderMeshSource> sources;
    for (const auto& entry : info.mesh_sources) {
        if (entry.node_name.empty())
            return Result::fail("mesh source node_name must not be empty");
        if (!(entry.weight >= 0.0f))
            return Result::fail("mesh source weight must be >= 0: " + entry.node_name);
        // Writing a dangling reference is refused; one that dangles LATER is
        // reported by getParticleRender as resolved = false.
        if (!particleMeshSourceResolves(entry.node_name))
            return Result::fail("mesh source object not found: " + entry.node_name);
        for (const auto& existing : sources) {
            if (existing.node_name == entry.node_name)
                return Result::fail("mesh source listed twice: " + entry.node_name);
        }
        SceneData::ParticleRenderMeshSource source;
        source.node_name = entry.node_name;
        source.weight = entry.weight;
        sources.push_back(std::move(source));
    }

    auto& rs = g_ctx->scene.particle_systems[index].render;
    rs.emitter_only = info.emitter_only;
    rs.render_in_raytrace = info.render_in_raytrace;
    rs.shape = shape;
    rs.size_multiplier = info.size_multiplier;
    rs.sphere_subdivisions = info.sphere_subdivisions;
    rs.emissive = info.emissive;
    rs.inherit_color_from_emitter = info.inherit_color_from_emitter;
    rs.base_color = info.base_color;
    rs.emission_strength = info.emission_strength;
    rs.roughness = info.roughness;
    rs.mesh_sources = std::move(sources);
    g_ctx->renderer.resetCPUAccumulation();
    ProjectManager::getInstance().markModified();
    return Result::success();
}

// Slug -> authored preset recipe. The slugs are the panel's list in lowercase
// snake_case; keeping them as data rather than an if-chain means adding a
// preset is one line here and the error message below stays complete.
namespace {
struct ParticlePresetSlug {
    const char* slug;
    SceneData::ParticleSystemPreset preset;
};
constexpr ParticlePresetSlug kParticlePresetSlugs[] = {
    {"campfire",            SceneData::ParticleSystemPreset::Campfire},
    {"explosion",           SceneData::ParticleSystemPreset::Explosion},
    {"smoke",               SceneData::ParticleSystemPreset::Smoke},
    {"ground_burst",        SceneData::ParticleSystemPreset::GroundBurst},
    {"fireball",            SceneData::ParticleSystemPreset::Fireball},
    {"flamethrower",        SceneData::ParticleSystemPreset::Flamethrower},
    {"burning_fuel_spill",  SceneData::ParticleSystemPreset::BurningFuelSpill},
    {"ignited_fuel_jet",    SceneData::ParticleSystemPreset::IgnitedFuelJet},
    {"nuclear",             SceneData::ParticleSystemPreset::Nuclear},
};
}  // namespace

Result addParticleSystemPreset(const std::string& preset, ParticleSystemInfo& out) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    std::string slug = preset;
    std::transform(slug.begin(), slug.end(), slug.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    // Accept the panel's spacing/hyphen spellings so a get -> set round trip and
    // an obvious guess both land, rather than failing on punctuation.
    std::replace(slug.begin(), slug.end(), ' ', '_');
    std::replace(slug.begin(), slug.end(), '-', '_');

    const ParticlePresetSlug* match = nullptr;
    for (const auto& entry : kParticlePresetSlugs) {
        if (slug == entry.slug) { match = &entry; break; }
    }
    if (!match) {
        std::string known;
        for (const auto& entry : kParticlePresetSlugs) {
            if (!known.empty()) known += ", ";
            known += entry.slug;
        }
        return Result::fail("unknown particle preset '" + preset + "'; known: " + known);
    }

    const std::size_t before = g_ctx->scene.particle_systems.size();
    SceneData::ParticleSystemObject& sys =
        g_ctx->scene.addParticleSystemPreset(match->preset);
    if (g_ctx->scene.particle_systems.size() != before + 1u) {
        // The scene logs this too, but a script must FAIL here rather than hand
        // back a plausible-looking info block for a system that was not created.
        return Result::fail("particle preset '" + slug + "' did not create a system");
    }

    out = infoFromSystem(sys, g_ctx->scene.particle_systems.size() - 1u);
    invalidateScriptSimulation();
    return Result::success();
}

Result setParticleSystemEmitterOnly(const std::string& index_or_name,
                                    bool emitter_only) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    std::size_t index = 0;
    if (Result result = resolveSystemIndex(index_or_name, index); !result) {
        return result;
    }
    if (!ParticleSystemUsage::setEmitterOnly(g_ctx->scene, index, emitter_only)) {
        return Result::fail("could not update particle system: " + index_or_name);
    }
    g_ctx->renderer.resetCPUAccumulation();
    return Result::success();
}

// Wipes every system scene-wide; the next add re-creates a clean one. Particle
// calls can target one system (ParticleSystemRef, removeParticleSystem), but
// flow sources, colliders and fluid domains in RtApiFluid.cpp still reach only
// the ACTIVE system through scriptSimulationRuntime(), so this remains the one
// scripted way to clear what those left in other systems.
Result clearParticleSystems() {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    g_ctx->scene.clearParticleSystemObjects();
    invalidateScriptSimulation();
    return Result::success();
}

// ── Appearance profiles ─────────────────────────────────────────────────────

namespace {

ParticleAppearanceInfo infoFromAppearance(const RayTrophiSim::ParticleAppearanceProfile& p,
                                          const RayTrophiSim::ParticleSimulationSystem& runtime,
                                          uint32_t system_id) {
    ParticleAppearanceInfo info;
    info.id = p.id;
    info.system_id = system_id;
    info.name = p.name;
    info.blend = RayTrophiSim::particleAppearanceBlendName(p.blend);
    for (const auto& stop : p.color_ramp) info.color_ramp.push_back({stop.t, stop.color});
    for (const auto& k : p.opacity_curve) info.opacity_curve.push_back({k.t, k.value});
    for (const auto& k : p.size_curve) info.size_curve.push_back({k.t, k.value});
    for (const auto& k : p.emission_curve) info.emission_curve.push_back({k.t, k.value});
    for (const auto& emitter : runtime.emitters()) {
        if (emitter.appearance_profile_id == p.id) {
            info.used_by_emitter_uids.push_back(emitter.timeline_uid);
        }
    }
    return info;
}

Result appearanceFromInfo(const ParticleAppearanceInfo& info,
                          RayTrophiSim::ParticleAppearanceProfile& out) {
    out.name = info.name;
    if (!RayTrophiSim::parseParticleAppearanceBlend(canonical(info.blend), out.blend))
        return Result::fail("unknown appearance blend: " + info.blend + " (additive|alpha)");
    out.color_ramp.clear();
    for (const auto& stop : info.color_ramp) out.color_ramp.push_back({stop.t, stop.color});
    auto copyCurve = [](const std::vector<ParticleCurveKeyInfo>& in,
                        std::vector<RayTrophiSim::ParticleCurveKey>& dst) {
        dst.clear();
        for (const auto& k : in) dst.push_back({k.t, k.value});
    };
    copyCurve(info.opacity_curve, out.opacity_curve);
    copyCurve(info.size_curve, out.size_curve);
    copyCurve(info.emission_curve, out.emission_curve);
    // Validate here too, so the error text is identical for IPC and Python and
    // nothing is written when it fails.
    RayTrophiSim::ParticleAppearanceProfile check = out;
    const std::string problem = RayTrophiSim::normalizeParticleAppearanceProfile(check);
    if (!problem.empty()) return Result::fail(problem);
    return Result::success();
}

// Appearance is read at draw time, so a look edit does NOT invalidate the
// simulation cache (dragging a colour must not throw the sim away every
// frame). The one exception is `opacity_changed`: the particle -> gas deposit
// is weighted by the opacity curve, so cached gas frames would be stale.
void appearanceChanged(bool opacity_changed) {
    g_ctx->renderer.resetCPUAccumulation();
    ProjectManager::getInstance().markModified();
    if (opacity_changed) invalidateScriptSimulation();
}

bool sameCurve(const std::vector<RayTrophiSim::ParticleCurveKey>& a,
               const std::vector<RayTrophiSim::ParticleCurveKey>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].t != b[i].t || a[i].value != b[i].value) return false;
    }
    return true;
}

} // namespace

Result listParticleAppearances(const ParticleSystemRef& system,
                               std::vector<ParticleAppearanceInfo>& out) {
    out.clear();
    if (!g_ctx) return notBound();
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    uint32_t system_id = 0;
    if (Result r = resolveRuntime(system, runtime, &system_id); !r) return r;
    for (const auto& profile : runtime->appearanceProfiles()) {
        out.push_back(infoFromAppearance(profile, *runtime, system_id));
    }
    return Result::success();
}

Result getParticleAppearance(uint32_t profile_id, ParticleAppearanceInfo& out,
                             const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    uint32_t system_id = 0;
    if (Result r = resolveRuntime(system, runtime, &system_id); !r) return r;
    const auto* profile = runtime->findAppearanceProfile(profile_id);
    if (!profile)
        return Result::fail("appearance profile " + std::to_string(profile_id) +
                            " does not exist in this particle system");
    out = infoFromAppearance(*profile, *runtime, system_id);
    return Result::success();
}

Result addParticleAppearance(const ParticleAppearanceInfo& info, ParticleAppearanceInfo& out,
                             const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    uint32_t system_id = 0;
    if (Result r = resolveRuntime(system, runtime, &system_id); !r) return r;
    RayTrophiSim::ParticleAppearanceProfile profile;
    if (Result r = appearanceFromInfo(info, profile); !r) return r;
    profile.id = 0;  // always a fresh id
    std::string error;
    const uint32_t id = runtime->addAppearanceProfile(profile, &error);
    if (id == 0) return Result::fail(error);
    out = infoFromAppearance(*runtime->findAppearanceProfile(id), *runtime, system_id);
    appearanceChanged(false);  // a new profile is referenced by nothing yet
    return Result::success();
}

Result updateParticleAppearance(uint32_t profile_id, const ParticleAppearanceInfo& info,
                                const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    if (Result r = resolveRuntime(system, runtime); !r) return r;
    const auto* existing = runtime->findAppearanceProfile(profile_id);
    if (!existing)
        return Result::fail("appearance profile " + std::to_string(profile_id) +
                            " does not exist in this particle system");
    RayTrophiSim::ParticleAppearanceProfile profile;
    if (Result r = appearanceFromInfo(info, profile); !r) return r;
    profile.id = profile_id;
    // Compare in the stored (sorted) form, so reordering keys is not a change.
    RayTrophiSim::ParticleAppearanceProfile normalized = profile;
    (void)RayTrophiSim::normalizeParticleAppearanceProfile(normalized);
    const bool opacity_changed = !sameCurve(existing->opacity_curve, normalized.opacity_curve);
    std::string error;
    if (!runtime->updateAppearanceProfile(profile, &error)) return Result::fail(error);
    appearanceChanged(opacity_changed);
    return Result::success();
}

Result removeParticleAppearance(uint32_t profile_id, const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    if (Result r = resolveRuntime(system, runtime); !r) return r;
    std::string error;
    if (!runtime->removeAppearanceProfile(profile_id, &error)) return Result::fail(error);
    appearanceChanged(false);  // only unreferenced profiles can be removed
    return Result::success();
}

Result getParticlePhysics(ParticlePhysicsInfo& out, const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    if (Result r = resolveRuntime(system, runtime); !r) return r;
    const ParticlePhysicsSettings& s = runtime->physicsSettings();
    out.mode = nameOf(kPhysicsModes, s.mode, "spark");
    out.quality = nameOf(kQualities, s.quality, "realtime");
    out.execution_policy = nameOf(kExecutionPolicies, s.execution_policy, "auto");
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
    out.inherit_atmosphere = s.inherit_atmosphere;
    out.effective_air_wind = s.inherit_atmosphere ? atmosphere::ambientWindMps()
                                                  : Vec3(0.0f, 0.0f, 0.0f);
    return Result::success();
}

Result updateParticlePhysics(const ParticlePhysicsInfo& info, const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    if (Result r = resolveRuntime(system, runtime); !r) return r;
    ParticlePhysicsSettings& s = runtime->physicsSettings();

    ParticlePhysicsMode mode = s.mode;
    if (!info.mode.empty() && !parseMode(kPhysicsModes, info.mode, mode))
        return Result::fail("unknown particle physics mode: " + info.mode +
                            " (" + optionList(kPhysicsModes) + ")");
    ParticleQualityMode quality = s.quality;
    if (!info.quality.empty() && !parseMode(kQualities, info.quality, quality))
        return Result::fail("unknown particle quality mode: " + info.quality +
                            " (" + optionList(kQualities) + ")");
    RayTrophiSim::ParticleExecutionPolicy policy = s.execution_policy;
    if (!info.execution_policy.empty() &&
        !parseMode(kExecutionPolicies, info.execution_policy, policy))
        return Result::fail("unknown particle execution_policy: " + info.execution_policy +
                            " (" + optionList(kExecutionPolicies) + ")");
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
    s.execution_policy = policy;
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
    s.inherit_atmosphere = info.inherit_atmosphere;
    s.grid_deposit_fade_with_age = info.grid_deposit_fade_with_age;
    invalidateScriptSimulation();
    return Result::success();
}

Result getParticleStats(ParticleStatsInfo& out, const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    RayTrophiSim::ParticleSimulationSystem* runtime_ptr = nullptr;
    uint32_t system_id = 0;
    if (Result r = resolveRuntime(system, runtime_ptr, &system_id); !r) return r;
    auto& runtime = *runtime_ptr;
    out.system_id = system_id;
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

    // Stage backends, written out explicitly so a script can assert on them.
    // A resident step runs forces + integration on the device; emission
    // (spawn decisions) stays on the CPU, and a resident system has no
    // collision stages by construction (they are what make it ineligible).
    out.execution_policy = nameOf(kExecutionPolicies, stats.execution_policy, "auto");
    out.compute_backend = stats.compute_backend ? stats.compute_backend : "none";
    out.gpu_status = stats.gpu_status ? stats.gpu_status : "not_attempted";
    out.device_resident = stats.device_resident;
    out.step_blocked = stats.step_blocked;
    bool any_collider = false;
    for (const auto& collider : runtime.colliders()) any_collider = any_collider || collider.enabled;
    out.emit_backend = "cpu";
    out.forces_backend = stats.device_resident ? "gpu" : "cpu";
    out.integrate_backend = stats.device_resident ? "gpu" : "cpu";
    out.scene_collision_backend = any_collider ? "cpu" : "off";
    out.self_collision_backend =
        runtime.physicsSettings().self_collision_enabled ? "cpu" : "off";
    if (stats.step_blocked) {
        out.forces_backend = "blocked";
        out.integrate_backend = "blocked";
        out.scene_collision_backend = "blocked";
        out.self_collision_backend = "blocked";
    }
    out.gpu_step_ms = stats.gpu_step_ms;
    out.residency = RayTrophiSim::particleKinematicResidencyName(runtime.kinematicResidency());
    out.resident_capacity = static_cast<int>(runtime.residentCapacity());
    out.slot_records = stats.slot_records;
    out.step_upload_bytes = stats.step_transfer.upload_bytes;
    out.step_download_bytes = stats.step_transfer.download_bytes;
    out.step_dispatch_calls = stats.step_transfer.dispatch_calls;
    out.step_synchronize_calls = stats.step_transfer.synchronize_calls;
    out.step_upload_call_ms = stats.step_transfer.upload_call_ms;
    out.step_download_call_ms = stats.step_transfer.download_call_ms;
    out.step_synchronize_ms = stats.step_transfer.synchronize_ms;
    const auto& snapshots = runtime.hostSnapshotStats();
    out.snapshot_sync_count = snapshots.synchronous_count;
    out.snapshot_mirrored_steps = snapshots.mirrored_steps;
    out.snapshot_download_bytes = snapshots.download_bytes;
    out.snapshot_last_reason = snapshots.last_reason ? snapshots.last_reason : "none";
    out.nonfinite_particles = stats.nonfinite_particles;
    out.nonfinite_measured = stats.nonfinite_measured;
    out.grid_deposit_landed = stats.grid_deposit_landed;
    out.grid_deposit_dropped_no_domain = stats.grid_deposit_dropped_no_domain;
    out.grid_deposit_dropped_no_channel = stats.grid_deposit_dropped_no_channel;
    return Result::success();
}

Result getParticleStateSample(int max_count, int offset, int stride,
                              ParticleStateSample& out, const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (max_count < 0) return Result::fail("max_count must be >= 0");
    if (offset < 0) return Result::fail("offset must be >= 0");
    if (stride < 1) return Result::fail("stride must be at least 1");
    max_count = std::min(max_count, 4096);

    RayTrophiSim::ParticleSimulationSystem* runtime_ptr = nullptr;
    if (Result r = resolveRuntime(system, runtime_ptr); !r) return r;
    // Device-resident kinematics come home first: an explicit, counted
    // snapshot, never a silent read of stale host positions.
    if (!runtime_ptr->syncHostState("state_sample")) {
        return Result::fail("particle state is device-resident and could not be read back "
                            "(see the console)");
    }
    const auto& runtime = *runtime_ptr;
    const RayTrophiSim::ParticleSoABuffers& b = runtime.buffers();
    out = ParticleStateSample{};
    out.capacity = static_cast<int>(b.alive.size());
    out.indices.reserve(static_cast<std::size_t>(max_count));
    out.positions.reserve(static_cast<std::size_t>(max_count));
    out.velocities.reserve(static_cast<std::size_t>(max_count));
    out.ages.reserve(static_cast<std::size_t>(max_count));

    double sum_p[3] = {0.0, 0.0, 0.0};
    double sum_v[3] = {0.0, 0.0, 0.0};
    int finite_count = 0;
    int alive_ordinal = 0;
    bool have_bounds = false;
    for (std::size_t i = 0; i < b.alive.size(); ++i) {
        if (b.alive[i] == 0u) continue;
        const Vec3 p(b.position_x[i], b.position_y[i], b.position_z[i]);
        const Vec3 v(b.velocity_x[i], b.velocity_y[i], b.velocity_z[i]);
        ++out.alive_count;
        const bool finite = std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
                            std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        if (!finite) {
            ++out.nonfinite;
        } else {
            // Aggregates skip non-finite particles so one NaN does not erase
            // the whole population's centroid; `nonfinite` reports them.
            ++finite_count;
            sum_p[0] += p.x; sum_p[1] += p.y; sum_p[2] += p.z;
            sum_v[0] += v.x; sum_v[1] += v.y; sum_v[2] += v.z;
            if (!have_bounds) {
                out.bounds_min = p;
                out.bounds_max = p;
                have_bounds = true;
            } else {
                out.bounds_min = Vec3::min(out.bounds_min, p);
                out.bounds_max = Vec3::max(out.bounds_max, p);
            }
        }
        const int ordinal = alive_ordinal++;
        if (ordinal < offset || ((ordinal - offset) % stride) != 0) continue;
        if (out.returned >= max_count) continue;
        out.indices.push_back(static_cast<int>(i));
        out.positions.push_back(p);
        out.velocities.push_back(v);
        out.ages.push_back(b.age_seconds[i]);
        ++out.returned;
    }
    if (finite_count > 0) {
        const double inv = 1.0 / static_cast<double>(finite_count);
        out.centroid = Vec3(static_cast<float>(sum_p[0] * inv),
                            static_cast<float>(sum_p[1] * inv),
                            static_cast<float>(sum_p[2] * inv));
        out.mean_velocity = Vec3(static_cast<float>(sum_v[0] * inv),
                                 static_cast<float>(sum_v[1] * inv),
                                 static_cast<float>(sum_v[2] * inv));
    }
    return Result::success();
}

Result getFluidStepStats(const std::string& domain_id_or_name, FluidStepStats& out) {
    if (!g_ctx) return notBound();
    auto& runtime = scriptSimulationRuntime();
    const auto& domains = runtime.gridDomains();
    const auto& states = runtime.gridDomainStates();

    std::size_t index = domains.size();
    for (std::size_t i = 0; i < domains.size(); ++i) {
        if (domains[i].name == domain_id_or_name) {
            index = i;
            break;
        }
    }
    if (index >= domains.size()) {
        char* end = nullptr;
        const long parsed = std::strtol(domain_id_or_name.c_str(), &end, 10);
        if (end && *end == '\0' && parsed >= 0 &&
            static_cast<std::size_t>(parsed) < domains.size()) {
            index = static_cast<std::size_t>(parsed);
        } else {
            return Result::fail("grid domain not found: " + domain_id_or_name);
        }
    }
    if (!RayTrophiSim::simulationDomainHasLiquid(domains[index].type)) {
        return Result::fail("domain is not fluid: " + domain_id_or_name);
    }
    if (index >= states.size()) return Result::fail("domain has no live state yet");

    const auto& state = states[index];
    const auto& transfer = state.fluid_transfer_stats;
    out = FluidStepStats{};
    out.measured = state.valid && transfer.measured;
    if (!out.measured) return Result::success();

    const auto& fluid = state.fluid_stats;
    out.normalize_window_used = fluid.normalize_window_used;
    out.normalize_window_cells = fluid.normalize_window_used ? fluid.normalize_window_cells : 0;
    out.full_grid_cells = fluid.grid_cell_count;
    out.pressure_window_used = fluid.pressure_on_gpu && fluid.pressure_window_used;
    out.pressure_window_cells = out.pressure_window_used ? fluid.pressure_window_cells : 0;
    out.transfer_sparse_used = fluid.p2g_on_gpu && fluid.transfer_sparse_used;
    out.flip_sparse_used = out.transfer_sparse_used && fluid.flip_sparse_used;
    out.transfer_sparse_active_tiles = out.transfer_sparse_used
        ? fluid.transfer_sparse_active_tiles : 0;
    out.transfer_sparse_allocated_tiles = out.transfer_sparse_used
        ? fluid.transfer_sparse_allocated_tiles : 0;
    out.transfer_sparse_resident_bytes = out.transfer_sparse_used
        ? fluid.transfer_sparse_resident_bytes : 0;
    out.transfer_sparse_status = fluid.transfer_sparse_status;
    out.transfer_sparse_canonical = out.transfer_sparse_used && fluid.transfer_sparse_canonical;
    out.transfer_sparse_blocked = fluid.transfer_sparse_blocked;
    out.pressure_sparse_used = fluid.pressure_on_gpu && fluid.pressure_sparse_used;
    out.pressure_sparse_active_tiles = out.pressure_sparse_used
        ? fluid.pressure_sparse_active_tiles : 0;
    out.pressure_sparse_allocated_tiles = out.pressure_sparse_used
        ? fluid.pressure_sparse_allocated_tiles : 0;
    out.pressure_sparse_resident_bytes = out.pressure_sparse_used
        ? fluid.pressure_sparse_resident_bytes : 0;
    out.occupancy_on_gpu = fluid.occupancy_on_gpu;
    out.viscosity_sparse_used = fluid.viscosity_on_gpu && fluid.viscosity_sparse_used;
    out.viscosity_sparse_active_tiles = out.viscosity_sparse_used
        ? fluid.viscosity_sparse_active_tiles : 0;
    out.viscosity_sparse_allocated_tiles = out.viscosity_sparse_used
        ? fluid.viscosity_sparse_allocated_tiles : 0;
    out.viscosity_sparse_resident_bytes = out.viscosity_sparse_used
        ? fluid.viscosity_sparse_resident_bytes : 0;
    out.resolution[0] = RayTrophiSim::Fluid::liquidGrid(state).nx;
    out.resolution[1] = RayTrophiSim::Fluid::liquidGrid(state).ny;
    out.resolution[2] = RayTrophiSim::Fluid::liquidGrid(state).nz;
    out.particle_count = state.particles.size();
    out.gpu_status = fluid.gpu_status;
    out.p2g_on_gpu = fluid.p2g_on_gpu;
    out.pressure_on_gpu = fluid.pressure_on_gpu;
    out.g2p_on_gpu = fluid.g2p_on_gpu;
    out.density_on_gpu = fluid.density_on_gpu;
    out.total_ms = fluid.total_ms;
    out.p2g_ms = fluid.p2g_ms;
    out.pressure_ms = fluid.pressure_ms;
    out.g2p_ms = fluid.g2p_ms;
    out.advect_ms = fluid.advect_ms;
    out.density_ms = fluid.density_ms;
    out.upload_bytes = transfer.upload_bytes;
    out.download_bytes = transfer.download_bytes;
    out.upload_calls = transfer.upload_calls;
    out.download_calls = transfer.download_calls;
    out.dispatch_calls = transfer.dispatch_calls;
    out.batch_end_calls = transfer.batch_end_calls;
    out.synchronize_calls = transfer.synchronize_calls;
    out.upload_call_ms = transfer.upload_call_ms;
    out.download_call_ms = transfer.download_call_ms;
    out.dispatch_call_ms = transfer.dispatch_call_ms;
    out.batch_end_ms = transfer.batch_end_ms;
    out.synchronize_ms = transfer.synchronize_ms;
    return Result::success();
}

Result getGasStepStats(const std::string& domain_id_or_name, GasStepStats& out) {
    if (!g_ctx) return notBound();
    auto& runtime = scriptSimulationRuntime();
    const auto& domains = runtime.gridDomains();
    const auto& states = runtime.gridDomainStates();

    std::size_t index = domains.size();
    for (std::size_t i = 0; i < domains.size(); ++i) {
        if (domains[i].name == domain_id_or_name) { index = i; break; }
    }
    if (index >= domains.size()) {
        // Fall back to a plain index so a script can drive an unnamed domain.
        char* end = nullptr;
        const long parsed = std::strtol(domain_id_or_name.c_str(), &end, 10);
        if (end && *end == '\0' && parsed >= 0 &&
            static_cast<std::size_t>(parsed) < domains.size()) {
            index = static_cast<std::size_t>(parsed);
        } else {
            return Result::fail("grid domain not found: " + domain_id_or_name);
        }
    }
    if (index >= states.size()) return Result::fail("domain has no live state yet");

    const auto& state = states[index];
    const auto& gs = state.gas_stats;
    // ★ `measured` is load-bearing and must be read before any number below.
    // Every timing is 0.0 both for "this stage is free" and for "no step ran",
    // and a caller that skips this flag records an idle domain as a 0 ms step —
    // which in an optimisation A/B reads as a total win.
    out.measured = state.valid && gs.stepped;
    if (!out.measured) return Result::success();

    out.resolution[0] = state.resolution_x;
    out.resolution[1] = state.resolution_y;
    out.resolution[2] = state.resolution_z;
    out.cell_count = gs.cell_count;
    out.active_blocks = gs.active_blocks;
    out.total_blocks = gs.total_blocks;
    out.active_density_cells = gs.active_density_cells;
    out.grid_memory_bytes = gs.grid_memory_bytes;

    out.total_ms = gs.total_ms;
    out.voxelize_ms = gs.voxelize_ms;
    out.inventory_advection_ms = gs.inventory_advection_ms;
    out.analysis_ms = gs.analysis_ms;

    out.gpu_collider_source_ms = gs.gpu_collider_source_ms;
    out.gpu_msf_ms = gs.gpu_msf_ms;
    out.gpu_source_upload_ms = gs.gpu_source_upload_ms;
    out.gpu_fluid_combustion_ms = gs.gpu_fluid_combustion_ms;
    out.gpu_velocity_advect_ms = gs.gpu_velocity_advect_ms;
    out.gpu_scalar_advect_ms = gs.gpu_scalar_advect_ms;
    out.gpu_combustion_ms = gs.gpu_combustion_ms;
    out.gpu_body_forces_ms = gs.gpu_body_forces_ms;
    out.gpu_dissipation_ms = gs.gpu_dissipation_ms;
    out.gpu_pressure_ms = gs.gpu_pressure_ms;
    out.gpu_publish_ms = gs.gpu_publish_ms;
    out.gpu_majorant_ms = gs.gpu_majorant_ms;
    out.gpu_host_sync_ms = gs.gpu_host_sync_ms;
    out.gpu_surface_dust_ms = gs.gpu_surface_dust_ms;
    out.gpu_scalar_dissipate_ms = gs.gpu_scalar_dissipate_ms;

    out.cpu_total_ms = gs.cpu.total_ms;
    out.cpu_advect_velocity_ms = gs.cpu.advect_velocity_ms;
    out.cpu_advect_scalar_ms = gs.cpu.advect_scalar_ms;
    out.cpu_boundary_ms = gs.cpu.boundary_ms;
    out.cpu_combustion_ms = gs.cpu.combustion_ms;
    out.cpu_surface_dust_ms = gs.cpu.surface_dust_ms;
    out.cpu_buoyancy_ms = gs.cpu.buoyancy_ms;
    out.cpu_force_fields_ms = gs.cpu.force_fields_ms;
    out.cpu_vorticity_ms = gs.cpu.vorticity_ms;
    out.cpu_turbulence_ms = gs.cpu.turbulence_ms;
    out.cpu_dissipation_ms = gs.cpu.dissipation_ms;
    out.cpu_pressure_ms = gs.cpu.pressure_ms;

    out.max_density = gs.max_density;
    out.max_temperature = gs.max_temperature;
    out.max_speed = gs.max_speed;
    out.cfl = gs.cfl;
    out.burning_cells = gs.burning_cells;
    out.solid_cells = gs.solid_cells;
    out.liquid_boundary_cells = gs.liquid_boundary_cells;
    out.liquid_boundary_mean_velocity = gs.liquid_boundary_mean_velocity;
    return Result::success();
}

Result spawnParticle(Vec3 position, Vec3 velocity, float lifetime_seconds, float mass,
                     float size, int& out_index, const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (lifetime_seconds <= 0.0f) return Result::fail("lifetime_seconds must be positive");
    if (mass <= 0.0f) return Result::fail("mass must be positive");
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    if (Result r = resolveRuntime(system, runtime); !r) return r;
    RayTrophiSim::ParticleSpawnDesc desc;
    desc.position = position;
    desc.velocity = velocity;
    desc.lifetime_seconds = lifetime_seconds;
    desc.mass = mass;
    // No profile: the fallback look (white, opacity 1 -> 0, unit size curve),
    // so size_scale is the width in metres — what `size` always meant here.
    desc.size_scale = size;
    out_index = static_cast<int>(runtime->spawn(desc));
    invalidateScriptSimulation();
    return Result::success();
}

Result clearParticles(const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    if (Result r = resolveRuntime(system, runtime); !r) return r;
    runtime->clear();
    invalidateScriptSimulation();
    return Result::success();
}

Result stepParticleSimulation(float dt, const ParticleSystemRef& system) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (dt <= 0.0f) dt = 0.0166667f;
    RayTrophiSim::ParticleSimulationSystem* runtime = nullptr;
    if (Result r = resolveRuntime(system, runtime); !r) return r;
    // Select the compute backend the current policies ask for. Without this a
    // scripted step ran on whatever the last frame-loop sync left behind, so
    // a policy change made over IPC was never reflected in the backend.
    g_ctx->scene.syncSimulationWorld();
    RayTrophiSim::SimulationContext context =
        g_ctx->scene.simulation_world.makeContext(dt, 0, 1);
    context.dt = dt;
    // Same frame boundary as SimulationWorld::executeStep: the Vulkan backend
    // submits and fences in endFrame(), so a scripted step must not leave
    // recorded dispatches open for a later cache restore to invalidate.
    auto& compute = g_ctx->scene.simulation_world.compute();
    compute.beginFrame(0);
    runtime->step(context);
    compute.endFrame();
    return Result::success();
}

} // namespace rtapi
