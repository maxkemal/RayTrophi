/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtPythonParticle.cpp
* License:       MIT
* =========================================================================
*
* rt.particle — emitters, systems, render settings, solver settings and live
* statistics. Split out of RtPython.cpp for particle roadmap Phase 1.
*
* Every per-system call takes two optional keywords:
*   system_id=<int>          the system's stable id (list_systems()["id"])
*   system=<str|int>         panel index or name
* Neither given = the ACTIVE system (the particle panel's selection), which is
* what every call did before Phase 1. An explicit reference never falls back to
* the active system; it raises.
*
* Particle colliders and grid domains live on the same runtime and are scripted
* under rt.fluid / rt.collider, not here.
*/

#include "RtPythonParticle.h"

#include "RtPyCommon.h"

#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

rtapi::ParticleSystemRef systemRef(const py::object& system, int system_id) {
    rtapi::ParticleSystemRef ref;
    ref.id = system_id;
    if (!system.is_none()) {
        if (py::isinstance<py::int_>(system))
            ref.index_or_name = std::to_string(py::cast<long long>(system));
        else
            ref.index_or_name = py::cast<std::string>(system);
    }
    return ref;
}

// For the **kwargs entry points: the addressing keys travel in the same dict
// as the fields being patched.
rtapi::ParticleSystemRef systemRefFromKwargs(const py::kwargs& kwargs) {
    py::object system = py::none();
    int system_id = -1;
    if (kwargs.contains("system")) system = kwargs["system"];
    if (kwargs.contains("system_id")) system_id = py::cast<int>(kwargs["system_id"]);
    return systemRef(system, system_id);
}

py::dict emitterToDict(const rtapi::ParticleEmitterInfo& info) {
    py::dict d;
    d["index"] = info.index;
    d["uid"] = info.uid;
    d["system_id"] = info.system_id;
    d["name"] = info.name;
    d["source_mode"] = info.source_mode;
    d["spawn_mode"] = info.spawn_mode;
    d["source_name"] = info.source_name;
    d["enabled"] = info.enabled;
    d["point"] = vec3ToPython(info.point);
    d["local_offset"] = vec3ToPython(info.local_offset);
    d["direction"] = vec3ToPython(info.direction);
    d["surface_offset"] = info.surface_offset;
    d["rate_per_second"] = info.rate_per_second;
    d["burst_count"] = info.burst_count;
    d["speed"] = info.speed;
    d["spread"] = info.spread;
    d["lifetime_seconds"] = info.lifetime_seconds;
    d["mass"] = info.mass;
    d["appearance_profile_id"] = info.appearance_profile_id;
    d["size_jitter"] = info.size_jitter;
    d["angular_velocity"] = info.angular_velocity;
    d["angular_jitter"] = info.angular_jitter;
    d["seed"] = info.seed;
    d["parent_object"] = info.parent_object;
    d["velocity_space"] = info.velocity_space;
    d["inherit_velocity"] = info.inherit_velocity;
    d["override_grid_deposit"] = info.override_grid_deposit;
    d["grid_density_deposit"] = info.grid_density_deposit;
    d["grid_temperature_deposit"] = info.grid_temperature_deposit;
    d["grid_fuel_deposit"] = info.grid_fuel_deposit;
    return d;
}

// Shared by add_emitter (patches a default desc) and set_emitter (patches the
// live one), so both accept exactly the same keyword set.
void patchEmitter(const py::kwargs& kwargs, rtapi::ParticleEmitterInfo& info) {
    auto str = [&](const char* key, std::string& target) {
        if (kwargs.contains(key)) target = py::cast<std::string>(kwargs[key]);
    };
    auto flt = [&](const char* key, float& target) {
        if (kwargs.contains(key)) target = py::cast<float>(kwargs[key]);
    };
    auto boolean = [&](const char* key, bool& target) {
        if (kwargs.contains(key)) target = py::cast<bool>(kwargs[key]);
    };
    auto vector = [&](const char* key, Vec3& target) {
        if (kwargs.contains(key)) target = vec3FromPython(kwargs[key]);
    };
    str("name", info.name); str("source_mode", info.source_mode);
    str("spawn_mode", info.spawn_mode); str("source_name", info.source_name);
    boolean("enabled", info.enabled);
    vector("point", info.point); vector("local_offset", info.local_offset);
    vector("direction", info.direction);
    flt("surface_offset", info.surface_offset);
    flt("rate_per_second", info.rate_per_second);
    if (kwargs.contains("burst_count")) info.burst_count = py::cast<int>(kwargs["burst_count"]);
    flt("speed", info.speed); flt("spread", info.spread);
    flt("lifetime_seconds", info.lifetime_seconds); flt("mass", info.mass);
    if (kwargs.contains("appearance_profile_id"))
        info.appearance_profile_id = py::cast<uint32_t>(kwargs["appearance_profile_id"]);
    flt("size_jitter", info.size_jitter);
    flt("angular_velocity", info.angular_velocity);
    flt("angular_jitter", info.angular_jitter);
    if (kwargs.contains("seed"))
        info.seed = py::cast<unsigned int>(kwargs["seed"]);
    str("parent_object", info.parent_object);
    str("velocity_space", info.velocity_space);
    flt("inherit_velocity", info.inherit_velocity);
    boolean("override_grid_deposit", info.override_grid_deposit);
    flt("grid_density_deposit", info.grid_density_deposit);
    flt("grid_temperature_deposit", info.grid_temperature_deposit);
    flt("grid_fuel_deposit", info.grid_fuel_deposit);
}

py::dict systemToDict(const rtapi::ParticleSystemInfo& s) {
    py::dict d;
    d["index"] = s.index;
    d["id"] = s.id;
    d["name"] = s.name;
    d["active"] = s.active;
    d["enabled"] = s.enabled;
    d["visible"] = s.visible;
    d["emitter_only"] = s.emitter_only;
    d["render_in_raytrace"] = s.render_in_raytrace;
    d["domain_count"] = s.domain_count;
    d["flow_source_count"] = s.flow_source_count;
    d["emitter_count"] = s.emitter_count;
    d["collider_count"] = s.collider_count;
    d["appearance_profile_count"] = s.appearance_profile_count;
    return d;
}

// Keys that Phase 1.5 moved to appearance profiles. Accepting and ignoring
// them would turn an old script into a silent no-op, so they are refused
// (same text as the IPC side).
void rejectMovedAppearanceKeys(const py::kwargs& kwargs) {
    static const char* kMoved[] = {"start_size", "end_size", "start_opacity",
                                   "end_opacity", "start_color", "end_color", "blend_mode"};
    for (const char* key : kMoved) {
        if (kwargs.contains(key)) {
            throw std::runtime_error(
                std::string(key) + " moved to appearance profiles: set it with "
                "particle.set_appearance (the emitter's appearance_profile_id)");
        }
    }
}

py::list curveToPython(const std::vector<rtapi::ParticleCurveKeyInfo>& keys) {
    py::list out;
    for (const auto& k : keys) out.append(py::make_tuple(k.t, k.value));
    return out;
}

void curveFromPython(const py::kwargs& kwargs, const char* key,
                     std::vector<rtapi::ParticleCurveKeyInfo>& out) {
    if (!kwargs.contains(key)) return;
    out.clear();
    for (const auto& item : py::cast<py::sequence>(kwargs[key])) {
        const auto pair = py::cast<py::sequence>(item);
        if (pair.size() != 2)
            throw std::runtime_error(std::string(key) + " must be [(t, value), ...]");
        out.push_back({py::cast<float>(pair[0]), py::cast<float>(pair[1])});
    }
}

py::dict appearanceToDict(const rtapi::ParticleAppearanceInfo& a) {
    py::dict d;
    d["id"] = a.id;
    d["system_id"] = a.system_id;
    d["name"] = a.name;
    d["blend"] = a.blend;
    py::list ramp;
    for (const auto& stop : a.color_ramp)
        ramp.append(py::make_tuple(stop.t, stop.color.x, stop.color.y, stop.color.z));
    d["color_ramp"] = ramp;
    d["opacity_curve"] = curveToPython(a.opacity_curve);
    d["size_curve"] = curveToPython(a.size_curve);
    d["emission_curve"] = curveToPython(a.emission_curve);
    d["used_by_emitter_uids"] = a.used_by_emitter_uids;
    return d;
}

void patchAppearance(const py::kwargs& kwargs, rtapi::ParticleAppearanceInfo& info) {
    if (kwargs.contains("name")) info.name = py::cast<std::string>(kwargs["name"]);
    if (kwargs.contains("blend")) info.blend = py::cast<std::string>(kwargs["blend"]);
    if (kwargs.contains("color_ramp")) {
        info.color_ramp.clear();
        for (const auto& item : py::cast<py::sequence>(kwargs["color_ramp"])) {
            const auto stop = py::cast<py::sequence>(item);
            if (stop.size() != 4)
                throw std::runtime_error("color_ramp must be [(t, r, g, b), ...]");
            info.color_ramp.push_back({py::cast<float>(stop[0]),
                                       Vec3(py::cast<float>(stop[1]), py::cast<float>(stop[2]),
                                            py::cast<float>(stop[3]))});
        }
    }
    curveFromPython(kwargs, "opacity_curve", info.opacity_curve);
    curveFromPython(kwargs, "size_curve", info.size_curve);
    curveFromPython(kwargs, "emission_curve", info.emission_curve);
}

py::dict renderToDict(const rtapi::ParticleRenderInfo& r) {
    py::dict d;
    d["emitter_only"] = r.emitter_only;
    d["render_in_raytrace"] = r.render_in_raytrace;
    d["shape"] = r.shape;
    d["size_multiplier"] = r.size_multiplier;
    d["sphere_subdivisions"] = r.sphere_subdivisions;
    d["emissive"] = r.emissive;
    d["inherit_color_from_emitter"] = r.inherit_color_from_emitter;
    d["base_color"] = vec3ToPython(r.base_color);
    d["emission_strength"] = r.emission_strength;
    d["roughness"] = r.roughness;
    py::list sources;
    for (const auto& s : r.mesh_sources) {
        py::dict e;
        e["node_name"] = s.node_name;
        e["weight"] = s.weight;
        e["resolved"] = s.resolved;
        sources.append(e);
    }
    d["mesh_sources"] = sources;
    return d;
}

} // namespace

namespace rtpy {

void registerParticleBindings(py::module_& module) {
    py::module_ particle = module.def_submodule(
        "particle", "Particle systems, emitters, render and solver settings, live statistics");

    // ── Emitters ─────────────────────────────────────────────────────────
    particle.def("emitters", [](const py::object& system, int system_id) -> py::list {
        std::vector<rtapi::ParticleEmitterInfo> emitters;
        requireResult(rtapi::listParticleEmitters(systemRef(system, system_id), emitters));
        py::list out;
        for (const rtapi::ParticleEmitterInfo& info : emitters) out.append(emitterToDict(info));
        return out;
    }, py::arg("system") = py::none(), py::arg("system_id") = -1);

    // `emitter` is an index, a name, or "uid:<n>" (stable across removal and
    // save/load).
    particle.def("get_emitter", [](const std::string& emitter, const py::object& system,
                                   int system_id) {
        rtapi::ParticleEmitterInfo info;
        requireResult(rtapi::getParticleEmitter(emitter, info, systemRef(system, system_id)));
        return emitterToDict(info);
    }, py::arg("emitter"), py::arg("system") = py::none(), py::arg("system_id") = -1);

    particle.def("add_emitter", [](const py::kwargs& kwargs) {
        rejectMovedAppearanceKeys(kwargs);
        rtapi::ParticleEmitterInfo info;   // facade defaults
        patchEmitter(kwargs, info);
        rtapi::ParticleEmitterInfo created;
        requireResult(rtapi::addParticleEmitter(info, created, systemRefFromKwargs(kwargs)));
        return emitterToDict(created);
    });

    particle.def("set_emitter", [](const std::string& emitter, const py::kwargs& kwargs) {
        rejectMovedAppearanceKeys(kwargs);
        const rtapi::ParticleSystemRef ref = systemRefFromKwargs(kwargs);
        rtapi::ParticleEmitterInfo info;
        requireResult(rtapi::getParticleEmitter(emitter, info, ref));
        patchEmitter(kwargs, info);
        requireResult(rtapi::updateParticleEmitter(emitter, info, ref));
    }, py::arg("emitter"));

    // Timeline keys. Only the channels actually passed are keyed, so
    //   rt.particle.key_emitter("Embers", 120, enabled=False)
    // keys ONLY the switch and leaves rate/speed free to be keyed separately.
    particle.def("key_emitter", [](const std::string& emitter, int frame,
                                   const py::kwargs& kw) {
        rtapi::ParticleEmitterKey key;
        key.frame = frame;
        if (kw.contains("enabled")) { key.has_enabled = true; key.enabled = py::cast<bool>(kw["enabled"]); }
        if (kw.contains("rate_per_second")) { key.has_rate = true; key.rate_per_second = py::cast<float>(kw["rate_per_second"]); }
        if (kw.contains("speed")) { key.has_speed = true; key.speed = py::cast<float>(kw["speed"]); }
        if (kw.contains("spread")) { key.has_spread = true; key.spread = py::cast<float>(kw["spread"]); }
        if (kw.contains("point")) { key.has_point = true; key.point = vec3FromPython(kw["point"]); }
        if (kw.contains("direction")) { key.has_direction = true; key.direction = vec3FromPython(kw["direction"]); }
        requireResult(rtapi::keyParticleEmitter(emitter, key, systemRefFromKwargs(kw)));
    }, py::arg("emitter"), py::arg("frame"));
    particle.def("clear_emitter_key", [](const std::string& emitter, int frame,
                                         const py::object& system, int system_id) {
        requireResult(rtapi::clearParticleEmitterKey(emitter, frame, systemRef(system, system_id)));
    }, py::arg("emitter"), py::arg("frame"), py::arg("system") = py::none(),
       py::arg("system_id") = -1);
    particle.def("remove_emitter", [](const std::string& emitter, const py::object& system,
                                      int system_id) {
        requireResult(rtapi::removeParticleEmitter(emitter, systemRef(system, system_id)));
    }, py::arg("emitter"), py::arg("system") = py::none(), py::arg("system_id") = -1);

    particle.def("clear_emitters", [](const py::object& system, int system_id) {
        requireResult(rtapi::clearParticleEmitters(systemRef(system, system_id)));
    }, py::arg("system") = py::none(), py::arg("system_id") = -1);

    // ── Systems ──────────────────────────────────────────────────────────
    particle.def("list_systems", []() -> py::list {
        std::vector<rtapi::ParticleSystemInfo> systems;
        requireResult(rtapi::listParticleSystems(systems));
        py::list out;
        for (const auto& s : systems) out.append(systemToDict(s));
        return out;
    });

    particle.def("get_system", [](const py::object& system, int system_id) -> py::dict {
        rtapi::ParticleSystemInfo info;
        requireResult(rtapi::getParticleSystem(systemRef(system, system_id), info));
        return systemToDict(info);
    }, py::arg("system") = py::none(), py::arg("system_id") = -1);

    // Patches name / visible (enabled is read-only, see RtApi.h).
    particle.def("set_system", [](const py::kwargs& kwargs) {
        rejectMovedAppearanceKeys(kwargs);
        const rtapi::ParticleSystemRef ref = systemRefFromKwargs(kwargs);
        rtapi::ParticleSystemInfo info;
        requireResult(rtapi::getParticleSystem(ref, info));
        if (kwargs.contains("name")) info.name = py::cast<std::string>(kwargs["name"]);
        if (kwargs.contains("visible")) info.visible = py::cast<bool>(kwargs["visible"]);
        requireResult(rtapi::updateParticleSystem(ref, info));
    });

    // -- Appearance profiles (Phase 1.5) ------------------------------------
    // Curves are [(t, value), ...] over normalized age; color_ramp is
    // [(t, r, g, b), ...]; blend is "additive" | "alpha". Same keys as IPC.
    particle.def("list_appearances", [](const py::object& system, int system_id) -> py::list {
        std::vector<rtapi::ParticleAppearanceInfo> list;
        requireResult(rtapi::listParticleAppearances(systemRef(system, system_id), list));
        py::list out;
        for (const auto& a : list) out.append(appearanceToDict(a));
        return out;
    }, py::arg("system") = py::none(), py::arg("system_id") = -1);
    particle.def("get_appearance", [](uint32_t profile_id, const py::object& system,
                                      int system_id) -> py::dict {
        rtapi::ParticleAppearanceInfo info;
        requireResult(rtapi::getParticleAppearance(profile_id, info,
                                                   systemRef(system, system_id)));
        return appearanceToDict(info);
    }, py::arg("profile_id"), py::arg("system") = py::none(), py::arg("system_id") = -1);
    particle.def("add_appearance", [](const py::kwargs& kwargs) -> py::dict {
        rtapi::ParticleAppearanceInfo info;
        patchAppearance(kwargs, info);
        rtapi::ParticleAppearanceInfo created;
        requireResult(rtapi::addParticleAppearance(info, created, systemRefFromKwargs(kwargs)));
        return appearanceToDict(created);
    });
    particle.def("set_appearance", [](uint32_t profile_id, const py::kwargs& kwargs) {
        const rtapi::ParticleSystemRef ref = systemRefFromKwargs(kwargs);
        rtapi::ParticleAppearanceInfo info;
        requireResult(rtapi::getParticleAppearance(profile_id, info, ref));
        patchAppearance(kwargs, info);
        requireResult(rtapi::updateParticleAppearance(profile_id, info, ref));
    }, py::arg("profile_id"));
    particle.def("remove_appearance", [](uint32_t profile_id, const py::object& system,
                                         int system_id) {
        requireResult(rtapi::removeParticleAppearance(profile_id, systemRef(system, system_id)));
    }, py::arg("profile_id"), py::arg("system") = py::none(), py::arg("system_id") = -1);

    particle.def("remove_system", [](const py::object& system, int system_id) {
        requireResult(rtapi::removeParticleSystem(systemRef(system, system_id)));
    }, py::arg("system") = py::none(), py::arg("system_id") = -1,
       "Remove one particle system (explicit reference required).");

    particle.def("set_active_system", [](const py::object& system, int system_id) {
        requireResult(rtapi::setActiveParticleSystem(systemRef(system, system_id)));
    }, py::arg("system") = py::none(), py::arg("system_id") = -1,
       "Move the particle panel's selection focus (the default target of every call).");

    particle.def("set_system_emitter_only",
                 [](const std::string& system, bool emitter_only) {
        requireResult(rtapi::setParticleSystemEmitterOnly(system, emitter_only));
    }, py::arg("system"), py::arg("emitter_only"),
       "Use the system only as a gas/fluid emitter source, hiding carrier "
       "particles from RayFusion, Solid and ray-traced renders.");

    particle.def("add_system", [](const std::string& name) -> py::dict {
        rtapi::ParticleSystemInfo info;
        requireResult(rtapi::addParticleSystem(name, info));
        return systemToDict(info);
    }, py::arg("name") = "Particle System",
       "Create an empty independent particle system without applying a preset.");

    particle.def("add_preset", [](const std::string& preset) -> py::dict {
        rtapi::ParticleSystemInfo info;
        requireResult(rtapi::addParticleSystemPreset(preset, info));
        return systemToDict(info);
    }, py::arg("preset"),
       "Create one of the authored particle presets additively. Slugs: campfire, "
       "explosion, smoke, ground_burst, fireball, flamethrower, burning_fuel_spill, "
       "ignited_fuel_jet, nuclear.");

    particle.def("clear_systems", []() { requireResult(rtapi::clearParticleSystems()); });

    // ── Render settings ──────────────────────────────────────────────────
    particle.def("get_render", [](const py::object& system, int system_id) -> py::dict {
        rtapi::ParticleRenderInfo info;
        requireResult(rtapi::getParticleRender(systemRef(system, system_id), info));
        return renderToDict(info);
    }, py::arg("system") = py::none(), py::arg("system_id") = -1);

    // mesh_sources replaces the whole list: [{"node_name": ..., "weight": ...}]
    // or [(node_name, weight)].
    particle.def("set_render", [](const py::kwargs& kwargs) {
        const rtapi::ParticleSystemRef ref = systemRefFromKwargs(kwargs);
        rtapi::ParticleRenderInfo info;
        requireResult(rtapi::getParticleRender(ref, info));
        auto flt = [&](const char* key, float& target) {
            if (kwargs.contains(key)) target = py::cast<float>(kwargs[key]);
        };
        auto boolean = [&](const char* key, bool& target) {
            if (kwargs.contains(key)) target = py::cast<bool>(kwargs[key]);
        };
        boolean("emitter_only", info.emitter_only);
        boolean("render_in_raytrace", info.render_in_raytrace);
        if (kwargs.contains("shape")) info.shape = py::cast<std::string>(kwargs["shape"]);
        flt("size_multiplier", info.size_multiplier);
        if (kwargs.contains("sphere_subdivisions"))
            info.sphere_subdivisions = py::cast<int>(kwargs["sphere_subdivisions"]);
        boolean("emissive", info.emissive);
        boolean("inherit_color_from_emitter", info.inherit_color_from_emitter);
        if (kwargs.contains("base_color")) info.base_color = vec3FromPython(kwargs["base_color"]);
        flt("emission_strength", info.emission_strength);
        flt("roughness", info.roughness);
        if (kwargs.contains("mesh_sources")) {
            info.mesh_sources.clear();
            const py::sequence list = py::reinterpret_borrow<py::sequence>(py::object(kwargs["mesh_sources"]));
            for (std::size_t i = 0; i < py::len(list); ++i) {
                const py::object item = list[i];   // owned: survives past this line
                rtapi::ParticleRenderMeshSourceInfo entry;
                if (py::isinstance<py::dict>(item)) {
                    py::dict e = py::reinterpret_borrow<py::dict>(item);
                    entry.node_name = py::cast<std::string>(e["node_name"]);
                    if (e.contains("weight")) entry.weight = py::cast<float>(e["weight"]);
                } else {
                    py::sequence pair = py::reinterpret_borrow<py::sequence>(item);
                    entry.node_name = py::cast<std::string>(pair[0]);
                    if (py::len(pair) > 1) entry.weight = py::cast<float>(pair[1]);
                }
                info.mesh_sources.push_back(std::move(entry));
            }
        }
        requireResult(rtapi::updateParticleRender(ref, info));
    });

    // ── Solver settings ──────────────────────────────────────────────────
    particle.def("get_physics", [](const py::object& system, int system_id) -> py::dict {
        rtapi::ParticlePhysicsInfo info;
        requireResult(rtapi::getParticlePhysics(info, systemRef(system, system_id)));
        py::dict d;
        d["mode"] = info.mode;
        d["quality"] = info.quality;
        d["execution_policy"] = info.execution_policy;
        d["particle_radius"] = info.particle_radius;
        d["self_collision_enabled"] = info.self_collision_enabled;
        d["solver_iterations"] = info.solver_iterations;
        d["max_neighbors_per_particle"] = info.max_neighbors_per_particle;
        d["viscosity"] = info.viscosity;
        d["cohesion"] = info.cohesion;
        d["pressure_stiffness"] = info.pressure_stiffness;
        d["rest_density"] = info.rest_density;
        d["buoyancy"] = info.buoyancy;
        d["gravity_scale"] = info.gravity_scale;
        d["vorticity"] = info.vorticity;
        d["grid_density_deposit"] = info.grid_density_deposit;
        d["grid_temperature_deposit"] = info.grid_temperature_deposit;
        d["grid_fuel_deposit"] = info.grid_fuel_deposit;
        d["grid_deposit_fade_with_age"] = info.grid_deposit_fade_with_age;
        return d;
    }, py::arg("system") = py::none(), py::arg("system_id") = -1);

    particle.def("set_physics", [](const py::kwargs& kwargs) {
        const rtapi::ParticleSystemRef ref = systemRefFromKwargs(kwargs);
        rtapi::ParticlePhysicsInfo info;
        requireResult(rtapi::getParticlePhysics(info, ref));
        auto str = [&](const char* key, std::string& target) {
            if (kwargs.contains(key)) target = py::cast<std::string>(kwargs[key]);
        };
        auto flt = [&](const char* key, float& target) {
            if (kwargs.contains(key)) target = py::cast<float>(kwargs[key]);
        };
        auto integer = [&](const char* key, int& target) {
            if (kwargs.contains(key)) target = py::cast<int>(kwargs[key]);
        };
        auto boolean = [&](const char* key, bool& target) {
            if (kwargs.contains(key)) target = py::cast<bool>(kwargs[key]);
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
        requireResult(rtapi::updateParticlePhysics(info, ref));
    });

    // ── Statistics and state ─────────────────────────────────────────────
    particle.def("stats", [](const py::object& system, int system_id) -> py::dict {
        rtapi::ParticleStatsInfo info;
        requireResult(rtapi::getParticleStats(info, systemRef(system, system_id)));
        py::dict d;
        d["system_id"] = info.system_id;
        d["alive_count"] = info.alive_count;
        d["capacity"] = info.capacity;
        d["emitter_count"] = info.emitter_count;
        d["collider_count"] = info.collider_count;
        d["domain_count"] = info.domain_count;
        d["total_ms"] = info.total_ms;
        d["emit_ms"] = info.emit_ms;
        d["integrate_ms"] = info.integrate_ms;
        d["self_collision_ms"] = info.self_collision_ms;
        d["grid_domain_ms"] = info.grid_domain_ms;
        d["execution_policy"] = info.execution_policy;
        d["compute_backend"] = info.compute_backend;
        d["gpu_status"] = info.gpu_status;
        d["device_resident"] = info.device_resident;
        d["step_blocked"] = info.step_blocked;
        py::dict stages;
        stages["emit"] = info.emit_backend;
        stages["forces"] = info.forces_backend;
        stages["integrate"] = info.integrate_backend;
        stages["scene_collision"] = info.scene_collision_backend;
        stages["self_collision"] = info.self_collision_backend;
        d["stage_backends"] = stages;
        d["gpu_step_ms"] = info.gpu_step_ms;
        d["residency"] = info.residency;
        d["resident_capacity"] = info.resident_capacity;
        d["slot_records"] = info.slot_records;
        d["step_upload_bytes"] = info.step_upload_bytes;
        d["step_download_bytes"] = info.step_download_bytes;
        d["step_dispatch_calls"] = info.step_dispatch_calls;
        d["step_synchronize_calls"] = info.step_synchronize_calls;
        d["step_upload_call_ms"] = info.step_upload_call_ms;
        d["step_download_call_ms"] = info.step_download_call_ms;
        d["step_synchronize_ms"] = info.step_synchronize_ms;
        d["snapshot_sync_count"] = info.snapshot_sync_count;
        d["snapshot_mirrored_steps"] = info.snapshot_mirrored_steps;
        d["snapshot_download_bytes"] = info.snapshot_download_bytes;
        d["snapshot_last_reason"] = info.snapshot_last_reason;
        d["nonfinite_particles"] = info.nonfinite_particles;
        d["nonfinite_measured"] = info.nonfinite_measured;
        d["grid_deposit_landed"] = info.grid_deposit_landed;
        d["grid_deposit_dropped_no_domain"] = info.grid_deposit_dropped_no_domain;
        d["grid_deposit_dropped_no_channel"] = info.grid_deposit_dropped_no_channel;
        return d;
    }, py::arg("system") = py::none(), py::arg("system_id") = -1);

    particle.def("get_state_sample", [](int max_count, int offset, int stride,
                                        const py::object& system, int system_id) -> py::dict {
        rtapi::ParticleStateSample sample;
        requireResult(rtapi::getParticleStateSample(max_count, offset, stride, sample,
                                                    systemRef(system, system_id)));
        py::list indices, positions, velocities, ages;
        for (int i = 0; i < sample.returned; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            indices.append(sample.indices[k]);
            positions.append(vec3ToPython(sample.positions[k]));
            velocities.append(vec3ToPython(sample.velocities[k]));
            ages.append(sample.ages[k]);
        }
        py::dict d;
        d["alive_count"] = sample.alive_count;
        d["capacity"] = sample.capacity;
        d["returned"] = sample.returned;
        d["indices"] = indices;
        d["positions"] = positions;
        d["velocities"] = velocities;
        d["ages"] = ages;
        d["centroid"] = vec3ToPython(sample.centroid);
        d["mean_velocity"] = vec3ToPython(sample.mean_velocity);
        d["bounds_min"] = vec3ToPython(sample.bounds_min);
        d["bounds_max"] = vec3ToPython(sample.bounds_max);
        d["nonfinite"] = sample.nonfinite;
        return d;
    }, py::arg("max_count") = 256, py::arg("offset") = 0, py::arg("stride") = 1,
       py::arg("system") = py::none(), py::arg("system_id") = -1);

    // ── Direct control ───────────────────────────────────────────────────
    particle.def("spawn", [](const py::handle& position, const py::handle& velocity,
                             float lifetime_seconds, float mass, float size,
                             const py::object& system, int system_id) {
        int index = -1;
        requireResult(rtapi::spawnParticle(vec3FromPython(position), vec3FromPython(velocity),
                                           lifetime_seconds, mass, size, index,
                                           systemRef(system, system_id)));
        return index;
    }, py::arg("position") = py::make_tuple(0.0f, 1.0f, 0.0f),
       py::arg("velocity") = py::make_tuple(0.0f, 0.0f, 0.0f),
       py::arg("lifetime_seconds") = 5.0f, py::arg("mass") = 1.0f, py::arg("size") = 0.05f,
       py::arg("system") = py::none(), py::arg("system_id") = -1);

    particle.def("clear", [](const py::object& system, int system_id) {
        requireResult(rtapi::clearParticles(systemRef(system, system_id)));
    }, py::arg("system") = py::none(), py::arg("system_id") = -1);

    particle.def("step", [](float dt, const py::object& system, int system_id) {
        requireResult(rtapi::stepParticleSimulation(dt, systemRef(system, system_id)));
    }, py::arg("dt") = 0.0166667f, py::arg("system") = py::none(), py::arg("system_id") = -1);
}

} // namespace rtpy
