/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiSimNodes.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 * Scripting surface for the simulation node graph (Faz N0/N1/N2), plus the
 * attribute resolver the node layer reads through.
 *
 * ★ This file is the SEAM. NodeSystem must not include solver headers and the
 * solvers must not know about nodes, so the application installs a resolver
 * here. Nodes stay drivers; nothing in NodeSystem gains a dependency.
 */

#include "Api/RtApiInternal.h"
#include "NodeSystem/SimulationNodes.h"
#include "NodeSystem/NodeRegistry.h"
#include "Fluid/FluidParticles.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>

namespace rtapi {

namespace {

// ── The scoped graph registry ───────────────────────────────────────────────
//
// Decision record: docs/dev/SIMULATION_NODE_OBJECT_MODEL.md, section 8 steps 1-2.
//
// ★★★ The graphs live on the SCENE (scene_data.h), not in a static here. The
// static this replaced outlived scene changes — the same shape as the fracture
// UI cache that went on describing a scene that no longer existed. Opening a
// project now replaces the graphs along with everything they name.
//
// ★★★ There is NO default scope and NO default owner. Every entry point below
// takes both. An "active domain" fallback is precisely the silent assumption
// this repository keeps paying for: the call succeeds, edits something else,
// and no reading ever contradicts it.

using SimGraphPtr = std::shared_ptr<NodeSystem::Sim::SimulationNodeGraph>;

// Where a scope's graphs are stored. Returns null for World, which has a single
// graph rather than a map — the caller handles that case explicitly.
std::unordered_map<std::string, SimGraphPtr>*
scopeStorage(UIContext& ctx, NodeSystem::Sim::GraphScope scope) {
    switch (scope) {
        case NodeSystem::Sim::GraphScope::Domain: return &ctx.scene.simulation_domain_graphs;
        case NodeSystem::Sim::GraphScope::Object: return &ctx.scene.simulation_object_graphs;
        case NodeSystem::Sim::GraphScope::World:  return nullptr;
    }
    return nullptr;
}

// ★ One message shape for every bad scope string, and it LISTS the choices.
// A caller that guessed wrong learns what the options are from the failure.
Result parseScopeArg(const std::string& scope_text,
                     NodeSystem::Sim::GraphScope& out) {
    if (!NodeSystem::Sim::parseScope(scope_text, out))
        return Result::fail("unknown graph scope '" + scope_text +
                            "' (expected 'object', 'domain' or 'world')");
    return Result::success();
}

// Does the scene entity a graph would name actually exist? Checked when a graph
// is CREATED, not on every edit.
//
// ★★ A graph whose owner does not exist is the failure this check exists to
// prevent: it looks healthy, accepts nodes, applies nothing, and reports no
// error — the "node forever pointing at nothing" shape from the sim node panel.
Result ownerExists(NodeSystem::Sim::GraphScope scope, const std::string& owner) {
    if (scope == NodeSystem::Sim::GraphScope::World) return Result::success();
    if (owner.empty())
        return Result::fail(std::string("scope '") + NodeSystem::Sim::scopeName(scope) +
                            "' needs an owner name");
    if (scope == NodeSystem::Sim::GraphScope::Object) {
        // ★★★ An object surface has TWO legitimate names: the scene object, and
        // the simulation collider that references it. Both are accepted here for
        // the same reason ObjectRefNode resolves both — the authored material
        // lives on the collider while the measured MSF field is keyed by its
        // source object. Accepting only one would make a graph silently work
        // under one name and silently do nothing under the other, with no error
        // either way, which this repository has already measured once.
        if (objectExists(owner)) return Result::success();
        SimulationColliderInfo collider;
        if (getSimulationCollider(owner, collider).ok) return Result::success();
        return Result::fail("no object or simulation collider named '" + owner +
                            "' in the scene");
    }
    std::vector<FluidDomainInfo> domains;
    Result listed = listFluidDomains(domains);
    if (!listed.ok) return listed;
    for (const auto& d : domains)
        if (d.name == owner) return Result::success();
    return Result::fail("no simulation domain named '" + owner + "' in the scene");
}

// Looks up an EXISTING graph. Does not create one.
//
// ★★★ Deliberately not create-on-demand. Auto-creating here would make a typo
// in a domain name produce a second, empty graph that silently accepts every
// later edit — the caller would see success on every call and no effect
// anywhere. Creation is its own explicit call.
Result findGraph(const std::string& scope_text, const std::string& owner,
                 SimGraphPtr& out) {
    if (!g_ctx) return notBound();
    NodeSystem::Sim::GraphScope scope;
    Result parsed = parseScopeArg(scope_text, scope);
    if (!parsed.ok) return parsed;

    if (scope == NodeSystem::Sim::GraphScope::World) {
        if (!g_ctx->scene.simulation_world_graph)
            return Result::fail("no world simulation graph (create it first)");
        out = g_ctx->scene.simulation_world_graph;
        return Result::success();
    }
    auto* storage = scopeStorage(*g_ctx, scope);
    if (!storage) return Result::fail("unsupported graph scope");
    auto it = storage->find(owner);
    if (it == storage->end() || !it->second)
        return Result::fail(std::string("no ") + NodeSystem::Sim::scopeName(scope) +
                            " graph for '" + owner + "' (create it first)");
    out = it->second;
    return Result::success();
}

// ── The naming layer, first instance ────────────────────────────────────────
//
// (the graph accessor itself is published at the bottom of this file)
//
// ★★ This table IS the attribute naming layer for per-particle data (plan D.3).
// It creates no storage: every entry points at an array that already exists.
// What it adds is DISCOVERABILITY — until now the only way to learn that
// `substance_tag` existed was to read the solver source, and that cost real
// debugging time.
//
// ★★★ Name at the boundary, index in the loop. Resolution by string happens
// HERE, once per query. No solver inner loop may ever look a name up.
struct AttributeBinding {
    const char* name;
    const std::vector<float>  RayTrophiSim::Fluid::FluidParticles::* float_array;
    const std::vector<uint32_t> RayTrophiSim::Fluid::FluidParticles::* uint_array;
};

const AttributeBinding kParticleAttributes[] = {
    {"temperature",           &RayTrophiSim::Fluid::FluidParticles::temperature,           nullptr},
    {"mass_fraction",         &RayTrophiSim::Fluid::FluidParticles::mass_fraction,         nullptr},
    {"combustible_fraction",  &RayTrophiSim::Fluid::FluidParticles::combustible_fraction,  nullptr},
    {"granular_softening",    &RayTrophiSim::Fluid::FluidParticles::granular_softening,    nullptr},
    {"granular_bond_scale",   &RayTrophiSim::Fluid::FluidParticles::granular_bond_scale,   nullptr},
    {"granular_damage",       &RayTrophiSim::Fluid::FluidParticles::granular_damage,       nullptr},
    {"granular_hardening",    &RayTrophiSim::Fluid::FluidParticles::granular_hardening,    nullptr},
    {"granular_yield_value",  &RayTrophiSim::Fluid::FluidParticles::granular_yield_value,  nullptr},
    {"substance_tag",         nullptr, &RayTrophiSim::Fluid::FluidParticles::substance_tag},
};

// Find the LIVE particle set for a named domain. Deliberately mirrors
// fluid.list_domains rather than fluid.get: on 2026-08-16 the latter answered
// from a stale same-named descriptor and reported an empty domain as fact.
const RayTrophiSim::Fluid::FluidParticles* liveParticles(const std::string& domain) {
    if (!g_ctx) return nullptr;
    const RayTrophiSim::Fluid::FluidParticles* best = nullptr;
    auto consider = [&](RayTrophiSim::ParticleSimulationSystem& sys) {
        const auto& domains = sys.gridDomains();
        const auto& states = sys.gridDomainStates();
        for (std::size_t i = 0; i < domains.size(); ++i) {
            if (domains[i].name != domain) continue;
            if (i < states.size() && states[i].valid) {
                best = &states[i].particles;
                return true;
            }
        }
        return false;
    };
    for (auto& system : g_ctx->scene.particle_systems) {
        if (!system.runtime) continue;
        if (consider(*system.runtime)) break;
    }
    return best;
}

// ★★★ `elements` is the AUTHORITY, not values.size(). A per-particle array can
// outlive the particles: on 2026-08-17 the granular arrays measured 3987 entries
// against 3981 live particles, and iterating the whole vector averaged in six
// dead slots. Nothing errored — the statistics simply described a population
// that no longer existed, which is precisely the failure shape nobody reports.
//
// The disagreement is carried out to the caller instead of being clamped away in
// silence; a stale tail is a fact about the solver, not noise to hide.
template <typename T>
void accumulate(const std::vector<T>& values, std::size_t elements,
                NodeSystem::Sim::FieldStats& out) {
    const std::size_t n = std::min(elements, values.size());
    double sum = 0.0;
    double lo = std::numeric_limits<double>::max();
    double hi = -std::numeric_limits<double>::max();
    uint32_t counted = 0;
    for (std::size_t i = 0; i < n; ++i) {
        // double, not float: a uint32 identity (substance_tag) is exact here and
        // rounds to a DIFFERENT identity in a float past 2^24.
        const double f = static_cast<double>(values[i]);
        if (!std::isfinite(f)) continue;   // a NaN must not poison min/max
        sum += f;
        lo = std::min(lo, f);
        hi = std::max(hi, f);
        ++counted;
    }
    out.count = counted;
    out.array_size = static_cast<uint32_t>(values.size());
    out.in_sync = values.size() == elements;
    out.min_value = counted ? lo : 0.0;
    out.max_value = counted ? hi : 0.0;
    out.mean_value = counted ? sum / counted : 0.0;
}

bool resolveAttributeStats(const std::string& domain, const std::string& attribute,
                           NodeSystem::Sim::FieldStats& out) {
    const auto* particles = liveParticles(domain);
    if (!particles) return false;          // ★ false, NOT zeros
    // FluidParticles::size() is position.size() — the one count every other
    // array is supposed to match. Attribute arrays are measured against it.
    const std::size_t elements = particles->size();
    for (const auto& binding : kParticleAttributes) {
        if (attribute != binding.name) continue;
        if (binding.float_array) {
            const auto& arr = particles->*(binding.float_array);
            if (arr.empty()) return false; // the array is not allocated here
            accumulate(arr, elements, out);
            return true;
        }
        if (binding.uint_array) {
            const auto& arr = particles->*(binding.uint_array);
            if (arr.empty()) return false;
            accumulate(arr, elements, out);
            return true;
        }
    }
    return false;                          // unknown name: say so, do not guess
}

std::vector<std::string> resolveAttributeList(const std::string& domain) {
    std::vector<std::string> names;
    const auto* particles = liveParticles(domain);
    if (!particles) return names;
    for (const auto& binding : kParticleAttributes) {
        const bool present =
            (binding.float_array && !(particles->*(binding.float_array)).empty()) ||
            (binding.uint_array  && !(particles->*(binding.uint_array)).empty());
        if (present) names.emplace_back(binding.name);
    }
    return names;
}

// ── N3: the override layer ──────────────────────────────────────────────────
//
// ★★★ An override MUST be reversible. Plan B.5: solver configuration is runtime
// state and has to stay resettable, so a graph may not write into the authored
// data. The authored value is captured the FIRST time a key is overridden and
// restored when overrides are cleared.
//
// ★★ Capture must happen BEFORE the first write, never after — reading it back
// afterwards would record the override as if it were the authored value, and
// the original would be gone with nothing to show it ever existed. That is the
// same shape as the melt bug where a derived value was stored as if authored.
struct OverrideKey {
    std::string domain;
    std::string key;
    bool operator==(const OverrideKey& o) const {
        return domain == o.domain && key == o.key;
    }
};
struct OverrideKeyHash {
    size_t operator()(const OverrideKey& k) const {
        return std::hash<std::string>{}(k.domain) ^
               (std::hash<std::string>{}(k.key) << 1);
    }
};
std::unordered_map<OverrideKey, float, OverrideKeyHash> g_authored_values;

// Reading a parameter's current value. Uses listFluidDomains, the authority
// that answers correctly for grid-based domains.
bool readParameter(const std::string& domain, const std::string& key, float& out) {
    std::vector<FluidDomainInfo> domains;
    if (!listFluidDomains(domains).ok) return false;
    for (const auto& d : domains) {
        if (d.name != domain) continue;
        if (key == "kinematic_viscosity")   { out = d.kinematic_viscosity;   return true; }
        if (key == "viscosity_wall_slip")   { out = d.viscosity_wall_slip;   return true; }
        if (key == "surface_offset_voxels") { out = d.surface_offset_voxels; return true; }
        if (key == "pore_amount")           { out = d.pore_amount;           return true; }
        if (key == "pore_scale")            { out = d.pore_scale;            return true; }
        if (key == "pore_detail")           { out = d.pore_detail;           return true; }
        if (key == "solid_phase_fill")      { out = d.solid_phase_fill;      return true; }
        if (key == "granular_cohesion")     { out = d.granular_cohesion;     return true; }
        if (key == "voxel_size")            { out = d.voxel_size;            return true; }
        // ★★ Integer and boolean parameters travel as floats because SimCommand
        // carries one numeric type. They are rounded back on the WRITE side, in
        // one place — a value that round-trips through a float here and is
        // truncated somewhere else would drift by one and nobody would look.
        if (key == "viscosity_sweeps")      { out = static_cast<float>(d.viscosity_sweeps);    return true; }
        if (key == "uvw_refresh_period")    { out = static_cast<float>(d.uvw_refresh_period);  return true; }
        if (key == "granular_friction_angle_degrees")
                                            { out = d.granular_friction_angle_degrees;        return true; }
        if (key == "solid_phase")           { out = d.solid_phase_enabled ? 1.0f : 0.0f;       return true; }
        if (key == "enabled")               { out = d.enabled ? 1.0f : 0.0f;                   return true; }
        if (key == "visible")               { out = d.visible ? 1.0f : 0.0f;                   return true; }
        return false;                       // unknown key: say so, do not guess
    }
    return false;                           // unknown domain
}

bool writeParameter(const std::string& domain, const std::string& key, float value) {
    const float* kinematic_viscosity = nullptr;
    const float* viscosity_wall_slip = nullptr;
    const float* surface_offset_voxels = nullptr;
    const float* pore_amount = nullptr;
    const float* pore_scale = nullptr;
    const float* pore_detail = nullptr;
    const float* solid_phase_fill = nullptr;
    const float* granular_cohesion = nullptr;
    const float* voxel_size = nullptr;
    const float* granular_friction_angle = nullptr;
    // ★ Non-float targets need their own storage: updateFluidDomain takes an
    // int/bool pointer, and handing it the address of `value` would reinterpret
    // a float's bits as an integer.
    int  sweeps_store = 0;      const int*  viscosity_sweeps = nullptr;
    int  refresh_store = 0;     const int*  uvw_refresh_period = nullptr;
    bool solid_store = false;   const bool* solid_phase = nullptr;
    bool enabled_store = false; const bool* enabled = nullptr;
    bool visible_store = false; const bool* visible = nullptr;

    // ★★ ONE rounding site for every integer parameter, and it rounds rather
    // than truncates: 7.999999 arriving from a float round-trip must land on 8,
    // not 7. A truncation here would make a value read back one lower than the
    // one just written, and the override layer would then "restore" the wrong
    // authored number.
    auto as_int = [&value]() { return static_cast<int>(value + (value < 0.0f ? -0.5f : 0.5f)); };
    // Anything above zero means on. There is no in-between to interpolate.
    auto as_bool = [&value]() { return value > 0.5f; };

    if (key == "kinematic_viscosity")        kinematic_viscosity = &value;
    else if (key == "viscosity_wall_slip")   viscosity_wall_slip = &value;
    else if (key == "surface_offset_voxels") surface_offset_voxels = &value;
    else if (key == "pore_amount")           pore_amount = &value;
    else if (key == "pore_scale")            pore_scale = &value;
    else if (key == "pore_detail")           pore_detail = &value;
    else if (key == "solid_phase_fill")      solid_phase_fill = &value;
    else if (key == "granular_cohesion")     granular_cohesion = &value;
    else if (key == "voxel_size")            voxel_size = &value;
    else if (key == "granular_friction_angle_degrees") granular_friction_angle = &value;
    else if (key == "viscosity_sweeps")   { sweeps_store  = as_int();  viscosity_sweeps   = &sweeps_store; }
    else if (key == "uvw_refresh_period") { refresh_store = as_int();  uvw_refresh_period = &refresh_store; }
    else if (key == "solid_phase")        { solid_store   = as_bool(); solid_phase        = &solid_store; }
    else if (key == "enabled")            { enabled_store = as_bool(); enabled            = &enabled_store; }
    else if (key == "visible")            { visible_store = as_bool(); visible            = &visible_store; }
    // ★★★ granular_enabled is deliberately NOT here. Toggling it changes which
    // solver representation the domain runs, and whether that invalidates the
    // accumulated state has not been MEASURED. Exposing it without knowing
    // would either discard a simulation silently or demand a restart nobody
    // needs — and a dial whose restart semantics are guessed is worse than a
    // missing one, because the guess is invisible.
    else return false;

    // ★ One argument per line, each labelled. updateFluidDomain takes 25
    // same-typed optional pointers; a positional call written compactly is a
    // silent-shift waiting to happen, and the symptom would be "setting the
    // porosity changed the wall slip" with nothing to point at.
    return updateFluidDomain(
        domain,
        /* domain_min                     */ nullptr,
        /* domain_max                     */ nullptr,
        /* voxel_size                     */ voxel_size,
        /* render_mode                    */ nullptr,
        /* backend                        */ nullptr,
        /* boundary                       */ nullptr,
        /* preset                         */ nullptr,
        /* kinematic_viscosity            */ kinematic_viscosity,
        /* viscosity_sweeps               */ viscosity_sweeps,
        /* viscosity_wall_slip            */ viscosity_wall_slip,
        /* surface_material               */ nullptr,
        /* surface_offset_voxels          */ surface_offset_voxels,
        /* pore_amount                    */ pore_amount,
        /* pore_scale                     */ pore_scale,
        /* pore_detail                    */ pore_detail,
        /* coord_space                    */ nullptr,
        /* uvw_refresh_period             */ uvw_refresh_period,
        /* solid_phase                    */ solid_phase,
        /* solid_phase_fill               */ solid_phase_fill,
        /* enabled                        */ enabled,
        /* visible                        */ visible,
        /* granular_enabled               */ nullptr,
        /* granular_friction_angle_degrees*/ granular_friction_angle,
        /* granular_cohesion              */ granular_cohesion).ok;
}

// ── World thermal ambient (section 7 item 1, section 8 step 4) ─────────────
//
// ★ Reads and writes go through rt.world.get_thermal/set_thermal's own
// implementation (getWorldThermal/setWorldThermal), not a second path into
// WorldThermalState — one place owns the field list and its validation.
bool readWorldThermalParameter(const std::string& key, float& out) {
    WorldThermalInfo t;
    if (!getWorldThermal(t).ok) return false;
    if (key == "ambient_kelvin")         { out = t.ambient_kelvin;         return true; }
    if (key == "kelvin_per_unit")        { out = t.kelvin_per_unit;        return true; }
    if (key == "convection_coefficient") { out = t.convection_coefficient; return true; }
    if (key == "oxygen_availability")    { out = t.oxygen_availability;    return true; }
    return false;                        // unknown key: say so, do not guess
}

bool writeWorldThermalParameter(const std::string& key, float value) {
    if (key == "ambient_kelvin")         return setWorldThermal(&value, nullptr, nullptr, nullptr).ok;
    if (key == "kelvin_per_unit")        return setWorldThermal(nullptr, &value, nullptr, nullptr).ok;
    if (key == "convection_coefficient") return setWorldThermal(nullptr, nullptr, &value, nullptr).ok;
    if (key == "oxygen_availability")    return setWorldThermal(nullptr, nullptr, nullptr, &value).ok;
    return false;
}

// ── N6: cache status seam ───────────────────────────────────────────────────
bool resolveCacheStatus(NodeSystem::Sim::CacheStatusReading& out) {
    if (!g_ctx) return false;
    out.valid = g_ctx->scene.simCacheValid();
    out.baking = g_ctx->scene.simBakeActive();
    out.ram_frames = static_cast<uint32_t>(g_ctx->scene.simFrameCacheCount());
    out.has_range = g_ctx->scene.simFrameCacheRange(out.first_frame, out.last_frame);
    out.config_signature = g_ctx->scene.simConfigSignature();
    // ★ Hashed fresh, not remembered. Comparing the cache's signature against a
    // remembered copy of itself would always agree; the question is whether the
    // SCENE has moved since the bake.
    out.live_signature = g_ctx->scene.computeSimConfigSignature();
    return true;
}

// ── N5: per-object surface (MSF) attributes ─────────────────────────────────
//
// ★★ The naming layer again, now over per-TEXEL data — and again it creates no
// storage. MaterialStateField keeps its host mirror as one interleaved array of
// 8 floats per element; naming it is what makes it discoverable.
struct SurfaceAttributeBinding {
    const char* name;
    std::size_t offset;   // index inside MaterialStateField::kStateStride
};

// state layout, 8 floats per element:
//   [0..3] temperature, fuel_remaining, char, moisture
//   [4..7] released_this_step, melt, mass_loss, transferred_mass
const SurfaceAttributeBinding kSurfaceAttributes[] = {
    {"temperature",      0},
    {"fuel_remaining",   1},
    {"char",             2},
    {"moisture",         3},
    {"released",         4},
    {"melt",             5},
    {"mass_loss",        6},
    {"transferred_mass", 7},
};

// ★★★ AN OBJECT HAS TWO NAMES HERE, and they are not interchangeable.
//
// The AUTHORED material lives on the collider ("SubstanceTestCollider"); the
// MEASURED state lives in an MSF field keyed by the collider's SOURCE OBJECT
// ("SubstanceTestBox"), because plan A.2 deliberately moved burn state out of
// the grid and onto the object. Measured 2026-08-17: naming the collider read
// back an empty attribute list while a 99072-element field sat right there.
//
// Resolving only one of them would make a node silently work under one name and
// silently do nothing under the other — with no error either way, which is the
// worst shape a naming layer can have. Both directions are resolved.
std::string colliderNameForIdentity(const std::string& identity) {
    std::vector<SimulationColliderInfo> colliders;
    if (!listSimulationColliders(colliders).ok) return identity;
    for (const auto& c : colliders) if (c.name == identity) return c.name;
    for (const auto& c : colliders) if (c.source_object == identity) return c.name;
    return identity;
}

std::string surfaceKeyForIdentity(const std::string& identity) {
    std::vector<SimulationColliderInfo> colliders;
    if (!listSimulationColliders(colliders).ok) return identity;
    // A collider name maps to the object it represents; that object key is what
    // the MSF map uses. An identity that is already an object key stays as-is.
    for (const auto& c : colliders) {
        if (c.name == identity && !c.source_object.empty()) return c.source_object;
    }
    return identity;
}

const RayTrophiSim::MaterialStateField* liveSurface(const std::string& object) {
    if (!g_ctx) return nullptr;
    const std::string key = surfaceKeyForIdentity(object);
    for (auto& system : g_ctx->scene.particle_systems) {
        if (!system.runtime) continue;
        const auto& fields = system.runtime->materialStateFields();
        auto it = fields.find(key);
        if (it != fields.end()) return &it->second;
        if (key != object) {
            it = fields.find(object);
            if (it != fields.end()) return &it->second;
        }
    }
    return nullptr;
}

bool resolveSurfaceStats(const std::string& object, const std::string& attribute,
                         NodeSystem::Sim::FieldStats& out) {
    const auto* field = liveSurface(object);
    if (!field) return false;              // ★ false, NOT zeros
    const std::size_t stride = RayTrophiSim::MaterialStateField::kStateStride;
    const std::size_t elements = field->state.size() / stride;
    if (elements == 0) return false;
    for (const auto& binding : kSurfaceAttributes) {
        if (attribute != binding.name) continue;
        double sum = 0.0;
        double lo = std::numeric_limits<double>::max();
        double hi = -std::numeric_limits<double>::max();
        uint32_t counted = 0;
        for (std::size_t i = 0; i < elements; ++i) {
            const double v = static_cast<double>(field->state[i * stride + binding.offset]);
            if (!std::isfinite(v)) continue;
            sum += v;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            ++counted;
        }
        out.count = counted;
        out.array_size = static_cast<uint32_t>(elements);
        // The MSF host mirror is one array, so its channels cannot disagree in
        // length by construction. Reported anyway, so the field means the same
        // thing on both inspectors.
        out.in_sync = true;
        // ★★★ Whether these numbers describe the CURRENT device state. A field
        // that has never been read back still returns its initialisation values
        // — fuel_remaining = -1 means "not seeded yet", not "negative fuel".
        out.host_fresh = field->host_state_fresh;
        out.min_value = counted ? lo : 0.0;
        out.max_value = counted ? hi : 0.0;
        out.mean_value = counted ? sum / counted : 0.0;
        return true;
    }
    return false;                          // unknown name: say so, do not guess
}

std::vector<std::string> resolveSurfaceList(const std::string& object) {
    std::vector<std::string> names;
    const auto* field = liveSurface(object);
    if (!field) return names;
    const std::size_t stride = RayTrophiSim::MaterialStateField::kStateStride;
    if (field->state.size() < stride) return names;
    for (const auto& binding : kSurfaceAttributes) names.emplace_back(binding.name);
    return names;
}

// Reading and writing the AUTHORED per-object settings. These live on the
// collider descriptor, which is the same thing the panel edits.
bool readSurfaceParameter(const std::string& object, const std::string& key,
                          float& out) {
    SimulationColliderInfo c;
    if (!getSimulationCollider(colliderNameForIdentity(object), c).ok) return false;
    if (key == "ignite_on_contact")   { out = c.gas_ignite_on_contact ? 1.0f : 0.0f; return true; }
    if (key == "override_ignition")   { out = c.msf_override_ignition ? 1.0f : 0.0f; return true; }
    if (key == "ignition_kelvin")     { out = c.msf_ignition_kelvin;     return true; }
    if (key == "burn_rate_scale")     { out = c.msf_burn_rate_scale;     return true; }
    if (key == "fuel_capacity_scale") { out = c.msf_fuel_capacity_scale; return true; }
    if (key == "melt_flow")           { out = c.msf_melt_flow_enabled ? 1.0f : 0.0f; return true; }
    if (key == "melt_height_loss")    { out = c.msf_melt_height_loss;   return true; }
    if (key == "melt_spread")         { out = c.msf_melt_spread;        return true; }
    return false;                     // unknown key: say so, do not guess
}

bool readSurfaceText(const std::string& object, const std::string& key,
                     std::string& out) {
    SimulationColliderInfo c;
    if (!getSimulationCollider(colliderNameForIdentity(object), c).ok) return false;
    if (key == "substance") { out = c.msf_substance; return true; }
    return false;
}

// ★ Read-modify-write of the WHOLE descriptor, on purpose. updateSimulationCollider
// takes a complete SimulationColliderInfo, so writing one field means handing
// back every other field unchanged — fetching first is what keeps this from
// resetting the object's shape and rates to their defaults.
bool writeSurfaceParameter(const std::string& object, const std::string& key,
                           float value) {
    const std::string collider = colliderNameForIdentity(object);
    SimulationColliderInfo c;
    if (!getSimulationCollider(collider, c).ok) return false;
    if      (key == "ignite_on_contact")   c.gas_ignite_on_contact = value > 0.5f;
    else if (key == "override_ignition")   c.msf_override_ignition = value > 0.5f;
    else if (key == "ignition_kelvin")     c.msf_ignition_kelvin = value;
    else if (key == "burn_rate_scale")     c.msf_burn_rate_scale = value;
    else if (key == "fuel_capacity_scale") c.msf_fuel_capacity_scale = value;
    else if (key == "melt_flow")           c.msf_melt_flow_enabled = value > 0.5f;
    else if (key == "melt_height_loss")    c.msf_melt_height_loss = value;
    else if (key == "melt_spread")         c.msf_melt_spread = value;
    else return false;
    return updateSimulationCollider(collider, c).ok;
}

bool writeSurfaceText(const std::string& object, const std::string& key,
                      const std::string& text) {
    const std::string collider = colliderNameForIdentity(object);
    SimulationColliderInfo c;
    if (!getSimulationCollider(collider, c).ok) return false;
    if (key != "substance") return false;
    c.msf_substance = text;
    // ★ An unknown substance name is REFUSED by updateSimulationCollider rather
    // than silently falling back, so a typo in a graph does not quietly turn a
    // steel beam into oak.
    return updateSimulationCollider(collider, c).ok;
}

// Authored values held for surface overrides. Text and float are separate
// tables because a substance NAME has no numeric identity to fall back on.
std::unordered_map<OverrideKey, float, OverrideKeyHash> g_authored_surface;
std::unordered_map<OverrideKey, std::string, OverrideKeyHash> g_authored_surface_text;

// ── Emitter (flow source) parameters ────────────────────────────────────────
//
// ★★ Read-modify-write against the LIVE source, never a freshly built one.
// SimulationFlowSourceInfo carries ~25 authored fields; constructing one to set
// a single value would reset every other field to its default, and the symptom
// would be "changing the rate also moved the emitter and dropped its substance".
bool readEmitterParameter(const std::string& emitter, const std::string& key,
                          float& out) {
    SimulationFlowSourceInfo src;
    if (!getSimulationFlowSource(emitter, src).ok) return false;
    if      (key == "enabled")                    { out = src.enabled ? 1.0f : 0.0f; return true; }
    else if (key == "radius")                     { out = src.radius; return true; }
    else if (key == "density")                    { out = src.density; return true; }
    else if (key == "temperature")                { out = src.temperature; return true; }
    else if (key == "fuel")                       { out = src.fuel; return true; }
    else if (key == "falloff")                    { out = src.falloff; return true; }
    else if (key == "velocity_coupling")          { out = src.velocity_coupling; return true; }
    else if (key == "inherit_velocity")           { out = src.inherit_velocity; return true; }
    else if (key == "fluid_particles_per_second") { out = src.fluid_particles_per_second; return true; }
    else if (key == "fluid_velocity_spread")      { out = src.fluid_velocity_spread; return true; }
    return false;                       // unknown key: say so, do not guess
}

bool writeEmitterParameter(const std::string& emitter, const std::string& key,
                           float value) {
    SimulationFlowSourceInfo src;
    if (!getSimulationFlowSource(emitter, src).ok) return false;
    if      (key == "enabled")                    src.enabled = value > 0.5f;
    else if (key == "radius")                     src.radius = value;
    else if (key == "density")                    src.density = value;
    else if (key == "temperature")                src.temperature = value;
    else if (key == "fuel")                       src.fuel = value;
    else if (key == "falloff")                    src.falloff = value;
    else if (key == "velocity_coupling")          src.velocity_coupling = value;
    else if (key == "inherit_velocity")           src.inherit_velocity = value;
    else if (key == "fluid_particles_per_second") src.fluid_particles_per_second = value;
    else if (key == "fluid_velocity_spread")      src.fluid_velocity_spread = value;
    // ★★★ `domain` is deliberately absent. Which region a source feeds resolves
    // an ambiguity (an object inside two overlapping domains has no geometric
    // answer) and is authored on the source. A graph silently rebinding it would
    // move emission to another domain with nothing on screen to say so.
    else return false;
    return updateSimulationFlowSource(emitter, src).ok;
}

bool readEmitterText(const std::string& emitter, const std::string& key,
                     std::string& out) {
    SimulationFlowSourceInfo src;
    if (!getSimulationFlowSource(emitter, src).ok) return false;
    if (key != "fluid_substance") return false;
    out = src.fluid_substance;
    return true;
}

bool writeEmitterText(const std::string& emitter, const std::string& key,
                      const std::string& text) {
    SimulationFlowSourceInfo src;
    if (!getSimulationFlowSource(emitter, src).ok) return false;
    if (key != "fluid_substance") return false;
    src.fluid_substance = text;
    return updateSimulationFlowSource(emitter, src).ok;
}

std::unordered_map<OverrideKey, float, OverrideKeyHash> g_authored_emitter;
std::unordered_map<OverrideKey, std::string, OverrideKeyHash> g_authored_emitter_text;

// ── N7: render binding ──────────────────────────────────────────────────────
//
// Binds EXISTING look parameters — the SDF surface material, the splat material
// and the gas volume shader settings. Reversible like every other override.
bool readRenderText(const std::string& domain, const std::string& key,
                    std::string& out) {
    if (key == "surface_material" || key == "splat_material") {
        std::vector<FluidDomainInfo> domains;
        if (!listFluidDomains(domains).ok) return false;
        for (const auto& d : domains) {
            if (d.name != domain) continue;
            out = (key == "surface_material") ? d.surface_material : d.splat_material;
            return true;
        }
        return false;
    }
    if (key == "volume_preset") {
        GasShaderSettings s;
        if (!getGasShaderSettings(domain, s).ok) return false;
        out = s.preset;
        return true;
    }
    return false;
}

bool readRenderParameter(const std::string& domain, const std::string& key,
                         float& out) {
    GasShaderSettings s;
    if (!getGasShaderSettings(domain, s).ok) return false;
    if (key == "density_multiplier") { out = s.density_multiplier; return true; }
    if (key == "density_cutoff")     { out = s.density_cutoff;     return true; }
    if (key == "temperature_min")    { out = s.temperature_min;    return true; }
    if (key == "temperature_max")    { out = s.temperature_max;    return true; }
    if (key == "scattering")         { out = s.scattering_coefficient; return true; }
    if (key == "absorption")         { out = s.absorption_coefficient; return true; }
    return false;
}

bool writeRenderText(const std::string& domain, const std::string& key,
                     const std::string& text) {
    if (key == "surface_material" || key == "splat_material") {
        const std::string* surface = (key == "surface_material") ? &text : nullptr;
        // ★ splat_material has no slot in updateFluidDomain's argument list, so
        // only the surface material can be bound here. Reported as a failure
        // rather than silently ignored — a graph edit that does nothing while
        // claiming success is worse than one that refuses.
        if (!surface) return false;
        return updateFluidDomain(
            domain,
            /* domain_min                     */ nullptr,
            /* domain_max                     */ nullptr,
            /* voxel_size                     */ nullptr,
            /* render_mode                    */ nullptr,
            /* backend                        */ nullptr,
            /* boundary                       */ nullptr,
            /* preset                         */ nullptr,
            /* kinematic_viscosity            */ nullptr,
            /* viscosity_sweeps               */ nullptr,
            /* viscosity_wall_slip            */ nullptr,
            /* surface_material               */ surface).ok;
    }
    if (key == "volume_preset") {
        GasShaderSettings s;
        if (!getGasShaderSettings(domain, s).ok) return false;
        s.preset = text;
        return updateGasShaderSettings(domain, s).ok;
    }
    return false;
}

bool writeRenderParameter(const std::string& domain, const std::string& key,
                          float value) {
    GasShaderSettings s;
    if (!getGasShaderSettings(domain, s).ok) return false;
    if      (key == "density_multiplier") s.density_multiplier = value;
    else if (key == "density_cutoff")     s.density_cutoff = value;
    else if (key == "temperature_min")    s.temperature_min = value;
    else if (key == "temperature_max")    s.temperature_max = value;
    else if (key == "scattering")         s.scattering_coefficient = value;
    else if (key == "absorption")         s.absorption_coefficient = value;
    else return false;
    return updateGasShaderSettings(domain, s).ok;
}

std::unordered_map<OverrideKey, float, OverrideKeyHash> g_authored_render;
std::unordered_map<OverrideKey, std::string, OverrideKeyHash> g_authored_render_text;

// ★★★ THE GAS SHADER IS CAPTURED WHOLE, PER DOMAIN — not key by key.
//
// A preset write is DESTRUCTIVE: it replaces the shader with a pristine recipe.
// So capturing only the preset NAME cannot undo it. The authored state here was
// "smoke, with scattering hand-tuned to 0.15"; restoring the name reinstalled
// pristine smoke and the tuning was gone, while clear_overrides reported
// success. Measured 2026-08-17, twice — the first fix (restore order) moved the
// wrong number without curing the cause.
//
// Per-key capture has a second, subtler failure: the first write of `scattering`
// may happen while a DIFFERENT preset is installed, so its "authored" value
// belongs to a recipe the restore is about to replace. Replaying it then
// corrupts the recipe it was just restored into.
//
// One snapshot of the whole struct, taken before the first write of any gas
// shader key, restores exactly and has neither problem.
bool isGasShaderKey(const std::string& key) {
    return key == "volume_preset" || key == "density_multiplier" ||
           key == "density_cutoff" || key == "temperature_min" ||
           key == "temperature_max" || key == "scattering" ||
           key == "absorption";
}
std::unordered_map<std::string, GasShaderSettings> g_authored_gas_shader;

bool captureGasShader(const std::string& domain) {
    if (g_authored_gas_shader.count(domain)) return true;
    GasShaderSettings s;
    if (!getGasShaderSettings(domain, s).ok) return false;
    g_authored_gas_shader.emplace(domain, s);
    return true;
}

bool restoreGasShader(const std::string& domain, const GasShaderSettings& s) {
    // Two calls, and both are needed: the first reinstalls the recipe (which
    // returns early, discarding values by design), the second puts the authored
    // numbers back on top of it.
    GasShaderSettings recipe = s;
    if (!updateGasShaderSettings(domain, recipe).ok) return false;
    GasShaderSettings values = s;
    values.preset.clear();          // empty = "no recipe change, apply values"
    return updateGasShaderSettings(domain, values).ok;
}

// ── N4: couplings ───────────────────────────────────────────────────────────
//
// ★★★ A coupling node DECLARES; it does not schedule. The order couplings run
// in is decided by stepGridDomains, and the solver reports what it ran
// (ParticleSimulationSystem::couplingTrace). Comparing the two is the whole
// value of this phase — a graph that displayed a user-chosen order while the
// solver ran another would look like control and be a lie.
//
// What a coupling node DOES write is the on/off switch that already exists
// behind the coupling, and it writes it through the same reversible capture as
// N3: coupling state is configuration, and configuration stays resettable.
struct CouplingAuthored {
    bool  known = false;
    bool  value = false;
};
std::unordered_map<OverrideKey, CouplingAuthored, OverrideKeyHash> g_authored_couplings;

// Which authored switch each coupling id maps onto. `nullptr` means the
// coupling is observable but has no script-writable switch yet — reported as
// such rather than silently doing nothing.
bool readCouplingSwitch(const std::string& domain, const std::string& coupling,
                        bool& out, std::string& why_not) {
    if (coupling == "fluid_to_gas" || coupling == "gas_to_fluid_ignition") {
        CombustibleFluidSettings s;
        if (!getCombustibleFluidSettings(domain, s).ok) {
            why_not = "no combustible-fluid settings on '" + domain + "'";
            return false;
        }
        out = (coupling == "fluid_to_gas") ? s.enabled : s.auto_ignite;
        return true;
    }
    if (coupling == "foam_from_fluid") {
        // ★ Honest gap, not a silent no-op: foam parameters have no scripting
        // surface yet, so this coupling can be OBSERVED in the trace but not
        // switched from a graph. Saying so is the difference between a known
        // limit and a mystery.
        why_not = "foam has no script-writable settings yet; the coupling can be "
                  "observed but not switched";
        return false;
    }
    why_not = "unknown coupling '" + coupling + "'";
    return false;
}

bool writeCouplingSwitch(const std::string& domain, const std::string& coupling,
                         bool value, std::string& why_not) {
    if (coupling == "fluid_to_gas" || coupling == "gas_to_fluid_ignition") {
        CombustibleFluidSettings s;
        if (!getCombustibleFluidSettings(domain, s).ok) {
            why_not = "no combustible-fluid settings on '" + domain + "'";
            return false;
        }
        if (coupling == "fluid_to_gas") s.enabled = value;
        else                            s.auto_ignite = value;
        if (!updateCombustibleFluidSettings(domain, s).ok) {
            why_not = "cannot write combustible-fluid settings on '" + domain + "'";
            return false;
        }
        return true;
    }
    return readCouplingSwitch(domain, coupling, value, why_not);  // fills why_not
}

} // namespace

SimApplyResult simGraphApply(const std::string& scope_text, const std::string& owner,
                             bool allow_restart) {
    SimApplyResult out;
    const SimGraphEvaluation evaluation = simGraphEvaluate(scope_text, owner);
    // ★ A graph that could not be found applied nothing, and must not read as a
    // successful apply of zero commands.
    if (!evaluation.evaluated) {
        out.failed.push_back(evaluation.error.empty() ? "graph not found"
                                                      : evaluation.error);
        return out;
    }

    for (const auto& cmd : evaluation.commands) {
        if (cmd.kind != "set_parameter") continue;
        // ★ World commands carry no target — there is exactly one world — so
        // they are told apart from a domain command by scope, not by an empty
        // target string (which would otherwise read as a missing domain name).
        const bool is_world = (cmd.scope == "world");
        const bool needs_restart =
            NodeSystem::Sim::SetParameterNode::keyRequiresRestart(cmd.key);
        if (needs_restart && !allow_restart) {
            // ★ Refused, and REPORTED. A graph edit must not silently discard a
            // running simulation; the caller has to ask for it.
            out.refused.push_back(cmd.key + " on '" + cmd.target +
                                  "' requires a simulation restart");
            continue;
        }
        // cmd.target is empty for a world command, which is exactly the key
        // this table already uses for "the world" — no real domain has an
        // empty name, so there is no collision.
        OverrideKey ok{cmd.target, cmd.key};
        if (is_world) {
            if (g_authored_values.find(ok) == g_authored_values.end()) {
                float authored = 0.0f;
                if (!readWorldThermalParameter(cmd.key, authored)) {
                    out.failed.push_back("cannot read authored value for " + cmd.key + " on the world");
                    continue;
                }
                g_authored_values.emplace(ok, authored);
            }
            if (!writeWorldThermalParameter(cmd.key, cmd.value)) {
                out.failed.push_back("cannot apply " + cmd.key + " on the world");
                continue;
            }
            ++out.applied;
            continue;
        }
        if (g_authored_values.find(ok) == g_authored_values.end()) {
            float authored = 0.0f;
            if (!readParameter(cmd.target, cmd.key, authored)) {
                out.failed.push_back("cannot read authored value for " + cmd.key +
                                     " on '" + cmd.target + "'");
                continue;
            }
            g_authored_values.emplace(ok, authored);   // capture BEFORE writing
        }
        if (!writeParameter(cmd.target, cmd.key, cmd.value)) {
            out.failed.push_back("cannot apply " + cmd.key + " on '" + cmd.target + "'");
            continue;
        }
        ++out.applied;
    }

    // Per-object surface settings (N5). Same capture-before-write contract.
    for (const auto& cmd : evaluation.commands) {
        if (cmd.kind != "set_surface") continue;
        OverrideKey ok{cmd.target, cmd.key};
        if (!cmd.text.empty()) {
            const bool text_inserted =
                g_authored_surface_text.find(ok) == g_authored_surface_text.end();
            if (text_inserted) {
                std::string authored;
                if (!readSurfaceText(cmd.target, cmd.key, authored)) {
                    out.failed.push_back("cannot read authored " + cmd.key +
                                         " on object '" + cmd.target + "'");
                    continue;
                }
                g_authored_surface_text.emplace(ok, authored);
            }
            if (!writeSurfaceText(cmd.target, cmd.key, cmd.text)) {
                // Rolled back for the same reason as the render path: a capture
                // for a key that was never written makes clear_overrides fail
                // forever.
                if (text_inserted) g_authored_surface_text.erase(ok);
                out.failed.push_back("cannot apply " + cmd.key + "='" + cmd.text +
                                     "' on object '" + cmd.target + "'");
                continue;
            }
            ++out.applied;
            continue;
        }
        const bool inserted =
            g_authored_surface.find(ok) == g_authored_surface.end();
        if (inserted) {
            float authored = 0.0f;
            if (!readSurfaceParameter(cmd.target, cmd.key, authored)) {
                out.failed.push_back("cannot read authored " + cmd.key +
                                     " on object '" + cmd.target + "'");
                continue;
            }
            g_authored_surface.emplace(ok, authored);
        }
        if (!writeSurfaceParameter(cmd.target, cmd.key, cmd.value)) {
            if (inserted) g_authored_surface.erase(ok);
            out.failed.push_back("cannot apply " + cmd.key + " on object '" +
                                 cmd.target + "'");
            continue;
        }
        ++out.applied;
    }

    // Emitter properties. Same capture-before-write contract.
    for (const auto& cmd : evaluation.commands) {
        if (cmd.kind != "set_emitter") continue;
        OverrideKey ok{cmd.target, cmd.key};
        if (cmd.key == "fluid_substance") {
            const bool inserted =
                g_authored_emitter_text.find(ok) == g_authored_emitter_text.end();
            if (inserted) {
                std::string authored;
                if (!readEmitterText(cmd.target, cmd.key, authored)) {
                    out.failed.push_back("cannot read authored " + cmd.key +
                                         " on emitter '" + cmd.target + "'");
                    continue;
                }
                g_authored_emitter_text.emplace(ok, authored);
            }
            if (!writeEmitterText(cmd.target, cmd.key, cmd.text)) {
                // ★ Roll the capture back. A captured key that was never written
                // makes clear_overrides try to restore a value it never
                // replaced, and that failure would repeat forever.
                if (inserted) g_authored_emitter_text.erase(ok);
                out.failed.push_back("cannot apply " + cmd.key + "='" + cmd.text +
                                     "' on emitter '" + cmd.target + "'");
                continue;
            }
            ++out.applied;
            continue;
        }
        const bool inserted = g_authored_emitter.find(ok) == g_authored_emitter.end();
        if (inserted) {
            float authored = 0.0f;
            if (!readEmitterParameter(cmd.target, cmd.key, authored)) {
                out.failed.push_back("cannot read authored " + cmd.key +
                                     " on emitter '" + cmd.target + "'");
                continue;
            }
            g_authored_emitter.emplace(ok, authored);
        }
        if (!writeEmitterParameter(cmd.target, cmd.key, cmd.value)) {
            if (inserted) g_authored_emitter.erase(ok);
            out.failed.push_back("cannot apply " + cmd.key + " on emitter '" +
                                 cmd.target + "'");
            continue;
        }
        ++out.applied;
    }

    // Render binding (N7). Same capture-before-write contract again.
    for (const auto& cmd : evaluation.commands) {
        if (cmd.kind != "set_render") continue;
        OverrideKey ok{cmd.target, cmd.key};
        // Gas shader keys share ONE per-domain snapshot; see the note above.
        if (isGasShaderKey(cmd.key) && !captureGasShader(cmd.target)) {
            out.failed.push_back("cannot read the authored gas shader on '" +
                                 cmd.target + "'");
            continue;
        }
        const bool is_text = cmd.key == "surface_material" ||
                             cmd.key == "splat_material" ||
                             cmd.key == "volume_preset";
        if (is_text && isGasShaderKey(cmd.key)) {
            if (!writeRenderText(cmd.target, cmd.key, cmd.text)) {
                out.failed.push_back("cannot apply " + cmd.key + " on '" +
                                     cmd.target + "'");
                continue;
            }
            ++out.applied;
            continue;
        }
        if (!is_text && isGasShaderKey(cmd.key)) {
            if (!writeRenderParameter(cmd.target, cmd.key, cmd.value)) {
                out.failed.push_back("cannot apply " + cmd.key + " on '" +
                                     cmd.target + "'");
                continue;
            }
            ++out.applied;
            continue;
        }
        if (is_text) {
            // ★★★ CAPTURE IS ROLLED BACK WHEN THE WRITE FAILS. Reading first is
            // the contract, but KEEPING a capture for a key that was never
            // written poisons clear_overrides forever: it will try to restore a
            // value nothing changed, through a writer that cannot write it, and
            // fail every time. Measured 2026-08-17 with `splat_material`, which
            // has no slot in updateFluidDomain — the graph applied cleanly and
            // then clear_overrides threw on every later call.
            const bool inserted =
                g_authored_render_text.find(ok) == g_authored_render_text.end();
            if (inserted) {
                std::string authored;
                if (!readRenderText(cmd.target, cmd.key, authored)) {
                    out.failed.push_back("cannot read authored " + cmd.key +
                                         " on '" + cmd.target + "'");
                    continue;
                }
                g_authored_render_text.emplace(ok, authored);
            }
            if (!writeRenderText(cmd.target, cmd.key, cmd.text)) {
                if (inserted) g_authored_render_text.erase(ok);
                out.failed.push_back("cannot apply " + cmd.key + " on '" +
                                     cmd.target + "'");
                continue;
            }
            ++out.applied;
            continue;
        }
        const bool inserted =
            g_authored_render.find(ok) == g_authored_render.end();
        if (inserted) {
            float authored = 0.0f;
            if (!readRenderParameter(cmd.target, cmd.key, authored)) {
                out.failed.push_back("cannot read authored " + cmd.key +
                                     " on '" + cmd.target + "'");
                continue;
            }
            g_authored_render.emplace(ok, authored);
        }
        if (!writeRenderParameter(cmd.target, cmd.key, cmd.value)) {
            if (inserted) g_authored_render.erase(ok);
            out.failed.push_back("cannot apply " + cmd.key + " on '" +
                                 cmd.target + "'");
            continue;
        }
        ++out.applied;
    }

    // Couplings, applied after parameters so a chain that configures a coupling
    // and then switches it on lands in that order.
    for (const auto& cmd : evaluation.commands) {
        if (cmd.kind != "couple") continue;
        const bool want = cmd.value > 0.5f;
        OverrideKey ok{cmd.target, cmd.key};
        if (g_authored_couplings.find(ok) == g_authored_couplings.end()) {
            bool authored = false;
            std::string why_not;
            if (!readCouplingSwitch(cmd.target, cmd.key, authored, why_not)) {
                out.failed.push_back(why_not);
                continue;
            }
            g_authored_couplings.emplace(ok, CouplingAuthored{true, authored});
        }
        std::string why_not;
        if (!writeCouplingSwitch(cmd.target, cmd.key, want, why_not)) {
            out.failed.push_back(why_not);
            continue;
        }
        ++out.applied;
    }

    out.overrides_held = static_cast<uint32_t>(g_authored_values.size() +
                                               g_authored_couplings.size() +
                                               g_authored_surface.size() +
                                               g_authored_surface_text.size() +
                                               g_authored_render.size() +
                                               g_authored_render_text.size() +
                                               g_authored_gas_shader.size() +
                                 g_authored_emitter.size() +
                                 g_authored_emitter_text.size());
    out.ok = out.failed.empty();
    return out;
}

Result simGraphClearOverrides() {
    std::string first_error;
    for (const auto& entry : g_authored_values) {
        // Empty domain is the world command's own key (see simGraphApply) —
        // not a domain nobody named.
        const bool is_world = entry.first.domain.empty();
        const bool restored = is_world
            ? writeWorldThermalParameter(entry.first.key, entry.second)
            : writeParameter(entry.first.domain, entry.first.key, entry.second);
        if (!restored && first_error.empty()) {
            first_error = is_world
                ? "cannot restore " + entry.first.key + " on the world"
                : "cannot restore " + entry.first.key + " on '" + entry.first.domain + "'";
        }
    }
    for (const auto& entry : g_authored_couplings) {
        std::string why_not;
        if (!writeCouplingSwitch(entry.first.domain, entry.first.key,
                                 entry.second.value, why_not) &&
            first_error.empty()) {
            first_error = why_not;
        }
    }
    // Text before numbers here too. A substance does not currently reset the
    // per-object scales the way a volume preset resets its values, but the rule
    // is the same one and keeping both restore paths in the same order means
    // there is no second ordering to reason about later.
    for (const auto& entry : g_authored_surface_text) {
        if (!writeSurfaceText(entry.first.domain, entry.first.key,
                              entry.second) && first_error.empty()) {
            first_error = "cannot restore " + entry.first.key + " on object '" +
                          entry.first.domain + "'";
        }
    }
    for (const auto& entry : g_authored_surface) {
        if (!writeSurfaceParameter(entry.first.domain, entry.first.key,
                                   entry.second) && first_error.empty()) {
            first_error = "cannot restore " + entry.first.key + " on object '" +
                          entry.first.domain + "'";
        }
    }
    // Emitters: text then numbers, matching every other restore path. A flow
    // source substance does not currently rewrite the numeric fields, but
    // keeping one ordering everywhere means there is no second rule to reason
    // about the next time a value turns out to be a recipe.
    for (const auto& entry : g_authored_emitter_text) {
        if (!writeEmitterText(entry.first.domain, entry.first.key,
                              entry.second) && first_error.empty()) {
            first_error = "cannot restore " + entry.first.key + " on emitter '" +
                          entry.first.domain + "'";
        }
    }
    for (const auto& entry : g_authored_emitter) {
        if (!writeEmitterParameter(entry.first.domain, entry.first.key,
                                   entry.second) && first_error.empty()) {
            first_error = "cannot restore " + entry.first.key + " on emitter '" +
                          entry.first.domain + "'";
        }
    }
    // ★★★ TEXT FIRST, THEN NUMBERS — restoring in the other order silently
    // loses the numbers. A volume preset is a RECIPE: putting "smoke" back
    // reinstalls smoke's pristine values, so any numeric restore that ran before
    // it is wiped. Measured 2026-08-17: scattering came back as the preset's 2.0
    // instead of the authored 0.15, and clear_overrides reported success.
    //
    // This mirrors the apply path, where the node emits the preset before the
    // values for exactly the same reason.
    for (const auto& entry : g_authored_gas_shader) {
        if (!restoreGasShader(entry.first, entry.second) && first_error.empty()) {
            first_error = "cannot restore the gas shader on '" + entry.first + "'";
        }
    }
    for (const auto& entry : g_authored_render_text) {
        if (!writeRenderText(entry.first.domain, entry.first.key,
                             entry.second) && first_error.empty()) {
            first_error = "cannot restore " + entry.first.key + " on '" +
                          entry.first.domain + "'";
        }
    }
    for (const auto& entry : g_authored_render) {
        if (!writeRenderParameter(entry.first.domain, entry.first.key,
                                  entry.second) && first_error.empty()) {
            first_error = "cannot restore " + entry.first.key + " on '" +
                          entry.first.domain + "'";
        }
    }
    // Cleared even on partial failure: keeping a half-restored table would make
    // the next clear restore values that are no longer authored.
    g_authored_values.clear();
    g_authored_couplings.clear();
    g_authored_surface.clear();
    g_authored_surface_text.clear();
    g_authored_render.clear();
    g_authored_render_text.clear();
    g_authored_gas_shader.clear();
    g_authored_emitter.clear();
    g_authored_emitter_text.clear();
    if (!first_error.empty()) return Result::fail(first_error);
    return Result::success();
}

uint32_t simGraphOverrideCount() {
    return static_cast<uint32_t>(g_authored_values.size() +
                                 g_authored_couplings.size() +
                                 g_authored_surface.size() +
                                 g_authored_surface_text.size() +
                                 g_authored_render.size() +
                                 g_authored_render_text.size() +
                                 g_authored_gas_shader.size());
}

std::vector<std::string> simListSurfaceAttributes(const std::string& object) {
    return resolveSurfaceList(object);
}

// ── N6: bake and cache state ────────────────────────────────────────────────

Result simCacheStatus(SimCacheStatus& out) {
    if (!g_ctx) return notBound();
    out = SimCacheStatus{};
    out.valid = g_ctx->scene.simCacheValid();
    out.baking = g_ctx->scene.simBakeActive();
    out.cache_dir = g_ctx->scene.simCacheDir();
    out.ram_frames = static_cast<uint32_t>(g_ctx->scene.simFrameCacheCount());
    out.has_range = g_ctx->scene.simFrameCacheRange(out.first_frame, out.last_frame);
    out.config_signature = g_ctx->scene.simConfigSignature();
    return Result::success();
}

Result simBake(const std::string& cache_dir, int start_frame, int end_frame,
               float fps) {
    if (!g_ctx) return notBound();
    if (cache_dir.empty()) return Result::fail("bake requires a cache directory");
    if (end_frame < start_frame)
        return Result::fail("bake range is inverted: end < start");
    if (fps <= 0.0f) return Result::fail("bake requires a positive fps");
    if (g_ctx->scene.simBakeActive())
        return Result::fail("a bake is already running");
    // ★ Reported as a failure rather than "succeeding" with nothing written. A
    // bake that silently produced zero frames would leave a script believing the
    // cache is warm and the next render pulling live physics instead.
    if (!g_ctx->scene.bakeSimulationToDisk(cache_dir, start_frame, end_frame, fps))
        return Result::fail("bake did not complete; nothing was cached");
    return Result::success();
}

Result simClearCache() {
    if (!g_ctx) return notBound();
    // Same call the panel's "Clear the bake cache" makes: drops the RAM timeline
    // cache and unbinds the disk bake, returning to free-run preview.
    g_ctx->scene.invalidateRigidBodySimulationCache();
    return Result::success();
}

SimCouplingReport simGraphCouplings() {
    SimCouplingReport out;

    // Declared: every graph's Couple commands, in dependency order.
    //
    // ★★★ This report spans ALL scopes on purpose — it is the answer to "I can't
    // see the whole simulation on one canvas" (section 4.3). A coupling joins two
    // domains, so it belongs to neither graph alone; the overview is a REPORT,
    // never a fourth editing surface. Scoping this to one graph would hide
    // exactly the cross-domain declarations it exists to show.
    for (const auto& ref : simGraphList()) {
        const SimGraphEvaluation evaluation = simGraphEvaluate(ref.scope, ref.owner);
        if (!evaluation.evaluated) continue;
        for (const auto& cmd : evaluation.commands) {
            if (cmd.kind != "couple") continue;
            SimCouplingEntry e;
            e.coupling = cmd.key;
            e.source_domain = cmd.target;
            e.target_domain = cmd.text;
            e.active = cmd.value > 0.5f;
            e.source_node = cmd.source_node;
            e.scope = ref.scope;
            e.owner = ref.owner;
            out.declared.push_back(std::move(e));
        }
    }

    // Actual: what the solver RAN in the last step, in execution order.
    //
    // ★★★ This is measured, not mirrored. An empty list is NOT "no couplings" —
    // it also happens when the sim has not stepped since the app started, and
    // treating the two as the same would report a healthy scene as broken.
    // `traced` says which of the two it is.
    if (g_ctx) {
        for (auto& system : g_ctx->scene.particle_systems) {
            if (!system.runtime) continue;
            for (const auto& t : system.runtime->couplingTrace()) {
                SimCouplingEntry e;
                e.coupling = t.name ? t.name : "";
                e.producer = t.producer ? t.producer : "";
                e.consumer = t.consumer ? t.consumer : "";
                e.source_domain = t.source_domain;
                e.target_domain = t.target_domain;
                e.active = true;               // it ran; that is the claim
                out.actual.push_back(std::move(e));
            }
            out.traced = true;
        }
    }

    // Cross-check. Two independent facts, kept apart on purpose:
    //
    //   declared_not_running — the graph asks for a coupling the solver never
    //       reached. Either it is switched off downstream, or the two ends do
    //       not overlap, or the apply never happened.
    //   running_not_declared — the solver runs a coupling no node mentions. NOT
    //       an error: the panels still author couplings directly. It matters
    //       because a user reading only the graph would not know it was there.
    for (const auto& d : out.declared) {
        if (!d.active) continue;
        bool found = false;
        for (const auto& a : out.actual) {
            if (a.coupling == d.coupling &&
                (d.source_domain.empty() || a.source_domain == d.source_domain)) {
                found = true;
                break;
            }
        }
        if (!found) out.declared_not_running.push_back(d.coupling + " on '" +
                                                       d.source_domain + "'");
    }
    for (const auto& a : out.actual) {
        bool found = false;
        for (const auto& d : out.declared) {
            if (d.coupling == a.coupling) { found = true; break; }
        }
        if (!found) out.running_not_declared.push_back(a.coupling + " on '" +
                                                       a.source_domain + "'");
    }

    // ★★ ORDER, which is the reason this phase exists. Compared only over the
    // couplings that appear in BOTH lists: a declaration the solver never ran
    // has no position to disagree about, and folding that into the order check
    // would report an ordering problem where the real problem is absence.
    std::vector<std::string> declared_order, actual_order;
    for (const auto& d : out.declared) {
        if (!d.active) continue;
        for (const auto& a : out.actual) {
            if (a.coupling == d.coupling) { declared_order.push_back(d.coupling); break; }
        }
    }
    for (const auto& a : out.actual) {
        for (const auto& d : out.declared) {
            if (d.active && d.coupling == a.coupling) {
                actual_order.push_back(a.coupling);
                break;
            }
        }
    }
    out.order_matches = (declared_order == actual_order);

    return out;
}

// ── Measured interaction: which forces/colliders geometrically reach a domain
//
// Decision record: SIMULATION_NODE_OBJECT_MODEL.md section 9.6 stage 3.
//
// ★ Bounding-volume overlap, not exact clipping. `ForceFieldInfo` and
// `SimulationColliderInfo` are read-only info structs with no mesh data, so an
// axis-aligned box is the honest limit of what they can answer -- an `obb`
// collider's box is measured WITHOUT its rotation, which can only
// over-report an intersection near a domain edge, never hide a real one.
// mesh_sdf/convex/mesh_bvh colliders carry no box at all here and are
// reported `measurable=false` rather than guessed as non-intersecting -- the
// same "false, NOT zeros" rule as resolveAttributeStats above.
bool simAabbOverlap(const Vec3& a_min, const Vec3& a_max,
                    const Vec3& b_min, const Vec3& b_max) {
    return a_min.x <= b_max.x && a_max.x >= b_min.x &&
           a_min.y <= b_max.y && a_max.y >= b_min.y &&
           a_min.z <= b_max.z && a_max.z >= b_min.z;
}

Result simDomainIntersections(const std::string& domain,
                              std::vector<SimIntersectionEntry>& out) {
    out.clear();
    std::vector<FluidDomainInfo> domains;
    if (!listFluidDomains(domains).ok) return Result::fail("could not list domains");
    const FluidDomainInfo* dom = nullptr;
    for (const auto& d : domains) if (d.name == domain) { dom = &d; break; }
    if (!dom) return Result::fail("no such domain: '" + domain + "'");
    const Vec3 d_min = dom->domain_min, d_max = dom->domain_max;

    for (const auto& f : listForceFields()) {
        SimIntersectionEntry e;
        e.name = f.name;
        e.kind = "force";
        e.measurable = true;
        if (f.shape == "infinite") {
            e.intersects = true;   // unbounded by definition -- reaches every domain
        } else {
            const float r = std::max(f.falloff_radius, 0.0f);
            const Vec3 f_min(f.position.x - r, f.position.y - r, f.position.z - r);
            const Vec3 f_max(f.position.x + r, f.position.y + r, f.position.z + r);
            e.intersects = simAabbOverlap(d_min, d_max, f_min, f_max);
        }
        out.push_back(std::move(e));
    }

    std::vector<SimulationColliderInfo> colliders;
    if (listSimulationColliders(colliders).ok) {
        for (const auto& c : colliders) {
            SimIntersectionEntry e;
            e.name = c.name;
            e.kind = "collider";
            if (c.source_mode == "plane") {
                // An infinite plane in X/Z: only the Y band matters, and this is
                // exact, not an approximation.
                e.measurable = true;
                e.intersects = c.plane_y >= d_min.y && c.plane_y <= d_max.y;
            } else if (c.source_mode == "sphere") {
                const float r = std::max(c.sphere_radius, 0.0f);
                const Vec3 c_min(c.sphere_center.x - r, c.sphere_center.y - r, c.sphere_center.z - r);
                const Vec3 c_max(c.sphere_center.x + r, c.sphere_center.y + r, c.sphere_center.z + r);
                e.measurable = true;
                e.intersects = simAabbOverlap(d_min, d_max, c_min, c_max);
            } else if (c.source_mode == "capsule") {
                const float r = std::max(c.capsule_radius, 0.0f);
                const Vec3 c_min(std::min(c.capsule_start.x, c.capsule_end.x) - r,
                                 std::min(c.capsule_start.y, c.capsule_end.y) - r,
                                 std::min(c.capsule_start.z, c.capsule_end.z) - r);
                const Vec3 c_max(std::max(c.capsule_start.x, c.capsule_end.x) + r,
                                 std::max(c.capsule_start.y, c.capsule_end.y) + r,
                                 std::max(c.capsule_start.z, c.capsule_end.z) + r);
                e.measurable = true;
                e.intersects = simAabbOverlap(d_min, d_max, c_min, c_max);
            } else if (c.source_mode == "aabb" || c.source_mode == "obb") {
                e.measurable = true;
                e.intersects = simAabbOverlap(d_min, d_max, c.bounds_min, c.bounds_max);
            } else {
                // mesh_sdf | convex | mesh_bvh
                e.measurable = false;
                e.intersects = false;
            }
            out.push_back(std::move(e));
        }
    }
    return Result::success();
}

void initSimulationNodes() {
    NodeSystem::Sim::registerSimulationNodeTypes();
    NodeSystem::Sim::setAttributeStatsResolver(&resolveAttributeStats);
    NodeSystem::Sim::setAttributeListResolver(&resolveAttributeList);
    NodeSystem::Sim::setSurfaceStatsResolver(&resolveSurfaceStats);
    NodeSystem::Sim::setSurfaceListResolver(&resolveSurfaceList);
    NodeSystem::Sim::setCacheStatusResolver(&resolveCacheStatus);
}

std::vector<std::string> simListAttributes(const std::string& domain) {
    return resolveAttributeList(domain);
}

// ── Unified attribute discovery + measurement (rt.attr, section 9.5-9.6) ────
//
// ★ Merges the domain/object halves the naming layer split above (kept split
// there only because that is how the Field/Surface Inspect node pickers
// happen to consume them), and adds the half that never had a caller-facing
// path: measuring a named attribute required placing an Inspect node in a
// graph and evaluating it. This calls the same resolvers directly.
std::vector<std::string> listAttributes(const std::string& scope, const std::string& id) {
    if (scope == "domain") return resolveAttributeList(id);
    if (scope == "object") return resolveSurfaceList(id);
    if (scope == "world")
        return {"ambient_kelvin", "kelvin_per_unit", "convection_coefficient",
                "oxygen_availability"};
    return {};
}

Result getAttributeStats(const std::string& scope, const std::string& id,
                         const std::string& name, AttrStatsInfo& out) {
    out = AttrStatsInfo{};
    // ★ "Could not measure this attribute" is a VALUE (`available=false`), not
    // an error -- the same contract as `stats_available` on a Field/Surface
    // Inspect node, which never throws for an unknown channel either. Only an
    // unsupported SCOPE string is a genuine caller mistake and fails loudly;
    // an unmeasurable attribute inside a valid scope must round-trip through
    // rt.attr.stats() as a reading, not an exception, or a caller that probes
    // for "does this attribute exist yet" would have to wrap every call in a
    // try/except to ask a question this field already answers.
    if (scope == "domain") {
        NodeSystem::Sim::FieldStats s;
        if (resolveAttributeStats(id, name, s)) {
            out.available = true; out.count = s.count; out.array_size = s.array_size;
            out.in_sync = s.in_sync; out.min_value = s.min_value;
            out.max_value = s.max_value; out.mean_value = s.mean_value;
        }
        return Result::success();
    }
    if (scope == "object") {
        NodeSystem::Sim::FieldStats s;
        if (resolveSurfaceStats(id, name, s)) {
            out.available = true; out.count = s.count; out.array_size = s.array_size;
            out.in_sync = s.in_sync; out.min_value = s.min_value;
            out.max_value = s.max_value; out.mean_value = s.mean_value;
        }
        return Result::success();
    }
    if (scope == "world") {
        // ★ A world thermal field is a single scalar, not a population -- min,
        // max and mean all read the same value on purpose, so a caller that
        // treats every rt.attr.stats() result uniformly gets a coherent answer
        // instead of a divide-by-zero mean over zero elements.
        float v = 0.0f;
        if (readWorldThermalParameter(name, v)) {
            out.available = true; out.count = 1; out.array_size = 1; out.in_sync = true;
            out.min_value = out.max_value = out.mean_value = static_cast<double>(v);
        }
        return Result::success();
    }
    return Result::fail("unsupported attribute scope '" + scope + "' (expected object|domain|world)");
}

// ── Graph lifecycle ─────────────────────────────────────────────────────────

std::vector<SimGraphRef> simGraphList() {
    std::vector<SimGraphRef> out;
    if (!g_ctx) return out;
    auto push = [&out](const SimGraphPtr& g) {
        if (!g) return;
        SimGraphRef ref;
        ref.scope = NodeSystem::Sim::scopeName(g->scope);
        ref.owner = g->owner;
        ref.node_count = static_cast<uint32_t>(g->nodes.size());
        ref.owner_node = g->ownerNodeId;
        ref.owner_missing = !ownerExists(g->scope, g->owner).ok;
        out.push_back(std::move(ref));
    };
    for (auto& [name, g] : g_ctx->scene.simulation_domain_graphs) push(g);
    for (auto& [name, g] : g_ctx->scene.simulation_object_graphs) push(g);
    push(g_ctx->scene.simulation_world_graph);
    std::sort(out.begin(), out.end(), [](const SimGraphRef& a, const SimGraphRef& b) {
        if (a.scope != b.scope) return a.scope < b.scope;
        return a.owner < b.owner;
    });
    return out;
}

Result simGraphCreate(const std::string& scope_text, const std::string& owner) {
    if (!g_ctx) return notBound();
    NodeSystem::Sim::GraphScope scope;
    Result parsed = parseScopeArg(scope_text, scope);
    if (!parsed.ok) return parsed;
    Result exists = ownerExists(scope, owner);
    if (!exists.ok) return exists;

    if (scope == NodeSystem::Sim::GraphScope::World) {
        if (!g_ctx->scene.simulation_world_graph)
            g_ctx->scene.simulation_world_graph =
                NodeSystem::Sim::makeScopedGraph(scope, std::string());
        return Result::success();
    }
    auto* storage = scopeStorage(*g_ctx, scope);
    if (!storage) return Result::fail("unsupported graph scope");
    auto it = storage->find(owner);
    // ★ Creating twice is not an error — it is how a caller says "make sure this
    // exists". Failing here would push every script into a look-then-create
    // dance whose two halves can disagree.
    if (it == storage->end() || !it->second)
        (*storage)[owner] = NodeSystem::Sim::makeScopedGraph(scope, owner);
    return Result::success();
}

Result simGraphDelete(const std::string& scope_text, const std::string& owner) {
    if (!g_ctx) return notBound();
    NodeSystem::Sim::GraphScope scope;
    Result parsed = parseScopeArg(scope_text, scope);
    if (!parsed.ok) return parsed;
    if (scope == NodeSystem::Sim::GraphScope::World) {
        if (!g_ctx->scene.simulation_world_graph)
            return Result::fail("no world simulation graph to delete");
        g_ctx->scene.simulation_world_graph.reset();
        return Result::success();
    }
    auto* storage = scopeStorage(*g_ctx, scope);
    if (!storage) return Result::fail("unsupported graph scope");
    if (storage->erase(owner) == 0)
        return Result::fail(std::string("no ") + NodeSystem::Sim::scopeName(scope) +
                            " graph for '" + owner + "'");
    return Result::success();
}

Result simGraphClear(const std::string& scope_text, const std::string& owner) {
    SimGraphPtr graph;
    Result found = findGraph(scope_text, owner, graph);
    if (!found.ok) return found;
    graph->clear();
    // ★ clear() empties the node list, which takes the owner node with it. The
    // graph must not come back ownerless: an empty canvas that no longer says
    // whose it is would let the next node be authored against nothing.
    graph->ownerNodeId = 0;
    NodeSystem::Sim::seedOwnerNode(*graph);
    return Result::success();
}

Result simGraphAddNode(const std::string& scope_text, const std::string& owner,
                       const std::string& type_id, uint32_t& out_id) {
    SimGraphPtr graph;
    Result found = findGraph(scope_text, owner, graph);
    if (!found.ok) return found;
    auto node = NodeSystem::NodeRegistry::instance().create(type_id);
    if (!node) return Result::fail("unknown simulation node type: " + type_id);
    NodeSystem::NodeBase* added = graph->registerNode(std::move(node));
    if (!added) return Result::fail("failed to add node: " + type_id);
    out_id = added->id;
    return Result::success();
}

Result simGraphSetNodeText(const std::string& scope_text, const std::string& owner,
                           uint32_t node_id, const std::string& key,
                           const std::string& value) {
    SimGraphPtr graph;
    Result found = findGraph(scope_text, owner, graph);
    if (!found.ok) return found;
    NodeSystem::NodeBase* node = graph->getNode(node_id);
    if (!node) return Result::fail("no simulation node with id " + std::to_string(node_id));
    // ★★ The owner node names the entity the graph BELONGS to. Letting a script
    // retarget it would leave a graph filed under one name that drives another —
    // and every later reading would agree with the wrong one.
    if (graph->isOwnerNode(node_id) && (key == "domain" || key == "object"))
        return Result::fail("node " + std::to_string(node_id) +
                            " is this graph's owner node; its target is the graph's "
                            "own scope and cannot be retargeted");
    if (auto* dom = dynamic_cast<NodeSystem::Sim::DomainRefNode*>(node)) {
        if (key == "domain") { dom->domainName = value; node->dirty = true; return Result::success(); }
    }
    if (auto* setter = dynamic_cast<NodeSystem::Sim::SetParameterNode*>(node)) {
        if (key == "key") { setter->key = value; node->dirty = true; return Result::success(); }
    }
    if (auto* obj = dynamic_cast<NodeSystem::Sim::ObjectRefNode*>(node)) {
        if (key == "object") { obj->objectName = value; node->dirty = true; return Result::success(); }
    }
    if (auto* cache = dynamic_cast<NodeSystem::Sim::CacheNode*>(node)) {
        if (key == "cache_dir") { cache->cacheDir = value; node->dirty = true; return Result::success(); }
    }
    if (auto* emitter = dynamic_cast<NodeSystem::Sim::EmitterNode*>(node)) {
        if (key == "emitter") {
            emitter->emitterName = value;
            emitter->refreshTitle();   // the canvas names the source it drives
            node->dirty = true;
            return Result::success();
        }
        if (key == "fluid_substance") {
            // ★ Setting the substance turns the override ON. An empty string is
            // a legitimate substance value, so emptiness cannot mean "unset" —
            // use fluid_substance.use = 0 to stop overriding it.
            emitter->substance = value;
            emitter->useSubstance = true;
            emitter->refreshTitle();
            node->dirty = true;
            return Result::success();
        }
    }
    if (auto* liquid = dynamic_cast<NodeSystem::Sim::LiquidMaterialNode*>(node)) {
        if (key == "surface_material") { liquid->surfaceMaterial = value; node->dirty = true; return Result::success(); }
        if (key == "splat_material")   { liquid->splatMaterial = value;   node->dirty = true; return Result::success(); }
    }
    if (auto* vol = dynamic_cast<NodeSystem::Sim::VolumeMaterialNode*>(node)) {
        if (key == "preset") { vol->preset = value; node->dirty = true; return Result::success(); }
    }
    if (auto* sub = dynamic_cast<NodeSystem::Sim::SubstanceNode*>(node)) {
        if (key == "substance") { sub->substanceName = value; node->dirty = true; return Result::success(); }
    }
    if (auto* surf = dynamic_cast<NodeSystem::Sim::SurfaceInspectNode*>(node)) {
        if (key == "channel") { surf->channel = value; node->dirty = true; return Result::success(); }
    }
    if (auto* field = dynamic_cast<NodeSystem::Sim::FieldReadNode*>(node)) {
        if (key == "channel") { field->channel = value; node->dirty = true; return Result::success(); }
        if (key == "source") {
            if (value == "grid") field->source = NodeSystem::Sim::FieldReadNode::Source::GridChannel;
            else if (value == "attribute") field->source = NodeSystem::Sim::FieldReadNode::Source::ElementAttribute;
            else return Result::fail("source must be 'grid' or 'attribute'");
            node->dirty = true;
            return Result::success();
        }
    }
    return Result::fail("node " + std::to_string(node_id) + " has no text parameter '" + key + "'");
}

Result simGraphSetNodeValue(const std::string& scope_text, const std::string& owner,
                            uint32_t node_id, const std::string& key, float value) {
    SimGraphPtr graph;
    Result found = findGraph(scope_text, owner, graph);
    if (!found.ok) return found;
    NodeSystem::NodeBase* node = graph->getNode(node_id);
    if (!node) return Result::fail("no simulation node with id " + std::to_string(node_id));
    if (auto* emitter = dynamic_cast<NodeSystem::Sim::EmitterNode*>(node)) {
        if (key == "fluid_substance.use") {
            emitter->useSubstance = value > 0.5f;
            emitter->refreshTitle();
            node->dirty = true;
            return Result::success();
        }
        const std::string suffix = ".use";
        if (key.size() > suffix.size() &&
            key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
            const std::string base = key.substr(0, key.size() - suffix.size());
            auto* f = emitter->find(base);
            if (!f) return Result::fail("node " + std::to_string(node_id) +
                                        " has no parameter '" + base + "'");
            f->use = value > 0.5f;
            emitter->refreshTitle();
            node->dirty = true;
            return Result::success();
        }
        if (auto* f = emitter->find(key)) {
            f->value = value;
            f->use = true;
            emitter->refreshTitle();
            node->dirty = true;
            return Result::success();
        }
        return Result::fail("node " + std::to_string(node_id) +
                            " has no parameter '" + key + "'");
    }
    if (auto* aspect = dynamic_cast<NodeSystem::Sim::DomainParamNodeBase*>(node)) {
        // "<key>.use" turns a field OFF again. Writing a value turns it ON,
        // because a value written into a field that stayed inert would be a
        // silent no-op — and a silent no-op is worse than a missing parameter.
        const std::string suffix = ".use";
        if (key.size() > suffix.size() &&
            key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
            const std::string base = key.substr(0, key.size() - suffix.size());
            auto* f = aspect->find(base);
            if (!f) return Result::fail("node " + std::to_string(node_id) +
                                        " has no parameter '" + base + "'");
            f->use = value > 0.5f;
            aspect->refreshTitle();
            node->dirty = true;
            return Result::success();
        }
        if (auto* f = aspect->find(key)) {
            f->value = value;
            f->use = true;
            aspect->refreshTitle();
            node->dirty = true;
            return Result::success();
        }
        return Result::fail("node " + std::to_string(node_id) +
                            " has no parameter '" + key + "'");
    }
    if (auto* world = dynamic_cast<NodeSystem::Sim::WorldThermalNode*>(node)) {
        // Same "<key>.use" / bare-key contract as DomainParamNodeBase above,
        // duplicated rather than shared because WorldThermalNode carries no
        // Domain pin and does not derive from that base — see its class
        // comment. EmitterNode duplicates the same shape for the same reason.
        const std::string suffix = ".use";
        if (key.size() > suffix.size() &&
            key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
            const std::string base = key.substr(0, key.size() - suffix.size());
            auto* f = world->find(base);
            if (!f) return Result::fail("node " + std::to_string(node_id) +
                                        " has no parameter '" + base + "'");
            f->use = value > 0.5f;
            world->refreshTitle();
            node->dirty = true;
            return Result::success();
        }
        if (auto* f = world->find(key)) {
            f->value = value;
            f->use = true;
            world->refreshTitle();
            node->dirty = true;
            return Result::success();
        }
        return Result::fail("node " + std::to_string(node_id) +
                            " has no parameter '" + key + "'");
    }
    if (auto* setter = dynamic_cast<NodeSystem::Sim::SetParameterNode*>(node)) {
        if (key == "value") { setter->value = value; node->dirty = true; return Result::success(); }
    }
    if (auto* sub = dynamic_cast<NodeSystem::Sim::SubstanceNode*>(node)) {
        if (key == "override_ignition") { sub->overrideIgnition = value > 0.5f; node->dirty = true; return Result::success(); }
        if (key == "ignition_kelvin")   { sub->ignitionKelvin = value;   node->dirty = true; return Result::success(); }
        if (key == "burn_rate_scale")   { sub->burnRateScale = value;    node->dirty = true; return Result::success(); }
        if (key == "fuel_capacity_scale") { sub->fuelCapacityScale = value; node->dirty = true; return Result::success(); }
    }
    if (auto* cache = dynamic_cast<NodeSystem::Sim::CacheNode*>(node)) {
        if (key == "start_frame") { cache->startFrame = static_cast<int>(value); node->dirty = true; return Result::success(); }
        if (key == "end_frame")   { cache->endFrame = static_cast<int>(value);   node->dirty = true; return Result::success(); }
    }
    if (auto* vol = dynamic_cast<NodeSystem::Sim::VolumeMaterialNode*>(node)) {
        if (key == "override_values")    { vol->overrideValues = value > 0.5f;  node->dirty = true; return Result::success(); }
        if (key == "density_multiplier") { vol->densityMultiplier = value; node->dirty = true; return Result::success(); }
        if (key == "density_cutoff")     { vol->densityCutoff = value;     node->dirty = true; return Result::success(); }
        if (key == "temperature_min")    { vol->temperatureMin = value;    node->dirty = true; return Result::success(); }
        if (key == "temperature_max")    { vol->temperatureMax = value;    node->dirty = true; return Result::success(); }
        if (key == "scattering")         { vol->scattering = value;        node->dirty = true; return Result::success(); }
        if (key == "absorption")         { vol->absorption = value;        node->dirty = true; return Result::success(); }
    }
    if (auto* pyro = dynamic_cast<NodeSystem::Sim::PyrolysisNode*>(node)) {
        if (key == "active") { pyro->active = value > 0.5f; node->dirty = true; return Result::success(); }
    }
    if (auto* phase = dynamic_cast<NodeSystem::Sim::PhaseChangeNode*>(node)) {
        if (key == "melt_flow")        { phase->meltFlow = value > 0.5f; node->dirty = true; return Result::success(); }
        if (key == "melt_height_loss") { phase->heightLoss = value;      node->dirty = true; return Result::success(); }
        if (key == "melt_spread")      { phase->spread = value;          node->dirty = true; return Result::success(); }
    }
    if (auto* coupling = dynamic_cast<NodeSystem::Sim::CouplingNodeBase*>(node)) {
        // `active` is a declaration, not a gain — see CouplingNodeBase. Anything
        // above zero means "couple"; there is no in-between to interpolate.
        if (key == "active") {
            coupling->active = value > 0.5f;
            node->dirty = true;
            return Result::success();
        }
    }
    return Result::fail("node " + std::to_string(node_id) +
                        " has no numeric parameter '" + key + "'");
}

Result simGraphConnect(const std::string& scope_text, const std::string& owner,
                       uint32_t from_node, int from_pin,
                       uint32_t to_node, int to_pin) {
    SimGraphPtr graph;
    Result found = findGraph(scope_text, owner, graph);
    if (!found.ok) return found;
    NodeSystem::NodeBase* src = graph->getNode(from_node);
    NodeSystem::NodeBase* dst = graph->getNode(to_node);
    if (!src) return Result::fail("no simulation node with id " + std::to_string(from_node));
    if (!dst) return Result::fail("no simulation node with id " + std::to_string(to_node));
    if (from_pin < 0 || from_pin >= static_cast<int>(src->outputs.size()))
        return Result::fail("output pin index out of range");
    if (to_pin < 0 || to_pin >= static_cast<int>(dst->inputs.size()))
        return Result::fail("input pin index out of range");
    // addLink returns 0 when the graph rejects the connection — type mismatch or
    // a cycle. ★ A cycle here is not merely invalid: the repo has already paid
    // for one as a stack overflow in the node-graph evaluator.
    if (graph->addLink(src->outputs[from_pin].id, dst->inputs[to_pin].id) == 0)
        return Result::fail("connection rejected (type mismatch or cycle)");
    return Result::success();
}

SimGraphEvaluation simGraphEvaluate(const std::string& scope_text,
                                    const std::string& owner) {
    SimGraphEvaluation out;
    SimGraphPtr found_graph;
    Result found = findGraph(scope_text, owner, found_graph);
    // ★ evaluated stays false and the error is carried, so a caller cannot read
    // an empty command list as "the graph declared nothing".
    if (!found.ok) { out.error = found.error; return out; }
    auto& graph = *found_graph;
    NodeSystem::EvaluationContext ctx(&graph);
    graph.evaluateSimulation(ctx);

    for (const auto& cmd : graph.collected) {
        SimCommandInfo info;
        switch (cmd.kind) {
            case NodeSystem::Sim::SimCommand::Kind::BindDomain:   info.kind = "bind_domain"; break;
            case NodeSystem::Sim::SimCommand::Kind::SetParameter: info.kind = "set_parameter"; break;
            case NodeSystem::Sim::SimCommand::Kind::Couple:       info.kind = "couple"; break;
            case NodeSystem::Sim::SimCommand::Kind::BindObject:   info.kind = "bind_object"; break;
            case NodeSystem::Sim::SimCommand::Kind::SetSurface:   info.kind = "set_surface"; break;
            case NodeSystem::Sim::SimCommand::Kind::SetRender:    info.kind = "set_render"; break;
            case NodeSystem::Sim::SimCommand::Kind::SetEmitter:   info.kind = "set_emitter"; break;
            default: info.kind = "none"; break;
        }
        switch (cmd.scope) {
            case NodeSystem::Sim::SimCommand::Scope::Domain: info.scope = "domain"; break;
            case NodeSystem::Sim::SimCommand::Scope::Object: info.scope = "object"; break;
            case NodeSystem::Sim::SimCommand::Scope::World:  info.scope = "world";  break;
        }
        info.target = cmd.target;
        info.key = cmd.key;
        info.value = cmd.value;
        info.text = cmd.text;
        info.source_node = cmd.sourceNode;
        out.commands.push_back(std::move(info));
    }
    // ★ Reported, never acted on. A node that needs a restart tells the caller;
    // deciding to discard a running simulation belongs to the user.
    for (const auto* node : graph.nodesRequiringRestart()) {
        SimRestartRequest req;
        req.node_id = node->id;
        req.reason = node->restartReason();
        out.restart_requests.push_back(std::move(req));
    }
    out.evaluated = true;
    return out;
}

Result simGraphNodes(const std::string& scope_text, const std::string& owner,
                     std::vector<SimNodeInfo>& out_nodes) {
    out_nodes.clear();
    std::vector<SimNodeInfo> out;
    SimGraphPtr graph;
    Result found = findGraph(scope_text, owner, graph);
    if (!found.ok) return found;
    for (const auto& node : graph->nodes) {
        SimNodeInfo info;
        info.id = node->id;
        info.is_owner_node = graph->isOwnerNode(node->id);
        if (const auto* emitter =
                dynamic_cast<const NodeSystem::Sim::EmitterNode*>(node.get())) {
            info.domain = emitter->emitterName;   // which flow source it names
            for (const auto& f : emitter->fields) {
                SimNodeField out_field;
                out_field.key = f.key;
                out_field.in_use = f.use;
                out_field.value = f.value;
                info.fields.push_back(std::move(out_field));
            }
            SimNodeField substance_field;
            substance_field.key = "fluid_substance";
            substance_field.in_use = emitter->useSubstance;
            info.fields.push_back(std::move(substance_field));
            info.channel = emitter->substance;    // the text value itself
        }
        if (const auto* aspect =
                dynamic_cast<const NodeSystem::Sim::DomainParamNodeBase*>(node.get())) {
            for (const auto& f : aspect->fields) {
                SimNodeField out_field;
                out_field.key = f.key;
                out_field.in_use = f.use;
                out_field.value = f.value;
                out_field.requires_restart =
                    NodeSystem::Sim::SetParameterNode::keyRequiresRestart(f.key);
                info.fields.push_back(std::move(out_field));
            }
        }
        if (const auto* world =
                dynamic_cast<const NodeSystem::Sim::WorldThermalNode*>(node.get())) {
            for (const auto& f : world->fields) {
                SimNodeField out_field;
                out_field.key = f.key;
                out_field.in_use = f.use;
                out_field.value = f.value;
                out_field.requires_restart =
                    NodeSystem::Sim::SetParameterNode::keyRequiresRestart(f.key);
                info.fields.push_back(std::move(out_field));
            }
        }
        info.type_id = node->getTypeId();
        info.display_name = node->metadata.displayName;
        info.enabled = node->enabled;
        info.input_count = static_cast<int>(node->inputs.size());
        info.output_count = static_cast<int>(node->outputs.size());
        if (auto* dom = dynamic_cast<NodeSystem::Sim::DomainRefNode*>(node.get()))
            info.domain = dom->domainName;
        if (auto* field = dynamic_cast<NodeSystem::Sim::FieldReadNode*>(node.get())) {
            info.channel = field->channel;
            info.source = field->source == NodeSystem::Sim::FieldReadNode::Source::GridChannel
                          ? "grid" : "attribute";
        }
        if (auto* obj = dynamic_cast<NodeSystem::Sim::ObjectRefNode*>(node.get()))
            info.domain = obj->objectName;   // identity slot; scope says which
        if (auto* surface = dynamic_cast<NodeSystem::Sim::SurfaceInspectNode*>(node.get())) {
            info.channel = surface->channel;
            info.source = "surface";
            info.has_stats = true;
            info.stats_available = surface->stats.available;
            info.particle_count = surface->stats.particle_count;
            info.array_size = surface->stats.array_size;
            info.array_in_sync = surface->stats.in_sync;
            info.host_fresh = surface->stats.host_fresh;
            info.min_value = surface->stats.min_value;
            info.max_value = surface->stats.max_value;
            info.mean_value = surface->stats.mean_value;
        }
        if (auto* cache = dynamic_cast<NodeSystem::Sim::CacheNode*>(node.get())) {
            info.has_cache_status = cache->status.available;
            info.cache_valid = cache->status.valid;
            info.cache_baking = cache->status.baking;
            info.cache_stale = cache->status.stale;
            info.cache_ram_frames = cache->status.ram_frames;
        }
        if (auto* inspect = dynamic_cast<NodeSystem::Sim::FieldInspectNode*>(node.get())) {
            info.has_stats = true;
            info.stats_available = inspect->stats.available;
            info.particle_count = inspect->stats.particle_count;
            info.array_size = inspect->stats.array_size;
            info.array_in_sync = inspect->stats.in_sync;
            info.min_value = inspect->stats.min_value;
            info.max_value = inspect->stats.max_value;
            info.mean_value = inspect->stats.mean_value;
        }
        out.push_back(std::move(info));
    }
    out_nodes = std::move(out);
    return Result::success();
}

// ★★★ Published for the editor panel (D.4), and deliberately as a REFERENCE to
// the one instance rather than a copy. The panel is a view: every edit it makes
// lands in the same graph `rt.sim_graph.*` reads, so a script can inspect what a
// human just wired and a human sees what a script wired. Handing the UI its own
// copy would recreate the exact failure this repo already paid for twice — a
// panel holding state the core does not know about.
// ★★★ Returns null rather than a fallback graph. The panel must render "no
// graph for this scope" instead of drawing a graph that belongs to something
// else — a canvas that silently showed the wrong owner's nodes would be the
// panel-lies failure class this layer was built to end.
NodeSystem::Sim::SimulationNodeGraph* simulationGraph(const std::string& scope_text,
                                                      const std::string& owner) {
    SimGraphPtr graph;
    if (!findGraph(scope_text, owner, graph).ok) return nullptr;
    return graph.get();
}

} // namespace rtapi
