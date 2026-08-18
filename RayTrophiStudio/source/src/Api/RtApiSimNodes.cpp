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

std::unique_ptr<NodeSystem::Sim::SimulationNodeGraph> g_sim_graph;

NodeSystem::Sim::SimulationNodeGraph& simGraph() {
    if (!g_sim_graph)
        g_sim_graph = std::make_unique<NodeSystem::Sim::SimulationNodeGraph>();
    return *g_sim_graph;
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

    if (key == "kinematic_viscosity")        kinematic_viscosity = &value;
    else if (key == "viscosity_wall_slip")   viscosity_wall_slip = &value;
    else if (key == "surface_offset_voxels") surface_offset_voxels = &value;
    else if (key == "pore_amount")           pore_amount = &value;
    else if (key == "pore_scale")            pore_scale = &value;
    else if (key == "pore_detail")           pore_detail = &value;
    else if (key == "solid_phase_fill")      solid_phase_fill = &value;
    else if (key == "granular_cohesion")     granular_cohesion = &value;
    else if (key == "voxel_size")            voxel_size = &value;
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
        /* viscosity_sweeps               */ nullptr,
        /* viscosity_wall_slip            */ viscosity_wall_slip,
        /* surface_material               */ nullptr,
        /* surface_offset_voxels          */ surface_offset_voxels,
        /* pore_amount                    */ pore_amount,
        /* pore_scale                     */ pore_scale,
        /* pore_detail                    */ pore_detail,
        /* coord_space                    */ nullptr,
        /* uvw_refresh_period             */ nullptr,
        /* solid_phase                    */ nullptr,
        /* solid_phase_fill               */ solid_phase_fill,
        /* enabled                        */ nullptr,
        /* visible                        */ nullptr,
        /* granular_enabled               */ nullptr,
        /* granular_friction_angle_degrees*/ nullptr,
        /* granular_cohesion              */ granular_cohesion).ok;
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

SimApplyResult simGraphApply(bool allow_restart) {
    SimApplyResult out;
    const SimGraphEvaluation evaluation = simGraphEvaluate();

    for (const auto& cmd : evaluation.commands) {
        if (cmd.kind != "set_parameter") continue;
        const bool needs_restart =
            NodeSystem::Sim::SetParameterNode::keyRequiresRestart(cmd.key);
        if (needs_restart && !allow_restart) {
            // ★ Refused, and REPORTED. A graph edit must not silently discard a
            // running simulation; the caller has to ask for it.
            out.refused.push_back(cmd.key + " on '" + cmd.target +
                                  "' requires a simulation restart");
            continue;
        }
        OverrideKey ok{cmd.target, cmd.key};
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
                                               g_authored_gas_shader.size());
    out.ok = out.failed.empty();
    return out;
}

Result simGraphClearOverrides() {
    std::string first_error;
    for (const auto& entry : g_authored_values) {
        if (!writeParameter(entry.first.domain, entry.first.key, entry.second) &&
            first_error.empty()) {
            first_error = "cannot restore " + entry.first.key +
                          " on '" + entry.first.domain + "'";
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

    // Declared: the graph's Couple commands, in dependency order.
    const SimGraphEvaluation evaluation = simGraphEvaluate();
    for (const auto& cmd : evaluation.commands) {
        if (cmd.kind != "couple") continue;
        SimCouplingEntry e;
        e.coupling = cmd.key;
        e.source_domain = cmd.target;
        e.target_domain = cmd.text;
        e.active = cmd.value > 0.5f;
        e.source_node = cmd.source_node;
        out.declared.push_back(std::move(e));
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

Result simGraphClear() {
    simGraph().clear();
    return Result::success();
}

Result simGraphAddNode(const std::string& type_id, uint32_t& out_id) {
    auto node = NodeSystem::NodeRegistry::instance().create(type_id);
    if (!node) return Result::fail("unknown simulation node type: " + type_id);
    NodeSystem::NodeBase* added = simGraph().registerNode(std::move(node));
    if (!added) return Result::fail("failed to add node: " + type_id);
    out_id = added->id;
    return Result::success();
}

Result simGraphSetNodeText(uint32_t node_id, const std::string& key,
                           const std::string& value) {
    NodeSystem::NodeBase* node = simGraph().getNode(node_id);
    if (!node) return Result::fail("no simulation node with id " + std::to_string(node_id));
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

Result simGraphSetNodeValue(uint32_t node_id, const std::string& key, float value) {
    NodeSystem::NodeBase* node = simGraph().getNode(node_id);
    if (!node) return Result::fail("no simulation node with id " + std::to_string(node_id));
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

Result simGraphConnect(uint32_t from_node, int from_pin,
                       uint32_t to_node, int to_pin) {
    NodeSystem::NodeBase* src = simGraph().getNode(from_node);
    NodeSystem::NodeBase* dst = simGraph().getNode(to_node);
    if (!src) return Result::fail("no simulation node with id " + std::to_string(from_node));
    if (!dst) return Result::fail("no simulation node with id " + std::to_string(to_node));
    if (from_pin < 0 || from_pin >= static_cast<int>(src->outputs.size()))
        return Result::fail("output pin index out of range");
    if (to_pin < 0 || to_pin >= static_cast<int>(dst->inputs.size()))
        return Result::fail("input pin index out of range");
    // addLink returns 0 when the graph rejects the connection — type mismatch or
    // a cycle. ★ A cycle here is not merely invalid: the repo has already paid
    // for one as a stack overflow in the node-graph evaluator.
    if (simGraph().addLink(src->outputs[from_pin].id, dst->inputs[to_pin].id) == 0)
        return Result::fail("connection rejected (type mismatch or cycle)");
    return Result::success();
}

SimGraphEvaluation simGraphEvaluate() {
    SimGraphEvaluation out;
    auto& graph = simGraph();
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
            default: info.kind = "none"; break;
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

std::vector<SimNodeInfo> simGraphNodes() {
    std::vector<SimNodeInfo> out;
    for (const auto& node : simGraph().nodes) {
        SimNodeInfo info;
        info.id = node->id;
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
    return out;
}

// ★★★ Published for the editor panel (D.4), and deliberately as a REFERENCE to
// the one instance rather than a copy. The panel is a view: every edit it makes
// lands in the same graph `rt.sim_graph.*` reads, so a script can inspect what a
// human just wired and a human sees what a script wired. Handing the UI its own
// copy would recreate the exact failure this repo already paid for twice — a
// panel holding state the core does not know about.
NodeSystem::Sim::SimulationNodeGraph& simulationGraph() { return simGraph(); }

} // namespace rtapi
