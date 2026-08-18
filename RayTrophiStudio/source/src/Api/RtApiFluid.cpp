/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiFluid.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 */

#include "RtApiInternal.h"
#include "Fluid/FluidObject.h"
#include "Fluid/FluidSimulationSystem.h"
#include "Fluid/SubstanceTag.h"
#include "Fluid/APICFluidSolver.h"
#include "Fluid/FluidSplatMaterialAuthoring.h"
#include "ParticleSimulation.h"
#include "MaterialStateField.h"
#include "MaterialManager.h"
#include "VolumeShader.h"
#include <algorithm>
#include <cctype>
#include <cmath>     // std::sqrt, for the material-coordinate drift measurement

namespace rtapi {

namespace {

// Preset enum -> the string scripts use. One definition, because this ternary
// chain was copy-pasted at three call sites and adding Chocolate to two of them
// would have produced a domain that reports "custom" from fluid.list and
// "chocolate" from fluid.get.
const char* fluidPresetName(RayTrophiSim::Fluid::APICSolverParams::FluidPreset p) {
    using FluidPreset = RayTrophiSim::Fluid::APICSolverParams::FluidPreset;
    switch (p) {
        case FluidPreset::Water:     return "water";
        case FluidPreset::Oil:       return "oil";
        case FluidPreset::Mud:       return "mud";
        case FluidPreset::Honey:     return "honey";
        case FluidPreset::Lava:      return "lava";
        case FluidPreset::Sand:      return "sand";
        case FluidPreset::Chocolate: return "chocolate";
        case FluidPreset::WetSand:      return "wet_sand";
        case FluidPreset::Gravel:       return "gravel";
        case FluidPreset::CohesiveSoil: return "cohesive_soil";
        case FluidPreset::MoltenPlastic: return "molten_plastic";
        case FluidPreset::Custom:
        default:                     return "custom";
    }
}

// Copy the rheology block a script can read back. Kept together so a new
// rheology field cannot reach one reporting path and miss another.
void fillFluidRheology(const RayTrophiSim::Fluid::APICSolverParams& params,
                       FluidDomainInfo& info) {
    info.preset              = fluidPresetName(params.current_preset);
    info.kinematic_viscosity = params.kinematic_viscosity;
    info.viscosity_sweeps    = params.viscosity_sweeps;
    info.viscosity_wall_slip = params.viscosity_wall_slip;
    info.granular_enabled = params.granular_enabled;
    info.granular_friction_angle_degrees = params.granular_friction_angle_degrees;
    info.granular_cohesion = params.granular_cohesion;
    info.granular_dilatancy_degrees = params.granular_dilatancy_degrees;
    info.granular_young_modulus = params.granular_young_modulus;
    info.granular_poisson_ratio = params.granular_poisson_ratio;
    info.granular_tensile_cutoff = params.granular_tensile_cutoff;
    info.granular_hardening = params.granular_hardening;
    info.granular_fracture_strain = params.granular_fracture_strain;
    info.granular_damage_rate = params.granular_damage_rate;
    info.granular_healing_rate = params.granular_healing_rate;
    info.granular_rebonding = params.granular_rebonding;
    info.granular_max_solver_substeps = params.granular_max_solver_substeps;
    info.granular_softening_temperature = params.granular_softening_temperature;
    info.granular_softening_range = params.granular_softening_range;
    info.granular_residual_strength = params.granular_residual_strength;
    info.granular_tack_peak = params.granular_tack_peak;
    info.granular_thermal_conductivity = params.granular_thermal_conductivity;
}

// Report the isosurface material by NAME. Lives on the grid descriptor rather
// than in APICSolverParams (it is a look, not rheology), so it needs its own
// helper next to fillFluidRheology instead of riding along inside it.
void fillFluidSurfaceMaterial(const RayTrophiSim::SimulationGridDomainDesc& d,
                              FluidDomainInfo& info) {
    // Porosity rides along here for the same reason: it is surface LOOK, and a
    // script has to be able to read back what it set or it cannot tell "the
    // value never landed" from "the value landed and did nothing".
    info.pore_amount = d.fluid_surface_pore_amount;
    info.pore_scale  = d.fluid_surface_pore_scale;
    info.pore_detail = d.fluid_surface_pore_detail;
    info.surface_offset_voxels = d.fluid_level_set_params.surface_offset_voxels;
    info.coord_space = d.fluid_surface_coord_space;
    info.solid_phase_enabled = d.fluid_solid_phase_enabled;
    info.solid_phase_fill = d.fluid_solid_phase_fill;
    info.uvw_refresh_period = d.fluid_params.uvw_refresh_period;
    info.splat_material = RayTrophiSim::Fluid::fluidSplatMaterialName(d);

    // Substance -> material bindings, with the material NAME resolved. A script
    // that could only read the id would have to keep its own copy of the
    // material table to make sense of it.
    info.substance_materials.clear();
    {
        const auto& all = MaterialManager::getInstance().getAllMaterials();
        for (const auto& b : d.fluid_substance_materials) {
            FluidDomainInfo::SubstanceMaterialBinding out_b;
            out_b.substance = b.substance;
            out_b.material_id = b.material_id;
            out_b.representation =
                b.representation == RayTrophiSim::Fluid::SubstanceRepresentation::Splat ? "splat" :
                b.representation == RayTrophiSim::Fluid::SubstanceRepresentation::SurfaceSDF ? "sdf" : "inherit";
            // Resolved exactly the way the render bridge resolves it: an explicit
            // value wins, Inherit follows the domain mode. `Volume` is not a valid
            // liquid mode and is normalised to the isosurface where it is
            // consumed, so it resolves to "sdf" here too rather than inventing a
            // third answer this one report would be alone in using.
            out_b.effective_representation =
                b.representation == RayTrophiSim::Fluid::SubstanceRepresentation::Splat ? "splat" :
                b.representation == RayTrophiSim::Fluid::SubstanceRepresentation::SurfaceSDF ? "sdf" :
                (d.fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles ? "splat" : "sdf");
            if (b.material_id >= 0 &&
                static_cast<std::size_t>(b.material_id) < all.size() &&
                all[static_cast<std::size_t>(b.material_id)]) {
                out_b.material = all[static_cast<std::size_t>(b.material_id)]->materialName;
            }
            out_b.kinematic_viscosity = b.kinematic_viscosity;
            out_b.miscibility = b.miscibility;
            out_b.phase =
                b.phase == RayTrophiSim::Fluid::SubstancePhase::Solid ? "solid" : "liquid";
            info.substance_materials.push_back(out_b);
        }
    }

    info.surface_material.clear();
    if (d.fluid_surface_material_id < 0) return;
    const auto& mats = MaterialManager::getInstance().getAllMaterials();
    const std::size_t mi = static_cast<std::size_t>(d.fluid_surface_material_id);
    if (mi < mats.size() && mats[mi]) info.surface_material = mats[mi]->materialName;
}

Result setFluidSplatMaterialImpl(const std::string& domain_id_or_name,
                                 const std::string& material_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    for (auto& system : g_ctx->scene.particle_systems) {
        if (!system.runtime) continue;
        const auto authored = RayTrophiSim::Fluid::setFluidSplatMaterial(
            *system.runtime, domain_id_or_name, material_name);
        if (authored.status ==
            RayTrophiSim::Fluid::SplatMaterialAuthoringStatus::MaterialNotFound) {
            return Result::fail("material not found: " + material_name);
        }
        if (!authored.ok()) continue;

        if (authored.changed) {
            g_ctx->scene.requestSimulationTimelineRenderResync();
            g_ctx->renderer.resetCPUAccumulation();
            if (g_ctx->backend_ptr) g_ctx->backend_ptr->resetAccumulation();
            g_ctx->start_render = true;
        }
        return Result::success();
    }

    return Result::fail("fluid domain not found: " + domain_id_or_name);
}

// Material-coordinate diagnostics. Read from the LIVE state, not the descriptor:
// the coordinate is carried by particles, so the descriptor knows nothing about
// it and a report sourced there would always say "fine".
//
// ★ The measurement is mean |uvw - position| — how far the average parcel of
// liquid has moved since birth. This is the number that distinguishes a working
// anchor from a silent fall back to world space, and it is deliberately a
// TREND, not a threshold: it must RISE while liquid pours. A non-zero check
// would also pass on a coordinate that was seeded once and then frozen by a
// broken inheritance path, which is the most likely way this breaks.
// Solid-phase measurement, read from the LIVE step statistics rather than from
// the descriptor.
//
// ★★★ THE DESCRIPTOR CANNOT ANSWER THIS. It knows a substance was DECLARED
// solid; only the step knows whether any parcel of it existed and whether any
// cell was full enough to block. Sourcing it from the binding would produce a
// report that always agrees with what the caller just wrote — the shape of
// measurement with none of its value.
void fillFluidSolidPhase(const RayTrophiSim::SimulationGridDomainState& state,
                         FluidDomainInfo& info) {
    info.solid_phase_particles =
        static_cast<uint64_t>(state.fluid_stats.solid_phase_particles);
    info.solid_phase_cells =
        static_cast<uint64_t>(state.fluid_stats.solid_phase_cells);
    // Same source, same reason: only the step knows whether a region ended up
    // without a pressure reference. Nothing in the descriptor implies it.
    info.sealed_pockets =
        static_cast<uint64_t>(state.fluid_stats.sealed_pockets);
    info.sealed_pocket_cells =
        static_cast<uint64_t>(state.fluid_stats.sealed_pocket_cells);
    info.sealed_pockets_measured = state.fluid_stats.sealed_pockets_measured;
    info.interior_fluid_cells =
        static_cast<uint64_t>(state.fluid_stats.interior_fluid_cells);
    info.reseed_added_particles =
        static_cast<uint64_t>(state.fluid_stats.reseed_added_particles);
    info.reseed_removed_particles =
        static_cast<uint64_t>(state.fluid_stats.reseed_removed_particles);
    info.granular_yielded_particles =
        static_cast<uint64_t>(state.fluid_stats.granular_yielded_particles);
    info.granular_detached_particles =
        static_cast<uint64_t>(state.fluid_stats.granular_detached_particles);
    info.granular_invalid_particles =
        static_cast<uint64_t>(state.fluid_stats.granular_invalid_particles);
    info.granular_sleeping_particles =
        static_cast<uint64_t>(state.fluid_stats.granular_sleeping_particles);
    info.granular_damaged_particles =
        static_cast<uint64_t>(state.fluid_stats.granular_damaged_particles);
    info.granular_damage_over_10_particles =
        static_cast<uint64_t>(state.fluid_stats.granular_damage_over_10_particles);
    info.granular_damage_over_50_particles =
        static_cast<uint64_t>(state.fluid_stats.granular_damage_over_50_particles);
    info.granular_damage_over_90_particles =
        static_cast<uint64_t>(state.fluid_stats.granular_damage_over_90_particles);
    info.granular_max_yield = state.fluid_stats.granular_max_yield_value;
    info.granular_max_plastic_increment = state.fluid_stats.granular_max_plastic_increment;
    info.granular_max_accumulated_plastic = state.fluid_stats.granular_max_accumulated_plastic;
    info.granular_mean_accumulated_plastic = state.fluid_stats.granular_mean_accumulated_plastic;
    info.granular_max_fracture_history = state.fluid_stats.granular_max_fracture_history;
    info.granular_mean_fracture_history = state.fluid_stats.granular_mean_fracture_history;
    info.granular_max_damage = state.fluid_stats.granular_max_damage;
    info.granular_mean_damage = state.fluid_stats.granular_mean_damage;
    info.granular_requested_young_modulus = state.fluid_stats.granular_requested_young_modulus;
    info.granular_effective_young_modulus = state.fluid_stats.granular_effective_young_modulus;
    info.granular_required_substeps = state.fluid_stats.granular_required_substeps;
    info.granular_solver_substeps = state.fluid_stats.granular_solver_substeps;
    info.granular_stiffness_capped = state.fluid_stats.granular_stiffness_capped;
    info.granular_wave_substeps = state.fluid_stats.granular_wave_substeps;
    info.granular_strain_substeps = state.fluid_stats.granular_strain_substeps;
    info.granular_strain_rate = state.fluid_stats.granular_strain_rate;
    info.granular_strain_limited_particles =
        static_cast<uint64_t>(state.fluid_stats.granular_strain_limited_particles);
    info.granular_compaction_capped_particles =
        static_cast<uint64_t>(state.fluid_stats.granular_compaction_capped_particles);
    info.granular_min_softening = state.fluid_stats.granular_min_softening;
    info.granular_softened_particles =
        static_cast<uint64_t>(state.fluid_stats.granular_softened_particles);
    info.granular_overburden_pressure = state.fluid_stats.granular_overburden_pressure;
    info.granular_young_modulus_for_load = state.fluid_stats.granular_young_modulus_for_load;
    info.granular_stiffness_below_load = state.fluid_stats.granular_stiffness_below_load;
}

void fillFluidMaterialCoords(const RayTrophiSim::SimulationGridDomainState& state,
                             FluidDomainInfo& info) {
    info.uvw_dim[0] = state.grid.nx;
    info.uvw_dim[1] = state.grid.ny;
    info.uvw_dim[2] = state.grid.nz;
    info.uvw_origin[0] = state.grid.origin.x;
    info.uvw_origin[1] = state.grid.origin.y;
    info.uvw_origin[2] = state.grid.origin.z;
    info.uvw_voxel = state.grid.voxel_size;

    const auto& parts = state.particles;
    const std::size_t n = parts.size();
    // A short sidecar means some producer emitted without a coordinate. Report
    // unavailable rather than averaging over the prefix: a drift computed from
    // the particles that HAPPEN to have one is a number that looks healthy while
    // the surface tears.
    if (n == 0 || parts.uvw.size() < n) {
        info.uvw_available = false;
        info.uvw_drift = 0.0f;
        info.uvw_particles = 0;
        return;
    }
    // ★★★ MEASURE THE BLEND, NOT ONE GENERATION. The shader is fed
    // w_a*(uvw_a - p) + w_b*(uvw_b - p), and that is what this number has to
    // describe. Reporting generation A alone would saw-tooth to zero every time
    // A is reset — a periodic collapse in a metric whose whole job is to catch
    // collapses, i.e. the instrument would raise an alarm about the schedule it
    // is supposed to be measuring through.
    //
    // ★ Blended, it is continuous BY CONSTRUCTION: a generation's weight is zero
    // exactly when it resets, so the discontinuity is multiplied away. The same
    // property that makes the refresh invisible on screen makes it invisible
    // here, which is the sign the two agree about what is happening.
    float w_a = 1.0f, w_b = 0.0f;
    parts.materialCoordWeights(w_a, w_b);
    const bool has_gen_b = parts.uvw_b.size() >= n;
    if (!has_gen_b) { w_a = 1.0f; w_b = 0.0f; }
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 da = parts.uvw[i] - parts.position[i];
        const Vec3 db = has_gen_b ? (parts.uvw_b[i] - parts.position[i]) : da;
        const Vec3 d(w_a * da.x + w_b * db.x,
                     w_a * da.y + w_b * db.y,
                     w_a * da.z + w_b * db.z);
        acc += std::sqrt(static_cast<double>(d.x) * d.x +
                         static_cast<double>(d.y) * d.y +
                         static_cast<double>(d.z) * d.z);
    }
    // ── Substance census ─────────────────────────────────────────────────────
    // Counted from the LIVE particles, not from the emitter list: an emitter
    // describes what is being poured, and this has to describe what is actually
    // in the domain. The two differ constantly — after a source is disabled its
    // liquid is still there, and that liquid is exactly what a test asking
    // "did identity survive?" needs to see.
    info.substances.clear();
    {
        const std::size_t tag_n = parts.substance_tag.size();
        for (std::size_t i = 0; i < n && i < tag_n; ++i) {
            const uint32_t tag = parts.substance_tag[i];
            bool found = false;
            for (auto& entry : info.substances) {
                if (entry.tag == tag) { ++entry.particles; found = true; break; }
            }
            if (!found) {
                FluidDomainInfo::SubstanceCount entry;
                entry.tag = tag;
                entry.particles = 1;
                info.substances.push_back(entry);
            }
        }
        // Resolve names from the sources feeding THIS domain. A linear scan over
        // a handful of emitters per distinct tag; both counts are tiny.
        const auto& sources = scriptSimulationRuntime().flowSources();
        for (auto& entry : info.substances) {
            if (entry.tag == RayTrophiSim::Fluid::kSubstanceUntagged) continue;
            for (const auto& src : sources) {
                if (src.fluid_substance.empty()) continue;
                if (RayTrophiSim::Fluid::substanceTag(src.fluid_substance) != entry.tag)
                    continue;
                entry.name = src.fluid_substance;
                break;
            }
        }
    }

    info.uvw_available = true;
    info.uvw_particles = static_cast<uint64_t>(n);
    info.uvw_drift = static_cast<float>(acc / static_cast<double>(n));
}

// Locate a Gas domain descriptor by id or name. Shared by the shader accessors
// below; mirrors the lookup updateGasDomainSettings performs.
RayTrophiSim::SimulationGridDomainDesc* findGasDomainDesc(
    const std::string& domain_id_or_name, Result& out_error) {
    if (!g_ctx) { out_error = notBound(); return nullptr; }
    FluidDomainInfo info;
    Result found = getFluidDomain(domain_id_or_name, info);
    if (!found.ok) { out_error = found; return nullptr; }
    auto& domains = g_ctx->scene.ensureParticleSimulationSystem().gridDomains();
    auto it = std::find_if(domains.begin(), domains.end(),
        [&info](const auto& d) { return d.name == info.name; });
    if (it == domains.end() || it->type != RayTrophiSim::SimulationDomainType::Gas) {
        out_error = Result::fail("gas domain not found: " + domain_id_or_name);
        return nullptr;
    }
    out_error = Result::success();
    return &(*it);
}

} // namespace

Result setFluidSplatMaterial(const std::string& domain_id_or_name,
                             const std::string& material_name) {
    return setFluidSplatMaterialImpl(domain_id_or_name, material_name);
}

Result listMaterialSubstances(std::vector<std::string>& out_names) {
    out_names.clear();
    for (const auto& profile : RayTrophiSim::substanceLibrary()) {
        out_names.push_back(profile.name);
    }
    return Result::success();
}

Result listMaterialFields(std::vector<MaterialFieldInfo>& out_fields) {
    if (!g_ctx) return notBound();
    out_fields.clear();
    auto& runtime = scriptSimulationRuntime();
    // Integrity/mass-loss live on the device during simulation. Request the
    // next scheduler step's readback before exposing the current host snapshot;
    // repeated script polling then observes a coherent completed generation
    // instead of reading the creation-time all-ones integrity forever.
    runtime.requestMaterialStateFieldReadback();
    const auto& fields = runtime.materialStateFields();
    out_fields.reserve(fields.size());
    for (const auto& entry : fields) {
        const auto& field = entry.second;
        MaterialFieldInfo info;
        info.object_key = field.object_key;
        info.substance = field.substance_name;
        info.topology_generation = field.topology_generation;
        info.content_generation = field.mask_revision;
        info.element_count = static_cast<uint32_t>(field.elementCount());
        info.mask_resolution = field.mask_resolution;
        info.centers_dirty = field.centers_dirty;
        const auto integrity =
            RayTrophiSim::MaterialStateFieldSystem::summarizeIntegrity(field);
        if (integrity.valid) {
            info.mean_integrity = integrity.mean_integrity;
            info.minimum_integrity = integrity.minimum_integrity;
            info.mass_loss = integrity.total_mass_loss;
        }
        const auto budget =
            RayTrophiSim::MaterialStateFieldSystem::summarizeMassBudget(field);
        if (budget.valid) {
            info.initial_mass = budget.initial_mass;
            info.solid_mass = budget.solid_mass;
            info.pyrolyzed_mass = budget.pyrolyzed_mass;
            info.molten_reservoir_mass = budget.molten_reservoir_mass;
            info.transferred_mass = budget.transferred_mass;
            info.mass_conservation_error = budget.conservation_error;
            info.mass_budget_overflow = budget.budget_overflow_mass;
            info.mass_negative = budget.negative_mass;
            info.mass_invalid_elements = budget.invalid_elements;
        }
        info.semantics = {
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::Temperature),
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::Moisture),
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::FuelRemaining),
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::Char),
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::Melt),
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::MassLoss),
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::Integrity)
        };
        out_fields.push_back(std::move(info));
    }
    return Result::success();
}

Result getGasShaderSettings(const std::string& domain_id_or_name,
                            GasShaderSettings& out_settings) {
    Result error;
    auto* domain = findGasDomainDesc(domain_id_or_name, error);
    if (!domain) return error;
    if (!domain->shader) {
        // Not an error: the render bridge creates the shader lazily on the
        // first sync. Report the defaults the domain's intent will produce.
        out_settings = GasShaderSettings{};
        // No shader yet: the domain's combustion intent is the only signal there
        // is, so it stands in for a preset that was never chosen.
        out_settings.preset = domain->fire_enabled ? "fire" : "smoke";
        return Result::success();
    }
    const auto& s = *domain->shader;
    // ★★★ Report the preset the SHADER was built from, not the combustion flag.
    // These are two different questions and the writer only ever touched the
    // first, so reading the second made a preset change look like a no-op.
    // `fire_enabled` remains the fallback for shaders that predate the field.
    out_settings.preset = !domain->shader_preset.empty()
        ? domain->shader_preset
        : (domain->fire_enabled ? "fire" : "smoke");
    out_settings.density_multiplier = s.density.multiplier;
    out_settings.density_cutoff = s.density.cutoff_threshold;
    out_settings.blackbody_intensity = s.emission.blackbody_intensity;
    out_settings.temperature_min = s.emission.temperature_min;
    out_settings.temperature_max = s.emission.temperature_max;
    out_settings.scattering_coefficient = s.scattering.coefficient;
    out_settings.absorption_coefficient = s.absorption.coefficient;
    return Result::success();
}

Result updateGasShaderSettings(const std::string& domain_id_or_name,
                               const GasShaderSettings& settings) {
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    Result error;
    auto* domain = findGasDomainDesc(domain_id_or_name, error);
    if (!domain) return error;

    std::string preset = settings.preset;
    std::transform(preset.begin(), preset.end(), preset.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    // ★★★ A PRESET IS A RECIPE, so switching to a different one must let its own
    // values stand. The usual caller reads the settings, edits one field and
    // writes the whole struct back — so on a preset CHANGE every numeric field
    // in `settings` still describes the OLD preset, and applying them would
    // install the fire recipe and immediately overwrite it with smoke's numbers.
    // The preset would be a label with no effect: measured 2026-08-17 through
    // rt.gas.set_shader, which reported success and changed nothing.
    //
    // So a change installs the recipe and returns. Setting a preset AND custom
    // values is two calls, which is also the honest description of what it is.
    const bool preset_changed =
        !preset.empty() && preset != domain->shader_preset;
    if (preset == "fire") {
        domain->shader = VolumeShader::createFirePreset();
        domain->shader_preset = "fire";
    } else if (preset == "smoke") {
        domain->shader = VolumeShader::createSmokePreset();
        domain->shader_preset = "smoke";
    } else if (!preset.empty()) {
        return Result::fail("unknown gas shader preset: " + settings.preset);
    } else if (!domain->shader) {
        domain->shader = domain->fire_enabled ? VolumeShader::createFirePreset()
                                              : VolumeShader::createSmokePreset();
        domain->shader_preset = domain->fire_enabled ? "fire" : "smoke";
    }
    if (preset_changed) {
        g_gas_volumes_dirty = true;
        return Result::success();
    }
    auto& s = *domain->shader;
    s.density.multiplier = std::max(0.0f, settings.density_multiplier);
    s.density.cutoff_threshold = std::max(0.0f, settings.density_cutoff);
    s.emission.blackbody_intensity = std::max(0.0f, settings.blackbody_intensity);
    // A zero-width or inverted range maps every temperature to the same colour,
    // which reads as "emission is broken". Keep the pair ordered and separated.
    s.emission.temperature_min = std::max(0.0f, settings.temperature_min);
    s.emission.temperature_max =
        std::max(s.emission.temperature_min + 1.0f, settings.temperature_max);
    s.scattering.coefficient = std::max(0.0f, settings.scattering_coefficient);
    s.absorption.coefficient = std::max(0.0f, settings.absorption_coefficient);
    g_gas_volumes_dirty = true;
    return Result::success();
}

Result createFluidDomain(const std::string& name, Vec3 domain_min, Vec3 domain_max,
                         float voxel_size, const std::string& type, rtapi::FluidDomainInfo& out_info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    std::string domain_name = name.empty() ? "Fluid" : name;
    std::string dom_type = type;
    std::transform(dom_type.begin(), dom_type.end(), dom_type.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    const bool is_gas = (dom_type == "gas" || dom_type == "smoke" || dom_type == "fire");

    // 1. Ensure high-level Physics UI Panel domain descriptor exists
    auto& p_sys = g_ctx->scene.ensureParticleSimulationSystem();
    auto& domains = p_sys.gridDomains();
    RayTrophiSim::SimulationGridDomainDesc* grid_dom = nullptr;
    for (auto& d : domains) {
        if (d.name == domain_name) { grid_dom = &d; break; }
    }
    if (!grid_dom) {
        RayTrophiSim::SimulationGridDomainDesc desc;
        desc.name = domain_name;
        desc.type = is_gas ? RayTrophiSim::SimulationDomainType::Gas : RayTrophiSim::SimulationDomainType::Fluid;
        // Both gas and liquid now start on the fastest solver the machine has —
        // scripted domains should not be slower than panel-created ones. Scene
        // synchronization selects CUDA first and Vulkan second, and retains the
        // deterministic CPU fallback when neither GPU backend is usable.
        desc.backend = RayTrophiSim::defaultSimulationDomainBackend();
        desc.boundary_mode = is_gas ? RayTrophiSim::SimulationGridDomainBoundaryMode::Open : RayTrophiSim::SimulationGridDomainBoundaryMode::Closed;
        desc.bounds_min = domain_min;
        desc.bounds_max = domain_max;
        if (voxel_size > 0.001f) desc.voxel_size = voxel_size;
        desc.enabled = true;
        g_ctx->scene.addSimulationGridDomain(desc);
        // addSimulationGridDomain may reallocate the descriptor vector; resolve
        // the pointer again so the returned API state reports the real backend.
        for (auto& d : p_sys.gridDomains()) {
            if (d.name == domain_name) { grid_dom = &d; break; }
        }
    } else {
        grid_dom->type = is_gas ? RayTrophiSim::SimulationDomainType::Gas : RayTrophiSim::SimulationDomainType::Fluid;
        grid_dom->boundary_mode = is_gas ? RayTrophiSim::SimulationGridDomainBoundaryMode::Open : RayTrophiSim::SimulationGridDomainBoundaryMode::Closed;
        grid_dom->bounds_min = domain_min;
        grid_dom->bounds_max = domain_max;
        if (voxel_size > 0.001f) grid_dom->voxel_size = voxel_size;
    }

    // ★ A GAS domain gets NO legacy FluidObject.
    //
    // FluidObject is the pre-grid-domain APIC LIQUID container (particles, seed
    // box, level set). Creating one for gas left a ghost that nothing simulated
    // but that syncFluidRenderVolumes still walked EVERY FRAME: it built a
    // density volume from the ghost's grid, found it empty, and called
    // destroyFluidRenderVolume — which unconditionally raises g_geometry_dirty +
    // g_vulkan_rebuild_pending. That is a full TLAS rebuild per frame, and it is
    // exactly why a scripted gas domain churned while a panel-created one (grid
    // domain only) did not. The grid domain is the authority; getFluidDomain
    // resolves gas domains from it directly.
    if (is_gas) {
        if (!grid_dom) return Result::fail("failed to create gas domain: " + domain_name);
        out_info.id = 0;
        out_info.name = grid_dom->name;
        out_info.type = "gas";
        out_info.domain_min = grid_dom->bounds_min;
        out_info.domain_max = grid_dom->bounds_max;
        out_info.voxel_size = grid_dom->voxel_size;
        out_info.particle_count = 0;
        out_info.render_mode = "volume";
        out_info.boundary =
            (grid_dom->boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Open) ? "open" :
            (grid_dom->boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Periodic) ? "periodic" : "closed";
        out_info.preset = "custom";
        out_info.backend =
            (grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Compute) ? "gpu" :
            (grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) ? "vulkan" :
            (grid_dom->backend == RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB) ? "cpu_sparse" : "cpu";
        out_info.enabled = grid_dom->enabled;
        out_info.visible = true;
        return Result::success();
    }

    // 2. Ensure low-level FluidObject exists (liquid domains only)
    auto existing = g_ctx->scene.findFluidObjectByName(domain_name);
    RayTrophiSim::Fluid::FluidObject* obj = existing;
    if (!obj) {
        obj = g_ctx->scene.addFluidObject(domain_name);
    }
    if (!obj) return Result::fail("failed to create fluid domain: " + domain_name);

    obj->params.boundary = is_gas ? RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Open : RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Closed;
    obj->domain_min = domain_min;
    obj->domain_max = domain_max;
    if (voxel_size > 0.001f) obj->voxel_size = voxel_size;
    obj->grid_dirty = true;
    obj->ensureGrid();

    out_info.id = obj->id;
    out_info.name = obj->name;
    out_info.type = is_gas ? "gas" : "fluid";
    out_info.domain_min = obj->domain_min;
    out_info.domain_max = obj->domain_max;
    out_info.voxel_size = obj->voxel_size;
    out_info.particle_count = obj->particles.size();
    out_info.render_mode = (obj->render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) ? "surface" :
                           (obj->render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) ? "particles" : "volume";
    out_info.boundary = (obj->params.boundary == RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Open) ? "open" :
                        (obj->params.boundary == RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Periodic) ? "periodic" : "closed";
    fillFluidRheology(obj->params, out_info);
    // ★★★ THE GRID DESCRIPTOR IS THE LIVE APIC AUTHORITY, and getFluidDomain
    // has always said so (see the same override there). This listing did not,
    // so it reported the legacy editor mirror for every rheology field.
    //
    // Measured divergence on a real scene: fluid.list_domains said
    // granular_enabled = true while fluid.get said false for the SAME domain,
    // with every other granular field agreeing. The solver reads the grid
    // descriptor, so the listing was the one lying — and it is the surface a
    // script reaches for first when it asks "which domains are granular?".
    // Filtering on it would pick a domain the solver is not running as granular
    // at all, and nothing would report an error.
    if (grid_dom) {
        out_info.render_mode =
            (grid_dom->fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) ? "surface" :
            (grid_dom->fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) ? "particles" : "volume";
        out_info.boundary =
            (grid_dom->fluid_params.boundary == RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Open) ? "open" :
            (grid_dom->fluid_params.boundary == RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Periodic) ? "periodic" : "closed";
        fillFluidRheology(grid_dom->fluid_params, out_info);
    }
    out_info.backend = (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Compute) ? "gpu" :
                       (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) ? "vulkan" :
                       (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB) ? "cpu_sparse" : "cpu";
    out_info.enabled = obj->enabled;
    out_info.visible = obj->visible;
    return Result::success();
}

Result getFluidDomain(const std::string& domain_id_or_name, rtapi::FluidDomainInfo& out_info) {
    if (!g_ctx) return notBound();

    RayTrophiSim::Fluid::FluidObject* obj = nullptr;
    try {
        size_t idx = 0;
        uint32_t id = static_cast<uint32_t>(std::stoul(domain_id_or_name, &idx));
        if (idx == domain_id_or_name.size()) {
            for (auto& fo : g_ctx->scene.fluid_objects) {
                if (fo.id == id) { obj = &fo; break; }
            }
        }
    } catch (...) {}

    if (!obj) obj = g_ctx->scene.findFluidObjectByName(domain_id_or_name);
    if (!obj) {
        // ★★★★★ THE DEAD TWIN. This branch is labelled "gas domains own no
        // FluidObject", but what it actually tests is "no FluidObject carries
        // this NAME" — and a GRID-BASED FLUID domain satisfies that too. So a
        // live fluid domain with 30,000 particles fell in here and was answered
        // from a bare descriptor: particle_count hardcoded to 0, preset forced
        // to "custom", every granular field left at its struct default.
        //
        // ★★★ It returns no error. It returns a PLAUSIBLE domain that reads
        // "granular disabled, nothing simulated", while fluid.list_domains
        // reports the same domain at the same instant as granular_enabled=true
        // with 30,000 particles. Measured 2026-08-16 over IPC.
        //
        // ★★ That false reading is what produced the diagnosis "the thermal
        // chain is off on this domain" — the chain was fine, the INSTRUMENT was
        // lying. This is the QA read path agents drive the app through, so a
        // silent wrong answer here costs a whole debugging round.
        //
        // ★ Same class as "a default is not a measurement": absence of a
        // FluidObject was silently converted into "I measured zero particles".
        // The cure is to answer from the same authority list_domains uses — the
        // grid domains of EVERY particle system, preferring one with a stepped
        // state — and to leave live_state false when there is genuinely no
        // measurement to report.
        RayTrophiSim::SimulationGridDomainDesc* gd = nullptr;
        RayTrophiSim::ParticleSimulationSystem* gd_sys = nullptr;
        std::size_t gd_index = 0u;
        auto considerNamed = [&](RayTrophiSim::ParticleSimulationSystem& sys) {
            const auto& domains = sys.gridDomains();
            const auto& states = sys.gridDomainStates();
            for (std::size_t i = 0; i < domains.size(); ++i) {
                if (domains[i].name != domain_id_or_name) continue;
                const bool live = i < states.size() && states[i].valid;
                if (gd && !live) continue;          // keep the better candidate
                gd = const_cast<RayTrophiSim::SimulationGridDomainDesc*>(&domains[i]);
                gd_sys = &sys;
                gd_index = i;
                if (live) return true;              // cannot do better
            }
            return false;
        };
        for (auto& system : g_ctx->scene.particle_systems) {
            if (!system.runtime) continue;
            if (considerNamed(*system.runtime)) break;
        }
        if (!gd) considerNamed(g_ctx->scene.ensureParticleSimulationSystem());
        if (!gd) return Result::fail("fluid domain not found: " + domain_id_or_name);

        out_info = FluidDomainInfo{};
        out_info.id = 0;
        out_info.name = gd->name;
        out_info.type = (gd->type == RayTrophiSim::SimulationDomainType::Gas) ? "gas" : "fluid";
        out_info.domain_min = gd->bounds_min;
        out_info.domain_max = gd->bounds_max;
        out_info.voxel_size = gd->voxel_size;
        out_info.particle_count = 0;
        out_info.render_mode =
            (gd->fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) ? "surface" :
            (gd->fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) ? "particles" : "volume";
        out_info.boundary =
            (gd->boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Open) ? "open" :
            (gd->boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Periodic) ? "periodic" : "closed";
        out_info.backend =
            (gd->backend == RayTrophiSim::SimulationDomainBackend::GPU_Compute) ? "gpu" :
            (gd->backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) ? "vulkan" :
            (gd->backend == RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB) ? "cpu_sparse" : "cpu";
        out_info.enabled = gd->enabled;
        out_info.visible = true;
        // The real parameters, not struct defaults: preset, rheology and every
        // granular dial come from the descriptor the solver is actually running.
        fillFluidRheology(gd->fluid_params, out_info);
        fillFluidSurfaceMaterial(*gd, out_info);
        if (gd_sys) {
            const auto& states = gd_sys->gridDomainStates();
            if (gd_index < states.size() && states[gd_index].valid) {
                out_info.particle_count = states[gd_index].particles.size();
                out_info.live_state = true;  // only now is the count a measurement
                fillFluidMaterialCoords(states[gd_index], out_info);
                fillFluidSolidPhase(states[gd_index], out_info);
            }
        }
        return Result::success();
    }

    // ★ Search EVERY particle system, not just the active one. fluid.list_domains
    // already walks them all, and resolving the same domain name through two
    // different scopes is how the two calls came to disagree (9963 vs 0) about
    // the same domain at the same instant. Sim APIs are otherwise locked to the
    // active system on purpose; a read-only lookup must not inherit that limit,
    // because the answer it falls back to is indistinguishable from a real zero.
    // ★★ Prefer the system that actually HAS a stepped state for this name, not
    // simply the first system that carries the name.
    //
    // Searching every system (rather than only the active one) was needed
    // because fluid.get and fluid.list_domains disagreed — 9963 vs 0 — about the
    // same domain at the same instant. But "first match wins" then picked up a
    // stale same-named descriptor from an inactive system, which reported
    // granular_enabled=false while the panel showed the granular solver plainly
    // running. Trading one wrong answer for a different wrong answer.
    //
    // A descriptor with a live state is the only one that can answer a question
    // about the simulation, so that is the one to return. The nameless fallback
    // below still resolves authored parameters for a domain that has never been
    // stepped — but live_state stays false there, which is the caller's signal
    // that no measurement was taken.
    RayTrophiSim::SimulationGridDomainDesc* grid_dom = nullptr;
    RayTrophiSim::ParticleSimulationSystem* owning_sys = nullptr;
    std::size_t grid_index = 0u;
    auto consider = [&](RayTrophiSim::ParticleSimulationSystem& sys) {
        const auto& domains = sys.gridDomains();
        const auto& states = sys.gridDomainStates();
        for (std::size_t i = 0; i < domains.size(); ++i) {
            if (domains[i].name != obj->name) continue;
            const bool live = i < states.size() && states[i].valid;
            if (grid_dom && !live) continue;   // keep the better candidate
            grid_dom = const_cast<RayTrophiSim::SimulationGridDomainDesc*>(&domains[i]);
            owning_sys = &sys;
            grid_index = i;
            if (live) return true;             // cannot do better than a live one
        }
        return false;
    };
    for (auto& system : g_ctx->scene.particle_systems) {
        if (!system.runtime) continue;
        if (consider(*system.runtime)) break;
    }
    if (!grid_dom) consider(g_ctx->scene.ensureParticleSimulationSystem());

    out_info.id = obj->id;
    out_info.name = obj->name;
    out_info.type = (grid_dom && grid_dom->type == RayTrophiSim::SimulationDomainType::Gas) ? "gas" : "fluid";
    out_info.domain_min = obj->domain_min;
    out_info.domain_max = obj->domain_max;
    out_info.voxel_size = obj->voxel_size;
    out_info.particle_count = obj->particles.size();
    out_info.render_mode = (obj->render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) ? "surface" :
                           (obj->render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) ? "particles" : "volume";
    out_info.boundary = (obj->params.boundary == RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Open) ? "open" :
                        (obj->params.boundary == RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Periodic) ? "periodic" : "closed";
    fillFluidRheology(obj->params, out_info);
    out_info.surface_offset_voxels = obj->level_set_params.surface_offset_voxels;
    out_info.backend = (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Compute) ? "gpu" :
                       (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) ? "vulkan" :
                       (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB) ? "cpu_sparse" : "cpu";
    out_info.enabled = obj->enabled;
    out_info.visible = obj->visible;
    // The grid descriptor is the live APIC authority. Automatic molten
    // transfer may specialize an empty domain without mutating the legacy
    // editor mirror, so report those live values when available.
    if (grid_dom) {
        out_info.render_mode =
            (grid_dom->fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) ? "surface" :
            (grid_dom->fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) ? "particles" : "volume";
        fillFluidRheology(grid_dom->fluid_params, out_info);
        fillFluidSurfaceMaterial(*grid_dom, out_info);
        const auto& states = owning_sys->gridDomainStates();
        // Index comes from the search above, not from pointer arithmetic: the
        // candidate may have been picked in an earlier system than the one this
        // pointer now belongs to, and `&d - data()` would then index the wrong
        // state array without any sign that it had.
        const std::size_t index = grid_index;
        if (index < states.size() && states[index].valid) {
            out_info.particle_count = states[index].particles.size();
            out_info.live_state = true;   // only now is the count a measurement
            fillFluidMaterialCoords(states[index], out_info);
            fillFluidSolidPhase(states[index], out_info);
        }
    }
    return Result::success();
}

Result listFluidDomains(std::vector<rtapi::FluidDomainInfo>& out_domains) {
    if (!g_ctx) return notBound();
    out_domains.clear();

    // The grid-domain list is the authority for BOTH types (a gas domain owns no
    // FluidObject at all). Walk every particle system, not just the active one —
    // presets and imported scenes can leave a second system behind, and a domain
    // the caller cannot see is a domain it cannot delete.
    for (auto& system : g_ctx->scene.particle_systems) {
        if (!system.runtime) continue;
        for (const auto& d : system.runtime->gridDomains()) {
            FluidDomainInfo info;
            info.name = d.name;
            info.type = (d.type == RayTrophiSim::SimulationDomainType::Gas) ? "gas" : "fluid";
            info.domain_min = d.bounds_min;
            info.domain_max = d.bounds_max;
            info.voxel_size = d.voxel_size;
            info.boundary =
                (d.boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Open) ? "open" :
                (d.boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Periodic) ? "periodic" : "closed";
            info.backend =
                (d.backend == RayTrophiSim::SimulationDomainBackend::GPU_Compute) ? "gpu" :
                (d.backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) ? "vulkan" :
                (d.backend == RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB) ? "cpu_sparse" : "cpu";
            info.enabled = d.enabled;
            info.visible = true;
            info.render_mode = "volume";
            info.preset = "custom";
            info.particle_count = 0;
            fillFluidSurfaceMaterial(d, info);
            // Liquid domains additionally carry a legacy FluidObject; take the
            // id/editor visibility from it. Live particles and rheology remain
            // grid-domain authoritative: the mirror intentionally owns no
            // independently stepped particle copy.
            if (d.type != RayTrophiSim::SimulationDomainType::Gas) {
                if (auto* obj = g_ctx->scene.findFluidObjectByName(d.name)) {
                    info.id = obj->id;
                    info.visible = obj->visible;
                }
                info.render_mode =
                    (d.fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) ? "surface" :
                    (d.fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) ? "particles" : "volume";
                fillFluidRheology(d.fluid_params, info);
            }

            // Same authoritative diagnostics fluid.get reports. Keep this
            // after the legacy metadata block so particle_count can never be
            // overwritten by the deliberately empty FluidObject mirror.
            {
                const auto& all = system.runtime->gridDomains();
                const auto& states = system.runtime->gridDomainStates();
                const std::size_t index = static_cast<std::size_t>(&d - all.data());
                if (index < states.size() && states[index].valid) {
                    info.particle_count = states[index].particles.size();
                    info.live_state = true;   // only now is the count a measurement
                    fillFluidMaterialCoords(states[index], info);
                    fillFluidSolidPhase(states[index], info);
                }
            }
            out_domains.push_back(std::move(info));
        }
    }
    return Result::success();
}

Result removeFluidDomain(const std::string& domain_id_or_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    rtapi::FluidDomainInfo info;
    if (!getFluidDomain(domain_id_or_name, info).ok) {
        return Result::fail("fluid domain not found: " + domain_id_or_name);
    }

    bool grid_removed = false;
    for (std::size_t si = 0; si < g_ctx->scene.particle_systems.size() && !grid_removed; ++si) {
        auto& system = g_ctx->scene.particle_systems[si];
        if (!system.runtime) continue;
        const auto& domains = system.runtime->gridDomains();
        for (std::size_t di = 0; di < domains.size(); ++di) {
            if (domains[di].name == info.name) {
                grid_removed = g_ctx->scene.removeSimulationGridDomain(si, di);
                break;
            }
        }
    }
    // Gas domains report id 0 because they own no FluidObject; do not hand that
    // to removeFluidObject, which would be a lookup for an unrelated object.
    const bool object_removed =
        (info.type == "gas") ? false : g_ctx->scene.removeFluidObject(info.id);
    if (!grid_removed && !object_removed)
        return Result::fail("failed to remove fluid domain: " + info.name);
    return Result::success();
}

Result seedFluidParticles(const std::string& domain_id_or_name, Vec3 seed_min, Vec3 seed_max,
                           int particles_per_cell, bool replace, bool persistent) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    rtapi::FluidDomainInfo info;
    if (!getFluidDomain(domain_id_or_name, info).ok) {
        return Result::fail("fluid domain not found: " + domain_id_or_name);
    }

    // `fluid.get` and every simulation authoring facade resolve against the
    // active particle system. Prefer that same runtime here; scanning the scene
    // from index zero first could seed a stale preset system carrying the same
    // domain name while get() honestly reported zero from the active one.
    const auto active_runtime = g_ctx->scene.getParticleSimulationSystem();
    bool grid_seeded = active_runtime &&
        active_runtime->seedFluidDomainParticles(
            info.name, seed_min, seed_max, particles_per_cell, replace,
            persistent);
    for (auto& system : g_ctx->scene.particle_systems) {
        if (!grid_seeded && system.runtime &&
            system.runtime != active_runtime &&
            system.runtime->seedFluidDomainParticles(
                info.name, seed_min, seed_max, particles_per_cell, replace,
                persistent)) {
            grid_seeded = true;
            break;
        }
    }
    if (!grid_seeded) {
        return Result::fail("authoritative fluid grid domain not found: " + info.name);
    }

    // FluidObject is a legacy authoring/render mirror. It must never own a
    // second live copy of a grid-domain seed: SimulationWorld steps both
    // systems, so the duplicate ignored domain motion and appeared as a second
    // stream passing through the real granular block.
    for (auto& obj : g_ctx->scene.fluid_objects) {
        if (obj.name != info.name) continue;
        obj.seed_min = Vec3::min(seed_min, seed_max);
        obj.seed_max = Vec3::max(seed_min, seed_max);
        obj.seed_particles_per_cell = std::clamp(particles_per_cell, 1, 64);
        obj.replace_on_seed = replace;
        obj.pending_seed = false;
        obj.resetState();
        break;
    }

    // Seed is an executed initial-state action, not merely a descriptor edit.
    // Drop stale baked frames, then rebase the signature watcher on this live
    // result so its next UI tick does not interpret the new PPC/recipe fields
    // as another edit and reset a one-shot seed back to zero.
    invalidateScriptSimulation();
    g_ctx->scene.preserveScriptSimulationPreview();
    return Result::success();
}

Result clearFluidParticles(const std::string& domain_id_or_name,
                           bool clear_seed_recipe) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    rtapi::FluidDomainInfo info;
    if (!getFluidDomain(domain_id_or_name, info).ok) {
        return Result::fail("fluid domain not found: " + domain_id_or_name);
    }

    const auto active_runtime = g_ctx->scene.getParticleSimulationSystem();
    bool grid_cleared = active_runtime &&
        active_runtime->clearFluidDomainParticles(info.name,
                                                  clear_seed_recipe);
    for (auto& system : g_ctx->scene.particle_systems) {
        if (!grid_cleared && system.runtime &&
            system.runtime != active_runtime &&
            system.runtime->clearFluidDomainParticles(info.name,
                                                       clear_seed_recipe)) {
            grid_cleared = true;
            break;
        }
    }

    bool mirror_cleared = false;
    for (auto& obj : g_ctx->scene.fluid_objects) {
        if (obj.name == info.name) {
            obj.pending_seed = false;
            obj.resetState();
            mirror_cleared = true;
            break;
        }
    }
    if (!grid_cleared && !mirror_cleared)
        return Result::fail("fluid domain not found: " + info.name);
    invalidateScriptSimulation();
    g_ctx->scene.preserveScriptSimulationPreview();
    return Result::success();
}

Result updateFluidDomain(const std::string& domain_id_or_name,
                         const Vec3* domain_min, const Vec3* domain_max,
                         const float* voxel_size, const std::string* render_mode,
                         const std::string* backend, const std::string* boundary,
                         const std::string* preset,
                         const float* kinematic_viscosity,
                         const int* viscosity_sweeps,
                         const float* viscosity_wall_slip,
                         const std::string* surface_material,
                         const float* surface_offset_voxels,
                         const float* pore_amount,
                         const float* pore_scale,
                         const float* pore_detail,
                         const int* coord_space,
                         const int* uvw_refresh_period,
                         const bool* solid_phase,
                         const float* solid_phase_fill,
                         const bool* enabled, const bool* visible,
                         const bool* granular_enabled,
                         const float* granular_friction_angle_degrees,
                         const float* granular_cohesion,
                         const float* granular_dilatancy_degrees,
                         const float* granular_young_modulus,
                         const float* granular_poisson_ratio,
                         const float* granular_tensile_cutoff,
                         const float* granular_hardening,
                         const float* granular_fracture_strain,
                         const float* granular_damage_rate,
                         const float* granular_healing_rate,
                         const bool* granular_rebonding,
                         const int* granular_max_solver_substeps,
                         const float* granular_softening_temperature,
                         const float* granular_softening_range,
                         const float* granular_residual_strength,
                         const float* granular_tack_peak,
                         const float* granular_thermal_conductivity) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (surface_offset_voxels &&
        (!std::isfinite(*surface_offset_voxels) ||
         *surface_offset_voxels < -0.75f || *surface_offset_voxels > 1.25f)) {
        return Result::fail("surface_offset_voxels must be finite and in [-0.75, 1.25]");
    }

    rtapi::FluidDomainInfo info;
    if (!getFluidDomain(domain_id_or_name, info).ok) {
        return Result::fail("fluid domain not found: " + domain_id_or_name);
    }

    RayTrophiSim::Fluid::FluidObject* obj = nullptr;
    for (auto& fo : g_ctx->scene.fluid_objects) {
        if (fo.id == info.id) { obj = &fo; break; }
    }

    auto& p_sys_upd = g_ctx->scene.ensureParticleSimulationSystem();
    RayTrophiSim::SimulationGridDomainDesc* grid_dom = nullptr;
    for (auto& d : p_sys_upd.gridDomains()) {
        if (d.name == info.name) { grid_dom = &d; break; }
    }
    if (!obj && !grid_dom) return Result::fail("fluid domain not found");

    if (obj) {
        if (domain_min) { obj->domain_min = *domain_min; obj->grid_dirty = true; }
        if (domain_max) { obj->domain_max = *domain_max; obj->grid_dirty = true; }
        if (voxel_size && *voxel_size > 0.001f) {
            obj->voxel_size = *voxel_size;
            obj->grid_dirty = true;
        }
    }
    if (grid_dom) {
        if (domain_min) grid_dom->bounds_min = *domain_min;
        if (domain_max) grid_dom->bounds_max = *domain_max;
        if (voxel_size && *voxel_size > 0.001f) grid_dom->voxel_size = *voxel_size;
    }

    if (render_mode) {
        std::string rm = *render_mode;
        std::transform(rm.begin(), rm.end(), rm.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        // ★★★ UNKNOWN VALUES ARE REJECTED, NOT FOLDED INTO 'volume'. Every
        // unrecognised string used to land on Volume — which is not a valid
        // liquid mode and is normalised to the isosurface downstream. So
        // render_mode="splat" (an entirely reasonable guess) silently produced
        // the SURFACE: the opposite of the request, with no error, and the
        // script's own read-back agreeing with the mistake.
        //
        // ★★ The two aliases exist because this project already has two words
        // for these two things: a substance's `representation` says
        // "splat"/"sdf" and a domain's render_mode says "particles"/"surface".
        // One vocabulary for one question — the panel confusion the user
        // reported starts here, in the API that taught the two names.
        RayTrophiSim::Fluid::FluidRenderMode grid_mode;
        if (rm == "surface" || rm == "sdf") {
            grid_mode = RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
        } else if (rm == "particles" || rm == "splat") {
            grid_mode = RayTrophiSim::Fluid::FluidRenderMode::Particles;
        } else if (rm == "volume") {
            // Legal for GAS only; a liquid domain normalises it to the
            // isosurface where it is consumed, so it is accepted but never a
            // way to ask a liquid for something new.
            grid_mode = RayTrophiSim::Fluid::FluidRenderMode::Volume;
        } else {
            return Result::fail(
                "render_mode '" + *render_mode + "' is not a render mode. "
                "Use 'particles' (alias 'splat'), 'surface' (alias 'sdf'), or "
                "'volume' for gas domains.");
        }
        if (grid_dom) grid_dom->fluid_render_mode = grid_mode;
        if (obj) {
            obj->render_mode = grid_mode;
        }
    }

    // ★ Rheology writes go to BOTH representations. The grid domain descriptor is
    // what the solver actually steps; the FluidObject is the authoring-side copy.
    // Writing only the FluidObject — which is what this did — meant a script
    // asking for "honey" on a grid-domain fluid got success() and no honey. A
    // silent no-op is worse than a rejection: nothing distinguishes it from a
    // preset that simply does not look like much, which is precisely how the
    // broken viscosity dial survived this long.
    if (preset) {
        std::string p = *preset;
        std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        using FluidPreset = RayTrophiSim::Fluid::APICSolverParams::FluidPreset;
        bool known = true;
        FluidPreset chosen = FluidPreset::Custom;
        if      (p == "water")     chosen = FluidPreset::Water;
        else if (p == "oil")       chosen = FluidPreset::Oil;
        else if (p == "mud")       chosen = FluidPreset::Mud;
        else if (p == "honey")     chosen = FluidPreset::Honey;
        else if (p == "lava")      chosen = FluidPreset::Lava;
        else if (p == "sand")      chosen = FluidPreset::Sand;
        else if (p == "chocolate") chosen = FluidPreset::Chocolate;
        else if (p == "wet_sand" || p == "wetsand")           chosen = FluidPreset::WetSand;
        else if (p == "gravel")                               chosen = FluidPreset::Gravel;
        else if (p == "cohesive_soil" || p == "cohesivesoil") chosen = FluidPreset::CohesiveSoil;
        else if (p == "molten_plastic" || p == "moltenplastic" || p == "plastic") chosen = FluidPreset::MoltenPlastic;
        // ★ "custom" is what fluid.get REPORTS for a hand-tuned domain, so
        // rejecting it here broke the obvious round trip: read a domain, write
        // it back, get an error on a value this very API produced. It applies
        // nothing (applyPreset(Custom) returns early) and is accepted as the
        // explicit "leave the material alone" request it reads as.
        else if (p == "custom")                               chosen = FluidPreset::Custom;
        else known = false;
        if (!known) {
            return Result::fail("unknown fluid preset: " + *preset +
                                " (water, oil, mud, honey, lava, sand, chocolate,"
                                " wet_sand, gravel, cohesive_soil, molten_plastic, custom)");
        }
        if (obj)      obj->params.applyPreset(chosen);
        if (grid_dom) grid_dom->fluid_params.applyPreset(chosen);
    }

    if (boundary) {
        std::string b = *boundary;
        std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        const auto grid_boundary = b == "open"
            ? RayTrophiSim::SimulationGridDomainBoundaryMode::Open
            : (b == "periodic" ? RayTrophiSim::SimulationGridDomainBoundaryMode::Periodic
                                : RayTrophiSim::SimulationGridDomainBoundaryMode::Closed);
        if (grid_dom) grid_dom->boundary_mode = grid_boundary;
        if (obj) {
            if (b == "open") obj->params.boundary = RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Open;
            else if (b == "periodic") obj->params.boundary = RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Periodic;
            else obj->params.boundary = RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Closed;
        }
    }

    // Hand-setting any rheology field means the domain is no longer the named
    // material — mirror the UI's rule so a script and a slider leave the same
    // state behind.
    auto markCustom = [&]() {
        using FluidPreset = RayTrophiSim::Fluid::APICSolverParams::FluidPreset;
        if (obj)      obj->params.current_preset      = FluidPreset::Custom;
        if (grid_dom) grid_dom->fluid_params.current_preset = FluidPreset::Custom;
    };
    if (kinematic_viscosity) {
        const float v = std::max(0.0f, *kinematic_viscosity);
        if (obj)      obj->params.kinematic_viscosity = v;
        if (grid_dom) grid_dom->fluid_params.kinematic_viscosity = v;
        markCustom();
    }
    if (viscosity_sweeps) {
        const int s = std::clamp(*viscosity_sweeps, 1, 64);
        if (obj)      obj->params.viscosity_sweeps = s;
        if (grid_dom) grid_dom->fluid_params.viscosity_sweeps = s;
        markCustom();
    }
    if (viscosity_wall_slip) {
        const float w = std::clamp(*viscosity_wall_slip, 0.0f, 1.0f);
        if (obj)      obj->params.viscosity_wall_slip = w;
        if (grid_dom) grid_dom->fluid_params.viscosity_wall_slip = w;
        markCustom();
    }
    auto applyGranularPatch = [&](RayTrophiSim::Fluid::APICSolverParams& p) {
        if (granular_enabled) p.granular_enabled = *granular_enabled;
        if (granular_friction_angle_degrees) p.granular_friction_angle_degrees = *granular_friction_angle_degrees;
        if (granular_cohesion) p.granular_cohesion = *granular_cohesion;
        if (granular_dilatancy_degrees) p.granular_dilatancy_degrees = *granular_dilatancy_degrees;
        if (granular_young_modulus) p.granular_young_modulus = *granular_young_modulus;
        if (granular_poisson_ratio) p.granular_poisson_ratio = *granular_poisson_ratio;
        if (granular_tensile_cutoff) p.granular_tensile_cutoff = *granular_tensile_cutoff;
        if (granular_hardening) p.granular_hardening = *granular_hardening;
        if (granular_fracture_strain) p.granular_fracture_strain = *granular_fracture_strain;
        if (granular_damage_rate) p.granular_damage_rate = *granular_damage_rate;
        if (granular_healing_rate) p.granular_healing_rate = *granular_healing_rate;
        if (granular_rebonding) p.granular_rebonding = *granular_rebonding;
        if (granular_max_solver_substeps) p.granular_max_solver_substeps = *granular_max_solver_substeps;
        if (granular_softening_temperature) p.granular_softening_temperature = *granular_softening_temperature;
        if (granular_softening_range) p.granular_softening_range = *granular_softening_range;
        if (granular_residual_strength) p.granular_residual_strength = *granular_residual_strength;
        if (granular_tack_peak) p.granular_tack_peak = *granular_tack_peak;
        if (granular_thermal_conductivity) p.granular_thermal_conductivity = *granular_thermal_conductivity;
        p.sanitizeGranularMaterial();
    };
    const bool granular_patch = granular_enabled || granular_friction_angle_degrees ||
        granular_cohesion || granular_dilatancy_degrees || granular_young_modulus ||
        granular_poisson_ratio || granular_tensile_cutoff || granular_hardening ||
        granular_fracture_strain || granular_damage_rate || granular_healing_rate ||
        granular_rebonding || granular_max_solver_substeps ||
        granular_softening_temperature || granular_softening_range ||
        granular_residual_strength || granular_tack_peak ||
        granular_thermal_conductivity;
    if (granular_patch) {
        if (obj) applyGranularPatch(obj->params);
        if (grid_dom) applyGranularPatch(grid_dom->fluid_params);
        markCustom();
    }

    if (backend) {
        std::string dev = *backend;
        std::transform(dev.begin(), dev.end(), dev.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        RayTrophiSim::SimulationDomainBackend be = RayTrophiSim::SimulationDomainBackend::CPU_Dense;
        if (dev == "gpu" || dev == "gpu_compute" || dev == "cuda" || dev == "compute") {
            be = RayTrophiSim::SimulationDomainBackend::GPU_Compute;
        } else if (dev == "vulkan" || dev == "gpu_vulkan") {
            be = RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
        } else if (dev == "cpu_sparse" || dev == "sparse" || dev == "vdb") {
            be = RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB;
        }

        if (grid_dom) grid_dom->backend = be;
    }

    if (surface_material && grid_dom) {
        if (surface_material->empty()) {
            grid_dom->fluid_surface_material_id = -1;
        } else {
            auto& mm = MaterialManager::getInstance();
            const auto& mats = mm.getAllMaterials();
            int found = -1;
            for (std::size_t mi = 0; mi < mats.size(); ++mi) {
                if (mats[mi] && mats[mi]->materialName == *surface_material) {
                    found = static_cast<int>(mi);
                    break;
                }
            }
            // Fail loudly. A missing material silently left as the built-in
            // dielectric would look like "the material had no effect", which is
            // exactly the report nobody can act on.
            if (found < 0) {
                return Result::fail("material not found: " + *surface_material);
            }
            grid_dom->fluid_surface_material_id = found;
        }
        g_ctx->scene.refreshFluidSurfaceMaterial();
    }

    // Canonical geometric fullness. Apply to both authoring representations so
    // switching between the legacy FluidObject and the live grid domain cannot
    // resurrect an older silhouette. This deliberately does not mark rheology
    // Custom: it is reconstruction state, not material physics.
    if (surface_offset_voxels) {
        if (obj) obj->level_set_params.surface_offset_voxels = *surface_offset_voxels;
        if (grid_dom) grid_dom->fluid_level_set_params.surface_offset_voxels = *surface_offset_voxels;

        // scene_data hashes the level-set parameters, so the next bridge update
        // rebuilds the field without a simulation reset. Clear both progressive
        // accumulators or the new silhouette would be averaged with the old one.
        g_ctx->renderer.resetCPUAccumulation();
        if (g_ctx->backend_ptr) g_ctx->backend_ptr->resetAccumulation();
        g_ctx->start_render = true;
    }

    // Procedural porosity. Clamped the same way the GPU packer clamps, so a
    // scripted value and a panel value cannot mean different things.
    if (grid_dom && pore_amount) {
        grid_dom->fluid_surface_pore_amount = (std::max)(0.0f, *pore_amount);
    }
    if (grid_dom && pore_scale) {
        grid_dom->fluid_surface_pore_scale = (std::max)(1e-4f, *pore_scale);
    }
    if (grid_dom && pore_detail) {
        grid_dom->fluid_surface_pore_detail =
            (std::max)(0.0f, (std::min)(1.0f, *pore_detail));
    }
    if (grid_dom && coord_space) {
        // Clamped rather than rejected. The shader clamps too, so a stray value
        // could never corrupt anything — but accepting it silently at one layer
        // and snapping it at another means fluid.get reports a number the render
        // does not use. Snap here, once, so the read-back is the truth.
        grid_dom->fluid_surface_coord_space =
            (std::max)(0, (std::min)(2, *coord_space));
    }
    if (grid_dom && solid_phase) {
        grid_dom->fluid_solid_phase_enabled = *solid_phase;
    }
    if (grid_dom && solid_phase_fill) {
        // ★ REJECTED, not clamped, outside a sane band. This dial decides
        // whether matter blocks flow at all; silently snapping 0 to 0.01 would
        // let a script believe it had disabled blocking while the solid kept
        // walling the domain off.
        if (*solid_phase_fill < 0.01f || *solid_phase_fill > 4.0f)
            return Result::fail("solid_phase_fill must be in [0.01, 4.0] (fraction of the seed density)");
        grid_dom->fluid_solid_phase_fill = *solid_phase_fill;
    }
    if (grid_dom && uvw_refresh_period) {
        // Same snap-here rule as coord_space. Below 2 there is no second half to
        // stagger against, so both generations would reset on the same step and
        // collapse into one — the stretch cure would report as on and do
        // nothing, which is the failure mode worth spending a clamp on.
        grid_dom->fluid_params.uvw_refresh_period =
            (std::max)(2, (std::min)(20000, *uvw_refresh_period));
    }
    // ★ PUSH AND RESET. Writing the descriptor is not the same as changing the
    // render: these values live in the volume table, and until
    // refreshFluidSurfaceMaterial copies them onto the render volume (and sets
    // g_gas_volumes_dirty) the shader is still reading the old ones.
    //
    // Clearing the accumulator matters just as much and is easier to miss. A
    // converged image is an AVERAGE over past samples; without a reset the new
    // value only dilutes into it, so a script that sets a knob and renders
    // immediately captures mostly the OLD look. That failure is worst exactly
    // where it is least visible — an automated visual test, which converges
    // longer than a human would wait and therefore hides the change best.
    //
    // Only when something actually changed: an unconditional reset here would
    // let any fluid.set_param call (a viscosity tweak, an enable toggle) throw
    // away a converging render.
    if (grid_dom && (pore_amount || pore_scale || pore_detail || coord_space ||
                     uvw_refresh_period)) {
        g_ctx->scene.refreshFluidSurfaceMaterial();
        g_ctx->renderer.resetCPUAccumulation();
        if (g_ctx->backend_ptr) g_ctx->backend_ptr->resetAccumulation();
        g_ctx->start_render = true;
    }

    if (obj && enabled) obj->enabled = *enabled;
    if (obj && visible) obj->visible = *visible;
    if (grid_dom && enabled) grid_dom->enabled = *enabled;

    if (obj) obj->ensureGrid();
    return Result::success();
}

Result setFluidSubstanceMaterial(const std::string& domain_id_or_name,
                                 const std::string& substance,
                                 const std::string& material_name,
                                 const std::string* representation,
                                 const float* kinematic_viscosity,
                                 const float* miscibility,
                                 const std::string* phase) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (substance.empty())
        return Result::fail("substance name is required");

    Result found;
    // Same lookup the gas settings use; the descriptor list is shared and this
    // helper already reports a usable error for a bad name.
    RayTrophiSim::SimulationGridDomainDesc* dom =
        findGasDomainDesc(domain_id_or_name, found);
    if (!dom) return found;

    // Resolve the material FIRST. Writing the binding and then discovering the
    // material does not exist would leave a table entry pointing at nothing,
    // and the producer would silently fall back to the domain material — the
    // script would have "succeeded" and changed no pixel.
    int material_id = -1;
    bool clear = material_name.empty();
    if (!clear && material_name != "dielectric") {
        const auto& mats = MaterialManager::getInstance().getAllMaterials();
        bool resolved = false;
        for (std::size_t i = 0; i < mats.size(); ++i) {
            if (!mats[i] || mats[i]->materialName != material_name) continue;
            material_id = static_cast<int>(i);
            resolved = true;
            break;
        }
        if (!resolved)
            return Result::fail("material not found: " + material_name);
    }

    auto& table = dom->fluid_substance_materials;
    auto it = std::find_if(table.begin(), table.end(),
        [&](const RayTrophiSim::SimulationGridDomainDesc::SubstanceMaterial& b) {
            return b.substance == substance;
        });
    RayTrophiSim::Fluid::SubstanceRepresentation rep =
        RayTrophiSim::Fluid::SubstanceRepresentation::Inherit;
    if (representation) {
        std::string r;
        for (unsigned char ch : *representation) {
            if (ch == '_' || ch == '-' || std::isspace(ch)) continue;
            r.push_back(static_cast<char>(std::tolower(ch)));
        }
        if (r == "splat" || r == "particles")
            rep = RayTrophiSim::Fluid::SubstanceRepresentation::Splat;
        else if (r == "sdf" || r == "surface" || r == "surfacesdf")
            rep = RayTrophiSim::Fluid::SubstanceRepresentation::SurfaceSDF;
        else if (r != "inherit" && r != "domain")
            return Result::fail("representation must be inherit, splat, or sdf");
    }
    if (miscibility && (*miscibility < 0.0f || *miscibility > 1.0f))
        return Result::fail("miscibility must be in [0, 1]");

    RayTrophiSim::Fluid::SubstancePhase phase_value =
        RayTrophiSim::Fluid::SubstancePhase::Liquid;
    if (phase) {
        std::string p;
        for (unsigned char ch : *phase) {
            if (ch == '_' || ch == '-' || std::isspace(ch)) continue;
            p.push_back(static_cast<char>(std::tolower(ch)));
        }
        if (p == "solid" || p == "frozen")
            phase_value = RayTrophiSim::Fluid::SubstancePhase::Solid;
        // ★ REJECTED, not snapped to liquid. Phase decides whether matter
        // blocks flow, so a typo silently read as "liquid" would leave a script
        // believing it froze something while the sim kept pouring. Unlike the
        // look controls elsewhere in this file, guessing here changes physics.
        else if (p != "liquid" && p != "fluid")
            return Result::fail("phase must be liquid or solid");
    }

    // ★ A physics-only call must NOT delete the binding. `clear` means "no
    // material name given"; combined with a viscosity, miscibility or phase
    // argument that is a request to author physics on an existing substance,
    // and erasing the row would throw away the material binding the caller
    // never mentioned.
    const bool physics_only = clear && !representation &&
                              (kinematic_viscosity || miscibility || phase);
    if (clear && !representation && !physics_only) {
        if (it != table.end()) table.erase(it);
    } else if (it != table.end()) {
        if (!clear) it->material_id = material_id;
        if (representation) it->representation = rep;
        // ★ Negative is a MEANINGFUL value here (inherit the domain), so it is
        // stored as given rather than clamped up to 0 — clamping would make
        // "inherit" unauthorable through the script layer while the panel could
        // still set it, which is exactly the parity gap the project's first rule
        // exists to prevent.
        if (kinematic_viscosity) it->kinematic_viscosity = *kinematic_viscosity;
        if (miscibility)         it->miscibility = *miscibility;
        if (phase)               it->phase = phase_value;
    } else {
        if (table.size() >= RayTrophiSim::Fluid::kMaxFluidSubstanceMaterials) {
            return Result::fail(
                "this domain already binds the maximum number of substances (" +
                std::to_string(RayTrophiSim::Fluid::kMaxFluidSubstanceMaterials) +
                "); the composition field carries two per cell and an unbounded "
                "table would make the gather cost invisible");
        }
        RayTrophiSim::SimulationGridDomainDesc::SubstanceMaterial entry;
        entry.substance = substance;
        entry.material_id = material_id;
        entry.representation = rep;
        if (kinematic_viscosity) entry.kinematic_viscosity = *kinematic_viscosity;
        if (miscibility)         entry.miscibility = *miscibility;
        if (phase)               entry.phase = phase_value;
        table.push_back(entry);
    }

    // ★ PUSH AND RESET, same rule as every other look control here: the binding
    // only reaches the shader through a surface rebuild, and a converged image
    // would otherwise dilute the change instead of showing it — worst exactly
    // in an automated visual test, which converges longer than a human waits.
    g_ctx->scene.refreshFluidSurfaceMaterial();
    g_ctx->renderer.resetCPUAccumulation();
    if (g_ctx->backend_ptr) g_ctx->backend_ptr->resetAccumulation();
    g_ctx->start_render = true;
    return Result::success();
}

Result getGasDomainSettings(const std::string& domain_id_or_name, GasDomainSettings& out) {
    if (!g_ctx) return notBound();
    FluidDomainInfo info;
    Result found = getFluidDomain(domain_id_or_name, info);
    if (!found.ok) return found;

    auto& domains = g_ctx->scene.ensureParticleSimulationSystem().gridDomains();
    auto it = std::find_if(domains.begin(), domains.end(),
        [&info](const auto& d) { return d.name == info.name; });
    if (it == domains.end() || it->type != RayTrophiSim::SimulationDomainType::Gas)
        return Result::fail("gas domain not found: " + domain_id_or_name);

    switch (it->quality_profile) {
        case RayTrophiSim::SimulationDomainQualityProfile::Interactive: out.quality_profile = "interactive"; break;
        case RayTrophiSim::SimulationDomainQualityProfile::Preview: out.quality_profile = "preview"; break;
        case RayTrophiSim::SimulationDomainQualityProfile::Final: out.quality_profile = "final"; break;
        case RayTrophiSim::SimulationDomainQualityProfile::Cinema: out.quality_profile = "cinema"; break;
        default: out.quality_profile = "custom"; break;
    }
    out.resource_budget_mb = it->resource_budget_mb;
    out.enforce_resource_budget = it->enforce_resource_budget;
    out.use_sparse_tiles = it->use_sparse_tiles;
    out.render_to_nanovdb = it->render_to_nanovdb;
    out.fire_enabled = it->fire_enabled;
    out.ignition_temperature = it->ignition_temperature;
    out.burn_rate = it->burn_rate;
    out.heat_release = it->heat_release;
    out.smoke_generation = it->smoke_generation;
    out.flame_dissipation = it->flame_dissipation;
    out.fire_max_temperature = it->fire_max_temperature;
    out.structural_coupling_enabled = it->structural_coupling_enabled;
    out.structural_pressure_scale = it->structural_pressure_scale;
    out.structural_min_intensity = it->structural_min_intensity;
    out.structural_event_interval = it->structural_event_interval;
    out.buoyancy_heat = it->gas_buoyancy_heat;
    out.buoyancy_density = it->gas_buoyancy_density;
    out.vorticity = it->gas_vorticity;
    out.fire_expansion = it->fire_expansion;
    out.turbulence_strength = it->turbulence_strength;
    out.turbulence_scale = it->turbulence_scale;
    out.turbulence_octaves = it->turbulence_octaves;
    out.turbulence_lacunarity = it->turbulence_lacunarity;
    out.turbulence_persistence = it->turbulence_persistence;
    out.turbulence_speed = it->turbulence_speed;
    return Result::success();
}

Result updateGasDomainSettings(const std::string& domain_id_or_name, const GasDomainSettings& s) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    FluidDomainInfo info;
    Result found = getFluidDomain(domain_id_or_name, info);
    if (!found.ok) return found;

    auto& domains = g_ctx->scene.ensureParticleSimulationSystem().gridDomains();
    auto it = std::find_if(domains.begin(), domains.end(),
        [&info](const auto& d) { return d.name == info.name; });
    if (it == domains.end() || it->type != RayTrophiSim::SimulationDomainType::Gas)
        return Result::fail("gas domain not found: " + domain_id_or_name);

    std::string profile = s.quality_profile;
    std::transform(profile.begin(), profile.end(), profile.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (profile == "interactive") it->quality_profile = RayTrophiSim::SimulationDomainQualityProfile::Interactive;
    else if (profile == "preview") it->quality_profile = RayTrophiSim::SimulationDomainQualityProfile::Preview;
    else if (profile == "final") it->quality_profile = RayTrophiSim::SimulationDomainQualityProfile::Final;
    else if (profile == "cinema") it->quality_profile = RayTrophiSim::SimulationDomainQualityProfile::Cinema;
    else if (profile == "custom") it->quality_profile = RayTrophiSim::SimulationDomainQualityProfile::Custom;
    else return Result::fail("unknown gas quality_profile: " + s.quality_profile);

    it->resource_budget_mb = s.resource_budget_mb;
    it->enforce_resource_budget = s.enforce_resource_budget;
    it->use_sparse_tiles = s.use_sparse_tiles;
    it->render_to_nanovdb = s.render_to_nanovdb;
    it->fire_enabled = s.fire_enabled;
    it->ignition_temperature = std::max(0.0f, s.ignition_temperature);
    it->burn_rate = std::max(0.0f, s.burn_rate);
    it->heat_release = std::max(0.0f, s.heat_release);
    it->smoke_generation = std::max(0.0f, s.smoke_generation);
    it->flame_dissipation = std::max(0.0f, s.flame_dissipation);
    it->fire_max_temperature = std::max(0.0f, s.fire_max_temperature);
    it->structural_coupling_enabled = s.structural_coupling_enabled;
    it->structural_pressure_scale = std::max(0.0f, s.structural_pressure_scale);
    it->structural_min_intensity = std::max(0.0f, s.structural_min_intensity);
    it->structural_event_interval = std::max(1.0f / 120.0f, s.structural_event_interval);
    it->gas_buoyancy_heat = s.buoyancy_heat;
    it->gas_buoyancy_density = s.buoyancy_density;
    it->gas_vorticity = std::max(0.0f, s.vorticity);
    it->fire_expansion = std::max(0.0f, s.fire_expansion);
    it->turbulence_strength = std::max(0.0f, s.turbulence_strength);
    it->turbulence_scale = std::max(0.001f, s.turbulence_scale);
    it->turbulence_octaves = std::clamp(s.turbulence_octaves, 1, 8);
    it->turbulence_lacunarity = std::max(1.0f, s.turbulence_lacunarity);
    it->turbulence_persistence = std::clamp(s.turbulence_persistence, 0.0f, 1.0f);
    it->turbulence_speed = s.turbulence_speed;
    return Result::success();
}

Result getCombustibleFluidSettings(
    const std::string& domain_id_or_name,
    CombustibleFluidSettings& out) {
    if (!g_ctx) return notBound();
    FluidDomainInfo info;
    Result found=getFluidDomain(domain_id_or_name,info);
    if(!found.ok) return found;
    auto& domains=g_ctx->scene.ensureParticleSimulationSystem().gridDomains();
    auto it=std::find_if(domains.begin(),domains.end(),
        [&info](const auto& d){return d.name==info.name;});
    if(it==domains.end() ||
       it->type!=RayTrophiSim::SimulationDomainType::Fluid)
        return Result::fail("fluid domain not found: "+domain_id_or_name);
    out.enabled=it->fluid_flammable;
    switch (it->fluid_params.chemistry_preset) {
        case RayTrophiSim::Fluid::FluidChemistryPreset::Water: out.chemistry_preset="water"; break;
        case RayTrophiSim::Fluid::FluidChemistryPreset::Gasoline: out.chemistry_preset="gasoline"; break;
        case RayTrophiSim::Fluid::FluidChemistryPreset::Alcohol: out.chemistry_preset="alcohol"; break;
        case RayTrophiSim::Fluid::FluidChemistryPreset::Oil: out.chemistry_preset="oil"; break;
        case RayTrophiSim::Fluid::FluidChemistryPreset::Plastic: out.chemistry_preset="plastic"; break;
        case RayTrophiSim::Fluid::FluidChemistryPreset::Wax: out.chemistry_preset="wax"; break;
        case RayTrophiSim::Fluid::FluidChemistryPreset::Custom: out.chemistry_preset="custom"; break;
        default: out.chemistry_preset="inert"; break;
    }
    out.auto_ignite=it->fluid_auto_ignite;
    out.ignition_temperature=it->fluid_ignition_temperature;
    out.evaporation_rate=it->fluid_evaporation_rate;
    out.surface_fuel_capacity=it->fluid_surface_fuel_capacity;
    out.heat_release=it->fluid_combustion_heat_release;
    out.smoke_yield=it->fluid_combustion_smoke_yield;
    out.surface_cooling=it->fluid_surface_cooling;
    return Result::success();
}

Result updateCombustibleFluidSettings(
    const std::string& domain_id_or_name,
    const CombustibleFluidSettings& s) {
    if(!g_ctx) return notBound();
    if(renderJobActive())
        return Result::fail("scene is locked by the final render job");
    FluidDomainInfo info;
    Result found=getFluidDomain(domain_id_or_name,info);
    if(!found.ok) return found;
    auto& domains=g_ctx->scene.ensureParticleSimulationSystem().gridDomains();
    auto it=std::find_if(domains.begin(),domains.end(),
        [&info](const auto& d){return d.name==info.name;});
    if(it==domains.end() ||
       it->type!=RayTrophiSim::SimulationDomainType::Fluid)
        return Result::fail("fluid domain not found: "+domain_id_or_name);
    std::string chemistry = s.chemistry_preset;
    std::transform(chemistry.begin(), chemistry.end(), chemistry.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    if (chemistry == "water") it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Water);
    else if (chemistry == "gasoline") it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Gasoline);
    else if (chemistry == "alcohol") it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Alcohol);
    else if (chemistry == "oil") it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Oil);
    else if (chemistry == "plastic") it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Plastic);
    else if (chemistry == "wax") it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Wax);
    else if (chemistry == "custom") it->fluid_params.chemistry_preset=RayTrophiSim::Fluid::FluidChemistryPreset::Custom;
    else it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Inert);
    // The chemistry profile owns the physical interaction mode.  In particular,
    // water must enter the extinguishing path even when the legacy `enabled`
    // flag is used by an older script.
    const auto& profile = it->fluid_params.fuel_profile;
    it->fluid_extinguishing = profile.extinguishing;
    it->fluid_flammable = s.enabled && !profile.extinguishing;
    it->fluid_cooling_power = profile.cooling_power;
    it->fluid_oxygen_dilution = profile.oxygen_dilution;
    it->fluid_auto_ignite=s.auto_ignite;
    it->fluid_ignition_temperature=std::max(0.0f,s.ignition_temperature);
    it->fluid_evaporation_rate=std::max(0.0f,s.evaporation_rate);
    it->fluid_surface_fuel_capacity=std::max(0.0f,s.surface_fuel_capacity);
    it->fluid_combustion_heat_release=std::max(0.0f,s.heat_release);
    it->fluid_combustion_smoke_yield=std::max(0.0f,s.smoke_yield);
    it->fluid_surface_cooling=std::max(0.0f,s.surface_cooling);
    return Result::success();
}

// Shared with RtApiParticle.cpp (declared in RtApiInternal.h): the emitters,
// colliders and grid domains all hang off the SAME runtime, so both facades
// must reach it through one accessor and invalidate it the same way.
RayTrophiSim::ParticleSimulationSystem& scriptSimulationRuntime() {
    return g_ctx->scene.ensureParticleSimulationSystem();
}

void invalidateScriptSimulation() {
    g_ctx->scene.clearSimFrameCache();
    g_ctx->scene.requestSimulationTimelineRenderResync();
}

namespace {

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

int scriptDomainIndex(RayTrophiSim::ParticleSimulationSystem& runtime,
                      const std::string& domain_name) {
    const auto& domains = runtime.gridDomains();
    for (std::size_t i = 0; i < domains.size(); ++i) {
        if (domains[i].name == domain_name) return static_cast<int>(i);
    }
    return -1;
}

RayTrophiSim::SimulationFlowSourceMode flowModeFromString(const std::string& value,
                                                          bool& ok) {
    const std::string mode = lowerCopy(value);
    ok = true;
    if (mode == "point") return RayTrophiSim::SimulationFlowSourceMode::Point;
    if (mode == "object_bounds" || mode == "bounds")
        return RayTrophiSim::SimulationFlowSourceMode::ObjectBounds;
    if (mode == "mesh_surface" || mode == "surface")
        return RayTrophiSim::SimulationFlowSourceMode::MeshSurface;
    ok = false;
    return RayTrophiSim::SimulationFlowSourceMode::Point;
}

std::string flowModeToString(RayTrophiSim::SimulationFlowSourceMode mode) {
    switch (mode) {
        case RayTrophiSim::SimulationFlowSourceMode::ObjectBounds: return "object_bounds";
        case RayTrophiSim::SimulationFlowSourceMode::MeshSurface: return "mesh_surface";
        default: return "point";
    }
}

SimulationFlowSourceInfo flowInfoFromDesc(
    const RayTrophiSim::SimulationFlowSourceDesc& source,
    const std::vector<RayTrophiSim::SimulationGridDomainDesc>& domains) {
    SimulationFlowSourceInfo out;
    out.name = source.name;
    if (source.domain_index >= 0 &&
        source.domain_index < static_cast<int>(domains.size()))
        out.domain = domains[static_cast<std::size_t>(source.domain_index)].name;
    out.source_mode = flowModeToString(source.source_mode);
    out.source_object = source.source_name;
    out.enabled = source.enabled;
    out.parent_object = source.parent_object;
    out.velocity_space =
        (source.velocity_space == RayTrophiSim::SimulationEmissionVelocitySpace::World)
            ? "world" : "local";
    out.inherit_velocity = source.inherit_velocity;
    out.position = source.position;
    out.velocity = source.velocity;
    out.radius = source.radius;
    out.velocity_coupling = source.velocity_coupling;
    out.density = source.density;
    out.temperature = source.temperature;
    out.fuel = source.fuel;
    out.falloff = source.falloff;
    out.fluid_particles_per_second = source.fluid_particles_per_second;
    out.fluid_velocity_spread = source.fluid_velocity_spread;
    out.fluid_emit_along_normal = source.fluid_emit_along_normal;
    out.fluid_substance = source.fluid_substance;
    out.use_time_limit = source.use_time_limit;
    out.start_time = source.start_time;
    out.end_time = source.end_time;
    out.use_particle_limit = source.use_particle_limit;
    out.max_emitted_particles = source.max_emitted_particles;
    return out;
}

Result flowDescFromInfo(const SimulationFlowSourceInfo& info,
                        RayTrophiSim::ParticleSimulationSystem& runtime,
                        RayTrophiSim::SimulationFlowSourceDesc& out) {
    const int domain_index = scriptDomainIndex(runtime, info.domain);
    if (domain_index < 0) return Result::fail("simulation domain not found: " + info.domain);
    bool mode_ok = false;
    out.name = info.name.empty() ? "Flow Source" : info.name;
    out.domain_index = domain_index;
    out.source_mode = flowModeFromString(info.source_mode, mode_ok);
    if (!mode_ok) return Result::fail("unknown flow source_mode: " + info.source_mode);
    out.source_name = info.source_object;
    out.enabled = info.enabled;
    out.parent_object = info.parent_object;
    {
        const std::string space = lowerCopy(info.velocity_space);
        if (space == "world") {
            out.velocity_space = RayTrophiSim::SimulationEmissionVelocitySpace::World;
        } else if (space.empty() || space == "local") {
            out.velocity_space = RayTrophiSim::SimulationEmissionVelocitySpace::Local;
        } else {
            return Result::fail("unknown flow velocity_space: " + info.velocity_space);
        }
    }
    out.inherit_velocity = info.inherit_velocity;
    out.position = info.position;
    out.velocity = info.velocity;
    out.radius = std::max(0.001f, info.radius);
    out.velocity_coupling = std::max(0.0f, info.velocity_coupling);
    out.density = std::max(0.0f, info.density);
    out.temperature = std::max(0.0f, info.temperature);
    out.fuel = std::max(0.0f, info.fuel);
    out.falloff = std::max(0.0f, info.falloff);
    out.fluid_particles_per_second = std::max(0.0f, info.fluid_particles_per_second);
    out.fluid_velocity_spread = std::max(0.0f, info.fluid_velocity_spread);
    out.fluid_emit_along_normal = info.fluid_emit_along_normal;
    out.fluid_substance = info.fluid_substance;
    out.use_time_limit = info.use_time_limit;
    out.start_time = info.start_time;
    out.end_time = std::max(info.start_time, info.end_time);
    out.use_particle_limit = info.use_particle_limit;
    out.max_emitted_particles = std::max(0, info.max_emitted_particles);
    return Result::success();
}

RayTrophiSim::ParticleColliderSourceMode colliderModeFromString(
    const std::string& value, bool& ok) {
    const std::string mode = lowerCopy(value);
    ok = true;
    if (mode == "plane" || mode == "plane_y") return RayTrophiSim::ParticleColliderSourceMode::PlaneY;
    if (mode == "sphere") return RayTrophiSim::ParticleColliderSourceMode::Sphere;
    if (mode == "capsule") return RayTrophiSim::ParticleColliderSourceMode::Capsule;
    if (mode == "aabb" || mode == "object_aabb") return RayTrophiSim::ParticleColliderSourceMode::ObjectAABB;
    if (mode == "obb" || mode == "object_obb") return RayTrophiSim::ParticleColliderSourceMode::ObjectOBB;
    if (mode == "mesh_sdf" || mode == "sdf") return RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF;
    if (mode == "convex" || mode == "convex_decomp") return RayTrophiSim::ParticleColliderSourceMode::ObjectConvexDecomp;
    if (mode == "mesh_bvh" || mode == "bvh") return RayTrophiSim::ParticleColliderSourceMode::ObjectMeshBVH;
    ok = false;
    return RayTrophiSim::ParticleColliderSourceMode::PlaneY;
}

std::string colliderModeToString(RayTrophiSim::ParticleColliderSourceMode mode) {
    switch (mode) {
        case RayTrophiSim::ParticleColliderSourceMode::ObjectAABB: return "aabb";
        case RayTrophiSim::ParticleColliderSourceMode::ObjectOBB: return "obb";
        case RayTrophiSim::ParticleColliderSourceMode::Sphere: return "sphere";
        case RayTrophiSim::ParticleColliderSourceMode::Capsule: return "capsule";
        case RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF: return "mesh_sdf";
        case RayTrophiSim::ParticleColliderSourceMode::ObjectConvexDecomp: return "convex";
        case RayTrophiSim::ParticleColliderSourceMode::ObjectMeshBVH: return "mesh_bvh";
        default: return "plane";
    }
}

SimulationColliderInfo colliderInfoFromDesc(const RayTrophiSim::ParticleColliderDesc& c) {
    SimulationColliderInfo out;
    out.name = c.name;
    out.source_mode = colliderModeToString(c.source_mode);
    out.source_object = c.source_name;
    out.enabled = c.enabled;
    out.fluid_collision_enabled = c.fluid_collision_enabled;
    out.plane_y = c.plane_y;
    out.sphere_center = c.sphere_center;
    out.sphere_radius = c.sphere_radius;
    out.capsule_start = c.capsule_start;
    out.capsule_end = c.capsule_end;
    out.capsule_radius = c.capsule_radius;
    out.bounds_min = c.bounds_min;
    out.bounds_max = c.bounds_max;
    out.friction = c.friction;
    out.restitution = c.restitution;
    out.thickness = c.thickness;
    out.sdf_resolution_mode = c.sdf_resolution_mode;
    out.sdf_ready = c.sdf_grid_data && !c.sdf_grid_data->empty();
    out.sdf_resolution = c.sdf_nx;
    out.gas_interaction_enabled = c.gas_interaction_enabled;
    out.gas_density_rate = c.gas_density_rate;
    out.gas_temperature_rate = c.gas_temperature_rate;
    out.gas_fuel_rate = c.gas_fuel_rate;
    out.gas_flame_rate = c.gas_flame_rate;
    out.gas_ignite_on_contact = c.gas_ignite_on_contact;
    out.msf_substance = c.msf_substance;
    out.msf_override_ignition = c.msf_override.override_ignition;
    out.msf_ignition_kelvin = c.msf_override.ignition_kelvin;
    out.msf_burn_rate_scale = c.msf_override.burn_rate_scale;
    out.msf_fuel_capacity_scale = c.msf_override.fuel_capacity_scale;
    out.msf_mask_resolution = c.msf_mask_resolution;
    out.msf_generate_char_mask = c.msf_generate_char_mask;
    out.msf_auto_transfer = c.msf_auto_transfer;
    out.msf_transfer_domain = c.msf_transfer_domain;
    out.msf_transfer_rate_kg_s = c.msf_transfer_rate_kg_s;
    out.msf_transfer_min_mass_kg = c.msf_transfer_min_mass_kg;
    out.msf_transfer_particles_per_kg = c.msf_transfer_particles_per_kg;
    out.msf_transfer_max_batch_particles = c.msf_transfer_max_batch_particles;
    out.msf_transfer_velocity = c.msf_transfer_velocity;
    out.msf_melt_flow_enabled = c.msf_melt_flow_enabled;
    out.msf_melt_height_loss = c.msf_melt_height_loss;
    out.msf_melt_spread = c.msf_melt_spread;
    out.msf_melt_sdf_refresh = c.msf_melt_sdf_refresh;
    out.msf_melt_sdf_revision_interval = c.msf_melt_sdf_revision_interval;
    out.msf_melt_sdf_change_threshold = c.msf_melt_sdf_change_threshold;
    return out;
}

Result colliderDescFromInfo(const SimulationColliderInfo& info,
                            RayTrophiSim::ParticleColliderDesc& out) {
    bool mode_ok = false;
    out.name = info.name.empty() ? "Simulation Collider" : info.name;
    out.source_mode = colliderModeFromString(info.source_mode, mode_ok);
    if (!mode_ok) return Result::fail("unknown collider source_mode: " + info.source_mode);
    out.source_name = info.source_object;
    out.enabled = info.enabled;
    out.fluid_collision_enabled = info.fluid_collision_enabled;
    out.plane_y = info.plane_y;
    out.sphere_center = info.sphere_center;
    out.sphere_radius = std::max(0.001f, info.sphere_radius);
    out.capsule_start = info.capsule_start;
    out.capsule_end = info.capsule_end;
    out.capsule_radius = std::max(0.001f, info.capsule_radius);
    out.bounds_min = Vec3::min(info.bounds_min, info.bounds_max);
    out.bounds_max = Vec3::max(info.bounds_min, info.bounds_max);
    out.friction = std::clamp(info.friction, 0.0f, 1.0f);
    out.restitution = std::clamp(info.restitution, 0.0f, 1.0f);
    out.thickness = std::max(0.0f, info.thickness);
    out.sdf_resolution_mode = std::clamp(info.sdf_resolution_mode, 0, 2);
    out.gas_interaction_enabled = info.gas_interaction_enabled;
    out.gas_density_rate = std::max(0.0f, info.gas_density_rate);
    out.gas_temperature_rate = std::max(0.0f, info.gas_temperature_rate);
    out.gas_fuel_rate = std::max(0.0f, info.gas_fuel_rate);
    out.gas_flame_rate = std::max(0.0f, info.gas_flame_rate);
    out.gas_ignite_on_contact = info.gas_ignite_on_contact;
    if (!info.msf_substance.empty()) {
        // Validate against the library rather than storing an unknown name that
        // would silently degrade to the "Custom" profile at simulation time.
        bool known = false;
        for (const auto& profile : RayTrophiSim::substanceLibrary()) {
            if (profile.name == info.msf_substance) { known = true; break; }
        }
        if (!known) return Result::fail("unknown msf_substance: " + info.msf_substance);
        out.msf_substance = info.msf_substance;
    }
    out.msf_override.override_ignition = info.msf_override_ignition;
    out.msf_override.ignition_kelvin = std::max(0.0f, info.msf_ignition_kelvin);
    out.msf_override.burn_rate_scale = std::max(0.0f, info.msf_burn_rate_scale);
    out.msf_override.fuel_capacity_scale = std::max(0.0f, info.msf_fuel_capacity_scale);
    out.msf_mask_resolution = std::clamp(info.msf_mask_resolution, 0, 4096);
    out.msf_generate_char_mask = info.msf_generate_char_mask;
    out.msf_auto_transfer = info.msf_auto_transfer;
    out.msf_transfer_domain = info.msf_transfer_domain;
    out.msf_transfer_rate_kg_s = std::max(0.0f, info.msf_transfer_rate_kg_s);
    out.msf_transfer_min_mass_kg = std::max(0.0f, info.msf_transfer_min_mass_kg);
    out.msf_transfer_particles_per_kg = std::clamp(
        info.msf_transfer_particles_per_kg, 1.0f, 100000.0f);
    out.msf_transfer_max_batch_particles = std::clamp(
        info.msf_transfer_max_batch_particles, 1u, 4096u);
    out.msf_transfer_velocity = info.msf_transfer_velocity;
    out.msf_melt_flow_enabled = info.msf_melt_flow_enabled;
    out.msf_melt_height_loss = std::clamp(info.msf_melt_height_loss, 0.0f, 0.92f);
    out.msf_melt_spread = std::clamp(info.msf_melt_spread, 0.0f, 2.5f);
    out.msf_melt_sdf_refresh = info.msf_melt_sdf_refresh;
    out.msf_melt_sdf_revision_interval = std::clamp<uint32_t>(
        info.msf_melt_sdf_revision_interval, 1u, 60u);
    out.msf_melt_sdf_change_threshold = std::clamp(
        info.msf_melt_sdf_change_threshold, 0.001f, 0.25f);
    return Result::success();
}

} // namespace

Result listSimulationFlowSources(std::vector<SimulationFlowSourceInfo>& out) {
    if (!g_ctx) return notBound();
    auto& runtime = scriptSimulationRuntime();
    out.clear();
    out.reserve(runtime.flowSources().size());
    for (const auto& source : runtime.flowSources())
        out.push_back(flowInfoFromDesc(source, runtime.gridDomains()));
    return Result::success();
}

Result getSimulationFlowSource(const std::string& name, SimulationFlowSourceInfo& out) {
    if (!g_ctx) return notBound();
    auto& runtime = scriptSimulationRuntime();
    const auto it = std::find_if(runtime.flowSources().begin(), runtime.flowSources().end(),
        [&name](const auto& source) { return source.name == name; });
    if (it == runtime.flowSources().end()) return Result::fail("flow source not found: " + name);
    out = flowInfoFromDesc(*it, runtime.gridDomains());
    return Result::success();
}

Result createSimulationFlowSource(const SimulationFlowSourceInfo& info,
                                  SimulationFlowSourceInfo& out) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    if (std::any_of(runtime.flowSources().begin(), runtime.flowSources().end(),
                    [&info](const auto& source) { return source.name == info.name; }))
        return Result::fail("flow source already exists: " + info.name);
    RayTrophiSim::SimulationFlowSourceDesc desc;
    Result converted = flowDescFromInfo(info, runtime, desc);
    if (!converted.ok) return converted;
    auto& created = runtime.addFlowSource(desc);
    out = flowInfoFromDesc(created, runtime.gridDomains());
    invalidateScriptSimulation();
    return Result::success();
}

Result updateSimulationFlowSource(const std::string& name,
                                  const SimulationFlowSourceInfo& info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    auto it = std::find_if(runtime.flowSources().begin(), runtime.flowSources().end(),
        [&name](const auto& source) { return source.name == name; });
    if (it == runtime.flowSources().end()) return Result::fail("flow source not found: " + name);
    RayTrophiSim::SimulationFlowSourceDesc desc;
    Result converted = flowDescFromInfo(info, runtime, desc);
    if (!converted.ok) return converted;
    desc.timeline_uid = it->timeline_uid;
    // Keys are animation, not configuration: a property update must not wipe
    // them. This whole-desc assignment silently discarded the keyframe map
    // before flow sources on Fluid domains were keyable and nobody noticed.
    desc.keyframes = it->keyframes;
    *it = std::move(desc);
    invalidateScriptSimulation();
    return Result::success();
}

Result keySimulationFlowSource(const std::string& name,
                               const SimulationFlowSourceKey& key) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    auto it = std::find_if(runtime.flowSources().begin(), runtime.flowSources().end(),
        [&name](const auto& source) { return source.name == name; });
    if (it == runtime.flowSources().end()) return Result::fail("flow source not found: " + name);

    // Merge into any key already at this frame so two calls can key different
    // channels on the same frame without erasing each other.
    auto& stored = it->keyframes[key.frame];
    if (key.has_enabled)  { stored.has_enabled = true;  stored.enabled = key.enabled; }
    if (key.has_position) { stored.has_position = true; stored.position = key.position; }
    if (key.has_velocity) { stored.has_velocity = true; stored.velocity = key.velocity; }
    if (key.has_radius)   { stored.has_radius = true;   stored.radius = std::max(0.001f, key.radius); }
    if (key.has_density)  { stored.has_density = true;  stored.density = std::max(0.0f, key.density); }
    if (key.has_temperature) { stored.has_temperature = true; stored.temperature = std::max(0.0f, key.temperature); }
    if (key.has_fuel)     { stored.has_fuel = true;     stored.fuel = std::max(0.0f, key.fuel); }
    if (key.has_falloff)  { stored.has_falloff = true;  stored.falloff = std::max(0.0f, key.falloff); }
    if (key.has_velocity_coupling) {
        stored.has_velocity_coupling = true;
        stored.velocity_coupling = std::max(0.0f, key.velocity_coupling);
    }
    if (key.has_flow_rate) { stored.has_flow_rate = true; stored.flow_rate = std::max(0.0f, key.flow_rate); }
    invalidateScriptSimulation();
    return Result::success();
}

Result clearSimulationFlowSourceKey(const std::string& name, int frame) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    auto it = std::find_if(runtime.flowSources().begin(), runtime.flowSources().end(),
        [&name](const auto& source) { return source.name == name; });
    if (it == runtime.flowSources().end()) return Result::fail("flow source not found: " + name);
    it->keyframes.erase(frame);
    invalidateScriptSimulation();
    return Result::success();
}

Result removeSimulationFlowSource(const std::string& name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    auto& sources = runtime.flowSources();
    const auto it = std::find_if(sources.begin(), sources.end(),
        [&name](const auto& source) { return source.name == name; });
    if (it == sources.end()) return Result::fail("flow source not found: " + name);
    sources.erase(it);
    invalidateScriptSimulation();
    return Result::success();
}

Result listSimulationColliders(std::vector<SimulationColliderInfo>& out) {
    if (!g_ctx) return notBound();
    const auto& colliders = scriptSimulationRuntime().colliders();
    out.clear();
    out.reserve(colliders.size());
    for (const auto& collider : colliders) out.push_back(colliderInfoFromDesc(collider));
    return Result::success();
}

Result getSimulationCollider(const std::string& name, SimulationColliderInfo& out) {
    if (!g_ctx) return notBound();
    const auto& colliders = scriptSimulationRuntime().colliders();
    const auto it = std::find_if(colliders.begin(), colliders.end(),
        [&name](const auto& collider) { return collider.name == name; });
    if (it == colliders.end()) return Result::fail("simulation collider not found: " + name);
    out = colliderInfoFromDesc(*it);
    return Result::success();
}

Result createSimulationCollider(const SimulationColliderInfo& info,
                                SimulationColliderInfo& out) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    if (std::any_of(runtime.colliders().begin(), runtime.colliders().end(),
                    [&info](const auto& collider) { return collider.name == info.name; }))
        return Result::fail("simulation collider already exists: " + info.name);
    RayTrophiSim::ParticleColliderDesc desc;
    Result converted = colliderDescFromInfo(info, desc);
    if (!converted.ok) return converted;
    auto& created = runtime.addCollider(desc);
    out = colliderInfoFromDesc(created);
    invalidateScriptSimulation();
    // UI creation starts the asynchronous SDF cook immediately. Scripted
    // creation must use the same path; invalidating the timeline alone never
    // populates sdf_grid_data, so mesh_sdf silently behaved like no collider.
    g_ctx->scene.rebuildSDFColliderAsync(created);
    return Result::success();
}

Result updateSimulationCollider(const std::string& name,
                                const SimulationColliderInfo& info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    auto& colliders = runtime.colliders();
    auto it = std::find_if(colliders.begin(), colliders.end(),
        [&name](const auto& collider) { return collider.name == name; });
    if (it == colliders.end()) return Result::fail("simulation collider not found: " + name);
    RayTrophiSim::ParticleColliderDesc desc;
    Result converted = colliderDescFromInfo(info, desc);
    if (!converted.ok) return converted;
    *it = std::move(desc);
    invalidateScriptSimulation();
    g_ctx->scene.rebuildSDFColliderAsync(*it);
    return Result::success();
}

Result rebuildSimulationColliderSDF(const std::string& name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& colliders = scriptSimulationRuntime().colliders();
    auto it = std::find_if(colliders.begin(), colliders.end(),
        [&name](const auto& collider) { return collider.name == name; });
    if (it == colliders.end()) return Result::fail("simulation collider not found: " + name);
    if (it->source_mode != RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF)
        return Result::fail("simulation collider is not mesh_sdf: " + name);
    it->sdf_grid_data.reset();
    it->sdf_nx = it->sdf_ny = it->sdf_nz = 0;
    g_ctx->scene.rebuildSDFColliderAsync(*it);
    invalidateScriptSimulation();
    return Result::success();
}

Result removeSimulationCollider(const std::string& name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    auto& colliders = runtime.colliders();
    const auto it = std::find_if(colliders.begin(), colliders.end(),
        [&name](const auto& collider) { return collider.name == name; });
    if (it == colliders.end()) return Result::fail("simulation collider not found: " + name);
    colliders.erase(it);
    invalidateScriptSimulation();
    return Result::success();
}

Result resetFluidSimulation() {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    // Reset both the legacy FluidObject bridge and the unified particle/grid
    // runtime used by scripted Gas + APIC domains, including source counters,
    // collider history and the timeline frame cache.
    g_ctx->scene.ensureFluidSimulationSystem();
    g_ctx->scene.resetSimulation();
    return Result::success();
}

Result stepFluidSimulation(float dt) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (dt <= 0.0f) dt = 0.0166667f;

    // Script-authored APIC/gas domains live on ParticleSimulationSystem, while
    // legacy FluidObject instances live on FluidSimulationSystem. Both are
    // registered with SimulationWorld. Calling only the legacy system here left
    // every scripted gas domain, MSF gather and coupling stage completely
    // unstepped even though rt.fluid.step() returned success. Use the same
    // one-shot scheduler path as the main timeline so system ordering, force
    // snapshots and all registered runtimes are exercised exactly once.
    g_ctx->scene.syncSimulationWorld();
    // Scripts panel runs while the timeline is commonly paused. stepOnce()
    // intentionally honours that global mode and would otherwise return before
    // executing any system, despite this API being an explicit manual-step
    // request. Temporarily unpause only for this call and restore the user's
    // timeline state immediately afterwards.
    const auto previousMode = g_ctx->scene.simulation_world.getMode();
    if (previousMode == RayTrophiSim::SimulationMode::Paused)
        g_ctx->scene.simulation_world.setMode(RayTrophiSim::SimulationMode::Realtime);
    g_ctx->scene.simulation_world.stepOnce(dt);
    if (previousMode == RayTrophiSim::SimulationMode::Paused)
        g_ctx->scene.simulation_world.setMode(previousMode);
    return Result::success();
}

} // namespace rtapi
