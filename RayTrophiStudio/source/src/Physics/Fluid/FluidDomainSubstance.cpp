// Domain <- substance. See FluidDomainSubstance.h.
#include "Fluid/FluidDomainSubstance.h"

#include "MaterialStateField.h"
#include "SubstanceLibrary.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace RayTrophiSim {
namespace Fluid {

const SubstanceProfile* domainSubstance(const APICSolverParams& params) {
    return tryFindSubstance(params.default_substance);
}

bool resolveDomainSubstancePhysics(APICSolverParams& params, float voxel_size,
                                   const MaterialTemperatureScale& scale,
                                   std::string& error) {
    const SubstanceProfile* s = domainSubstance(params);
    if (!s) {
        error = "domain default substance '" + params.default_substance +
            "' is not in the substance library";
        return false;
    }
    params.granular_enabled =
        s->default_constitutive_model == MatterConstitutiveModel::Granular;

    // Viscosity. A granular skeleton has none (its friction is a yield
    // criterion, not a velocity Laplacian). A liquid whose diffusion number at
    // this voxel is negligible skips the solve: water's 1e-6 is orders below
    // the numerical diffusion of any renderable grid, and paying for sweeps buys
    // motion nobody can see. Not with the thermal chain on: its viscosity curve
    // ramps from this value in log space.
    float nu = std::max(s->liquid_kinematic_viscosity, 0.0f);
    const float h = voxel_size > 0.0f ? voxel_size : 0.05f;
    if (params.granular_enabled) {
        nu = 0.0f;
    } else if (!params.thermal_liquid_enabled &&
               nu * kViscosityReferenceDt / (h * h) < kUnresolvedViscosityDiffusion) {
        nu = 0.0f;
    }
    params.kinematic_viscosity = nu;

    params.granular_friction_angle_degrees = s->granular_friction_degrees;
    params.granular_cohesion = s->granular_cohesion;
    params.granular_dilatancy_degrees = s->granular_dilatancy_degrees;
    params.granular_young_modulus = s->granular_young_modulus;
    params.granular_poisson_ratio = s->granular_poisson_ratio;
    params.granular_tensile_cutoff = s->granular_tensile_cutoff;
    params.granular_hardening = s->granular_hardening;
    params.granular_compaction_hardening = s->granular_compaction_hardening;
    params.granular_compaction_limit = s->granular_compaction_limit;
    params.granular_fracture_strain = s->granular_fracture_strain;
    params.granular_damage_rate = s->granular_damage_rate;
    params.granular_healing_rate = s->granular_healing_rate;
    params.granular_rebonding = s->granular_rebonding;
    params.granular_softening_temperature = s->granular_softening_kelvin;
    params.granular_softening_range = s->granular_softening_range;
    params.granular_residual_strength = s->granular_residual_strength;
    params.granular_tack_peak = s->granular_tack_peak;
    params.granular_thermal_conductivity = s->parcel_conduction;

    // Freezing is the substance's own melt point. A substance that cannot melt
    // never freezes (its melt_kelvin field keeps an inert default).
    params.thermal_freeze_kelvin = s->meltable ? s->melt_kelvin : 1.0f;
    params.thermal_viscosity_range = s->liquid_freeze_viscosity_range;
    params.thermal_cold_viscosity = s->liquid_cold_viscosity;

    params.applySubstanceProfile(*s, scale);
    params.sanitizeGranularMaterial();
    params.sanitizeThermalLiquid();
    if (s->meltable) {
        // sanitizeThermalLiquid floors the freeze point at 1 K; the inert value
        // above relies on that floor, a real melt point must pass through.
        params.thermal_freeze_kelvin = std::clamp(s->melt_kelvin, 1.0f, 5000.0f);
    }
    error.clear();
    return true;
}

bool resolveGridDomainSubstancePhysics(SimulationGridDomainDesc& domain, float voxel_size,
                                      const MaterialTemperatureScale& scale,
                                      std::string& error) {
    if (!resolveDomainSubstancePhysics(domain.fluid_params, voxel_size, scale, error)) {
        return false;
    }
    const auto& chemistry = domain.fluid_params.fuel_profile;
    domain.fluid_flammable = chemistry.flammable;
    domain.fluid_extinguishing = chemistry.extinguishing;
    domain.fluid_ignition_temperature = chemistry.flash_temperature;
    domain.fluid_evaporation_rate = chemistry.vaporization_rate;
    domain.fluid_cooling_power = chemistry.cooling_power;
    domain.fluid_oxygen_dilution = chemistry.oxygen_dilution;
    return true;
}

void applySubstanceSolverHints(APICSolverParams& params, const SubstanceProfile& s) {
    params.flip_blend = s.solver_flip_blend;
    params.apic_blend = s.solver_apic_blend;
    params.velocity_damping = s.solver_velocity_damping;
    params.density_correction = s.solver_density_correction;
    params.air_drag = s.solver_air_drag;
    params.wall_damping = s.solver_wall_damping;
    params.affine_damping = s.solver_affine_damping;
    params.max_velocity = s.solver_max_velocity;
    params.viscosity_sweeps = std::max(1, static_cast<int>(std::lround(s.solver_viscosity_sweeps)));
    params.viscosity_wall_slip = s.solver_viscosity_wall_slip;
    params.internal_friction = s.solver_internal_friction;
    params.granular_max_solver_substeps =
        std::clamp(static_cast<int>(std::lround(s.solver_granular_max_substeps)), 1, 64);
    params.thermal_liquid_enabled = s.solver_thermal_chain;
    params.thermal_air_cooling_rate = s.solver_air_cooling_rate;
    params.thermal_contact_cooling_rate = s.solver_contact_cooling_rate;
    params.sanitizeThermalLiquid();
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy projects
// ─────────────────────────────────────────────────────────────────────────────
namespace {

using nlohmann::json;

// The FluidPreset integers as SceneSerializer/ProjectManager stored them
// (append-only enum, removed 2026-10-07).
enum LegacyPreset : int {
    kCustom = 0, kWater, kOil, kMud, kHoney, kLava, kSand, kChocolate,
    kWetSand, kGravel, kCohesiveSoil, kMoltenPlastic, kWax
};

struct LegacySubstance {
    const char* name;
    const char* based_on;
    json overrides;
};

// Presets that were never built-in substances. Values are the removed
// applyPreset table, verbatim.
bool legacyPresetSubstance(int preset, LegacySubstance& out) {
    switch (preset) {
        case kLava:
            out = {"Lava (legacy)", "Stone", {
                {"liquid_kinematic_viscosity", 0.5}, {"solver_viscosity_sweeps", 24.0},
                {"solver_viscosity_wall_slip", 0.0}, {"solver_internal_friction", 0.2},
                {"solver_flip_blend", 0.85}, {"solver_apic_blend", 0.85},
                {"solver_air_drag", 1.0}, {"solver_wall_damping", 0.5},
                {"solver_affine_damping", 0.92}}};
            return true;
        case kWetSand:
            out = {"Wet Sand (legacy)", "Sand", {
                {"granular_friction_degrees", 37.0}, {"granular_cohesion", 1500.0},
                {"granular_dilatancy_degrees", 6.0}, {"granular_young_modulus", 2.5e5},
                {"granular_tensile_cutoff", 400.0}, {"granular_fracture_strain", 0.010},
                {"granular_damage_rate", 14.0}, {"granular_healing_rate", 0.5},
                {"granular_rebonding", true}, {"solver_air_drag", 0.12},
                {"solver_wall_damping", 0.45}}};
            return true;
        case kCohesiveSoil:
            out = {"Cohesive Soil (legacy)", "Soil", {
                {"granular_friction_degrees", 20.0}, {"granular_cohesion", 12000.0},
                {"granular_dilatancy_degrees", 0.0}, {"granular_young_modulus", 1.2e5},
                {"granular_poisson_ratio", 0.35}, {"granular_tensile_cutoff", 3000.0},
                {"granular_hardening", 0.5}, {"granular_fracture_strain", 0.006},
                {"granular_damage_rate", 20.0}, {"granular_healing_rate", 0.0},
                {"solver_air_drag", 0.10}, {"solver_wall_damping", 0.60}}};
            return true;
        case kMoltenPlastic:
            out = {"Molten Plastic (legacy)", "Plastic (PE)", {
                {"default_constitutive_model", "granular"}, {"category", "granular"},
                {"granular_friction_degrees", 30.0}, {"granular_cohesion", 6000.0},
                {"granular_dilatancy_degrees", 2.0}, {"granular_young_modulus", 2.0e5},
                {"granular_poisson_ratio", 0.40}, {"granular_tensile_cutoff", 1500.0},
                {"granular_hardening", 0.0}, {"granular_fracture_strain", 0.02},
                {"granular_damage_rate", 6.0}, {"granular_healing_rate", 2.0},
                {"granular_rebonding", true}, {"granular_softening_kelvin", 420.0},
                {"granular_softening_range", 90.0}, {"granular_residual_strength", 0.0},
                {"granular_tack_peak", 3.5}, {"parcel_conduction", 2.5},
                {"solver_flip_blend", 0.0}, {"solver_viscosity_wall_slip", 0.0},
                {"solver_air_drag", 0.05}, {"solver_wall_damping", 0.70}}};
            return true;
        default:
            return false;
    }
}

const char* legacyPresetBuiltin(int preset) {
    switch (preset) {
        case kWater: return "Water";
        case kOil: return "Oil";
        case kMud: return "Mud";
        case kHoney: return "Honey";
        case kSand: return "Sand";
        case kChocolate: return "Chocolate";
        case kGravel: return "Gravel";
        case kWax: return "Wax";
        default: return nullptr;
    }
}

// FluidChemistryPreset integers: Inert, Water, Gasoline, Alcohol, Oil, Custom,
// Plastic, Wax. Inert/Custom carried no substance.
const char* legacyChemistrySubstance(int chemistry) {
    switch (chemistry) {
        case 1: return "Water";
        case 2: return "Gasoline";
        case 3: return "Alcohol";
        case 4: return "Oil";
        case 6: return "Plastic (PE)";
        case 7: return "Wax";
        default: return nullptr;
    }
}

bool ensureProjectSubstance(const std::string& name, const std::string& based_on,
                            const json& overrides, std::string& error) {
    if (!tryFindSubstance(name)) {
        if (!deriveSubstance(name, based_on, error)) return false;
    }
    if (!overrides.empty() && !patchSubstance(name, overrides, error)) return false;
    return true;
}

bool nearlyEqual(double a, double b) {
    // Viscosity is often below 1e-5. A unit-sized absolute tolerance would
    // silently erase custom water/oil values during migration.
    return std::fabs(a - b) <= 1.0e-5 * std::max({1.0e-12, std::fabs(a), std::fabs(b)});
}

} // namespace

bool migrateLegacyDomainMaterial(const json& f, const std::string& domain_name,
                                 float voxel_size, std::string& out_default_substance,
                                 std::string& error) {
    error.clear();
    const int preset = f.value("current_preset", static_cast<int>(kWater));
    const int chemistry = f.value("chemistry_preset", 0);

    std::string base;
    LegacySubstance legacy;
    if (const char* builtin = legacyPresetBuiltin(preset)) {
        base = builtin;
    } else if (legacyPresetSubstance(preset, legacy)) {
        if (!ensureProjectSubstance(legacy.name, legacy.based_on, legacy.overrides, error))
            return false;
        base = legacy.name;
    } else {
        // Custom: the chemistry said what the liquid was, if anything.
        const char* chem = legacyChemistrySubstance(chemistry);
        base = chem ? chem : "Water";
    }

    // The base's physics as this domain would now resolve them.
    APICSolverParams resolved;
    resolved.default_substance = base;
    resolved.thermal_liquid_enabled = f.value("thermal_liquid_enabled", false);
    if (!resolveDomainSubstancePhysics(resolved, voxel_size, MaterialTemperatureScale{}, error))
        return false;

    // Stored physics that differ from the base become overrides of a domain
    // material. Each group is compared only where it acted: every old file
    // stored thermal_freeze_kelvin = 330 whether the chain was on or not, and
    // granular fields are inert in a liquid domain.
    json overrides = json::object();
    const bool stored_granular = f.value("granular_enabled", resolved.granular_enabled);
    if (stored_granular != resolved.granular_enabled) {
        overrides["default_constitutive_model"] = stored_granular ? "granular" : "fluid";
    }
    const auto compare = [&](const char* stored_key, float resolved_value,
                             const char* substance_key) {
        if (!f.contains(stored_key) || !f[stored_key].is_number()) return;
        const double stored = f[stored_key].get<double>();
        if (!nearlyEqual(stored, resolved_value)) overrides[substance_key] = stored;
    };
    if (!stored_granular) {
        compare("kinematic_viscosity", resolved.kinematic_viscosity, "liquid_kinematic_viscosity");
    } else {
        compare("granular_friction_angle_degrees", resolved.granular_friction_angle_degrees,
                "granular_friction_degrees");
        compare("granular_cohesion", resolved.granular_cohesion, "granular_cohesion");
        compare("granular_dilatancy_degrees", resolved.granular_dilatancy_degrees,
                "granular_dilatancy_degrees");
        compare("granular_young_modulus", resolved.granular_young_modulus, "granular_young_modulus");
        compare("granular_poisson_ratio", resolved.granular_poisson_ratio, "granular_poisson_ratio");
        compare("granular_tensile_cutoff", resolved.granular_tensile_cutoff, "granular_tensile_cutoff");
        compare("granular_hardening", resolved.granular_hardening, "granular_hardening");
        compare("granular_fracture_strain", resolved.granular_fracture_strain, "granular_fracture_strain");
        compare("granular_damage_rate", resolved.granular_damage_rate, "granular_damage_rate");
        compare("granular_healing_rate", resolved.granular_healing_rate, "granular_healing_rate");
        compare("granular_softening_temperature", resolved.granular_softening_temperature,
                "granular_softening_kelvin");
        compare("granular_softening_range", resolved.granular_softening_range, "granular_softening_range");
        compare("granular_residual_strength", resolved.granular_residual_strength,
                "granular_residual_strength");
        compare("granular_tack_peak", resolved.granular_tack_peak, "granular_tack_peak");
        if (f.contains("granular_rebonding") && f["granular_rebonding"].is_boolean() &&
            f["granular_rebonding"].get<bool>() != resolved.granular_rebonding) {
            overrides["granular_rebonding"] = f["granular_rebonding"].get<bool>();
        }
    }
    if (stored_granular || resolved.thermal_liquid_enabled) {
        compare("granular_thermal_conductivity", resolved.granular_thermal_conductivity,
                "parcel_conduction");
    }
    if (resolved.thermal_liquid_enabled) {
        if (f.contains("thermal_freeze_kelvin") && f["thermal_freeze_kelvin"].is_number()) {
            const double stored = f["thermal_freeze_kelvin"].get<double>();
            if (!nearlyEqual(stored, resolved.thermal_freeze_kelvin)) {
                overrides["meltable"] = true;
                overrides["melt_kelvin"] = stored;
            }
        }
        compare("thermal_viscosity_range", resolved.thermal_viscosity_range,
                "liquid_freeze_viscosity_range");
        compare("thermal_cold_viscosity", resolved.thermal_cold_viscosity, "liquid_cold_viscosity");
    }
    // "Oil physics + Gasoline chemistry": the fuel half of a different
    // substance. Inert/Custom chemistry carried nothing to keep.
    if (const char* chem_name = legacyChemistrySubstance(chemistry)) {
        const SubstanceProfile* chem = tryFindSubstance(chem_name);
        const SubstanceProfile* base_profile = tryFindSubstance(base);
        if (chem && base_profile && chem->name != base_profile->name &&
            chem->name != base_profile->based_on) {
            overrides["fluid_flammable"] = chem->fluid_flammable;
            overrides["fluid_extinguishing"] = chem->fluid_extinguishing;
            overrides["flash_kelvin"] = chem->flash_kelvin;
            overrides["autoignition_kelvin"] = chem->autoignition_kelvin;
            overrides["vaporization_rate"] = chem->vaporization_rate;
            overrides["latent_heat_vaporization"] = chem->latent_heat_vaporization;
            overrides["cooling_power"] = chem->cooling_power;
            overrides["oxygen_dilution"] = chem->oxygen_dilution;
            overrides["flame_persistence"] = chem->flame_persistence;
        }
    }

    if (overrides.empty()) {
        out_default_substance = base;
        return true;
    }
    const std::string material = domain_name + " material";
    // A re-load of the same old file must not stack a second copy.
    if (const SubstanceProfile* existing = tryFindSubstance(material)) {
        if (existing->based_on != base) {
            error = "cannot migrate domain '" + domain_name + "': a substance named '" +
                material + "' already exists with a different base";
            return false;
        }
    }
    if (!ensureProjectSubstance(material, base, overrides, error)) return false;
    out_default_substance = material;
    return true;
}

} // namespace Fluid
} // namespace RayTrophiSim
