#include "Fluid/MudMaterial.h"
#include "MaterialStateField.h"

namespace RayTrophiSim::Fluid {
namespace {

SubstanceProfile mudBase() {
    SubstanceProfile p;
    p.name = "Mud";
    p.density = 1700.0f;
    p.liquid_density = 1700.0f;
    p.specific_heat = 1500.0f;
    p.conductivity = 1.0f;
    p.vapor_molar_mass_kg_mol = 0.018015f;
    p.thermal_expansion_per_k = 0.0002f;
    p.boiling_kelvin = 373.15f;
    p.latent_heat_vaporization = 2.26e6f;
    p.solver_apic_blend = 0.90f;
    p.solver_air_drag = 0.8f;
    p.solver_affine_damping = 0.95f;
    p.solver_internal_friction = 0.0f;
    return p;
}

} // namespace

SubstanceProfile makeDenseMud() {
    auto p = mudBase();
    // Existing EVP law: a finite yield strain keeps a supported shape; yielded
    // deformation persists. Kelvin-Voigt stress supplies rate resistance.
    // Do not enable Maxwell creep: it would relax sub-yield support at rest.
    p.category = SubstanceCategory::Solid;
    p.default_constitutive_model = MatterConstitutiveModel::Elastic;
    p.elastic_young_modulus = 8.0e4f;
    p.elastic_poisson_ratio = 0.40f;
    p.elastic_yield_strain = 0.025f;
    p.elastic_relaxation_s = 0.0f;
    p.elastic_viscosity_pa_s = 150.0f;
    p.elastic_fracture_strain = 0.35f;
    p.elastic_damage_rate = 20.0f;
    p.solver_flip_blend = 0.0f;
    p.liquid_kinematic_viscosity = 0.0f;
    return p;
}

SubstanceProfile makeMudSlurry() {
    auto p = mudBase();
    p.name = "Mud Slurry";
    p.category = SubstanceCategory::Liquid;
    p.default_constitutive_model = MatterConstitutiveModel::Fluid;
    p.liquid_kinematic_viscosity = 2.0e-3f;
    p.solver_viscosity_sweeps = 12.0f;
    p.solver_viscosity_wall_slip = 0.2f;
    p.solver_flip_blend = 0.90f;
    return p;
}

} // namespace RayTrophiSim::Fluid
