#pragma once

// A liquid/matter domain's material is one substance from the library
// (APICSolverParams::default_substance). This file is the only writer of the
// domain's PHYSICAL solver fields from it, and the only reader of the legacy
// FluidPreset / FluidChemistryPreset integers old projects carry.
// Design: docs/dev/MADDE_TIPLERI_TASARIMI.md §6a.

#include "Fluid/APICFluidSolver.h"
#include "json.hpp"

#include <string>

namespace RayTrophiSim {
struct SubstanceProfile;
struct MaterialTemperatureScale;
struct SimulationGridDomainDesc;
namespace Fluid {

// The domain's substance, or nullptr when the name is not in the library.
const SubstanceProfile* domainSubstance(const APICSolverParams& params);

// Resolve the continuum's constitutive law from its canonical particle substance.
// The domain default remains the fallback for untagged particles and liquids.
void resolveParticleGranularPhysics(APICSolverParams& params, const FluidParticles& particles);

// Writes the substance's physics into `params`: kinematic viscosity (0 when it
// is below what the voxel can resolve and the thermal chain is off), the
// granular skeleton and granular_enabled, the freeze point (melt_kelvin; never
// for a substance that cannot melt), the near-freeze viscosity curve, parcel
// conduction and the fuel profile. Deterministic for a given substance and
// voxel: these values are hashed into the bake signature.
// Fails (and leaves params untouched) when the substance is unknown.
class FluidParticles;
// Liquid physics follow the liquid that is actually there. A domain whose
// default substance is a liquid but holds none of its parcels (a Honey seed
// layer in a Water-default domain) ran Water's physics on the honey: nu 0,
// free-slip walls -- "honey flows like water" (2026-11-09). Then the dominant
// liquid parcel substance's physics are resolved instead; default_substance
// itself is untouched. Returns true when it adopted.
bool adoptParcelLiquidSubstance(APICSolverParams& params, const FluidParticles& particles,
                                float voxel_size, const MaterialTemperatureScale& scale);

struct SubstanceViscosityEntry;
// Every liquid substance among the parcels that no entry describes yet gets
// its own catalogue viscosity (resolveSubstanceViscosity), so a honey + water
// mixture diffuses each at its own nu even without a material binding. Before,
// only bindings fed the per-cell field: parcels of an unbound substance took the
// domain value. Returns the new entry count.
std::size_t appendParcelSubstanceViscosities(const FluidParticles& particles,
    const APICSolverParams& params, float voxel_size, SubstanceViscosityEntry* entries,
    std::size_t count, std::size_t capacity);

bool resolveDomainSubstancePhysics(APICSolverParams& params, float voxel_size,
                                   const MaterialTemperatureScale& scale,
                                   std::string& error);

// Keep the combustion descriptor mirrors consistent with the same material
// resolution. Solver hints remain authored numerical settings.
bool resolveGridDomainSubstancePhysics(SimulationGridDomainDesc& domain, float voxel_size,
                                      const MaterialTemperatureScale& scale,
                                      std::string& error);

// Writes the substance's solver hints (numerical tuning) into `params`. Called
// once when a domain chooses the substance; the domain may then edit them.
void applySubstanceSolverHints(APICSolverParams& params, const SubstanceProfile& substance);

// ν·dt/h² below this, at the reference frame step, is numerical-diffusion
// territory: the viscous solve is skipped (ν treated as 0).
constexpr float kUnresolvedViscosityDiffusion = 1.0e-3f;
constexpr float kViscosityReferenceDt = 1.0f / 24.0f;

// Old projects: chooses the default substance from the stored FluidPreset and
// FluidChemistryPreset integers and from the physical fields the domain stored
// itself. Creates project substances when needed ("Lava (legacy)" and friends
// for presets that are not built-ins; "<domain> material" when the stored
// physics differ from the chosen base, so a hand-tuned domain keeps its
// values). Returns false with `error` only when a substance cannot be created.
bool migrateLegacyDomainMaterial(const nlohmann::json& old_fluid_params,
                                 const std::string& domain_name, float voxel_size,
                                 std::string& out_default_substance, std::string& error);

} // namespace Fluid
} // namespace RayTrophiSim
