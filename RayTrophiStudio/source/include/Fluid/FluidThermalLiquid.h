#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// THERMAL LIQUID — cooling, temperature-dependent viscosity, freezing.
// ═══════════════════════════════════════════════════════════════════════════
// The candle-wax chain for the APIC liquid. Parameters live on
// APICSolverParams (thermal_*); the caller runs these once per frame, before
// the solid-phase overlay is built, in this order:
//
//   coolThermalLiquid      surface/contact parcels lose heat (Newton)
//   updateThermalFreeze    fusion: parcels below their melt point solidify
//                          (where nucleated), solid parcels above it melt,
//                          and either way the latent heat is paid
//   buildThermalViscosityField   ν(T) per cell, handed to the viscous solve
//
// Conduction between parcels is NOT here: diffuseParticleTemperature already
// runs for every fluid domain and carries heat from a fresh pour into the wax
// it lands on (which is what re-melts a frozen layer under a hot one).
//
// ★★★ WHY FREEZING NEEDS A SUPPORT. A frozen parcel is pinned — velocity held
// at zero, stamped into grid.solid. Wax does set in the air, but a parcel that
// froze in free fall under this rule would hang there, and the result would be
// a stalactite growing out of nothing. Real wax that sets mid-air is still
// attached to the stream above it and falls with it; the solver has no bond
// to express that. So a parcel only freezes where it touches something that
// can hold it: a collider, a closed domain wall, or a parcel that is already
// frozen. The freeze front therefore grows out of the contact surface, one
// cell per frame at most, which is also how a real pour sets — from the cold
// surface outward. A stream that cools in flight still thickens (ν(T)), it just
// does not turn to stone until it lands.

#include "APICFluidSolver.h"
#include "FluidParticles.h"
#include "../FluidGrid.h"
#include "../MaterialStateField.h"
#include <cstddef>
#include <vector>

namespace RayTrophiSim {
namespace Fluid {

// Last frame's thermal readout for one domain. `measured` false means the
// chain did not run (disabled, granular, or no particles) — the numbers are
// then zero because nothing was measured, not because nothing is hot.
struct ThermalLiquidStats {
    bool        measured = false;
    std::size_t frozen_particles = 0;      // flagged frozen after this frame
    std::size_t froze_this_frame = 0;
    std::size_t melted_this_frame = 0;
    // Parcels below the freeze point that did NOT freeze for lack of a
    // support. Non-zero with frozen_particles == 0 is the reading that says
    // "the wax is cold enough but touches nothing": a missing collider, or a
    // collider the fluid does not collide with.
    std::size_t cold_unsupported = 0;
    // Parcels part solid, part liquid (0 < fusion_solid_fraction < 1): held
    // at their melt point while the latent heat crosses.
    std::size_t mushy_particles = 0;
    double      fusion_released_j = 0.0;   // latent heat freezing gave back this step
    double      fusion_absorbed_j = 0.0;   // latent heat melting consumed this step
    std::size_t air_cooled_particles = 0;      // in a surface cell this frame
    std::size_t contact_cooled_particles = 0;  // touching a collider/closed wall
    std::size_t field_heated_particles = 0;    // inside a Thermal force field
    std::size_t convecting_particles = 0;      // got a Boussinesq kick this step
    std::size_t conducting_cells = 0;          // liquid cells in the conduction pass
    std::size_t gas_exchange_particles = 0;    // surface parcels trading heat with gas
    double      heat_to_gas_j = 0.0;           // net heat liquid -> gas this step
    double      evaporated_kg = 0.0;           // below-boiling evaporation this step
    double      condensed_kg = 0.0;            // vapour condensed onto surface parcels this step
    std::size_t condensation_splits = 0;       // parcels grown past 2x canonical mass and split
    std::size_t solid_contact_particles = 0;   // parcels trading heat with a surface (MSF)
    double      heat_to_solid_j = 0.0;         // net heat liquid -> surfaces this step
    // Heat the walls and the outside air (infinite reservoirs at ambient)
    // took this step through their films; negative = they warmed the liquid.
    double      heat_to_ambient_j = 0.0;
    bool        grain_air_humid = false;  // grain-air film read a vapour inventory (P9)
    // What the touching parcels SAW, contact-weighted (K; 0 = no contact).
    // Tells "the shell looks cold where the water is" apart from "heat does
    // not flow" when heat_to_solid_j is small (Ek 11, 2026-10-18).
    double      solid_contact_shell_kelvin = 0.0;
    double      solid_contact_parcel_kelvin = 0.0;
    std::size_t solid_contact_shell_cells = 0;  // distinct shell cells touched
    // Grain carriers (exchangeMatterGrainHeat, 2026-10-19). grain_measured
    // false: no grains, or the pass did not run - the rows are then unmeasured.
    bool        grain_measured = false;
    std::size_t grain_count = 0;
    std::size_t grain_wetted = 0;              // grains sharing a cell with liquid
    float       grain_min_kelvin = 0.0f;
    float       grain_mean_kelvin = 0.0f;
    float       grain_max_kelvin = 0.0f;
    double      grain_heat_to_liquid_j = 0.0;  // net grains -> liquid this step
    double      grain_heat_to_gas_j = 0.0;     // net grains -> the domain's air
    double      grain_heat_to_walls_j = 0.0;   // grains -> closed walls / bare colliders
    // sum of C dT over grains, liquid and air for this pass: zero to rounding.
    // Anything else means one side was written without the other.
    double      grain_energy_residual_j = 0.0;
    float       grain_thermal_ms = 0.0f;
    float       min_kelvin = 0.0f;
    float       mean_kelvin = 0.0f;
    float       max_kelvin = 0.0f;
    // Viscosity actually handed to the solver (cells with particle support).
    bool        viscosity_field_built = false;
    float       min_viscosity = 0.0f;
    float       max_viscosity = 0.0f;
};

// Walls and outside air as FILMS against each parcel's heat capacity
// (2026-10-19; the old thermal_air/contact_cooling_rate were capacity-free and
// are retired). A face on a closed domain wall or on a collider without an MSF
// shell trades through 500 W/(m^2 K) with a container at `ambient_kelvin`; a
// face onto empty air through 20 W/(m^2 K) plus the parcel's linearized
// radiation - only where the domain has no gas grid (air_through_gas). Each
// parcel takes its share of the cell face (h^2 / parcels in the cell). Grain
// carriers are skipped (exchangeMatterGrainHeat owns them).
// `heat_sources` (Thermal force fields, the same list and thermalSourceWeight()
// MSF uses) are authored heat SOURCES: every parcel inside one relaxes toward
// ambient + the stacked deltas at the fields' own rate_per_s, surface or not,
// with the thermal-liquid chain on or off.
// Reads grid.solid, so call it AFTER the collider voxelize and BEFORE the
// solid-phase overlay is stamped (the overlay is not a cold surface).
void coolThermalLiquid(FluidParticles& particles,
                       const FluidSim::FluidGrid& grid,
                       const APICSolverParams& params,
                       float ambient_kelvin,
                       float dt,
                       ThermalLiquidStats& stats,
                       const std::vector<MaterialStateFieldSystem::ThermalSource>*
                           heat_sources = nullptr,
                       bool air_through_gas = false,
                       const std::vector<float>* solid_capacity = nullptr);

// Liquid <-> collider surface (MSF shell) contact, per substance, energy
// conserved: a parcel whose cell or face neighbour holds shell capacity
// (solid_capacity, J/K, from MaterialStateFieldSystem::gatherThermalContact)
// relaxes with that shell toward the shared equilibrium through a contact film
// of 500 W/(m^2 K) (liquid on a solid surface, natural convection) over its
// share of one cell face. solid_kelvin is updated in place as parcels draw on
// it; the heat each shell cell gained goes to solid_heat_j for
// depositThermalContactHeat. Such contacts are NOT also relaxed toward the
// ambient in coolThermalLiquid (pass the same solid_capacity there).
void exchangeLiquidSolidContact(FluidParticles& particles,
                                const FluidSim::FluidGrid& grid,
                                const APICSolverParams& params,
                                float dt,
                                const std::vector<float>& solid_capacity,
                                std::vector<float>& solid_kelvin,
                                std::vector<float>& solid_heat_j,
                                ThermalLiquidStats& stats);

// Heat capacity of one parcel, J/K: its current mass times its OWN substance's
// specific heat. Mass is rest_mass_kg * mass_fraction when known, else the
// substance's liquid density times the parcel's share of a cell.
double parcelHeatCapacity(const FluidParticles& particles, std::size_t p,
                          const SubstanceProfile& profile, float voxel_size,
                          int particles_per_cell);

// Fourier conduction between neighbouring liquid cells. Each cell gathers its
// parcels' heat capacity (m*c, per substance) and energy; the flux across a
// face is k_face * A * (T_a - T_b) / h with k_face the harmonic mean of the two
// cells' capacity-weighted conductivities, integrated implicitly per face pair
// so it never overshoots. Parcels receive their cell's temperature CHANGE
// (FLIP-style), keeping their own deviation. Energy is conserved exactly.
// Water at 0.1 m voxels conducts slowly - that is the physics; convection does
// the mixing.
void conductThermalLiquid(FluidParticles& particles,
                          const FluidSim::FluidGrid& grid,
                          const APICSolverParams& params,
                          float dt,
                          ThermalLiquidStats& stats);

// Natural convection (Boussinesq): every fluid-model, unfrozen parcel whose
// substance has thermal_expansion_per_k > 0 gets dv = -g * beta * (T - Tmean) * dt,
// Tmean being the mass-free mean temperature of those parcels. Relative to the
// mean, so the body as a whole is not lifted; warm parcels rise, cold sink.
// Runs whenever temperatures differ (thermal chain or a heat field), on the
// HOST velocities: the caller re-uploads them like the frozen-pin path.
// Returns the number of parcels kicked.
std::size_t applyThermalConvection(FluidParticles& particles,
                                   const APICSolverParams& params,
                                   float dt,
                                   ThermalLiquidStats& stats);

// Fusion (melting/freezing) with its latent heat, per parcel and per
// substance (melt_kelvin, latent_heat_fusion, specific_heat).
// fusion_solid_fraction F is the parcel's solid share. Above the melt point
// the superheat melts dF = min(F, c (T - Tm) / L) and the parcel holds at Tm;
// below it the undercooling freezes the same way. So 1 kg of ice at 0 C takes
// 334 kJ to become water, and a pond does not freeze the moment it touches
// 0 C. L = 0 (no data) keeps the old free switch at Tm.
// State follows the majority phase (F >= kFusionSolidThreshold):
//  * a meltable liquid (Water, Wax) is FROZEN - pinned, stamped into the
//    solid mask. Freezing needs a nucleus (support below, or crystals already
//    in the parcel); without one it supercools (cold_unsupported).
//  * a skeleton carrier (Snow, Ice) changes constitutive model at the next
//    refreshMatterConstitutiveModels; it never uses the frozen flag.
// With the chain off and no heat field, liquid parcels are unfrozen and
// fused back to liquid, so the switch never leaves pinned parcels behind.
// The half-parcel of latent heat between the two state switches is what keeps
// a parcel at the threshold from flickering in grid.solid.
void updateThermalFreeze(FluidParticles& particles,
                         const FluidSim::FluidGrid& grid,
                         const APICSolverParams& params,
                         ThermalLiquidStats& stats,
                         bool heat_fields_present = false,
                         std::vector<float>* solid_change = nullptr);

// Cell-centred ν(T): evaluate each parcel's substance curve using its own
// melt point, hot/cold viscosity and temperature, then gather with trilinear
// weights. Empty cells retain the base field (or domain hot viscosity).
// Returns false when disabled, granular, or without valid particle storage.
// `base` must not alias `viscosity_out`.
bool buildThermalViscosityField(const FluidParticles& particles,
                                const FluidSim::FluidGrid& grid,
                                const APICSolverParams& params,
                                const std::vector<float>* base,
                                std::vector<float>& viscosity_out,
                                ThermalLiquidStats* stats = nullptr);

inline bool isFrozenParticle(const FluidParticles& particles, std::size_t p) {
    return p < particles.flags.size() &&
           (particles.flags[p] & kParticleFlagFrozen) != 0u;
}

} // namespace Fluid
} // namespace RayTrophiSim
