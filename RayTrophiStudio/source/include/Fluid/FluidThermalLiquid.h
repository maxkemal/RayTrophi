#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// THERMAL LIQUID — cooling, temperature-dependent viscosity, freezing.
// ═══════════════════════════════════════════════════════════════════════════
// The candle-wax chain for the APIC liquid. Parameters live on
// APICSolverParams (thermal_*); the caller runs these once per frame, before
// the solid-phase overlay is built, in this order:
//
//   coolThermalLiquid      surface/contact parcels lose heat (Newton)
//   updateThermalFreeze    parcels below the freeze point solidify where
//                          supported, frozen parcels above it melt
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
    std::size_t air_cooled_particles = 0;      // in a surface cell this frame
    std::size_t contact_cooled_particles = 0;  // touching a collider/closed wall
    float       min_kelvin = 0.0f;
    float       mean_kelvin = 0.0f;
    float       max_kelvin = 0.0f;
    // Viscosity actually handed to the solver (cells with particle support).
    bool        viscosity_field_built = false;
    float       min_viscosity = 0.0f;
    float       max_viscosity = 0.0f;
};

// Newton cooling toward `ambient_kelvin`: parcels in a cell with an empty
// (air) face neighbour at thermal_air_cooling_rate, parcels whose cell touches
// a collider or a CLOSED domain wall at thermal_contact_cooling_rate. Both
// apply when both hold. Exponential form: stable at any dt, never overshoots.
// Runs for granular domains too (a molten plastic re-stiffens as it cools).
// Reads grid.solid, so call it AFTER the collider voxelize and BEFORE the
// solid-phase overlay is stamped (the overlay is not a cold surface).
void coolThermalLiquid(FluidParticles& particles,
                       const FluidSim::FluidGrid& grid,
                       const APICSolverParams& params,
                       float ambient_kelvin,
                       float dt,
                       ThermalLiquidStats& stats);

// Freeze / melt transitions and pinning. Liquid domains only: a granular
// domain (or a disabled chain) has every frozen flag cleared instead, so
// switching the feature off can never leave invisible pinned parcels behind.
// Melting uses a small hysteresis above the freeze point so a parcel sitting
// on the threshold does not flicker between solid and liquid every frame —
// each flip would be a step change in grid.solid and a kick to the liquid.
void updateThermalFreeze(FluidParticles& particles,
                         const FluidSim::FluidGrid& grid,
                         const APICSolverParams& params,
                         ThermalLiquidStats& stats);

// Cell-centred ν(T): the parcels' temperature gathered with the transfer's
// trilinear weights (the same support as buildSubstanceViscosityField), then
// log-interpolated from the hot value (the `base` field when non-null — the
// per-substance field — else params.kinematic_viscosity) down to
// thermal_cold_viscosity at the freeze point. Cells no parcel supports keep
// the hot value. Returns false (and clears) when the chain is off, granular,
// or there are no particles; the caller then keeps whatever field it had.
// `base` may alias neither `viscosity_out` nor be resized by this call.
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
