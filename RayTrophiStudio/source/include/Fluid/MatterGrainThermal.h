#pragma once

// Grain carriers in the thermal chain (2026-10-19, HANDOFF_DEM_THERMAL 1-3).
// Until this a grain domain refused the thermal-liquid chain outright: hot
// gravel did not warm its water, sand in a fire stayed at room temperature and
// a tank of ice cubes could not melt. The DEM step does not need temperature,
// so grains keep theirs on the host, in particles.temperature, and every heat
// path below runs on the same canonical array as the liquid's.
//
// What already carried grains before this file:
//  * conductThermalLiquid - cell-to-cell conduction with each parcel's own
//    capacity and conductivity (Sand 0.27 W/(m K) is the bed value);
//  * exchangeLiquidSolidContact - parcels next to an MSF shell;
//  * updateThermalFreeze - skeleton carriers (Ice) melt paying their latent
//    heat and refreshMatterConstitutiveModels hands the melt to the liquid.
// What did not, and is done here, inside each liquid cell:
//  * grain <-> liquid: a 500 W/(m^2 K) film over the wetted share of the
//    grain's surface, against the cell's liquid as one pool;
//  * grain <-> air (the domain's own gas grid): 20 W/(m^2 K) plus the flame's
//    linearized radiation over the dry share, against the cell's air;
//  * grain <-> grain: contact conduction G = k * 4 pi r toward the cell's
//    grain mean (conduction between cells stays in conductThermalLiquid);
//  * grain <-> wall: a closed domain wall or a collider face without an MSF
//    shell is a container at `wall_kelvin`, reached through the grain contact
//    film (200 W/(m^2 K)) over a sixth of the grain's surface per touching
//    face. coolThermalLiquid skips grains: its capacity-free wall rate drained
//    a hot bed into the floor in seconds.
// Each exchange is the exact two-body relaxation, so nothing overshoots and
// the energy one side loses the other gains: sum C dT over grains, liquid and
// air is zero to rounding (reported per step).

#include "FluidThermalLiquid.h"
#include "MatterGrainGasCoupling.h"

namespace RayTrophiSim::Fluid {

// Grain area is 3 m / (rho r): a carrier heavier than one sphere of
// `params.grain.radius_m` (a super-grain) still exposes the surface of its
// mass in grains of that radius. `gas` may be null (no gas lane): the dry
// share then trades nothing. Writes temperature deltas into gas->temperature
// (normalized units, `air` maps them); the gas step uploads its host grid.
// Fills the grain_* rows of `stats`.
// `shell_capacity` (gatherThermalContact, may be null): collider cells with an
// MSF shell are not walls here - exchangeLiquidSolidContact trades with them.
// wall_kelvin <= 0: no wall exchange (the thermal chain is off).
void exchangeMatterGrainHeat(FluidParticles& particles, const FluidSim::FluidGrid& grid,
                             const APICSolverParams& params, FluidSim::FluidGrid* gas,
                             const MatterGrainGasAir& air, float wall_kelvin,
                             const std::vector<float>* shell_capacity, float dt,
                             ThermalLiquidStats& stats);

} // namespace RayTrophiSim::Fluid
