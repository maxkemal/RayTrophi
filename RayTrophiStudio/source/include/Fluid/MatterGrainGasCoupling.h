#pragma once

// T5 / FG1 (2026-10-18): grains in a Matter domain feel the domain's own gas.
// Before this the grain lane sampled only the liquid (buildMatterGrainLiquidField):
// wind, smoke and a fire's updraft did not move sand, and sand did not push air.

#include "MatterGrain.h"
#include "../Vec3.h"

#include <vector>

namespace FluidSim {
class FluidGrid;
}
namespace RayTrophiSim::Fluid {
class FluidParticles;

// The gas lane's temperature field holds the VIRTUAL temperature Tv
// (GasSaturation.h); the air density at 1 atm is exactly P Ma / (R Tv), so no
// vapour bookkeeping is needed for it. kelvin_per_unit <= 0: assume 293.15 K.
struct MatterGrainGasAir {
    float ambient_kelvin = 293.15f;  // K at gas_ambient
    float gas_ambient = 0.0f;        // normalized field value of the ambient
    float kelvin_per_unit = 0.0f;
};

// One frame of grain <-> gas drag, staggered like the liquid coupling: the gas
// stepped last frame and its host grid is current (the gas step ends with a
// host sync); grains step next; the gas steps after them this frame.
//
// Per grain: Di Felice drag (matterGrainDragCoefficient, voidage 1) with air
// density from Tv and Sutherland viscosity, integrated exactly over the frame
// against the frame-start relative velocity:
//     dv = (u_gas - v)(1 - exp(-k dt / m)),
// handed to the DEM substeps as a constant ACCELERATION dv/dt (added to
// `acceleration`, resized to the grain count). A force, not a velocity nudge,
// so static friction in a resting pile can hold against a breeze.
//
// Per gas cell the grains' total impulse is limited so the gas cannot be
// pushed past the mixture's equilibrium velocity (M_gas u + sum m v)/(M + sum m):
// a dense sand cloud in a light cell cannot make the air overshoot. The same
// factor scales those grains' impulses, so the pair stays exactly opposite.
// The reaction -J is splatted onto the gas MAC faces with the adjoint of the
// trilinear velocity sample (momentum / (rho V)).
//
// Not modelled: buoyancy in air (rho_air / rho_grain ~ 5e-4), the grains'
// volume in the gas projection, drag dissipation as heat.
void exchangeMatterGrainGasDrag(const FluidParticles& grains, float grain_radius,
                                FluidSim::FluidGrid& gas, const MatterGrainGasAir& air,
                                float dt, std::vector<Vec3>& acceleration,
                                MatterGrainStepReport& report);

} // namespace RayTrophiSim::Fluid
