#pragma once

// Liquid -> surface water (2026-10-18). Water a surface soaks up is taken out of
// the liquid parcels that touch it, kilogram for kilogram. Before this the MSF
// wetting pass raised a dimensionless moisture wherever a surface sat in water,
// and the liquid never lost anything.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace FluidSim {
class FluidGrid;
}
namespace RayTrophiSim::Fluid {
class FluidParticles;
struct APICSolverParams;

struct SurfaceWaterExchangeStats {
    std::size_t parcels = 0;      // water parcels that gave mass this step
    double absorbed_kg = 0.0;
    double energy_j = 0.0;        // the absorbed water's sensible energy (c T)
    double heat_to_surface_j = 0.0;  // c dm (T_parcel - T_surface), into the surface
};

// Per cell of `grid` (cell-centred, as MaterialStateFieldSystem::
// gatherWaterContact binned it): how much water the surfaces there can still
// take (kg), how fast (kg/s) and their temperature (K). Water parcels in a
// cell, or in one of its six face neighbours, give up to half their mass per
// step, the cell taking at most min(deficit, rate dt). A parcel is never left
// below the live-parcel floor: it is either kept at >= 3 % of its rest mass or
// taken whole. Writes what each cell absorbed and the sensible heat it brings.
// Only parcels of the substance "Water" take part: the surface holds water,
// and its vapour joins the gas inventory's water pool when it dries.
SurfaceWaterExchangeStats exchangeLiquidSurfaceWater(
    FluidParticles& particles, const FluidSim::FluidGrid& grid,
    const APICSolverParams& params, float dt,
    const std::vector<float>& cell_deficit_kg, const std::vector<float>& cell_rate_kg_s,
    const std::vector<float>& cell_kelvin,
    std::vector<float>& cell_absorbed_kg, std::vector<float>& cell_absorbed_heat_j);

} // namespace RayTrophiSim::Fluid
