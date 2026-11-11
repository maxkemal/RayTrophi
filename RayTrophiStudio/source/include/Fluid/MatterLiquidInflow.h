// Liquid inflow / outflow regions of a flow source (2026-11-09).
//
// A "rate" emitter scatters N parcels per second into a sphere once per frame:
// its flux has nothing to do with its velocity or nozzle size, its parcels are
// born as a puff per frame, and a nozzle under one cell silently clamped the
// rate (over-pack guard). A physical inflow is a NOZZLE: the liquid fills its
// cross-section A and leaves at velocity v, so the flux is Q = A * |v|. This is
// what FLIP sources do in production (Houdini source volume + velocity,
// Bifrost continuous emitter, Mantaflow inflow): keep the inflow region FULL
// at the target parcel density and hold the velocity inside it every step.
//
//  * Region: a disk nozzle (Point mode: center = the face, axis = flow
//    direction, radius) or an axis-aligned box (ObjectBounds mode), swept
//    UPSTREAM by `sweep` = max(|v| dt, h): the liquid that crosses the face
//    during the coming step. Topping the swept region up to particles_per_cell
//    requests births where material is missing. After initial reservoir fill,
//    the physical birth budget is at most A |v| dt, with fractional parcel carry.
//    Fill noise or redistribution cannot create extra source volume; a blocked
//    region can accept less, and unused capacity never becomes a backlog.
//  * Hold: parcels inside the region get v - g dt, so that after the step's
//    gravity they leave the face at v (the nozzle walls carry the column's
//    weight). Without it the whole slug fell g dt^2 (17 mm at 24 fps) during
//    its birth step and the stream broke into one blob per frame.
//  * Outflow: parcels inside a sphere (Point) or box (ObjectBounds) are
//    removed; the mass is recorded as Drainage.
#pragma once

#include "Fluid/FluidParticles.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace RayTrophiSim::Fluid {

struct LiquidFlowRegion {
    enum class Shape { Disk, Box, Sphere };
    Shape shape = Shape::Disk;
    Vec3 center = Vec3(0.0f);   // disk face / sphere centre
    Vec3 axis = Vec3(0.0f, -1.0f, 0.0f);  // unit flow direction (Disk, Box)
    float radius = 0.0f;        // disk / sphere
    Vec3 box_min = Vec3(0.0f);  // Box
    Vec3 box_max = Vec3(0.0f);
    float sweep = 0.0f;         // upstream length (Disk, Box), m
};

// The cell lattice parcels are counted on (the domain's liquid grid).
struct LiquidLattice {
    Vec3 origin = Vec3(0.0f);
    float h = 0.0f;
    int nx = 0, ny = 0, nz = 0;
};

bool insideLiquidFlowRegion(const LiquidFlowRegion& region, const Vec3& p);
void liquidFlowRegionBounds(const LiquidFlowRegion& region, Vec3& lo, Vec3& hi);
// Cross-section the flow crosses, m^2: pi r^2 for a disk, the box's
// silhouette normal to the axis for a box.
float liquidFlowRegionArea(const LiquidFlowRegion& region);

struct LiquidInflowPlan {
    std::vector<Vec3> births;          // where the missing parcels go
    std::vector<std::size_t> held;     // parcels already inside the region
    std::size_t region_parcels = 0;    // parcels of a full region
};

// Jittered-stratified particles_per_cell samples in every lattice cell the
// region touches; a cell's samples inside the region, minus the parcels
// already inside the region in that cell, are the births. `seed` decorrelates
// steps (identical sample sites every step stacked parcels that reseed then
// trimmed, see MatterDomainSources.inl).
void planLiquidInflow(const LiquidFlowRegion& region, const LiquidLattice& lattice,
                      int particles_per_cell, const FluidParticles& particles,
                      uint64_t seed, LiquidInflowPlan& out);

// First activation may fill the upstream reservoir. Later steps may add at most
// Q*dt/parcel_volume parcels, carrying only the fractional parcel between steps.
// Unused whole-parcel capacity is discarded: blocked or pool-limited nozzles
// must not release a backlog. Subsampling covers the entire candidate region.
void limitLiquidInflowBirths(LiquidInflowPlan& plan, double requested_parcels,
                            bool& initialized, double& fractional_parcel,
                            uint64_t seed);

// Marks every parcel inside the region for removal; returns how many.
std::size_t markLiquidOutflow(const LiquidFlowRegion& region,
                              const FluidParticles& particles,
                              std::vector<uint8_t>& remove_mask);

} // namespace RayTrophiSim::Fluid
