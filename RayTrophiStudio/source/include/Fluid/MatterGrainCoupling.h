#pragma once

#include "MatterGrain.h"
#include "../FluidGrid.h"
#include "../Vec3.h"

#include <cstddef>
#include <string>
#include <vector>

namespace RayTrophiSim::Fluid {
class FluidParticles;
enum class FluidChemistryPreset : int;

// One frame of the grain <-> liquid exchange in a shared Matter domain.
//
// Unresolved CFD-DEM, staggered per frame: the liquid steps first, its end
// state is binned on the domain grid (cell-centred mass/volume/momentum, and
// the grains' solid volume), each grain samples it with trilinear weights,
// the grain step integrates drag implicitly against a private liquid lump,
// and the opposite impulse is returned to the liquid parcels of the same
// cells with the same weights. Every exchanged impulse lands on liquid mass,
// so the pair conserves momentum exactly (up to float rounding).
//
// Each cell's liquid mass is divided among the grains that overlap it in
// proportion to their volume weight; the lumps of one cell never exceed the
// cell's liquid mass (equal once a cell holds a grain volume or more), so a
// dense pile cannot pull more liquid momentum than the cell holds and the
// implicit pair cannot overshoot. A lone grain sees the trilinearly sampled
// liquid of its cells.
//
// B5 (volume_exclusion): the liquid's projection sees the grains' volume.
// The variational face weights carry the pore fraction eps and the closed
// part of a face moves with the grains, so the projection enforces
// div(eps u_l + (1 - eps) u_g) = 0; the grains then take the solver's real
// pressure gradient, -V rho grad(p), instead of hydrostatic buoyancy, and the
// liquid gets it back (unresolved CFD-DEM "model A": the liquid ends up with
// -eps grad p). Without it the liquid sees the pile only as drag.
//
// Not modelled (stated, not hidden): rotational drag, added mass, and
// wetting -- that is H1-G2 wet response (B6).
struct MatterGrainLiquidField {
    int nx = 0, ny = 0, nz = 0;
    Vec3 origin;
    float h = 0.0f;
    std::vector<double> mass;         // liquid kg
    std::vector<double> volume;       // liquid m^3
    std::vector<double> momentum[3];  // liquid kg m/s
    std::vector<double> solid;        // grain m^3 (trilinear volume weights)
    std::vector<int> parcel_cell;     // liquid parcel -> cell, -1 outside
};

struct MatterGrainCouplingFrame {
    MatterGrainLiquidField field;
    std::vector<MatterGrainCouplingInput> inputs;   // one per grain
    std::vector<float> density;       // liquid density sampled per grain, kg/m^3
    std::vector<float> submerged;     // submerged fraction per grain, 0..1
    // Per grain: the eight neighbour cells and their share of this grain's
    // liquid lump (sums to 1 over cells with liquid; empty when uncoupled).
    std::vector<int> cells;           // 8 per grain, -1 unused
    std::vector<float> shares;        // 8 per grain
    std::vector<Vec3> buoyancy_impulse;  // per grain, whole frame
};

// Bins the liquid and the grains. `liquid` and `grains` are the two
// transport-owner subsets of one domain.
bool buildMatterGrainLiquidField(const FluidParticles& liquid, const FluidParticles& grains,
    float grain_radius, FluidChemistryPreset chemistry_preset, const Vec3& origin,
    int nx, int ny, int nz, float h, MatterGrainLiquidField& field, std::string& error);

// Per-grain drag coefficient (Di Felice 1994 with voidage), Archimedes
// buoyancy and the liquid lump. Fills `report` coupling counters.
void prepareMatterGrainCoupling(const FluidParticles& grains, const MatterGrainParams& params,
    const Vec3& gravity, float dt, MatterGrainCouplingFrame& frame,
    MatterGrainStepReport& report);

// Di Felice drag magnitude / relative speed (kg/s) for one sphere. Exposed
// for the contract test; `voidage` in (0, 1].
double matterGrainDragCoefficient(double diameter, double relative_speed, double density,
    double viscosity, double voidage);

// Returns -(drag + buoyancy) impulse of every grain to the liquid parcels of
// the grain's cells. Velocities of `liquid` change; nothing else does.
void applyMatterGrainLiquidReaction(FluidParticles& liquid, const MatterGrainCouplingFrame& frame,
    const std::vector<MatterGrainCouplingOutput>& drag, MatterGrainStepReport& report);

// Grid state the porous projection overwrites for one liquid step.
struct MatterGrainPorosityBackup {
    bool active = false;
    std::vector<uint8_t> u_weight, v_weight, w_weight;
    std::vector<Vec3> solid_vel;
    bool weights_init = false;
    uint64_t weights_sig = 0;
};

// Writes eps into the face weights (on top of the collider weights when the
// domain maintains them, else on fully open faces) and the grains' mean
// velocity into solid_vel of non-solid cells. Pore fraction never drops
// below `minimum_voidage`. Fills report.porous_cells/max_solid_fraction.
void applyMatterGrainPorosity(FluidSim::FluidGrid& grid, const FluidParticles& grains,
    float grain_radius, bool keep_collider_weights, float minimum_voidage,
    MatterGrainPorosityBackup& backup, MatterGrainStepReport& report);
// Restores exactly what applyMatterGrainPorosity replaced.
void restoreMatterGrainPorosity(FluidSim::FluidGrid& grid, MatterGrainPorosityBackup& backup);

// Replaces hydrostatic buoyancy by the projection's pressure gradient:
// F = -V rho grad(p_kin), p_kin the solver's kinematic pressure (p / rho, the
// value sim_fluid_subtract_gradient uses), mask the device fluid mask
// (> .5 liquid, < -.5 solid -> Neumann, else air p = 0); closed domain walls
// are Neumann. Grains with no liquid share keep zero force.
void applyMatterGrainPressureForce(const std::vector<float>& pressure,
    const std::vector<float>& mask, const FluidParticles& grains, float grain_radius,
    float dt, MatterGrainCouplingFrame& frame, MatterGrainStepReport& report);

// B6 wet grains, after the drag reaction (so the reaction's cell masses are
// the ones it was computed with). Absorption moves water from the liquid
// parcels of a grain's cells (the same shares as its lump) into the grain's
// pore_water_mass_kg, at most half a cell's liquid per frame; the moved
// water carries the liquid's cell velocity, so momentum is exact. Drying
// removes held water from the domain (reported, not hidden). Capacity is
// water_capacity_fraction of the sphere volume (pore_capacity_kg).
void exchangeMatterGrainWater(FluidParticles& liquid, FluidParticles& grains,
    const MatterGrainCouplingFrame& frame, const MatterGrainParams& params, float dt,
    MatterGrainStepReport& report);

// Order of the grain-owned carriers: granular model, sorted by stable
// identity. Liquid parcels keep their order. Fails on carriers that neither
// owner accepts (elastic, frozen; wet grains unless wet_grains) or on Auto in
// a legacy-granular domain.
bool partitionMatterGrainOwners(const FluidParticles& particles, bool legacy_granular,
    bool wet_grains, std::vector<std::size_t>& liquid, std::vector<std::size_t>& grains,
    std::string& error);

// B8: orders grain indices by the Morton code of their contact cell (size
// `cell`, from `origin`), ties by identity. Neighbours then sit close in the
// device arrays, so the step's neighbour reads hit nearby memory; the
// runtime gathers contact history to the new order by identity.
void orderMatterGrainsByCell(const FluidParticles& particles, const Vec3& origin, float cell,
    std::vector<std::size_t>& grains);

// Exact snapshot copies of `order` (identity, every sidecar).
FluidParticles selectMatterParticles(const FluidParticles& particles,
    const std::vector<std::size_t>& order);

// particles := [liquid..., grains...]. Both subsets must be the full
// population (no births or removals); the merged order is stable frame to
// frame, so the next partition is the identity permutation.
bool mergeMatterGrainOwners(FluidParticles& particles, const FluidParticles& liquid,
    const FluidParticles& grains, std::string& error);

} // namespace RayTrophiSim::Fluid
