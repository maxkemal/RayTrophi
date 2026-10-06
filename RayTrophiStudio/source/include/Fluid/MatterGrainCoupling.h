#pragma once

#include "MatterGrain.h"
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
// Not modelled (stated, not hidden): volume exclusion in the liquid's
// pressure projection (the liquid sees the pile only as drag, i.e. as a
// porous medium), pressure-gradient force beyond hydrostatic buoyancy,
// rotational drag, added mass, and wetting -- that is H1-G2 wet response.
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

// Order of the grain-owned carriers: granular model, sorted by stable
// identity. Liquid parcels keep their order. Fails on carriers that neither
// owner accepts (elastic, frozen, wet) or on Auto in a legacy-granular domain.
bool partitionMatterGrainOwners(const FluidParticles& particles, bool legacy_granular,
    std::vector<std::size_t>& liquid, std::vector<std::size_t>& grains, std::string& error);

// Exact snapshot copies of `order` (identity, every sidecar).
FluidParticles selectMatterParticles(const FluidParticles& particles,
    const std::vector<std::size_t>& order);

// particles := [liquid..., grains...]. Both subsets must be the full
// population (no births or removals); the merged order is stable frame to
// frame, so the next partition is the identity permutation.
bool mergeMatterGrainOwners(FluidParticles& particles, const FluidParticles& liquid,
    const FluidParticles& grains, std::string& error);

} // namespace RayTrophiSim::Fluid
