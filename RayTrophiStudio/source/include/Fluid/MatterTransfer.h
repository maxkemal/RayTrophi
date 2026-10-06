#pragma once

#include "FluidParticles.h"

#include <array>
#include <map>
#include <string>
#include <vector>

namespace RayTrophiSim::Fluid {

struct MatterMomentum {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct MatterModelTotals {
    std::size_t particles = 0;
    double mass_kg = 0.0;
    MatterMomentum momentum;
};

struct MatterCellField {
    double mass_kg = 0.0;
    MatterMomentum momentum;
    MatterMomentum mass_gradient;
};

using MatterCellKey = std::array<int, 3>;

struct MatterTransferCell {
    // Fluid and granular fields never alias. Elastic/unknown parcels are
    // reported but cannot accidentally enter either transfer accumulator.
    std::array<MatterCellField, 2> model;
};

struct MatterTransferFrame {
    std::array<MatterModelTotals, 4> totals;
    std::array<std::vector<std::size_t>, 4> indices;
    std::map<MatterCellKey, MatterTransferCell> cells;
    std::map<MatterCellKey, std::vector<uint64_t>> occupants;
    std::size_t overlapping_cells = 0;
    std::size_t outside_particles = 0;
    double outside_mass_kg = 0.0;
    double deposited_mass_kg = 0.0;
    MatterMomentum deposited_momentum;
};

// Read-only C4a transfer reference. Weights use the same quadratic 3x3x3
// support as APIC. Bounds clipping is normalized to preserve parcel mass.
// Index lanes: fluid, granular, elastic, unresolved.
bool buildMatterTransfer(const FluidParticles& particles, const Vec3& origin,
                         float voxel, const std::array<int, 3>& dimensions,
                         MatterConstitutiveModel legacy_model,
                         double legacy_mass_kg, MatterTransferFrame& result,
                         std::string& error, bool include_transfer = true);

struct MatterContactResult {
    std::size_t pairs = 0;
    double impulse_norm = 0.0;
    double kinetic_energy_loss = 0.0;
    MatterMomentum momentum_error;
};

// Pairwise equal/opposite normal contact + Coulomb tangential impulse between
// the two fields. This does not run liquid pressure on granular matter.
// Input validation is transactional. Not yet attached to the live solver.
bool applyMatterGridContact(MatterTransferFrame& frame, double friction,
                           MatterContactResult& result, std::string& error);

} // namespace RayTrophiSim::Fluid
