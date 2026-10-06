#pragma once

#include "MatterSolverStages.h"

#include <functional>

namespace RayTrophiSim::Fluid {

using MatterBatchContact = std::function<bool(
    const FluidParticles&, FluidSim::FluidGrid&,
    const FluidParticles&, FluidSim::FluidGrid&, std::string&)>;

struct MatterModelBatchResult {
    std::array<APICSolverStats, 2> model_stats;
    std::array<std::size_t, 2> before_count{};
    std::array<std::size_t, 2> after_count{};
    std::size_t removed_particles = 0;
};

// Atomic CPU coordinator for ONE resolved common substep. The caller provides
// independent model params (including borrowed fields) and the canonical MAC
// contact service. Missing contact is an error, never silent uncoupled motion.
// Canonical particles/grid are committed only after prepare/contact/gather all
// succeed. Single-model domains should use the existing direct fast path.
bool stepMatterModelBatch(FluidParticles& particles, FluidSim::FluidGrid& liquid_grid,
                         const std::array<APICSolverParams, 2>& model_params,
                         MatterConstitutiveModel legacy_model, float dt,
                         const SimulationForceFieldSnapshot* forces, float time_seconds,
                         const MatterBatchContact& contact, MatterModelBatchResult& result,
                         std::string& error);

} // namespace RayTrophiSim::Fluid
