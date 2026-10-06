#pragma once

#include "APICFluidSolver.h"
#include <string>

namespace RayTrophiSim::Fluid {

MatterConstitutiveModel resolveSingleMatterModel(const FluidParticles& particles,
                                                 bool legacy_granular);
bool hasMixedMatterModels(const FluidParticles& particles, bool legacy_granular);
std::size_t estimateMixedMatterWorkingSet(const FluidParticles& particles,
                                         const FluidSim::FluidGrid& grid);
bool stepMixedMatter(FluidParticles& particles, FluidSim::FluidGrid& grid,
    const APICSolverParams& params, float dt, const SimulationForceFieldSnapshot* forces,
    float time_seconds, APICSolverStats& stats, std::string& error);

} // namespace RayTrophiSim::Fluid
