#pragma once

#include "MatterTransfer.h"
#include "../FluidGrid.h"

namespace RayTrophiSim::Fluid {

// Conservative reference lift from cell impulses to MAC faces. Momentum is
// conserved in the physical face-mass metric; a global energy line search
// prevents overlap averaging from injecting kinetic energy.
bool applyMatterMacContact(const FluidParticles& particles,
    FluidSim::FluidGrid& liquid, FluidSim::FluidGrid& granular,
    MatterConstitutiveModel legacy_model, double friction,
    MatterContactResult& result, std::string& error);

} // namespace RayTrophiSim::Fluid
