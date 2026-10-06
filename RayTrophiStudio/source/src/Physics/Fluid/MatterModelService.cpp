#include "Fluid/MatterModelService.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/FluidPhysicalMass.h"

namespace RayTrophiSim::Fluid {

bool inspectMatterModels(const SimulationGridDomainDesc& domain,
                         const SimulationGridDomainState* state,
                         bool include_transfer, MatterTransferFrame& frame,
                         std::string& error) {
    if (!simulationDomainHasLiquid(domain.type)) {
        error = "domain has no particle phase";
        return false;
    }
    if (!state || !state->valid) {
        frame = {};
        error.clear();
        return true;
    }
    if (include_transfer && state->particles.size() > 250000) {
        error = "CPU transfer reference is limited to 250000 particles";
        return false;
    }
    const auto& grid = liquidGrid(*state);
    const auto legacy_model = domain.fluid_params.granular_enabled
        ? MatterConstitutiveModel::Granular : MatterConstitutiveModel::Fluid;
    // Existing parcels' physical rest masses remain authoritative. Only old
    // zero/unwritten mass sidecars use the established domain mass policy.
    const double fallback_mass = fluidParticleRestMassKg(
        0u, domain.fluid_params.chemistry_preset, grid.voxel_size,
        domain.fluid_params.particles_per_cell, legacy_model);
    return buildMatterTransfer(state->particles, grid.origin, grid.voxel_size,
        {grid.nx, grid.ny, grid.nz}, legacy_model, fallback_mass,
        frame, error, include_transfer);
}

} // namespace RayTrophiSim::Fluid
