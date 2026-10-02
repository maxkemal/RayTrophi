#include "Fluid/FluidParticleLabels.h"
#include "ParticleSimulation.h"

namespace RayTrophiSim {
namespace Fluid {

ParticleLabelReport inspectParticleLabels(const SimulationGridDomainState* state) {
    ParticleLabelReport report;
    if (!state || !state->valid || !simulationDomainHasLiquid(state->type)) {
        return report;
    }
    report.available = true;
    report.primary_particles = state->particles.size();
    report.secondary_particles = state->foam.size();
    report.primary = countParticleLabels(state->particles);
    report.secondary = countSecondaryParticleLabels(state->foam);
    report.last_step = state->particle_label_stats;
    return report;
}

} // namespace Fluid
} // namespace RayTrophiSim
