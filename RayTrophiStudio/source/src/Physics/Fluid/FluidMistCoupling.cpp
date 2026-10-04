#include "Fluid/FluidMistCoupling.h"

#include "Fluid/FluidParticleLabels.h"
#include "ParticleSimulation.h"
#include "Fluid/MatterPhaseGrid.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim::Fluid {

std::size_t applyMistGasDrag(SimulationGridDomainState& fluid_state,
                             const SimulationGridDomainDesc& gas_desc,
                             const SimulationGridDomainState& gas_state,
                             float dt) {
    if (!simulationDomainHasGas(gas_desc.type) || !gas_desc.enabled ||
        !gas_state.valid || !(dt > 0.0f) || !std::isfinite(dt)) {
        return 0;
    }

    FluidParticles& particles = fluid_state.particles;
    const std::size_t count = particles.size();
    if (particles.velocity.size() < count || particles.flags.size() < count) {
        return 0;
    }

    std::size_t carried = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (particleLabel(particles.flags[i]) != ParticleLabel::Mist) {
            continue;
        }
        const Vec3& p = particles.position[i];
        if (!gridContains(gasGrid(gas_state), p)) {
            continue;
        }

        const float mass = i < particles.mass_fraction.size()
            ? std::clamp(particles.mass_fraction[i], 0.02f,
                         kParticleMistMassFraction)
            : kParticleMistMassFraction;
        // Smaller droplets approach the carrier velocity faster. At the mist
        // boundary the response time is ~0.08 s; the smallest live parcel is
        // effectively entrained in one or two 60 Hz steps.
        const float response_seconds = 0.55f * mass;
        const float blend = 1.0f - std::exp(-dt / response_seconds);
        const Vec3 gas_velocity = gasGrid(gas_state).sampleVelocity(p);
        particles.velocity[i] +=
            (gas_velocity - particles.velocity[i]) * std::clamp(blend, 0.0f, 1.0f);
        ++carried;
    }
    return carried;
}

} // namespace RayTrophiSim::Fluid
