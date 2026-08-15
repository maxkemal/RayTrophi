#include "ParticleSimulation.h"

#include <algorithm>

namespace RayTrophiSim {

bool ParticleSimulationSystem::seedFluidDomainParticles(
    const std::string& domain_name,
    const Vec3& seed_min,
    const Vec3& seed_max,
    int particles_per_cell,
    bool replace,
    bool persistent) {
    for (auto& domain : grid_domains_) {
        if (domain.name != domain_name ||
            domain.type != SimulationDomainType::Fluid) {
            continue;
        }

        domain.fluid_seed_mode = FluidSeedMode::SeedBox;
        domain.fluid_seed_min = Vec3::min(seed_min, seed_max);
        domain.fluid_seed_max = Vec3::max(seed_min, seed_max);
        domain.fluid_seed_particles_per_cell =
            std::clamp(particles_per_cell, 1, 64);
        domain.fluid_replace_on_seed = replace;
        domain.fluid_reseed_on_reset = persistent;
        domain.fluid_pending_seed = true;

        // Seed synchronously so API/IPC read-back immediately observes the
        // authoritative particle count. The old API seeded FluidObject now and
        // this state on the next tick, creating two independently simulated
        // blocks with the same name.
        synchronizeGridDomainsNow();
        return true;
    }
    return false;
}

bool ParticleSimulationSystem::clearFluidDomainParticles(
    const std::string& domain_name,
    bool clear_seed_recipe) {
    for (std::size_t i = 0; i < grid_domains_.size(); ++i) {
        auto& domain = grid_domains_[i];
        if (domain.name != domain_name ||
            domain.type != SimulationDomainType::Fluid) {
            continue;
        }

        domain.fluid_pending_seed = false;
        if (clear_seed_recipe) {
            domain.fluid_reseed_on_reset = false;
            // FillLevel is itself a persistent initial-state recipe. Returning
            // to SeedBox is necessary; clearing only the boolean would let the
            // next reset refill the domain through the mode predicate.
            domain.fluid_seed_mode = FluidSeedMode::SeedBox;
        }

        if (i < grid_domain_states_.size()) {
            auto& state = grid_domain_states_[i];
            state.particles.clear();
            state.foam.clear();
            state.fluid_stats = Fluid::APICSolverStats{};
            state.foam_stats = Fluid::FoamStats{};
            state.domain_motion_delta = Vec3(0.0f, 0.0f, 0.0f);
            state.molten_scan_tag = 0u;
            state.molten_scan_count = 0u;
            ++state.version;
        }
        if (i < grid_domain_compute_buffers_.size()) {
            auto& buffers = grid_domain_compute_buffers_[i];
            // A clear is a CPU-side authority change. Do not publish the last
            // Vulkan particle/density view while the domain is empty and has
            // not run another solver step yet.
            buffers.gpu_resident_fields_valid = false;
            buffers.foam_render.count = 0;
            ++buffers.foam_render.version;
            buffers.foam_expected_pending = false;
            buffers.foam_neigh_pending = false;
        }
        return true;
    }
    return false;
}

} // namespace RayTrophiSim
