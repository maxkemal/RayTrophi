#include "Fluid/FluidGasMovingBoundary.h"

#include "Fluid/FluidParticleLabels.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"
#include "ParticleSimulation.h"
#include "Fluid/MatterPhaseGrid.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace RayTrophiSim::Fluid {
namespace {

bool overlaps(const SimulationGridDomainDesc& a,
              const SimulationGridDomainDesc& b) {
    const Vec3 lo = Vec3::max(a.bounds_min, b.bounds_min);
    const Vec3 hi = Vec3::min(a.bounds_max, b.bounds_max);
    return lo.x < hi.x && lo.y < hi.y && lo.z < hi.z;
}

} // namespace

FluidGasMovingBoundaryStats buildFluidGasMovingBoundary(
    const std::vector<SimulationGridDomainDesc>& domains,
    const std::vector<SimulationGridDomainState>& states,
    std::size_t gas_domain_index,
    std::vector<uint32_t>& cells_out,
    std::vector<Vec3>& velocities_out) {
    FluidGasMovingBoundaryStats stats;
    cells_out.clear();
    velocities_out.clear();
    if (gas_domain_index >= domains.size() || gas_domain_index >= states.size()) {
        return stats;
    }

    const SimulationGridDomainDesc& gas_domain = domains[gas_domain_index];
    const SimulationGridDomainState& gas_state = states[gas_domain_index];
    const FluidSim::FluidGrid& gas = gasGrid(gas_state);
    const std::size_t cell_count = gas.getCellCount();
    if (!gas_domain.enabled || !simulationDomainHasGas(gas_domain.type) ||
        !gas_state.valid || cell_count == 0 || !(gas.voxel_size > 0.0f)) {
        return stats;
    }

    std::vector<double> volume(cell_count, 0.0);
    std::vector<double> mass(cell_count, 0.0);
    std::vector<Vec3> momentum(cell_count, Vec3(0.0f));
    const float inv_h = 1.0f / gas.voxel_size;

    for (std::size_t domain_index = 0;
         domain_index < domains.size() && domain_index < states.size();
         ++domain_index) {
        if (domain_index == gas_domain_index &&
            gas_domain.type != SimulationDomainType::Matter) {
            continue;
        }
        const SimulationGridDomainDesc& fluid_domain = domains[domain_index];
        const SimulationGridDomainState& fluid_state = states[domain_index];
        if (!fluid_domain.enabled || !simulationDomainHasLiquid(fluid_domain.type) ||
            !fluid_state.valid || fluid_state.particles.empty() ||
            !overlaps(fluid_domain, gas_domain)) {
            continue;
        }
        ++stats.source_domains;

        const FluidParticles& particles = fluid_state.particles;
        for (std::size_t particle = 0; particle < particles.size(); ++particle) {
            if (particle < particles.flags.size() &&
                particleLabel(particles.flags[particle]) == ParticleLabel::Mist) {
                continue;
            }
            const float fraction = particle < particles.mass_fraction.size()
                ? std::clamp(particles.mass_fraction[particle], 0.0f, 1.0f) : 1.0f;
            const float rest_mass = particle < particles.rest_mass_kg.size()
                ? particles.rest_mass_kg[particle] : 0.0f;
            const float particle_mass = rest_mass * fraction;
            if (!(particle_mass > 0.0f) || !std::isfinite(particle_mass)) {
                continue;
            }
            const uint32_t tag = particle < particles.substance_tag.size()
                ? particles.substance_tag[particle] : kSubstanceUntagged;
            const SubstanceProfile* profile = resolveFluidSubstanceProfile(
                tag, fluid_domain.fluid_params.chemistry_preset);
            const float density = profile && profile->liquid_density > 0.0f
                ? profile->liquid_density : 1000.0f;
            const Vec3 local = (particles.position[particle] - gas.origin) * inv_h;
            const int i = static_cast<int>(std::floor(local.x));
            const int j = static_cast<int>(std::floor(local.y));
            const int k = static_cast<int>(std::floor(local.z));
            if (i < 0 || i >= gas.nx || j < 0 || j >= gas.ny ||
                k < 0 || k >= gas.nz) {
                continue;
            }
            const std::size_t cell = gas.cellIndex(i, j, k);
            volume[cell] += static_cast<double>(particle_mass) / density;
            mass[cell] += particle_mass;
            if (particle < particles.velocity.size()) {
                momentum[cell] += particles.velocity[particle] * particle_mass;
            }
            ++stats.source_particles;
        }
    }

    const double cell_volume = static_cast<double>(gas.voxel_size) *
        gas.voxel_size * gas.voxel_size;
    const double enter_volume = 0.25 * cell_volume;
    const double release_volume = 0.15 * cell_volume;
    std::vector<uint8_t> was_boundary(cell_count, 0u);
    for (uint32_t cell : gas.substance_solid_prev_cells) {
        if (cell < cell_count) {
            was_boundary[cell] = 1u;
        }
    }

    Vec3 velocity_sum(0.0f);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        const double threshold = was_boundary[cell] ? release_volume : enter_volume;
        if (volume[cell] < threshold || !(mass[cell] > 0.0)) {
            continue;
        }
        const Vec3 velocity = momentum[cell] * static_cast<float>(1.0 / mass[cell]);
        cells_out.push_back(static_cast<uint32_t>(cell));
        velocities_out.push_back(velocity);
        velocity_sum += velocity;
    }
    stats.boundary_cells = cells_out.size();
    if (stats.boundary_cells > 0) {
        stats.mean_velocity = velocity_sum *
            (1.0f / static_cast<float>(stats.boundary_cells));
    }
    return stats;
}

} // namespace RayTrophiSim::Fluid
