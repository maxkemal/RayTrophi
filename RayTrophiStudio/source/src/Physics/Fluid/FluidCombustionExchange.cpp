#include "Fluid/FluidCombustionExchange.h"

#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"
#include "MatterExchangeLedger.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>

namespace RayTrophiSim {
namespace Fluid {

namespace {

struct ExchangeAccumulation {
    double mass_kg = 0.0;
    double sensible_energy_j = 0.0;
    double latent_energy_j = 0.0;
    Vec3 momentum_kg_m_s;
};

bool overlaps(const SimulationGridDomainDesc& a,
              const SimulationGridDomainDesc& b) {
    const Vec3 lo = Vec3::max(a.bounds_min, b.bounds_min);
    const Vec3 hi = Vec3::min(a.bounds_max, b.bounds_max);
    return lo.x < hi.x && lo.y < hi.y && lo.z < hi.z;
}

float gasTemperatureKelvin(float temperature, float ambient_kelvin) {
    if (temperature > 20.0f) return temperature;
    return ambient_kelvin + std::clamp(temperature, 0.0f, 1.0f) * 1500.0f;
}

bool isFreeSurfaceParticle(const Vec3& position,
                           const FluidSim::FluidGrid& liquid) {
    const std::size_t cell_count = liquid.getCellCount();
    if (liquid.nx <= 0 || liquid.ny <= 0 || liquid.nz <= 0 ||
        liquid.density.size() != cell_count ||
        !(liquid.voxel_size > 1.0e-6f)) {
        return false;
    }

    const Vec3 local = (position - liquid.origin) / liquid.voxel_size;
    const int x = static_cast<int>(std::floor(local.x));
    const int y = static_cast<int>(std::floor(local.y));
    const int z = static_cast<int>(std::floor(local.z));
    auto occupied = [&](int cx, int cy, int cz) {
        if (cx < 0 || cx >= liquid.nx || cy < 0 || cy >= liquid.ny ||
            cz < 0 || cz >= liquid.nz) {
            return false;
        }
        return liquid.density[liquid.cellIndex(cx, cy, cz)] > 0.02f;
    };
    if (!occupied(x, y, z)) {
        return false;
    }

    // Match the established GPU surface bridge: upper and lateral exposure
    // emit vapour, while the lower face against a floor does not.
    return !occupied(x, y + 1, z) || !occupied(x + 1, y, z) ||
           !occupied(x - 1, y, z) || !occupied(x, y, z + 1) ||
           !occupied(x, y, z - 1);
}

} // namespace

FluidCombustionExchangeStats transferFluidCombustionToGas(
    const SimulationGridDomainDesc& fluid_domain,
    SimulationGridDomainState& fluid_state,
    const SimulationGridDomainDesc& gas_domain,
    SimulationGridDomainState& gas_state,
    float dt,
    uint64_t event_id_base,
    MatterExchangeLedger& ledger) {
    FluidCombustionExchangeStats stats;
    if (!fluid_domain.enabled || !gas_domain.enabled ||
        !fluid_domain.fluid_flammable || fluid_domain.fluid_extinguishing ||
        !simulationDomainHasLiquid(fluid_domain.type) ||
        !simulationDomainHasGas(gas_domain.type) || !gas_domain.fire_enabled ||
        !fluid_state.valid || !gas_state.valid || fluid_state.particles.empty() ||
        !(dt > 0.0f) || !std::isfinite(dt) || !overlaps(fluid_domain, gas_domain)) {
        return stats;
    }

    FluidParticles& particles = fluid_state.particles;
    ensureFluidParticleRestMasses(
        particles, fluid_domain.fluid_params.chemistry_preset,
        fluid_state.voxel_size, fluid_domain.fluid_params.particles_per_cell);

    FluidSim::FluidGrid& gas = gas_state.grid;
    const FluidSim::FluidGrid& liquid =
        fluid_domain.type == SimulationDomainType::Matter
            ? fluid_state.matter_liquid_grid
            : fluid_state.grid;
    const std::size_t cell_count = gas.getCellCount();
    if (gas.temperature.size() != cell_count || gas.fuel.size() != cell_count ||
        gas.density.size() != cell_count || gas.interaction.size() != cell_count ||
        !(gas.voxel_size > 1.0e-6f)) {
        return stats;
    }
    gas_state.gas_phase_mass_kg.resize(cell_count, 0.0f);
    gas_state.gas_phase_energy_j.resize(cell_count, 0.0f);
    stats.ran = true;

    const float ignition = std::max(0.0f, fluid_domain.fluid_ignition_temperature);
    const float rate = std::max(0.0f, fluid_domain.fluid_evaporation_rate);
    const float ambient_kelvin = std::max(1.0f, fluid_domain.thermal_ambient_kelvin);
    const float cooling = std::max(0.0f, fluid_domain.fluid_surface_cooling);
    const float conductivity = std::max(
        0.0f, fluid_domain.fluid_params.granular_thermal_conductivity);
    const float visual_heat = std::max(
        0.0f, fluid_domain.fluid_combustion_heat_release);
    const float visual_smoke = std::max(
        0.0f, fluid_domain.fluid_combustion_smoke_yield);
    const float inv_gas_voxel = 1.0f / gas.voxel_size;
    const float fluid_cell_volume = std::max(
        fluid_state.voxel_size * fluid_state.voxel_size * fluid_state.voxel_size,
        1.0e-12f);
    std::map<std::string, ExchangeAccumulation> exchanges;

    for (std::size_t particle = particles.size(); particle-- > 0;) {
        // Combustible vapour is an interface flux. Applying the authored rate
        // to every parcel makes a uniformly seeded pool reach the removal
        // threshold on one frame and vanish as a single block.
        if (!isFreeSurfaceParticle(particles.position[particle], liquid)) {
            continue;
        }
        const uint32_t tag = particle < particles.substance_tag.size()
            ? particles.substance_tag[particle] : kSubstanceUntagged;
        const bool tagged = tag != kSubstanceUntagged;
        const float combustible = tagged &&
            particle < particles.combustible_fraction.size()
            ? std::clamp(particles.combustible_fraction[particle], 0.0f, 1.0f)
            : 1.0f;
        if (!(combustible > 0.0f)) continue;

        const Vec3 local = (particles.position[particle] - gas.origin) *
            inv_gas_voxel;
        const int gx = static_cast<int>(std::floor(local.x));
        const int gy = static_cast<int>(std::floor(local.y));
        const int gz = static_cast<int>(std::floor(local.z));
        if (gx < 0 || gx >= gas.nx || gy < 0 || gy >= gas.ny ||
            gz < 0 || gz >= gas.nz) {
            continue;
        }
        const std::size_t cell = gas.cellIndex(gx, gy, gz);
        const float gas_temperature = gas.temperature[cell];
        const float gas_kelvin = gasTemperatureKelvin(
            gas_temperature, ambient_kelvin);
        if (particle < particles.temperature.size()) {
            float& particle_kelvin = particles.temperature[particle];
            if (gas_kelvin > particle_kelvin) {
                particle_kelvin += (gas_kelvin - particle_kelvin) *
                    (1.0f - std::exp(-conductivity * dt));
            }
            if (cooling > 0.0f && particle_kelvin > ambient_kelvin) {
                particle_kelvin += (ambient_kelvin - particle_kelvin) *
                    (1.0f - std::exp(-cooling * dt));
            }
        }
        if ((!fluid_domain.fluid_auto_ignite && gas_temperature < ignition) ||
            !(rate > 0.0f)) {
            continue;
        }

        // Auto Ignite is the authored pilot for a combustible liquid. It must
        // be able to seed an empty gas phase; otherwise a newly unified Matter
        // domain deadlocks with no hot gas to evaporate the first parcel and
        // therefore no fuel that could create hot gas. Once a pilot or flame
        // exists, the ordinary temperature response remains unchanged.
        const float heat_factor = fluid_domain.fluid_auto_ignite
            ? 1.0f
            : std::clamp(
                  (gas_temperature - ignition) / std::max(ignition, 1.0f),
                  0.0f, 1.0f);
        float& fraction = particles.mass_fraction[particle];
        const float before = std::clamp(fraction, 0.0f, 1.0f);
        float after = std::clamp(
            before - rate * heat_factor * combustible * dt, 0.0f, 1.0f);
        if (after <= 0.02f) after = 0.0f;
        const float lost_fraction = std::max(before - after, 0.0f);
        if (!(lost_fraction > 0.0f)) continue;

        const float mass_kg = particles.rest_mass_kg[particle] * lost_fraction;
        if (!(mass_kg > 0.0f) || !std::isfinite(mass_kg)) continue;
        fraction = after;
        const SubstanceProfile* profile = resolveFluidSubstanceProfile(
            tag, fluid_domain.fluid_params.chemistry_preset);
        const float density = profile && profile->liquid_density > 0.0f
            ? profile->liquid_density : 1000.0f;
        const float specific_heat = profile && profile->specific_heat > 0.0f
            ? profile->specific_heat
            : std::max(1.0f, fluid_domain.fluid_params.fuel_profile.heat_capacity *
                                  1000.0f);
        const float latent_heat = profile
            ? std::max(0.0f, profile->latent_heat_vaporization)
            : std::max(0.0f, fluid_domain.fluid_params.fuel_profile.latent_heat *
                                  1.0e6f);
        const float particle_kelvin = particle < particles.temperature.size()
            ? std::max(0.0f, particles.temperature[particle]) : ambient_kelvin;
        const double sensible_energy = static_cast<double>(mass_kg) *
            static_cast<double>(specific_heat) * particle_kelvin;
        const double latent_energy = static_cast<double>(mass_kg) * latent_heat;
        const float tracer = mass_kg /
            std::max(density * fluid_cell_volume, 1.0e-12f);

        gas.fuel[cell] += tracer;
        gas.temperature[cell] += tracer * visual_heat;
        gas.density[cell] += tracer * visual_smoke;
        gas.interaction[cell] += tracer;
        gas_state.gas_phase_mass_kg[cell] += mass_kg;
        gas_state.gas_phase_energy_j[cell] +=
            static_cast<float>(sensible_energy + latent_energy);

        const std::string substance = profile ? profile->name : "Custom";
        ExchangeAccumulation& exchange = exchanges[substance];
        exchange.mass_kg += mass_kg;
        exchange.sensible_energy_j += sensible_energy;
        exchange.latent_energy_j += latent_energy;
        exchange.momentum_kg_m_s += particles.velocity[particle] * mass_kg;
        ++stats.particles_changed;
        stats.transferred_mass_kg += mass_kg;
        stats.transferred_energy_j += sensible_energy + latent_energy;

        if (particle < particles.temperature.size()) {
            particles.temperature[particle] = std::min(
                particles.temperature[particle] + visual_heat * 100.0f *
                    lost_fraction,
                ambient_kelvin + 1500.0f);
        }
        if (after <= 0.0f) {
            particles.removeSwap(particle);
            ++fluid_state.burned_particles;
            ++stats.particles_removed;
        }
    }

    uint64_t event_id = event_id_base;
    for (const auto& [substance, values] : exchanges) {
        MatterExchangeRecord record;
        record.event_id = event_id++;
        record.kind = MatterExchangeKind::Vaporization;
        record.source = fluid_domain.name;
        record.target = gas_domain.name;
        record.substance = substance;
        record.source_mass_kg = values.mass_kg;
        record.target_mass_kg = values.mass_kg;
        record.source_energy_j = values.sensible_energy_j;
        record.target_energy_j = values.sensible_energy_j;
        record.latent_required_j = values.latent_energy_j;
        record.latent_accounted_j = values.latent_energy_j;
        record.source_momentum_kg_m_s = values.momentum_kg_m_s;
        record.target_momentum_kg_m_s = values.momentum_kg_m_s;
        ledger.record(std::move(record));
    }
    if (stats.particles_changed > 0) ++fluid_state.version;
    return stats;
}

} // namespace Fluid
} // namespace RayTrophiSim
