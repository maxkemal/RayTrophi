#include "Fluid/FluidMistPhaseExchange.h"

#include "Fluid/FluidParticleLabels.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/SubstanceTag.h"
#include "GridFluidSolver.h"
#include "MaterialStateField.h"
#include "MatterExchangeLedger.h"
#include "ParticleSimulation.h"
#include "Fluid/MatterPhaseGrid.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace RayTrophiSim::Fluid {
namespace {

struct ExchangeAccumulation {
    double mass_kg = 0.0;
    double sensible_energy_j = 0.0;
    double latent_energy_j = 0.0;
    Vec3 momentum_kg_m_s;
};


} // namespace

FluidMistPhaseExchangeStats transferMistToGas(
    const SimulationGridDomainDesc& fluid_domain,
    SimulationGridDomainState& fluid_state,
    const SimulationGridDomainDesc& gas_domain,
    SimulationGridDomainState& gas_state,
    uint64_t event_id_base,
    MatterExchangeLedger& ledger) {
    FluidMistPhaseExchangeStats stats;
    if (!fluid_domain.enabled || !gas_domain.enabled ||
        !simulationDomainHasLiquid(fluid_domain.type) ||
        !simulationDomainHasGas(gas_domain.type) ||
        !fluid_state.valid || !gas_state.valid ||
        fluid_state.particles.empty() ||
        !gridsOverlap(liquidGrid(fluid_state), gasGrid(gas_state))) {
        return stats;
    }

    FluidParticles& particles = fluid_state.particles;
    const float liquid_voxel = liquidGrid(fluid_state).voxel_size;
    ensureFluidParticleRestMasses(
        particles, fluid_domain.fluid_params.chemistry_preset,
        liquid_voxel, fluid_domain.fluid_params.particles_per_cell);

    FluidSim::FluidGrid& gas = gasGrid(gas_state);
    const std::size_t cell_count = gas.getCellCount();
    if (gas.density.size() != cell_count || gas.temperature.size() != cell_count ||
        gas.fuel.size() != cell_count || !(gas.voxel_size > 1.0e-6f)) {
        return stats;
    }
    gas_state.gas_phase_mass_kg.resize(cell_count, 0.0f);
    gas_state.gas_phase_energy_j.resize(cell_count, 0.0f);
    stats.ran = true;

    const float ambient_kelvin = std::max(
        1.0f, fluid_domain.thermal_ambient_kelvin);
    const float inv_gas_voxel = 1.0f / gas.voxel_size;
    const float fluid_cell_volume = std::max(
        liquid_voxel * liquid_voxel *
            liquid_voxel,
        1.0e-12f);
    std::map<std::string, ExchangeAccumulation> exchanges;

    for (std::size_t particle = particles.size(); particle-- > 0;) {
        if (particle >= particles.flags.size() ||
            particleLabel(particles.flags[particle]) != ParticleLabel::Mist) {
            continue;
        }

        const Vec3 local = (particles.position[particle] - gas.origin) *
            inv_gas_voxel;
        const int gx = static_cast<int>(std::floor(local.x));
        const int gy = static_cast<int>(std::floor(local.y));
        const int gz = static_cast<int>(std::floor(local.z));
        if (gx < 0 || gx >= gas.nx || gy < 0 || gy >= gas.ny ||
            gz < 0 || gz >= gas.nz) {
            continue;
        }

        const float fraction = particle < particles.mass_fraction.size()
            ? std::clamp(particles.mass_fraction[particle], 0.0f, 1.0f)
            : 1.0f;
        const float mass_kg = particles.rest_mass_kg[particle] * fraction;
        if (!(mass_kg > 0.0f) || !std::isfinite(mass_kg)) {
            continue;
        }

        const uint32_t tag = particle < particles.substance_tag.size()
            ? particles.substance_tag[particle] : kSubstanceUntagged;
        const SubstanceProfile* profile = resolveFluidSubstanceProfile(
            tag, fluid_domain.fluid_params.chemistry_preset);
        const float density = profile && profile->liquid_density > 0.0f
            ? profile->liquid_density : 1000.0f;
        const float specific_heat = profile && profile->specific_heat > 0.0f
            ? profile->specific_heat : 1000.0f;
        const float latent_heat = profile
            ? std::max(0.0f, profile->latent_heat_vaporization) : 0.0f;
        const float particle_kelvin = particle < particles.temperature.size()
            ? std::max(0.0f, particles.temperature[particle]) : ambient_kelvin;
        const float combustible = particle < particles.combustible_fraction.size()
            ? std::clamp(particles.combustible_fraction[particle], 0.0f, 1.0f)
            : (profile && profile->fluid_flammable ? 1.0f : 0.0f);
        const double sensible_energy = static_cast<double>(mass_kg) *
            static_cast<double>(specific_heat) * particle_kelvin;
        const double latent_energy = static_cast<double>(mass_kg) * latent_heat;
        const float tracer = mass_kg /
            std::max(density * fluid_cell_volume, 1.0e-12f);
        const std::size_t cell = gas.cellIndex(gx, gy, gz);

        gas.density[cell] += tracer;
        gas.fuel[cell] += tracer * combustible;
        gas.temperature[cell] += tracer * std::clamp(
            (particle_kelvin - ambient_kelvin) / 1500.0f, 0.0f, 1.0f);
        gas_state.gas_phase_mass_kg[cell] += mass_kg;
        gas_state.gas_phase_energy_j[cell] +=
            static_cast<float>(sensible_energy + latent_energy);

        const std::string substance = profile ? profile->name : "Custom";
        ExchangeAccumulation& exchange = exchanges[substance];
        exchange.mass_kg += mass_kg;
        exchange.sensible_energy_j += sensible_energy;
        exchange.latent_energy_j += latent_energy;
        exchange.momentum_kg_m_s += particles.velocity[particle] * mass_kg;
        stats.transferred_mass_kg += mass_kg;
        stats.transferred_energy_j += sensible_energy + latent_energy;
        particles.removeSwap(particle);
        ++stats.particles_removed;
    }

    uint64_t event_id = event_id_base;
    for (const auto& [substance, values] : exchanges) {
        MatterExchangeRecord record;
        record.event_id = event_id++;
        record.kind = MatterExchangeKind::MistToGas;
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
    if (stats.particles_removed > 0) {
        ++fluid_state.version;
    }
    return stats;
}

bool advectGasPhaseInventory(
    SimulationGridDomainState& gas_state,
    const GridFluid::SolverParams& params,
    float dt) {
    if (!gas_state.valid || !simulationDomainHasGas(gas_state.type) ||
        !(dt > 0.0f) || !std::isfinite(dt)) {
        return false;
    }
    const std::size_t cell_count = gasGrid(gas_state).getCellCount();
    if (gas_state.gas_phase_mass_kg.size() != cell_count ||
        gas_state.gas_phase_energy_j.size() != cell_count) {
        return false;
    }
    auto advectInventory = [&](std::vector<float>& field) {
        double before = 0.0;
        for (float& value : field) {
            if (!std::isfinite(value) || value <= 0.0f) {
                value = 0.0f;
            } else {
                before += static_cast<double>(value);
            }
        }
        if (!(before > 0.0)) return;
        GridFluid::advectPassiveScalarField(
            gasGrid(gas_state), params, field, 0.0f, dt);
        double after = 0.0;
        for (float& value : field) {
            if (!std::isfinite(value) || value < 0.0f) {
                value = 0.0f;
            }
            after += value;
        }
        const bool preserve_total =
            params.boundary != GridFluid::Boundary::Open;
        if (after > 0.0 && before > 0.0 &&
            (preserve_total || after > before)) {
            const float scale = static_cast<float>(before / after);
            for (float& value : field) {
                value *= scale;
            }
        }
    };
    advectInventory(gas_state.gas_phase_mass_kg);
    advectInventory(gas_state.gas_phase_energy_j);
    return true;
}

} // namespace RayTrophiSim::Fluid
