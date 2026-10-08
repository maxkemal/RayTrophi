#include "Fluid/FluidThermalPhaseExchange.h"
#include "Fluid/FluidDomainSubstance.h"

#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/SubstanceTag.h"
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

struct PhaseAccumulation {
    double mass_kg = 0.0;
    double sensible_energy_j = 0.0;
    double latent_energy_j = 0.0;
    Vec3 momentum_kg_m_s;
    std::size_t particles = 0;
};

struct PhaseKey {
    MatterExchangeKind kind = MatterExchangeKind::Freezing;
    std::string substance;

    bool operator<(const PhaseKey& other) const {
        if (kind != other.kind) {
            return static_cast<uint8_t>(kind) < static_cast<uint8_t>(other.kind);
        }
        return substance < other.substance;
    }
};

} // namespace

FluidThermalPhaseExchangeStats updateThermalFreezeAndRecord(
    const SimulationGridDomainDesc& domain,
    SimulationGridDomainState& state,
    const APICSolverParams& params,
    uint64_t event_id_base,
    MatterExchangeLedger& ledger) {
    FluidThermalPhaseExchangeStats stats;
    FluidParticles& particles = state.particles;
    const std::size_t count = particles.size();

    ensureFluidParticleRestMasses(
        particles, Fluid::domainSubstance(params), liquidGrid(state).voxel_size,
        params.particles_per_cell, params.granular_enabled);

    std::vector<uint8_t> was_frozen(count, 0u);
    std::vector<Vec3> previous_velocity(count, Vec3(0.0f));
    for (std::size_t particle = 0; particle < count; ++particle) {
        was_frozen[particle] = isFrozenParticle(particles, particle) ? 1u : 0u;
        if (particle < particles.velocity.size()) {
            previous_velocity[particle] = particles.velocity[particle];
        }
    }

    updateThermalFreeze(particles, liquidGrid(state), params, state.thermal_stats);

    std::map<PhaseKey, PhaseAccumulation> exchanges;
    for (std::size_t particle = 0; particle < count; ++particle) {
        const bool frozen = isFrozenParticle(particles, particle);
        if (frozen == (was_frozen[particle] != 0u)) {
            continue;
        }

        const MatterExchangeKind kind = frozen
            ? MatterExchangeKind::Freezing : MatterExchangeKind::Melting;
        const float fraction = particle < particles.mass_fraction.size()
            ? std::clamp(particles.mass_fraction[particle], 0.0f, 1.0f) : 1.0f;
        const float mass_kg = particle < particles.rest_mass_kg.size()
            ? particles.rest_mass_kg[particle] * fraction : 0.0f;
        if (!(mass_kg > 0.0f) || !std::isfinite(mass_kg)) {
            continue;
        }

        const uint32_t tag = particle < particles.substance_tag.size()
            ? particles.substance_tag[particle] : kSubstanceUntagged;
        const SubstanceProfile* profile = resolveFluidSubstanceProfile(
            tag, Fluid::domainSubstance(params));
        const float specific_heat = profile && profile->specific_heat > 0.0f
            ? profile->specific_heat : 1000.0f;
        const float latent_heat = profile
            ? std::max(0.0f, profile->latent_heat_fusion) : 0.0f;
        const float kelvin = particle < particles.temperature.size() &&
            std::isfinite(particles.temperature[particle])
            ? std::max(0.0f, particles.temperature[particle])
            : std::max(1.0f, domain.thermal_ambient_kelvin);
        const std::string substance = profile ? profile->name : "Custom";

        PhaseAccumulation& exchange = exchanges[{kind, substance}];
        exchange.mass_kg += mass_kg;
        exchange.sensible_energy_j += static_cast<double>(mass_kg) *
            static_cast<double>(specific_heat) * kelvin;
        exchange.latent_energy_j += static_cast<double>(mass_kg) * latent_heat;
        exchange.momentum_kg_m_s += previous_velocity[particle] * mass_kg;
        ++exchange.particles;

        if (frozen) {
            ++stats.frozen_particles;
            stats.frozen_mass_kg += mass_kg;
        } else {
            ++stats.melted_particles;
            stats.melted_mass_kg += mass_kg;
        }
    }

    uint64_t event_id = event_id_base;
    for (const auto& [key, values] : exchanges) {
        MatterExchangeRecord record;
        record.event_id = event_id++;
        record.kind = key.kind;
        record.source = domain.name +
            (key.kind == MatterExchangeKind::Freezing ? ":liquid" : ":solid");
        record.target = domain.name +
            (key.kind == MatterExchangeKind::Freezing ? ":solid" : ":liquid");
        record.substance = key.substance;
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
    return stats;
}

} // namespace RayTrophiSim::Fluid
