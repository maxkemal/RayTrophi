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
    MatterExchangeLedger& ledger,
    bool heat_fields_present) {
    FluidThermalPhaseExchangeStats stats;
    FluidParticles& particles = state.particles;
    const std::size_t count = particles.size();

    ensureFluidParticleRestMasses(
        particles, Fluid::domainSubstance(params), liquidGrid(state).voxel_size,
        params.particles_per_cell, params.granular_enabled);

    std::vector<Vec3> previous_velocity(count, Vec3(0.0f));
    for (std::size_t particle = 0; particle < count && particle < particles.velocity.size();
         ++particle) {
        previous_velocity[particle] = particles.velocity[particle];
    }

    static std::vector<float> s_solid_change;
    updateThermalFreeze(particles, liquidGrid(state), params, state.thermal_stats,
                        heat_fields_present, &s_solid_change);

    // One record per substance and direction: the mass that actually changed
    // phase this step (m * dF) and the latent heat that actually moved through
    // the parcel temperatures. A parcel that crossed the solid threshold
    // without latent exchange (L = 0) still moves mass, with zero latent.
    std::map<PhaseKey, PhaseAccumulation> exchanges;
    for (std::size_t particle = 0; particle < count && particle < s_solid_change.size();
         ++particle) {
        const float changed = s_solid_change[particle];
        if (changed == 0.0f || !std::isfinite(changed)) {
            continue;
        }
        const MatterExchangeKind kind = changed > 0.0f
            ? MatterExchangeKind::Freezing : MatterExchangeKind::Melting;
        const float fraction = particle < particles.mass_fraction.size()
            ? std::clamp(particles.mass_fraction[particle], 0.0f, 1.0f) : 1.0f;
        const float parcel_kg = particle < particles.rest_mass_kg.size()
            ? particles.rest_mass_kg[particle] * fraction : 0.0f;
        const double mass_kg = static_cast<double>(parcel_kg) * std::fabs(changed);
        if (!(mass_kg > 0.0) || !std::isfinite(mass_kg)) {
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
        exchange.sensible_energy_j += mass_kg * specific_heat * kelvin;
        exchange.latent_energy_j += mass_kg * latent_heat;
        exchange.momentum_kg_m_s += previous_velocity[particle] * static_cast<float>(mass_kg);
        ++exchange.particles;

        if (kind == MatterExchangeKind::Freezing) {
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
        // Paid through the parcel temperature by updateThermalFreeze (freezing
        // warms the parcel back toward Tm, melting cools it to Tm).
        record.latent_required_j = values.latent_energy_j;
        record.latent_accounted_j = values.latent_energy_j;
        record.source_momentum_kg_m_s = values.momentum_kg_m_s;
        record.target_momentum_kg_m_s = values.momentum_kg_m_s;
        ledger.record(std::move(record));
    }
    return stats;
}

} // namespace RayTrophiSim::Fluid
