#include "Fluid/FluidMistPhaseExchange.h"
#include "Fluid/FluidDomainSubstance.h"
#include "Fluid/GasInventoryGpu.h"
#include "Fluid/GasSurfaceGeometry.h"
#include "Fluid/GasSurfaceExchangePlan.h"
#include "Fluid/GasPhaseMomentum.h"

#include "Fluid/FluidParticleLabels.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/SubstanceTag.h"
#include "GridFluidSolver.h"
#include "MaterialStateField.h"
#include "MatterExchangeLedger.h"
#include "ParticleSimulation.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/MatterConstitutive.h"
#include "Fluid/FluidFoam.h"
#include "Fluid/FluidThermalLiquid.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <stdexcept>
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
        particles, Fluid::domainSubstance(fluid_domain.fluid_params),
        liquid_voxel, fluid_domain.fluid_params.particles_per_cell,
        fluid_domain.fluid_params.granular_enabled);

    FluidSim::FluidGrid& gas = gasGrid(gas_state);
    const std::size_t cell_count = gas.getCellCount();
    if (gas.density.size() != cell_count || gas.temperature.size() != cell_count ||
        gas.fuel.size() != cell_count || !(gas.voxel_size > 1.0e-6f)) {
        return stats;
    }
    prepareGasInventoryHostProducer(gas_state);
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
            tag, Fluid::domainSubstance(fluid_domain.fluid_params));
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
        // Mist is liquid water carried into the air: it joins the fog-water
        // pool with its sensible heat only. It becomes vapour later, in the
        // saturation adjustment, and pays its latent heat from the air there.
        // ★ Before 2026-10-11 it went straight to vapour with L*m of energy
        // that nothing paid for, and the ledger called it accounted.
        (void)latent_heat;
        const double latent_energy = 0.0;
        const float tracer = mass_kg /
            std::max(density * fluid_cell_volume, 1.0e-12f);
        const std::size_t cell = gas.cellIndex(gx, gy, gz);

        gas.density[cell] += tracer;
        gas.fuel[cell] += tracer * combustible;
        gas.temperature[cell] += tracer * std::clamp(
            (particle_kelvin - ambient_kelvin) / 1500.0f, 0.0f, 1.0f);
        depositGasCondensate(gas_state, cell, mass_kg,
                             static_cast<float>(sensible_energy));
        queueGasPhaseMomentum(gas_state, cell, mass_kg, particles.velocity[particle] * mass_kg);

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

bool hasBoilingParcel(const SimulationGridDomainDesc& fluid_domain,
                      const SimulationGridDomainState& fluid_state) {
    const FluidParticles& particles = fluid_state.particles;
    const std::size_t n = std::min(particles.size(), particles.temperature.size());
    const SubstanceProfile* fallback = Fluid::domainSubstance(fluid_domain.fluid_params);
    uint32_t cached_tag = ~0u;
    float cached_boil = 0.0f;
    for (std::size_t p = 0; p < n; ++p) {
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        if (tag != cached_tag) {
            const SubstanceProfile* profile = resolveFluidSubstanceProfile(tag, fallback);
            cached_tag = tag;
            cached_boil = profile && profile->latent_heat_vaporization > 0.0f
                ? profile->boiling_kelvin : 0.0f;
        }
        if (cached_boil > 0.0f && particles.temperature[p] > cached_boil) {
            return true;
        }
    }
    return false;
}

FluidBoilingExchangeStats transferBoilingToGas(
    const SimulationGridDomainDesc& fluid_domain,
    SimulationGridDomainState& fluid_state,
    const SimulationGridDomainDesc& gas_domain,
    SimulationGridDomainState& gas_state,
    const MaterialTemperatureScale& scale,
    float dt,
    uint64_t event_id_base,
    MatterExchangeLedger& ledger) {
    FluidBoilingExchangeStats stats;
    // Sparse nucleation: a boiling parcel seeds a bubble with probability
    // void_fraction * kBubbleChance per step (hash of parcel and step), so
    // bubbles mark where it boils without filling the surface volume's foam
    // channel - every parcel every step turned the whole body milky white.
    constexpr std::size_t kMaxBubblesPerStep = 512;
    constexpr float kBubbleVoidFraction = 0.2f;
    constexpr float kBubbleChance = 0.05f;
    constexpr float kBubbleLifeSeconds = 0.75f;
    const Vec3 gravity = fluid_domain.fluid_params.gravity;
    if (!fluid_domain.enabled || !gas_domain.enabled ||
        !simulationDomainHasLiquid(fluid_domain.type) ||
        !simulationDomainHasGas(gas_domain.type) ||
        !fluid_state.valid || !gas_state.valid ||
        fluid_state.particles.empty() ||
        !gridsOverlap(liquidGrid(fluid_state), gasGrid(gas_state))) {
        return stats;
    }

    FluidParticles& particles = fluid_state.particles;
    const std::size_t n = particles.size();
    if (particles.temperature.size() < n) {
        return stats;   // no thermal state: nothing can be above boiling
    }
    const float liquid_voxel = liquidGrid(fluid_state).voxel_size;
    ensureFluidParticleRestMasses(
        particles, Fluid::domainSubstance(fluid_domain.fluid_params),
        liquid_voxel, fluid_domain.fluid_params.particles_per_cell,
        fluid_domain.fluid_params.granular_enabled);
    if (particles.mass_fraction.size() < particles.size()) {
        particles.mass_fraction.resize(particles.size(), 1.0f);
    }

    FluidSim::FluidGrid& gas = gasGrid(gas_state);
    const std::size_t cell_count = gas.getCellCount();
    if (gas.density.size() != cell_count || gas.temperature.size() != cell_count ||
        !(gas.voxel_size > 1.0e-6f)) {
        return stats;
    }
    prepareGasInventoryHostProducer(gas_state);
    stats.ran = true;

    constexpr float kGasConstant = 8.314462f;     // J/(mol K)
    constexpr float kAtmosphere = 101325.0f;      // Pa
    constexpr float kAirMolarMass = 0.028965f;    // kg/mol
    const float inv_gas_voxel = 1.0f / gas.voxel_size;
    const float gas_cell_volume = std::max(
        gas.voxel_size * gas.voxel_size * gas.voxel_size, 1.0e-12f);
    std::map<std::string, ExchangeAccumulation> exchanges;

    GasSurfaceGeometry surface(fluid_state, gas,
        gas_domain.boundary_mode == SimulationGridDomainBoundaryMode::Periodic,
        &fluid_domain.fluid_params);

    for (std::size_t particle = particles.size(); particle-- > 0;) {
        if (particle < particles.flags.size() &&
            (particles.flags[particle] & kParticleFlagFrozen) != 0u) {
            continue;
        }
        if (particle < particles.constitutive_model.size() &&
            particles.constitutive_model[particle] !=
                static_cast<uint8_t>(MatterConstitutiveModel::Fluid) &&
            particles.constitutive_model[particle] !=
                static_cast<uint8_t>(MatterConstitutiveModel::Auto)) {
            continue;   // a solid or grain does not boil; it melts first
        }
        const uint32_t tag = particle < particles.substance_tag.size()
            ? particles.substance_tag[particle] : kSubstanceUntagged;
        const SubstanceProfile* profile = resolveFluidSubstanceProfile(
            tag, Fluid::domainSubstance(fluid_domain.fluid_params));
        if (!profile || !(profile->boiling_kelvin > 0.0f) ||
            !(profile->latent_heat_vaporization > 0.0f)) {
            continue;
        }
        const float boil = profile->boiling_kelvin;
        const float kelvin = particles.temperature[particle];
        if (!std::isfinite(kelvin) || !(kelvin > boil)) {
            continue;
        }
        const Vec3 local = (particles.position[particle] - gas.origin) * inv_gas_voxel;
        const int gx = static_cast<int>(std::floor(local.x));
        const int gy = static_cast<int>(std::floor(local.y));
        const int gz = static_cast<int>(std::floor(local.z));
        if (gx < 0 || gx >= gas.nx || gy < 0 || gy >= gas.ny ||
            gz < 0 || gz >= gas.nz) {
            continue;
        }

        const float rest = particles.rest_mass_kg[particle];
        const float fraction = std::clamp(particles.mass_fraction[particle], 0.0f, 1.0f);
        const float mass_kg = rest * fraction;
        if (!(mass_kg > 0.0f) || !std::isfinite(mass_kg)) {
            continue;
        }
        const float specific_heat = profile->specific_heat > 0.0f
            ? profile->specific_heat : 1000.0f;
        const float latent = profile->latent_heat_vaporization;
        // Superheat pays the latent heat; the parcel stays at Tb.
        const double superheat_j = static_cast<double>(mass_kg) * specific_heat *
            (kelvin - boil);
        float boiled_kg = static_cast<float>(
            std::min<double>(mass_kg, superheat_j / latent));
        if (static_cast<double>(boiled_kg) * latent > superheat_j) {
            boiled_kg = std::nextafter(boiled_kg, 0.0f);
        }
        if (!(boiled_kg > 0.0f)) {
            continue;
        }
        const double latent_j = static_cast<double>(boiled_kg) * latent;
        // All superheat must leave the source: after complete vaporization or
        // float-rounded dm, carry the surplus as vapour sensible energy.
        const double surplus_j = std::max(0.0, superheat_j - latent_j);
        const double sensible_j = static_cast<double>(boiled_kg) * specific_heat * boil + surplus_j;
        const float vapour_kelvin = static_cast<float>(
            boil + surplus_j / (static_cast<double>(boiled_kg) * specific_heat));

        // Steam tracer: the vapour's volume at 1 atm as a fraction of the gas
        // cell, never more than the cell's remaining headroom (the kg/J
        // inventory below carries the full amount regardless).
        // Unknown molar mass: neutral vapour (air), never liquid-dense.
        const float molar_mass = profile->vapor_molar_mass_kg_mol > 0.0f
            ? profile->vapor_molar_mass_kg_mol : kAirMolarMass;
        const float vapour_density = kAtmosphere * molar_mass / (kGasConstant * vapour_kelvin);
        uint32_t cell;
        if (!surface.nearestAir(static_cast<uint32_t>(gas.cellIndex(gx, gy, gz)), cell)) {
            continue; // No accessible gas cell: retain liquid mass and superheat.
        }
        const float tracer = std::min(
            boiled_kg / (vapour_density * gas_cell_volume),
            std::max(0.0f, 1.0f - gas.density[cell]));
        // Mix the cell toward the steam's virtual temperature by the share of
        // the cell the new steam fills (bounded, never overshoots).
        const float virtual_kelvin = vapour_kelvin * kAirMolarMass / molar_mass;
        const float steam_t = scale.toNormalized(virtual_kelvin);
        const float share = tracer / std::max(gas.density[cell] + tracer, 1.0e-6f);
        gas.density[cell] += tracer;
        gas.temperature[cell] += share * (steam_t - gas.temperature[cell]);
        // A flammable liquid's vapour is fuel for the gas combustion.
        if (profile->fluid_flammable && gas.fuel.size() == cell_count) {
            gas.fuel[cell] += tracer;
        }
        depositGasInventory(gas_state, cell, boiled_kg,
                            static_cast<float>(sensible_j + latent_j));
        queueGasPhaseMomentum(gas_state, cell, boiled_kg,
                               particles.velocity[particle] * boiled_kg);

        ExchangeAccumulation& exchange = exchanges[profile->name];
        exchange.mass_kg += boiled_kg;
        exchange.sensible_energy_j += sensible_j;
        exchange.latent_energy_j += latent_j;
        exchange.momentum_kg_m_s += particles.velocity[particle] * boiled_kg;

        // Vapour first fills the parcel's own volume: lift by the void fraction
        // and nucleate a whitewater bubble where the vapour is a real pocket.
        const float liquid_density = profile->liquid_density > 0.0f
            ? profile->liquid_density : 1000.0f;
        const float vapour_volume = boiled_kg / vapour_density;
        const float liquid_volume = std::max(mass_kg - boiled_kg, 0.0f) / liquid_density;
        const float void_fraction =
            vapour_volume / std::max(vapour_volume + liquid_volume, 1.0e-12f);
        if (dt > 0.0f) {
            particles.velocity[particle] -= gravity * (void_fraction * dt);
        }
        if (fluid_domain.fluid_foam_params.enabled) {
            uint32_t h = static_cast<uint32_t>(particle) * 2654435761u ^
                         static_cast<uint32_t>(event_id_base >> 32u) * 2246822519u;
            h ^= h >> 15u; h *= 2654435761u; h ^= h >> 13u;
            const float roll = static_cast<float>(h & 0xffffu) / 65535.0f;
            if (void_fraction > kBubbleVoidFraction &&
                roll < void_fraction * kBubbleChance &&
                stats.bubbles_emitted < kMaxBubblesPerStep) {
                fluid_state.foam.emit(particles.position[particle],
                                      particles.velocity[particle],
                                      kBubbleLifeSeconds, FoamType::Bubble);
                ++stats.bubbles_emitted;
            }
        }
        stats.vaporized_mass_kg += boiled_kg;
        fluid_state.thermal_stats.evaporated_kg += boiled_kg;
        stats.latent_energy_j += latent_j;
        ++stats.boiling_particles;

        const float remaining_kg = mass_kg - boiled_kg;
        if (!(remaining_kg > 1.0e-6f * rest)) {
            particles.removeSwap(particle);
            ++stats.particles_removed;
            continue;
        }
        particles.mass_fraction[particle] = remaining_kg / rest;
        particles.temperature[particle] = boil;
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
        // Paid from the parcel's superheat, which the clamp to Tb removed.
        record.latent_required_j = values.latent_energy_j;
        record.latent_accounted_j = values.latent_energy_j;
        record.source_momentum_kg_m_s = values.momentum_kg_m_s;
        record.target_momentum_kg_m_s = values.momentum_kg_m_s;
        ledger.record(std::move(record));
    }
    if (stats.boiling_particles > 0) {
        ++fluid_state.version;
    }
    return stats;
}

void exchangeLiquidGasSurface(
    const SimulationGridDomainDesc& fluid_domain,
    SimulationGridDomainState& fluid_state,
    const SimulationGridDomainDesc& gas_domain,
    SimulationGridDomainState& gas_state,
    const MaterialTemperatureScale& scale,
    float ambient_kelvin,
    float gas_ambient_normalized,
    float drying_scale,
    float dt,
    uint64_t event_id_base,
    MatterExchangeLedger& ledger) {
    if (!exchangeGasSurfaceCpu(fluid_domain, fluid_state, gas_domain, gas_state, scale,
                              ambient_kelvin, gas_ambient_normalized, drying_scale, dt,
                              event_id_base, ledger)) {
        throw std::runtime_error("CPU gas surface exchange failed");
    }
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
    gas_state.gas_phase_condensate_kg.resize(cell_count, 0.0f);
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
        const bool preserve_total = params.boundary != GridFluid::Boundary::Open;
        const std::vector<float> previous = preserve_total ? field : std::vector<float>();
        GridFluid::advectPassiveScalarField(
            gasGrid(gas_state), params, field, 0.0f, dt);
        double after = 0.0;
        for (float& value : field) {
            if (!std::isfinite(value) || value < 0.0f) {
                value = 0.0f;
            }
            after += value;
        }
        if (preserve_total && !(after > 0.0)) {
            field = previous;
            return;
        }
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
    advectInventory(gas_state.gas_phase_condensate_kg);
    return true;
}

} // namespace RayTrophiSim::Fluid
