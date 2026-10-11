#include "Fluid/GasSurfaceExchangePlan.h"
#include "Fluid/GasSurfaceGeometry.h"
#include "Fluid/GasInventoryGpu.h"
#include "Fluid/GasPhaseMomentum.h"
#include "Fluid/GasSaturation.h"
#include "Fluid/FluidDomainSubstance.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/FluidThermalLiquid.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/SubstanceTag.h"
#include "Fluid/MatterSubstanceState.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace RayTrophiSim::Fluid {
namespace {
// Condensate first refills what the parcel lost (mass_fraction back to 1),
// then raises its physical kg (rest_mass_kg). Returns true when it grew past 1.
bool addCondensedMass(FluidParticles& particles, uint32_t p, double gained_kg) {
    if (!(gained_kg > 0.0) || p >= particles.rest_mass_kg.size() ||
        !(particles.rest_mass_kg[p] > 0.0f)) {
        return false;
    }
    float& fraction = particles.mass_fraction[p];
    const double rest = particles.rest_mass_kg[p];
    const double refill = std::min(gained_kg, rest * std::max(0.0f, 1.0f - fraction));
    fraction = std::min(1.0f, fraction + static_cast<float>(refill / rest));
    const double extra = gained_kg - refill;
    if (!(extra > 0.0)) {
        return false;
    }
    particles.rest_mass_kg[p] = static_cast<float>(rest * fraction + extra);
    fraction = 1.0f;
    return true;
}

// Splits a parcel heavier than twice `canonical_kg` into two equal parcels a
// quarter voxel apart along the vertical. Both carry every sidecar of the
// original (temperature, substance, uvw, fusion state); identity is new for
// the second, as for any birth.
bool splitHeavyParcel(FluidParticles& particles, uint32_t p, float canonical_kg, float voxel) {
    if (!(canonical_kg > 0.0f) || p >= particles.size() ||
        !(particles.rest_mass_kg[p] > 2.0f * canonical_kg)) {
        return false;
    }
    const float half = 0.5f * particles.rest_mass_kg[p];
    particles.rest_mass_kg[p] = half;
    const Vec3 offset(0.0f, 0.25f * voxel, 0.0f);
    const Vec3 position = particles.position[p];
    const Vec3 uvw = p < particles.uvw.size() ? particles.uvw[p] : position;
    const Vec3 uvw_b = p < particles.uvw_b.size() ? particles.uvw_b[p] : uvw;
    const auto model = p < particles.constitutive_model.size()
        ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[p])
        : MatterConstitutiveModel::Fluid;
    particles.emit(position - offset, particles.velocity[p], particles.temperature[p],
        p < particles.combustible_fraction.size() ? particles.combustible_fraction[p] : 0.0f,
        p < particles.substance_tag.size() ? particles.substance_tag[p] : kSubstanceUntagged,
        &uvw, &uvw_b, half, model);
    particles.position[p] = position + offset;
    const std::size_t born = particles.size() - 1;
    if (p < particles.affine.size()) particles.affine[born] = particles.affine[p];
    if (born < particles.fusion_solid_fraction.size() && p < particles.fusion_solid_fraction.size()) {
        particles.fusion_solid_fraction[born] = particles.fusion_solid_fraction[p];
    }
    return true;
}
} // namespace

static_assert(sizeof(GasSurfaceParcel) == 64 && offsetof(GasSurfaceParcel, capacity) == 40);
static_assert(offsetof(GasSurfaceParcel, velocity) == 48);
static_assert(alignof(GasSurfaceParcel) <= alignof(std::max_align_t));
static_assert(sizeof(GasSurfaceGroup) == 16);
static_assert(sizeof(GasSurfaceResult) == 64 && offsetof(GasSurfaceResult, kelvin) == 32);
static_assert(alignof(GasSurfaceResult) <= alignof(std::max_align_t));

GasSurfaceExchangePlan buildGasSurfaceExchangePlan(
    const SimulationGridDomainDesc& liquid_domain, SimulationGridDomainState& liquid,
    const SimulationGridDomainDesc& gas_domain, const SimulationGridDomainState& gas) {
    GasSurfaceExchangePlan plan;
    auto& particles = liquid.particles;
    const auto& grid = gasGrid(gas);
    if (particles.temperature.size() < particles.size() ||
        particles.velocity.size() < particles.size()) {
        plan.error = "gas surface source temperature or velocity array is incomplete";
        return plan;
    }
    const auto* fallback = domainSubstance(liquid_domain.fluid_params);
    const float voxel = liquidGrid(liquid).voxel_size;
    ensureFluidParticleRestMasses(particles, fallback, voxel,
                                 liquid_domain.fluid_params.particles_per_cell,
                                 liquid_domain.fluid_params.granular_enabled);
    particles.mass_fraction.resize(particles.size(), 1.0f);
    const CondensableVapour vapour = resolveCondensableVapour(gas_domain);
    const std::string condensable_name = vapour.valid() ? vapour.substance : std::string();
    GasSurfaceGeometry geometry(liquid, grid,
        gas_domain.boundary_mode == SimulationGridDomainBoundaryMode::Periodic,
        &liquid_domain.fluid_params);
    plan.records.reserve(particles.size());
    const double area = static_cast<double>(grid.voxel_size) * grid.voxel_size;
    for (std::size_t p = 0; p < particles.size(); ++p) {
        uint32_t cell;
        if (!(particles.rest_mass_kg[p] > 0.0f) ||
            !std::isfinite(particles.rest_mass_kg[p]) || !geometry.particleCell(p, cell) ||
            (p < particles.flags.size() && (particles.flags[p] & kParticleFlagFrozen)) ||
            !(particles.temperature[p] > 0.0f) || !std::isfinite(particles.temperature[p])) {
            continue;
        }
        auto model = p < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[p])
            : MatterConstitutiveModel::Auto;
        if (model == MatterConstitutiveModel::Auto) {
            const uint32_t tag = p < particles.substance_tag.size() ? particles.substance_tag[p] : 0u;
            model = substanceBirthModel(tag, liquid_domain.fluid_params, particles.temperature[p]);
        }
        if (model != MatterConstitutiveModel::Fluid) {
            continue;
        }
        const auto faces = geometry.exposedFaces(cell);
        if (!faces.count) {
            continue;
        }
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        const auto* profile = resolveFluidSubstanceProfile(tag, fallback);
        if (!profile) {
            continue;
        }
        // Disjoint thermal/mass shares, so six air faces cannot pay latent
        // heat six times or race while modifying one canonical particle.
        const float split = 1.0f / faces.count;
        const double capacity = parcelHeatCapacity(
            particles, p, *profile, voxel, liquid_domain.fluid_params.particles_per_cell);
        const Vec3 v = particles.velocity[p];
        if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
            plan.error = "non-finite gas surface source velocity";
            return plan;
        }
        // Evaporation and condensation only for the air's condensable vapour:
        // the vapour pool is that one species. Another liquid exchanges heat
        // and takes rain but does not evaporate here (a combustible one
        // vaporizes through the combustion exchange, into the
        // non-condensable pool). ★ Before 2026-10-11 part 4 any substance with
        // vapour data evaporated INTO the water vapour pool.
        const bool condensable = condensable_name == profile->name;
        for (uint32_t f = 0; f < faces.count; ++f) {
            plan.records.push_back({
                static_cast<uint32_t>(p), faces.air[f], particles.temperature[p],
                particles.rest_mass_kg[p] * split, particles.mass_fraction[p],
                profile->specific_heat > 0.0f ? profile->specific_heat : 1000.0f,
                condensable ? profile->boiling_kelvin : 0.0f,
                condensable ? profile->latent_heat_vaporization : 0.0f,
                condensable ? profile->vapor_molar_mass_kg_mol : 0.0f,
                static_cast<float>(area / std::max(1u, geometry.parcelsInCell(cell))),
                capacity * split, {v.x, v.y, v.z, 0.0f}
            });
        }
        plan.substances[static_cast<uint32_t>(p)] = profile->name;
    }
    // Flat records avoid one allocation/copy per air cell. A stable order is
    // shared by CPU and GPU, so each air cell has exactly one commit writer.
    std::stable_sort(plan.records.begin(), plan.records.end(),
        [](const GasSurfaceParcel& a, const GasSurfaceParcel& b) {
            return a.air_cell < b.air_cell;
        });
    for (std::size_t first = 0; first < plan.records.size();) {
        std::size_t last = first + 1;
        while (last < plan.records.size() &&
               plan.records[last].air_cell == plan.records[first].air_cell) {
            ++last;
        }
        plan.groups.push_back({static_cast<uint32_t>(first),
            static_cast<uint32_t>(last - first), plan.records[first].air_cell});
        first = last;
    }
    return plan;
}

bool validateGasSurfaceResults(const GasSurfaceExchangePlan& plan,
                               const std::vector<GasSurfaceResult>& results) {
    if (results.size() != plan.records.size()) {
        return false;
    }
    for (std::size_t r = 0; r < results.size(); ++r) {
        const auto& value = results[r];
        const auto& input = plan.records[r];
        const double budget = static_cast<double>(input.rest_mass) * input.fraction;
        // Negative mass is condensation onto the parcel: at most doubling it.
        if (value.error != 0.0f || !(value.kelvin > 0.0f) ||
            !std::isfinite(value.kelvin) || !std::isfinite(value.fraction) ||
            value.fraction < 0.0f || value.fraction > input.fraction + 1.0e-5f ||
            !std::isfinite(value.mass) || value.mass < -(budget + 1.0e-8) ||
            value.mass > budget + 1.0e-8 ||
            !std::isfinite(value.sensible) || !std::isfinite(value.latent) ||
            !std::isfinite(value.heat_to_gas) ||
            !std::isfinite(value.rain_kg) || value.rain_kg < 0.0f ||
            !std::isfinite(value.rain_energy_j) || value.rain_energy_j < 0.0f) {
            return false;
        }
    }
    return true;
}

void publishGasSurfaceResults(const GasSurfaceExchangePlan& plan,
                              const std::vector<GasSurfaceResult>& results,
                              const SimulationGridDomainDesc& liquid_domain,
                              SimulationGridDomainState& liquid,
                              const SimulationGridDomainDesc& gas_domain,
                              SimulationGridDomainState& gas,
                              uint64_t event_id, MatterExchangeLedger& ledger,
                              bool inventory_committed) {
    struct Mix { double capacity = 0, thermal = 0, evaporated = 0; bool touched = false; };
    struct Exchange { double mass = 0, sensible = 0, latent = 0; Vec3 momentum; };
    std::map<uint32_t, Mix> mixes;
    std::map<std::string, Exchange> exchanges;
    std::map<std::string, Exchange> condensations;
    std::map<std::string, Exchange> rains;
    double heat = 0, evaporated = 0, condensed = 0, rained = 0;
    for (std::size_t r = 0; r < results.size(); ++r) {
        const auto& input = plan.records[r];
        const auto& value = results[r];
        auto& mix = mixes[input.particle];
        const double rain = value.rain_kg;
        const double capacity = std::max(0.0,
            input.capacity - (value.mass - rain) * input.specific_heat);
        mix.capacity += capacity;
        mix.thermal += capacity * value.kelvin;
        // Net mass leaving the parcel: evaporation minus condensation and rain.
        mix.evaporated += value.mass - rain;
        if (rain > 0.0) {
            const Vec3 momentum = Vec3(input.velocity[0], input.velocity[1], input.velocity[2]) *
                static_cast<float>(rain);
            auto& exchange = rains[plan.substances.at(input.particle)];
            exchange.mass += rain;
            exchange.sensible += value.rain_energy_j;
            exchange.momentum += momentum;
            rained += rain;
        }
        mix.touched = mix.touched || value.exchanged > 0;
        heat += value.heat_to_gas;
        if (value.mass > 0) {
            const Vec3 momentum = Vec3(input.velocity[0], input.velocity[1], input.velocity[2]) *
                static_cast<float>(value.mass);
            if (inventory_committed) {
                queueGasPhaseMomentum(gas, input.air_cell, static_cast<float>(value.mass), momentum);
            } else {
                depositGasInventory(gas, input.air_cell, static_cast<float>(value.mass),
                    static_cast<float>(value.sensible + value.latent));
                queueGasPhaseMomentum(gas, input.air_cell, static_cast<float>(value.mass), momentum);
            }
            auto& exchange = exchanges[plan.substances.at(input.particle)];
            exchange.mass += value.mass;
            exchange.sensible += value.sensible;
            exchange.latent += value.latent;
            exchange.momentum += momentum;
            evaporated += value.mass;
        } else if (value.mass < 0) {
            // The vapour joins the parcel at the parcel's velocity; its own
            // (tiny) gas momentum is left in the carrier.
            const double gained = -value.mass;
            const Vec3 momentum = Vec3(input.velocity[0], input.velocity[1], input.velocity[2]) *
                static_cast<float>(gained);
            auto& exchange = condensations[plan.substances.at(input.particle)];
            exchange.mass += gained;
            exchange.sensible += value.sensible;
            exchange.latent += value.latent;
            exchange.momentum += momentum;
            condensed += gained;
        }
    }
    std::size_t touched = 0;
    std::vector<uint32_t> grown;
    for (const auto& entry : mixes) {
        const uint32_t p = entry.first;
        const auto& mix = entry.second;
        if (mix.capacity > 0) {
            liquid.particles.temperature[p] = static_cast<float>(mix.thermal / mix.capacity);
        }
        if (mix.evaporated >= 0.0) {
            liquid.particles.mass_fraction[p] = std::max(0.0f,
                liquid.particles.mass_fraction[p] -
                    static_cast<float>(mix.evaporated / liquid.particles.rest_mass_kg[p]));
        } else if (addCondensedMass(liquid.particles, p, -mix.evaporated)) {
            grown.push_back(p);
        }
        touched += mix.touched;
    }
    // A parcel that condensation has grown past twice the substance's canonical
    // parcel mass splits in two: mass_fraction never exceeds 1 (every solver
    // lane relies on it), and an ever-heavier parcel would carry kg its APIC
    // weight does not know about.
    const float voxel = liquidGrid(liquid).voxel_size;
    const auto* fallback = domainSubstance(liquid_domain.fluid_params);
    std::size_t split = 0;
    for (const uint32_t p : grown) {
        const uint32_t tag = p < liquid.particles.substance_tag.size()
            ? liquid.particles.substance_tag[p] : kSubstanceUntagged;
        const float canonical = fluidParticleRestMassKg(tag, fallback, voxel,
            liquid_domain.fluid_params.particles_per_cell, MatterConstitutiveModel::Fluid);
        split += splitHeavyParcel(liquid.particles, p, canonical, voxel) ? 1u : 0u;
    }
    for (const auto& entry : condensations) {
        MatterExchangeRecord record;
        record.event_id = event_id++;
        record.kind = MatterExchangeKind::Condensation;
        record.source = gas_domain.name;
        record.target = liquid_domain.name;
        record.substance = entry.first;
        record.source_mass_kg = record.target_mass_kg = entry.second.mass;
        record.source_energy_j = record.target_energy_j = entry.second.sensible;
        // Released into the parcel: it is part of the vapour energy handed over.
        record.latent_required_j = record.latent_accounted_j = entry.second.latent;
        record.source_momentum_kg_m_s = record.target_momentum_kg_m_s = entry.second.momentum;
        ledger.record(std::move(record));
    }
    // Rain water that landed on the liquid (sim_inventory_rain_fall stopped it
    // in this air cell): it joins at its own temperature, no phase change.
    for (const auto& entry : rains) {
        MatterExchangeRecord record;
        record.event_id = event_id++;
        record.kind = MatterExchangeKind::Precipitation;
        record.source = gas_domain.name + ":rain";
        record.target = liquid_domain.name;
        record.substance = entry.first;
        record.source_mass_kg = record.target_mass_kg = entry.second.mass;
        record.source_energy_j = record.target_energy_j = entry.second.sensible;
        record.source_momentum_kg_m_s = record.target_momentum_kg_m_s = entry.second.momentum;
        ledger.record(std::move(record));
    }
    gas.gas_rain_absorbed_pending_kg += rained;
    liquid.thermal_stats.condensed_kg += condensed;
    liquid.thermal_stats.condensation_splits += split;
    for (const auto& entry : exchanges) {
        MatterExchangeRecord record;
        record.event_id = event_id++;
        record.kind = MatterExchangeKind::Vaporization;
        record.source = liquid_domain.name;
        record.target = gas_domain.name;
        record.substance = entry.first;
        record.source_mass_kg = record.target_mass_kg = entry.second.mass;
        record.source_energy_j = record.target_energy_j = entry.second.sensible;
        record.latent_required_j = record.latent_accounted_j = entry.second.latent;
        record.source_momentum_kg_m_s = record.target_momentum_kg_m_s = entry.second.momentum;
        ledger.record(std::move(record));
    }
    liquid.thermal_stats.gas_exchange_particles += touched;
    liquid.thermal_stats.heat_to_gas_j += heat;
    liquid.thermal_stats.evaporated_kg += evaporated;
    if (touched || evaporated > 0 || condensed > 0 || rained > 0) {
        ++liquid.version;
    }
}
}
