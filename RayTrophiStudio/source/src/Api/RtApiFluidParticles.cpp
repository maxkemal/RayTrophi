/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiFluidParticles.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 */

// Readouts of a domain's live state: the determinism fingerprint
// (fluid.state_digest), the hottest/coldest particles with what they are
// (fluid.particle_extremes) and the hottest / most vapour-laden gas cells
// (fluid.gas_cell_extremes). Particles are read from the host arrays; the gas
// inventory through a host image (a device-resident one is copied out).

#include "RtApiInternal.h"
#include "ParticleSimulation.h"
#include "MaterialStateField.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/MatterConstitutive.h"
#include "Fluid/SubstanceTag.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/GasInventoryGpu.h"
#include "Fluid/GasSaturation.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <string>

namespace rtapi {

namespace {

// Same name resolution as fluid.get; liquid-bearing domains only. Null state
// with success = the domain exists but has not been simulated.
Result resolveLiveLiquidState(const std::string& domain_id_or_name,
                              const RayTrophiSim::SimulationGridDomainState*& state) {
    state = nullptr;
    if (!g_ctx) return notBound();
    FluidDomainInfo info;
    Result found = getFluidDomain(domain_id_or_name, info);
    if (!found.ok) return found;
    auto& sys = g_ctx->scene.ensureParticleSimulationSystem();
    const auto& domains = sys.gridDomains();
    const auto& states = sys.gridDomainStates();
    for (std::size_t i = 0; i < domains.size(); ++i) {
        if (domains[i].name != info.name) continue;
        if (!RayTrophiSim::simulationDomainHasLiquid(domains[i].type)) break;
        if (i < states.size() && states[i].valid) state = &states[i];
        return Result::success();
    }
    return Result::fail("liquid (fluid) domain not found: " + domain_id_or_name);
}

const char* modelName(uint8_t model) {
    using RayTrophiSim::Fluid::MatterConstitutiveModel;
    switch (static_cast<MatterConstitutiveModel>(model)) {
        case MatterConstitutiveModel::Fluid: return "fluid";
        case MatterConstitutiveModel::Granular: return "granular";
        case MatterConstitutiveModel::Elastic: return "elastic";
        default: return "auto";
    }
}

FluidParticleProbe probeParticle(const RayTrophiSim::Fluid::FluidParticles& ps, std::size_t i) {
    FluidParticleProbe out;
    out.index = i;
    out.particle_id = i < ps.particle_id.size() ? ps.particle_id[i] : 0;
    out.substance_tag = i < ps.substance_tag.size()
        ? ps.substance_tag[i] : RayTrophiSim::Fluid::kSubstanceUntagged;
    const uint8_t model = i < ps.constitutive_model.size() ? ps.constitutive_model[i] : 0;
    out.model = modelName(model);
    if (const auto* profile = RayTrophiSim::tryFindSubstanceByTag(out.substance_tag)) {
        out.substance = profile->name;
        if (model == static_cast<uint8_t>(
                RayTrophiSim::Fluid::MatterConstitutiveModel::Granular)) {
            out.granular_transport = profile->granular_transport ==
                RayTrophiSim::Fluid::MatterGranularTransport::Dem ? "dem" : "mpm";
        }
    }
    out.kelvin = ps.temperature[i];
    out.mass_fraction = i < ps.mass_fraction.size() ? ps.mass_fraction[i] : 1.0f;
    out.rest_mass_kg = i < ps.rest_mass_kg.size() ? ps.rest_mass_kg[i] : 0.0f;
    if (i < ps.fusion_solid_fraction.size() && std::isfinite(ps.fusion_solid_fraction[i])) {
        out.fusion_solid_fraction = ps.fusion_solid_fraction[i];
    }
    out.pore_water_kg = i < ps.pore_water_mass_kg.size() ? ps.pore_water_mass_kg[i] : 0.0f;
    out.frozen = i < ps.flags.size() &&
        (ps.flags[i] & RayTrophiSim::Fluid::kParticleFlagFrozen) != 0u;
    if (i < ps.position.size()) {
        out.position[0] = ps.position[i].x;
        out.position[1] = ps.position[i].y;
        out.position[2] = ps.position[i].z;
    }
    if (i < ps.velocity.size()) {
        out.velocity[0] = ps.velocity[i].x;
        out.velocity[1] = ps.velocity[i].y;
        out.velocity[2] = ps.velocity[i].z;
    }
    return out;
}

} // namespace

Result getFluidStateDigest(const std::string& domain_id_or_name, FluidStateDigest& out) {
    out = FluidStateDigest{};
    const RayTrophiSim::SimulationGridDomainState* state = nullptr;
    Result found = resolveLiveLiquidState(domain_id_or_name, state);
    if (!found.ok || !state) return found;
    auto fnv = [](uint64_t h, const Vec3& v) {
        const float f[3] = { v.x, v.y, v.z };
        const unsigned char* b = reinterpret_cast<const unsigned char*>(f);
        for (std::size_t i = 0; i < sizeof(f); ++i) { h ^= b[i]; h *= 1099511628211ull; }
        return h;
    };
    const auto& ps = state->particles;
    out.live = true;
    out.particles = ps.position.size();
    out.position_hash = out.velocity_hash = 1469598103934665603ull;
    double cx = 0.0, cy = 0.0, cz = 0.0, speed = 0.0;
    for (std::size_t i = 0; i < ps.position.size(); ++i) {
        out.position_hash = fnv(out.position_hash, ps.position[i]);
        cx += ps.position[i].x; cy += ps.position[i].y; cz += ps.position[i].z;
        if (i < ps.velocity.size()) {
            out.velocity_hash = fnv(out.velocity_hash, ps.velocity[i]);
            speed += ps.velocity[i].length();
        }
    }
    if (out.particles > 0) {
        const double n = static_cast<double>(out.particles);
        out.centroid[0] = cx / n; out.centroid[1] = cy / n; out.centroid[2] = cz / n;
        out.mean_speed = speed / n;
    }
    out.whitewater_particles = state->foam.size();
    out.whitewater_position_hash = 1469598103934665603ull;
    for (const auto& p : state->foam.position)
        out.whitewater_position_hash = fnv(out.whitewater_position_hash, p);
    return Result::success();
}

Result getFluidParticleExtremes(const std::string& domain_id_or_name, uint32_t count,
                                FluidParticleExtremes& out) {
    out = FluidParticleExtremes{};
    if (count < 1 || count > 64) return Result::fail("count must be 1..64");
    const RayTrophiSim::SimulationGridDomainState* state = nullptr;
    Result found = resolveLiveLiquidState(domain_id_or_name, state);
    if (!found.ok || !state) return found;
    const auto& ps = state->particles;
    const std::size_t n = std::min(ps.size(), ps.temperature.size());
    out.live = true;
    out.particles = ps.size();
    // 0 K marks an unwritten parcel, not a cold one (FluidParticles::temperature).
    std::vector<std::size_t> written;
    written.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float t = ps.temperature[i];
        if (std::isfinite(t) && t > 0.0f) written.push_back(i);
    }
    out.measured = written.size();
    const std::size_t k = std::min<std::size_t>(count, written.size());
    if (k == 0) return Result::success();
    // Index breaks ties so the lists are deterministic.
    const auto hotter = [&](std::size_t a, std::size_t b) {
        return ps.temperature[a] != ps.temperature[b]
            ? ps.temperature[a] > ps.temperature[b] : a < b;
    };
    const auto colder = [&](std::size_t a, std::size_t b) {
        return ps.temperature[a] != ps.temperature[b]
            ? ps.temperature[a] < ps.temperature[b] : a < b;
    };
    std::vector<std::size_t> ranked = written;
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(), hotter);
    for (std::size_t r = 0; r < k; ++r) out.hottest.push_back(probeParticle(ps, ranked[r]));
    ranked = written;
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(), colder);
    for (std::size_t r = 0; r < k; ++r) out.coldest.push_back(probeParticle(ps, ranked[r]));
    return Result::success();
}

Result getFluidGasCellExtremes(const std::string& domain_id_or_name, uint32_t count,
                               FluidGasCellExtremes& out) {
    out = FluidGasCellExtremes{};
    if (count < 1 || count > 64) return Result::fail("count must be 1..64");
    const RayTrophiSim::SimulationGridDomainState* state = nullptr;
    Result found = resolveLiveLiquidState(domain_id_or_name, state);
    if (!found.ok || !state) return found;
    const auto& gas = RayTrophiSim::Fluid::gasGrid(*state);
    const std::size_t cells = static_cast<std::size_t>(std::max(0, gas.nx)) *
        std::max(0, gas.ny) * std::max(0, gas.nz);
    out.live = true;
    out.cells = cells;
    out.voxel_m = gas.voxel_size;
    if (cells == 0 || gas.temperature.size() != cells || !(gas.voxel_size > 0.0f)) {
        return Result::success();
    }
    RayTrophiSim::Fluid::GasInventoryHostImage image;
    try {
        // A device-resident inventory is copied out; the live one is untouched.
        image = RayTrophiSim::Fluid::captureGasInventoryHostImage(*state);
    } catch (const std::exception& e) {
        return Result::fail(std::string("gas inventory unreadable: ") + e.what());
    }
    out.inventory = image.mass.size() == cells;
    out.condensable_substance = state->gas_stats.condensable_substance;
    const auto* vapour_profile = RayTrophiSim::tryFindSubstance(
        out.condensable_substance.empty() ? std::string("Water") : out.condensable_substance);
    const float molar = vapour_profile && vapour_profile->vapor_molar_mass_kg_mol > 0.0f
        ? vapour_profile->vapor_molar_mass_kg_mol : 0.018015f;
    const float specific_heat = vapour_profile && vapour_profile->specific_heat > 0.0f
        ? vapour_profile->specific_heat : 4186.0f;
    // Same Tv mapping as the grain air films (FluidDomainStep): domain ambient
    // at the world's normalized ambient, kelvin_per_unit per field unit.
    auto& sys = g_ctx->scene.ensureParticleSimulationSystem();
    const auto& world = sys.worldThermal();
    float ambient = world.ambientKelvin();
    const auto& domains = sys.gridDomains();
    const auto& states = sys.gridDomainStates();
    for (std::size_t i = 0; i < domains.size() && i < states.size(); ++i) {
        if (&states[i] == state) {
            ambient = RayTrophiSim::fluidDomainAmbientKelvin(domains[i], world);
            break;
        }
    }
    const float gas_ambient = world.ambientNormalized();
    const float per_unit = world.kelvin_per_unit;
    const double volume = double(gas.voxel_size) * gas.voxel_size * gas.voxel_size;
    constexpr double kAtm = 101325.0, kR = 8.314462618;
    const auto at = [](const std::vector<float>& v, std::size_t c) {
        return c < v.size() ? v[c] : 0.0f;
    };
    std::vector<FluidGasCellProbe> rows;
    rows.reserve(cells / 8 + 16);
    for (std::size_t c = 0; c < cells; ++c) {
        const float field = gas.temperature[c];
        if (!std::isfinite(field)) continue;
        const float vapour = std::max(0.0f, at(image.mass, c));
        const bool solid = c < gas.solid.size() && gas.solid[c] != 0u;
        // Cells at the ambient with nothing in them are not worth a row.
        if (field == gas_ambient && vapour <= 0.0f && at(image.condensate, c) <= 0.0f &&
            at(image.rain, c) <= 0.0f) {
            continue;
        }
        FluidGasCellProbe p;
        p.cell = c;
        p.ijk[0] = static_cast<int>(c % gas.nx);
        p.ijk[1] = static_cast<int>((c / gas.nx) % gas.ny);
        p.ijk[2] = static_cast<int>(c / (static_cast<std::size_t>(gas.nx) * gas.ny));
        p.center[0] = gas.origin.x + (p.ijk[0] + 0.5) * gas.voxel_size;
        p.center[1] = gas.origin.y + (p.ijk[1] + 0.5) * gas.voxel_size;
        p.center[2] = gas.origin.z + (p.ijk[2] + 0.5) * gas.voxel_size;
        p.virtual_kelvin = std::max(1.0f, ambient + (field - gas_ambient) * per_unit);
        p.air_kelvin = vapour > 0.0f
            ? RayTrophiSim::Fluid::airKelvinFromVirtual(p.virtual_kelvin, vapour,
                  static_cast<float>(volume), molar, 1.0f)
            : p.virtual_kelvin;
        p.vapour_kg = vapour;
        p.vapour_kg_m3 = static_cast<float>(vapour / volume);
        p.pure_vapour_kg_m3 = static_cast<float>(kAtm * molar / (kR * std::max(1.0f, p.air_kelvin)));
        p.condensate_kg = at(image.condensate, c);
        p.rain_kg = at(image.rain, c);
        p.rain_kelvin = p.rain_kg > 0.0f
            ? at(image.rain_energy, c) / (specific_heat * p.rain_kg) : 0.0f;
        p.foreign_kg = at(image.foreign, c);
        p.energy_j = at(image.energy, c);
        p.solid = solid;
        if (!solid && p.pure_vapour_kg_m3 > 0.0f) {
            out.max_vapour_over_pure = std::max(out.max_vapour_over_pure,
                                                p.vapour_kg_m3 / p.pure_vapour_kg_m3);
        }
        rows.push_back(p);
    }
    const std::size_t k = std::min<std::size_t>(count, rows.size());
    if (k == 0) return Result::success();
    std::vector<FluidGasCellProbe> ranked = rows;
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
        [](const FluidGasCellProbe& a, const FluidGasCellProbe& b) {
            return a.air_kelvin != b.air_kelvin ? a.air_kelvin > b.air_kelvin : a.cell < b.cell;
        });
    out.hottest.assign(ranked.begin(), ranked.begin() + k);
    ranked = rows;
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
        [](const FluidGasCellProbe& a, const FluidGasCellProbe& b) {
            return a.vapour_kg_m3 != b.vapour_kg_m3 ? a.vapour_kg_m3 > b.vapour_kg_m3
                                                    : a.cell < b.cell;
        });
    out.most_vapour.assign(ranked.begin(), ranked.begin() + k);
    return Result::success();
}

} // namespace rtapi
