#include "Fluid/GasSurfaceExchangePlan.h"
#include "Fluid/GasInventoryGpu.h"
#include "Fluid/MatterPhaseGrid.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace RayTrophiSim::Fluid {
bool exchangeGasSurfaceCpu(
    const SimulationGridDomainDesc& liquid_domain, SimulationGridDomainState& liquid,
    const SimulationGridDomainDesc& gas_domain, SimulationGridDomainState& gas,
    const MaterialTemperatureScale& scale, float ambient_kelvin,
    float gas_ambient_normalized, float drying_scale, float dt,
    uint64_t event_id, MatterExchangeLedger& ledger) {
    if (gasInventoryDevice(gas)) {
        return false;
    }
    if (!(dt > 0) || !liquid_domain.enabled || !gas_domain.enabled ||
        !simulationDomainHasLiquid(liquid_domain.type) ||
        !simulationDomainHasGas(gas_domain.type) || !liquid.valid || !gas.valid ||
        liquid.particles.empty() || !gridsOverlap(liquidGrid(liquid), gasGrid(gas))) {
        return true;
    }
    auto& grid = gasGrid(gas);
    if (!std::isfinite(dt) || !(scale.kelvin_per_unit > 0) || !(grid.voxel_size > 0) ||
        grid.temperature.size() != grid.getCellCount() || grid.density.size() != grid.getCellCount()) {
        return false;
    }
    const auto plan = buildGasSurfaceExchangePlan(liquid_domain, liquid, gas_domain, gas);
    if (!plan.error.empty()) {
        return false;
    }
    if (plan.records.empty()) {
        return true;
    }
    prepareGasInventoryHostProducer(gas);
    const double volume = static_cast<double>(grid.voxel_size) * grid.voxel_size * grid.voxel_size;
    const double humidity = std::clamp(1.0 - static_cast<double>(drying_scale), 0.0, 1.0);
    constexpr double atmosphere = 101325.0, gas_constant = 8.314462, air_molar = 0.028965;
    constexpr double pi = 3.14159265358979;
    // Effective Hertz-Knudsen coefficient. 4e-5 is the open-pan value: in air
    // the vapour must DIFFUSE through the air layer, and that resistance, not
    // the interface, sets the rate. It scales with the air's share of the
    // cell; in nearly pure steam the interface coefficient (0.04) is the limit,
    // so steam on cold water condenses hundreds of times faster than a pan
    // evaporates (2026-10-11: 68 kg boiled, 0.010 kg condensed with 4e-5).
    constexpr double kOpenPanCoefficient = 4.0e-5, kInterfaceCoefficient = 0.04;
    const auto coefficient = [&](double vapour_pressure) {
        const double air_share = std::max(1.0 - vapour_pressure / atmosphere, 1.0e-3);
        return std::min(kInterfaceCoefficient, kOpenPanCoefficient / air_share);
    };
    std::vector<GasSurfaceResult> results(plan.records.size());
    std::vector<std::array<float, 2>> columns;
    columns.reserve(plan.groups.size());
    // Vapour each air cell gave back to the liquid; committed to the host
    // inventory only after every result validated.
    std::vector<std::array<double, 2>> condensed(plan.groups.size(), {0.0, 0.0});
    // Rain landing (sim_inventory_surface_exchange): the air cell above a
    // surface parcel stops the rain fall, and its rain joins the parcels by
    // surface share. Absorbed cells are emptied after validation.
    gas.gas_rain_landing.resize(grid.getCellCount(), 0u);
    std::vector<uint8_t> rain_absorbed(plan.groups.size(), 0u);
    for (std::size_t g = 0; g < plan.groups.size(); ++g) {
        const auto& group = plan.groups[g];
        float m = gas.gas_phase_mass_kg[group.cell];
        float e = gas.gas_phase_energy_j[group.cell];
        float temperature = grid.temperature[group.cell];
        float density = grid.density[group.cell];
        gas.gas_rain_landing[group.cell] = gas.gas_rain_stamp;
        const double rain_mass = std::max(0.0f, gas.gas_phase_rain_kg[group.cell]);
        const double rain_heat = std::max(0.0f, gas.gas_phase_rain_energy_j[group.cell]);
        double share_sum = 0.0;
        for (uint32_t r = group.first; r < group.first + group.count; ++r) {
            const auto& input = plan.records[r];
            if (std::isfinite(input.kelvin) && input.kelvin > 0.0f && input.capacity > 0.0 &&
                input.rest_mass > 0.0f) {
                share_sum += std::max(0.0f, input.share);
            }
        }
        rain_absorbed[g] = rain_mass > 0.0 && share_sum > 0.0;
        for (uint32_t r = group.first; r < group.first + group.count; ++r) {
            const auto& input = plan.records[r];
            auto& value = results[r];
            value = {};
            value.kelvin = input.kelvin;
            value.fraction = input.fraction;
            double tp = input.kelvin;
            const double ta = std::max(1.0, static_cast<double>(ambient_kelvin) +
                (temperature - gas_ambient_normalized) * scale.kelvin_per_unit);
            const double cp = input.capacity;
            const double ca = atmosphere * air_molar / (gas_constant * ta) * volume * 1005.0;
            if (cp > 0 && ca > 0) {
                const double decay = -std::expm1(-20.0 * input.share * dt * (1.0 / cp + 1.0 / ca));
                const double q = (tp - ta) * (cp * ca / (cp + ca)) * decay;
                tp -= q / cp;
                temperature += static_cast<float>(q / ca / scale.kelvin_per_unit);
                value.heat_to_gas = q;
                value.exchanged = 1;
            }
            const double boil = input.boiling, latent = input.latent, molar = input.molar;
            const double rest = input.rest_mass;
            if (boil > 0 && latent > 0 && molar > 0 && rest > 0 && tp <= boil) {
                const double a = latent * molar / gas_constant;
                const double psat = atmosphere * std::exp(a * (1.0 / boil - 1.0 / tp));
                const double psat_air = atmosphere * std::exp(a * (1.0 / boil - 1.0 / std::min(ta, boil)));
                const double pv = std::max(humidity * psat_air,
                    static_cast<double>(m) / volume * gas_constant * ta / molar);
                const double flux = coefficient(pv) * std::max(0.0, psat - pv) *
                    std::sqrt(molar / (2.0 * pi * gas_constant * tp));
                const double mass = rest * input.fraction;
                const double c = input.specific_heat;
                const double thermal_budget = cp * std::max(0.0, tp - 1.0) / latent;
                const double dm = std::min({flux * input.share * dt, 0.5 * mass, thermal_budget});
                if (dm > 0 && cp > 0) {
                    tp -= dm * latent / cp;
                    value.fraction = static_cast<float>((mass - dm) / rest);
                    value.mass = dm;
                    value.sensible = dm * c * tp;
                    value.latent = dm * latent;
                    m += static_cast<float>(dm);
                    e += static_cast<float>(value.sensible + value.latent);
                    const double vapour_density = atmosphere * molar / (gas_constant * tp);
                    density += static_cast<float>(std::min(dm / (vapour_density * volume),
                        std::max(0.0, 1.0 - static_cast<double>(density))));
                }
            }
            // Condensation: the same Hertz-Knudsen flux run backwards when the
            // cell's vapour is above saturation at the SURFACE temperature
            // (steam over cold water, humid air over an ice bath). Only real
            // inventory vapour condenses (the ambient humidity carries no
            // mass), its partial pressure capped at the 1 atm total. Bounded by
            // the cell's excess over saturation, by doubling the parcel, and by
            // heating it to its boiling point. The vapour arrives at the air
            // temperature with its latent heat (c*Ta + L per kg); the energy
            // pool holds all the cell's water, so e/m is not the vapour's.
            if (!(value.mass > 0.0) && boil > 0 && latent > 0 && molar > 0 && rest > 0 &&
                tp < boil && cp > 0 && m > 0.0f) {
                const double a = latent * molar / gas_constant;
                const double psat = atmosphere * std::exp(a * (1.0 / boil - 1.0 / tp));
                const double pv = std::min(atmosphere,
                    static_cast<double>(m) / volume * gas_constant * ta / molar);
                if (pv > psat) {
                    const double c = input.specific_heat;
                    const double flux = coefficient(pv) * (pv - psat) *
                        std::sqrt(molar / (2.0 * pi * gas_constant * tp));
                    const double saturated = psat * volume * molar / (gas_constant * ta);
                    const double mass = rest * input.fraction;
                    const double eps = std::max(c * ta + latent, c * tp);
                    const double heat_budget = cp * (boil - tp) / std::max(eps - c * tp, 1.0);
                    const double dm = std::min({flux * input.share * dt,
                        std::max(0.0, static_cast<double>(m) - saturated), mass, heat_budget});
                    if (dm > 0) {
                        tp = (cp * tp + dm * eps) / (cp + dm * c);
                        value.mass = -dm;
                        value.sensible = dm * std::max(0.0, eps - latent);
                        value.latent = dm * latent;
                        m = std::max(0.0f, m - static_cast<float>(dm));
                        e = std::max(0.0f, e - static_cast<float>(dm * eps));
                        const double vapour_density = atmosphere * molar / (gas_constant * ta);
                        density = std::max(0.0f, density -
                            static_cast<float>(dm / (vapour_density * volume)));
                        condensed[g][0] += dm;
                        condensed[g][1] += dm * eps;
                    }
                }
            }
            // Rain absorbed at its own temperature (sensible only).
            if (rain_absorbed[g] && cp > 0 && rest > 0 && std::isfinite(input.kelvin) &&
                input.kelvin > 0.0f) {
                const double portion = std::max(0.0f, input.share) / share_sum;
                const double dr = rain_mass * portion;
                const double er = rain_heat * portion;
                const double c = input.specific_heat;
                const double capacity_now = cp + std::max(0.0, -value.mass) * c;
                tp = (capacity_now * tp + er) / (capacity_now + dr * c);
                value.rain_kg = static_cast<float>(dr);
                value.rain_energy_j = static_cast<float>(er);
            }
            value.kelvin = static_cast<float>(tp);
        }
        if (!std::isfinite(m) || !std::isfinite(e) || !std::isfinite(temperature) ||
            !std::isfinite(density) || m < 0 || e < 0) {
            return false;
        }
        columns.push_back({temperature, density});
    }
    if (!validateGasSurfaceResults(plan, results)) {
        return false;
    }
    for (std::size_t g = 0; g < plan.groups.size(); ++g) {
        const uint32_t cell = plan.groups[g].cell;
        grid.temperature[cell] = columns[g][0];
        grid.density[cell] = columns[g][1];
        // Evaporation deposits through publish; condensation is a withdrawal.
        if (condensed[g][0] > 0.0) {
            gas.gas_phase_mass_kg[cell] = std::max(0.0f,
                gas.gas_phase_mass_kg[cell] - static_cast<float>(condensed[g][0]));
            gas.gas_phase_energy_j[cell] = std::max(0.0f,
                gas.gas_phase_energy_j[cell] - static_cast<float>(condensed[g][1]));
        }
        if (rain_absorbed[g]) {
            gas.gas_phase_rain_kg[cell] = 0.0f;
            gas.gas_phase_rain_energy_j[cell] = 0.0f;
        }
    }
    publishGasSurfaceResults(plan, results, liquid_domain, liquid, gas_domain, gas,
                             event_id, ledger, false);
    return true;
}
}
