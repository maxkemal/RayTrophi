#include "SimulationDomainLanes.h"
#include "ParticleSimulation.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/MatterPhaseGrid.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace RayTrophiSim {
namespace {
template <class Domain>
bool hasGas(const Domain& domain) {
    return domain.type == SimulationDomainType::Matter
        ? domain.gas_lane_enabled : domain.type == SimulationDomainType::LegacyGas;
}
template <class Domain>
bool hasLiquid(const Domain& domain) {
    return domain.type == SimulationDomainType::Matter
        ? domain.liquid_lane_enabled : domain.type == SimulationDomainType::LegacyFluid;
}
} // namespace

bool simulationDomainHasGas(const SimulationGridDomainDesc& domain) {
    return hasGas(domain);
}
bool simulationDomainHasGas(const SimulationGridDomainState& state) {
    return hasGas(state);
}
bool simulationDomainHasLiquid(const SimulationGridDomainDesc& domain) {
    return hasLiquid(domain);
}
bool simulationDomainHasLiquid(const SimulationGridDomainState& state) {
    return hasLiquid(state);
}
bool simulationDomainHasBothPhases(const SimulationGridDomainDesc& domain) {
    return hasGas(domain) && hasLiquid(domain);
}
bool simulationDomainHasBothPhases(const SimulationGridDomainState& state) {
    return hasGas(state) && hasLiquid(state);
}

SimulationDomainType parseSavedSimulationDomainType(const std::string& name) {
    if (name == "Matter") {
        return SimulationDomainType::Matter;
    }
    if (name == "Gas") {
        return SimulationDomainType::LegacyGas;
    }
    if (name == "Fluid") {
        return SimulationDomainType::LegacyFluid;
    }
    throw std::invalid_argument("unknown saved simulation domain type: " + name);
}

SimulationDomainType parseSavedSimulationDomainType(int value) {
    if (value == static_cast<int>(SimulationDomainType::Matter) ||
        value == static_cast<int>(SimulationDomainType::LegacyGas) ||
        value == static_cast<int>(SimulationDomainType::LegacyFluid)) {
        return static_cast<SimulationDomainType>(value);
    }
    throw std::invalid_argument("unknown saved simulation domain type value: " +
                                std::to_string(value));
}

void canonicalizeMatterDomain(SimulationGridDomainDesc& domain,
    std::vector<SimulationFlowSourceDesc>& sources, int domain_index) {
    if (domain.type == SimulationDomainType::Matter) {
        return;
    }
    if (domain.type != SimulationDomainType::LegacyGas &&
        domain.type != SimulationDomainType::LegacyFluid) {
        throw std::invalid_argument("unknown saved simulation domain type");
    }
    const bool liquid = domain.type == SimulationDomainType::LegacyFluid;
    domain.gas_lane_enabled = !liquid;
    domain.liquid_lane_enabled = liquid;
    for (auto& source : sources) {
        if (source.domain_index == domain_index) {
            source.phase = liquid ? SimulationFlowSourceDesc::Phase::Liquid
                                  : SimulationFlowSourceDesc::Phase::Gas;
        }
    }
    if (!liquid && !domain.fluid_fog_shader && domain.shader) {
        domain.fluid_fog_shader = domain.shader;
    }
    domain.type = SimulationDomainType::Matter;
}

bool setMatterDomainLanes(SimulationGridDomainDesc& domain,
    const std::vector<SimulationFlowSourceDesc>& sources, int domain_index,
    bool gas, bool liquid, bool has_live_state, std::string& error) {
    if (!gas && !liquid) {
        error = "at least one Matter lane must remain enabled";
        return false;
    }
    if (gas == hasGas(domain) && liquid == hasLiquid(domain)) {
        error.clear();
        return true;
    }
    if (has_live_state) {
        error = "reset the simulation before changing Matter lanes";
        return false;
    }
    if (!liquid && Fluid::matterGrainOwnership(domain, sources, domain_index, false).wanted) {
        error = "DEM substances require the liquid/particle lane";
        return false;
    }
    for (const auto& source : sources) {
        if (source.domain_index != domain_index) {
            continue;
        }
        if ((source.phase == SimulationFlowSourceDesc::Phase::Gas && !gas) ||
            (source.phase == SimulationFlowSourceDesc::Phase::Liquid && !liquid)) {
            error = "source '" + source.name +
                "' targets the lane being disabled; retarget or remove it first";
            return false;
        }
    }
    const bool was_dual = hasGas(domain) && hasLiquid(domain);
    if (gas && liquid && hasGas(domain) && !was_dual) {
        domain.fluid_fog_shader = domain.shader;
    } else if (gas && !liquid && was_dual && domain.fluid_fog_shader) {
        domain.shader = domain.fluid_fog_shader;
    }
    domain.type = SimulationDomainType::Matter;
    domain.gas_lane_enabled = gas;
    domain.liquid_lane_enabled = liquid;
    error.clear();
    return true;
}

bool validateMatterCacheLane(const SimulationGridDomainState& state,
    const SimulationGridDomainDesc& domain, std::string& error) {
    if (state.type != SimulationDomainType::Matter ||
        hasGas(state) != hasGas(domain) || hasLiquid(state) != hasLiquid(domain)) {
        error = "cache lane layout differs for '" + domain.name +
            "'; legacy Gas/Fluid or changed Matter lanes require a rebake";
        return false;
    }
    error.clear();
    return true;
}

bool validateMatterCacheLanes(const std::vector<SimulationGridDomainState>& states,
    const std::vector<SimulationGridDomainDesc>& domains, std::string& error) {
    if (states.size() != domains.size()) {
        error = "cached domain count differs from the current scene; rebake simulation";
        return false;
    }
    for (std::size_t i = 0; i < states.size(); ++i) {
        if (!validateMatterCacheLane(states[i], domains[i], error)) {
            return false;
        }
    }
    error.clear();
    return true;
}

bool resolveMatterSourcePhase(const SimulationGridDomainDesc& domain,
    const std::string& requested, bool& liquid, std::string& error) {
    std::string phase = requested;
    std::transform(phase.begin(), phase.end(), phase.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    bool candidate = false;
    if (phase.empty() || phase == "auto") {
        candidate = !hasGas(domain) && hasLiquid(domain);
    } else if (phase == "liquid" || phase == "fluid") {
        candidate = true;
    } else if (phase != "gas") {
        error = "flow source phase must be auto, gas, or liquid";
        return false;
    }
    if (candidate ? !hasLiquid(domain) : !hasGas(domain)) {
        error = "flow source targets a disabled Matter lane; enable the lane first";
        return false;
    }
    liquid = candidate;
    error.clear();
    return true;
}

std::vector<int> resolveMatterGasExchangeOwners(
    const std::vector<SimulationGridDomainDesc>& domains,
    const std::vector<SimulationGridDomainState>& states) {
    std::vector<int> result(domains.size(), -1);
    const auto available = [&](std::size_t liquid, std::size_t gas) {
        return gas < states.size() && domains[gas].enabled && states[gas].valid &&
            hasGas(domains[gas]) && Fluid::gridsOverlap(
                Fluid::liquidGrid(states[liquid]), Fluid::gasGrid(states[gas]));
    };
    for (std::size_t liquid = 0; liquid < domains.size() && liquid < states.size(); ++liquid) {
        if (!domains[liquid].enabled || !states[liquid].valid || !hasLiquid(domains[liquid])) {
            continue;
        }
        if (available(liquid, liquid)) {
            result[liquid] = static_cast<int>(liquid);
            continue;
        }
        for (std::size_t gas = 0; gas < domains.size(); ++gas) {
            if (available(liquid, gas)) {
                result[liquid] = static_cast<int>(gas);
                break;
            }
        }
    }
    return result;
}
} // namespace RayTrophiSim
