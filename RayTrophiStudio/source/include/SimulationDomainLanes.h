#pragma once

#include <string>
#include <vector>

namespace RayTrophiSim {
enum class SimulationDomainType;
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
struct SimulationFlowSourceDesc;

bool simulationDomainHasGas(const SimulationGridDomainDesc& domain);
bool simulationDomainHasGas(const SimulationGridDomainState& state);
bool simulationDomainHasLiquid(const SimulationGridDomainDesc& domain);
bool simulationDomainHasLiquid(const SimulationGridDomainState& state);
bool simulationDomainHasBothPhases(const SimulationGridDomainDesc& domain);
bool simulationDomainHasBothPhases(const SimulationGridDomainState& state);

// Gas/Fluid enum values are legacy input only. Canonical live domains are Matter.
SimulationDomainType parseSavedSimulationDomainType(const std::string& name);
SimulationDomainType parseSavedSimulationDomainType(int value);
void canonicalizeMatterDomain(SimulationGridDomainDesc& domain,
    std::vector<SimulationFlowSourceDesc>& sources, int domain_index);
bool setMatterDomainLanes(SimulationGridDomainDesc& domain,
    const std::vector<SimulationFlowSourceDesc>& sources, int domain_index,
    bool gas, bool liquid, bool has_live_state, std::string& error);
bool validateMatterCacheLane(const SimulationGridDomainState& state,
    const SimulationGridDomainDesc& domain, std::string& error);
bool validateMatterCacheLanes(const std::vector<SimulationGridDomainState>& states,
    const std::vector<SimulationGridDomainDesc>& domains, std::string& error);
bool resolveMatterSourcePhase(const SimulationGridDomainDesc& domain,
    const std::string& requested, bool& liquid, std::string& error);
std::vector<int> resolveMatterGasExchangeOwners(
    const std::vector<SimulationGridDomainDesc>& domains,
    const std::vector<SimulationGridDomainState>& states);
} // namespace RayTrophiSim
