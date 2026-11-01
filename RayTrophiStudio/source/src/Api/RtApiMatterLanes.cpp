#include "RtMatterLanes.h"
#include "RtApiInternal.h"
#include "Fluid/GasInventoryGpu.h"
#include "SimulationDomainLanes.h"

#include <stdexcept>

namespace rtapi {
namespace {
std::size_t findLaneDomain(RayTrophiSim::ParticleSimulationSystem& runtime,
                         const std::string& name) {
    const auto& domains = runtime.gridDomains();
    for (std::size_t i = 0; i < domains.size(); ++i) {
        if (domains[i].name == name) {
            return i;
        }
    }
    throw std::runtime_error("grid domain not found: " + name);
}
} // namespace

nlohmann::json getMatterDomainLanes(const std::string& name) {
    if (!g_ctx) {
        throw std::runtime_error("rtapi is not bound to a UIContext");
    }
    auto& runtime = scriptSimulationRuntime();
    const auto index = findLaneDomain(runtime, name);
    const auto& domain = runtime.gridDomains()[index];
    const auto& states = runtime.gridDomainStates();
    const bool measured = index < states.size() && states[index].valid;
    const bool gas = RayTrophiSim::simulationDomainHasGas(domain);
    const bool liquid = RayTrophiSim::simulationDomainHasLiquid(domain);
    return {{"type", "matter"}, {"measured", measured},
        {"primary_cells", measured ? states[index].grid.getCellCount() : 0},
        {"secondary_cells", measured ? states[index].matter_liquid_grid.getCellCount() : 0},
        {"gas_lane_enabled", gas},
        {"liquid_lane_enabled", liquid},
        {"preset", gas && liquid ? "matter" : gas ? "gas" : "liquid"}};
}

nlohmann::json setMatterDomainLanes(const std::string& name, bool gas, bool liquid) {
    if (!g_ctx) {
        throw std::runtime_error("rtapi is not bound to a UIContext");
    }
    if (renderJobActive()) {
        throw std::runtime_error("cannot change Matter lanes during a render job");
    }
    auto& runtime = scriptSimulationRuntime();
    const auto index = findLaneDomain(runtime, name);
    const auto& states = runtime.gridDomainStates();
    const bool live = index < states.size() &&
        (!states[index].particles.empty() ||
         RayTrophiSim::Fluid::gasInventoryHasContent(states[index]) ||
         states[index].gas_stats.active_density_cells > 0 ||
         states[index].gas_stats.active_fuel_cells > 0 ||
         states[index].gas_stats.max_density > 0.0f ||
         states[index].gas_stats.max_temperature > 0.0f ||
         states[index].gas_stats.max_speed > 0.0f);
    auto& domain = runtime.gridDomains()[index];
    const bool changed = gas != RayTrophiSim::simulationDomainHasGas(domain) ||
        liquid != RayTrophiSim::simulationDomainHasLiquid(domain);
    std::string error;
    if (!RayTrophiSim::setMatterDomainLanes(domain, runtime.flowSources(),
            static_cast<int>(index), gas, liquid, live, error)) {
        throw std::runtime_error(error);
    }
    if (changed) {
        runtime.synchronizeGridDomainsNow();
        invalidateScriptSimulation();
    }
    return getMatterDomainLanes(name);
}
} // namespace rtapi
