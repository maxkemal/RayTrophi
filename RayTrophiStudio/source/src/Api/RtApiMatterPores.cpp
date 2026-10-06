#include "RtMatterModels.h"
#include "RtApiInternal.h"
#include "Fluid/MatterGrain.h"
#include "Fluid/MatterPoreAuthoring.h"

#include <stdexcept>

namespace rtapi {

nlohmann::json setMatterPoreExchange(const std::string& domain, const nlohmann::json& patch) {
    if (!g_ctx) {
        throw std::runtime_error("rtapi is not bound to a UIContext");
    }
    if (renderJobActive()) {
        throw std::runtime_error("scene is locked by the final render job");
    }
    auto& runtime = scriptSimulationRuntime();
    auto& domains = runtime.gridDomains();
    for (std::size_t i = 0; i < domains.size(); ++i) {
        auto& descriptor = domains[i];
        if (descriptor.name != domain) {
            continue;
        }
        if (descriptor.type != RayTrophiSim::SimulationDomainType::Matter) {
            throw std::runtime_error("C5 authoring requires a Matter domain");
        }
        auto candidate = descriptor.fluid_params.pore_exchange;
        std::string error;
        if (!RayTrophiSim::Fluid::patchMatterPoreParams(patch, candidate, error)) {
            throw std::runtime_error(error);
        }
        if ((candidate.enabled || candidate.wet_response_enabled) &&
            descriptor.fluid_params.grain.enabled) {
            throw std::runtime_error("grains are enabled: pore water exchange cannot run with "
                                     "grains; use wet grains");
        }
        if ((candidate.enabled || candidate.wet_response_enabled) &&
            (descriptor.backend != RayTrophiSim::SimulationDomainBackend::GPU_Vulkan ||
             descriptor.boundary_mode != RayTrophiSim::SimulationGridDomainBoundaryMode::Closed)) {
            throw std::runtime_error("Pore exchange / wet physics requires Closed Vulkan Matter");
        }
        const auto& states = runtime.gridDomainStates();
        if (i < states.size() && candidate.porosity != descriptor.fluid_params.pore_exchange.porosity &&
            RayTrophiSim::Fluid::needsMatterPoreTransport(states[i].particles, candidate)) {
            throw std::runtime_error("Drain pore water before changing porosity");
        }
        descriptor.fluid_params.pore_exchange = candidate;
        invalidateScriptSimulation();
        return RayTrophiSim::Fluid::matterPoreParamsToJson(candidate);
    }
    throw std::runtime_error("grid domain not found: " + domain);
}

} // namespace rtapi
