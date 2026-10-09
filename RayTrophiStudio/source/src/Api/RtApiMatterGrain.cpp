#include "RtMatterModels.h"
#include "RtApiInternal.h"
#include "Fluid/MatterGrain.h"

#include <stdexcept>

namespace rtapi {

nlohmann::json matterGrainSettings(const std::string& domain, const nlohmann::json& patch,
                                 bool write) {
    if (!g_ctx) {
        throw std::runtime_error("rtapi is not bound to a UIContext");
    }
    if (write && renderJobActive()) {
        throw std::runtime_error("scene is locked by the final render job");
    }
    auto& runtime = scriptSimulationRuntime();
    auto& domains = runtime.gridDomains();
    for (std::size_t i = 0; i < domains.size(); ++i) {
        auto& d = domains[i];
        if (d.name != domain) {
            continue;
        }
        if (!write) {
            return RayTrophiSim::Fluid::matterGrainParamsToJson(d.fluid_params.grain);
        }
        if (d.type != RayTrophiSim::SimulationDomainType::Matter) {
            throw std::runtime_error("grain authoring requires a Matter domain");
        }
        auto candidate = d.fluid_params.grain;
        std::string error;
        if (!RayTrophiSim::Fluid::patchMatterGrainParams(patch, candidate, error)) {
            throw std::runtime_error(error);
        }
        if (!RayTrophiSim::Fluid::validateMatterGrainDomain(d, candidate, error)) {
            throw std::runtime_error(error);
        }
        // Only what is fixed at a grain's birth needs an empty domain: the
        // radius (packing is the substance's; substance.set guards it) (mass = density / packing x
        // sphere), and dropping wet grains while grains may hold water. The
        // material, contact and coupling settings act on the next step.
        const auto& states = runtime.gridDomainStates();
        if (i < states.size() && !states[i].particles.empty()) {
            const auto before = RayTrophiSim::Fluid::matterGrainParamsToJson(d.fluid_params.grain);
            const auto after = RayTrophiSim::Fluid::matterGrainParamsToJson(candidate);
            std::string fixed;
            for (const char* key : {"radius_m"}) {
                if (before.at(key) != after.at(key)) {
                    fixed += (fixed.empty() ? "" : ", ") + std::string(key);
                }
            }
            if (!fixed.empty()) {
                throw std::runtime_error("reset domain particles before changing " + fixed +
                    " (fixed when a grain is born)");
            }
        }
        d.fluid_params.grain = candidate;
        invalidateScriptSimulation();
        return RayTrophiSim::Fluid::matterGrainParamsToJson(candidate);
    }
    throw std::runtime_error("grid domain not found: " + domain);
}

} // namespace rtapi
