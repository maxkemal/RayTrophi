#include "RtPhaseGrid.h"
#include "RtApiInternal.h"
#include "Fluid/MatterPhaseConfig.h"

#include <stdexcept>

namespace rtapi {
namespace {

std::size_t findDomain(RayTrophiSim::ParticleSimulationSystem& runtime,
                       const std::string& name) {
    const auto& domains = runtime.gridDomains();
    for (std::size_t index = 0; index < domains.size(); ++index) {
        if (domains[index].name == name) {
            return index;
        }
    }
    throw std::runtime_error("grid domain not found: " + name);
}

Vec3 boundsVector(const nlohmann::json& value) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("bounds must contain three numbers");
    }
    for (const auto& component : value) {
        if (!component.is_number()) {
            throw std::runtime_error("bounds must contain three numbers");
        }
    }
    return Vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
}

} // namespace

nlohmann::json getPhaseGrids(const std::string& domain) {
    if (!g_ctx) {
        throw std::runtime_error("rtapi is not bound to a UIContext");
    }
    auto& runtime = scriptSimulationRuntime();
    const auto index = findDomain(runtime, domain);
    const auto& states = runtime.gridDomainStates();
    return RayTrophiSim::Fluid::phaseGridInfo(runtime.gridDomains()[index],
        index < states.size() ? &states[index] : nullptr);
}

nlohmann::json setPhaseGrid(const std::string& domain, const std::string& phase,
                           bool inherit, const nlohmann::json& bounds_min,
                           const nlohmann::json& bounds_max, float voxel) {
    if (!g_ctx) {
        throw std::runtime_error("rtapi is not bound to a UIContext");
    }
    if (renderJobActive()) {
        throw std::runtime_error("cannot change phase grids during a render job");
    }
    auto& runtime = scriptSimulationRuntime();
    const auto index = findDomain(runtime, domain);
    RayTrophiSim::Fluid::GridPhase selected;
    std::string error;
    if (!RayTrophiSim::Fluid::parseGridPhase(phase, selected, error)) {
        throw std::runtime_error(error);
    }
    const Vec3 lo = inherit ? Vec3(0.0f) : boundsVector(bounds_min);
    const Vec3 hi = inherit ? Vec3(1.0f) : boundsVector(bounds_max);
    if (!RayTrophiSim::Fluid::setPhaseGrid(runtime.gridDomains()[index], selected,
                                         inherit, lo, hi, voxel, error)) {
        throw std::runtime_error(error);
    }
    runtime.resetGridDomainStates();
    invalidateScriptSimulation();
    return getPhaseGrids(domain);
}

} // namespace rtapi
