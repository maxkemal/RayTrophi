#include "Fluid/GasInventoryGpu.h"
#include "scene_data.h"

#include <stdexcept>

namespace RayTrophiSim::Fluid {
std::string gasInventoryStepError(const ::SceneData& scene) {
    for (const auto& system : scene.particle_systems) {
        if (!system.runtime) {
            continue;
        }
        for (const auto& state : system.runtime->gridDomainStates()) {
            if (!state.gas_inventory_error.empty()) {
                return state.gas_inventory_error;
            }
        }
    }
    return {};
}

GasInventoryHostImage captureGasInventoryHostImage(const SimulationGridDomainState& state) {
    if (!state.gas_inventory_error.empty()) {
        throw std::runtime_error("cannot serialize held gas inventory");
    }
    // Explicit serialization is a snapshot consumer, never a physics stage.
    if (state.gas_inventory_revision == 0) {
        return {state.gas_phase_mass_kg, state.gas_phase_energy_j,
                state.gas_phase_condensate_kg, state.gas_phase_rain_kg,
                state.gas_phase_rain_energy_j, state.gas_phase_foreign_kg};
    }
    SimulationGridDomainState snapshot;
    snapshot.gas_inventory_device = state.gas_inventory_device;
    snapshot.gas_inventory_revision = state.gas_inventory_revision;
    if (!snapshotGasInventory(snapshot)) {
        throw std::runtime_error("cannot serialize current gas inventory device image");
    }
    return {std::move(snapshot.gas_phase_mass_kg), std::move(snapshot.gas_phase_energy_j),
            std::move(snapshot.gas_phase_condensate_kg), std::move(snapshot.gas_phase_rain_kg),
            std::move(snapshot.gas_phase_rain_energy_j),
            std::move(snapshot.gas_phase_foreign_kg)};
}
bool captureGasInventoryHostImages(const std::vector<SimulationGridDomainState>& states,
                                   std::vector<GasInventoryHostImage>& images) {
    std::vector<GasInventoryHostImage> candidate;
    candidate.reserve(states.size());
    try {
        for (const auto& state : states) {
            candidate.push_back(captureGasInventoryHostImage(state));
        }
    } catch (const std::exception& error) {
        SCENE_LOG_ERROR(std::string("Gas inventory serialization rejected: ") + error.what());
        return false;
    }
    images = std::move(candidate);
    return true;
}
} // namespace RayTrophiSim::Fluid
