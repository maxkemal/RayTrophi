#include "Fluid/GasInventoryFrameCapture.h"
#include "scene_data.h"

#include <exception>

namespace RayTrophiSim::Fluid {
GasInventoryFrameCapture captureGasInventoryFrame(::SceneData& scene) {
    GasInventoryFrameCapture frame;
    frame.error = gasInventoryStepError(scene);
    if (!frame.error.empty()) {
        return frame;
    }
    frame.states.reserve(scene.particle_systems.size());
    try {
        for (const auto& system : scene.particle_systems) {
            if (system.runtime) {
                frame.states.push_back(system.runtime->captureGridDomainStatesForCache(
                    scene.simulation_world.compute()));
            } else {
                frame.states.emplace_back();
            }
        }
    } catch (const std::exception& error) {
        frame.error = error.what();
        frame.states.clear();
        return frame;
    }
    frame.ok = true;
    return frame;
}
}
