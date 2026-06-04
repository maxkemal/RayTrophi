#include "SimulationSystems.h"

#include "GasVolume.h"
#include "GasSimulator.h"

namespace RayTrophiSim {

void GasVolumeSimulationSystem::setVolumes(std::vector<std::shared_ptr<GasVolume>>* volumes) {
    volumes_ = volumes;
}

void GasVolumeSimulationSystem::setUploadStream(void* stream) {
    stream_ = stream;
}

bool GasVolumeSimulationSystem::enabled() const {
    return volumes_ != nullptr && !volumes_->empty();
}

void GasVolumeSimulationSystem::step(const SimulationContext& context) {
    if (!volumes_) {
        return;
    }

    for (auto& gas : *volumes_) {
        if (!gas) {
            continue;
        }

        gas->getSimulator().setExternalForceFieldManager(context.force_fields);
        gas->getSimulator().setExternalForceFieldSnapshot(context.force_snapshot);
        gas->update(context.dt, stream_);
    }
}

} // namespace RayTrophiSim
