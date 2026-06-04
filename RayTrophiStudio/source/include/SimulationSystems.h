#pragma once

#include "SimulationWorld.h"

#include <memory>
#include <vector>

class GasVolume;

namespace RayTrophiSim {

class GasVolumeSimulationSystem final : public ISimulationSystem {
public:
    void setVolumes(std::vector<std::shared_ptr<GasVolume>>* volumes);
    void setUploadStream(void* stream);

    const char* name() const override { return "Gas Volumes"; }
    SimulationSystemKind kind() const override { return SimulationSystemKind::Gas; }
    int order() const override { return 100; }
    bool enabled() const override;
    void step(const SimulationContext& context) override;

private:
    std::vector<std::shared_ptr<GasVolume>>* volumes_ = nullptr;
    void* stream_ = nullptr;
};

} // namespace RayTrophiSim
