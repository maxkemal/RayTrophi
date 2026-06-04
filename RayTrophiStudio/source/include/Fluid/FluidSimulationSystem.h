/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FluidSimulationSystem.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * ISimulationSystem adapter that drives APIC liquid objects each step. Lives
 * alongside GasVolumeSimulationSystem / ParticleSimulationSystem; gets
 * registered with the SimulationWorld by the scene layer that owns the
 * fluid_objects list.
 */

#pragma once

#include "../SimulationWorld.h"
#include "FluidObject.h"

#include <vector>

namespace RayTrophiSim {

class FluidSimulationSystem final : public ISimulationSystem {
public:
    void setObjects(std::vector<Fluid::FluidObject>* objects);

    const char* name() const override { return "Fluid Objects"; }
    SimulationSystemKind kind() const override { return SimulationSystemKind::Fluid; }
    int  order() const override { return 110; }   // after gas (100)
    bool enabled() const override;

    void step(const SimulationContext& context) override;

private:
    std::vector<Fluid::FluidObject>* objects_ = nullptr;
};

} // namespace RayTrophiSim
