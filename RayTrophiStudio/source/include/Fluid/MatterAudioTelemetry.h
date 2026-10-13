#pragma once

#include "Vec3.h"
#include <cstdint>
#include <string>
#include <unordered_map>

namespace RayTrophiSim {
struct SimulationGridDomainState;
struct SimulationGridDomainDesc;
struct SimulationGridDomainComputeBuffers;
class SimulationComputeContext;
namespace Fluid {

struct DomainAudioMeasurement {
    bool measured = false;
    bool physical_power = true;
    bool energy_measured = false;
    double value = 0.0;
    double energy_j = 0.0;
    double duration_seconds = 0.0;
    double position_weight = 0.0;
    Vec3 position = Vec3(0);
    std::uint64_t carriers = 0;
    std::string reason = "not_published";
};

struct MatterAudioStats {
    std::uint64_t publication_version = 0;
    DomainAudioMeasurement fluid_flow;
    DomainAudioMeasurement gas_flow;
    DomainAudioMeasurement mpm_deformation;
    DomainAudioMeasurement mpm_fracture;
};

// Main-thread solver publication, never an audio-callback or renderer scan.
class MatterAudioFrame {
public:
    MatterAudioFrame(SimulationGridDomainState& state,
                    const SimulationGridDomainDesc& domain, double duration);
    void publishLiquid();
    void publishGas(SimulationComputeContext* compute,
                    SimulationGridDomainComputeBuffers* buffers);

private:
    SimulationGridDomainState& state_;
    const SimulationGridDomainDesc& domain_;
    double duration_;
    std::unordered_map<std::uint64_t, float> damage_;
    bool damage_valid_ = true;
};

bool measureGasAudioGpu(SimulationGridDomainState& state,
                       SimulationComputeContext& compute,
                       SimulationGridDomainComputeBuffers& buffers,
                       double duration, DomainAudioMeasurement& output);

} // namespace Fluid
} // namespace RayTrophiSim
