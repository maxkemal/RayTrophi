#pragma once

#include "SimulationCompute.h"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace RayTrophiSim {
struct SimulationGridDomainState;
struct SimulationGridDomainComputeBuffers;
namespace GridFluid { struct SolverParams; }
namespace Fluid {

// Passive render ownership, never a second physical mass/temperature solver.
// UID 0 owns background, phase exchange and other unattributed producers.
struct GasSourceImage {
    std::string name;
    std::vector<float> density;
    std::vector<float> fuel;
    std::vector<float> pending_density;
    std::vector<float> pending_fuel;
};
using GasSourceImages = std::map<uint64_t, GasSourceImage>;
struct GasSourceDevice;

enum class GasSourceStage { Prepare, Advect, Transported, Reacted, Finish };

GasSourceImage* prepareGasSourceDeposit(SimulationGridDomainState& state,
    uint64_t uid, const std::string& name);
void gasSourceStage(SimulationGridDomainState& state,
    const GridFluid::SolverParams& params, float dt, GasSourceStage stage,
    SimulationComputeContext* compute = nullptr,
    SimulationGridDomainComputeBuffers* buffers = nullptr);
// Snapshot into a COPY, detach the weak live view. Failure must abort the consumer.
bool snapshotGasSources(SimulationGridDomainState& snapshot);
void materializeGasSources(SimulationGridDomainState& state);
void releaseGasSources(SimulationGridDomainComputeBuffers& buffers);
void releaseGasSourceDomains(std::vector<SimulationGridDomainState>& states,
    std::vector<SimulationGridDomainComputeBuffers>& buffers);
void validateGasSourceBudget(std::size_t cells, std::size_t streams);

} // namespace Fluid
} // namespace RayTrophiSim
