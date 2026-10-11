#pragma once
#include <cstddef>

namespace RayTrophiSim {
struct SimulationGridDomainState;
struct SimulationGridDomainComputeBuffers;
class SimulationComputeContext;
namespace Fluid {
class FluidParticles;
struct APICSolverParams;
struct WhitewaterBinWindow {
    int nx = 0, ny = 0, nz = 0;
    int x = 0, y = 0, z = 0;
};
bool isWhitewaterFluidParcel(const FluidParticles& particles, std::size_t index,
                            const APICSolverParams* source_params = nullptr);
WhitewaterBinWindow whitewaterBinWindow(const SimulationGridDomainState& state,
                                        float radius_voxels, bool device_only,
                                        const APICSolverParams* source_params = nullptr);
void disableWhitewaterRuntime(SimulationGridDomainState& state,
                             SimulationComputeContext* compute,
                             SimulationGridDomainComputeBuffers* buffers);
}
}
