// CUDA simulation compute backend factory: Player (no-CUDA) link alternative.
//
// PLAYER BUILD ONLY. Studio links the real factory from SimulationComputeCuda.cu;
// the two must never be linked into the same target (Studio lists this file as
// None). Returning nullptr is the factory's documented "no CUDA device" result:
// callers (SceneData backend selection) fall back to Vulkan compute.
// Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md (Faz 3b).

#include "SimulationCompute.h"

namespace RayTrophiSim {

std::unique_ptr<ISimulationComputeBackend> createCudaSimulationComputeBackend() {
    return nullptr;
}

} // namespace RayTrophiSim
