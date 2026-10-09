#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace FluidSim {
class FluidGrid;
}

namespace RayTrophiSim {
class SimulationComputeContext;
struct SimulationGridDomainComputeBuffers;

namespace Fluid {

// Host grid remains dense in S1b. Pack only faces owned by the supplied MAC
// tiles, in their GPU slot order. Padding is 1 (fully open), never a closed wall.
// The output is replaced only on success. No domain-sized occupancy allocation.
bool packSparseMacSolidWeights(
    const std::array<int, 3>& cells,
    const std::vector<uint32_t>& tile_keys,
    const std::array<const std::vector<uint8_t>*, 3>& weights,
    std::array<std::vector<float>, 3>& pages,
    std::string& error);

// Called after canonical P2G rebuilt its topology and before variational
// divergence. Uses pages allocated/budgeted by P2G; no dense GPU weight bank.
bool uploadSparseMacSolidWeights(SimulationComputeContext& compute,
                                 SimulationGridDomainComputeBuffers& buffers,
                                 const FluidSim::FluidGrid& grid,
                                 std::string& error);

} // namespace Fluid
} // namespace RayTrophiSim
