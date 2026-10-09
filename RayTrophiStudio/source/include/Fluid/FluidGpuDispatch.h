#pragma once

#include "SimulationCompute.h"
#include <algorithm>
#include <cstdint>

namespace RayTrophiSim::FluidGpuDispatch {

inline ComputeDispatchSize groups256(uint32_t elements) {
    const uint64_t groups = std::max<uint64_t>(1u, (uint64_t(elements) + 255u) / 256u);
    // Vulkan guarantees >=65535 workgroups per dimension. Balance the final
    // rectangle: 65536 logical groups become 32768x2, not 65535x2.
    const uint32_t rows = static_cast<uint32_t>((groups + 65534u) / 65535u);
    const uint32_t columns = static_cast<uint32_t>((groups + rows - 1u) / rows);
    return {columns, rows, 1u};
}

} // namespace RayTrophiSim::FluidGpuDispatch
