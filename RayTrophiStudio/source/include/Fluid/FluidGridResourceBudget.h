#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace RayTrophiSim::Fluid {

// This is the dense working-set model enforced by ParticleSimulation. Keep the
// UI and the clamp on this one definition: the old panel used 44 bytes/cell
// while the Matter solver reserved 352, so its colour and warning had no
// relationship to the limit that actually changed the grid.
constexpr std::size_t kGasWorkingBytesPerCell = 128u;
constexpr std::size_t kLiquidWorkingBytesPerCell = 224u;

struct GridResourceEstimate {
    std::size_t cell_count = 0u;
    std::size_t bytes_per_cell = 0u;
    std::size_t working_bytes = 0u;
};

inline GridResourceEstimate estimateGridResources(
    int resolution_x,
    int resolution_y,
    int resolution_z,
    bool has_gas,
    bool has_liquid) {
    const std::size_t x = static_cast<std::size_t>(std::max(resolution_x, 0));
    const std::size_t y = static_cast<std::size_t>(std::max(resolution_y, 0));
    const std::size_t z = static_cast<std::size_t>(std::max(resolution_z, 0));
    const std::size_t bytes_per_cell =
        (has_gas ? kGasWorkingBytesPerCell : 0u) +
        (has_liquid ? kLiquidWorkingBytesPerCell : 0u);
    const std::size_t cells = x * y * z;
    return {cells, bytes_per_cell, cells * bytes_per_cell};
}

inline std::size_t gridCellBudget(
    std::size_t budget_bytes,
    bool has_gas,
    bool has_liquid) {
    const std::size_t bytes_per_cell =
        (has_gas ? kGasWorkingBytesPerCell : 0u) +
        (has_liquid ? kLiquidWorkingBytesPerCell : 0u);
    if (bytes_per_cell == 0u) {
        return 8u * 8u * 8u;
    }
    return std::max<std::size_t>(8u * 8u * 8u, budget_bytes / bytes_per_cell);
}

} // namespace RayTrophiSim::Fluid
