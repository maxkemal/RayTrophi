#pragma once

#include "Vec3.h"
#include "SimulationCompute.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace RayTrophiSim::Fluid::ActiveWindow {

// Half-open bounds in the full grid's coordinates. Storage is never cropped.
struct Bounds {
    int begin[3] = {0, 0, 0};
    int end[3] = {0, 0, 0};
    bool bounded = false;

    uint64_t cells() const {
        return uint64_t(end[0] - begin[0]) * uint64_t(end[1] - begin[1]) *
               uint64_t(end[2] - begin[2]);
    }
};

inline Bounds full(int nx, int ny, int nz) {
    Bounds result;
    result.end[0] = std::max(0, nx);
    result.end[1] = std::max(0, ny);
    result.end[2] = std::max(0, nz);
    return result;
}

inline Bounds plan(const std::vector<Vec3>& positions,
                   const std::vector<Vec3>& velocities,
                   const Vec3& origin, float voxel, float dt,
                   int nx, int ny, int nz, bool host_positions_valid) {
    const Bounds fallback = full(nx, ny, nz);
    if (!host_positions_valid || positions.empty() || velocities.size() != positions.size() ||
        nx <= 0 || ny <= 0 || nz <= 0 || !std::isfinite(voxel) || voxel <= 0.0f ||
        !std::isfinite(dt) || dt < 0.0f) {
        return fallback;
    }
    const int dims[3] = {nx, ny, nz};
    const double offsets[3] = {origin.x, origin.y, origin.z};
    Bounds result = fallback;
    for (int axis = 0; axis < 3; ++axis) {
        result.begin[axis] = dims[axis];
        result.end[axis] = 0;
        if (!std::isfinite(offsets[axis])) {
            return fallback;
        }
    }
    for (size_t particle = 0; particle < positions.size(); ++particle) {
        const auto& p = positions[particle];
        const auto& v = velocities[particle];
        const double coordinates[3] = {p.x, p.y, p.z};
        const double speeds[3] = {v.x, v.y, v.z};
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(coordinates[axis]) || !std::isfinite(speeds[axis])) {
                return fallback;
            }
            const double cell = (coordinates[axis] - offsets[axis]) / voxel;
            // Quadratic MAC support plus pressure neighbour and CFL travel halo.
            const double halo = 2.0 + std::ceil(std::abs(speeds[axis]) * dt / voxel);
            const double lower = std::floor(cell) - halo;
            const double upper = std::floor(cell) + halo + 2.0;
            if (!std::isfinite(lower) || !std::isfinite(upper)) {
                return fallback;
            }
            // Clamp before integer conversion, including particles outside the box.
            const int first = static_cast<int>(std::clamp(lower, 0.0, double(dims[axis])));
            const int last = static_cast<int>(std::clamp(upper, 0.0, double(dims[axis])));
            result.begin[axis] = std::min(result.begin[axis], first);
            result.end[axis] = std::max(result.end[axis], last);
        }
    }
    if (result.cells() == 0) {
        return fallback;
    }
    result.bounded = result.cells() < fallback.cells();
    return result;
}

struct NormalizeConstants {
    int nx, ny, nz, component;
    int begin_x, begin_y, begin_z;
    int extent_x, extent_y, extent_z;
};
static_assert(sizeof(NormalizeConstants) == 40, "Active normalize push-constant ABI");

inline bool normalize(RayTrophiSim::SimulationComputeContext& compute, const Bounds& window,
                      int nx, int ny, int nz, int component,
                      RayTrophiSim::ComputeBufferHandle velocity,
                      RayTrophiSim::ComputeBufferHandle weight) {
    NormalizeConstants constants = {
        nx, ny, nz, component,
        window.begin[0], window.begin[1], window.begin[2],
        window.end[0] - window.begin[0] + (component == 0 ? 1 : 0),
        window.end[1] - window.begin[1] + (component == 1 ? 1 : 0),
        window.end[2] - window.begin[2] + (component == 2 ? 1 : 0)
    };
    const uint64_t count = uint64_t(constants.extent_x) * constants.extent_y *
                           constants.extent_z;
    if (count == 0 || count > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    RayTrophiSim::ComputeBufferHandle buffers[] = {velocity, weight};
    RayTrophiSim::ComputeDispatch command;
    command.kernel = "sim_fluid_normalize_window";
    command.buffers = buffers;
    command.buffer_count = 2;
    command.constants = &constants;
    command.constants_size = sizeof(constants);
    command.groups.groups_x = static_cast<uint32_t>((count + 255) / 256);
    return compute.dispatch(command);
}

template <typename Block, typename Stats>
inline void drawDiagnostics(Block& block, const Stats& stats) {
    if (stats.normalize_window_used && stats.grid_cell_count > 0) {
        block.Value("P2G normalize cells", "%llu / %zu (%.1f%%)",
                    static_cast<unsigned long long>(stats.normalize_window_cells),
                    stats.grid_cell_count,
                    100.0 * double(stats.normalize_window_cells) / stats.grid_cell_count);
    }
    if (stats.pressure_window_used && stats.grid_cell_count > 0) {
        block.Value("Pressure window cells", "%llu / %zu (%.1f%%)",
                    static_cast<unsigned long long>(stats.pressure_window_cells),
                    stats.grid_cell_count,
                    100.0 * double(stats.pressure_window_cells) / stats.grid_cell_count);
    }
    if (stats.occupancy_on_gpu) {
        block.Value("Occupancy mask", "%s", "GPU (resident)");
    }
}

} // namespace RayTrophiSim::Fluid::ActiveWindow
