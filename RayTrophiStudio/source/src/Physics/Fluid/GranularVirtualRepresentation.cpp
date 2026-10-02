#include "Fluid/GranularVirtualRepresentation.h"

#include "Fluid/FluidParticles.h"
#include "FluidGrid.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>

namespace RayTrophiSim::Fluid {
namespace {

std::size_t columnIndex(int x, int z, int nx) {
    return static_cast<std::size_t>(x) +
           static_cast<std::size_t>(z) * static_cast<std::size_t>(nx);
}

bool worldToCell(
    const Vec3& position,
    const FluidSim::FluidGrid& grid,
    int& x,
    int& y,
    int& z) {
    if (!std::isfinite(position.x) ||
        !std::isfinite(position.y) ||
        !std::isfinite(position.z)) {
        return false;
    }

    x = static_cast<int>(std::floor((position.x - grid.origin.x) / grid.voxel_size));
    y = static_cast<int>(std::floor((position.y - grid.origin.y) / grid.voxel_size));
    z = static_cast<int>(std::floor((position.z - grid.origin.z) / grid.voxel_size));
    return x >= 0 && x < grid.nx &&
           y >= 0 && y < grid.ny &&
           z >= 0 && z < grid.nz;
}

bool touchesCollider(
    int x,
    int y,
    int z,
    const FluidSim::FluidGrid& grid) {
    static constexpr int kOffsets[6][3] = {
        {-1, 0, 0}, {1, 0, 0},
        {0, -1, 0}, {0, 1, 0},
        {0, 0, -1}, {0, 0, 1}
    };
    if (grid.solid.empty()) {
        return false;
    }

    for (const auto& offset : kOffsets) {
        const int nx = x + offset[0];
        const int ny = y + offset[1];
        const int nz = z + offset[2];
        if (nx < 0 || nx >= grid.nx ||
            ny < 0 || ny >= grid.ny ||
            nz < 0 || nz >= grid.nz) {
            continue;
        }
        const std::size_t neighbor = grid.cellIndex(nx, ny, nz);
        if (neighbor < grid.solid.size() &&
            grid.solid[neighbor] == FluidSim::FluidGrid::kSolidCollider) {
            return true;
        }
    }
    return false;
}

void smoothHeightField(
    std::vector<float>& height,
    const std::vector<std::uint8_t>& valid,
    int nx,
    int nz,
    std::uint32_t iterations) {
    std::vector<float> scratch(height.size(), 0.0f);
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        scratch = height;
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                const std::size_t column = columnIndex(x, z, nx);
                if (valid[column] == 0) {
                    continue;
                }

                float weighted_sum = height[column] * 4.0f;
                float total_weight = 4.0f;
                static constexpr int kOffsets[4][2] = {
                    {-1, 0}, {1, 0}, {0, -1}, {0, 1}
                };
                for (const auto& offset : kOffsets) {
                    const int neighbor_x = x + offset[0];
                    const int neighbor_z = z + offset[1];
                    if (neighbor_x < 0 || neighbor_x >= nx ||
                        neighbor_z < 0 || neighbor_z >= nz) {
                        continue;
                    }
                    const std::size_t neighbor = columnIndex(neighbor_x, neighbor_z, nx);
                    if (valid[neighbor] != 0) {
                        weighted_sum += height[neighbor];
                        total_weight += 1.0f;
                    }
                }
                scratch[column] = weighted_sum / total_weight;
            }
        }
        height.swap(scratch);
    }
}

} // namespace

void GranularVirtualRepresentation::clear() {
    bounds_min = Vec3(0.0f);
    bounds_max = Vec3(0.0f);
    resolution_x = 0;
    resolution_z = 0;
    cell_size = 0.0f;
    height.clear();
    column_valid.clear();
    particle_class.clear();
    surface_grain_indices.clear();
    detached_grain_indices.clear();
    stats = {};
}

bool buildGranularVirtualRepresentation(
    const FluidParticles& particles,
    const FluidSim::FluidGrid& grid,
    const GranularVirtualSettings& settings,
    GranularVirtualRepresentation& output,
    std::string& error) {
    output.clear();
    error.clear();
    const auto started = std::chrono::steady_clock::now();

    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        !std::isfinite(grid.voxel_size) || grid.voxel_size <= 0.0f) {
        error = "Granular virtual representation requires a valid solver grid.";
        return false;
    }
    if (settings.min_particles_per_cell == 0) {
        error = "min_particles_per_cell must be greater than zero.";
        return false;
    }
    if (particles.position.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        error = "Granular particle count exceeds the 32-bit compact-index contract.";
        return false;
    }

    const std::size_t nx = static_cast<std::size_t>(grid.nx);
    const std::size_t ny = static_cast<std::size_t>(grid.ny);
    const std::size_t nz = static_cast<std::size_t>(grid.nz);
    if (nx > std::numeric_limits<std::size_t>::max() / ny ||
        nx * ny > std::numeric_limits<std::size_t>::max() / nz) {
        error = "Granular solver grid dimensions overflow addressable memory.";
        return false;
    }
    const std::size_t cell_count = nx * ny * nz;
    const std::size_t column_count = nx * nz;
    if (cell_count >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        error = "Granular solver grid exceeds the 32-bit compact-cell contract.";
        return false;
    }

    std::vector<std::uint32_t> occupancy(cell_count, 0);
    for (const Vec3& position : particles.position) {
        int x = 0;
        int y = 0;
        int z = 0;
        if (worldToCell(position, grid, x, y, z)) {
            ++occupancy[grid.cellIndex(x, y, z)];
        }
    }

    std::vector<std::uint8_t> supported(cell_count, 0);
    std::deque<std::uint32_t> frontier;
    const int support_gap = static_cast<int>(std::min<std::uint32_t>(
        settings.support_gap_cells,
        static_cast<std::uint32_t>(std::max(0, grid.ny - 1))));

    for (int z = 0; z < grid.nz; ++z) {
        for (int y = 0; y < grid.ny; ++y) {
            for (int x = 0; x < grid.nx; ++x) {
                const std::size_t cell = grid.cellIndex(x, y, z);
                if (occupancy[cell] < settings.min_particles_per_cell) {
                    continue;
                }
                if (y <= support_gap || touchesCollider(x, y, z, grid)) {
                    supported[cell] = 1;
                    frontier.push_back(static_cast<std::uint32_t>(cell));
                }
            }
        }
    }

    static constexpr int kOffsets[6][3] = {
        {-1, 0, 0}, {1, 0, 0},
        {0, -1, 0}, {0, 1, 0},
        {0, 0, -1}, {0, 0, 1}
    };
    while (!frontier.empty()) {
        const std::size_t cell = frontier.front();
        frontier.pop_front();
        const int z = static_cast<int>(cell / (nx * ny));
        const std::size_t plane_remainder = cell % (nx * ny);
        const int y = static_cast<int>(plane_remainder / nx);
        const int x = static_cast<int>(plane_remainder % nx);

        for (const auto& offset : kOffsets) {
            const int neighbor_x = x + offset[0];
            const int neighbor_y = y + offset[1];
            const int neighbor_z = z + offset[2];
            if (neighbor_x < 0 || neighbor_x >= grid.nx ||
                neighbor_y < 0 || neighbor_y >= grid.ny ||
                neighbor_z < 0 || neighbor_z >= grid.nz) {
                continue;
            }
            const std::size_t neighbor = grid.cellIndex(
                neighbor_x,
                neighbor_y,
                neighbor_z);
            if (supported[neighbor] != 0 ||
                occupancy[neighbor] < settings.min_particles_per_cell) {
                continue;
            }
            supported[neighbor] = 1;
            frontier.push_back(static_cast<std::uint32_t>(neighbor));
        }
    }

    const float missing_height = std::numeric_limits<float>::lowest();
    std::vector<float> column_peak(column_count, missing_height);
    output.particle_class.assign(
        particles.position.size(),
        GranularParticleClass::DetachedGrain);

    for (std::size_t particle_index = 0;
         particle_index < particles.position.size();
         ++particle_index) {
        const Vec3& position = particles.position[particle_index];
        int x = 0;
        int y = 0;
        int z = 0;
        if (!worldToCell(position, grid, x, y, z)) {
            continue;
        }
        if (supported[grid.cellIndex(x, y, z)] == 0) {
            continue;
        }
        const std::size_t column = columnIndex(x, z, grid.nx);
        column_peak[column] = std::max(column_peak[column], position.y);
    }

    output.bounds_min = grid.origin;
    output.bounds_max = grid.origin + Vec3(
        static_cast<float>(grid.nx) * grid.voxel_size,
        static_cast<float>(grid.ny) * grid.voxel_size,
        static_cast<float>(grid.nz) * grid.voxel_size);
    output.resolution_x = grid.nx;
    output.resolution_z = grid.nz;
    output.cell_size = grid.voxel_size;
    output.height.resize(column_count, grid.origin.y);
    output.column_valid.resize(column_count, 0);
    for (std::size_t column = 0; column < column_count; ++column) {
        if (column_peak[column] == missing_height) {
            continue;
        }
        output.column_valid[column] = 1;
        output.height[column] = column_peak[column] +
                                settings.surface_offset_voxels * grid.voxel_size;
        ++output.stats.surface_columns;
    }

    const float surface_band =
        static_cast<float>(settings.surface_band_cells) * grid.voxel_size;
    output.surface_grain_indices.reserve(particles.position.size() / 8);
    output.detached_grain_indices.reserve(particles.position.size() / 16);
    for (std::size_t particle_index = 0;
         particle_index < particles.position.size();
         ++particle_index) {
        const Vec3& position = particles.position[particle_index];
        int x = 0;
        int y = 0;
        int z = 0;
        const bool in_grid = worldToCell(position, grid, x, y, z);
        const bool is_supported = in_grid &&
            supported[grid.cellIndex(x, y, z)] != 0;
        if (!is_supported) {
            output.detached_grain_indices.push_back(
                static_cast<std::uint32_t>(particle_index));
            continue;
        }

        ++output.stats.supported_particles;
        const std::size_t column = columnIndex(x, z, grid.nx);
        if (position.y >= column_peak[column] - surface_band) {
            output.particle_class[particle_index] = GranularParticleClass::SurfaceGrain;
            output.surface_grain_indices.push_back(
                static_cast<std::uint32_t>(particle_index));
        } else {
            output.particle_class[particle_index] = GranularParticleClass::HiddenBulk;
        }
    }

    smoothHeightField(
        output.height,
        output.column_valid,
        grid.nx,
        grid.nz,
        settings.height_smoothing_iterations);

    output.stats.simulation_particles = particles.position.size();
    output.stats.surface_grains = output.surface_grain_indices.size();
    output.stats.detached_grains = output.detached_grain_indices.size();
    output.stats.hidden_bulk_particles =
        output.stats.simulation_particles -
        output.stats.surface_grains -
        output.stats.detached_grains;
    output.stats.supported_cells = static_cast<std::size_t>(std::count(
        supported.begin(),
        supported.end(),
        static_cast<std::uint8_t>(1)));
    output.stats.measured = true;
    output.stats.build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return true;
}

} // namespace RayTrophiSim::Fluid
