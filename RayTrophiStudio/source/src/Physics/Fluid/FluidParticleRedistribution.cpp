#include "Fluid/FluidParticleRedistribution.h"
#include "Fluid/APICFluidSolver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace RayTrophiSim::Fluid {

std::size_t redistributeFluidParticles(FluidParticles& particles,
                                      const FluidSim::FluidGrid& grid,
                                      const APICSolverParams& params,
                                      uint32_t step_seed) {
    if (!params.reseed_enabled || params.granular_enabled || particles.empty() ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        !(grid.voxel_size > 0.0f) || !std::isfinite(grid.voxel_size)) {
        return 0;
    }
    const int target = std::max(1, params.reseed_target_per_cell > 0
        ? params.reseed_target_per_cell : params.particles_per_cell);
    const int minimum = std::clamp(params.reseed_min_per_cell, 1, target);
    const int maximum = std::max(target + 1, params.reseed_max_per_cell);
    const std::size_t cells = static_cast<std::size_t>(grid.nx) * grid.ny * grid.nz;
    if (grid.solid.size() < cells) {
        return 0;
    }
    constexpr std::size_t none = std::numeric_limits<std::size_t>::max();
    // Reuse allocations per simulation thread, including when domains alternate.
    static thread_local std::vector<int> counts;
    static thread_local std::vector<std::size_t> heads;
    static thread_local std::vector<std::size_t> next;
    counts.assign(cells, 0);
    heads.assign(cells, none);
    next.assign(particles.size(), none);

    const auto tagOf = [&](std::size_t p) {
        return p < particles.substance_tag.size() ? particles.substance_tag[p] : 0u;
    };
    const auto movable = [&](std::size_t p) {
        if (p < particles.flags.size() && (particles.flags[p] & kParticleFlagFrozen)) {
            return false;
        }
        const uint32_t tag = tagOf(p);
        return tag == 0u || !params.solid_substance_tags ||
            std::find(params.solid_substance_tags->begin(),
                      params.solid_substance_tags->end(), tag) ==
                params.solid_substance_tags->end();
    };
    const auto inside = [&](int x, int y, int z) {
        return x >= 0 && x < grid.nx && y >= 0 && y < grid.ny &&
            z >= 0 && z < grid.nz;
    };
    const float invH = 1.0f / grid.voxel_size;
    for (std::size_t p = 0; p < particles.size(); ++p) {
        const Vec3 local = (particles.position[p] - grid.origin) * invH;
        // Check float bounds before converting NaNs or extreme values to int.
        if (!(local.x >= 0 && local.x < grid.nx &&
              local.y >= 0 && local.y < grid.ny &&
              local.z >= 0 && local.z < grid.nz)) {
            continue;
        }
        const auto c = grid.cellIndex(static_cast<int>(local.x),
                                      static_cast<int>(local.y),
                                      static_cast<int>(local.z));
        if (grid.solid[c] || !movable(p)) {
            continue;
        }
        ++counts[c];
        next[p] = heads[c];
        heads[c] = p;
    }

    constexpr int faces[6][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
        {0, 1, 0}, {0, 0, -1}, {0, 0, 1}
    };
    std::mt19937 rng(step_seed ^ 0x9E3779B9u);
    std::size_t moved = 0;
    for (int z = 0; z < grid.nz; ++z) {
        for (int y = 0; y < grid.ny; ++y) {
            for (int x = 0; x < grid.nx; ++x) {
                const auto c = grid.cellIndex(x, y, z);
                if (grid.solid[c] || counts[c] == 0 || counts[c] >= minimum) {
                    continue;
                }
                bool surface = false;
                for (const auto& face : faces) {
                    const int nx = x + face[0], ny = y + face[1], nz = z + face[2];
                    if (!inside(nx, ny, nz)) {
                        continue;
                    }
                    const auto neighbor = grid.cellIndex(nx, ny, nz);
                    if (!grid.solid[neighbor] && counts[neighbor] == 0) {
                        surface = true;
                        break;
                    }
                }
                if (surface) {
                    continue;
                }
                const uint32_t tag = tagOf(heads[c]);
                // Only borrow matching parcels across a shared fluid face.
                // No global relocation across disconnected pools or solids.
                for (const auto& face : faces) {
                    const int nx = x + face[0], ny = y + face[1], nz = z + face[2];
                    if (!inside(nx, ny, nz)) {
                        continue;
                    }
                    const auto donor = grid.cellIndex(nx, ny, nz);
                    if (grid.solid[donor] || counts[donor] <= maximum) {
                        continue;
                    }
                    std::size_t* link = &heads[donor];
                    while (*link != none && counts[c] < target && counts[donor] > maximum) {
                        const std::size_t p = *link;
                        if (tagOf(p) != tag) {
                            link = &next[p];
                            continue;
                        }
                        *link = next[p];
                        next[p] = heads[c];
                        heads[c] = p;
                        // Explicit RNG conversion, strictly inside the target cell.
                        const auto jitter = [&]() {
                            return (static_cast<float>(rng() >> 9) + 0.5f) / 8388608.0f;
                        };
                        const float jx = jitter(), jy = jitter(), jz = jitter();
                        particles.position[p] = grid.origin +
                            Vec3(x + jx, y + jy, z + jz) * grid.voxel_size;
                        // Velocity, affine, both material coordinates, chemistry,
                        // labels and all other SoA fields remain with the parcel.
                        --counts[donor];
                        ++counts[c];
                        ++moved;
                    }
                    if (counts[c] >= target) {
                        break;
                    }
                }
            }
        }
    }
    return moved;
}

} // namespace RayTrophiSim::Fluid
