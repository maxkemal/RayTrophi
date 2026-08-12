#include "Fluid/FluidParticleSolidRecovery.h"

#include "Fluid/FluidParticles.h"
#include "FluidGrid.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace RayTrophiSim::Fluid {

std::size_t recoverParticlesFromSolidCells(FluidParticles& particles,
                                           const FluidSim::FluidGrid& grid,
                                           int maximum_search_radius) {
    if (particles.empty() || grid.solid.empty() || !(grid.voxel_size > 0.0f) ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) return 0u;

    const float inv_h = 1.0f / grid.voxel_size;
    const int search_limit = std::clamp(maximum_search_radius, 1, 12);
    std::size_t recovered = 0u;

    for (std::size_t pi = 0; pi < particles.position.size(); ++pi) {
        const Vec3 gp = (particles.position[pi] - grid.origin) * inv_h;
        const int ci = static_cast<int>(std::floor(gp.x));
        const int cj = static_cast<int>(std::floor(gp.y));
        const int ck = static_cast<int>(std::floor(gp.z));
        if (ci < 0 || ci >= grid.nx || cj < 0 || cj >= grid.ny ||
            ck < 0 || ck >= grid.nz) continue;
        if (grid.solid[grid.cellIndex(ci, cj, ck)] == 0u) continue;

        int best_i = -1, best_j = -1, best_k = -1;
        float best_distance2 = std::numeric_limits<float>::max();
        // ★ WALK THE SHELL, NOT THE CUBE.
        //
        // This used to iterate the full (2r+1)^3 cube at every radius and
        // `continue` past everything that was not on the shell. Summed over radii
        // that is ~60k cell tests per embedded particle at the old limit of 12 —
        // and this routine only has work to do when MANY particles are embedded,
        // i.e. it spiked hardest exactly when the sim was already misbehaving.
        // Stepping dz by 2r whenever dx and dy are both interior visits the shell
        // and nothing else, for the identical result.
        for (int radius = 1; radius <= search_limit && best_i < 0; ++radius) {
            for (int dx = -radius; dx <= radius; ++dx) {
                for (int dy = -radius; dy <= radius; ++dy) {
                    const bool rim = std::abs(dx) == radius || std::abs(dy) == radius;
                    const int dz_step = rim ? 1 : 2 * radius;
                    for (int dz = -radius; dz <= radius; dz += dz_step) {
                        const int i = ci + dx, j = cj + dy, k = ck + dz;
                        if (i < 0 || i >= grid.nx || j < 0 || j >= grid.ny ||
                            k < 0 || k >= grid.nz || grid.solid[grid.cellIndex(i, j, k)] != 0u)
                            continue;
                        // Prefer gravity-side exits when distances tie, but never
                        // choose a farther cell merely to move downward.
                        const float distance2 = static_cast<float>(dx * dx + dy * dy + dz * dz) +
                            (dy > 0 ? 0.05f : 0.0f);
                        if (distance2 < best_distance2) {
                            best_distance2 = distance2;
                            best_i = i; best_j = j; best_k = k;
                        }
                    }
                }
            }
        }
        if (best_i < 0) continue;

        const float jitter = (static_cast<float>((pi * 37u) % 101u) / 100.0f - 0.5f) *
                             grid.voxel_size * 0.12f;
        particles.position[pi] = grid.origin + Vec3(
            (static_cast<float>(best_i) + 0.5f) * grid.voxel_size + jitter,
            (static_cast<float>(best_j) + 0.5f) * grid.voxel_size,
            (static_cast<float>(best_k) + 0.5f) * grid.voxel_size - jitter);
        // Deeply embedded particles often carry a wall-projection velocity that
        // would put them straight back into the solid. Preserve direction/heat/
        // chemistry and mass, but dissipate that invalid collision impulse.
        particles.velocity[pi] = particles.velocity[pi] * 0.25f;
        ++recovered;
    }
    return recovered;
}

} // namespace RayTrophiSim::Fluid
