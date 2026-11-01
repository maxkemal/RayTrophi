#pragma once

#include "Fluid/APICFluidSolver.h"
#include "Fluid/MatterSubstanceState.h"
#include "FluidGrid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace RayTrophiSim::Fluid {

// Reuse the pressure operator with zero input velocity to obtain a positional
// density correction. Its field never enters G2P, FLIP, affine state or momentum.
// The callback runs the same backend as the physical pressure solve.
template <class Solve>
bool projectLiquidDensityPositions(FluidParticles& particles, FluidSim::FluidGrid& grid,
                                   const APICSolverParams& params, float dt, Solve solve) {
    if (!(params.density_relaxation_s > 0.0f) || !(dt > 0.0f) ||
        !(grid.voxel_size > 0.0f) || particles.empty()) {
        return true;
    }
    const auto cells = grid.getCellCount();
    std::vector<float> mask(cells, 0.0f);
    const auto cellAt = [&](const Vec3& position, std::size_t& cell) {
        const Vec3 local = (position - grid.origin) * (1.0f / grid.voxel_size);
        const int i = static_cast<int>(std::floor(local.x));
        const int j = static_cast<int>(std::floor(local.y));
        const int k = static_cast<int>(std::floor(local.z));
        if (i < 0 || j < 0 || k < 0 || i >= grid.nx || j >= grid.ny || k >= grid.nz) {
            return false;
        }
        cell = grid.cellIndex(i, j, k);
        return true;
    };
    std::vector<std::size_t> mobile;
    mobile.reserve(particles.size());
    for (std::size_t p = 0; p < particles.size(); ++p) {
        const auto model = p < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[p])
            : MatterConstitutiveModel::Auto;
        if (model != MatterConstitutiveModel::Fluid &&
            !(model == MatterConstitutiveModel::Auto && !params.granular_enabled)) {
            continue;
        }
        if (isMatterObstacle(particles, p,
                params.solid_substance_tags ? params.solid_substance_tags->data() : nullptr,
                params.solid_substance_tags ? params.solid_substance_tags->size() : 0)) {
            continue;
        }
        std::size_t cell = 0;
        if (cellAt(particles.position[p], cell)) {
            mask[cell] += 1.0f;
            mobile.push_back(p);
        }
    }
    if (mobile.empty()) {
        return true;
    }
    for (std::size_t c = 0; c < cells; ++c) {
        if (c < grid.solid.size() && grid.solid[c]) {
            mask[c] = -1.0f;
        }
    }

    // Restore the physical grid even if allocation or a backend callback throws.
    struct GridTransaction {
        FluidSim::FluidGrid& grid;
        std::array<std::vector<float>, 5> saved;
        std::vector<Vec3> solid_velocity;

        explicit GridTransaction(FluidSim::FluidGrid& value) : grid(value) {
            saved[0].swap(grid.vel_x);
            saved[1].swap(grid.vel_y);
            saved[2].swap(grid.vel_z);
            saved[3].swap(grid.pressure);
            saved[4].swap(grid.divergence);
            solid_velocity.swap(grid.solid_vel);
        }
        ~GridTransaction() {
            saved[0].swap(grid.vel_x);
            saved[1].swap(grid.vel_y);
            saved[2].swap(grid.vel_z);
            saved[3].swap(grid.pressure);
            saved[4].swap(grid.divergence);
            solid_velocity.swap(grid.solid_vel);
        }
    } transaction(grid);
    grid.vel_x.assign(transaction.saved[0].size(), 0.0f);
    grid.vel_y.assign(transaction.saved[1].size(), 0.0f);
    grid.vel_z.assign(transaction.saved[2].size(), 0.0f);
    grid.pressure.assign(cells, 0.0f);
    grid.divergence.assign(cells, 0.0f);
    grid.solid_vel.assign(transaction.solid_velocity.size(), Vec3(0.0f));
    APICSolverStats density_stats;
    if (!solve(mask, mobile, density_stats)) {
        return false;
    }
    std::vector<Vec3> corrected;
    corrected.reserve(mobile.size());
    const float limit = 0.25f * grid.voxel_size;
    Vec3 low, high;
    grid.getWorldBounds(low, high);
    const Vec3 pad(0.01f * grid.voxel_size);
    for (const auto p : mobile) {
        Vec3 shift = grid.sampleVelocity(particles.position[p]) * dt;
        const float length = shift.length();
        if (!std::isfinite(length)) {
            return false;
        }
        if (length > limit) {
            shift = shift * (limit / length);
        }
        Vec3 target = particles.position[p] + shift;
        if (params.boundary == APICSolverParams::BoundaryMode::Closed) {
            target = Vec3::max(low + pad, Vec3::min(high - pad, target));
        }
        std::size_t cell = 0;
        if (!cellAt(target, cell) || mask[cell] < 0.0f) {
            target = particles.position[p];
        }
        corrected.push_back(target);
    }
    for (std::size_t i = 0; i < mobile.size(); ++i) {
        particles.position[mobile[i]] = corrected[i];
    }
    return true;
}

// The CPU reference pressure operator, also used by the position-only pass.
std::size_t projectPressureFreeSurface(const FluidParticles& particles,
    FluidSim::FluidGrid& grid, const APICSolverParams& params, float dt,
    std::size_t& sealed_pockets, std::size_t& sealed_cells, std::size_t& interior_cells,
    APICSolverStats& stats);

} // namespace RayTrophiSim::Fluid
