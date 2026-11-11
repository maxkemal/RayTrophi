#pragma once

#include "Fluid/DomainWallSettings.h"

namespace RayTrophiSim::Fluid {

inline void applyDomainWallGridContact(FluidSim::FluidGrid& grid,
    const APICSolverParams& params, float dt) {
    const auto wall = domainWallContact(params, grid.voxel_size);
    if (!wall.closed) {
        return;
    }
    const std::vector<float> snapshot[] = {grid.vel_x, grid.vel_y, grid.vel_z};
    const int size[] = {grid.nx, grid.ny, grid.nz};
    const auto sample = [&](int axis, const int q[3]) {
        const int i = std::clamp(q[0], 0, grid.nx - (axis != 0));
        const int j = std::clamp(q[1], 0, grid.ny - (axis != 1));
        const int k = std::clamp(q[2], 0, grid.nz - (axis != 2));
        const auto index = axis == 0 ? grid.velXIndex(i, j, k)
            : axis == 1 ? grid.velYIndex(i, j, k) : grid.velZIndex(i, j, k);
        return snapshot[axis][index];
    };
    for (int axis = 0; axis < 3; ++axis) {
        auto& output = axis == 0 ? grid.vel_x : (axis == 1 ? grid.vel_y : grid.vel_z);
        for (int k = 0; k < grid.nz + (axis == 2); ++k) {
            for (int j = 0; j < grid.ny + (axis == 1); ++j) {
                for (int i = 0; i < grid.nx + (axis == 0); ++i) {
                    const int face[] = {i, j, k};
                    const auto index = axis == 0 ? grid.velXIndex(i, j, k)
                        : axis == 1 ? grid.velYIndex(i, j, k) : grid.velZIndex(i, j, k);
                    output[index] = resolveDomainWallFace(snapshot[axis][index], axis,
                        face, size, wall, params.gravity, dt, sample);
                }
            }
        }
    }
}

} // namespace RayTrophiSim::Fluid
