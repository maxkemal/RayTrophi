#pragma once

#include "Fluid/FluidParticles.h"
#include "FluidGrid.h"

#include <cmath>
#include <vector>

namespace RayTrophiSim::Fluid {

// The transfer's liquid support is fixed during a frame. Beyond it, retaining
// the post-G2P velocity matches the Vulkan tail instead of stopping a fast slab.
class LiquidAdvectionSupport {
public:
    LiquidAdvectionSupport(const FluidParticles& particles, const FluidSim::FluidGrid& grid,
                           bool enabled)
        : grid_(grid) {
        if (!enabled || !(grid.voxel_size > 0.0f)) {
            return;
        }
        occupied_.assign(grid.getCellCount(), 0);
        for (std::size_t p = 0; p < particles.size(); ++p) {
            if (p < particles.flags.size() &&
                (particles.flags[p] & kParticleFlagFrozen) != 0) {
                continue;
            }
            std::size_t cell = 0;
            if (cellAt(particles.position[p], cell)) {
                occupied_[cell] = 1;
            }
        }
    }

    template <class Sample>
    Vec3 midpoint(const Vec3& position, const Vec3& own, float dt, Sample sample) const {
        const Vec3 first = velocity(position, own, sample);
        return position + velocity(position + first * (0.5f * dt), own, sample) * dt;
    }

private:
    bool cellAt(const Vec3& position, std::size_t& cell) const {
        const Vec3 local = (position - grid_.origin) * (1.0f / grid_.voxel_size);
        const int i = static_cast<int>(std::floor(local.x));
        const int j = static_cast<int>(std::floor(local.y));
        const int k = static_cast<int>(std::floor(local.z));
        if (i < 0 || j < 0 || k < 0 || i >= grid_.nx || j >= grid_.ny || k >= grid_.nz) {
            return false;
        }
        cell = grid_.cellIndex(i, j, k);
        return true;
    }

    template <class Sample>
    Vec3 velocity(const Vec3& position, const Vec3& own, Sample sample) const {
        std::size_t cell = 0;
        if (occupied_.empty() || !cellAt(position, cell) || !occupied_[cell]) {
            return own;
        }
        return sample((position - grid_.origin) * (1.0f / grid_.voxel_size));
    }

    const FluidSim::FluidGrid& grid_;
    std::vector<unsigned char> occupied_;
};

} // namespace RayTrophiSim::Fluid
