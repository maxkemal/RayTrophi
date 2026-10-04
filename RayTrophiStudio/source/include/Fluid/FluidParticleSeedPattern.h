#pragma once

#include <cstdint>
#include <vector>

namespace RayTrophiSim::Fluid {

struct FluidSeedSlot {
    int x = 0;
    int y = 0;
    int z = 0;
};

struct FluidSeedPattern {
    int subdivisions = 1;
    std::vector<FluidSeedSlot> slots;
};

FluidSeedPattern buildFluidSeedPattern(int particles_per_cell);

FluidSeedSlot orientFluidSeedSlot(const FluidSeedSlot& slot,
                                  int subdivisions,
                                  int cell_x,
                                  int cell_y,
                                  int cell_z,
                                  std::uint32_t seed);

} // namespace RayTrophiSim::Fluid
