#pragma once

#include "FluidGrid.h"
#include <array>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace RayTrophiSim {
struct SimulationGridDomainState;
namespace Fluid {
struct APICSolverParams;
struct GasSurfaceFaces {
    std::array<uint32_t, 6> air{};
    uint32_t count = 0;
};

// Sparse occupied cells; no full gas-grid mirror or per-face Triangle geometry.
class GasSurfaceGeometry {
public:
    GasSurfaceGeometry(const SimulationGridDomainState& liquid,
                       const FluidSim::FluidGrid& gas, bool periodic = false,
                       const APICSolverParams* source_params = nullptr);
    bool particleCell(std::size_t particle, uint32_t& cell) const;
    GasSurfaceFaces exposedFaces(uint32_t cell) const;
    uint32_t parcelsInCell(uint32_t cell) const;
    bool nearestAir(uint32_t cell, uint32_t& air);

private:
    const FluidSim::FluidGrid& grid_;
    std::vector<uint32_t> particle_cells_;
    std::unordered_map<uint32_t, uint32_t> occupied_;
    std::unordered_set<uint32_t> liquid_cells_;
    std::unordered_map<uint32_t, uint32_t> release_;
    bool release_built_ = false;
    bool periodic_ = false;
    bool neighbour(uint32_t cell, int face, uint32_t& result) const;
};
}
}
