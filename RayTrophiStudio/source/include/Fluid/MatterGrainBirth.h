#pragma once

#include "FluidParticles.h"
#include "../FluidGrid.h"

#include <unordered_map>
#include <vector>

namespace RayTrophiSim::Fluid {

// Host authoring/birth boundary only. Existing simulation state stays unchanged;
// accepted new centers occupy the hash immediately, before subsequent births.
class MatterGrainBirthFilter {
public:
    MatterGrainBirthFilter(const FluidParticles& particles,
                          const FluidSim::FluidGrid& grid, float radius);
    bool accept(const Vec3& position);

private:
    struct Cell {
        int x, y, z;
        bool operator==(const Cell& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    struct Hash {
        std::size_t operator()(const Cell& cell) const;
    };
    Cell cell(const Vec3& position) const;
    Vec3 low_, high_;
    float radius_ = 0.0f;
    float spacing_ = 0.0f;
    std::unordered_map<Cell, std::vector<Vec3>, Hash> cells_;
};

} // namespace RayTrophiSim::Fluid
