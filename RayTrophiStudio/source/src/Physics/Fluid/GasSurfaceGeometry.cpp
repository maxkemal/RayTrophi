#include "Fluid/GasSurfaceGeometry.h"
#include "ParticleSimulation.h"
#include "Fluid/MatterSubstanceState.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace RayTrophiSim::Fluid {
namespace {
constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();
constexpr int offsets[6][3] = {
    {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}, {0, 0, -1}, {0, 0, 1}, {0, -1, 0}
};
}
GasSurfaceGeometry::GasSurfaceGeometry(const SimulationGridDomainState& liquid,
                                       const FluidSim::FluidGrid& gas, bool periodic,
                                       const APICSolverParams* source_params)
    : grid_(gas), particle_cells_(liquid.particles.size(), invalid), periodic_(periodic) {
    if (!(gas.voxel_size > 0.0f) || gas.nx <= 0 || gas.ny <= 0 || gas.nz <= 0 ||
        gas.getCellCount() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return;
    }
    occupied_.reserve(liquid.particles.size());
    const auto& particles = liquid.particles;
    for (std::size_t p = 0; p < particles.size(); ++p) {
        if (p < particles.mass_fraction.size() &&
            !(particles.mass_fraction[p] > 0.02f)) {
            continue;
        }
        const Vec3 local = (particles.position[p] - gas.origin) / gas.voxel_size;
        if (!std::isfinite(local.x) || !std::isfinite(local.y) || !std::isfinite(local.z) ||
            local.x < 0 || local.x >= gas.nx || local.y < 0 || local.y >= gas.ny ||
            local.z < 0 || local.z >= gas.nz) {
            continue;
        }
        const uint32_t cell = static_cast<uint32_t>(gas.cellIndex(
            static_cast<int>(local.x), static_cast<int>(local.y), static_cast<int>(local.z)));
        particle_cells_[p] = cell;
        ++occupied_[cell];
        const bool frozen = p < particles.flags.size() &&
            (particles.flags[p] & kParticleFlagFrozen);
        auto model = p < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[p])
            : MatterConstitutiveModel::Auto;
        if (model == MatterConstitutiveModel::Auto && source_params) {
            const uint32_t tag = p < particles.substance_tag.size() ? particles.substance_tag[p] : 0u;
            const float kelvin = p < particles.temperature.size() ? particles.temperature[p] : 0.0f;
            model = substanceBirthModel(tag, *source_params, kelvin);
        }
        const bool fluid = model == MatterConstitutiveModel::Fluid ||
            model == MatterConstitutiveModel::Auto;
        if (!frozen && fluid) liquid_cells_.insert(cell);
    }
}

bool GasSurfaceGeometry::particleCell(std::size_t particle, uint32_t& cell) const {
    if (particle >= particle_cells_.size() || particle_cells_[particle] == invalid) {
        return false;
    }
    cell = particle_cells_[particle];
    return true;
}

bool GasSurfaceGeometry::neighbour(uint32_t cell, int face, uint32_t& result) const {
    int x = static_cast<int>(cell % grid_.nx) + offsets[face][0];
    int y = static_cast<int>((cell / grid_.nx) % grid_.ny) + offsets[face][1];
    int z = static_cast<int>(cell / (grid_.nx * grid_.ny)) + offsets[face][2];
    if (periodic_) {
        x = (x + grid_.nx) % grid_.nx;
        y = (y + grid_.ny) % grid_.ny;
        z = (z + grid_.nz) % grid_.nz;
    }
    if (x < 0 || x >= grid_.nx || y < 0 || y >= grid_.ny || z < 0 || z >= grid_.nz) {
        return false;
    }
    result = static_cast<uint32_t>(grid_.cellIndex(x, y, z));
    return true;
}

GasSurfaceFaces GasSurfaceGeometry::exposedFaces(uint32_t cell) const {
    GasSurfaceFaces result;
    for (int face = 0; face < 6; ++face) {
        uint32_t air;
        if (!neighbour(cell, face, air) || occupied_.find(air) != occupied_.end() ||
            (grid_.solid.size() == grid_.getCellCount() && grid_.solid[air] != 0)) {
            continue;
        }
        result.air[result.count++] = air;
    }
    return result;
}

uint32_t GasSurfaceGeometry::parcelsInCell(uint32_t cell) const {
    const auto found = occupied_.find(cell);
    return found == occupied_.end() ? 0u : found->second;
}

bool GasSurfaceGeometry::nearestAir(uint32_t cell, uint32_t& air) {
    if (!release_built_) {
        // A multi-source shortest path through liquid cells. Buried boiling
        // vents at its nearest accessible surface, never at an unrelated jet top.
        std::vector<uint32_t> cells;
        cells.reserve(liquid_cells_.size());
        for (const uint32_t cell : liquid_cells_) {
            cells.push_back(cell);
        }
        std::sort(cells.begin(), cells.end());
        std::vector<uint32_t> queue;
        queue.reserve(cells.size());
        for (const uint32_t occupied : cells) {
            const auto faces = exposedFaces(occupied);
            if (faces.count) {
                release_[occupied] = faces.air[0];
                queue.push_back(occupied);
            }
        }
        for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
            const uint32_t source = queue[cursor];
            for (int face = 0; face < 6; ++face) {
                uint32_t next;
                if (neighbour(source, face, next) && liquid_cells_.find(next) != liquid_cells_.end() &&
                    release_.find(next) == release_.end()) {
                    release_[next] = release_.at(source);
                    queue.push_back(next);
                }
            }
        }
        release_built_ = true;
    }
    const auto found = release_.find(cell);
    if (found == release_.end()) {
        return false; // Sealed liquid: no gas destination, so latent is not paid yet.
    }
    air = found->second;
    return true;
}
}
