#include "Fluid/MatterGrainBirth.h"
#include "Fluid/APICFluidSolver.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/SubstanceTag.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace RayTrophiSim::Fluid {

std::size_t MatterGrainBirthFilter::Hash::operator()(const Cell& c) const {
    return (uint32_t(c.x) * 73856093u) ^ (uint32_t(c.y) * 19349663u) ^
        (uint32_t(c.z) * 83492791u);
}

MatterGrainBirthFilter::Cell MatterGrainBirthFilter::cell(const Vec3& p) const {
    return {static_cast<int>(std::floor((p.x - low_.x) / spacing_)),
        static_cast<int>(std::floor((p.y - low_.y) / spacing_)),
        static_cast<int>(std::floor((p.z - low_.z) / spacing_))};
}

MatterGrainBirthFilter::MatterGrainBirthFilter(const FluidParticles& particles,
    const FluidSim::FluidGrid& grid, float radius, const APICSolverParams* params)
    : radius_(radius), spacing_(2.0002f * radius) {
    if (params && radius > 0.0f) {
        spacing_ = 2.0002f * matterGrainMaximumRadius(params->grain);
    }
    grid.getWorldBounds(low_, high_);
    if (radius_ <= 0.0f) {
        return;
    }
    // Grains space against grains only. Liquid parcels are not rigid: a grain
    // born inside water is a valid state, and parcels are far denser than
    // grains, so counting them would starve a source poured into a pool.
    for (std::size_t i = 0; i < particles.position.size(); ++i) {
        if (i >= particles.constitutive_model.size() ||
            particles.constitutive_model[i] !=
                static_cast<uint8_t>(MatterConstitutiveModel::Granular)) {
            continue;
        }
        if (i < particles.mass_fraction.size() && !(particles.mass_fraction[i] > 0.02f)) {
            continue;
        }
        const uint32_t tag = i < particles.substance_tag.size()
            ? particles.substance_tag[i] : kSubstanceUntagged;
        if (params && substanceTransportOwner(tag, MatterConstitutiveModel::Granular, *params) !=
            MatterTransportOwner::Grain) {
            continue;
        }
        const Vec3& p = particles.position[i];
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
            p.x >= low_.x - spacing_ && p.x <= high_.x + spacing_ &&
            p.y >= low_.y - spacing_ && p.y <= high_.y + spacing_ &&
            p.z >= low_.z - spacing_ && p.z <= high_.z + spacing_) {
            const float existing_radius = params
                ? matterGrainRadius(params->grain, particles, i) : radius_;
            cells_[cell(p)].push_back({p, existing_radius});
        }
    }
}

bool MatterGrainBirthFilter::accept(const Vec3& p, float radius) {
    if (radius <= 0.0f) {
        radius = radius_;
    }
    if (radius_ <= 0.0f) {
        return true;
    }
    if (!std::isfinite(radius) || radius * 2.0002f > spacing_ ||
        !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        p.x < low_.x + radius || p.x > high_.x - radius ||
        p.y < low_.y + radius || p.y > high_.y - radius ||
        p.z < low_.z + radius || p.z > high_.z - radius) {
        return false;
    }
    const auto own = cell(p);
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                const auto found = cells_.find({own.x + x, own.y + y, own.z + z});
                if (found == cells_.end()) {
                    continue;
                }
                for (const auto& other : found->second) {
                    const auto d = p - other.position;
                    if (d.x * d.x + d.y * d.y + d.z * d.z < (1.0001f * (radius + other.radius)) *
                        (1.0001f * (radius + other.radius))) {
                        return false;
                    }
                }
            }
        }
    }
    cells_[own].push_back({p, radius});
    return true;
}

MatterGrainSeedResult seedMatterGrainBox(FluidParticles& particles,
    const FluidSim::FluidGrid& grid, const Vec3& low, const Vec3& high,
    const APICSolverParams& params, float kelvin, std::size_t budget) {
    MatterGrainSeedResult result;
    Vec3 grid_low, grid_high;
    grid.getWorldBounds(grid_low, grid_high);
    const Vec3 lo = Vec3::max(Vec3::min(low, high), grid_low);
    const Vec3 hi = Vec3::min(Vec3::max(low, high), grid_high);
    const float radius = matterGrainMaximumRadius(params.grain);
    const double spacing = 2.2002 * radius;
    const double jitter = 0.1 * radius;
    if (!(radius > 0.0f) || !std::isfinite(spacing) || !(grid.voxel_size > 0.0f)) {
        return result;
    }
    // A perfect cubic lattice forms vertical force chains with no lateral
    // force or spin under gravity. Break that symmetry at birth, not by adding
    // artificial solver forces. Adjacent centers remain at least 2.0002 R
    // apart along their separating axis, including both jitter extremes.
    // Coordinates depend on seed/candidate, never particle ID or voxel/PPC.
    auto noise = [](uint64_t key) -> double {
        key += 0x9e3779b97f4a7c15ull;
        key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ull;
        key = (key ^ (key >> 27)) * 0x94d049bb133111ebull;
        key ^= key >> 31;
        return double(key >> 40) / 16777215.0 * 2.0 - 1.0;
    };
    auto extent = [&](float a, float b) -> std::size_t {
        const double span = double(b) - a - 2.0 * (radius + jitter);
        if (!std::isfinite(span) || span < 0.0) {
            return 0;
        }
        return static_cast<std::size_t>(std::min(1000000.0, std::floor(span / spacing) + 1));
    };
    const std::size_t nx = extent(lo.x, hi.x);
    const std::size_t ny = extent(lo.y, hi.y);
    const std::size_t nz = extent(lo.z, hi.z);
    MatterGrainBirthFilter filter(particles, grid, params.grain.radius_m, &params);
    particles.ensureParticleIdentities();
    // Bound authoring work even for a huge box filled with solid cells.
    const std::size_t limit = std::max<std::size_t>(1000000,
        std::min<std::size_t>(budget, 1000000) * 64);
    std::size_t visited = 0;
    for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t z = 0; z < nz; ++z) {
            for (std::size_t x = 0; x < nx; ++x) {
                if (++visited > limit) {
                    result.truncated = true;
                    return result;
                }
                const uint64_t key = (uint64_t(params.grain.size_seed) << 32) ^
                    uint64_t(visited);
                const Vec3 p(
                    float(lo.x + radius + jitter + x * spacing + jitter * noise(key)),
                    float(lo.y + radius + jitter + y * spacing +
                        jitter * noise(key ^ 0xd1b54a32d192ed03ull)),
                    float(lo.z + radius + jitter + z * spacing +
                        jitter * noise(key ^ 0x94d049bb133111ebull)));
                const int cx = static_cast<int>((p.x - grid.origin.x) / grid.voxel_size);
                const int cy = static_cast<int>((p.y - grid.origin.y) / grid.voxel_size);
                const int cz = static_cast<int>((p.z - grid.origin.z) / grid.voxel_size);
                if (cx < 0 || cy < 0 || cz < 0 || cx >= grid.nx || cy >= grid.ny ||
                    cz >= grid.nz) {
                    continue;
                }
                const auto cell = grid.cellIndex(cx, cy, cz);
                if (cell < grid.solid.size() && grid.solid[cell]) {
                    continue;
                }
                auto grain = params.grain;
                grain.radius_m = matterGrainRadius(params.grain, particles.next_particle_id);
                if (!filter.accept(p, grain.radius_m)) {
                    continue;
                }
                if (result.emitted == budget) {
                    result.truncated = true;
                    return result;
                }
                const float mass = matterGrainRestMassKg(grain, kSubstanceUntagged,
                    particleSubstance(kSubstanceUntagged, params),
                    MatterConstitutiveModel::Granular, false);
                particles.emit(p, Vec3(0), kelvin, 0.0f, kSubstanceUntagged,
                    nullptr, nullptr, mass, MatterConstitutiveModel::Granular);
                initMatterGrainBirthWater(particles, particles.size() - 1, 0.0f, params.grain);
                ++result.emitted;
            }
        }
    }
    return result;
}

} // namespace RayTrophiSim::Fluid
