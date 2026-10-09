#include "Fluid/MatterGrainBirth.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/SubstanceTag.h"

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
            cells_[cell(p)].push_back(p);
        }
    }
}

bool MatterGrainBirthFilter::accept(const Vec3& p) {
    if (radius_ <= 0.0f) {
        return true;
    }
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        p.x < low_.x + radius_ || p.x > high_.x - radius_ ||
        p.y < low_.y + radius_ || p.y > high_.y - radius_ ||
        p.z < low_.z + radius_ || p.z > high_.z - radius_) {
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
                    const auto d = p - other;
                    if (d.x * d.x + d.y * d.y + d.z * d.z < spacing_ * spacing_) {
                        return false;
                    }
                }
            }
        }
    }
    cells_[own].push_back(p);
    return true;
}

} // namespace RayTrophiSim::Fluid
