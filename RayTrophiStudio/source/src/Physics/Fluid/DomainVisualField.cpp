#include "Fluid/DomainVisualField.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/MatterSubstanceState.h"
#include "ParticleSimulation.h"
#include "MaterialStateField.h"

namespace RayTrophiSim::Fluid {
using namespace DomainFields;

std::size_t DomainVisualField::Hash::operator()(const std::array<int, 3>& cell) const {
    uint64_t key = 0;
    for (const auto axis : cell) {
        key = visualHash(key ^ static_cast<uint32_t>(axis));
    }
    return static_cast<std::size_t>(key);
}

DomainVisualField::DomainVisualField(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState& state, const std::vector<std::size_t>& indices,
    uint64_t stream, uint64_t byte_budget) {
    const auto& grid = liquidGrid(state);
    geometry_ = {{grid.origin.x, grid.origin.y, grid.origin.z},
                 {grid.nx, grid.ny, grid.nz}, grid.voxel_size};
    if (!(grid.voxel_size > 0.0f) || grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
        throw std::runtime_error("visual field grid is unavailable");
    }
    identity_.domain_uid = domain.name + ":" + std::to_string(stream);
    identity_.state_generation = state.version;
    identity_.step = state.version;
    identity_.physical_time = state.render_physical_time;
    closed_ = domain.boundary_mode == SimulationGridDomainBoundaryMode::Closed;
    const auto& particles = state.particles;
    for (const auto index : indices) {
        const auto& p = particles.position.at(index);
        const auto& v = particles.velocity.at(index);
        const auto tag = index < particles.substance_tag.size()
            ? particles.substance_tag[index] : 0;
        const auto* profile = particleSubstance(tag, domain.fluid_params);
        const auto model = index < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[index])
            : MatterConstitutiveModel::Fluid;
        const bool solid = substanceTransportOwner(tag, model, domain.fluid_params) ==
            MatterTransportOwner::Mpm;
        identity_.lane = solid ? Lane::Mpm : Lane::Liquid;
        const double density = profile ? (solid ? profile->density : profile->liquid_density) : 0.0;
        if (!profile || !std::isfinite(density) || density <= 0.0 ||
            index >= particles.rest_mass_kg.size() ||
            !std::isfinite(particles.rest_mass_kg[index]) || particles.rest_mass_kg[index] <= 0.0) {
            throw std::runtime_error("visual volume requires valid physical mass and substance density");
        }
        const double fraction = index < particles.mass_fraction.size()
            ? particles.mass_fraction[index] : 1.0;
        if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0) {
            throw std::runtime_error("invalid physical mass fraction in visual field");
        }
        double volume = particles.rest_mass_kg[index] * fraction / density;
        if (solid) {
            if (index >= particles.granular_deformation_col0.size() ||
                index >= particles.granular_deformation_col1.size() ||
                index >= particles.granular_deformation_col2.size() ||
                index >= particles.granular_plastic_volume.size()) {
                throw std::runtime_error("MPM volume state unavailable; reset and rebake");
            }
            const auto& a = particles.granular_deformation_col0[index];
            const auto& b = particles.granular_deformation_col1[index];
            const auto& c = particles.granular_deformation_col2[index];
            const double jacobian = Vec3::dot(a, Vec3::cross(b, c)) *
                particles.granular_plastic_volume[index];
            if (!std::isfinite(jacobian) || jacobian <= 0.0) {
                throw std::runtime_error("invalid MPM deformation volume in visual field");
            }
            volume *= jacobian;
        }
        std::array<int, 3> base{};
        std::array<double, 3> fraction_cell{};
        for (int axis = 0; axis < 3; ++axis) {
            const double local = (p[axis] - geometry_.origin[axis]) / geometry_.voxel_size;
            if (!std::isfinite(local) || !std::isfinite(v[axis]) || local < 0.0 ||
                local >= geometry_.cells[axis]) {
                throw std::runtime_error("invalid/outside physical carrier in visual field");
            }
            base[axis] = static_cast<int>(std::floor(local - 0.5));
            fraction_cell[axis] = local - 0.5 - base[axis];
        }
        // Conservative trilinear scatter. Fold boundary contributions into the
        // nearest interior cell rather than losing physical volume at the wall.
        for (int corner = 0; corner < 8; ++corner) {
            std::array<int, 3> cell{};
            double weight = 1.0;
            for (int axis = 0; axis < 3; ++axis) {
                const bool upper = (corner & (1 << axis)) != 0;
                cell[axis] = std::clamp(base[axis] + (upper ? 1 : 0),
                                        0, geometry_.cells[axis] - 1);
                weight *= upper ? fraction_cell[axis] : 1.0 - fraction_cell[axis];
            }
            if (weight == 0.0 || volume == 0.0) {
                continue;
            }
            // Conservative capacity estimate includes hash nodes/buckets. The
            // service separately reserves checkpoint/particle staging memory.
            if (cells_.find(cell) == cells_.end() && cells_.size() >= byte_budget / 192) {
                throw std::length_error("visual field working budget exceeded");
            }
            auto& target = cells_[cell];
            target.volume += volume * weight;
            for (int axis = 0; axis < 3; ++axis) {
                target.momentum[axis] += volume * weight * v[axis];
            }
        }
    }
}

ChannelDescription DomainVisualField::describe(Channel channel) const {
    const auto component = static_cast<int>(channel);
    constexpr Location locations[] = {Location::FaceX, Location::FaceY, Location::FaceZ};
    if (component >= 0 && component < 3) {
        return {locations[component], Unit::LengthPerSecond, Residence::Host, 0.0, true};
    }
    return {};
}

Scalar DomainVisualField::read(Channel channel, int x, int y, int z) const {
    const int component = static_cast<int>(channel);
    if (component < 0 || component > 2) {
        return {};
    }
    std::array<int, 3> face{x, y, z};
    if (closed_ && (face[component] == 0 || face[component] == geometry_.cells[component])) {
        return {0.0, Validity::ValidData};
    }
    double volume = 0.0;
    double momentum = 0.0;
    for (int side = 0; side < 2; ++side) {
        auto cell = face;
        cell[component] -= side;
        for (int axis = 0; axis < 3; ++axis) {
            cell[axis] = std::clamp(cell[axis], 0, geometry_.cells[axis] - 1);
        }
        const auto found = cells_.find(cell);
        if (found != cells_.end()) {
            volume += found->second.volume;
            momentum += found->second.momentum[component];
        }
    }
    if (volume > 0.0) {
        return {momentum / volume, Validity::ValidData};
    }
    // Explicit one-cell velocity extrapolation from this same publication.
    // No old snapshot or another stream/lane contributes to this halo.
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const auto found = cells_.find({x + dx, y + dy, z + dz});
                if (found != cells_.end()) {
                    volume += found->second.volume;
                    momentum += found->second.momentum[component];
                }
            }
        }
    }
    return volume > 0.0 ? Scalar{momentum / volume, Validity::ValidData} : Scalar{};
}

std::vector<SampleRegion> DomainVisualField::regions(uint64_t stream) const {
    std::vector<SampleRegion> result;
    result.reserve(cells_.size());
    for (const auto& entry : cells_) {
        result.push_back({entry.first, stream, entry.second.volume});
    }
    return result;
}

MatterSupport DomainVisualField::support(const std::array<double, 3>& position) const {
    std::array<int, 3> cell{};
    for (int axis = 0; axis < 3; ++axis) {
        const double local = (position[axis] - geometry_.origin[axis]) / geometry_.voxel_size;
        if (!std::isfinite(local) || local < 0.0 || local >= geometry_.cells[axis]) {
            return {false, Validity::OutsideDomain};
        }
        cell[axis] = static_cast<int>(std::floor(local));
    }
    return {cells_.find(cell) != cells_.end(), Validity::ValidData};
}

} // namespace RayTrophiSim::Fluid
