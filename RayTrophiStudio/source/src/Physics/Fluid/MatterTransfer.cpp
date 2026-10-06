#include "Fluid/MatterTransfer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace RayTrophiSim::Fluid {
namespace {

MatterMomentum add(MatterMomentum a, MatterMomentum b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

MatterMomentum scale(MatterMomentum a, double value) {
    return {a.x * value, a.y * value, a.z * value};
}

double dot(MatterMomentum a, MatterMomentum b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

bool finite(MatterMomentum value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

std::size_t lane(MatterConstitutiveModel model) {
    switch (model) {
        case MatterConstitutiveModel::Fluid: return 0;
        case MatterConstitutiveModel::Granular: return 1;
        case MatterConstitutiveModel::Elastic: return 2;
        default: return 3;
    }
}

MatterMomentum totalMomentum(const MatterTransferFrame& frame) {
    MatterMomentum total;
    for (const auto& entry : frame.cells) {
        for (const auto& field : entry.second.model) {
            total = add(total, field.momentum);
        }
    }
    return total;
}

} // namespace

bool buildMatterTransfer(const FluidParticles& particles, const Vec3& origin,
                         float voxel, const std::array<int, 3>& dimensions,
                         MatterConstitutiveModel legacy_model,
                         double legacy_mass_kg, MatterTransferFrame& result,
                         std::string& error, bool include_transfer) {
    error.clear();
    if (!finite(origin) || !std::isfinite(voxel) || voxel < 1e-6f ||
        !std::isfinite(legacy_mass_kg) || legacy_mass_kg < 0.0 ||
        legacy_mass_kg > std::numeric_limits<float>::max() ||
        dimensions[0] <= 0 || dimensions[1] <= 0 || dimensions[2] <= 0) {
        error = "invalid matter transfer grid or fallback mass";
        return false;
    }
    const std::size_t count = particles.size();
    if (particles.velocity.size() != count || particles.particle_id.size() != count) {
        error = "matter velocity/identity sidecars must match particle count";
        return false;
    }
    MatterTransferFrame candidate;
    std::unordered_set<uint64_t> identities;
    identities.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const Vec3& position = particles.position[index];
        const Vec3& velocity = particles.velocity[index];
        const uint64_t identity = particles.particle_id[index];
        if (!finite(position) || !finite(velocity) || identity == 0 ||
            !identities.insert(identity).second) {
            error = "matter particles require finite state and unique nonzero identities";
            return false;
        }
        const double rest_mass = index < particles.rest_mass_kg.size() &&
            particles.rest_mass_kg[index] > 0.0f
                ? particles.rest_mass_kg[index] : legacy_mass_kg;
        const double fraction = index < particles.mass_fraction.size()
            ? particles.mass_fraction[index] : 1.0;
        if (!std::isfinite(rest_mass) || !std::isfinite(fraction) ||
            fraction < 0.0 || fraction > 1.0 ||
            (index < particles.rest_mass_kg.size() &&
             (!std::isfinite(particles.rest_mass_kg[index]) ||
              particles.rest_mass_kg[index] < 0.0f))) {
            error = "matter mass must be finite and nonnegative";
            return false;
        }
        auto model = index < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[index])
            : MatterConstitutiveModel::Auto;
        if (model == MatterConstitutiveModel::Auto) {
            model = legacy_model;
        }
        const std::size_t model_lane = lane(model);
        const double pore = index < particles.pore_water_mass_kg.size()
            ? particles.pore_water_mass_kg[index] : 0.0;
        if (!std::isfinite(pore) || pore < 0.0) {
            error = "matter pore water must be finite and nonnegative";
            return false;
        }
        const double mass = rest_mass * fraction + pore;
        const MatterMomentum momentum{mass * velocity.x, mass * velocity.y, mass * velocity.z};
        auto& totals = candidate.totals[model_lane];
        ++totals.particles;
        totals.mass_kg += mass;
        totals.momentum = add(totals.momentum, momentum);
        if (include_transfer) {
            candidate.indices[model_lane].push_back(index);
        }
        if (model_lane > 1 || !include_transfer) {
            continue;
        }

        const double coordinate[3] = {
            (static_cast<double>(position.x) - origin.x) / voxel,
            (static_cast<double>(position.y) - origin.y) / voxel,
            (static_cast<double>(position.z) - origin.z) / voxel};
        bool outside = false;
        for (int axis = 0; axis < 3; ++axis) {
            outside = outside || coordinate[axis] < 0.0 ||
                coordinate[axis] >= dimensions[axis];
        }
        if (outside) {
            ++candidate.outside_particles;
            candidate.outside_mass_kg += mass;
            continue;
        }
        MatterCellKey home;
        int base[3];
        double weight[3][3];
        double derivative[3][3];
        for (int axis = 0; axis < 3; ++axis) {
            home[axis] = static_cast<int>(std::floor(coordinate[axis]));
            // Cell-centered quadratic basis, matching APIC mass scatter.
            base[axis] = static_cast<int>(std::floor(coordinate[axis] - 1.0));
            const double offset = coordinate[axis] - 0.5 - base[axis];
            weight[axis][0] = 0.5 * (1.5 - offset) * (1.5 - offset);
            weight[axis][1] = 0.75 - (offset - 1.0) * (offset - 1.0);
            weight[axis][2] = 0.5 * (offset - 0.5) * (offset - 0.5);
            derivative[axis][0] = (offset - 1.5) / voxel;
            derivative[axis][1] = -2.0 * (offset - 1.0) / voxel;
            derivative[axis][2] = (offset - 0.5) / voxel;
        }
        candidate.occupants[home].push_back(identity);
        struct Support {
            MatterCellKey cell;
            double weight;
            MatterMomentum gradient;
        };
        std::array<Support, 27> support;
        std::size_t support_count = 0;
        double normalization = 0.0;
        MatterMomentum normalization_gradient;
        for (int z = 0; z < 3; ++z) {
            for (int y = 0; y < 3; ++y) {
                for (int x = 0; x < 3; ++x) {
                    const MatterCellKey cell{base[0] + x, base[1] + y, base[2] + z};
                    if (cell[0] < 0 || cell[1] < 0 || cell[2] < 0 ||
                        cell[0] >= dimensions[0] || cell[1] >= dimensions[1] ||
                        cell[2] >= dimensions[2]) {
                        continue;
                    }
                    const double w = weight[0][x] * weight[1][y] * weight[2][z];
                    const MatterMomentum gradient{
                        derivative[0][x] * weight[1][y] * weight[2][z],
                        weight[0][x] * derivative[1][y] * weight[2][z],
                        weight[0][x] * weight[1][y] * derivative[2][z]};
                    support[support_count++] = {cell, w, gradient};
                    normalization += w;
                    normalization_gradient = add(normalization_gradient, gradient);
                }
            }
        }
        if (!(normalization > 0.0)) {
            error = "matter particle has no valid transfer support";
            return false;
        }
        for (std::size_t entry = 0; entry < support_count; ++entry) {
            const auto& sample = support[entry];
            const double w = sample.weight / normalization;
            auto& field = candidate.cells[sample.cell].model[model_lane];
            field.mass_kg += mass * w;
            field.momentum = add(field.momentum, scale(momentum, w));
            const auto gradient = scale(add(scale(sample.gradient, normalization),
                scale(normalization_gradient, -sample.weight)),
                mass / (normalization * normalization));
            field.mass_gradient = add(field.mass_gradient, gradient);
        }
        candidate.deposited_mass_kg += mass;
        candidate.deposited_momentum = add(candidate.deposited_momentum, momentum);
    }
    for (auto& entry : candidate.occupants) {
        std::sort(entry.second.begin(), entry.second.end());
    }
    for (const auto& entry : candidate.cells) {
        if (entry.second.model[0].mass_kg > 0.0 && entry.second.model[1].mass_kg > 0.0) {
            ++candidate.overlapping_cells;
        }
    }
    result = std::move(candidate);
    return true;
}

bool applyMatterGridContact(MatterTransferFrame& frame, double friction,
                           MatterContactResult& result, std::string& error) {
    error.clear();
    if (!std::isfinite(friction) || friction < 0.0) {
        error = "contact friction must be finite and nonnegative";
        return false;
    }
    for (const auto& entry : frame.cells) {
        for (const auto& field : entry.second.model) {
            if (!std::isfinite(field.mass_kg) || field.mass_kg < 0.0 ||
                !finite(field.momentum) || !finite(field.mass_gradient)) {
                error = "contact field contains invalid mass, momentum or gradient";
                return false;
            }
        }
    }
    MatterContactResult candidate;
    const auto before = totalMomentum(frame);
    if (!finite(before)) {
        error = "contact total momentum exceeds numerical range";
        return false;
    }
    auto cells = frame.cells;
    for (auto& entry : cells) {
        auto& fluid = entry.second.model[0];
        auto& granular = entry.second.model[1];
        if (fluid.mass_kg <= 0.0 || granular.mass_kg <= 0.0) {
            continue;
        }
        auto normal = add(scale(granular.mass_gradient, 1.0 / granular.mass_kg),
            scale(fluid.mass_gradient, -1.0 / fluid.mass_kg));
        const double normal_length = std::sqrt(dot(normal, normal));
        if (!finite(normal) || !std::isfinite(normal_length)) {
            error = "contact normal exceeds numerical range";
            return false;
        }
        if (normal_length < 1e-12) {
            continue;
        }
        normal = scale(normal, 1.0 / normal_length);
        const auto fluid_velocity = scale(fluid.momentum, 1.0 / fluid.mass_kg);
        const auto granular_velocity = scale(granular.momentum, 1.0 / granular.mass_kg);
        const auto relative = add(fluid_velocity, scale(granular_velocity, -1.0));
        const double closing_speed = dot(relative, normal);
        if (!finite(relative) || !std::isfinite(closing_speed)) {
            error = "contact velocity exceeds numerical range";
            return false;
        }
        if (closing_speed <= 0.0) {
            continue;
        }
        const double reduced_mass = fluid.mass_kg <= granular.mass_kg
            ? fluid.mass_kg / (1.0 + fluid.mass_kg / granular.mass_kg)
            : granular.mass_kg / (1.0 + granular.mass_kg / fluid.mass_kg);
        const double normal_impulse = reduced_mass * closing_speed;
        const auto tangent = add(relative, scale(normal, -closing_speed));
        const double tangent_speed = std::sqrt(dot(tangent, tangent));
        if (!std::isfinite(tangent_speed) || !std::isfinite(normal_impulse)) {
            error = "contact impulse exceeds numerical range";
            return false;
        }
        const double tangent_impulse = std::min(
            reduced_mass * tangent_speed, friction * normal_impulse);
        auto impulse = scale(normal, normal_impulse);
        if (tangent_speed > 1e-12) {
            impulse = add(impulse, scale(tangent, tangent_impulse / tangent_speed));
        }
        fluid.momentum = add(fluid.momentum, scale(impulse, -1.0));
        granular.momentum = add(granular.momentum, impulse);
        candidate.kinetic_energy_loss += dot(impulse, relative) -
            0.5 * dot(impulse, impulse) / reduced_mass;
        candidate.impulse_norm += std::sqrt(dot(impulse, impulse));
        ++candidate.pairs;
    }
    if (!std::isfinite(candidate.kinetic_energy_loss) ||
        !std::isfinite(candidate.impulse_norm)) {
        error = "contact energy exceeds numerical range";
        return false;
    }
    frame.cells = std::move(cells);
    candidate.momentum_error = add(totalMomentum(frame), scale(before, -1.0));
    result = candidate;
    return true;
}

} // namespace RayTrophiSim::Fluid
