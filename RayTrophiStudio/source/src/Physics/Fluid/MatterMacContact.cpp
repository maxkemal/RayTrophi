#include "Fluid/MatterMacContact.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace RayTrophiSim::Fluid {
namespace {

using FaceArrays = std::array<std::vector<double>, 3>;

std::array<std::size_t, 2> faces(const FluidSim::FluidGrid& grid,
                               const MatterCellKey& cell, int axis) {
    const int i = cell[0], j = cell[1], k = cell[2];
    if (axis == 0) {
        return {grid.velXIndex(i, j, k), grid.velXIndex(i + 1, j, k)};
    }
    if (axis == 1) {
        return {grid.velYIndex(i, j, k), grid.velYIndex(i, j + 1, k)};
    }
    return {grid.velZIndex(i, j, k), grid.velZIndex(i, j, k + 1)};
}

} // namespace

bool applyMatterMacContact(const FluidParticles& particles,
    FluidSim::FluidGrid& liquid, FluidSim::FluidGrid& granular,
    MatterConstitutiveModel legacy_model, double friction,
    MatterContactResult& result, std::string& error) {
    if (liquid.nx != granular.nx || liquid.ny != granular.ny ||
        liquid.nz != granular.nz || liquid.voxel_size != granular.voxel_size ||
        liquid.origin.x != granular.origin.x || liquid.origin.y != granular.origin.y ||
        liquid.origin.z != granular.origin.z ||
        liquid.vel_x.size() != granular.vel_x.size() ||
        liquid.vel_y.size() != granular.vel_y.size() ||
        liquid.vel_z.size() != granular.vel_z.size()) {
        error = "MAC contact requires identical model layouts";
        return false;
    }
    const std::size_t nx = liquid.nx > 0 ? liquid.nx : 0;
    const std::size_t ny = liquid.ny > 0 ? liquid.ny : 0;
    const std::size_t nz = liquid.nz > 0 ? liquid.nz : 0;
    if (!nx || !ny || !nz || liquid.vel_x.size() != (nx + 1) * ny * nz ||
        liquid.vel_y.size() != nx * (ny + 1) * nz ||
        liquid.vel_z.size() != nx * ny * (nz + 1)) {
        error = "MAC contact velocity cardinality does not match layout";
        return false;
    }
    MatterTransferFrame frame;
    if (!buildMatterTransfer(particles, liquid.origin, liquid.voxel_size,
        {liquid.nx, liquid.ny, liquid.nz}, legacy_model, 1.0, frame, error)) {
        return false;
    }
    std::array<std::array<std::vector<float>*, 3>, 2> velocity{{
        {&liquid.vel_x, &liquid.vel_y, &liquid.vel_z},
        {&granular.vel_x, &granular.vel_y, &granular.vel_z}}};
    std::array<FaceArrays, 2> mass;
    FaceArrays impulse;
    for (int axis = 0; axis < 3; ++axis) {
        const auto count = velocity[0][axis]->size();
        impulse[axis].assign(count, 0.0);
        for (auto& lane : mass) {
            lane[axis].assign(count, 0.0);
        }
    }
    // Average face velocities at the cell centre, exactly the adjoint of the
    // half-impulse lift below. This identity is needed by the energy bound.
    for (auto& [cell, entry] : frame.cells) {
        for (int axis = 0; axis < 3; ++axis) {
            const auto pair = faces(liquid, cell, axis);
            for (int lane = 0; lane < 2; ++lane) {
                auto& field = entry.model[lane];
                const double component = 0.5 * (
                    (*velocity[lane][axis])[pair[0]] + (*velocity[lane][axis])[pair[1]]);
                double* momentum[] = {&field.momentum.x, &field.momentum.y,
                                      &field.momentum.z};
                *momentum[axis] = field.mass_kg * component;
                for (const auto face : pair) {
                    mass[lane][axis][face] += 0.5 * field.mass_kg;
                }
            }
        }
    }
    std::map<MatterCellKey, MatterMomentum> before;
    for (const auto& [cell, entry] : frame.cells) {
        before.emplace(cell, entry.model[0].momentum);
    }
    MatterContactResult candidate;
    if (!applyMatterGridContact(frame, friction, candidate, error)) {
        return false;
    }
    for (const auto& [cell, entry] : frame.cells) {
        const auto initial = before.at(cell);
        const auto after = entry.model[0].momentum;
        const double delta[] = {after.x - initial.x, after.y - initial.y,
                                after.z - initial.z};
        for (int axis = 0; axis < 3; ++axis) {
            for (const auto face : faces(liquid, cell, axis)) {
                impulse[axis][face] += 0.5 * delta[axis];
            }
        }
    }
    double linear = 0.0, quadratic = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t face = 0; face < impulse[axis].size(); ++face) {
            const double value = impulse[axis][face];
            if (value == 0.0) {
                continue;
            }
            const double ml = mass[0][axis][face], mg = mass[1][axis][face];
            if (ml <= 0.0 || mg <= 0.0) {
                error = "MAC impulse reached a face without both physical masses";
                return false;
            }
            linear += value * ((*velocity[0][axis])[face] - (*velocity[1][axis])[face]);
            quadratic += value * value * (1.0 / ml + 1.0 / mg);
        }
    }
    if (!std::isfinite(linear) || !std::isfinite(quadratic)) {
        error = "MAC contact energy exceeds numerical range";
        return false;
    }
    const double scale = quadratic > 0.0
        ? std::clamp(-linear / quadratic, 0.0, 1.0) : 0.0;
    // Validate every proposed float before writing either model field.
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t face = 0; face < impulse[axis].size(); ++face) {
            for (int lane = 0; lane < 2; ++lane) {
                const double m = mass[lane][axis][face];
                const double delta = m > 0.0 ? scale * impulse[axis][face] / m : 0.0;
                const double proposed = (*velocity[lane][axis])[face] +
                    (lane == 0 ? delta : -delta);
                if (!std::isfinite(proposed) ||
                    std::abs(proposed) > std::numeric_limits<float>::max()) {
                    error = "MAC contact velocity exceeds float range";
                    return false;
                }
            }
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t face = 0; face < impulse[axis].size(); ++face) {
            for (int lane = 0; lane < 2; ++lane) {
                const double m = mass[lane][axis][face];
                const double delta = m > 0.0 ? scale * impulse[axis][face] / m : 0.0;
                (*velocity[lane][axis])[face] += static_cast<float>(lane == 0 ? delta : -delta);
            }
        }
    }
    candidate.impulse_norm *= scale;
    candidate.kinetic_energy_loss = -(scale * linear + 0.5 * scale * scale * quadratic);
    result = candidate;
    return true;
}

} // namespace RayTrophiSim::Fluid
