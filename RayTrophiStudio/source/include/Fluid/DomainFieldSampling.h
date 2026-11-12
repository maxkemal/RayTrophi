#pragma once

#include "Fluid/DomainFieldSnapshot.h"

#include <cmath>
#include <limits>

namespace RayTrophiSim::Fluid::DomainFields {

struct Velocity {
    std::array<double, 3> value{};
    Validity validity = Validity::Unavailable;
};

// World-space domain is half-open. Boundary extension belongs to the snapshot
// adapter: this sampler never clamps indices or invents missing halo velocities.
inline Velocity sampleMacVelocity(const Snapshot& snapshot,
                                  const std::array<double, 3>& position) {
    const auto& geometry = snapshot.geometry();
    if (!std::isfinite(geometry.voxel_size) || geometry.voxel_size <= 0.0) {
        return {};
    }
    std::array<double, 3> local{};
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(position[axis]) || !std::isfinite(geometry.origin[axis]) ||
            geometry.cells[axis] <= 0) {
            return {};
        }
        local[axis] = (position[axis] - geometry.origin[axis]) / geometry.voxel_size;
        if (!std::isfinite(local[axis])) {
            return {};
        }
        if (local[axis] < 0.0 || local[axis] >= geometry.cells[axis]) {
            return {{}, Validity::OutsideDomain};
        }
    }

    constexpr Channel channels[] = {Channel::VelocityX, Channel::VelocityY,
                                    Channel::VelocityZ};
    constexpr Location locations[] = {Location::FaceX, Location::FaceY, Location::FaceZ};
    Velocity result;
    bool all_background = true;
    bool missing = false;
    bool outside = false;
    for (int component = 0; component < 3; ++component) {
        const auto description = snapshot.describe(channels[component]);
        if (!description.available || description.location != locations[component] ||
            description.unit != Unit::LengthPerSecond ||
            description.residence == Residence::Device) {
            return {};
        }
        std::array<int, 3> base{};
        std::array<double, 3> fraction{};
        for (int axis = 0; axis < 3; ++axis) {
            const double coordinate = local[axis] - (axis == component ? 0.0 : 0.5);
            const double floor_value = std::floor(coordinate);
            if (floor_value < std::numeric_limits<int>::min() ||
                floor_value > std::numeric_limits<int>::max() - 1) {
                return {};
            }
            base[axis] = static_cast<int>(floor_value);
            fraction[axis] = coordinate - floor_value;
        }
        for (int corner = 0; corner < 8; ++corner) {
            double weight = 1.0;
            std::array<int, 3> index = base;
            for (int axis = 0; axis < 3; ++axis) {
                const bool upper = (corner & (1 << axis)) != 0;
                weight *= upper ? fraction[axis] : 1.0 - fraction[axis];
                index[axis] += upper ? 1 : 0;
            }
            // An exact face needs no data from its zero-weight neighbours.
            if (weight == 0.0) {
                continue;
            }
            const auto sample = snapshot.read(channels[component], index[0], index[1],
                                              index[2]);
            if (!usable(sample.validity)) {
                missing |= sample.validity == Validity::Unavailable;
                outside |= sample.validity == Validity::OutsideDomain;
                continue;
            }
            if (!std::isfinite(sample.value)) {
                missing = true;
                continue;
            }
            all_background &= sample.validity == Validity::KnownBackground;
            result.value[component] += weight * sample.value;
        }
    }
    // No partial velocity escapes on a failed stencil. Missing data takes
    // precedence over an outside halo, regardless of component/corner order.
    if (missing || outside) {
        return {{}, missing ? Validity::Unavailable : Validity::OutsideDomain};
    }
    result.validity = all_background ? Validity::KnownBackground : Validity::ValidData;
    return result;
}

} // namespace RayTrophiSim::Fluid::DomainFields
