#pragma once

#include "Fluid/DomainFieldSampling.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>

namespace RayTrophiSim::Fluid::DomainFields {

struct VisualSample {
    std::uint64_t uid = 0;
    std::uint64_t stream_uid = 0;
    std::array<double, 3> position{};
    std::array<double, 3> previous_position{};
    double represented_volume = 0.0;
    double birth_time = 0.0;
    // Birth-seeded fixed orientation; quaternion order is x,y,z,w.
    std::array<double, 4> orientation{0.0, 0.0, 0.0, 1.0};
};

struct VisualSampleSettings {
    double target_volume = 0.001;
    std::uint64_t max_samples = 1000000;
    std::uint64_t seed = 1;
    double cfl = 0.5;
    double reseed_hysteresis = 0.2;
    std::uint32_t max_substeps = 256;
};

struct TransportStats {
    std::uint64_t moved = 0;
    std::uint64_t held = 0;
    std::uint64_t retired = 0;
    std::uint64_t substeps = 0;
    bool frozen_velocity = true;
};

// Swept collision/boundary contract. The provider owns moving-wall response,
// periodic wrapping and support validity. It cannot modify physical state.
// Unavailable holds the complete old position. OutsideDomain retires it.
using TraceVisualSegment = std::function<Validity(
    std::uint64_t stream, const std::array<double, 3>& from,
    std::array<double, 3>& to, double from_time, double to_time)>;

inline void validateVisualSettings(const VisualSampleSettings& settings) {
    if (!std::isfinite(settings.target_volume) || settings.target_volume <= 0.0 ||
        settings.max_samples == 0 || settings.max_samples > 32000000 ||
        !std::isfinite(settings.cfl) || settings.cfl <= 0.0 || settings.cfl > 1.0 ||
        !std::isfinite(settings.reseed_hysteresis) || settings.reseed_hysteresis < 0.0 ||
        settings.reseed_hysteresis > 0.5 ||
        settings.max_substeps == 0 || settings.max_substeps > 65536) {
        throw std::invalid_argument("invalid visual sample volume, budget or CFL settings");
    }
}

inline double visualSpeed(const std::array<double, 3>& velocity) {
    return std::max({std::abs(velocity[0]), std::abs(velocity[1]), std::abs(velocity[2])});
}

// Reference transport against one immutable end-of-step velocity publication.
// No force integration. frozen-velocity RK2 is not second-order in physical time.
// Results commit as a transaction; callback/allocation exceptions change nothing.
inline TransportStats transportVisualSamples(
    const Snapshot& snapshot, const VisualSampleSettings& settings,
    double from_time, double to_time, const TraceVisualSegment& trace,
    std::vector<VisualSample>& samples) {
    validateVisualSettings(settings);
    if (!trace || !std::isfinite(from_time) || !std::isfinite(to_time) ||
        from_time < 0.0 || to_time < from_time ||
        snapshot.identity().physical_time != to_time || samples.size() > 32000000) {
        throw std::invalid_argument("invalid visual sample transport interval or publication");
    }
    TransportStats stats;
    if (to_time == from_time) {
        return stats;
    }
    const double distance_limit = snapshot.geometry().voxel_size * settings.cfl;
    if (!std::isfinite(distance_limit) || distance_limit <= 0.0) {
        throw std::invalid_argument("invalid visual sample transport geometry");
    }
    std::vector<VisualSample> next;
    next.reserve(samples.size());
    for (const auto& old : samples) {
        VisualSample sample = old;
        sample.previous_position = old.position;
        double time = from_time;
        bool held = false;
        bool retired = false;
        std::uint32_t substeps = 0;
        while (time < to_time) {
            if (substeps >= settings.max_substeps) {
                held = true;
                break;
            }
            const auto velocity = sampleMacVelocity(snapshot, sample.position);
            if (!usable(velocity.validity)) {
                retired = velocity.validity == Validity::OutsideDomain;
                held = !retired;
                break;
            }
            const double speed = visualSpeed(velocity.value);
            double dt = to_time - time;
            if (speed > 0.0) {
                dt = std::min(dt, distance_limit / speed);
            }
            bool accepted = false;
            // Midpoint velocity can exceed start velocity. Backtrack to enforce
            // the same CFL bound at both samples rather than crossing a wall.
            for (int retry = 0; retry < 32; ++retry) {
                if (!(dt > 0.0) || time + dt == time) {
                    break;
                }
                auto midpoint = sample.position;
                for (int axis = 0; axis < 3; ++axis) {
                    midpoint[axis] += velocity.value[axis] * (0.5 * dt);
                }
                // Validate the midpoint sweep before reading through a solid.
                const auto midpoint_status = trace(sample.stream_uid, sample.position,
                    midpoint, time, time + 0.5 * dt);
                if (!usable(midpoint_status)) {
                    retired = midpoint_status == Validity::OutsideDomain;
                    held = !retired;
                    break;
                }
                const auto midpoint_velocity = sampleMacVelocity(snapshot, midpoint);
                if (!usable(midpoint_velocity.validity)) {
                    retired = midpoint_velocity.validity == Validity::OutsideDomain;
                    held = !retired;
                    break;
                }
                if (visualSpeed(midpoint_velocity.value) * dt > distance_limit) {
                    dt *= 0.5;
                    continue;
                }
                auto destination = sample.position;
                for (int axis = 0; axis < 3; ++axis) {
                    destination[axis] += midpoint_velocity.value[axis] * dt;
                }
                const auto status = trace(sample.stream_uid, sample.position, destination,
                                          time, time + dt);
                if (!usable(status)) {
                    retired = status == Validity::OutsideDomain;
                    held = !retired;
                    break;
                }
                bool finite = true;
                for (const auto value : destination) {
                    finite &= std::isfinite(value);
                }
                if (!finite) {
                    held = true;
                    break;
                }
                sample.position = destination;
                time = dt >= to_time - time ? to_time : time + dt;
                ++substeps;
                accepted = true;
                break;
            }
            if (!accepted) {
                held |= !retired;
                break;
            }
        }
        stats.substeps += substeps;
        if (retired) {
            ++stats.retired;
        } else if (held) {
            // A failed late substep must not leave the sample at a partial time.
            sample = old;
            sample.previous_position = old.position;
            next.push_back(sample);
            ++stats.held;
        } else {
            next.push_back(sample);
            ++stats.moved;
        }
    }
    samples.swap(next);
    return stats;
}

} // namespace RayTrophiSim::Fluid::DomainFields
