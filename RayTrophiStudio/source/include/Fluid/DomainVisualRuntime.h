#pragma once

#include "Fluid/DomainVisualReseed.h"

#include <utility>

namespace RayTrophiSim::Fluid::DomainFields {

struct VisualCheckpoint {
    Identity identity;
    Geometry geometry;
    VisualSampleSettings settings;
    std::vector<VisualSample> samples;
    std::uint64_t next_uid = 1;
    bool initialized = false;
};

struct VisualUpdateStats {
    TransportStats transport;
    ReseedStats reseed;
    bool unchanged = false;
};

// Internal CPU reference owner. A renderer cannot advance the same physical
// publication twice. Seek/rewind requires an explicit checkpoint/reset, never a
// new RNG seed pretending to recover the previous advected positions.
class VisualRuntime {
public:
    const VisualCheckpoint& state() const {
        return state_;
    }

    void configure(const VisualSampleSettings& settings) {
        validateVisualSettings(settings);
        state_.settings = settings;
    }

    void reset(const VisualSampleSettings& settings) {
        validateVisualSettings(settings);
        VisualCheckpoint empty;
        empty.settings = settings;
        state_ = std::move(empty);
    }

    void restore(const VisualCheckpoint& checkpoint) {
        validateVisualSettings(checkpoint.settings);
        if (checkpoint.next_uid == 0 ||
            checkpoint.samples.size() > checkpoint.settings.max_samples ||
            (!checkpoint.initialized && !checkpoint.samples.empty())) {
            throw std::invalid_argument("invalid visual sample checkpoint");
        }
        std::set<std::uint64_t> identities;
        for (const auto& sample : checkpoint.samples) {
            if (sample.uid == 0 || sample.uid >= checkpoint.next_uid ||
                !identities.insert(sample.uid).second ||
                !std::isfinite(sample.represented_volume) || sample.represented_volume <= 0.0 ||
                !std::isfinite(sample.birth_time) || sample.birth_time < 0.0 ||
                sample.birth_time > checkpoint.identity.physical_time) {
                throw std::invalid_argument("invalid visual sample checkpoint entry");
            }
            double norm = 0.0;
            for (const auto value : sample.orientation) {
                norm += value * value;
            }
            if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1e-6) {
                throw std::invalid_argument("invalid visual sample checkpoint orientation");
            }
            for (int axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(sample.position[axis]) ||
                    !std::isfinite(sample.previous_position[axis])) {
                    throw std::invalid_argument("invalid visual sample checkpoint position");
                }
            }
        }
        if (checkpoint.initialized) {
            const auto& geometry = checkpoint.geometry;
            if (checkpoint.identity.domain_uid.empty() ||
                !std::isfinite(checkpoint.identity.physical_time) ||
                checkpoint.identity.physical_time < 0.0 ||
                !std::isfinite(geometry.voxel_size) || geometry.voxel_size <= 0.0) {
                throw std::invalid_argument("invalid visual sample checkpoint publication");
            }
            for (int axis = 0; axis < 3; ++axis) {
                if (geometry.cells[axis] <= 0 || !std::isfinite(geometry.origin[axis])) {
                    throw std::invalid_argument("invalid visual sample checkpoint geometry");
                }
            }
        }
        auto copy = checkpoint;
        state_ = std::move(copy);
    }

    VisualUpdateStats update(const Snapshot& snapshot,
                             const std::vector<SampleRegion>& regions,
                             const TraceVisualSegment& trace, const VisualSupport& support,
                             bool force_reseed = false) {
        const auto& identity = snapshot.identity();
        const auto& geometry = snapshot.geometry();
        if (identity.domain_uid.empty() || !std::isfinite(identity.physical_time) ||
            identity.physical_time < 0.0) {
            throw std::invalid_argument("invalid visual sample publication identity");
        }
        if (state_.initialized) {
            if (identity.domain_uid != state_.identity.domain_uid ||
                identity.lane != state_.identity.lane ||
                identity.topology_generation != state_.identity.topology_generation ||
                identity.transform_revision != state_.identity.transform_revision ||
                geometry.origin != state_.geometry.origin ||
                geometry.cells != state_.geometry.cells ||
                geometry.voxel_size != state_.geometry.voxel_size) {
                throw std::invalid_argument(
                    "visual sample publication requires reset or checkpoint");
            }
            if (!std::isfinite(identity.physical_time) ||
                identity.physical_time < state_.identity.physical_time ||
                identity.state_generation < state_.identity.state_generation ||
                identity.step < state_.identity.step ||
                (identity.step == state_.identity.step &&
                 identity.substep < state_.identity.substep)) {
                throw std::invalid_argument("visual sample rewind requires a checkpoint");
            }
            const bool same_publication =
                identity.state_generation == state_.identity.state_generation &&
                identity.physical_time == state_.identity.physical_time &&
                identity.step == state_.identity.step &&
                identity.substep == state_.identity.substep;
            if (same_publication) {
                VisualUpdateStats stats;
                stats.unchanged = true;
                if (!force_reseed) {
                    return stats;
                }
            }
            if (identity.state_generation == state_.identity.state_generation &&
                !same_publication) {
                throw std::invalid_argument("visual sample publication changed without generation");
            }
        }
        VisualCheckpoint next = state_;
        VisualUpdateStats stats;
        if (next.initialized) {
            stats.transport = transportVisualSamples(snapshot, next.settings,
                next.identity.physical_time, identity.physical_time, trace, next.samples);
            // A held sample remains at the old physical time. Publishing that
            // partial population as current would silently lose elapsed time.
            if (stats.transport.held != 0) {
                throw std::runtime_error(
                    "visual sample transport unavailable or CFL budget exceeded");
            }
        }
        stats.reseed = reseedVisualSamples(geometry, next.settings, identity.physical_time,
            regions, support, next.next_uid, next.samples);
        next.identity = identity;
        next.geometry = geometry;
        next.initialized = true;
        state_ = std::move(next);
        return stats;
    }

private:
    VisualCheckpoint state_;
};

} // namespace RayTrophiSim::Fluid::DomainFields
