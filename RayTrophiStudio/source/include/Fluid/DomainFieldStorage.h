#pragma once

#include "Fluid/DomainFieldSnapshot.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace RayTrophiSim::Fluid::DomainFields {

struct ChannelBuffer {
    Channel channel = Channel::VolumeFraction;
    ChannelDescription description;
    // Storage coordinates may include explicit halos. X is the fastest axis.
    std::array<int, 3> first{};
    std::array<int, 3> extent{};
    std::vector<float> values;
    // Empty means every stored value is ValidData. Otherwise one status per value.
    std::vector<Validity> validity;
};

// An owning CPU publication. Copying input deliberately prevents mutable aliases
// to producer vectors. This is not the GPU/sparse runtime adapter.
class FrozenSnapshot final : public Snapshot {
public:
    static std::shared_ptr<const FrozenSnapshot> publish(
        const Identity& identity, const Geometry& geometry,
        const std::vector<ChannelBuffer>& channels, std::uint64_t byte_budget) {
        if (identity.domain_uid.empty() || !std::isfinite(identity.physical_time) ||
            identity.physical_time < 0.0 || !std::isfinite(geometry.voxel_size) ||
            geometry.voxel_size <= 0.0) {
            throw std::invalid_argument("invalid domain field identity or geometry");
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (geometry.cells[axis] <= 0 || !std::isfinite(geometry.origin[axis]) ||
                !std::isfinite(geometry.origin[axis] +
                    geometry.cells[axis] * geometry.voxel_size)) {
                throw std::invalid_argument("invalid domain field bounds");
            }
        }
        std::uint64_t bytes = 0;
        std::array<bool, 7> seen{};
        for (const auto& channel : channels) {
            const auto index = static_cast<std::size_t>(channel.channel);
            if (index >= seen.size() || seen[index]) {
                throw std::invalid_argument("duplicate or unknown domain field channel");
            }
            seen[index] = true;
            if (!channel.description.available ||
                channel.description.residence != Residence::Host ||
                !std::isfinite(channel.description.background)) {
                throw std::invalid_argument("CPU publication requires available host channels");
            }
            std::size_t count = 1;
            for (int axis = 0; axis < 3; ++axis) {
                const int extent = channel.extent[axis];
                if (extent <= 0 || channel.first[axis] >
                        std::numeric_limits<int>::max() - (extent - 1) ||
                    count > std::numeric_limits<std::size_t>::max() /
                        static_cast<std::size_t>(extent)) {
                    throw std::invalid_argument("invalid domain field storage extent");
                }
                count *= static_cast<std::size_t>(extent);
            }
            if (channel.values.size() != count ||
                (!channel.validity.empty() && channel.validity.size() != count)) {
                throw std::invalid_argument("domain field buffer size mismatch");
            }
            const std::uint64_t stride = sizeof(float) +
                (channel.validity.empty() ? 0 : sizeof(Validity));
            if (count > (std::numeric_limits<std::uint64_t>::max() - bytes) / stride) {
                throw std::overflow_error("domain field storage size overflow");
            }
            bytes += static_cast<std::uint64_t>(count) * stride;
            if (bytes > byte_budget) {
                throw std::length_error("domain field CPU publication exceeds byte budget");
            }
            for (std::size_t i = 0; i < count; ++i) {
                const auto status = channel.validity.empty()
                    ? Validity::ValidData : channel.validity[i];
                if (status != Validity::ValidData && status != Validity::KnownBackground &&
                    status != Validity::Unavailable && status != Validity::OutsideDomain) {
                    throw std::invalid_argument("invalid domain field validity");
                }
                if (usable(status) && (!std::isfinite(channel.values[i]) ||
                    (status == Validity::KnownBackground &&
                     channel.values[i] != channel.description.background))) {
                    throw std::invalid_argument("invalid domain field value/background");
                }
            }
        }
        return std::shared_ptr<const FrozenSnapshot>(
            new FrozenSnapshot(identity, geometry, channels, bytes));
    }

    const Identity& identity() const override { return identity_; }
    const Geometry& geometry() const override { return geometry_; }
    std::uint64_t valueBytes() const { return bytes_; }

    ChannelDescription describe(Channel channel) const override {
        const auto* buffer = find(channel);
        return buffer ? buffer->description : ChannelDescription{};
    }

    Scalar read(Channel channel, int x, int y, int z) const override {
        const auto* buffer = find(channel);
        if (!buffer) {
            return {};
        }
        const int coordinates[] = {x, y, z};
        std::array<std::uint64_t, 3> local{};
        for (int axis = 0; axis < 3; ++axis) {
            const auto delta = static_cast<std::int64_t>(coordinates[axis]) -
                buffer->first[axis];
            if (delta < 0 || delta >= buffer->extent[axis]) {
                // No implicit zero halo, even when the declared background is zero.
                return {};
            }
            local[axis] = static_cast<std::uint64_t>(delta);
        }
        const std::size_t index = static_cast<std::size_t>(local[0] +
            buffer->extent[0] * (local[1] + buffer->extent[1] * local[2]));
        const auto status = buffer->validity.empty()
            ? Validity::ValidData : buffer->validity[index];
        return {usable(status) ? buffer->values[index] : 0.0, status};
    }

private:
    FrozenSnapshot(const Identity& identity, const Geometry& geometry,
                   const std::vector<ChannelBuffer>& channels, std::uint64_t bytes)
        : identity_(identity), geometry_(geometry), channels_(channels), bytes_(bytes) {}

    const ChannelBuffer* find(Channel channel) const {
        for (const auto& buffer : channels_) {
            if (buffer.channel == channel) {
                return &buffer;
            }
        }
        return nullptr;
    }

    Identity identity_;
    Geometry geometry_;
    std::vector<ChannelBuffer> channels_;
    std::uint64_t bytes_ = 0;
};

} // namespace RayTrophiSim::Fluid::DomainFields
