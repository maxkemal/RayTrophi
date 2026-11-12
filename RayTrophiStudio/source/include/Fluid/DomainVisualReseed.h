#pragma once

#include "Fluid/DomainVisualSamples.h"

#include <map>
#include <set>
#include <tuple>

namespace RayTrophiSim::Fluid::DomainFields {

struct SampleRegion {
    std::array<int, 3> cell{};
    std::uint64_t stream_uid = 0;
    // Physical reconstruction volume, not particle count or binary fluid mask.
    double volume = 0.0;
};

struct MatterSupport {
    bool occupied = false;
    Validity validity = Validity::Unavailable;
};

inline bool occupiedSupport(const MatterSupport& support) {
    return support.occupied && usable(support.validity);
}

using VisualSupport = std::function<MatterSupport(
    std::uint64_t stream, const std::array<double, 3>& position)>;

struct ReseedStats {
    std::uint64_t born = 0;
    std::uint64_t retired = 0;
    std::uint64_t support_rejections = 0;
    double represented_volume = 0.0;
    double unrepresented_volume = 0.0;
};

inline std::uint64_t visualHash(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

inline double visualUnitRandom(std::uint64_t key) {
    return static_cast<double>(visualHash(key) >> 11) * 0x1.0p-53;
}

using RegionKey = std::tuple<std::uint64_t, int, int, int>;

inline RegionKey regionKey(const SampleRegion& region) {
    return {region.stream_uid, region.cell[0], region.cell[1], region.cell[2]};
}

// CPU oracle for cell allocation and stable sample retention. Input is already
// reconstructed physical support; this function never derives volume from count.
// Missing support is reported, never filled with parent offsets. Allocations and
// UID changes commit together. Cost/memory are bounded by regions + sample budget.
inline ReseedStats reseedVisualSamples(
    const Geometry& geometry, const VisualSampleSettings& settings,
    double physical_time, const std::vector<SampleRegion>& regions,
    const VisualSupport& support, std::uint64_t& next_uid,
    std::vector<VisualSample>& samples) {
    validateVisualSettings(settings);
    if (!support || !std::isfinite(physical_time) || physical_time < 0.0 ||
        !std::isfinite(geometry.voxel_size) || geometry.voxel_size <= 0.0 ||
        samples.size() > 32000000 || next_uid == 0) {
        throw std::invalid_argument("invalid visual reseed input");
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (geometry.cells[axis] <= 0 || !std::isfinite(geometry.origin[axis])) {
            throw std::invalid_argument("invalid visual reseed geometry");
        }
    }
    struct Bin {
        SampleRegion region;
        std::vector<VisualSample> retained;
        std::uint64_t wanted = 0;
        std::uint64_t granted = 0;
        double remainder = 0.0;
    };
    std::map<RegionKey, Bin> bins;
    long double total_wanted = 0.0L;
    double total_volume = 0.0;
    for (const auto& region : regions) {
        if (!std::isfinite(region.volume) || region.volume <= 0.0) {
            throw std::invalid_argument("visual region volume must be finite and positive");
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (region.cell[axis] < 0 || region.cell[axis] >= geometry.cells[axis]) {
                throw std::invalid_argument("visual region outside domain");
            }
        }
        const auto inserted = bins.emplace(regionKey(region), Bin{region});
        if (!inserted.second) {
            throw std::invalid_argument("duplicate visual region/stream");
        }
        auto& bin = inserted.first->second;
        total_volume += region.volume;
        if (!std::isfinite(total_volume)) {
            throw std::overflow_error("visual region total volume overflow");
        }
        const double desired = region.volume / settings.target_volume;
        bin.wanted = desired >= static_cast<double>(settings.max_samples)
            ? settings.max_samples
            : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(std::floor(desired + 0.5)));
    }
    std::set<std::uint64_t> identities;
    for (const auto& sample : samples) {
        if (sample.uid == 0 || sample.uid >= next_uid ||
            !identities.insert(sample.uid).second) {
            throw std::invalid_argument("invalid or duplicate visual sample identity");
        }
        SampleRegion region;
        region.stream_uid = sample.stream_uid;
        bool inside = true;
        for (int axis = 0; axis < 3; ++axis) {
            const double local = (sample.position[axis] - geometry.origin[axis]) /
                geometry.voxel_size;
            if (!std::isfinite(local) || local < 0.0 || local >= geometry.cells[axis]) {
                inside = false;
                break;
            }
            region.cell[axis] = static_cast<int>(std::floor(local));
        }
        if (inside) {
            auto found = bins.find(regionKey(region));
            if (found != bins.end()) {
                const auto status = support(sample.stream_uid, sample.position);
                if (status.validity == Validity::Unavailable) {
                    throw std::runtime_error("retained visual sample support unavailable");
                }
                if (occupiedSupport(status)) {
                    found->second.retained.push_back(sample);
                }
            }
        }
    }
    std::uint64_t allocated = 0;
    std::vector<Bin*> allocation_order;
    for (auto& entry : bins) {
        auto& bin = entry.second;
        const auto retained = static_cast<std::uint64_t>(bin.retained.size());
        const double desired = bin.region.volume / settings.target_volume;
        if (retained > 0 && retained >= desired * (1.0 - settings.reseed_hysteresis) &&
            retained <= desired * (1.0 + settings.reseed_hysteresis)) {
            bin.wanted = retained;
        }
        total_wanted += bin.wanted;
    }
    for (auto& entry : bins) {
        auto& bin = entry.second;
        std::sort(bin.retained.begin(), bin.retained.end(),
            [](const VisualSample& a, const VisualSample& b) { return a.uid < b.uid; });
        const long double ideal = total_wanted <= settings.max_samples
            ? static_cast<long double>(bin.wanted)
            : static_cast<long double>(bin.wanted) * settings.max_samples / total_wanted;
        bin.granted = static_cast<std::uint64_t>(std::floor(ideal));
        bin.remainder = static_cast<double>(ideal - bin.granted);
        allocated += bin.granted;
        allocation_order.push_back(&bin);
    }
    // Largest remainder allocation is deterministic in sorted stream/cell order.
    std::stable_sort(allocation_order.begin(), allocation_order.end(),
        [](const Bin* a, const Bin* b) { return a->remainder > b->remainder; });
    const auto target = static_cast<std::uint64_t>(
        std::min(total_wanted, static_cast<long double>(settings.max_samples)));
    for (auto* bin : allocation_order) {
        if (allocated >= target) {
            break;
        }
        if (bin->granted < bin->wanted) {
            ++bin->granted;
            ++allocated;
        }
    }
    std::vector<VisualSample> next;
    next.reserve(static_cast<std::size_t>(target));
    auto uid_cursor = next_uid;
    ReseedStats stats;
    std::uint64_t retained_count = 0;
    for (auto& entry : bins) {
        auto& bin = entry.second;
        const std::size_t begin = next.size();
        const auto keep = std::min<std::uint64_t>(bin.retained.size(), bin.granted);
        for (std::uint64_t i = 0; i < keep; ++i) {
            next.push_back(bin.retained[static_cast<std::size_t>(i)]);
        }
        retained_count += keep;
        // Candidate locations depend on cell/stream/seed and birth UID. They do
        // not depend on carrier order, camera, redraw count or source position.
        std::uint64_t cell_key = visualHash(settings.seed ^ bin.region.stream_uid);
        for (const auto coordinate : bin.region.cell) {
            cell_key = visualHash(cell_key ^ static_cast<std::uint64_t>(coordinate));
        }
        for (std::uint64_t i = keep; i < bin.granted; ++i) {
            if (uid_cursor == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error("visual sample identity exhausted");
            }
            bool accepted = false;
            VisualSample sample;
            sample.uid = uid_cursor;
            sample.stream_uid = bin.region.stream_uid;
            sample.birth_time = physical_time;
            const double u1 = visualUnitRandom(cell_key ^ visualHash(uid_cursor) ^ 0x1001ull);
            const double u2 = visualUnitRandom(cell_key ^ visualHash(uid_cursor) ^ 0x1002ull);
            const double u3 = visualUnitRandom(cell_key ^ visualHash(uid_cursor) ^ 0x1003ull);
            constexpr double tau = 6.28318530717958647692;
            sample.orientation = {std::sqrt(1.0 - u1) * std::sin(tau * u2),
                std::sqrt(1.0 - u1) * std::cos(tau * u2),
                std::sqrt(u1) * std::sin(tau * u3), std::sqrt(u1) * std::cos(tau * u3)};
            for (std::uint64_t trial = 0; trial < 32; ++trial) {
                for (int axis = 0; axis < 3; ++axis) {
                    const double offset = visualUnitRandom(cell_key ^ visualHash(uid_cursor) ^
                        visualHash(trial * 3 + static_cast<std::uint64_t>(axis)));
                    sample.position[axis] = geometry.origin[axis] +
                        (bin.region.cell[axis] + offset) * geometry.voxel_size;
                }
                if (occupiedSupport(support(sample.stream_uid, sample.position))) {
                    accepted = true;
                    break;
                }
                ++stats.support_rejections;
            }
            ++uid_cursor;
            if (accepted) {
                sample.previous_position = sample.position;
                next.push_back(sample);
                ++stats.born;
            }
        }
        const std::size_t count = next.size() - begin;
        if (count == 0) {
            stats.unrepresented_volume += bin.region.volume;
        } else {
            const double volume = bin.region.volume / static_cast<double>(count);
            for (std::size_t index = begin; index < next.size(); ++index) {
                next[index].represented_volume = volume;
            }
            stats.represented_volume += bin.region.volume;
        }
    }
    stats.retired = samples.size() - retained_count;
    samples.swap(next);
    next_uid = uid_cursor;
    return stats;
}

} // namespace RayTrophiSim::Fluid::DomainFields
