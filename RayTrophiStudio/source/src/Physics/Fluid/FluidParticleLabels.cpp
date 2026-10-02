#include "Fluid/FluidParticleLabels.h"

#include "Fluid/FluidFoam.h"
#include "Fluid/FluidParticles.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace RayTrophiSim {
namespace Fluid {
namespace {

struct Cell {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const Cell& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

constexpr std::size_t kInvalidParticle = (std::numeric_limits<std::size_t>::max)();

// Open addressing avoids an allocation per occupied cell (and emplace's
// temporary node on duplicate keys). Reused storage is scratch, never state:
// every invocation starts a new generation and rebuilds all live memberships.
struct LabelBins {
    struct Bucket {
        Cell cell;
        std::size_t head = kInvalidParticle;
        uint32_t generation = 0;
    };

    std::vector<Bucket> table;
    std::vector<std::size_t> next;
    std::vector<std::size_t> particle_bucket;
    uint32_t generation = 0;
    uint64_t occupied = 0;

    void begin(std::size_t count) {
        // At most one occupied bucket per parcel, with load <= 1/2. Multiplying
        // only while capacity/2 < count avoids overflowing count * 2.
        std::size_t capacity = 16;
        while (capacity / 2 < count) {
            if (capacity > table.max_size() / 2) {
                throw std::length_error("particle label bins exceed addressable storage");
            }
            capacity *= 2;
        }
        if (table.size() < capacity) {
            table.assign(capacity, Bucket{});
            generation = 0;
        }
        if (++generation == 0) {
            std::fill(table.begin(), table.end(), Bucket{});
            generation = 1;
        }
        next.resize(count);
        particle_bucket.assign(count, kInvalidParticle);
        occupied = 0;
    }

    std::size_t hash(const Cell& cell) const {
        uint64_t value = static_cast<uint32_t>(cell.x) * uint64_t{73856093} ^
                         static_cast<uint32_t>(cell.y) * uint64_t{19349663} ^
                         static_cast<uint32_t>(cell.z) * uint64_t{83492791};
        value ^= value >> 33;
        value *= 0xff51afd7ed558ccdULL;
        value ^= value >> 33;
        return static_cast<std::size_t>(value) & (table.size() - 1);
    }

    void insert(const Cell& cell, std::size_t particle) {
        std::size_t slot = hash(cell);
        while (table[slot].generation == generation && !(table[slot].cell == cell)) {
            slot = (slot + 1) & (table.size() - 1);
        }
        auto& bucket = table[slot];
        if (bucket.generation != generation) {
            bucket.cell = cell;
            bucket.head = kInvalidParticle;
            bucket.generation = generation;
            ++occupied;
        }
        next[particle] = bucket.head;
        bucket.head = particle;
        particle_bucket[particle] = slot;
    }

    std::size_t findHead(const Cell& cell) const {
        std::size_t slot = hash(cell);
        while (table[slot].generation == generation) {
            if (table[slot].cell == cell) {
                return table[slot].head;
            }
            slot = (slot + 1) & (table.size() - 1);
        }
        return kInvalidParticle;
    }
};

bool cellAt(const Vec3& position, double radius, Cell& cell) {
    const double x = std::floor(static_cast<double>(position.x) / radius);
    const double y = std::floor(static_cast<double>(position.y) / radius);
    const double z = std::floor(static_cast<double>(position.z) / radius);
    const double limit = static_cast<double>((std::numeric_limits<int>::max)()) - 2.0;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        std::abs(x) > limit || std::abs(y) > limit || std::abs(z) > limit) {
        return false;
    }
    cell = {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)};
    return true;
}

} // namespace

const char* particleLabelName(ParticleLabel label) {
    switch (label) {
        case ParticleLabel::Body: return "body";
        case ParticleLabel::Spray: return "spray";
        case ParticleLabel::Foam: return "foam";
        case ParticleLabel::Bubble: return "bubble";
        case ParticleLabel::Mist: return "mist";
        case ParticleLabel::Frozen: return "frozen";
        default: return "unknown";
    }
}

ParticleLabel particleLabel(uint32_t flags) {
    const uint32_t value = (flags & kParticleLabelMask) >> kParticleLabelShift;
    return value < static_cast<uint32_t>(ParticleLabel::Count)
        ? static_cast<ParticleLabel>(value) : ParticleLabel::Unknown;
}

ParticleLabel secondaryParticleLabel(uint8_t foam_type) {
    switch (static_cast<FoamType>(foam_type)) {
        case FoamType::Spray: return ParticleLabel::Spray;
        case FoamType::Foam: return ParticleLabel::Foam;
        case FoamType::Bubble: return ParticleLabel::Bubble;
        default: return ParticleLabel::Unknown;
    }
}

void setParticleLabel(uint32_t& flags, ParticleLabel label) {
    flags = (flags & ~kParticleLabelMask) |
        ((static_cast<uint32_t>(label) << kParticleLabelShift) & kParticleLabelMask);
}

ParticleLabelStepStats updateParticleLabels(FluidParticles& particles,
                                           float voxel_size, bool granular,
                                           const std::vector<uint32_t>* solid_tags) {
    static_assert((kParticleLabelMask & kParticleFlagFrozen) == 0u);
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const std::size_t count = particles.size();
    particles.flags.resize(count, 0u);
    const double radius = static_cast<double>(voxel_size) * kParticleLabelRadiusVoxels;
    const bool classify = !granular && std::isfinite(radius) && radius > 0.0;
    // Thread-local scratch can serve consecutive domains without mixing their
    // contents, and does not race with another thread's independent simulation.
    static thread_local LabelBins scratch;
    const auto solid = [&](std::size_t i) {
        return solid_tags && i < particles.substance_tag.size() &&
            std::find(solid_tags->begin(), solid_tags->end(), particles.substance_tag[i]) !=
                solid_tags->end();
    };
    const auto mist = [&](std::size_t i) {
        return i < particles.mass_fraction.size() &&
            std::isfinite(particles.mass_fraction[i]) &&
            particles.mass_fraction[i] > 0.0f &&
            particles.mass_fraction[i] <= kParticleMistMassFraction;
    };
    if (classify) {
        scratch.begin(count);
        for (std::size_t i = 0; i < count; ++i) {
            Cell cell;
            if (solid(i) || mist(i) || !cellAt(particles.position[i], radius, cell)) {
                continue;
            }
            scratch.insert(cell, i);
        }
    }

    ParticleLabelStepStats stats;
    stats.particles = count;
    const auto bin_end = Clock::now();
    stats.bin_milliseconds = std::chrono::duration<double, std::milli>(bin_end - start).count();
    stats.occupied_bins = classify ? scratch.occupied : 0;
    uint64_t changed = 0;
    uint64_t center_resolved = 0;
    const LabelBins& bins = scratch;
    const double radius_squared = radius * radius;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count >= 4096) reduction(+:changed,center_resolved)
#endif
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(count); ++index) {
        const std::size_t i = static_cast<std::size_t>(index);
        const ParticleLabel previous = particleLabel(particles.flags[i]);
        ParticleLabel label = ParticleLabel::Unknown;
        if ((particles.flags[i] & kParticleFlagFrozen) != 0u) {
            label = ParticleLabel::Frozen;
        } else if (!granular && !solid(i) && mist(i)) {
            label = ParticleLabel::Mist;
        } else if (classify && bins.particle_bucket[i] != kInvalidParticle) {
            unsigned neighbors = 0u;
            const auto& own_bucket = bins.table[bins.particle_bucket[i]];
            const Cell center = own_bucket.cell;
            const Vec3& p = particles.position[i];
            const auto gather = [&](std::size_t head) {
                for (std::size_t j = head;
                     j != kInvalidParticle && neighbors < kParticleLabelRejoinNeighbors;
                     j = bins.next[j]) {
                    if (i == j) {
                        continue;
                    }
                    const Vec3& q = particles.position[j];
                    const double dx = static_cast<double>(p.x) - q.x;
                    const double dy = static_cast<double>(p.y) - q.y;
                    const double dz = static_cast<double>(p.z) - q.z;
                    if (dx * dx + dy * dy + dz * dz <= radius_squared) {
                        ++neighbors;
                    }
                }
            };
            // Dense liquid usually supplies six neighbours in its own cell.
            // Check it directly before hashing the 26 surrounding cells. The
            // former corner-first order paid for remote candidates first.
            gather(own_bucket.head);
            center_resolved += neighbors >= kParticleLabelRejoinNeighbors ? 1u : 0u;
            for (int z = -1; z <= 1 && neighbors < kParticleLabelRejoinNeighbors; ++z) {
                for (int y = -1; y <= 1 && neighbors < kParticleLabelRejoinNeighbors; ++y) {
                    for (int x = -1; x <= 1 && neighbors < kParticleLabelRejoinNeighbors; ++x) {
                        if (x == 0 && y == 0 && z == 0) {
                            continue;
                        }
                        gather(bins.findHead({center.x + x, center.y + y, center.z + z}));
                    }
                }
            }
            if (neighbors <= kParticleLabelDetachNeighbors) {
                label = ParticleLabel::Spray;
            } else if (neighbors >= kParticleLabelRejoinNeighbors) {
                label = ParticleLabel::Body;
            } else {
                label = previous == ParticleLabel::Spray
                    ? ParticleLabel::Spray : ParticleLabel::Body;
            }
        }
        changed += previous != label ? 1u : 0u;
        setParticleLabel(particles.flags[i], label);
    }
    stats.changed = changed;
    stats.center_resolved = center_resolved;
    const auto end = Clock::now();
    stats.classify_milliseconds = std::chrono::duration<double, std::milli>(end - bin_end).count();
    stats.milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    return stats;
}

ParticleLabelCounts countParticleLabels(const FluidParticles& particles) {
    ParticleLabelCounts counts{};
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto label = i < particles.flags.size()
            ? particleLabel(particles.flags[i]) : ParticleLabel::Unknown;
        ++counts[static_cast<std::size_t>(label)];
    }
    return counts;
}

ParticleLabelCounts countSecondaryParticleLabels(const FoamParticles& particles) {
    ParticleLabelCounts counts{};
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto label = i < particles.type.size()
            ? secondaryParticleLabel(particles.type[i]) : ParticleLabel::Unknown;
        ++counts[static_cast<std::size_t>(label)];
    }
    return counts;
}

} // namespace Fluid
} // namespace RayTrophiSim
