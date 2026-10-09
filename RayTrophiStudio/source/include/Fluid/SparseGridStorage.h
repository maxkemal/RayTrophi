#pragma once

#include "Fluid/SparseTileGrid.h"

#include <array>
#include <atomic>
#include <memory>
#include <unordered_map>

namespace RayTrophiSim::Fluid::Sparse {

// These are physical channels, not renderer activity flags. In particular,
// pressure support must not be inferred from visible smoke density.
enum class Channel : std::size_t {
    VelocityX, VelocityY, VelocityZ,
    Density, Temperature, Fuel, Flame, Pressure, Divergence,
    FluidMask, SolidPhi, SolidVelocityX, SolidVelocityY, SolidVelocityZ,
    OpenWeightX, OpenWeightY, OpenWeightZ,
    Viscosity, PorousDrag, PorousSaturation, PorousReaction,
    FlipX, FlipY, FlipZ,
    AdvectionX, AdvectionY, AdvectionZ,
    CorrectorX, CorrectorY, CorrectorZ,
    ScalarAdvection, ScalarCorrector,
    Count
};

struct ChannelDescription {
    Location location = Location::Cell;
    float background = 0.0f;
    bool enabled = false;
};

constexpr std::size_t channelCount = static_cast<std::size_t>(Channel::Count);
using ChannelDescriptions = std::array<ChannelDescription, channelCount>;

ChannelDescriptions liquidChannels(float viscosity);
ChannelDescriptions gasChannels(float ambient_temperature, float viscosity);

struct StorageStatistics {
    uint64_t generation = 0;
    uint64_t topology_tiles = 0;
    uint64_t resident_pages = 0;
    // Float page capacity only. Topology/map/container overhead is excluded.
    // retained_value_bytes includes pages retained by immutable snapshots.
    uint64_t current_value_bytes = 0;
    uint64_t retained_value_bytes = 0;
};

class GridStorage {
    struct Page;
    struct Ledger;
    struct State;

public:
    class Snapshot {
    public:
        float read(Channel channel, int x, int y, int z) const;
        const Topology& topology() const;
        uint64_t generation() const;

    private:
        friend class GridStorage;
        explicit Snapshot(std::shared_ptr<const State> state);
        std::shared_ptr<const State> state_;
    };

    GridStorage(std::shared_ptr<const Topology> topology,
                const ChannelDescriptions& channels, uint64_t value_budget_bytes = 0);

    float read(Channel channel, int x, int y, int z) const;
    void write(Channel channel, int x, int y, int z, float value);
    void clear(Channel channel);
    void pruneBackgroundPages();

    // Either every channel sees the new topology, or none does. Retiring any
    // non-background page rejects the complete transaction before publication.
    void rebind(std::shared_ptr<const Topology> next);
    void setValueBudget(uint64_t value_budget_bytes);
    Snapshot snapshot() const;
    StorageStatistics statistics() const;
    ChannelDescription description(Channel channel) const;

    // Explicit tile-page interchange for device adapters. No domain-sized
    // array is exposed and no missing-page write silently discards a value.
    void exportPage(Channel channel, const Coordinate& tile,
                    float* destination, std::size_t value_count) const;
    void importPage(Channel channel, const Coordinate& tile,
                    const float* source, std::size_t value_count);

private:
    static std::size_t index(Channel channel);
    static float readState(const State& state, Channel channel, int x, int y, int z);
    static bool isBackground(const Page& page, float background);
    static std::pair<Coordinate, std::size_t> physicalAddress(
        const State& state, Channel channel, int x, int y, int z);
    std::shared_ptr<Page> makePage(std::size_t values, float background);
    void ensureUniqueState();

    std::shared_ptr<Ledger> ledger_;
    std::shared_ptr<State> state_;
};

// Gas pressure can communicate through all air in the physical domain. This
// conservative support is valid for open, closed and periodic boundaries.
// Adaptive physical bounds can reduce it; a smoke threshold cannot.
std::shared_ptr<const Topology> gasPressureTopology(const std::array<int, 3>& dimensions);

} // namespace RayTrophiSim::Fluid::Sparse
