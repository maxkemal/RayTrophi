#include "Fluid/SparseGridStorage.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace RayTrophiSim::Fluid::Sparse {
namespace {
std::size_t pageValues(Location location) {
    return faceAxis(location) >= 0 ? 576u : 512u;
}

ChannelDescriptions commonChannels(float viscosity) {
    if (!std::isfinite(viscosity) || viscosity < 0.0f) {
        throw std::invalid_argument("Sparse viscosity must be finite and nonnegative");
    }
    ChannelDescriptions channels{};
    const auto enable = [&](Channel channel, Location location, float background) {
        channels[static_cast<std::size_t>(channel)] = {location, background, true};
    };
    for (Channel channel : {Channel::Density, Channel::Pressure, Channel::Divergence,
                            Channel::FluidMask, Channel::PorousDrag,
                            Channel::PorousSaturation, Channel::PorousReaction,
                            Channel::SolidVelocityX, Channel::SolidVelocityY,
                            Channel::SolidVelocityZ}) {
        enable(channel, Location::Cell, 0.0f);
    }
    enable(Channel::SolidPhi, Location::Cell, std::numeric_limits<float>::max());
    enable(Channel::Viscosity, Location::Cell, viscosity);
    const std::array<Location, 3> locations = {
        Location::XFace, Location::YFace, Location::ZFace};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        enable(static_cast<Channel>(static_cast<std::size_t>(Channel::VelocityX) + axis),
               locations[axis], 0.0f);
        enable(static_cast<Channel>(static_cast<std::size_t>(Channel::OpenWeightX) + axis),
               locations[axis], 1.0f);
    }
    return channels;
}
} // namespace

ChannelDescriptions liquidChannels(float viscosity) {
    auto channels = commonChannels(viscosity);
    const std::array<Location, 3> locations = {
        Location::XFace, Location::YFace, Location::ZFace};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        channels[static_cast<std::size_t>(Channel::FlipX) + axis] = {
            locations[axis], 0.0f, true};
    }
    return channels;
}

ChannelDescriptions gasChannels(float ambient_temperature, float viscosity) {
    if (!std::isfinite(ambient_temperature) || ambient_temperature < 0.0f) {
        throw std::invalid_argument("Sparse ambient temperature must be finite and nonnegative");
    }
    auto channels = commonChannels(viscosity);
    channels[static_cast<std::size_t>(Channel::Temperature)] = {
        Location::Cell, ambient_temperature, true};
    channels[static_cast<std::size_t>(Channel::Fuel)] = {Location::Cell, 0.0f, true};
    channels[static_cast<std::size_t>(Channel::Flame)] = {Location::Cell, 0.0f, true};
    const std::array<Location, 3> locations = {
        Location::XFace, Location::YFace, Location::ZFace};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        channels[static_cast<std::size_t>(Channel::AdvectionX) + axis] = {
            locations[axis], 0.0f, true};
        channels[static_cast<std::size_t>(Channel::CorrectorX) + axis] = {
            locations[axis], 0.0f, true};
    }
    channels[static_cast<std::size_t>(Channel::ScalarAdvection)] = {
        Location::Cell, 0.0f, true};
    channels[static_cast<std::size_t>(Channel::ScalarCorrector)] = {
        Location::Cell, 0.0f, true};
    return channels;
}

struct GridStorage::Ledger {
    uint64_t value_budget = 0;
    std::atomic<uint64_t> value_bytes{0};
};

struct GridStorage::Page {
    std::vector<float> values;
    std::shared_ptr<Ledger> ledger;
    uint64_t bytes = 0;

    ~Page() {
        if (ledger) {
            ledger->value_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        }
    }
};

struct GridStorage::State {
    using Pages = std::unordered_map<Coordinate, std::shared_ptr<Page>, CoordinateHash>;
    std::shared_ptr<const Topology> topology;
    ChannelDescriptions descriptions{};
    std::array<Pages, channelCount> pages;
    uint64_t generation = 1;
};

std::size_t GridStorage::index(Channel channel) {
    const auto value = static_cast<std::size_t>(channel);
    if (value >= channelCount) {
        throw std::out_of_range("Invalid sparse channel");
    }
    return value;
}

GridStorage::GridStorage(std::shared_ptr<const Topology> topology,
                         const ChannelDescriptions& channels, uint64_t value_budget_bytes)
    : ledger_(std::make_shared<Ledger>()), state_(std::make_shared<State>()) {
    if (!topology) {
        throw std::invalid_argument("Sparse grid requires a topology");
    }
    for (const auto& channel : channels) {
        if (!std::isfinite(channel.background) ||
            (channel.location != Location::Cell && faceAxis(channel.location) < 0)) {
            throw std::invalid_argument("Invalid sparse channel description");
        }
    }
    ledger_->value_budget = value_budget_bytes;
    state_->topology = std::move(topology);
    state_->descriptions = channels;
}

void GridStorage::ensureUniqueState() {
    if (state_.use_count() != 1) {
        state_ = std::make_shared<State>(*state_);
    }
}

std::shared_ptr<GridStorage::Page> GridStorage::makePage(std::size_t values, float background) {
    auto page = std::make_shared<Page>();
    page->ledger = ledger_;
    const auto reserve = [&](uint64_t bytes) {
        auto retained = ledger_->value_bytes.load(std::memory_order_relaxed);
        for (;;) {
            if (bytes > std::numeric_limits<uint64_t>::max() - retained ||
                (ledger_->value_budget != 0 && bytes > ledger_->value_budget -
                    std::min(ledger_->value_budget, retained))) {
                throw std::length_error("Sparse pages exceed the authored value budget");
            }
            if (ledger_->value_bytes.compare_exchange_weak(
                    retained, retained + bytes, std::memory_order_relaxed)) {
                page->bytes += bytes;
                return;
            }
        }
    };
    reserve(values * sizeof(float));
    page->values.assign(values, background);
    const auto capacity_bytes = page->values.capacity() * sizeof(float);
    if (capacity_bytes > page->bytes) {
        reserve(capacity_bytes - page->bytes);
    }
    return page;
}

std::pair<Coordinate, std::size_t> GridStorage::physicalAddress(
    const State& state, Channel channel, int x, int y, int z) {
    const auto& description = state.descriptions[index(channel)];
    if (!description.enabled) {
        throw std::invalid_argument("Sparse channel is disabled for this phase");
    }
    const auto& cells = state.topology->dimensions();
    const int face_axis = faceAxis(description.location);
    std::array<int, 3> face = {x, y, z};
    auto cell = face;
    for (int axis = 0; axis < 3; ++axis) {
        if (face[axis] < 0 ||
            (face[axis] >= cells[axis] && !(axis == face_axis && face[axis] == cells[axis]))) {
            throw std::out_of_range("Sparse field coordinate is outside the physical grid");
        }
    }
    if (face_axis >= 0) {
        cell[face_axis] = std::min(cell[face_axis], cells[face_axis] - 1);
    }
    const Coordinate tile = {cell[0] / 8, cell[1] / 8, cell[2] / 8};
    const std::array<int, 3> tile_origin = {tile.x * 8, tile.y * 8, tile.z * 8};
    std::array<std::size_t, 3> dimensions = {8, 8, 8};
    if (face_axis >= 0) {
        ++dimensions[face_axis];
    }
    for (int axis = 0; axis < 3; ++axis) {
        face[axis] -= tile_origin[axis];
    }
    const auto local = static_cast<std::size_t>(face[0]) + dimensions[0] *
        (static_cast<std::size_t>(face[1]) + dimensions[1] * face[2]);
    return {tile, local};
}

float GridStorage::readState(const State& state, Channel channel, int x, int y, int z) {
    const auto address = physicalAddress(state, channel, x, y, z);
    const auto& pages = state.pages[index(channel)];
    const auto page = pages.find(address.first);
    return page == pages.end() ? state.descriptions[index(channel)].background
                              : page->second->values[address.second];
}

float GridStorage::read(Channel channel, int x, int y, int z) const {
    return readState(*state_, channel, x, y, z);
}

void GridStorage::write(Channel channel, int x, int y, int z, float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("Sparse field write must be finite");
    }
    const auto address = physicalAddress(*state_, channel, x, y, z);
    if (state_->topology->slot(address.first) == Topology::missing) {
        throw std::out_of_range("Sparse write requires allocated physical support");
    }
    const auto field = index(channel);
    const auto description = state_->descriptions[field];
    if (value == description.background && state_->pages[field].count(address.first) == 0) {
        return;
    }
    ensureUniqueState();
    auto& pages = state_->pages[field];
    const auto found = pages.find(address.first);
    if (found == pages.end()) {
        auto page = makePage(pageValues(description.location), description.background);
        page->values[address.second] = value;
        pages.emplace(address.first, std::move(page));
    } else {
        if (found->second.use_count() != 1) {
            auto page = makePage(found->second->values.size(), description.background);
            std::copy(found->second->values.begin(), found->second->values.end(),
                      page->values.begin());
            found->second = std::move(page);
        }
        found->second->values[address.second] = value;
    }
}

bool GridStorage::isBackground(const Page& page, float background) {
    return std::all_of(page.values.begin(), page.values.end(),
                       [background](float value) { return value == background; });
}

void GridStorage::clear(Channel channel) {
    const auto field = index(channel);
    if (!state_->descriptions[field].enabled) {
        throw std::invalid_argument("Cannot clear a disabled sparse channel");
    }
    ensureUniqueState();
    state_->pages[field].clear();
}

void GridStorage::pruneBackgroundPages() {
    ensureUniqueState();
    for (std::size_t field = 0; field < channelCount; ++field) {
        auto& pages = state_->pages[field];
        for (auto page = pages.begin(); page != pages.end();) {
            if (isBackground(*page->second, state_->descriptions[field].background)) {
                page = pages.erase(page);
            } else {
                ++page;
            }
        }
    }
}

void GridStorage::rebind(std::shared_ptr<const Topology> next) {
    if (!next || next->dimensions() != state_->topology->dimensions()) {
        throw std::invalid_argument("Sparse transaction requires the same physical layout");
    }
    auto candidate = std::make_shared<State>(*state_);
    for (std::size_t field = 0; field < channelCount; ++field) {
        auto& pages = candidate->pages[field];
        for (auto page = pages.begin(); page != pages.end();) {
            if (next->slot(page->first) != Topology::missing) {
                ++page;
            } else if (isBackground(*page->second, candidate->descriptions[field].background)) {
                page = pages.erase(page);
            } else {
                throw std::runtime_error("Sparse topology transaction would discard a live channel");
            }
        }
    }
    if (candidate->generation == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("Sparse topology generation overflow");
    }
    candidate->topology = std::move(next);
    ++candidate->generation;
    state_.swap(candidate);
}

void GridStorage::setValueBudget(uint64_t value_budget_bytes) {
    if (value_budget_bytes != 0 &&
        ledger_->value_bytes.load(std::memory_order_relaxed) > value_budget_bytes) {
        throw std::length_error("Live sparse pages and snapshots exceed the requested budget");
    }
    ledger_->value_budget = value_budget_bytes;
}

GridStorage::Snapshot::Snapshot(std::shared_ptr<const State> state) : state_(std::move(state)) {
}

float GridStorage::Snapshot::read(Channel channel, int x, int y, int z) const {
    return readState(*state_, channel, x, y, z);
}

const Topology& GridStorage::Snapshot::topology() const {
    return *state_->topology;
}

uint64_t GridStorage::Snapshot::generation() const {
    return state_->generation;
}

GridStorage::Snapshot GridStorage::snapshot() const {
    return Snapshot(state_);
}

StorageStatistics GridStorage::statistics() const {
    StorageStatistics result;
    result.generation = state_->generation;
    result.topology_tiles = state_->topology->tiles().size();
    result.retained_value_bytes = ledger_->value_bytes.load(std::memory_order_relaxed);
    for (const auto& pages : state_->pages) {
        result.resident_pages += pages.size();
        for (const auto& page : pages) {
            result.current_value_bytes += page.second->bytes;
        }
    }
    return result;
}

ChannelDescription GridStorage::description(Channel channel) const {
    return state_->descriptions[index(channel)];
}

void GridStorage::exportPage(Channel channel, const Coordinate& tile,
                             float* destination, std::size_t value_count) const {
    const auto field = index(channel);
    const auto& description = state_->descriptions[field];
    if (!destination || !description.enabled ||
        value_count != pageValues(description.location) ||
        state_->topology->slot(tile) == Topology::missing) {
        throw std::invalid_argument("Invalid sparse page export");
    }
    const auto found = state_->pages[field].find(tile);
    if (found == state_->pages[field].end()) {
        std::fill(destination, destination + value_count, description.background);
    } else {
        std::copy(found->second->values.begin(), found->second->values.end(), destination);
    }
}

void GridStorage::importPage(Channel channel, const Coordinate& tile,
                             const float* source, std::size_t value_count) {
    const auto field = index(channel);
    const auto description = state_->descriptions[field];
    if (!source || !description.enabled || value_count != pageValues(description.location) ||
        state_->topology->slot(tile) == Topology::missing ||
        !std::all_of(source, source + value_count, [](float value) { return std::isfinite(value); })) {
        throw std::invalid_argument("Invalid sparse page import");
    }
    // Validate padding as well. Padding is never a second owner or hidden live
    // state that can later prevent a physically empty tile from retiring.
    const auto& cells = state_->topology->dimensions();
    const int face_axis = faceAxis(description.location);
    std::array<int, 3> shape = {8, 8, 8};
    if (face_axis >= 0) {
        ++shape[face_axis];
    }
    bool background = true;
    for (std::size_t local = 0; local < value_count; ++local) {
        const std::array<int64_t, 3> face = {
            int64_t(tile.x) * 8 + int64_t(local % shape[0]),
            int64_t(tile.y) * 8 + int64_t((local / shape[0]) % shape[1]),
            int64_t(tile.z) * 8 + int64_t(local / (shape[0] * shape[1]))};
        bool physical = true;
        for (int axis = 0; axis < 3; ++axis) {
            physical = physical && (face[axis] < cells[axis] ||
                (axis == face_axis && face[axis] == cells[axis]));
        }
        if (physical) {
            auto owner = face;
            if (face_axis >= 0) {
                owner[face_axis] = std::min(owner[face_axis], int64_t(cells[face_axis]) - 1);
            }
            physical = Coordinate{int(owner[0] / 8), int(owner[1] / 8), int(owner[2] / 8)} == tile;
        }
        if (!physical && source[local] != description.background) {
            throw std::invalid_argument("Sparse page padding must contain its background");
        }
        background = background && source[local] == description.background;
    }
    if (background) {
        ensureUniqueState();
        state_->pages[field].erase(tile);
        return;
    }
    auto page = makePage(value_count, description.background);
    std::copy(source, source + value_count, page->values.begin());
    ensureUniqueState();
    state_->pages[field].insert_or_assign(tile, std::move(page));
}

std::shared_ptr<const Topology> gasPressureTopology(const std::array<int, 3>& dimensions) {
    return std::make_shared<Topology>(dimensions,
        std::vector<CellBox>{{{0, 0, 0}, dimensions}});
}

} // namespace RayTrophiSim::Fluid::Sparse
