#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace RayTrophiSim::Fluid::Sparse {

// Solver storage, independent of the dense FluidGrid and its activity threshold.
// Callers supply the complete physical support, including pressure, emitters,
// moving boundaries and advection halos. A density threshold is not that support.
constexpr int tileEdge = 8;

struct Coordinate {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const Coordinate& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator<(const Coordinate& other) const {
        if (z != other.z) {
            return z < other.z;
        }
        if (y != other.y) {
            return y < other.y;
        }
        return x < other.x;
    }
};

struct CoordinateHash {
    std::size_t operator()(const Coordinate& coordinate) const {
        uint64_t value = 1469598103934665603ull;
        for (int axis : {coordinate.x, coordinate.y, coordinate.z}) {
            value ^= static_cast<uint32_t>(axis);
            value *= 1099511628211ull;
        }
        return static_cast<std::size_t>(value);
    }
};

// Half-open physical cell bounds. No dense tile bitmap or domain-size reserve.
struct CellBox {
    std::array<int, 3> begin{};
    std::array<int, 3> end{};
};

// Conservative quadratic transfer + pressure neighbour + travel support.
// This seed describes liquid parcels, not the complete influence region of gas
// pressure. Gas callers must also include their pressure domain explicitly.
inline CellBox sweptSupport(const std::array<double, 3>& position,
                            const std::array<double, 3>& velocity,
                            const std::array<double, 3>& origin,
                            const std::array<int, 3>& dimensions,
                            double voxel, double dt) {
    if (!std::isfinite(voxel) || voxel <= 0.0 || !std::isfinite(dt) || dt < 0.0) {
        throw std::invalid_argument("Sparse support requires finite voxel and timestep");
    }
    CellBox result;
    for (int axis = 0; axis < 3; ++axis) {
        if (dimensions[axis] <= 0 || !std::isfinite(position[axis]) ||
            !std::isfinite(velocity[axis]) || !std::isfinite(origin[axis])) {
            throw std::invalid_argument("Sparse support requires finite spatial data");
        }
        const double cell = (position[axis] - origin[axis]) / voxel;
        const double halo = 2.0 + std::ceil(std::abs(velocity[axis]) * dt / voxel);
        const double low = std::floor(cell) - halo;
        const double high = std::floor(cell) + halo + 2.0;
        if (!std::isfinite(low) || !std::isfinite(high)) {
            throw std::overflow_error("Sparse support travel is not representable");
        }
        result.begin[axis] = static_cast<int>(std::clamp(low, 0.0, double(dimensions[axis])));
        result.end[axis] = static_cast<int>(std::clamp(high, 0.0, double(dimensions[axis])));
    }
    return result;
}

class Topology {
public:
    Topology(std::array<int, 3> dimensions, const std::vector<CellBox>& support)
        : dimensions_(dimensions) {
        for (int dimension : dimensions_) {
            if (dimension <= 0) {
                throw std::invalid_argument("Sparse grid dimensions must be positive");
            }
        }
        for (const auto& box : support) {
            std::array<int, 3> first{};
            std::array<int, 3> last{};
            bool empty = false;
            for (int axis = 0; axis < 3; ++axis) {
                if (box.end[axis] < box.begin[axis]) {
                    throw std::invalid_argument("Sparse support bounds are reversed");
                }
                const int low = std::clamp(box.begin[axis], 0, dimensions_[axis]);
                const int high = std::clamp(box.end[axis], 0, dimensions_[axis]);
                empty = empty || low >= high;
                first[axis] = low / tileEdge;
                last[axis] = high > low ? (high - 1) / tileEdge : first[axis];
            }
            if (empty) {
                continue;
            }
            for (int z = first[2]; z <= last[2]; ++z) {
                for (int y = first[1]; y <= last[1]; ++y) {
                    for (int x = first[0]; x <= last[0]; ++x) {
                        slots_.emplace(Coordinate{x, y, z}, 0);
                    }
                }
            }
        }
        tiles_.reserve(slots_.size());
        for (const auto& entry : slots_) {
            tiles_.push_back(entry.first);
        }
        std::sort(tiles_.begin(), tiles_.end());
        for (std::size_t slot = 0; slot < tiles_.size(); ++slot) {
            slots_.at(tiles_[slot]) = slot;
        }
    }

    const std::array<int, 3>& dimensions() const {
        return dimensions_;
    }

    const std::vector<Coordinate>& tiles() const {
        return tiles_;
    }

    std::size_t slot(const Coordinate& coordinate) const {
        const auto found = slots_.find(coordinate);
        return found == slots_.end() ? missing : found->second;
    }

    static constexpr std::size_t missing = std::numeric_limits<std::size_t>::max();

private:
    std::array<int, 3> dimensions_;
    std::vector<Coordinate> tiles_;
    std::unordered_map<Coordinate, std::size_t, CoordinateHash> slots_;
};

enum class Location { Cell, XFace, YFace, ZFace };

inline int faceAxis(Location location) {
    switch (location) {
    case Location::XFace:
        return 0;
    case Location::YFace:
        return 1;
    case Location::ZFace:
        return 2;
    default:
        return -1;
    }
}

// Scalar pages can represent density, fuel, temperature, pressure or a MAC
// component. Background is explicit (e.g. ambient temperature rather than zero).
// Interior faces belong to the tile on their positive side. The domain's final
// face belongs to the final cell's tile. Thus every physical face has one owner;
// an unused extra row in a face page is padding, never a second authority.
class Field {
public:
    Field(std::shared_ptr<const Topology> topology, Location location, float background)
        : topology_(std::move(topology)), location_(location), background_(background) {
        if (!topology_) {
            throw std::invalid_argument("Sparse field requires a topology");
        }
        if (!std::isfinite(background_)) {
            throw std::invalid_argument("Sparse field background must be finite");
        }
        stride_.fill(tileEdge);
        const int axis = faceAxis(location_);
        if (axis >= 0) {
            ++stride_[axis];
        }
        page_size_ = std::size_t(stride_[0]) * stride_[1] * stride_[2];
        values_.assign(allocationSize(*topology_), background_);
    }

    const Topology& topology() const {
        return *topology_;
    }

    std::size_t allocatedValues() const {
        return values_.size();
    }

    float read(int x, int y, int z) const {
        const auto index = address(x, y, z);
        return index == Topology::missing ? background_ : values_[index];
    }

    void write(int x, int y, int z, float value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Sparse field value must be finite");
        }
        const auto index = address(x, y, z);
        if (index == Topology::missing) {
            throw std::out_of_range("Sparse write requires an allocated physical tile");
        }
        values_[index] = value;
    }

    // Preserve pages by coordinate, not transient slot. Refuse to silently lose
    // mass, momentum, heat or pressure when a caller supplies incomplete support.
    // The solver must explicitly reset an eligible page before retiring it.
    // Build replacement storage first so rejection/allocation failure is atomic.
    void rebind(std::shared_ptr<const Topology> next) {
        if (!next || next->dimensions() != topology_->dimensions()) {
            throw std::invalid_argument("Sparse remap requires the same physical layout");
        }
        std::vector<float> replacement(allocationSize(*next), background_);
        for (std::size_t old_slot = 0; old_slot < topology_->tiles().size(); ++old_slot) {
            const auto new_slot = next->slot(topology_->tiles()[old_slot]);
            const auto begin = values_.begin() + old_slot * page_size_;
            const auto end = begin + page_size_;
            if (new_slot == Topology::missing) {
                if (std::any_of(begin, end, [this](float value) {
                        return value != background_;
                    })) {
                    throw std::runtime_error("Sparse remap would discard a live field page");
                }
                continue;
            }
            std::copy(begin, end, replacement.begin() + new_slot * page_size_);
        }
        topology_ = std::move(next);
        values_.swap(replacement);
    }

private:
    std::size_t allocationSize(const Topology& topology) const {
        if (topology.tiles().size() > values_.max_size() / page_size_) {
            throw std::length_error("Sparse field exceeds host addressable storage");
        }
        return topology.tiles().size() * page_size_;
    }

    std::size_t address(int x, int y, int z) const {
        std::array<int, 3> cell = {x, y, z};
        const auto& dimensions = topology_->dimensions();
        const int face_axis = faceAxis(location_);
        for (int axis = 0; axis < 3; ++axis) {
            const bool final_face = axis == face_axis && cell[axis] == dimensions[axis];
            if (cell[axis] < 0 || (cell[axis] >= dimensions[axis] && !final_face)) {
                return Topology::missing;
            }
        }
        auto owner = cell;
        if (face_axis >= 0) {
            owner[face_axis] = std::min(owner[face_axis], dimensions[face_axis] - 1);
        }
        const Coordinate tile = {
            owner[0] / tileEdge, owner[1] / tileEdge, owner[2] / tileEdge
        };
        const auto slot = topology_->slot(tile);
        if (slot == Topology::missing) {
            return slot;
        }
        cell[0] -= tile.x * tileEdge;
        cell[1] -= tile.y * tileEdge;
        cell[2] -= tile.z * tileEdge;
        return slot * page_size_ + cell[0] +
            std::size_t(stride_[0]) * (cell[1] + std::size_t(stride_[1]) * cell[2]);
    }

    std::shared_ptr<const Topology> topology_;
    Location location_;
    float background_;
    std::array<int, 3> stride_{};
    std::size_t page_size_ = 0;
    std::vector<float> values_;
};

} // namespace RayTrophiSim::Fluid::Sparse
