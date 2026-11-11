#include "Fluid/SparseMacSolidWeightSource.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace RayTrophiSim::Fluid {
bool packSparseMacSolidWeightSource(const std::array<int, 3>& cells,
    const std::array<const std::vector<uint8_t>*, 3>& weights,
    std::vector<uint32_t>& output, std::string& error, uint64_t max_bytes) {
    error.clear();
    std::array<uint64_t, 3> dims{}, tiles{};
    uint64_t count = 1;
    for (int axis = 0; axis < 3; ++axis) {
        if (cells[axis] <= 0 || count > uint64_t(INT32_MAX) / uint64_t(cells[axis])) {
            error = "solid weight source: invalid or oversized dimensions";
            return false;
        }
        dims[axis] = uint64_t(cells[axis]);
        tiles[axis] = (dims[axis] + 7u) / 8u;
        count *= dims[axis];
    }
    for (int axis = 0; axis < 3; ++axis) {
        const auto faces = count / dims[axis] * (dims[axis] + 1u);
        if (!weights[axis] || weights[axis]->size() != faces) {
            error = "solid weight source: host face array size mismatch";
            return false;
        }
    }
    const uint64_t tile_count = tiles[0] * tiles[1] * tiles[2];
    if (tile_count > UINT32_MAX) {
        error = "solid weight source: lookup exceeds shader width";
        return false;
    }
    if (max_bytes != 0 && tile_count * sizeof(uint32_t) > max_bytes) {
        error = "solid weight source: lookup exceeds byte budget";
        return false;
    }
    std::vector<uint32_t> candidate(static_cast<std::size_t>(tile_count), 0u);
    for (uint64_t key = 0; key < tile_count; ++key) {
        const std::array<uint64_t, 3> base = {key % tiles[0] * 8u,
            key / tiles[0] % tiles[1] * 8u, key / (tiles[0] * tiles[1]) * 8u};
        std::array<uint32_t, kSolidWeightWordsPerTile> page;
        page.fill(UINT32_MAX);
        bool non_open = false;
        for (int axis = 0; axis < 3; ++axis) {
            auto extent = dims;
            ++extent[axis];
            std::array<uint64_t, 3> local_dims = {8u, 8u, 8u};
            ++local_dims[axis];
            for (uint32_t local = 0; local < 576u; ++local) {
                const std::array<uint64_t, 3> face = {base[0] + local % local_dims[0],
                    base[1] + local / local_dims[0] % local_dims[1],
                    base[2] + local / (local_dims[0] * local_dims[1])};
                if (face[0] >= extent[0] || face[1] >= extent[1] || face[2] >= extent[2]) {
                    continue;
                }
                auto owner = face;
                owner[axis] = std::min(owner[axis], dims[axis] - 1u);
                const auto owner_key = owner[0] / 8u + tiles[0] *
                    (owner[1] / 8u + tiles[1] * (owner[2] / 8u));
                if (owner_key != key) {
                    continue;
                }
                const auto index = face[0] + extent[0] * (face[1] + extent[1] * face[2]);
                const uint32_t byte = (*weights[axis])[static_cast<std::size_t>(index)];
                const auto word = axis * kSolidWeightWordsPerAxis + local / 4u;
                const auto shift = (local % 4u) * 8u;
                page[word] = (page[word] & ~(255u << shift)) | (byte << shift);
                non_open = non_open || byte != 255u;
            }
        }
        if (non_open) {
            if (candidate.size() > UINT32_MAX - kSolidWeightWordsPerTile) {
                error = "solid weight source: packed offsets exceed shader width";
                return false;
            }
            if (max_bytes != 0 &&
                (candidate.size() + kSolidWeightWordsPerTile) * sizeof(uint32_t) > max_bytes) {
                error = "solid weight source: sparse pages exceed byte budget";
                return false;
            }
            candidate[static_cast<std::size_t>(key)] = static_cast<uint32_t>(candidate.size());
            candidate.insert(candidate.end(), page.begin(), page.end());
        }
    }
    output = std::move(candidate);
    return true;
}
} // namespace RayTrophiSim::Fluid
