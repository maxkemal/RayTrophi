#include "Fluid/SparseMacSolidWeightsGpu.h"
#include "FluidGrid.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace RayTrophiSim::Fluid {

bool packSparseMacSolidWeights(
    const std::array<int, 3>& cells,
    const std::vector<uint32_t>& tile_keys,
    const std::array<const std::vector<uint8_t>*, 3>& weights,
    std::array<std::vector<float>, 3>& pages,
    std::string& error) {
    error.clear();
    const auto fail = [&](const char* message) {
        error = message;
        return false;
    };
    std::array<uint64_t, 3> dims{};
    std::array<uint64_t, 3> tiles{};
    uint64_t cell_count = 1;
    for (int axis = 0; axis < 3; ++axis) {
        if (cells[axis] <= 0) {
            return fail("compact solid weights: invalid grid dimensions");
        }
        dims[axis] = static_cast<uint64_t>(cells[axis]);
        // Match the signed cell-width gate of the pressure consumer.
        if (cell_count > uint64_t(std::numeric_limits<int32_t>::max()) / dims[axis]) {
            return fail("compact solid weights: grid exceeds signed shader width");
        }
        cell_count *= dims[axis];
        tiles[axis] = (dims[axis] + 7u) / 8u;
    }
    const uint64_t tile_count = tiles[0] * tiles[1] * tiles[2];
    if (tile_keys.size() > tile_count || tile_keys.size() >
        std::size_t(std::numeric_limits<int32_t>::max()) / 576u) {
        return fail("compact solid weights: tile count exceeds page index width");
    }
    for (int axis = 0; axis < 3; ++axis) {
        const uint64_t faces = cell_count / dims[axis] * (dims[axis] + 1u);
        if (!weights[axis] || weights[axis]->size() != faces) {
            return fail("compact solid weights: host face array size mismatch");
        }
    }
    // A repeated slot key would alias ownership. Check active keys only.
    auto sorted_keys = tile_keys;
    std::sort(sorted_keys.begin(), sorted_keys.end());
    if ((!sorted_keys.empty() && sorted_keys.back() >= tile_count) ||
        std::adjacent_find(sorted_keys.begin(), sorted_keys.end()) != sorted_keys.end()) {
        return fail("compact solid weights: invalid or duplicate tile key");
    }
    const std::size_t values = tile_keys.size() * 576u;
    std::array<std::vector<float>, 3> candidate;
    for (auto& page : candidate) {
        page.assign(values, 1.0f);
    }
    for (std::size_t slot = 0; slot < tile_keys.size(); ++slot) {
        const uint64_t key = tile_keys[slot];
        const std::array<uint64_t, 3> base = {
            (key % tiles[0]) * 8u,
            ((key / tiles[0]) % tiles[1]) * 8u,
            (key / (tiles[0] * tiles[1])) * 8u
        };
        for (int axis = 0; axis < 3; ++axis) {
            auto face_dims = dims;
            ++face_dims[axis];
            std::array<uint64_t, 3> local_dims = {8u, 8u, 8u};
            ++local_dims[axis];
            for (std::size_t local = 0; local < 576u; ++local) {
                const std::array<uint64_t, 3> face = {
                    base[0] + local % local_dims[0],
                    base[1] + (local / local_dims[0]) % local_dims[1],
                    base[2] + local / (local_dims[0] * local_dims[1])
                };
                if (face[0] >= face_dims[0] || face[1] >= face_dims[1] ||
                    face[2] >= face_dims[2]) {
                    continue;
                }
                auto owner = face;
                owner[axis] = std::min(owner[axis], dims[axis] - 1u);
                const uint64_t owner_key = owner[0] / 8u + tiles[0] *
                    (owner[1] / 8u + tiles[1] * (owner[2] / 8u));
                if (owner_key != key) {
                    continue;
                }
                const std::size_t dense = static_cast<std::size_t>(face[0] +
                    face_dims[0] * (face[1] + face_dims[1] * face[2]));
                candidate[axis][slot * 576u + local] =
                    FluidSim::FluidGrid::weightToFloat((*weights[axis])[dense]);
            }
        }
    }
    pages = std::move(candidate);
    return true;
}

} // namespace RayTrophiSim::Fluid
