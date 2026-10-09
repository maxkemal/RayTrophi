#include "Fluid/SparseMacSolidWeightsGpu.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

using RayTrophiSim::Fluid::packSparseMacSolidWeights;

namespace {

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

void checkLayout(const std::array<int, 3>& cells, std::mt19937& random) {
    const std::array<int, 3> tiles = {
        (cells[0] + 7) / 8, (cells[1] + 7) / 8, (cells[2] + 7) / 8
    };
    std::vector<uint32_t> keys(tiles[0] * tiles[1] * tiles[2]);
    std::iota(keys.begin(), keys.end(), 0u);
    std::array<std::vector<uint8_t>, 3> weights;
    for (int axis = 0; axis < 3; ++axis) {
        auto dims = cells;
        ++dims[axis];
        weights[axis].resize(dims[0] * dims[1] * dims[2]);
        for (auto& value : weights[axis]) {
            value = static_cast<uint8_t>(random() % 256u);
        }
    }
    const std::array<const std::vector<uint8_t>*, 3> input = {
        &weights[0], &weights[1], &weights[2]
    };
    std::array<std::vector<float>, 3> pages;
    std::string error;
    // Full, reordered, retired/disconnected and empty topologies.
    for (int pass = 0; pass < 4; ++pass) {
        if (pass == 1) {
            std::shuffle(keys.begin(), keys.end(), random);
        } else if (pass == 2) {
            keys.resize((keys.size() + 1u) / 2u);
            std::shuffle(keys.begin(), keys.end(), random);
        } else if (pass == 3) {
            keys.clear();
        }
        error = "old error";
        require(packSparseMacSolidWeights(cells, keys, input, pages, error),
                "Valid MAC topology rejected");
        require(error.empty(), "Successful packing retained an old error");
        for (int axis = 0; axis < 3; ++axis) {
            require(pages[axis].size() == keys.size() * 576u, "Noncompact page allocation");
            auto dims = cells;
            ++dims[axis];
            std::vector<bool> visited(pages[axis].size());
            // Oracle enumerates dense faces, independently of the packer's
            // slot/lane traversal. It also verifies ownership at tile planes
            // and the final domain face of clipped tiles.
            for (int z = 0; z < dims[2]; ++z) {
                for (int y = 0; y < dims[1]; ++y) {
                    for (int x = 0; x < dims[0]; ++x) {
                        const std::array<int, 3> face = {x, y, z};
                        auto owner = face;
                        owner[axis] = std::min(owner[axis], cells[axis] - 1);
                        const uint32_t key = owner[0] / 8 + tiles[0] *
                            (owner[1] / 8 + tiles[1] * (owner[2] / 8));
                        const auto found = std::find(keys.begin(), keys.end(), key);
                        if (found == keys.end()) {
                            continue;
                        }
                        std::array<int, 3> local{};
                        std::array<int, 3> local_dims = {8, 8, 8};
                        ++local_dims[axis];
                        for (int coordinate = 0; coordinate < 3; ++coordinate) {
                            local[coordinate] = face[coordinate] - owner[coordinate] / 8 * 8;
                        }
                        const std::size_t address = std::size_t(found - keys.begin()) * 576u +
                            local[0] + local_dims[0] * (local[1] + local_dims[1] * local[2]);
                        const std::size_t dense = x + dims[0] * (y + dims[1] * z);
                        const float expected = float(weights[axis][dense]) * (1.0f / 255.0f);
                        require(!visited[address], "Two faces share a compact weight slot");
                        require(pages[axis][address] == expected, "Fractional face weight changed");
                        visited[address] = true;
                    }
                }
            }
            for (std::size_t address = 0; address < visited.size(); ++address) {
                if (!visited[address]) {
                    require(pages[axis][address] == 1.0f, "Padding became a closed wall");
                }
            }
        }
    }
    // Rejection is atomic: do not publish partially repacked axes.
    const auto before = pages;
    require(!packSparseMacSolidWeights(cells, {0u, 0u}, input, pages, error),
            "Duplicate MAC tile accepted");
    require(!error.empty() && pages == before, "Rejected topology mutated output");
    require(!packSparseMacSolidWeights(cells, {uint32_t(tiles[0] * tiles[1] * tiles[2])},
                                      input, pages, error), "Out-of-range tile accepted");
    auto bad_input = input;
    bad_input[1] = nullptr;
    require(!packSparseMacSolidWeights(cells, {0u}, bad_input, pages, error),
            "Missing host weight accepted");
    weights[0].pop_back();
    require(!packSparseMacSolidWeights(cells, {0u}, input, pages, error),
            "Mismatched face layout accepted");
    require(!packSparseMacSolidWeights({0, cells[1], cells[2]}, {}, input, pages, error),
            "Zero grid dimension accepted");
    require(!packSparseMacSolidWeights({2147483647, 2, 2}, {}, input, pages, error),
            "Signed shader-width overflow accepted");
    require(pages == before, "Validation failure changed canonical output");
}

} // namespace

int main() {
    try {
        std::mt19937 random(20261009);
        const std::array<int, 3> layouts[] = {
            {1, 1, 1}, {8, 8, 8}, {16, 8, 17}, {17, 10, 9}, {25, 19, 8}
        };
        for (const auto& cells : layouts) {
            checkLayout(cells, random);
        }
        for (int sample = 0; sample < 30; ++sample) {
            const std::array<int, 3> cells = {
                int(random() % 25u) + 1, int(random() % 25u) + 1, int(random() % 25u) + 1
            };
            checkLayout(cells, random);
        }
        std::cout << "PASS compact solid weights: fractional faces, tile ownership, "
                     "reordering/retirement, padding and atomic validation\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
