#include "Fluid/SparseGridStorage.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

using namespace RayTrophiSim::Fluid::Sparse;

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Operation>
void rejects(Operation operation, const char* message) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, message);
}

std::shared_ptr<const Topology> support(const std::array<int, 3>& cells,
                                      std::vector<CellBox> boxes) {
    return std::make_shared<Topology>(cells, boxes);
}
} // namespace

int main() {
    try {
        const std::array<int, 3> cells = {24, 17, 9};
        auto first = support(cells, {{{0, 0, 0}, {8, 8, 8}}});
        auto both = support(cells, {
            {{0, 0, 0}, {8, 8, 8}}, {{16, 8, 0}, {24, 16, 8}}});
        auto second = support(cells, {{{16, 8, 0}, {24, 16, 8}}});
        GridStorage gas(first, gasChannels(293.15f, 0.0f));
        require(gas.statistics().resident_pages == 0, "Background allocated value pages");
        require(gas.read(Channel::Temperature, 3, 2, 1) == 293.15f, "Wrong ambient");
        require(gas.read(Channel::OpenWeightX, 8, 2, 1) == 1.0f, "Missing face default");
        gas.write(Channel::Density, 1, 2, 3, 2.0f);
        gas.write(Channel::Temperature, 1, 2, 3, 350.0f);
        gas.write(Channel::Pressure, 1, 2, 3, 0.25f);
        auto frozen = gas.snapshot();
        gas.write(Channel::Temperature, 1, 2, 3, 400.0f);
        require(frozen.read(Channel::Temperature, 1, 2, 3) == 350.0f,
                "Canonical mutation changed a retained snapshot");
        gas.rebind(both);
        require(gas.read(Channel::Density, 1, 2, 3) == 2.0f, "Remap lost density");
        require(gas.read(Channel::Temperature, 17, 10, 3) == 293.15f,
                "New support did not initialize ambient");
        const auto generation = gas.statistics().generation;
        rejects([&] { gas.rebind(second); }, "Retirement discarded live channels");
        require(gas.statistics().generation == generation,
                "Rejected multi-channel transaction changed topology");
        require(gas.read(Channel::Pressure, 1, 2, 3) == 0.25f,
                "Rejected transaction changed a later pressure channel");
        rejects([&] { gas.write(Channel::Density, 9, 10, 3, 1.0f); },
                "Missing support silently dropped mass");
        gas.clear(Channel::Density);
        gas.clear(Channel::Temperature);
        gas.clear(Channel::Pressure);
        gas.rebind(second);
        require(gas.statistics().current_value_bytes == 0, "Cleared pages remained resident");
        require(gas.statistics().retained_value_bytes != 0, "Snapshot memory was hidden");
        require(frozen.read(Channel::Density, 1, 2, 3) == 2.0f,
                "Retirement destroyed an older immutable snapshot");

        GridStorage limited(first, liquidChannels(0.0f), 512u * sizeof(float));
        limited.write(Channel::Density, 1, 2, 3, 1.0f);
        rejects([&] { limited.write(Channel::Pressure, 1, 2, 3, 1.0f); },
                "Second page ignored the authored value budget");
        require(limited.read(Channel::Pressure, 1, 2, 3) == 0.0f,
                "Budget failure published a partial field");
        rejects([&] { limited.setValueBudget(1); }, "Budget shrink discarded a live page");
        limited.clear(Channel::Density);
        limited.setValueBudget(576u * sizeof(float));
        limited.write(Channel::VelocityX, 7, 2, 3, 5.0f);
        std::array<float, 576> page{};
        limited.exportPage(Channel::VelocityX, {0, 0, 0}, page.data(), page.size());
        // x=8 is owned by the positive tile and must remain page padding.
        const std::size_t padding = 8u + 9u * (2u + 8u * 3u);
        page[padding] = 1.0f;
        rejects([&] {
            limited.importPage(Channel::VelocityX, {0, 0, 0}, page.data(), page.size());
        }, "Non-owner padding became canonical velocity");
        require(limited.read(Channel::VelocityX, 7, 2, 3) == 5.0f,
                "Rejected page import changed canonical velocity");

        auto gas_pressure = gasPressureTopology(cells);
        require(gas_pressure->tiles().size() == 3u * 3u * 2u,
                "Gas pressure omitted invisible air tiles");
        GridStorage air(gas_pressure, gasChannels(293.15f, 0.0f));
        require(air.statistics().current_value_bytes == 0,
                "Full pressure topology eagerly allocated all ambient scalar channels");
        air.write(Channel::Pressure, 23, 16, 8, 1.0f);
        require(air.statistics().resident_pages == 1, "One pressure write allocated every field");

        const int maximum = std::numeric_limits<int>::max();
        const std::array<int, 3> large_dimensions = {maximum, 8, 8};
        auto last = support(large_dimensions,
                            {{{maximum - 7, 0, 0}, {maximum, 8, 8}}});
        GridStorage last_face(last, liquidChannels(0.0f));
        last_face.write(Channel::VelocityX, maximum, 2, 3, 7.0f);
        page.fill(0.0f);
        const Coordinate last_tile = {(maximum - 1) / 8, 0, 0};
        last_face.exportPage(Channel::VelocityX, last_tile, page.data(), page.size());
        last_face.importPage(Channel::VelocityX, last_tile, page.data(), page.size());
        require(last_face.read(Channel::VelocityX, maximum, 2, 3) == 7.0f,
                "Clipped MAC page import overflowed at signed coordinate boundary");
        std::cout << "PASS sparse shared topology, atomic channel remap, snapshots and budget\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
