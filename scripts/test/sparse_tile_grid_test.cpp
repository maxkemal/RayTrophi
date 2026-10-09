#include "Fluid/SparseTileGrid.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace RayTrophiSim::Fluid::Sparse;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Operation>
void requireRejected(Operation operation, const char* message) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, message);
}

int main() {
    try {
        // Two disconnected pools must not allocate the enormous intervening box.
        const std::array<int, 3> huge = {1000000, 1000000, 1000000};
        auto islands = std::make_shared<Topology>(huge, std::vector<CellBox>{
            {{0, 0, 0}, {8, 8, 8}},
            {{999992, 999992, 999992}, {1000000, 1000000, 1000000}}
        });
        Field density(islands, Location::Cell, 0.0f);
        require(islands->tiles().size() == 2, "Disconnected support became dense");
        require(density.allocatedValues() == 1024, "Storage depends on domain volume");
        density.write(999999, 999999, 999999, 3.0f);
        require(density.read(999999, 999999, 999999) == 3.0f, "Far tile address failed");
        require(density.read(500000, 500000, 500000) == 0.0f, "Missing density background");
        requireRejected([&] { density.write(500000, 500000, 500000, 1.0f); },
                        "Unallocated write silently dropped mass");

        // Slot insertion/removal must not scramble canonical coordinates.
        const std::array<int, 3> dimensions = {17, 19, 21};
        auto far = std::make_shared<Topology>(dimensions, std::vector<CellBox>{
            {{8, 8, 8}, {16, 16, 16}}
        });
        Field heat(far, Location::Cell, 293.0f);
        heat.write(9, 10, 11, 350.0f);
        auto expanded = std::make_shared<Topology>(dimensions, std::vector<CellBox>{
            {{0, 0, 0}, {8, 8, 8}}, {{8, 8, 8}, {16, 16, 16}}
        });
        heat.rebind(expanded);
        require(heat.read(9, 10, 11) == 350.0f, "Remap lost heat after slot change");
        require(heat.read(0, 0, 0) == 293.0f, "New tile did not use ambient temperature");
        auto empty = std::make_shared<Topology>(dimensions, std::vector<CellBox>{});
        requireRejected([&] { heat.rebind(empty); }, "Live tile retirement lost heat");
        require(heat.read(9, 10, 11) == 350.0f, "Rejected remap was not atomic");
        auto changed_layout = std::make_shared<Topology>(std::array<int, 3>{18, 19, 21},
            std::vector<CellBox>{});
        requireRejected([&] { heat.rebind(changed_layout); },
                        "Layout change reused incompatible physical coordinates");
        require(heat.read(9, 10, 11) == 350.0f, "Layout rejection lost old state");
        heat.write(9, 10, 11, 293.0f);
        heat.rebind(empty);
        require(heat.allocatedValues() == 0, "Quiescent pages were not released");

        // Dense reference values over clipped tiles and every staggered face.
        auto full = std::make_shared<Topology>(dimensions, std::vector<CellBox>{
            {{0, 0, 0}, dimensions}
        });
        for (Location location : {Location::Cell, Location::XFace,
                                  Location::YFace, Location::ZFace}) {
            Field field(full, location, 0.0f);
            auto extent = dimensions;
            const int axis = faceAxis(location);
            if (axis >= 0) {
                ++extent[axis];
            }
            for (int z = 0; z < extent[2]; ++z) {
                for (int y = 0; y < extent[1]; ++y) {
                    for (int x = 0; x < extent[0]; ++x) {
                        field.write(x, y, z, float(1 + x + 100 * y + 10000 * z));
                    }
                }
            }
            for (int z = 0; z < extent[2]; ++z) {
                for (int y = 0; y < extent[1]; ++y) {
                    for (int x = 0; x < extent[0]; ++x) {
                        require(field.read(x, y, z) == float(1 + x + 100 * y + 10000 * z),
                                "MAC face alias or clipped tile address mismatch");
                    }
                }
            }
        }
        // Exact tile boundary: the final face is stored in the negative tile.
        auto edge = std::make_shared<Topology>(std::array<int, 3>{8, 8, 8},
            std::vector<CellBox>{{{0, 0, 0}, {8, 8, 8}}});
        Field u(edge, Location::XFace, 0.0f);
        u.write(8, 7, 7, 42.0f);
        require(u.read(8, 7, 7) == 42.0f, "Outer face at exact tile edge is missing");

        const auto swept = sweptSupport({15.5, 15.5, 15.5}, {16.0, 0.0, 0.0},
            {0.0, 0.0, 0.0}, {64, 64, 64}, 1.0, 1.0);
        require(swept.begin[0] == 0 && swept.end[0] == 35,
                "CFL travel halo omitted swept support");
        require(swept.begin[1] == 13 && swept.end[1] == 19,
                "Quadratic transfer and pressure neighbour halo missing");
        requireRejected([&] {
            sweptSupport({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
                         {64, 64, 64}, 1.0, -1.0);
        }, "Negative timestep was accepted");
        std::cout << "PASS sparse tile storage, MAC ownership, remap and swept support\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
