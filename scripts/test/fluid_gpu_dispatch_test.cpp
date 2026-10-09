#include "Fluid/FluidGpuDispatch.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check(uint32_t count) {
    const auto shape = RayTrophiSim::FluidGpuDispatch::groups256(count);
    const uint64_t needed = count == 0 ? 1u : 1u + (uint64_t(count) - 1u) / 256u;
    const uint64_t dispatched = uint64_t(shape.groups_x) * shape.groups_y;
    require(shape.groups_x > 0 && shape.groups_x <= 65535, "Invalid workgroup X extent");
    require(shape.groups_y > 0 && shape.groups_y <= 65535, "Invalid workgroup Y extent");
    require(shape.groups_z == 1, "Unexpected Z dispatch");
    require(dispatched >= needed, "Dispatch missed logical workgroups");
    require(dispatched - needed < shape.groups_y, "Excess padding at row transition");
    if (count != 0) {
        const uint64_t final_group = needed - 1;
        const uint64_t x = final_group % shape.groups_x;
        const uint64_t y = final_group / shape.groups_x;
        require(x + y * shape.groups_x == final_group, "Final group mapping is not invertible");
        const uint64_t final_lane = final_group * 256u + (uint64_t(count) - 1u) % 256u;
        require(final_lane == uint64_t(count) - 1u, "Final valid lane was lost");
    }
}

int main() {
    try {
        for (uint32_t count : {0u, 1u, 255u, 256u, 257u, 65535u * 256u,
                               65535u * 256u + 1u, 65536u * 256u + 1u,
                               std::numeric_limits<uint32_t>::max() - 255u,
                               std::numeric_limits<uint32_t>::max()}) {
            check(count);
        }
        const auto edge = RayTrophiSim::FluidGpuDispatch::groups256(65535u * 256u + 1u);
        require(edge.groups_x == 32768 && edge.groups_y == 2,
                "First two-row dispatch doubled nearly all workgroups");
        std::cout << "PASS balanced GPU dispatch coverage, padding and 32-bit boundary\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
