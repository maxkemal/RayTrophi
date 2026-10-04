// Standalone source test; compilation is performed by the user.
#include "Fluid/FluidActiveWindow.h"
#include "Fluid/FluidActivePressure.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <random>
#include <set>

namespace Window = RayTrophiSim::Fluid::ActiveWindow;

static void checkSupport(const std::vector<Vec3>& positions, const Vec3& origin,
                         float voxel, int nx, int ny, int nz) {
    std::vector<Vec3> velocities(positions.size(), Vec3(0.0f, 0.0f, 0.0f));
    const auto bounds = Window::plan(positions, velocities, origin, voxel,
                                     1.0f / 24.0f, nx, ny, nz, true);
    assert(bounds.cells() > 0 && bounds.cells() <= uint64_t(nx) * ny * nz);
    for (int component = 0; component < 3; ++component) {
        const int dims[3] = {nx + (component == 0), ny + (component == 1),
                             nz + (component == 2)};
        const int extent[3] = {
            bounds.end[0] - bounds.begin[0] + (component == 0),
            bounds.end[1] - bounds.begin[1] + (component == 1),
            bounds.end[2] - bounds.begin[2] + (component == 2)
        };
        std::set<uint64_t> visited;
        const uint64_t count = uint64_t(extent[0]) * extent[1] * extent[2];
        for (uint64_t local = 0; local < count; ++local) {
            // Shader mapping against full MAC storage, including the last face.
            const int i = bounds.begin[0] + int(local % extent[0]);
            const int j = bounds.begin[1] + int((local / extent[0]) % extent[1]);
            const int k = bounds.begin[2] + int(local / (uint64_t(extent[0]) * extent[1]));
            assert(i >= 0 && i < dims[0] && j >= 0 && j < dims[1]);
            assert(k >= 0 && k < dims[2]);
            const uint64_t index = i + uint64_t(dims[0]) * (j + uint64_t(dims[1]) * k);
            assert(visited.insert(index).second);
        }
        for (const auto& position : positions) {
            const Vec3 gp = (position - origin) / voxel;
            const float coordinates[3] = {gp.x, gp.y, gp.z};
            int base[3];
            for (int axis = 0; axis < 3; ++axis) {
                const float shifted = coordinates[axis] - (axis != component ? 0.5f : 0.0f);
                base[axis] = int(std::floor(shifted - 0.5f));
            }
            for (int z = 0; z < 3; ++z) {
                for (int y = 0; y < 3; ++y) {
                    for (int x = 0; x < 3; ++x) {
                        const int i = base[0] + x, j = base[1] + y, k = base[2] + z;
                        if (i < 0 || j < 0 || k < 0 || i >= dims[0] ||
                            j >= dims[1] || k >= dims[2]) {
                            continue;
                        }
                        const uint64_t index = i + uint64_t(dims[0]) *
                            (j + uint64_t(dims[1]) * k);
                        assert(visited.count(index) == 1);
                    }
                }
            }
        }
    }
}

int main() {
    const Vec3 origin(-4.0f, 2.0f, -1.0f);
    constexpr float voxel = 0.25f;
    constexpr int nx = 19, ny = 13, nz = 11;
    for (float x : {-0.2f, 0.0f, 0.49f, 0.5f, 1.0f, 18.9f, 19.0f, 19.2f}) {
        for (float y : {-0.2f, 0.0f, 6.5f, 13.2f}) {
            for (float z : {-0.2f, 0.0f, 5.5f, 11.2f}) {
                checkSupport({origin + Vec3(x, y, z) * voxel}, origin, voxel, nx, ny, nz);
            }
        }
    }
    std::mt19937 random(312);
    std::uniform_real_distribution<float> coordinate(-1.0f, 20.0f);
    for (int sample = 0; sample < 100; ++sample) {
        checkSupport({origin + Vec3(coordinate(random), coordinate(random),
                                   coordinate(random)) * voxel}, origin, voxel, nx, ny, nz);
    }
    std::vector<Vec3> positions{origin + Vec3(8.0f, 6.0f, 5.0f) * voxel};
    std::vector<Vec3> velocities{Vec3(0.0f, 0.0f, 0.0f)};
    const auto local = Window::plan(positions, velocities, origin, voxel, 0.1f, nx, ny, nz, true);
    assert(local.bounded);
    velocities[0] = Vec3(100.0f, 0.0f, 0.0f);
    const auto moving = Window::plan(positions, velocities, origin, voxel, 0.1f, nx, ny, nz, true);
    assert(moving.cells() >= local.cells());
    assert(!Window::plan(positions, velocities, origin, voxel, 0.1f, nx, ny, nz, false).bounded);
    positions[0].x = std::numeric_limits<float>::quiet_NaN();
    assert(!Window::plan(positions, velocities, origin, voxel, 0.1f, nx, ny, nz, true).bounded);
    assert(!Window::plan({}, {}, origin, voxel, 0.1f, nx, ny, nz, true).bounded);
    std::vector<float> mask(size_t(nx) * ny * nz, 0.0f);
    const size_t centre = 8 + nx * (6 + ny * 5);
    mask[centre] = 1.0f;
    const auto pressure = Window::pressureMaskBounds(mask, nx, ny, nz);
    assert(pressure.bounded && pressure.cells() == 27);
    assert(pressure.begin[0] == 7 && pressure.end[0] == 10);
    const Window::PressureDispatch dispatcher(pressure);
    assert(dispatcher.groups(200) == 1);
    assert(Window::pressureVariant("sim_fluid_cg_spmv_dot_var") != nullptr);
    assert(Window::pressureVariant("sim_fluid_cg_scalar_step") == nullptr);
    assert(Window::pressureVariant("sim_fluid_cg_residual_init") == nullptr);
    assert(Window::pressureVariant("sim_fluid_subtract_gradient") == nullptr);
    mask[centre] = 0.0f;
    assert(!Window::pressureMaskBounds(mask, nx, ny, nz).bounded);
    mask[centre] = std::numeric_limits<float>::infinity();
    assert(!Window::pressureMaskBounds(mask, nx, ny, nz).bounded);
    std::cout << "Fluid active-window support and MAC indexing PASS\n";
}
