#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>

namespace RayTrophiSim::Fluid {

// A signed interior is meaningful only for a closed, nonzero-volume triangle
// soup. Weld position seams conservatively; UV/normal splits are not holes.
// Used only for one-off signed cooks, never the per-frame unsigned shell query.
template <class Triangles>
bool colliderMeshHasClosedVolume(const Triangles& triangles) {
    std::array<double, 3> low;
    std::array<double, 3> high;
    low.fill(std::numeric_limits<double>::infinity());
    high.fill(-std::numeric_limits<double>::infinity());
    for (const auto& triangle : triangles) {
        for (const auto& point : {triangle.p0, triangle.p1, triangle.p2}) {
            const std::array<double, 3> p{point.x, point.y, point.z};
            for (std::size_t axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(p[axis])) {
                    return false;
                }
                low[axis] = std::min(low[axis], p[axis]);
                high[axis] = std::max(high[axis], p[axis]);
            }
        }
    }
    const double span = std::max({high[0] - low[0], high[1] - low[1], high[2] - low[2]});
    if (!std::isfinite(span) || span <= 0.0) {
        return false;
    }
    const double epsilon = std::max(1e-7, span * 1e-6);
    using Point = std::array<int64_t, 3>;
    using Edge = std::array<Point, 2>;
    std::map<Edge, uint32_t> uses;
    double volume6 = 0.0;
    for (const auto& triangle : triangles) {
        std::array<Point, 3> keys;
        std::array<std::array<double, 3>, 3> positions;
        std::size_t corner = 0;
        for (const auto& point : {triangle.p0, triangle.p1, triangle.p2}) {
            positions[corner] = {point.x - low[0], point.y - low[1], point.z - low[2]};
            for (std::size_t axis = 0; axis < 3; ++axis) {
                keys[corner][axis] = std::llround(positions[corner][axis] / epsilon);
            }
            ++corner;
        }
        for (std::size_t edge = 0; edge < 3; ++edge) {
            Point a = keys[edge];
            Point b = keys[(edge + 1) % 3];
            if (a == b) {
                return false;
            }
            if (b < a) {
                std::swap(a, b);
            }
            if (++uses[Edge{a, b}] > 2) {
                return false;
            }
        }
        const auto& a = positions[0];
        const auto& b = positions[1];
        const auto& c = positions[2];
        volume6 += a[0] * (b[1] * c[2] - b[2] * c[1]) +
                   a[1] * (b[2] * c[0] - b[0] * c[2]) +
                   a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    return !uses.empty() && std::abs(volume6) > epsilon * epsilon * epsilon &&
           std::all_of(uses.begin(), uses.end(), [](const auto& edge) {
               return edge.second == 2;
           });
}

} // namespace RayTrophiSim::Fluid
