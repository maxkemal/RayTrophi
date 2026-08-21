#include "MeshEdit/PolygonTriangulation.h"

#include <algorithm>
#include <cstddef>
#include <cmath>

namespace MeshEdit {
namespace {

struct Vec2d { double x = 0.0; double y = 0.0; };

double cross2(const Vec2d& a, const Vec2d& b, const Vec2d& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool pointInTriangle(const Vec2d& p, const Vec2d& a, const Vec2d& b, const Vec2d& c) {
    constexpr double eps = 1e-12;
    const double ab = cross2(a, b, p);
    const double bc = cross2(b, c, p);
    const double ca = cross2(c, a, p);
    return ab >= -eps && bc >= -eps && ca >= -eps;
}

} // namespace

std::vector<std::array<int, 3>> triangulatePlanarPolygon(
    const std::vector<Vec3>& points, const Vec3& reference_normal) {
    std::vector<std::array<int, 3>> result;
    if (points.size() < 3 || reference_normal.length_squared() <= 1e-20f) return result;

    const Vec3 n = reference_normal;
    const float ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
    std::vector<Vec2d> projected(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        if (ax >= ay && ax >= az) projected[i] = { points[i].y, points[i].z };
        else if (ay >= az) projected[i] = { points[i].x, points[i].z };
        else projected[i] = { points[i].x, points[i].y };
    }

    std::vector<int> loop;
    loop.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        if (!loop.empty()) {
            const Vec3 d = points[i] - points[static_cast<size_t>(loop.back())];
            if (d.length_squared() <= 1e-20f) continue;
        }
        loop.push_back(static_cast<int>(i));
    }
    if (loop.size() >= 2) {
        const Vec3 d = points[static_cast<size_t>(loop.front())] -
                       points[static_cast<size_t>(loop.back())];
        if (d.length_squared() <= 1e-20f) loop.pop_back();
    }
    if (loop.size() < 3) return result;

    double area = 0.0;
    for (size_t i = 0; i < loop.size(); ++i) {
        const Vec2d& a = projected[static_cast<size_t>(loop[i])];
        const Vec2d& b = projected[static_cast<size_t>(loop[(i + 1) % loop.size()])];
        area += a.x * b.y - b.x * a.y;
    }
    if (std::fabs(area) <= 1e-14) return result;
    if (area < 0.0) std::reverse(loop.begin(), loop.end());

    std::vector<int> remaining = loop;
    result.reserve(loop.size() - 2);
    const size_t guard_limit = loop.size() * loop.size() + 4;
    size_t guard = 0;
    while (remaining.size() > 3 && guard++ < guard_limit) {
        bool clipped = false;
        for (size_t i = 0; i < remaining.size(); ++i) {
            const int ia = remaining[(i + remaining.size() - 1) % remaining.size()];
            const int ib = remaining[i];
            const int ic = remaining[(i + 1) % remaining.size()];
            const Vec2d& a = projected[static_cast<size_t>(ia)];
            const Vec2d& b = projected[static_cast<size_t>(ib)];
            const Vec2d& c = projected[static_cast<size_t>(ic)];
            if (cross2(a, b, c) <= 1e-12) continue;

            bool contains_other = false;
            for (const int candidate : remaining) {
                if (candidate == ia || candidate == ib || candidate == ic) continue;
                if (pointInTriangle(projected[static_cast<size_t>(candidate)], a, b, c)) {
                    contains_other = true;
                    break;
                }
            }
            if (contains_other) continue;
            result.push_back({ ia, ib, ic });
            remaining.erase(remaining.begin() + static_cast<std::ptrdiff_t>(i));
            clipped = true;
            break;
        }
        if (!clipped) return {};
    }
    if (remaining.size() == 3) result.push_back({ remaining[0], remaining[1], remaining[2] });

    // Projection handedness differs by dominant axis. Make the final 3D winding
    // authoritative so render normals never depend on which axis was selected.
    for (auto& tri : result) {
        const Vec3 c = Vec3::cross(points[static_cast<size_t>(tri[1])] - points[static_cast<size_t>(tri[0])],
                                    points[static_cast<size_t>(tri[2])] - points[static_cast<size_t>(tri[0])]);
        if (c.dot(n) < 0.0f) std::swap(tri[1], tri[2]);
    }
    return result;
}

} // namespace MeshEdit
