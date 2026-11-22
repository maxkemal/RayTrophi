#pragma once

#include "AABB.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

struct SceneData;

namespace Paint {

struct BalloonRoomFront {
    Vec3 inward = Vec3(0.0f);
    std::vector<Vec3> triangles;

    float distance(const Vec3& point) const {
        return Vec3::dot(point - triangles.front(), inward);
    }
};

// Find a single open end of a three-sided, axis-aligned paint room.
inline BalloonRoomFront balloonRoomFront(const std::vector<AABB>& surfaces) {
    std::array<std::vector<const AABB*>, 3> walls;
    for (const auto& surface : surfaces) {
        const auto size = surface.max - surface.min;
        for (const int axis : {0, 2}) {
            if (size[axis] < 0.3f && size[2 - axis] > 1.0f && size.y > 1.0f) {
                walls[axis].push_back(&surface);
            }
        }
    }

    for (const int axis : {0, 2}) {
        const int cross = 2 - axis;
        const auto& sides = walls[cross];
        if (walls[axis].size() != 1 || sides.size() != 2) {
            continue;
        }
        const float low = std::max(sides[0]->min[axis], sides[1]->min[axis]);
        const float high = std::min(sides[0]->max[axis], sides[1]->max[axis]);
        const auto& rear = *walls[axis].front();
        const float rear_center = (rear.min[axis] + rear.max[axis]) * 0.5f;
        if (high - low < 2.0f || std::min(std::abs(rear_center - low),
                                          std::abs(rear_center - high)) > 0.75f) {
            continue;
        }
        const bool open_high = rear_center < (low + high) * 0.5f;
        const float plane = open_high ? high + 0.25f : low - 0.25f;
        const float side_low = std::min(sides[0]->min[cross], sides[1]->min[cross]) - 0.3f;
        const float side_high = std::max(sides[0]->max[cross], sides[1]->max[cross]) + 0.3f;
        const float floor = std::min(sides[0]->min.y, sides[1]->min.y) - 0.5f;
        const float ceiling = std::max(sides[0]->max.y, sides[1]->max.y) + 0.5f;
        if (side_high - side_low < 2.0f) {
            continue;
        }
        const auto point = [&](float side, float y) {
            auto p = Vec3(0.0f, y, 0.0f);
            p[axis] = plane;
            p[cross] = side;
            return p;
        };
        const auto a = point(side_low, floor);
        const auto b = point(side_high, floor);
        const auto c = point(side_high, ceiling);
        const auto d = point(side_low, ceiling);
        BalloonRoomFront front;
        front.inward[axis] = open_high ? -1.0f : 1.0f;
        if (Vec3::dot((b - a).cross(c - a), front.inward) > 0.0f) {
            front.triangles = {a, b, c, a, c, d};
        } else {
            front.triangles = {a, c, b, a, d, c};
        }
        return front;
    }
    return {};
}

BalloonRoomFront balloonPaintRoomFront(SceneData& scene,
                                       const std::vector<std::string>& surfaces);

} // namespace Paint
