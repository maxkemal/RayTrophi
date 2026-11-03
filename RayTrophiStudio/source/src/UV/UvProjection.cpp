// SPDX-License-Identifier: MIT
#include "UV/UvLayout.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace UV {

bool projectFromView(Layout& layout, const std::vector<Vec3>& world_positions,
                     Vec3 eye, Vec3 forward, Vec3 up, std::string& error,
                     bool orthographic) {
    if (!validate(layout, error)) {
        return false;
    }
    if (world_positions.size() != layout.coordinates.size()) {
        error = "uv_projection_missing_positions";
        return false;
    }
    auto finite = [](Vec3 v) {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    };
    if (!finite(eye) || !finite(forward) || !finite(up) ||
        forward.lengthSquared() < 1.0e-12f) {
        error = "uv_projection_invalid_camera";
        return false;
    }
    forward = forward.normalize();
    Vec3 right = Vec3::cross(forward, up);
    if (right.lengthSquared() < 1.0e-12f) {
        error = "uv_projection_invalid_camera";
        return false;
    }
    right = right.normalize();
    up = Vec3::cross(right, forward).normalize();
    std::vector<uint8_t> used(world_positions.size(), 0);
    for (uint32_t index : layout.indices) {
        used[index] = 1;
    }
    std::vector<Vec2> projected = layout.coordinates;
    double min_u = (std::numeric_limits<double>::max)();
    double min_v = min_u;
    double max_u = -min_u;
    double max_v = -min_u;
    for (size_t i = 0; i < used.size(); ++i) {
        if (!used[i]) {
            continue;
        }
        const Vec3 delta = world_positions[i] - eye;
        const double depth = Vec3::dot(delta, forward);
        if (!finite(delta) || !std::isfinite(depth) || depth <= 0.0001) {
            error = "uv_projection_behind_camera: frame the entire mesh in front of the camera";
            return false;
        }
        const double divisor = orthographic ? 1.0 : depth;
        projected[i] = Vec2(float(Vec3::dot(delta, right) / divisor),
                            float(Vec3::dot(delta, up) / divisor));
        if (!std::isfinite(projected[i].u) || !std::isfinite(projected[i].v)) {
            error = "uv_projection_non_finite";
            return false;
        }
        min_u = (std::min)(min_u, double(projected[i].u));
        min_v = (std::min)(min_v, double(projected[i].v));
        max_u = (std::max)(max_u, double(projected[i].u));
        max_v = (std::max)(max_v, double(projected[i].v));
    }
    const double span = (std::max)(max_u - min_u, max_v - min_v);
    if (!std::isfinite(span) || span <= 1.0e-12) {
        error = "uv_projection_degenerate";
        return false;
    }
    for (size_t i = 0; i < used.size(); ++i) {
        if (used[i]) {
            projected[i] = Vec2(float((projected[i].u - min_u) / span),
                                float((projected[i].v - min_v) / span));
        }
    }
    layout.coordinates = std::move(projected);
    return true;
}

} // namespace UV
