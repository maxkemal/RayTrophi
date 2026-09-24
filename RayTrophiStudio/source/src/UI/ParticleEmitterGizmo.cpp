#include "UI/ParticleEmitterGizmo.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace ParticleEmitterGizmo {
namespace {

constexpr float kTau = 6.28318530718f;

bool projectSegment(ImDrawList* draw_list,
                    const ProjectWorldPoint& project,
                    const Vec3& a,
                    const Vec3& b,
                    ImU32 color,
                    float thickness) {
    ImVec2 screen_a;
    ImVec2 screen_b;
    if (!project(a, screen_a) || !project(b, screen_b)) {
        return false;
    }
    draw_list->AddLine(screen_a, screen_b, color, thickness);
    return true;
}

Vec3 safeNormal(const Vec3& value, const Vec3& fallback) {
    const float length = value.length();
    return length > 1.0e-5f ? value * (1.0f / length) : fallback;
}

bool acceptConeCandidate(float distance,
                         const Vec3& ray_origin,
                         const Vec3& ray_direction,
                         const DirectionalEmitterGeometry& geometry,
                         float& closest) {
    if (distance <= 0.001f || distance >= closest) {
        return false;
    }
    const Vec3 point = ray_origin + ray_direction * distance;
    const float axial = (point - geometry.collar_center).dot(geometry.forward);
    if (axial < 0.0f || axial > geometry.length) {
        return false;
    }
    closest = distance;
    return true;
}

} // namespace

DirectionalEmitterGeometry makeDirectionalEmitterGeometry(
    const Vec3& position,
    const Vec3& direction,
    float speed,
    float spread) {
    DirectionalEmitterGeometry geometry;
    geometry.forward = safeNormal(direction, Vec3(0.0f, 1.0f, 0.0f));
    geometry.right = Vec3::cross(
        geometry.forward, Vec3(0.0f, 1.0f, 0.0f));
    if (geometry.right.length() <= 1.0e-5f) {
        geometry.right = Vec3::cross(
            geometry.forward, Vec3(1.0f, 0.0f, 0.0f));
    }
    geometry.right = safeNormal(geometry.right, Vec3(1.0f, 0.0f, 0.0f));
    geometry.up = safeNormal(
        Vec3::cross(geometry.right, geometry.forward),
        Vec3(0.0f, 0.0f, 1.0f));
    const float display_length = std::clamp(
        0.45f + std::max(0.0f, speed) * 0.18f, 0.45f, 3.0f);
    geometry.length = std::max(0.01f, display_length - 0.08f);
    const float half_angle = std::clamp(std::abs(spread), 0.08f, 1.15f);
    geometry.mouth_radius = std::clamp(
        std::tan(half_angle) * display_length,
        0.12f,
        display_length * 1.6f);
    geometry.collar_radius = std::clamp(
        geometry.mouth_radius * 0.22f, 0.07f, 0.24f);
    geometry.collar_center = position + geometry.forward * 0.08f;
    geometry.mouth_center = position + geometry.forward * display_length;
    return geometry;
}

bool intersectDirectionalEmitter(const Vec3& ray_origin,
                                 const Vec3& ray_direction,
                                 const DirectionalEmitterGeometry& geometry,
                                 float padding,
                                 float& out_distance) {
    const float pad = std::max(0.0f, padding);
    const float collar_radius = geometry.collar_radius + pad;
    const float mouth_radius = geometry.mouth_radius + pad;
    const float radius_slope =
        (mouth_radius - collar_radius) / std::max(geometry.length, 1.0e-5f);
    const Vec3 offset = ray_origin - geometry.collar_center;
    const float ray_axial = ray_direction.dot(geometry.forward);
    const float offset_axial = offset.dot(geometry.forward);
    const Vec3 ray_radial = ray_direction - geometry.forward * ray_axial;
    const Vec3 offset_radial = offset - geometry.forward * offset_axial;
    const float radius_at_origin = collar_radius + radius_slope * offset_axial;
    const float radius_along_ray = radius_slope * ray_axial;

    const float a = ray_radial.dot(ray_radial) -
                    radius_along_ray * radius_along_ray;
    const float b = 2.0f * (offset_radial.dot(ray_radial) -
                            radius_at_origin * radius_along_ray);
    const float c = offset_radial.dot(offset_radial) -
                    radius_at_origin * radius_at_origin;
    float closest = std::numeric_limits<float>::infinity();
    if (std::abs(a) > 1.0e-7f) {
        const float discriminant = b * b - 4.0f * a * c;
        if (discriminant >= 0.0f) {
            const float root = std::sqrt(discriminant);
            acceptConeCandidate((-b - root) / (2.0f * a),
                                ray_origin, ray_direction, geometry, closest);
            acceptConeCandidate((-b + root) / (2.0f * a),
                                ray_origin, ray_direction, geometry, closest);
        }
    } else if (std::abs(b) > 1.0e-7f) {
        acceptConeCandidate(-c / b, ray_origin, ray_direction, geometry, closest);
    }

    auto testCap = [&](const Vec3& center, float radius) {
        const float denominator = ray_direction.dot(geometry.forward);
        if (std::abs(denominator) <= 1.0e-7f) {
            return;
        }
        const float distance =
            (center - ray_origin).dot(geometry.forward) / denominator;
        if (distance <= 0.001f || distance >= closest) {
            return;
        }
        const Vec3 radial = ray_origin + ray_direction * distance - center;
        if (radial.dot(radial) <= radius * radius) {
            closest = distance;
        }
    };
    testCap(geometry.collar_center, collar_radius);
    testCap(geometry.mouth_center, mouth_radius);

    if (!std::isfinite(closest)) {
        return false;
    }
    out_distance = closest;
    return true;
}

ScopedViewportClip::ScopedViewportClip(ImDrawList* background,
                                       ImDrawList* foreground,
                                       const ImVec2& minimum,
                                       const ImVec2& maximum)
    : background_(background), foreground_(foreground) {
    if (background_) {
        background_->PushClipRect(minimum, maximum, true);
    }
    if (foreground_ && foreground_ != background_) {
        foreground_->PushClipRect(minimum, maximum, true);
    }
}

ScopedViewportClip::~ScopedViewportClip() {
    if (foreground_ && foreground_ != background_) {
        foreground_->PopClipRect();
    }
    if (background_) {
        background_->PopClipRect();
    }
}

void drawDirectionalEmitter(ImDrawList* draw_list,
                            const ProjectWorldPoint& project,
                            const Vec3& position,
                            const Vec3& direction,
                            float speed,
                            float spread,
                            ImU32 color,
                            bool selected) {
    if (!draw_list) {
        return;
    }

    const DirectionalEmitterGeometry geometry =
        makeDirectionalEmitterGeometry(position, direction, speed, spread);
    const float thickness = selected ? 2.4f : 1.45f;
    constexpr int segments = 20;

    std::array<Vec3, segments> collar_points;
    std::array<Vec3, segments> mouth_points;
    for (int i = 0; i < segments; ++i) {
        const float angle = kTau * static_cast<float>(i) /
                            static_cast<float>(segments);
        const Vec3 radial = geometry.right * std::cos(angle) +
                            geometry.up * std::sin(angle);
        collar_points[static_cast<std::size_t>(i)] =
            geometry.collar_center + radial * geometry.collar_radius;
        mouth_points[static_cast<std::size_t>(i)] =
            geometry.mouth_center + radial * geometry.mouth_radius;
    }

    for (int i = 0; i < segments; ++i) {
        const int next = (i + 1) % segments;
        projectSegment(draw_list, project,
                       collar_points[static_cast<std::size_t>(i)],
                       collar_points[static_cast<std::size_t>(next)],
                       color, thickness);
        projectSegment(draw_list, project,
                       mouth_points[static_cast<std::size_t>(i)],
                       mouth_points[static_cast<std::size_t>(next)],
                       color, thickness);
    }

    for (int i = 0; i < segments; i += 5) {
        projectSegment(draw_list, project,
                       collar_points[static_cast<std::size_t>(i)],
                       mouth_points[static_cast<std::size_t>(i)],
                       color, thickness);
    }
    projectSegment(draw_list, project, position, geometry.mouth_center,
                   color, thickness);

    ImVec2 tip;
    ImVec2 shaft;
    const Vec3 arrow_base = position +
        geometry.forward * (geometry.length * 0.68f);
    if (project(geometry.mouth_center, tip) && project(arrow_base, shaft)) {
        ImVec2 delta(tip.x - shaft.x, tip.y - shaft.y);
        const float screen_length = std::sqrt(delta.x * delta.x + delta.y * delta.y);
        if (screen_length > 1.0e-3f) {
            delta.x /= screen_length;
            delta.y /= screen_length;
            const ImVec2 normal(-delta.y, delta.x);
            const float head = selected ? 12.0f : 9.0f;
            draw_list->AddTriangleFilled(
                tip,
                ImVec2(tip.x - delta.x * head + normal.x * head * 0.48f,
                       tip.y - delta.y * head + normal.y * head * 0.48f),
                ImVec2(tip.x - delta.x * head - normal.x * head * 0.48f,
                       tip.y - delta.y * head - normal.y * head * 0.48f),
                color);
        }
    }
}

} // namespace ParticleEmitterGizmo
