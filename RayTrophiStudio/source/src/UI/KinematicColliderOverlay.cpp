#include "UI/KinematicColliderOverlay.h"

#include "KinematicColliderScene.h"
#include "globals.h"
#include "imgui.h"
#include "scene_ui.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_set>
#include <vector>

namespace KinematicColliderUI {
namespace {

struct Projection {
    const Camera& camera;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float width = 0.0f;
    float height = 0.0f;
    float aspect = 1.0f;
    float tangent = 1.0f;
    bool orthographic = false;

    bool point(const Vec3& value, ImVec2& screen) const {
        const Vec3 delta = value - camera.lookfrom;
        const float depth = delta.dot(forward);
        if (!orthographic && depth <= 0.01f) {
            return false;
        }
        const float half_height = orthographic
            ? std::max(camera.ortho_height, 0.0001f) * 0.5f
            : depth * tangent;
        if (!std::isfinite(half_height) || half_height <= 1.0e-8f) {
            return false;
        }
        screen = ImVec2(
            (delta.dot(right) / (half_height * aspect) * 0.5f + 0.5f) * width,
            (0.5f - delta.dot(up) / half_height * 0.5f) * height);
        return std::isfinite(screen.x) && std::isfinite(screen.y);
    }

    float radius(const Vec3& center, float world_radius) const {
        ImVec2 middle;
        ImVec2 edge;
        if (!point(center, middle) ||
            !point(center + right * world_radius, edge)) {
            return 0.0f;
        }
        const float dx = edge.x - middle.x;
        const float dy = edge.y - middle.y;
        return std::min(500.0f, std::sqrt(dx * dx + dy * dy));
    }
};

void drawSphere(ImDrawList& draw,
                const Projection& projection,
                const RayTrophiSim::KinematicProxySample& sample,
                ImU32 color) {
    ImVec2 center;
    if (!projection.point(sample.center, center)) {
        return;
    }
    const float radius = projection.radius(sample.center, sample.radius);
    if (radius > 0.25f) {
        draw.AddCircle(center, radius, color, 28, 1.5f);
    }
}

void drawCapsule(ImDrawList& draw,
                 const Projection& projection,
                 const RayTrophiSim::KinematicProxySample& sample,
                 ImU32 color) {
    ImVec2 start;
    ImVec2 end;
    if (!projection.point(sample.capsule_start, start) ||
        !projection.point(sample.capsule_end, end)) {
        return;
    }
    const float start_radius =
        projection.radius(sample.capsule_start, sample.radius);
    const float end_radius =
        projection.radius(sample.capsule_end, sample.radius);
    const float dx = end.x - start.x;
    const float dy = end.y - start.y;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length <= 1.0e-4f || start_radius <= 0.25f ||
        end_radius <= 0.25f) {
        return;
    }
    const ImVec2 normal(-dy / length, dx / length);
    const ImVec2 a0(
        start.x + normal.x * start_radius,
        start.y + normal.y * start_radius);
    const ImVec2 a1(
        start.x - normal.x * start_radius,
        start.y - normal.y * start_radius);
    const ImVec2 b0(
        end.x + normal.x * end_radius,
        end.y + normal.y * end_radius);
    const ImVec2 b1(
        end.x - normal.x * end_radius,
        end.y - normal.y * end_radius);
    draw.AddLine(a0, b0, color, 1.5f);
    draw.AddLine(a1, b1, color, 1.5f);
    draw.AddCircle(start, start_radius, color, 24, 1.5f);
    draw.AddCircle(end, end_radius, color, 24, 1.5f);
}

void drawBox(ImDrawList& draw,
             const Projection& projection,
             const RayTrophiSim::KinematicProxySample& sample,
             ImU32 color) {
    std::array<Vec3, 8> local = {
        Vec3(-1.0f, -1.0f, -1.0f), Vec3(1.0f, -1.0f, -1.0f),
        Vec3(1.0f, 1.0f, -1.0f), Vec3(-1.0f, 1.0f, -1.0f),
        Vec3(-1.0f, -1.0f, 1.0f), Vec3(1.0f, -1.0f, 1.0f),
        Vec3(1.0f, 1.0f, 1.0f), Vec3(-1.0f, 1.0f, 1.0f)};
    std::array<ImVec2, 8> screen;
    std::array<bool, 8> visible{};
    for (std::size_t index = 0; index < local.size(); ++index) {
        local[index].x *= sample.half_extents.x;
        local[index].y *= sample.half_extents.y;
        local[index].z *= sample.half_extents.z;
        visible[index] = projection.point(
            sample.world_transform.transform_point(local[index]),
            screen[index]);
    }
    constexpr int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    for (const auto& edge : edges) {
        if (visible[edge[0]] && visible[edge[1]]) {
            draw.AddLine(screen[edge[0]], screen[edge[1]], color, 1.5f);
        }
    }
}

} // namespace

void drawViewportOverlay(UIContext& context) {
    if (!context.scene.camera || context.scene.kinematic_colliders.sets().empty()) {
        return;
    }
    std::unordered_set<uint64_t> visible_sets;
    for (const auto& set : context.scene.kinematic_colliders.sets()) {
        if (set.enabled && set.viewport_visible) {
            visible_sets.insert(set.id);
        }
    }
    if (visible_sets.empty()) {
        return;
    }

    std::vector<RayTrophiSim::KinematicProxySample> samples;
    std::string error;
    if (!RayTrophiSim::inspectAllKinematicProxySets(
            context.scene, 1.0f / 60.0f, samples, error)) {
        return;
    }
    const ImGuiIO& io = ImGui::GetIO();
    if (io.DisplaySize.x <= 0.0f || io.DisplaySize.y <= 0.0f) {
        return;
    }
    const Camera& camera = *context.scene.camera;
    const Vec3 forward = (camera.lookat - camera.lookfrom).normalize();
    const Vec3 right = forward.cross(camera.vup).normalize();
    const Vec3 up = right.cross(forward).normalize();
    const float aspect = image_height > 0
        ? static_cast<float>(image_width) / static_cast<float>(image_height)
        : io.DisplaySize.x / io.DisplaySize.y;
    if (!std::isfinite(aspect) || aspect <= 0.0f) {
        return;
    }
    const Projection projection{
        camera,
        forward,
        right,
        up,
        io.DisplaySize.x,
        io.DisplaySize.y,
        aspect,
        std::tan(camera.vfov * 3.14159265359f / 360.0f),
        camera.orthographic};

    ImDrawList& draw = *ImGui::GetBackgroundDrawList();
    const ImU32 color = IM_COL32(55, 220, 245, 205);
    for (const auto& sample : samples) {
        if (!sample.resolved || visible_sets.count(sample.set_id) == 0) {
            continue;
        }
        switch (sample.shape) {
            case RayTrophiSim::KinematicProxyShape::Sphere:
                drawSphere(draw, projection, sample, color);
                break;
            case RayTrophiSim::KinematicProxyShape::Box:
                drawBox(draw, projection, sample, color);
                break;
            case RayTrophiSim::KinematicProxyShape::Capsule:
            default:
                drawCapsule(draw, projection, sample, color);
                break;
        }
    }
}

} // namespace KinematicColliderUI
