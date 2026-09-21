#include "UI/RigEnvelopeOverlay.h"

#include "Animation/RigEnvelopeWeights.h"
#include "globals.h"
#include "imgui.h"
#include "scene_ui.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace RigUI {
void drawRigEnvelopeOverlay(UIContext &ctx, int shadingMode) {
    const auto &state = ctx.scene.rigView;
    if (!state.envelope_overlay_visible || state.envelope_overlay_character.empty() ||
        !ctx.scene.camera) {
        return;
    }

    RigAuthoring::EnvelopeWeightSettings settings;
    settings.torsoRadius = state.envelope_overlay_torso_radius;
    settings.limbRadius = state.envelope_overlay_limb_radius;
    settings.extremityRadius = state.envelope_overlay_extremity_radius;
    settings.falloff = state.envelope_overlay_falloff;
    std::vector<RigAuthoring::EnvelopeSegmentView> segments;
    std::string error;
    if (!RigAuthoring::envelopeSegmentViews(ctx.scene, state.envelope_overlay_character,
                                            settings, segments, error)) {
        return;
    }

    const auto &camera = *ctx.scene.camera;
    const auto &io = ImGui::GetIO();
    if (io.DisplaySize.x <= 0 || io.DisplaySize.y <= 0)
        return;
    const Vec3 forward = (camera.lookat - camera.lookfrom).normalize();
    const Vec3 right = forward.cross(camera.vup).normalize();
    const Vec3 up = right.cross(forward).normalize();
    const bool ortho = camera.orthographic && shadingMode != 2;
    const float aspect = image_height > 0 ? static_cast<float>(image_width) / image_height
                                          : io.DisplaySize.x / io.DisplaySize.y;
    const float tangent = std::tan(camera.vfov * 3.14159265359f / 360.f);
    if (!std::isfinite(aspect) || aspect <= 0)
        return;

    auto project = [&](const Vec3 &point, ImVec2 &screen) {
        const auto delta = point - camera.lookfrom;
        const float depth = delta.dot(forward);
        if (!ortho && depth <= .01f)
            return false;
        const float halfHeight =
            ortho ? (camera.ortho_height > 1e-4f ? camera.ortho_height : 10.f) * .5f
                  : depth * tangent;
        if (!std::isfinite(halfHeight) || halfHeight <= 1e-8f)
            return false;
        screen = ImVec2((delta.dot(right) / (halfHeight * aspect) * .5f + .5f) *
                            io.DisplaySize.x,
                        (.5f - delta.dot(up) / halfHeight * .5f) * io.DisplaySize.y);
        return std::isfinite(screen.x) && std::isfinite(screen.y);
    };
    auto screenRadius = [&](const Vec3 &center, float radius) {
        ImVec2 middle, edge;
        if (!project(center, middle) || !project(center + right * radius, edge))
            return 0.f;
        const float dx = edge.x - middle.x;
        const float dy = edge.y - middle.y;
        return (std::min)(500.f, std::sqrt(dx * dx + dy * dy));
    };

    auto *draw = ImGui::GetBackgroundDrawList();
    for (const auto &segment : segments) {
        ImVec2 a, b;
        if (!project(segment.start, a) || !project(segment.end, b))
            continue;
        const float radiusA = screenRadius(segment.start, segment.startRadius);
        const float radiusB = screenRadius(segment.end, segment.endRadius);
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        const float length = std::sqrt(dx * dx + dy * dy);
        if (length <= 1e-4f || radiusA <= .25f || radiusB <= .25f)
            continue;
        const ImVec2 normal(-dy / length, dx / length);
        const ImVec2 a0(a.x + normal.x * radiusA, a.y + normal.y * radiusA);
        const ImVec2 a1(a.x - normal.x * radiusA, a.y - normal.y * radiusA);
        const ImVec2 b0(b.x + normal.x * radiusB, b.y + normal.y * radiusB);
        const ImVec2 b1(b.x - normal.x * radiusB, b.y - normal.y * radiusB);
        const bool active = state.character == state.envelope_overlay_character &&
                            state.bone == segment.bone;
        const ImU32 outline =
            active ? IM_COL32(255, 176, 55, 235) : IM_COL32(70, 215, 245, 105);
        const ImU32 fill =
            active ? IM_COL32(255, 176, 55, 22) : IM_COL32(70, 215, 245, 8);
        draw->AddQuadFilled(a0, b0, b1, a1, fill);
        draw->AddCircleFilled(a, radiusA, fill, 32);
        draw->AddCircleFilled(b, radiusB, fill, 32);
        draw->AddLine(a0, b0, outline, active ? 2.f : 1.f);
        draw->AddLine(a1, b1, outline, active ? 2.f : 1.f);
        draw->AddCircle(a, radiusA, outline, 32, active ? 2.f : 1.f);
        draw->AddCircle(b, radiusB, outline, 32, active ? 2.f : 1.f);
        if (active) {
            const std::string label = segment.category + " envelope";
            draw->AddText(ImVec2(a.x + radiusA + 6.f, a.y - 7.f), outline, label.c_str());
        }
    }
}
} // namespace RigUI
