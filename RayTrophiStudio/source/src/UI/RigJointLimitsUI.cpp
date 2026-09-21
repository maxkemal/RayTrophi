#include "UI/RigJointLimitsUI.h"
#include "Animation/RigJointLimits.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "globals.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
namespace RigUI {
namespace {
struct Drag {
    bool active = false;
    int handle = -1, load = -1, frame = -1;
    uint64_t serial = 0;
    float radius = 0, lastAngle = 0;
    ImVec2 mouse;
    RigAuthoring::JointLimitView view;
    std::string error;
} drag;
Vec3 origin(const Matrix4x4& m) {
    return Vec3(m.m[0][3], m.m[1][3], m.m[2][3]);
}
float clamp(float x, float a, float b) {
    return (std::max)(a, (std::min)(b, x));
}
}
void cancelRigJointLimitDrag() {
    drag.active = false;
}
void drawRigJointLimits(UIContext& ctx, int shadingMode, bool& hit) {
    if (!drag.error.empty())
        ImGui::GetForegroundDrawList()->AddText(ImVec2(30, 130), IM_COL32(255, 120, 80, 255),
                                                drag.error.c_str());
    const auto& state = ctx.scene.rigView;
    if (!state.joint_limits_visible || state.character.empty() || state.bone.empty() ||
        !ctx.scene.camera) {
        drag.active = false;
        return;
    }
    RigAuthoring::JointLimitView view;
    std::string error;
    if (!RigAuthoring::getJointLimitView(ctx.scene, state.character, state.bone, view, error)) {
        drag.active = false;
        return;
    }
    auto& io = ImGui::GetIO();
    const auto& cam = *ctx.scene.camera;
    const auto f = (cam.lookat - cam.lookfrom).normalize(), r = f.cross(cam.vup).normalize(),
               u = r.cross(f).normalize();
    const float aspect =
        image_height > 0 ? float(image_width) / image_height : io.DisplaySize.x / io.DisplaySize.y;
    if (!std::isfinite(aspect) || aspect <= 0 || io.DisplaySize.y <= 0) {
        drag.active = false;
        return;
    }
    const bool ortho = cam.orthographic && shadingMode != 2;
    const float tangent = std::tan(cam.vfov * 3.14159265359f / 360.f);
    auto project = [&](const Vec3& p, ImVec2& s) {
        const auto d = p - cam.lookfrom;
        const float depth = d.dot(f);
        if (!ortho && depth <= .01f)
            return false;
        const float half =
            ortho ? (cam.ortho_height > 1e-4f ? cam.ortho_height : 10.f) * .5f : depth * tangent;
        if (!std::isfinite(half) || half <= 1e-8f)
            return false;
        s = ImVec2((d.dot(r) / (half * aspect) * .5f + .5f) * io.DisplaySize.x,
                   (.5f - d.dot(u) / half * .5f) * io.DisplaySize.y);
        return std::isfinite(s.x) && std::isfinite(s.y);
    };
    const auto center = origin(view.jointWorld);
    ImVec2 screen;
    if (!project(center, screen)) {
        drag.active = false;
        return;
    }
    const float half = ortho ? (cam.ortho_height > 1e-4f ? cam.ortho_height : 10.f) * .5f
                             : (center - cam.lookfrom).dot(f) * tangent;
    float radius = half * 2 / io.DisplaySize.y * 65.f;
    const bool editable = state.joint_limits_edit && view.owned && view.hasRule &&
                          (view.rule.type == "hinge" || view.rule.type == "ball") &&
                          !state.pose.hasPreview;
    if (drag.active && (!editable || drag.load != ctx.scene.load_counter ||
                        drag.frame != ctx.scene.timeline.current_frame ||
                        drag.serial != state.pose.serial || drag.view.revision != view.revision ||
                        drag.view.character != view.character || drag.view.bone != view.bone))
        drag.active = false;
    if (drag.active)
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                if (drag.view.neutralWorld.m[row][col] != view.neutralWorld.m[row][col] ||
                    drag.view.jointWorld.m[row][col] != view.jointWorld.m[row][col])
                    drag.active = false;
    if (drag.active && ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        drag.active = false;
        hit = true;
        return;
    }
    if (drag.active) {
        view = drag.view;
        radius = drag.radius;
    }
    view.outside = RigAuthoring::jointOutsideLimits(view.rule, view.twist, view.swing);
    auto* draw = ImGui::GetBackgroundDrawList();
    const ImU32 axes[3] = {IM_COL32(245, 80, 80, 240), IM_COL32(100, 235, 110, 240),
                           IM_COL32(100, 160, 255, 240)};
    for (int i = 0; i < 3; ++i) {
        const auto axis =
            Vec3(view.jointWorld.m[0][i], view.jointWorld.m[1][i], view.jointWorld.m[2][i])
                .normalize();
        ImVec2 p;
        if (project(center + axis * radius * .55f, p)) {
            draw->AddLine(screen, p, axes[i], 2.f);
            const char* labels[3] = {"X", "Y", "Z"};
            draw->AddText(p, axes[i], labels[i]);
        }
    }
    const ImU32 boundary = view.outside        ? IM_COL32(255, 85, 65, 230)
                           : view.rule.enabled ? IM_COL32(235, 210, 110, 220)
                                               : IM_COL32(155, 155, 165, 140);
    auto segment = [&](const Vec3& a, const Vec3& b, ImU32 color, float width) {
        ImVec2 p, q;
        if (project(a, p) && project(b, q))
            draw->AddLine(p, q, color, width);
    };
    auto arc = [&](float from, float to, float scale, ImU32 color) {
        Vec3 previous = RigAuthoring::jointLimitArcPoint(view, radius * scale, from);
        for (int i = 1; i <= 64; ++i) {
            const auto p = RigAuthoring::jointLimitArcPoint(view, radius * scale,
                                                            from + (to - from) * i / 64.f);
            segment(previous, p, color, 1.5f);
            previous = p;
        }
    };
    const bool angular = view.hasRule && (view.rule.type == "hinge" || view.rule.type == "ball");
    if (angular) {
        arc(-180, 180, 1, IM_COL32(125, 135, 150, 65));
        arc(view.rule.minimum, view.rule.maximum, 1, boundary);
        segment(center, RigAuthoring::jointLimitArcPoint(view, radius, 0),
                IM_COL32(185, 185, 195, 140), 1.f);
        segment(center, RigAuthoring::jointLimitArcPoint(view, radius, view.twist),
                IM_COL32(255, 190, 65, 230), 2.f);
        segment(center, center + RigAuthoring::jointLimitAxis(view) * radius * .7f, boundary, 1.5f);
        if (view.rule.type == "ball") {
            const auto axis = view.rule.axis;
            const auto& matrix = view.jointWorld;
            const auto currentAxis =
                Vec3(matrix.m[0][0] * axis.x + matrix.m[0][1] * axis.y + matrix.m[0][2] * axis.z,
                     matrix.m[1][0] * axis.x + matrix.m[1][1] * axis.y + matrix.m[1][2] * axis.z,
                     matrix.m[2][0] * axis.x + matrix.m[2][1] * axis.y + matrix.m[2][2] * axis.z)
                    .normalize();
            segment(center, center + currentAxis * radius * 1.4f, IM_COL32(115, 245, 150, 230),
                    2.f);
            Vec3 previous =
                RigAuthoring::jointLimitConePoint(view, radius * 1.4f, view.rule.swing, 0);
            for (int i = 1; i <= 64; ++i) {
                const auto p = RigAuthoring::jointLimitConePoint(view, radius * 1.4f,
                                                                 view.rule.swing, i * 360.f / 64);
                segment(previous, p, boundary, 1.5f);
                previous = p;
            }
            for (int i = 0; i < 4; ++i)
                segment(center,
                        RigAuthoring::jointLimitConePoint(view, radius * 1.4f, view.rule.swing,
                                                          i * 90.f),
                        IM_COL32(110, 210, 145, 110), 1.f);
        }
    }

    const bool atLimit = view.rule.enabled && angular &&
                         (std::fabs(view.twist - view.rule.minimum) < .2f ||
                          std::fabs(view.twist - view.rule.maximum) < .2f ||
                          (view.rule.type == "ball" && view.swing >= view.rule.swing - .2f));
    std::string label = (view.ik ? "IK | " : "FK | ") + view.rule.type;
    if (view.hasRule)
        label += view.rule.enabled ? " | rule enabled" : " | rule disabled";
    else
        label += " | no rule";
    if (view.outside)
        label += " | outside limits";
    else if (atLimit)
        label += " | at limit";
    draw->AddText(ImVec2(screen.x + 12, screen.y + 14), boundary, label.c_str());
    if (angular) {
        char text[100];
        std::snprintf(text, sizeof(text), "twist %.1f [%.1f, %.1f] | swing %.1f", view.twist,
                      view.rule.minimum, view.rule.maximum, view.swing);
        draw->AddText(ImVec2(screen.x + 12, screen.y + 30), boundary, text);
    }
    if (!editable) {
        if (state.joint_limits_edit && !angular)
            draw->AddText(ImVec2(screen.x + 12, screen.y + 46), boundary,
                          "Set a hinge/ball rule in Joint Motion");
        return;
    }
    const Vec3 handles[3] = {
        RigAuthoring::jointLimitArcPoint(view, radius, view.rule.minimum),
        RigAuthoring::jointLimitArcPoint(view, radius * 1.18f, view.rule.maximum),
        RigAuthoring::jointLimitConePoint(view, radius * 1.4f, view.rule.swing, 0)};
    const ImU32 colors[3] = {IM_COL32(100, 160, 255, 255), IM_COL32(255, 160, 65, 255),
                             IM_COL32(105, 240, 145, 255)};
    const char* names[3] = {"min", "max", "swing"};
    int nearest = -1;
    float best = 12.f;
    for (int i = 0; i < (view.rule.type == "ball" ? 3 : 2); ++i) {
        ImVec2 p;
        if (!project(handles[i], p))
            continue;
        draw->AddCircleFilled(p, 6.f, colors[i]);
        draw->AddText(ImVec2(p.x + 8, p.y - 5), colors[i], names[i]);
        const float dx = p.x - io.MousePos.x, dy = p.y - io.MousePos.y,
                    d = std::sqrt(dx * dx + dy * dy);
        if (d < best) {
            best = d;
            nearest = i;
        }
    }
    if (!drag.active && nearest >= 0 && !io.KeyAlt &&
        !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) && !ImGuizmo::IsUsing()) {
        hit = true;
        if (ImGui::IsMouseClicked(0)) {
            drag.active = true;
            drag.handle = nearest;
            drag.view = view;
            drag.radius = radius;
            drag.load = ctx.scene.load_counter;
            drag.frame = ctx.scene.timeline.current_frame;
            drag.serial = state.pose.serial;
            drag.mouse = io.MousePos;
            drag.error.clear();
            drag.lastAngle = nearest == 0   ? view.rule.minimum
                             : nearest == 1 ? view.rule.maximum
                                            : view.rule.swing;
        }
    }
    if (!drag.active)
        return;
    hit = true;
    if (ImGui::IsMouseDown(0)) {
        const auto axis = RigAuthoring::jointLimitAxis(drag.view),
                   reference = RigAuthoring::jointLimitReference(drag.view),
                   tangentAxis = axis.cross(reference).normalize();
        const float x = io.MousePos.x / io.DisplaySize.x * 2 - 1,
                    y = 1 - io.MousePos.y / io.DisplaySize.y * 2;
        Vec3 rayOrigin = cam.lookfrom, rayDirection = f;
        if (ortho)
            rayOrigin += r * (x * aspect * half) + u * (y * half);
        else
            rayDirection = (f + r * (x * aspect * tangent) + u * (y * tangent)).normalize();
        const auto normal = drag.handle == 2 ? tangentAxis : axis;
        const float denominator = rayDirection.dot(normal);
        float angle = drag.lastAngle + (io.MousePos.x - drag.mouse.x) * .5f;
        if (std::fabs(denominator) > 1e-4f) {
            const float distance = (center - rayOrigin).dot(normal) / denominator;
            const auto delta = rayOrigin + rayDirection * distance - center;
            if (delta.length_squared() > 1e-10f) {
                const float raw = drag.handle == 2
                                      ? std::atan2(delta.dot(reference), delta.dot(axis))
                                      : std::atan2(delta.dot(tangentAxis), delta.dot(reference));
                angle = raw * 180.f / 3.14159265359f;
                while (angle - drag.lastAngle > 180)
                    angle -= 360;
                while (angle - drag.lastAngle < -180)
                    angle += 360;
            }
        }
        angle = clamp(angle, drag.handle == 0 ? -180.f : 0.f, drag.handle == 0 ? 0.f : 180.f);
        drag.lastAngle = angle;
        drag.mouse = io.MousePos;
        if (drag.handle == 0)
            drag.view.rule.minimum = angle;
        else if (drag.handle == 1)
            drag.view.rule.maximum = angle;
        else
            drag.view.rule.swing = angle;
    } else {
        const auto& rule = drag.view.rule;
        RigAuthoring::JointLimitView original;
        if (RigAuthoring::getJointLimitView(ctx.scene, drag.view.character, drag.view.bone,
                                            original, error) &&
            (rule.minimum != original.rule.minimum || rule.maximum != original.rule.maximum ||
             rule.swing != original.rule.swing)) {
            const auto result =
                rtapi::setRigJointLimits(drag.view.character, drag.view.bone, rule.minimum,
                                         rule.maximum, rule.swing, drag.view.revision);
            drag.error = result.ok ? "" : result.error;
        }
        drag.active = false;
    }
    if (!drag.error.empty())
        ImGui::GetForegroundDrawList()->AddText(ImVec2(30, 130), IM_COL32(255, 120, 80, 255),
                                                drag.error.c_str());
}
}
