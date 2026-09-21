#include "UI/RigIKUI.h"
#include "UI/RigFingerFKUI.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "globals.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include <algorithm>
#include <cmath>
namespace RigUI {
namespace {
struct IKDrag {
    bool active = false, changed = false, blocked = false;
    int load = -1, frame = -1;
    uint64_t revision = 0, serial = 0;
    std::string character, control, handle, error;
    Vec3 target, pole, world;
    Quaternion orientation;
    std::vector<Vec3> spline;
} drag;
bool projectPoint(const UIContext& ctx, int shadingMode, const Vec3& world, ImVec2& screen) {
    const auto& io = ImGui::GetIO();
    const auto& camera = *ctx.scene.camera;
    const auto forward = (camera.lookat - camera.lookfrom).normalize();
    const auto right = forward.cross(camera.vup).normalize();
    const auto up = right.cross(forward).normalize();
    const auto delta = world - camera.lookfrom;
    const float depth = delta.dot(forward);
    const bool ortho = camera.orthographic && shadingMode != 2;
    if (!ortho && depth <= .01f) {
        return false;
    }
    const float aspect = image_height > 0
                             ? static_cast<float>(image_width) / image_height
                             : io.DisplaySize.x / io.DisplaySize.y;
    const float halfHeight = ortho
                                 ? (camera.ortho_height > 1e-4f ? camera.ortho_height : 10.f) * .5f
                                 : depth * std::tan(camera.vfov * 3.14159265359f / 360.f);
    if (!std::isfinite(aspect) || aspect <= 0 || !std::isfinite(halfHeight) ||
        std::fabs(halfHeight) < 1e-6f) {
        return false;
    }
    screen.x = (delta.dot(right) / (halfHeight * aspect) * .5f + .5f) * io.DisplaySize.x;
    screen.y = (.5f - delta.dot(up) / halfHeight * .5f) * io.DisplaySize.y;
    return std::isfinite(screen.x) && std::isfinite(screen.y);
}
Vec3 jsonPoint(const nlohmann::json& row, const char* key) {
    return Vec3(row[key][0].get<float>(), row[key][1].get<float>(), row[key][2].get<float>());
}

float controlVisualScale(const UIContext& ctx, int shadingMode, const nlohmann::json& row) {
    float worldLength = row.value("length_world", 0.f);
    if (worldLength <= 1e-6f && row.contains("fk_handles") &&
        row["fk_handles"].is_array() && row["fk_handles"].size() >= 2) {
        Vec3 previous = jsonPoint(row["fk_handles"][0], "world");
        for (size_t index = 1; index < row["fk_handles"].size(); ++index) {
            const Vec3 current = jsonPoint(row["fk_handles"][index], "world");
            worldLength += (current - previous).length();
            previous = current;
        }
        if (row.contains("tip_world")) {
            worldLength += (jsonPoint(row, "tip_world") - previous).length();
        }
    }
    if (worldLength <= 1e-6f) {
        return 1.f;
    }
    const auto& camera = *ctx.scene.camera;
    const auto forward = (camera.lookat - camera.lookfrom).normalize();
    const auto right = forward.cross(camera.vup).normalize();
    const Vec3 anchor = jsonPoint(row, "target_world");
    ImVec2 start;
    ImVec2 end;
    if (!projectPoint(ctx, shadingMode, anchor, start) ||
        !projectPoint(ctx, shadingMode, anchor + right * worldLength, end)) {
        return 1.f;
    }
    const float dpiScale = std::max(1.f, ImGui::GetFontSize() / 13.f);
    const float pixels = std::fabs(end.x - start.x);
    return std::clamp(pixels / (130.f * dpiScale), .65f, 1.65f);
}

void drawTargetShape(ImDrawList* draw, const ImVec2& point, const std::string& shape,
                     float scale, ImU32 color, bool selected) {
    const float dpiScale = std::max(1.f, ImGui::GetFontSize() / 13.f);
    const float size = (selected ? 15.f : 13.f) * scale * dpiScale;
    const float thickness = (selected ? 3.f : 2.f) * dpiScale;
    if (shape == "hand") {
        draw->AddRect(ImVec2(point.x - size, point.y - size * .72f),
                      ImVec2(point.x + size, point.y + size * .72f), color, 3.f, 0, thickness);
    } else if (shape == "foot") {
        const ImVec2 points[] = {ImVec2(point.x - size, point.y - size * .55f),
                                 ImVec2(point.x + size * 1.25f, point.y - size * .35f),
                                 ImVec2(point.x + size, point.y + size * .55f),
                                 ImVec2(point.x - size, point.y + size * .55f)};
        draw->AddPolyline(points, 4, color, ImDrawFlags_Closed, thickness);
    } else if (shape == "aim") {
        draw->AddTriangle(ImVec2(point.x, point.y - size),
                          ImVec2(point.x + size, point.y + size),
                          ImVec2(point.x - size, point.y + size), color, thickness);
    } else if (shape == "root") {
        draw->AddCircle(point, size, color, 24, thickness);
        draw->AddCircle(point, size * .62f, color, 24, thickness);
        draw->AddLine(ImVec2(point.x - size, point.y), ImVec2(point.x + size, point.y), color,
                      thickness);
    } else if (shape == "arc") {
        draw->PathArcTo(point, size, 3.65f, 5.78f, 14);
        draw->PathStroke(color, 0, thickness);
        draw->AddLine(ImVec2(point.x - size * .85f, point.y - size * .48f),
                      ImVec2(point.x - size * .45f, point.y - size * .9f), color, thickness);
    } else {
        draw->AddCircle(point, size, color, 20, thickness);
    }
    draw->AddCircleFilled(point, selected ? 3.f : 2.f, color);
}
void drawPoleShape(ImDrawList* draw, const ImVec2& point, float scale, ImU32 color) {
    const float dpiScale = std::max(1.f, ImGui::GetFontSize() / 13.f);
    const float size = 12.f * scale * dpiScale;
    const ImVec2 points[] = {ImVec2(point.x, point.y - size),
                             ImVec2(point.x + size, point.y),
                             ImVec2(point.x, point.y + size),
                             ImVec2(point.x - size, point.y)};
    draw->AddPolyline(points, 4, color, ImDrawFlags_Closed, 2.f * dpiScale);
}
bool drawControlShapes(UIContext& ctx, int shadingMode, bool available, bool& hit) {
    const auto& session = ctx.scene.rigView.pose;
    if (!available || !session.active || !ctx.scene.camera) {
        return false;
    }
    nlohmann::json data;
    if (!rtapi::getRigControls(session.character, data).ok) {
        return false;
    }
    auto* draw = ImGui::GetBackgroundDrawList();
    const auto& io = ImGui::GetIO();
    const float dpiScale = std::max(1.f, ImGui::GetFontSize() / 13.f);
    float nearest = 24.f * dpiScale;
    std::string hoveredControl;
    std::string hoveredHandle;
    std::string hoveredBone;
    std::string hoveredSwitch;
    bool switchToIK = false;
    for (const auto& row : data["controls"]) {
        const auto name = row["name"].get<std::string>();
        const bool selected = name == session.control;
        const auto& display = row["display"];
        const auto& targetDisplay = display["target"];
        ImVec2 target;
        if (!projectPoint(ctx, shadingMode, jsonPoint(row, "target_world"), target)) {
            continue;
        }
        const float visualScale = controlVisualScale(ctx, shadingMode, row);
        const auto side = display.value("color_role", std::string("center"));
        const bool enabled = row["enabled"].get<bool>() && row["blend"].get<float>() > 0.f;
        const bool contact = row["contact"].get<bool>();
        const int alpha = enabled ? 235 : 125;
        const ImU32 color = selected ? IM_COL32(255, 175, 50, 255)
                            : side == "left" ? IM_COL32(90, 175, 255, alpha)
                            : side == "right" ? IM_COL32(255, 105, 105, alpha)
                                                : IM_COL32(245, 220, 100, alpha);
        drawTargetShape(draw, target, targetDisplay.value("shape", std::string("ring")),
                        targetDisplay.value("scale", 1.f) * visualScale, color, selected);
        if (contact) {
            draw->AddCircleFilled(target,
                                  (selected ? 4.5f : 4.f) * visualScale * dpiScale,
                                  IM_COL32(90, 245, 160, 255));
            draw->AddCircle(target,
                            (selected ? 6.5f : 6.f) * visualScale * dpiScale,
                            IM_COL32(225, 255, 238, 230), 16, 1.5f * dpiScale);
        }
        const float dx = io.MousePos.x - target.x;
        const float dy = io.MousePos.y - target.y;
        const float distance = std::sqrt(dx * dx + dy * dy);
        if (distance < nearest) {
            nearest = distance;
            hoveredControl = name;
            hoveredHandle = "target";
            hoveredBone.clear();
        }
        if (!enabled && row["solver"] == "two_bone") {
            for (const auto& handle : row["fk_handles"]) {
                ImVec2 fkPoint;
                if (!projectPoint(ctx, shadingMode, jsonPoint(handle, "world"), fkPoint)) {
                    continue;
                }
                const auto bone = handle["bone"].get<std::string>();
                const bool activeBone = ctx.scene.rigView.character == session.character &&
                                        ctx.scene.rigView.bone == bone;
                const auto fkColor = activeBone ? IM_COL32(255, 175, 50, 255)
                                                : IM_COL32(215, 225, 245, 185);
                const float fkScale = display["fk"].value("scale", .72f);
                draw->AddCircle(fkPoint,
                                (activeBone ? 13.f : 12.f) * fkScale * visualScale * dpiScale,
                                fkColor, 18,
                                (activeBone ? 3.f : 2.f) * dpiScale);
                const float fkX = io.MousePos.x - fkPoint.x;
                const float fkY = io.MousePos.y - fkPoint.y;
                const float fkDistance = std::sqrt(fkX * fkX + fkY * fkY);
                if (fkDistance < nearest) {
                    nearest = fkDistance;
                    hoveredBone = bone;
                    hoveredControl.clear();
                    hoveredHandle.clear();
                }
            }
        }
        if (!selected) {
            continue;
        }
        if (row["solver"] == "two_bone") {
            const float badgeOffset = 17.f * visualScale * dpiScale;
            const ImVec2 badgeMin(target.x + badgeOffset, target.y + 8.f * dpiScale);
            const ImVec2 badgeMax(badgeMin.x + 30.f * dpiScale,
                                  badgeMin.y + 18.f * dpiScale);
            draw->AddRectFilled(badgeMin, badgeMax,
                                enabled ? IM_COL32(45, 125, 90, 230)
                                        : IM_COL32(80, 85, 100, 220),
                                3.f);
            draw->AddRect(badgeMin, badgeMax, color, 3.f);
            draw->AddText(ImVec2(badgeMin.x + 5.f, badgeMin.y + 1.f), IM_COL32_WHITE,
                          enabled ? "IK" : "FK");
            if (io.MousePos.x >= badgeMin.x && io.MousePos.x <= badgeMax.x &&
                io.MousePos.y >= badgeMin.y && io.MousePos.y <= badgeMax.y) {
                hoveredSwitch = name;
                switchToIK = !enabled;
            }
        }
        ImVec2 pole;
        if (projectPoint(ctx, shadingMode, jsonPoint(row, "pole_world"), pole)) {
            draw->AddLine(target, pole, IM_COL32(255, 255, 255, 70), 1.f);
            drawPoleShape(draw, pole,
                          display["pole"].value("scale", .78f) * visualScale,
                          IM_COL32(255, 215, 90, 235));
            const float poleX = io.MousePos.x - pole.x;
            const float poleY = io.MousePos.y - pole.y;
            const float poleDistance = std::sqrt(poleX * poleX + poleY * poleY);
            if (poleDistance < nearest) {
                nearest = poleDistance;
                hoveredControl = name;
                hoveredHandle = "pole";
                hoveredBone.clear();
            }
        }
        const std::string label = name;
        draw->AddText(ImVec2(target.x + 18.f * visualScale * dpiScale,
                             target.y - 8.f * dpiScale),
                      color, label.c_str());
    }
    if (hoveredControl.empty() && hoveredBone.empty() && hoveredSwitch.empty()) {
        return false;
    }
    hit = true;
    const bool uiOwnsMouse = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) ||
                             ImGui::IsPopupOpen(
                                 "", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
    bool claimedClick = false;
    if (!uiOwnsMouse && !ImGuizmo::IsOver() && !ImGuizmo::IsUsing() &&
        ImGui::IsMouseClicked(0)) {
        if (!hoveredSwitch.empty()) {
            if (session.hasPreview) {
                drag.error = "Apply or cancel the current pose preview first";
            } else {
                const auto matched =
                    switchToIK
                        ? rtapi::matchRigIKToFK(session.character, hoveredSwitch,
                                               data["rig_revision"].get<uint64_t>())
                        : rtapi::matchRigFKToIK(session.character, hoveredSwitch,
                                               data["rig_revision"].get<uint64_t>());
                if (!matched.ok) {
                    drag.error = matched.error;
                } else {
                    const auto applied = rtapi::applyRigPosePreview(session.character);
                    drag.error = applied.ok ? "" : applied.error;
                }
            }
        } else if (!hoveredBone.empty()) {
            rtapi::selectRigBone(session.character, hoveredBone);
        } else {
            rtapi::selectRigControl(session.character, hoveredControl, hoveredHandle);
        }
        claimedClick = true;
    }
    return claimedClick;
}
void cancel() {
    if (drag.active) {
        rtapi::cancelRigPosePreview(drag.character);
        ImGuizmo::Enable(false);
        ImGuizmo::Enable(true);
    }
    drag.active = false;
    drag.changed = false;
    drag.blocked = ImGui::IsMouseDown(0);
}
}
bool drawRigIKGizmo(UIContext& ctx, int shadingMode, bool available, bool& hit) {
    const bool shapeHit = drawControlShapes(ctx, shadingMode, available, hit);
    const bool fingerHit = drawRigFingerFKOverlay(ctx, shadingMode, available, hit);
    const auto& session = ctx.scene.rigView.pose;
    const bool selected = session.active && !session.control.empty();
    if (fingerHit) {
        if (drag.active)
            cancel();
        return true;
    }
    if (!selected || !available) {
        if (drag.active)
            cancel();
        return selected || shapeHit;
    }
    if (drag.active &&
        (drag.load != ctx.scene.load_counter || drag.frame != ctx.scene.timeline.current_frame ||
         drag.serial != session.serial || drag.character != session.character ||
         drag.control != session.control || drag.handle != session.controlHandle)) {
        cancel();
        return true;
    }
    if (drag.active && ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        cancel();
        hit = true;
        return true;
    }
    // Keep the selected handle visible while the pointer is over a dock/popup.
    // UI ownership blocks drag startup below; it must not suppress rendering.
    const bool uiOwnsMouse =
        !drag.active &&
        (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) ||
         ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel));
    if (drag.blocked) {
        if (!ImGui::IsMouseDown(0))
            drag.blocked = false;
        else
            return true;
    }
    nlohmann::json data;
    auto result = rtapi::getRigControls(session.character, data);
    if (!result.ok) {
        if (drag.active)
            cancel();
        return true;
    }
    nlohmann::json row;
    for (const auto& c : data["controls"])
        if (c["name"] == session.control) {
            row = c;
            break;
        }
    if (row.is_null()) {
        if (drag.active)
            cancel();
        return true;
    }
    if (drag.active && data["rig_revision"].get<uint64_t>() != drag.revision) {
        cancel();
        return true;
    }
    auto point = [&](const char* key) {
        return Vec3(row[key][0].get<float>(), row[key][1].get<float>(), row[key][2].get<float>());
    };
    const auto target = point("target_world"), pole = point("pole_world");
    const bool rotating = session.controlHandle == "orientation";
    const auto& q =
        row[row["orientation_enabled"].get<bool>() ? "orientation_world" : "tip_orientation_world"];
    const Quaternion orientation = drag.active ? drag.orientation
                                               : Quaternion(q[0].get<float>(), q[1].get<float>(),
                                                            q[2].get<float>(), q[3].get<float>());
    const int shape = session.controlHandle == "spline_0"   ? 0
                      : session.controlHandle == "spline_1" ? 1
                                                            : -1;
    std::vector<Vec3> spline;
    for (const auto& point : row["spline_world"])
        spline.emplace_back(point[0].get<float>(), point[1].get<float>(), point[2].get<float>());
    if (shape >= 0 && spline.size() != 2) {
        if (drag.active)
            cancel();
        return true;
    }
    const auto position = drag.active  ? drag.world
                          : shape >= 0 ? spline[shape]
                                       : (session.controlHandle == "pole" ? pole : target);
    auto& io = ImGui::GetIO();
    const auto& cam = *ctx.scene.camera;
    const auto f = (cam.lookat - cam.lookfrom).normalize(), r = f.cross(cam.vup).normalize(),
               u = r.cross(f);
    const float view[16] = {r.x,
                            u.x,
                            -f.x,
                            0,
                            r.y,
                            u.y,
                            -f.y,
                            0,
                            r.z,
                            u.z,
                            -f.z,
                            0,
                            -r.dot(cam.lookfrom),
                            -u.dot(cam.lookfrom),
                            f.dot(cam.lookfrom),
                            1};
    const float aspect =
        image_height > 0 ? float(image_width) / image_height : io.DisplaySize.x / io.DisplaySize.y;
    if (!std::isfinite(aspect) || aspect <= 0)
        return true;
    const bool ortho = cam.orthographic && shadingMode != 2;
    constexpr float nearZ = .1f, farZ = 10000.f;
    float projection[16] = {};
    if (ortho) {
        const float h = cam.ortho_height > 1e-4f ? cam.ortho_height : 10.f;
        projection[0] = 2 / (h * aspect);
        projection[5] = 2 / h;
        projection[10] = -2 / (farZ - nearZ);
        projection[14] = -(farZ + nearZ) / (farZ - nearZ);
        projection[15] = 1;
    } else {
        const float tangent = std::tan(cam.vfov * 3.14159265359f / 360.f);
        if (!std::isfinite(tangent) || tangent <= 0)
            return true;
        projection[0] = 1 / (aspect * tangent);
        projection[5] = 1 / tangent;
        projection[10] = -(farZ + nearZ) / (farZ - nearZ);
        projection[11] = -1;
        projection[14] = -2 * farZ * nearZ / (farZ - nearZ);
    }
    float matrix[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, position.x, position.y, position.z, 1};
    if (rotating) {
        const auto rotation = orientation.toMatrix();
        for (int col = 0; col < 3; ++col)
            for (int row = 0; row < 3; ++row)
                matrix[col * 4 + row] = rotation.m[row][col];
    }
    ImGuizmo::BeginFrame();
    ImGuizmo::SetOrthographic(ortho);
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
    drawRigSplineGuide(row, view, projection, session.controlHandle);
    ImGuizmo::Manipulate(view, projection, rotating ? ImGuizmo::ROTATE : ImGuizmo::TRANSLATE,
                         ImGuizmo::WORLD, matrix);
    const bool usingNow = ImGuizmo::IsUsing();
    hit = hit || usingNow || ImGuizmo::IsOver();
    if (!drag.error.empty())
        ImGui::GetForegroundDrawList()->AddText(ImVec2(30, 110), IM_COL32(255, 120, 80, 255),
                                                drag.error.c_str());
    if (usingNow && !drag.active) {
        if (uiOwnsMouse) {
            ImGuizmo::Enable(false);
            ImGuizmo::Enable(true);
            drag.blocked = ImGui::IsMouseDown(0);
            return true;
        }
        if (session.hasPreview) {
            drag.error = "Apply or cancel the current pose preview first";
            ImGuizmo::Enable(false);
            ImGuizmo::Enable(true);
            drag.blocked = true;
            return true;
        }
        drag.active = true;
        drag.changed = false;
        drag.error.clear();
        drag.character = session.character;
        drag.control = session.control;
        drag.handle = session.controlHandle;
        drag.serial = session.serial;
        drag.frame = ctx.scene.timeline.current_frame;
        drag.load = ctx.scene.load_counter;
        drag.revision = data["rig_revision"].get<uint64_t>();
        drag.target = target;
        drag.pole = pole;
        drag.world = position;
        drag.orientation = orientation;
        drag.spline = spline;
    }
    if (!drag.active)
        return true;
    const Vec3 moved(matrix[12], matrix[13], matrix[14]);
    if (!std::isfinite(moved.x) || !std::isfinite(moved.y) || !std::isfinite(moved.z)) {
        cancel();
        return true;
    }
    if (rotating) {
        Matrix4x4 rotation;
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                rotation.m[row][col] = matrix[col * 4 + row];
        auto next = Quaternion::fromMatrix(rotation);
        next.normalize();
        const auto& old = drag.orientation;
        const float dot =
            std::fabs(next.w * old.w + next.x * old.x + next.y * old.y + next.z * old.z);
        if (dot < 1.f - 1e-7f) {
            const auto result =
                rtapi::setRigIKOrientation(drag.character, drag.control, next, true, drag.revision);
            if (!result.ok) {
                drag.error = result.error;
                cancel();
                return true;
            }
            drag.orientation = next;
            drag.changed = true;
        }
    } else if ((moved - drag.world).length_squared() > 1e-12f) {
        auto nextSpline = drag.spline;
        if (shape >= 0)
            nextSpline[shape] = moved;
        const auto result =
            shape >= 0
                ? rtapi::setRigIKSpline(drag.character, drag.control, nextSpline, true,
                                        drag.revision)
                : rtapi::setRigIKTarget(drag.character, drag.control,
                                        drag.handle == "target" ? moved : drag.target,
                                        drag.handle == "pole" ? moved : drag.pole, drag.revision);
        if (result.ok && shape >= 0)
            drag.spline = std::move(nextSpline);
        if (!result.ok) {
            drag.error = result.error;
            cancel();
            return true;
        }
        drag.world = moved;
        drag.changed = true;
    }
    if (!usingNow) {
        if (drag.changed) {
            const auto r = rtapi::applyRigPosePreview(drag.character);
            if (!r.ok) {
                drag.error = r.error;
                rtapi::cancelRigPosePreview(drag.character);
            }
        }
        drag.active = false;
        drag.changed = false;
    }
    return true;
}
}
