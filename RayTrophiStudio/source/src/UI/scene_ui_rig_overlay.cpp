#include "scene_ui.h"
#include "Api/RtApi.h"
#include "Animation/RigEditing.h"
#include "Animation/RigSelection.h"
#include <algorithm>
#include "UI/RigViewportUI.h"
#include "UI/RigEnvelopeOverlay.h"
#include "UI/RigJointLimitsUI.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include <cmath>
#include <unordered_map>

namespace {
struct RigBox {
    bool active = false, add = false;
    int load = -1;
    std::string character;
    ImVec2 start;
} box;
}
void SceneUI::drawSkeletonOverlay(UIContext& ctx, bool& gizmo_hit) {
    if (ctx.scene.rigView.edit_mode) {
        std::string error;
        if (!RigAuthoring::canEditRig(ctx.scene, ctx.scene.rigView.edit_character, error) ||
            mesh_overlay_settings.edit_mode || sculpt_mode_state.enabled ||
            paint_mode_state.enabled)
            rtapi::setRigMode("scene");
    }
    if (!ctx.scene.camera ||
        (!ctx.scene.rigView.visible && !ctx.scene.rigView.envelope_overlay_visible) ||
        rtapi::renderOutputPending() ||
        rtapi::renderStatus().state == rtapi::RenderJobState::Rendering) {
        box.active = false;
        RigUI::cancelRigJointLimitDrag();
        return;
    }
    const Camera& cam = *ctx.scene.camera;
    const ImGuiIO& io = ImGui::GetIO();
    const float width = io.DisplaySize.x, height = io.DisplaySize.y;
    if (width <= 0 || height <= 0) {
        box.active = false;
        return;
    }
    if (!mesh_overlay_settings.edit_mode && !sculpt_mode_state.enabled && !paint_mode_state.enabled) {
        RigUI::drawRigEnvelopeOverlay(ctx, viewport_settings.shading_mode);
        if (ctx.scene.rigView.visible)
            RigUI::drawRigJointLimits(ctx, viewport_settings.shading_mode, gizmo_hit);
        else
            RigUI::cancelRigJointLimitDrag();
    } else {
        RigUI::cancelRigJointLimitDrag();
    }
    if (!ctx.scene.rigView.visible) {
        box.active = false;
        return;
    }
    const Vec3 forward = (cam.lookat - cam.lookfrom).normalize();
    const Vec3 right = forward.cross(cam.vup).normalize(), up = right.cross(forward).normalize();
    const bool ortho = cam.orthographic && viewport_settings.shading_mode != 2;
    const float aspect =
        image_height > 0 ? static_cast<float>(image_width) / image_height : width / height;
    auto project = [&](const Matrix4x4& world, ImVec2& screen) {
        const Vec3 delta = Vec3(world.m[0][3], world.m[1][3], world.m[2][3]) - cam.lookfrom;
        const float depth = delta.dot(forward);
        if (!ortho && depth <= 0.01f)
            return false;
        const float halfH = ortho ? (cam.ortho_height > 1e-4f ? cam.ortho_height : 10.0f) * 0.5f
                                  : depth * std::tan(cam.vfov * 0.5f * 3.14159265359f / 180.0f);
        if (std::fabs(halfH) < 1e-6f || aspect <= 0)
            return false;
        screen = ImVec2((delta.dot(right) / (halfH * aspect) * 0.5f + 0.5f) * width,
                        (0.5f - delta.dot(up) / halfH * 0.5f) * height);
        return std::isfinite(screen.x) && std::isfinite(screen.y);
    };
    if (box.active && (!ctx.scene.rigView.edit_mode || box.load != ctx.scene.load_counter ||
                       box.character != ctx.scene.rigView.edit_character))
        box.active = false;
    std::vector<std::pair<std::string, ImVec2>> boxPoints;
    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    float nearestJoint = 8.0f, nearestSegment = 8.0f;
    std::string jointCharacter, jointBone, segmentCharacter, segmentBone;
    for (const auto& model : ctx.scene.importedModelContexts) {
        if (!model.visible || !model.hasSkeletonRepresentation)
            continue;
        std::vector<RigAuthoring::BoneView> bones;
        std::string error;
        if (!RigAuthoring::listBones(ctx.scene, model.importName, bones, error))
            continue;
        RigUI::applyRigDragPreview(ctx.scene, bones);
        std::unordered_map<std::string, ImVec2> points;
        std::unordered_map<std::string, const RigAuthoring::BoneView*> records;
        for (const auto& b : bones)
            records.emplace(b.name, &b);
        for (const auto& b : bones) {
            ImVec2 p;
            if (project(b.world, p))
                points.emplace(b.name, p);
        }
        for (const auto& b : bones) {
            const auto it = points.find(b.name);
            if (it == points.end())
                continue;
            const ImVec2 p = it->second;
            if (ctx.scene.rigView.edit_mode && b.character == ctx.scene.rigView.edit_character)
                boxPoints.emplace_back(b.name, p);
            const bool selected =
                RigAuthoring::isBoneSelected(ctx.scene.rigView, b.character, b.name);
            const bool active =
                ctx.scene.rigView.character == b.character && ctx.scene.rigView.bone == b.name;
            const ImU32 color = active       ? IM_COL32(255, 170, 45, 255)
                                : selected   ? IM_COL32(150, 245, 135, 255)
                                : b.weighted ? IM_COL32(90, 215, 245, 215)
                                             : IM_COL32(170, 145, 245, 210);
            const auto parent = points.find(b.parent);
            // A bone segment belongs to its starting joint (parent), matching
            // skin weights and the gizmo pivot. The endpoint remains a joint handle.
            if (parent != points.end()) {
                const auto owner = records.find(b.parent);
                if (owner != records.end()) {
                    const auto& start = *owner->second;
                    const bool segmentSelected = RigAuthoring::isBoneSelected(
                        ctx.scene.rigView, start.character, start.name);
                    const bool segmentActive = ctx.scene.rigView.character == start.character &&
                                               ctx.scene.rigView.bone == start.name;
                    const ImU32 segmentColor = segmentActive     ? IM_COL32(255, 170, 45, 255)
                                               : segmentSelected ? IM_COL32(150, 245, 135, 255)
                                               : start.weighted  ? IM_COL32(90, 215, 245, 215)
                                                                 : IM_COL32(170, 145, 245, 210);
                    draw->AddLine(parent->second, p, segmentColor, segmentSelected ? 3.f : 1.5f);
                }
            }
            draw->AddCircleFilled(p, selected ? 5.5f : 3.5f, color);
            if (active)
                draw->AddText(ImVec2(p.x + 9, p.y - 8), color, b.name.c_str());
            const float dx = io.MousePos.x - p.x, dy = io.MousePos.y - p.y;
            const bool selectable =
                !ctx.scene.rigView.edit_mode || ctx.scene.rigView.edit_character == b.character;
            const float distance = std::sqrt(dx * dx + dy * dy);
            if (selectable && distance < nearestJoint) {
                nearestJoint = distance;
                jointCharacter = b.character;
                jointBone = b.name;
            }
            if (parent != points.end()) {
                const float vx = p.x - parent->second.x, vy = p.y - parent->second.y;
                const float length2 = vx * vx + vy * vy;
                if (length2 > 1e-6f) {
                    float t = ((io.MousePos.x - parent->second.x) * vx +
                               (io.MousePos.y - parent->second.y) * vy) /
                              length2;
                    t = t < 0 ? 0 : (t > 1 ? 1 : t);
                    const float sx = io.MousePos.x - parent->second.x - t * vx,
                                sy = io.MousePos.y - parent->second.y - t * vy;
                    const float segmentDistance = std::sqrt(sx * sx + sy * sy);
                    if (selectable && records.count(b.parent) && segmentDistance < nearestSegment) {
                        nearestSegment = segmentDistance;
                        segmentCharacter = b.character;
                        segmentBone = b.parent;
                    }
                }
            }
        }
    }
    // Joint handles take precedence globally, including branch/terminal endpoints.
    const auto& pickBone = jointBone.empty() ? segmentBone : jointBone;
    const auto& pickCharacter = jointBone.empty() ? segmentCharacter : jointCharacter;
    const bool editLocked =
        mesh_overlay_settings.edit_mode || sculpt_mode_state.enabled || paint_mode_state.enabled;
    // ImGuizmo also sets WantCaptureMouse. Its current hover/drag checks below
    // own gizmo gestures; floating UI windows own ordinary UI gestures.
    const bool uiOwnsMouse =
        ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) ||
        ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);

    if (box.active) {
        gizmo_hit = true;
        if (editLocked || uiOwnsMouse || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            box.active = false;
            return;
        }
        const ImVec2 lo(std::min(box.start.x, io.MousePos.x), std::min(box.start.y, io.MousePos.y));
        const ImVec2 hi(std::max(box.start.x, io.MousePos.x), std::max(box.start.y, io.MousePos.y));
        draw->AddRectFilled(lo, hi, IM_COL32(90, 180, 245, 25));
        draw->AddRect(lo, hi, IM_COL32(90, 180, 245, 200));
        if (!ImGui::IsMouseDown(0)) {
            std::vector<std::string> names;
            for (const auto& entry : boxPoints)
                if (entry.second.x >= lo.x && entry.second.x <= hi.x && entry.second.y >= lo.y &&
                    entry.second.y <= hi.y)
                    names.push_back(entry.first);
            rtapi::selectRigBones(box.character, names, "", box.add ? "add" : "replace");
            box.active = false;
        }
        return;
    }
    if (ctx.scene.rigView.edit_mode && pickBone.empty() && !gizmo_hit && !editLocked &&
        !uiOwnsMouse && io.KeyShift && !io.KeyAlt && ImGui::IsMouseClicked(0) &&
        !ImGuizmo::IsOver() && !ImGuizmo::IsUsing()) {
        box.active = true;
        box.add = io.KeyCtrl;
        box.character = ctx.scene.rigView.edit_character;
        box.load = ctx.scene.load_counter;
        box.start = io.MousePos;
        gizmo_hit = true;
        return;
    }
    if (!pickBone.empty() && !gizmo_hit && !editLocked && !uiOwnsMouse &&
        ImGui::IsMouseClicked(0) && !ImGuizmo::IsOver() && !ImGuizmo::IsUsing()) {
        if (rtapi::selectRigBones(pickCharacter, {pickBone}, "",
                                  io.KeyShift  ? "range"
                                  : io.KeyCtrl ? "toggle"
                                               : "replace")
                .ok)
            gizmo_hit = true;
    }
}
