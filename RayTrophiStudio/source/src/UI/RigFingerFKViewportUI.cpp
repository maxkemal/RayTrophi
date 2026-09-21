#include "UI/RigFingerFKUI.h"
#include "Animation/RigAnatomy.h"
#include "Api/RtApi.h"
#include "globals.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "scene_ui.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace RigUI {
namespace {

struct DrivenOverlayDrag {
    bool active = false;
    bool changed = false;
    int frame = -1;
    uint64_t revision = 0;
    ImVec2 origin;
    std::string character;
    std::string control;
    std::string interaction;
    float minimum = -1.f;
    float maximum = 1.f;
    float value = 0.f;
};

struct DrivenOverlayGroup {
    std::string group;
    std::string anchor;
    std::string side;
    std::string shape;
    ImVec2 anchorPoint;
    ImVec2 point;
    float visualScale = 1.f;
    uint64_t revision = 0;
    std::vector<const RigAuthoring::RigDrivenControl*> controls;
};

DrivenOverlayDrag drag;

float overlayScale() {
    return std::max(1.f, ImGui::GetFontSize() / 13.f);
}

bool projectDrivenPoint(const UIContext& context, int shadingMode, const Vec3& world,
                        ImVec2& screen) {
    const auto& io = ImGui::GetIO();
    const auto& camera = *context.scene.camera;
    const auto forward = (camera.lookat - camera.lookfrom).normalize();
    const auto right = forward.cross(camera.vup).normalize();
    const auto up = right.cross(forward).normalize();
    const auto delta = world - camera.lookfrom;
    const float depth = delta.dot(forward);
    const bool orthographic = camera.orthographic && shadingMode != 2;
    if (!orthographic && depth <= .01f) {
        return false;
    }
    const float aspect = image_height > 0
                             ? static_cast<float>(image_width) / image_height
                             : io.DisplaySize.x / io.DisplaySize.y;
    const float halfHeight = orthographic
                                 ? (camera.ortho_height > 1e-4f ? camera.ortho_height : 10.f) * .5f
                                 : depth * std::tan(camera.vfov * 3.14159265359f / 360.f);
    if (!std::isfinite(aspect) || aspect <= 0.f || !std::isfinite(halfHeight) ||
        std::fabs(halfHeight) < 1e-6f) {
        return false;
    }
    screen.x = (delta.dot(right) / (halfHeight * aspect) * .5f + .5f) * io.DisplaySize.x;
    screen.y = (.5f - delta.dot(up) / halfHeight * .5f) * io.DisplaySize.y;
    return std::isfinite(screen.x) && std::isfinite(screen.y);
}

Vec3 bonePoint(const RigAuthoring::BoneView& bone) {
    return Vec3(bone.world.m[0][3], bone.world.m[1][3], bone.world.m[2][3]);
}

const RigAuthoring::BoneView* findBone(const std::vector<RigAuthoring::BoneView>& bones,
                                      const std::string& name) {
    const auto found = std::find_if(bones.begin(), bones.end(), [&](const auto& bone) {
        return bone.name == name;
    });
    return found == bones.end() ? nullptr : &*found;
}

float groupWorldLength(const DrivenOverlayGroup& group,
                       const std::vector<RigAuthoring::BoneView>& bones) {
    float maximum = 0.f;
    for (const auto* control : group.controls) {
        for (const auto& driver : control->drivers) {
            const auto* current = findBone(bones, driver.bone);
            float length = 0.f;
            for (int depth = 0; current && current->name != group.anchor && depth < 64; ++depth) {
                const auto* parent = findBone(bones, current->parent);
                if (!parent) {
                    length = 0.f;
                    break;
                }
                length += (bonePoint(*current) - bonePoint(*parent)).length();
                current = parent;
            }
            if (current && current->name == group.anchor) {
                maximum = std::max(maximum, length);
            }
        }
    }
    return maximum;
}

float projectedGroupScale(const UIContext& context, int shadingMode, const Vec3& anchor,
                          float worldLength) {
    const float dpiScale = overlayScale();
    if (worldLength <= 1e-6f) {
        return dpiScale;
    }
    const auto& camera = *context.scene.camera;
    const auto forward = (camera.lookat - camera.lookfrom).normalize();
    const auto right = forward.cross(camera.vup).normalize();
    ImVec2 start;
    ImVec2 end;
    if (!projectDrivenPoint(context, shadingMode, anchor, start) ||
        !projectDrivenPoint(context, shadingMode, anchor + right * worldLength, end)) {
        return dpiScale;
    }
    const float pixels = std::fabs(end.x - start.x);
    const float anatomical = std::clamp(pixels / (52.f * dpiScale), .62f, 1.65f);
    return dpiScale * anatomical;
}

ImU32 controlColor(const std::string& side, bool selected, int alpha = 235) {
    if (selected) {
        return IM_COL32(255, 175, 50, 255);
    }
    if (side == "left") {
        return IM_COL32(90, 175, 255, alpha);
    }
    if (side == "right") {
        return IM_COL32(255, 105, 105, alpha);
    }
    return IM_COL32(245, 220, 100, alpha);
}

void drawHandShape(ImDrawList* draw, const ImVec2& point, bool left, ImU32 color,
                   bool selected, float scale) {
    const float direction = left ? -1.f : 1.f;
    const float thickness = selected ? 3.f : 2.f;
    const float size = (selected ? 1.08f : 1.f) * scale;
    const ImVec2 palmMin(point.x - 9.f * size, point.y - 8.f * size);
    const ImVec2 palmMax(point.x + 9.f * size, point.y + 9.f * size);
    draw->AddRectFilled(palmMin, palmMax, IM_COL32(20, 25, 35, selected ? 105 : 65),
                        5.f * scale);
    draw->AddRect(palmMin, palmMax, color, 5.f * scale, 0, thickness * scale);
    static constexpr float FingerX[] = {-7.5f, -2.5f, 2.5f, 7.5f};
    static constexpr float FingerLength[] = {10.f, 14.f, 13.f, 9.f};
    for (int index = 0; index < 4; ++index) {
        const float x = point.x + FingerX[index] * size;
        draw->AddLine(ImVec2(x, palmMin.y + scale),
                      ImVec2(x, palmMin.y - FingerLength[index] * size), color,
                      thickness * scale);
    }
    draw->AddLine(ImVec2(point.x + direction * 8.f * size, point.y - size),
                  ImVec2(point.x + direction * 17.f * size, point.y + 6.f * size), color,
                  thickness * scale);
    draw->AddCircleFilled(point, selected ? 3.5f * scale : 2.5f * scale, color);
}

void drawAnchorShape(ImDrawList* draw, const DrivenOverlayGroup& group, ImU32 color,
                     bool selected, float scale) {
    if (group.shape == "hand") {
        drawHandShape(draw, group.point, group.side == "left", color, selected, scale);
        return;
    }
    const float radius = (selected ? 14.f : 12.f) * scale;
    draw->AddCircleFilled(group.point, radius, IM_COL32(20, 25, 35, selected ? 105 : 65), 24);
    draw->AddCircle(group.point, radius, color, 24, (selected ? 3.f : 2.f) * scale);
    draw->AddCircleFilled(group.point, selected ? 3.5f * scale : 2.5f * scale, color);
}

std::string controlSemantic(const RigAuthoring::RigDrivenControl& control) {
    const auto dot = control.id.find_last_of('.');
    return dot == std::string::npos ? control.id : control.id.substr(dot + 1);
}

std::string interactionFor(const RigAuthoring::RigDrivenControl& control) {
    const auto semantic = controlSemantic(control);
    if (semantic == "curl" || semantic == "thumb") {
        return "vertical";
    }
    return "horizontal";
}

ImVec2 controlHandlePoint(const DrivenOverlayGroup& group, size_t index, float scale) {
    const float layoutScale = std::max(scale, overlayScale() * .8f);
    const float direction = group.side == "left" ? -1.f : 1.f;
    if (group.controls.size() == 3) {
        static const ImVec2 Offsets[] = {
            ImVec2(0.f, -48.f), ImVec2(48.f, 0.f), ImVec2(0.f, 48.f)};
        const auto offset = Offsets[index];
        return ImVec2(group.point.x + offset.x * direction * layoutScale,
                      group.point.y + offset.y * layoutScale);
    }
    const float divisor = static_cast<float>(std::max<size_t>(1, group.controls.size() - 1));
    const float angle = -2.45f + static_cast<float>(index) * 1.75f / divisor;
    const float radius = 49.f * layoutScale;
    return ImVec2(group.point.x + std::cos(angle) * radius * direction,
                  group.point.y + std::sin(angle) * radius);
}

void drawArrowHead(ImDrawList* draw, const ImVec2& tip, const ImVec2& direction,
                   ImU32 color, float scale) {
    const ImVec2 normal(-direction.y, direction.x);
    const ImVec2 base(tip.x - direction.x * 5.f * scale,
                      tip.y - direction.y * 5.f * scale);
    draw->AddTriangleFilled(tip,
                            ImVec2(base.x + normal.x * 3.f * scale,
                                   base.y + normal.y * 3.f * scale),
                            ImVec2(base.x - normal.x * 3.f * scale,
                                   base.y - normal.y * 3.f * scale),
                            color);
}

void drawControlGlyph(ImDrawList* draw, const ImVec2& point,
                      const RigAuthoring::RigDrivenControl& control, ImU32 color,
                      float scale) {
    const auto semantic = controlSemantic(control);
    if (semantic == "curl") {
        draw->PathArcTo(point, 8.f * scale, 2.55f, 6.05f, 18);
        draw->PathStroke(color, 0, 2.f * scale);
        drawArrowHead(draw, ImVec2(point.x + 7.8f * scale, point.y - 1.8f * scale),
                      ImVec2(.35f, .94f), color, scale);
    } else if (semantic == "spread") {
        draw->AddLine(ImVec2(point.x - 8.f * scale, point.y),
                      ImVec2(point.x + 8.f * scale, point.y), color, 2.f * scale);
        drawArrowHead(draw, ImVec2(point.x - 9.f * scale, point.y), ImVec2(-1.f, 0.f),
                      color, scale);
        drawArrowHead(draw, ImVec2(point.x + 9.f * scale, point.y), ImVec2(1.f, 0.f),
                      color, scale);
    } else if (semantic == "thumb") {
        draw->AddLine(ImVec2(point.x - 5.f * scale, point.y + 6.f * scale),
                      ImVec2(point.x + 5.f * scale, point.y - 5.f * scale), color,
                      3.f * scale);
        draw->AddCircle(ImVec2(point.x + 6.f * scale, point.y - 6.f * scale),
                        3.f * scale, color, 12, 2.f * scale);
    } else {
        const char glyph[] = {control.label.empty() ? '?' : control.label.front(), '\0'};
        const auto size = ImGui::CalcTextSize(glyph);
        draw->AddText(ImVec2(point.x - size.x * .5f, point.y - size.y * .5f), color, glyph);
    }
}

void drawControlHandle(ImDrawList* draw, const ImVec2& point,
                       const RigAuthoring::RigDrivenControl& control, ImU32 color,
                       bool hovered, bool active, float scale) {
    const float radius = (active ? 16.f : hovered ? 15.f : 14.f) * scale;
    const int alpha = active ? 145 : hovered ? 105 : 55;
    draw->AddCircleFilled(point, radius, IM_COL32(20, 25, 35, alpha), 24);
    draw->AddCircle(point, radius, color, 24, (active ? 3.f : 2.f) * scale);
    drawControlGlyph(draw, point, control, color, scale);
}

bool pointerNear(const ImVec2& pointer, const ImVec2& point, float radius) {
    const float x = pointer.x - point.x;
    const float y = pointer.y - point.y;
    return x * x + y * y <= radius * radius;
}

void cancelDrag() {
    if (drag.active && drag.changed) {
        rtapi::cancelRigPosePreview(drag.character);
    }
    drag = {};
}

float dragValue(const ImVec2& mouse) {
    const float distance = drag.interaction == "vertical" ? drag.origin.y - mouse.y
                                                           : mouse.x - drag.origin.x;
    const float sensitivity = ImGui::GetIO().KeyShift ? 320.f : 110.f;
    return std::clamp(distance / sensitivity, drag.minimum, drag.maximum);
}

} // namespace

bool drawRigFingerFKOverlay(UIContext& context, int shadingMode, bool available, bool& hit) {
    const auto& pose = context.scene.rigView.pose;
    if (!available || !pose.active || !context.scene.camera) {
        if (drag.active) {
            cancelDrag();
        }
        return false;
    }

    const SceneData::ImportedModelContext* model = nullptr;
    for (const auto& candidate : context.scene.importedModelContexts) {
        if (candidate.importName == pose.character &&
            !candidate.rigAnatomy.drivenControls.empty()) {
            model = &candidate;
            break;
        }
    }
    if (!model) {
        return false;
    }

    std::vector<RigAuthoring::BoneView> bones;
    if (!rtapi::listRigBones(pose.character, bones).ok) {
        return false;
    }
    std::vector<DrivenOverlayGroup> groups;
    for (const auto& control : model->rigAnatomy.drivenControls) {
        auto group = std::find_if(groups.begin(), groups.end(), [&](const auto& item) {
            return item.group == control.group && item.anchor == control.anchor;
        });
        if (group == groups.end()) {
            DrivenOverlayGroup newGroup;
            newGroup.group = control.group;
            newGroup.anchor = control.anchor;
            newGroup.side = control.side;
            newGroup.shape = control.shape;
            groups.push_back(std::move(newGroup));
            group = groups.end() - 1;
        }
        group->controls.push_back(&control);
    }
    for (auto& group : groups) {
        const auto bone = std::find_if(bones.begin(), bones.end(), [&](const auto& item) {
            return item.name == group.anchor;
        });
        if (bone == bones.end()) {
            group.controls.clear();
            continue;
        }
        group.revision = bone->rig_revision;
        const Vec3 world = bonePoint(*bone);
        if (!projectDrivenPoint(context, shadingMode, world, group.anchorPoint)) {
            group.controls.clear();
            continue;
        }
        group.visualScale = projectedGroupScale(
            context, shadingMode, world, groupWorldLength(group, bones));
        group.point = group.anchorPoint;
        const float direction = group.side == "left" ? -1.f : 1.f;
        const float offset = std::max(25.f * overlayScale(), 31.f * group.visualScale);
        group.point.x += direction * offset;
    }

    if (drag.active && (drag.character != pose.character || drag.frame != pose.frame ||
                        ImGui::IsKeyPressed(ImGuiKey_Escape))) {
        cancelDrag();
        hit = true;
        return true;
    }

    const auto& io = ImGui::GetIO();
    auto* draw = ImGui::GetBackgroundDrawList();
    const float scale = overlayScale();
    DrivenOverlayGroup* hoveredGroup = nullptr;
    const RigAuthoring::RigDrivenControl* hoveredControl = nullptr;
    float nearest = 24.f * scale;

    for (auto& group : groups) {
        if (group.controls.empty()) {
            continue;
        }
        const bool selected = context.scene.rigView.character == pose.character &&
                              context.scene.rigView.bone == group.anchor;
        const auto color = controlColor(group.side, selected);
        draw->AddLine(group.anchorPoint, group.point, IM_COL32(255, 255, 255, 55), scale);
        drawAnchorShape(draw, group, color, selected, group.visualScale);
        if (pointerNear(io.MousePos, group.point, 24.f * scale)) {
            const float x = io.MousePos.x - group.point.x;
            const float y = io.MousePos.y - group.point.y;
            const float distance = std::sqrt(x * x + y * y);
            if (distance < nearest) {
                nearest = distance;
                hoveredGroup = &group;
                hoveredControl = nullptr;
            }
        }
        if (!selected) {
            continue;
        }
        for (size_t index = 0; index < group.controls.size(); ++index) {
            const auto point = controlHandlePoint(group, index, group.visualScale);
            draw->AddLine(group.point, point, IM_COL32(255, 255, 255, 55), scale);
            const auto* control = group.controls[index];
            const bool active = drag.active && drag.control == control->id;
            const bool hovered = pointerNear(io.MousePos, point, 18.f * scale);
            drawControlHandle(draw, point, *control, controlColor(group.side, active), hovered,
                              active, group.visualScale);
            if (hovered) {
                hoveredGroup = &group;
                hoveredControl = control;
            }
            if (hovered || active) {
                const std::string label = active
                                              ? control->label + "  " +
                                                    std::to_string(drag.value).substr(0, 5)
                                              : control->label;
                const auto size = ImGui::CalcTextSize(label.c_str());
                const float textOffset = std::max(20.f * scale, 20.f * group.visualScale);
                const ImVec2 textPoint(point.x - size.x * .5f, point.y + textOffset);
                draw->AddRectFilled(ImVec2(textPoint.x - 4.f, textPoint.y - 2.f),
                                    ImVec2(textPoint.x + size.x + 4.f,
                                           textPoint.y + size.y + 2.f),
                                    IM_COL32(15, 18, 25, 195), 3.f);
                draw->AddText(textPoint, controlColor(group.side, active), label.c_str());
            }
        }
    }

    const bool uiOwnsMouse = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) ||
                             ImGui::IsPopupOpen(
                                 "", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
    bool claimed = false;
    if (!drag.active && hoveredGroup && !uiOwnsMouse && !ImGuizmo::IsOver() &&
        !ImGuizmo::IsUsing() && ImGui::IsMouseClicked(0)) {
        if (hoveredControl && !pose.hasPreview) {
            drag.active = true;
            drag.frame = pose.frame;
            drag.revision = hoveredGroup->revision;
            drag.origin = io.MousePos;
            drag.character = pose.character;
            drag.control = hoveredControl->id;
            drag.interaction = interactionFor(*hoveredControl);
            drag.minimum = hoveredControl->minimum;
            drag.maximum = hoveredControl->maximum;
            drag.value = 0.f;
        } else if (!hoveredControl) {
            rtapi::selectRigBone(pose.character, hoveredGroup->anchor);
        }
        claimed = true;
    }

    if (drag.active) {
        hit = true;
        claimed = true;
        if (ImGui::IsMouseDown(0)) {
            const float value = dragValue(io.MousePos);
            if (std::fabs(value - drag.value) > 1e-5f) {
                if (std::fabs(value) <= 1e-5f) {
                    if (drag.changed) {
                        rtapi::cancelRigPosePreview(drag.character);
                    }
                    drag.value = 0.f;
                    drag.changed = false;
                } else {
                    const auto result = rtapi::previewRigControlValues(
                        drag.character, {{drag.control, value}}, drag.revision);
                    if (result.ok) {
                        drag.value = value;
                        drag.changed = true;
                    }
                }
            }
        } else {
            if (drag.changed) {
                const auto result = rtapi::applyRigPosePreview(drag.character);
                if (!result.ok) {
                    rtapi::cancelRigPosePreview(drag.character);
                }
            }
            drag = {};
        }
    } else if (hoveredGroup) {
        hit = true;
    }

    if (hoveredControl && !drag.active) {
        ImGui::SetTooltip("%s\nDrag %s | Shift: fine control",
                          hoveredControl->label.c_str(),
                          interactionFor(*hoveredControl) == "vertical" ? "vertically"
                                                                         : "horizontally");
    }
    return claimed;
}

} // namespace RigUI
