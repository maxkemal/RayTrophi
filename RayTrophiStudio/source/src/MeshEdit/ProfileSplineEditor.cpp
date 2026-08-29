#include "MeshEdit/ProfileSplineEditor.h"
#include "Api/RtApi.h"

#include "MeshEdit/SplineObject.h"
#include "MeshEdit/SplineEditService.h"
#include "MeshEdit/ProfileAuthoringService.h"
#include "MeshEdit/ProfileLoft.h"
#include "MeshEdit/ProfileRevolve.h"
#include "MeshEdit/ProfileSweep.h"
#include "MeshEdit/SplineAnimation.h"
#include "SceneSelection.h"
#include "ProjectManager.h"
#include "scene_ui.h"
#include "imgui.h"

#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>

namespace MeshEdit {
namespace {

template <typename Result>
std::string operationStatus(const Result& result) {
    if (result.report.ok && result.geometry) {
        return "Ready: " + std::to_string(result.geometry->get_vertex_count()) +
            " vertices, " + std::to_string(result.geometry->indices.size() / 3) + " triangles.";
    }
    if (!result.report.diagnostics.empty()) {
        std::string status;
        for (const auto& diagnostic : result.report.diagnostics) {
            if (!status.empty()) status += "\n";
            status += diagnostic.code + ": " + diagnostic.message;
        }
        return status;
    }
    return "Operation failed without a diagnostic.";
}

std::vector<std::shared_ptr<SplineObject>> splineSources(UIContext& ctx, bool closed,
                                                         const SplineObject* exclude = nullptr) {
    std::vector<std::shared_ptr<SplineObject>> result;
    for (const auto& object : ctx.scene.world.objects) {
        auto spline = std::dynamic_pointer_cast<SplineObject>(object);
        if (!spline || spline.get() == exclude || spline->spline.isClosed != closed) continue;
        result.push_back(std::move(spline));
    }
    return result;
}

// Geometry kernels consume profiles in canonical XY (X=radius/lateral,
// Y=height). Authoring objects retain their chosen viewport plane.
BezierSpline canonicalProfile(const SplineObject& object, bool applyObjectScale,
                               bool relativeToObjectPivot = false) {
    BezierSpline result = object.spline;
    auto remap = [&](const Vec3& value) {
        if (object.plane == SplinePlane::XZ) return Vec3(value.x, value.z, 0.0f);
        if (object.plane == SplinePlane::YZ) return Vec3(value.z, value.y, 0.0f);
        return value;
    };
    Vec3 profileScale(1.0f);
    if (applyObjectScale && object.transform) {
        const Matrix4x4 final = object.transform->getFinal();
        const Vec3 scale(
            std::sqrt(final.m[0][0] * final.m[0][0] + final.m[1][0] * final.m[1][0] + final.m[2][0] * final.m[2][0]),
            std::sqrt(final.m[0][1] * final.m[0][1] + final.m[1][1] * final.m[1][1] + final.m[2][1] * final.m[2][1]),
            std::sqrt(final.m[0][2] * final.m[0][2] + final.m[1][2] * final.m[1][2] + final.m[2][2] * final.m[2][2]));
        if (object.plane == SplinePlane::XY) profileScale = Vec3(scale.x, scale.y, 1.0f);
        else if (object.plane == SplinePlane::XZ) profileScale = Vec3(scale.x, scale.z, 1.0f);
        else if (object.plane == SplinePlane::YZ) profileScale = Vec3(scale.z, scale.y, 1.0f);
        // Free is not a planar profile plane; profile-generating consumers
        // (Sweep/Revolve/Loft) must reject it before reaching this point
        // rather than have it silently misread as YZ.
    }
    const Vec3 localPivot = relativeToObjectPivot && object.transform
        ? object.transform->pivot_offset : Vec3(0.0f);
    for (auto& point : result.points) {
        point.position = remap(point.position - localPivot) * profileScale;
        point.tangentIn = remap(point.tangentIn) * profileScale;
        point.tangentOut = remap(point.tangentOut) * profileScale;
    }
    return result;
}

BezierSpline worldSpline(const SplineObject& object) {
    BezierSpline result = object.spline;
    if (!object.transform) return result;
    const Matrix4x4 transform = object.transform->getFinal();
    for (auto& point : result.points) {
        point.position = transform.transform_point(point.position);
        point.tangentIn = transform.transform_vector(point.tangentIn);
        point.tangentOut = transform.transform_vector(point.tangentOut);
    }
    return result;
}

void hashCombine(size_t& seed, size_t value) {
    seed ^= value + static_cast<size_t>(0x9e3779b9u) + (seed << 6) + (seed >> 2);
}

void hashFloat(size_t& seed, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    hashCombine(seed, bits);
}

size_t sourceSignature(const SplineObject& object) {
    size_t seed = object.spline.points.size();
    hashCombine(seed, static_cast<size_t>(object.spline.curveType));
    hashCombine(seed, static_cast<size_t>(object.plane));
    hashCombine(seed, object.spline.isClosed ? 1u : 0u);
    for (const auto& point : object.spline.points) {
        for (const Vec3 value : {point.position, point.tangentIn, point.tangentOut}) {
            hashFloat(seed, value.x); hashFloat(seed, value.y); hashFloat(seed, value.z);
        }
    }
    if (object.transform) {
        const Matrix4x4 transform = object.transform->getFinal();
        for (int row = 0; row < 4; ++row)
            for (int column = 0; column < 4; ++column)
                hashFloat(seed, transform.m[row][column]);
        hashFloat(seed, object.transform->scale.x);
        hashFloat(seed, object.transform->scale.y);
        hashFloat(seed, object.transform->scale.z);
        hashFloat(seed, object.transform->pivot_offset.x);
        hashFloat(seed, object.transform->pivot_offset.y);
        hashFloat(seed, object.transform->pivot_offset.z);
    }
    return seed;
}

template <typename Result>
void setPreview(const std::shared_ptr<SplineObject>& owner,
                const std::shared_ptr<SplineObject>& counterpart,
                const Result& result,
                const char* operation, const Matrix4x4& transform, size_t signature) {
    const std::string status = operationStatus(result);
    auto assign = [&](const std::shared_ptr<SplineObject>& target,
                      const std::shared_ptr<SplineObject>& other) {
        if (!target) return;
        target->profile_preview_operation = operation;
        target->profile_preview_counterpart = other ? other->nodeName : std::string{};
        target->profile_preview_signature = signature;
        target->profile_preview_status = status;
        target->profile_preview_geometry = result.report.ok ? result.geometry : nullptr;
        target->profile_preview_transform = transform;
    };
    assign(owner, counterpart);
    assign(counterpart, owner);
}

void clearPreviewState(const std::shared_ptr<SplineObject>& object) {
    if (!object) return;
    object->profile_preview_geometry.reset();
    object->profile_preview_operation.clear();
    object->profile_preview_counterpart.clear();
    object->profile_preview_status.clear();
    object->profile_preview_signature = 0;
}

void clearLinkedPreview(UIContext& ctx, const std::shared_ptr<SplineObject>& owner) {
    const std::string counterpartName = owner ? owner->profile_preview_counterpart : std::string{};
    clearPreviewState(owner);
    if (counterpartName.empty()) return;
    for (const auto& object : ctx.scene.world.objects) {
        auto spline = std::dynamic_pointer_cast<SplineObject>(object);
        if (spline && spline->nodeName == counterpartName) {
            clearPreviewState(spline);
            return;
        }
    }
}

void drawProfilePlaneScale(SplineObject& object, const char* label) {
    if (!object.transform) return;
    float values[2];
    if (object.plane == SplinePlane::XY) { values[0] = object.transform->scale.x; values[1] = object.transform->scale.y; }
    else if (object.plane == SplinePlane::XZ) { values[0] = object.transform->scale.x; values[1] = object.transform->scale.z; }
    else { values[0] = object.transform->scale.z; values[1] = object.transform->scale.y; }
    if (!ImGui::DragFloat2(label, values, 0.01f, 0.001f, 1000.0f, "%.3f")) return;
    values[0] = std::max(0.001f, values[0]);
    values[1] = std::max(0.001f, values[1]);
    if (object.plane == SplinePlane::XY) { object.transform->scale.x = values[0]; object.transform->scale.y = values[1]; }
    else if (object.plane == SplinePlane::XZ) { object.transform->scale.x = values[0]; object.transform->scale.z = values[1]; }
    else { object.transform->scale.z = values[0]; object.transform->scale.y = values[1]; }
    object.transform->updateMatrix();
    ProjectManager::getInstance().markModified();
}

bool drawPlaneVector(const char* label, Vec3& value, SplinePlane plane) {
    float components[2] = {value.x, value.y};
    const char* suffix = "X/Y";
    if (plane == SplinePlane::XZ) {
        components[0] = value.x; components[1] = value.z; suffix = "X/Z";
    } else if (plane == SplinePlane::YZ) {
        components[0] = value.y; components[1] = value.z; suffix = "Y/Z";
    }
    const std::string fullLabel = std::string(label) + " (" + suffix + ")";
    if (!ImGui::DragFloat2(fullLabel.c_str(), components, 0.05f)) return false;
    if (plane == SplinePlane::XY) value = Vec3(components[0], components[1], 0.0f);
    else if (plane == SplinePlane::XZ) value = Vec3(components[0], 0.0f, components[1]);
    else value = Vec3(0.0f, components[0], components[1]);
    return true;
}

void remapSplinePlane(BezierSpline& spline, SplinePlane from, SplinePlane to) {
    auto remap = [from, to](Vec3& value) {
        float lateral = value.x;
        float vertical = value.y;
        if (from == SplinePlane::XZ) { lateral = value.x; vertical = value.z; }
        else if (from == SplinePlane::YZ) { lateral = value.z; vertical = value.y; }
        if (to == SplinePlane::XY) value = Vec3(lateral, vertical, 0.0f);
        else if (to == SplinePlane::XZ) value = Vec3(lateral, 0.0f, vertical);
        else value = Vec3(0.0f, vertical, lateral);
    };
    for (auto& point : spline.points) {
        remap(point.position);
        remap(point.tangentIn);
        remap(point.tangentOut);
    }
}

} // namespace

void drawProfileSplineEditControls(UIContext& ctx) {
    auto splineObject = ctx.selection.selected.spline_object;
    if (!splineObject) return;

    ImGui::TextColored(ImVec4(0.35f, 1.0f, 0.85f, 1.0f), "Spline Workspace");
    ImGui::TextDisabled("Shape -> Skin & Deform -> Animate");
    ImGui::Separator();
    if (ImGui::CollapsingHeader("1  Shape & Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Object: %s", splineObject->nodeName.c_str());
    const char* planeNames[] = {"XY (Front)", "XZ (Top)", "YZ (Right / Left)"};
    bool changed = false;
    int authoringPlane = static_cast<int>(splineObject->plane);
    if (ImGui::Combo("Authoring Plane", &authoringPlane, planeNames, 3)) {
        const SplinePlane oldPlane = splineObject->plane;
        splineObject->plane = static_cast<SplinePlane>(authoringPlane);
        remapSplinePlane(splineObject->spline, oldPlane, splineObject->plane);
        changed = true;
    }
    ImGui::TextDisabled("Changing plane preserves the profile's 2D lateral/height coordinates.");
    if (splineObject->transform) {
        Vec3 pivot = splineObject->transform->getPivotMatrix().getTranslation();
        ImGui::Text("Pivot: %.3f, %.3f, %.3f", pivot.x, pivot.y, pivot.z);
        if (ImGui::DragFloat3("Pivot World Position", &pivot.x, 0.05f)) {
            const Matrix4x4 geometryTransform = splineObject->transform->getFinal();
            const Vec3 localPivot = geometryTransform.inverse().transform_point(pivot);
            splineObject->transform->setPivotOffset(localPivot, true);
            splineObject->has_insert_preview = false;
            ProjectManager::getInstance().markModified();
        }
        ImGui::TextDisabled("Object mode: G moves spline + pivot; P edits only the pivot.");
        if (ctx.scene_ui_ptr) {
            const bool editingPivot = ctx.scene_ui_ptr->pivot_edit_mode;
            if (ImGui::Button(editingPivot ? "Finish Pivot Edit (P)" : "Edit Pivot in Viewport (P)",
                              ImVec2(-1.0f, 0.0f))) {
                ctx.scene_ui_ptr->pivot_edit_mode = !editingPivot;
            }
        }
    }
    const char* curveTypeNames[] = {"Linear", "Bezier", "B-Spline"};
    int curveType = static_cast<int>(splineObject->spline.curveType);
    if (ImGui::Combo("Spline Type", &curveType, curveTypeNames, 3)) {
        splineObject->spline.curveType = static_cast<SplineCurveType>(curveType);
        splineObject->spline.knots.clear();
        changed = true;
    }
    if (splineObject->spline.curveType == SplineCurveType::BSpline &&
        splineObject->spline.points.size() < 4) {
        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.25f, 1.0f),
                           "B-Spline requires at least 4 control points.");
    }
    if (splineObject->edit_mode) {
        if (ImGui::Button("Disable Spline Edit (Tab)", ImVec2(-1.0f, 0.0f))) {
            splineObject->edit_mode = false;
            splineObject->selected_point = -1;
            splineObject->selected_points.clear();
        }
    } else {
        if (ImGui::Button("Edit Spline (Tab)", ImVec2(-1.0f, 0.0f))) {
            splineObject->edit_mode = true;
        }
        ImGui::TextDisabled("Point picking and point gizmos are disabled while editing is off.");
    }

    bool preserveTangents = false;
    changed |= ImGui::Checkbox("Closed Profile", &splineObject->spline.isClosed);
    changed |= ImGui::Checkbox("Edit Control Points", &splineObject->edit_controls);

    if (splineObject->edit_mode) {
        ImGui::Separator();
        ImGui::Text("Viewport Tool");
        const char* toolNames[] = {"Select", "Insert Point", "Subdivide", "Extrude"};
        int tool = static_cast<int>(splineObject->edit_tool);
        if (ImGui::Combo("Tool", &tool, toolNames, 4)) {
            splineObject->edit_tool = static_cast<SplineEditTool>(tool);
        }
        if (splineObject->edit_tool == SplineEditTool::InsertPoint) {
            ImGui::TextDisabled("Hover a segment, then left-click to insert.");
            if (splineObject->spline.curveType == SplineCurveType::BSpline)
                ImGui::TextDisabled("Cubic knot insertion preserves the B-Spline shape.");
        } else if (splineObject->edit_tool == SplineEditTool::Subdivide) {
            ImGui::SliderInt("Cuts", &splineObject->subdivide_cuts, 1, 32);
            ImGui::TextDisabled("Selected point starts the segment.");
        } else if (splineObject->edit_tool == SplineEditTool::Extrude) {
            ImGui::TextDisabled("Select an open endpoint, then click the button.");
        }
    }

    if (splineObject->edit_mode) {
    if (ImGui::Button("Add Control Point", ImVec2(-1.0f, 0.0f))) {
        Vec3 position(0.0f, 0.0f, 0.0f);
        if (!splineObject->spline.points.empty()) {
            Vec3 offset(0.5f, 0.5f, 0.0f);
            if (splineObject->plane == SplinePlane::XZ) offset = Vec3(0.5f, 0.0f, 0.5f);
            else if (splineObject->plane == SplinePlane::YZ) offset = Vec3(0.0f, 0.5f, 0.5f);
            position = splineObject->spline.points.back().position + offset;
        }
        splineObject->spline.knots.clear();
        splineObject->spline.points.emplace_back(position);
        splineObject->spline.calculateAutoTangents();
        splineObject->selected_point = static_cast<int>(splineObject->spline.points.size()) - 1;
        splineObject->selected_points = {splineObject->selected_point};
        changed = true;
    }

    ImGui::Separator();
    ImGui::TextDisabled("Control Points: %zu | Selected: %zu",
                        splineObject->spline.points.size(), splineObject->selected_points.size());
    ImGui::BeginChild("SplinePointList", ImVec2(0.0f, 180.0f), true,
                      ImGuiWindowFlags_HorizontalScrollbar);
    for (size_t i = 0; i < splineObject->spline.points.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        const bool selected = std::find(splineObject->selected_points.begin(),
                                        splineObject->selected_points.end(),
                                        static_cast<int>(i)) != splineObject->selected_points.end();
        if (ImGui::Selectable((std::string("Point ") + std::to_string(i)).c_str(), selected)) {
            const int pointIndex = static_cast<int>(i);
            if (ImGui::GetIO().KeyCtrl) {
                auto it = std::find(splineObject->selected_points.begin(),
                                    splineObject->selected_points.end(), pointIndex);
                if (it == splineObject->selected_points.end()) {
                    splineObject->selected_points.push_back(pointIndex);
                } else {
                    splineObject->selected_points.erase(it);
                }
            } else {
                splineObject->selected_points.clear();
                splineObject->selected_points.push_back(pointIndex);
            }
            splineObject->selected_point = pointIndex;
        }
        ImGui::PopID();
    }
    ImGui::EndChild();

    if (!splineObject->selected_points.empty()) {
        if (ImGui::Button("Subdivide Selected Segments", ImVec2(-1.0f, 0.0f))) {
            std::vector<int> segments = splineObject->selected_points;
            std::sort(segments.begin(), segments.end(), std::greater<int>());
            bool subdivided = false;
            int lastInserted = -1;
            for (const int segment : segments) {
                if (SplineEditService::subdivideSegment(
                        splineObject->spline, segment, splineObject->subdivide_cuts, &lastInserted)) {
                    propagateSplineSubdivideToKeys(
                        ctx.scene.timeline, splineObject->nodeName, segment,
                        splineObject->subdivide_cuts);
                    subdivided = true;
                }
            }
            if (subdivided) {
                splineObject->selected_point = lastInserted;
                splineObject->selected_points = {lastInserted};
                changed = true;
                preserveTangents = true;
            }
        }
    }
    if (splineObject->edit_mode && splineObject->selected_point >= 0 &&
        splineObject->selected_point < static_cast<int>(splineObject->spline.points.size())) {
        const int index = splineObject->selected_point;
        auto& point = splineObject->spline.points[static_cast<size_t>(index)];
        ImGui::Separator();
        ImGui::Text("Selected Point %d", index);
        changed |= drawPlaneVector("Position", point.position, splineObject->plane);
        float pointRadius = point.userData1;
        if (ImGui::DragFloat("Curve Radius", &pointRadius, 0.01f, 0.0f, 10000.0f, "%.3f")) {
            pointRadius = std::max(0.0f, pointRadius);
            for (const int selectedIndex : splineObject->selected_points) {
                if (selectedIndex >= 0 && selectedIndex < static_cast<int>(splineObject->spline.points.size()))
                    splineObject->spline.points[static_cast<size_t>(selectedIndex)].userData1 = pointRadius;
            }
            point.userData1 = pointRadius;
            changed = true;
        }
        ImGui::TextDisabled("Curve to Mesh: point radius multiplier (animatable source data).");
        if (splineObject->spline.curveType == SplineCurveType::Bezier) {
            const bool inChanged = drawPlaneVector(
                "Incoming Handle", point.tangentIn, splineObject->plane);
            const bool outChanged = drawPlaneVector(
                "Outgoing Handle", point.tangentOut, splineObject->plane);
            if (inChanged || outChanged) {
                point.autoTangent = false;
                changed = true;
            }
            ImGui::TextDisabled("Bezier handles are relative to the anchor.");
        } else if (splineObject->spline.curveType == SplineCurveType::BSpline) {
            ImGui::TextDisabled("B-Spline control point: the point position is the control value.");
        }
        if (ImGui::Button("Subdivide From Selected", ImVec2(-1.0f, 0.0f))) {
            int inserted = -1;
                if (SplineEditService::subdivideSegment(
                        splineObject->spline, index, splineObject->subdivide_cuts, &inserted)) {
                    propagateSplineSubdivideToKeys(
                        ctx.scene.timeline, splineObject->nodeName, index,
                        splineObject->subdivide_cuts);
                    splineObject->selected_point = inserted;
                    splineObject->selected_points = {inserted};
                preserveTangents = true;
                changed = true;
            }
        }
        const bool endpoint = index == 0 || index == static_cast<int>(splineObject->spline.points.size()) - 1;
        if (ImGui::Button("Extrude Selected Endpoint", ImVec2(-1.0f, 0.0f))) {
            if (endpoint && !splineObject->spline.isClosed) {
                const Vec3 direction = index == 0
                    ? point.position - splineObject->spline.points[1].position
                    : point.position - splineObject->spline.points[splineObject->spline.points.size() - 2].position;
                const BezierSpline beforeSpline = splineObject->spline;
                int inserted = -1;
                if (SplineEditService::extrudeEndpoint(
                        splineObject->spline, index, point.position + direction, &inserted)) {
                    propagateSplineExtrudeToKeys(
                        ctx.scene.timeline, splineObject->nodeName, beforeSpline,
                        index, beforeSpline.points[static_cast<size_t>(index)].position + direction);
                    splineObject->selected_point = inserted;
                    preserveTangents = true;
                    changed = true;
                }
            }
        }
        if (ImGui::Button("Delete Selected Point", ImVec2(-1.0f, 0.0f))) {
            splineObject->spline.removePoint(index);
            propagateSplineRemovePointToKeys(
                ctx.scene.timeline, splineObject->nodeName, index);
            splineObject->selected_point = -1;
            splineObject->selected_points.clear();
            changed = true;
        }
    }
    }
    if (changed && !preserveTangents) {
        splineObject->spline.calculateAutoTangents();
    }
    if (changed) {
        ProjectManager::getInstance().markModified();
    }
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("2  Skin & Deform", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto& skin = splineObject->skin_display;
        bool skinChanged = false;
        ImGui::TextDisabled("Non-destructive spline display. Convert only when a final mesh is needed.");
        const auto profiles = splineSources(ctx, true, splineObject.get());
        const char* profilePreview = skin.custom_profile.empty()
            ? "Circular" : skin.custom_profile.c_str();
        if (ImGui::BeginCombo("Profile", profilePreview)) {
            const bool circular = skin.custom_profile.empty();
            if (ImGui::Selectable("Circular", circular)) {
                skin.custom_profile.clear(); skinChanged = true;
            }
            if (circular) ImGui::SetItemDefaultFocus();
            for (const auto& profile : profiles) {
                const bool selected = skin.custom_profile == profile->nodeName;
                if (ImGui::Selectable(profile->nodeName.c_str(), selected)) {
                    skin.custom_profile = profile->nodeName; skinChanged = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        skinChanged |= ImGui::DragFloat(
            "Bevel Radius", &skin.radius, 0.01f, 0.0001f, 10000.0f, "%.4f");
        skinChanged |= ImGui::SliderInt("Path Resolution", &skin.path_samples, 2, 256);
        skinChanged |= ImGui::SliderInt(
            skin.custom_profile.empty() ? "Radial Segments" : "Profile Resolution",
            &skin.radial_segments, 3, 64);
        skinChanged |= ImGui::Checkbox("Cap Start", &skin.cap_start);
        ImGui::SameLine();
        skinChanged |= ImGui::Checkbox("Cap End", &skin.cap_end);
        skinChanged |= ImGui::Checkbox("Use Point Radius", &skin.use_point_radius);
        ImGui::TextDisabled("Final radius = Bevel Radius x animated Curve Radius per point.");
        if (ImGui::TreeNode("Taper")) {
            skinChanged |= ImGui::DragFloat(
                "Start Scale", &skin.taper_start, 0.01f, 0.0f, 1000.0f, "%.3f");
            skinChanged |= ImGui::DragFloat(
                "End Scale", &skin.taper_end, 0.01f, 0.0f, 1000.0f, "%.3f");
            skinChanged |= ImGui::DragFloat(
                "Taper Falloff", &skin.taper_falloff, 0.01f, 0.01f, 32.0f, "%.3f");
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Twist")) {
            skinChanged |= ImGui::DragFloat(
                "Twist Start", &skin.twist_start_degrees,
                1.0f, -100000.0f, 100000.0f, "%.1f deg");
            skinChanged |= ImGui::DragFloat(
                "Twist End", &skin.twist_end_degrees,
                1.0f, -100000.0f, 100000.0f, "%.1f deg");
            ImGui::TextDisabled("Most visible on a custom/non-circular profile or textured UVs.");
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Wave + Noise")) {
            const char* axes[] = {"X", "Y", "Z"};
            skinChanged |= ImGui::Combo("Wave Axis", &skin.wave_axis, axes, 3);
            skinChanged |= ImGui::DragFloat(
                "Wave Amplitude", &skin.wave_amplitude, 0.01f, -10000.0f, 10000.0f, "%.3f");
            skinChanged |= ImGui::DragFloat(
                "Wave Cycles", &skin.wave_cycles, 0.05f, -1000.0f, 1000.0f, "%.3f");
            skinChanged |= ImGui::DragFloat(
                "Wave Phase", &skin.wave_phase_degrees,
                1.0f, -100000.0f, 100000.0f, "%.1f deg");
            skinChanged |= ImGui::DragFloat(
                "Noise Amount", &skin.wave_noise, 0.01f, 0.0f, 10000.0f, "%.3f");
            skinChanged |= ImGui::DragInt("Noise Seed", &skin.wave_seed, 1.0f);
            ImGui::TreePop();
        }
        auto apiSettings = [&]() {
            rtapi::SplineSkinSettings settings;
            settings.custom_profile = skin.custom_profile;
            settings.radius = std::max(0.0001f, skin.radius);
            settings.path_samples = skin.path_samples;
            settings.radial_segments = skin.radial_segments;
            settings.cap_start = skin.cap_start; settings.cap_end = skin.cap_end;
            settings.use_point_radius = skin.use_point_radius;
            settings.taper_start = skin.taper_start; settings.taper_end = skin.taper_end;
            settings.taper_falloff = skin.taper_falloff;
            settings.twist_start_degrees = skin.twist_start_degrees;
            settings.twist_end_degrees = skin.twist_end_degrees;
            settings.wave_amplitude = skin.wave_amplitude;
            settings.wave_cycles = skin.wave_cycles;
            settings.wave_phase_degrees = skin.wave_phase_degrees;
            settings.wave_noise = skin.wave_noise; settings.wave_seed = skin.wave_seed;
            settings.wave_axis = skin.wave_axis;
            return settings;
        };
        auto updateDisplay = [&]() {
            rtapi::SplineSkinInfo info;
            const auto result = rtapi::createSplineSkinAdvanced(
                splineObject->nodeName, "", apiSettings(), info);
            splineObject->skin_display_status = result.ok
                ? "Live display: " + info.object_name + " (" +
                    std::to_string(info.vertex_count) + " vertices)."
                : result.error;
        };
        if (skin.enabled && skinChanged) updateDisplay();
        if (skinChanged) ProjectManager::getInstance().markModified();

        if (!skin.enabled) {
            const bool validPath = !splineObject->spline.isClosed;
            if (!validPath) ImGui::BeginDisabled();
            if (ImGui::Button("Enable Skin Display", ImVec2(-1.0f, 0.0f))) updateDisplay();
            if (!validPath) ImGui::EndDisabled();
            if (!validPath) ImGui::TextDisabled("Skin display requires an open spline path.");
        } else {
            ImGui::TextDisabled("Preview host: %s", skin.host_name.c_str());
            if (ImGui::Button("Convert to Mesh", ImVec2(-1.0f, 0.0f))) {
                rtapi::SplineSkinInfo info;
                const auto result = rtapi::finalizeSplineSkinPreview(
                    splineObject->nodeName, info);
                splineObject->skin_display_status = result.ok
                    ? "Converted to mesh: " + info.object_name : result.error;
            }
            if (ImGui::Button("Remove Skin Display", ImVec2(-1.0f, 0.0f))) {
                const auto result = rtapi::clearSplineSkinPreview(splineObject->nodeName);
                if (!result.ok) splineObject->skin_display_status = result.error;
            }
        }
        if (!splineObject->skin_display_status.empty())
            ImGui::TextWrapped("%s", splineObject->skin_display_status.c_str());
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("3  Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
        const int frame = ctx.scene.timeline.current_frame;
        ImGui::Text("Current Frame: %d", frame);
        auto insertKey = [&](bool objectTransform, bool points, const char* label) {
            std::string error;
            if (insertSplineAnimationKey(ctx.scene.timeline, *splineObject, frame,
                                         objectTransform, points, &error)) {
                splineObject->animation_status = std::string(label) + " keyed at frame " +
                    std::to_string(frame) + ".";
                ProjectManager::getInstance().markModified();
                ctx.start_render = true;
            } else {
                splineObject->animation_status = error;
            }
        };
        if (ImGui::Button("Key Object + All Points", ImVec2(-1.0f, 0.0f)))
            insertKey(true, true, "Spline object and controls");
        if (ImGui::Button("Key Object Transform", ImVec2(-1.0f, 0.0f)))
            insertKey(true, false, "Spline object transform");
        if (ImGui::Button("Key All Points / Handles / Radius", ImVec2(-1.0f, 0.0f)))
            insertKey(false, true, "Spline controls");
        if (ImGui::Button("Remove Spline Key at Frame", ImVec2(-1.0f, 0.0f))) {
            std::string error;
            if (removeSplineAnimationKey(ctx.scene.timeline, splineObject->nodeName,
                                         frame, true, true, &error)) {
                splineObject->animation_status = "Spline key removed at frame " +
                    std::to_string(frame) + ".";
                ProjectManager::getInstance().markModified();
                ctx.start_render = true;
            } else splineObject->animation_status = error;
        }
        const auto keys = listSplineAnimationKeys(ctx.scene.timeline, splineObject->nodeName);
        ImGui::TextDisabled("Keys: %d | topology must stay constant between point keys.",
                            static_cast<int>(keys.size()));
        if (!splineObject->animation_status.empty())
            ImGui::TextWrapped("%s", splineObject->animation_status.c_str());
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Advanced Surface Tools")) {
        ImGui::TextDisabled("Sweep, Revolve and Loft. Open only when direct mesh generation is needed.");
        static int operation = 0;
        static int pathSamples = 32;
        static int profileSamples = 24;
        static int angleSegments = 32;
        static int revolveAxis = 1;
        static float startAngleDegrees = 0.0f;
        static float endAngleDegrees = 360.0f;
        static float revolveRadiusOffset = 0.0f;
        static float revolvePivotOffset[3] = {0.0f, 0.0f, 0.0f};
        static int loftSamples = 24;
        static float profileScale = 1.0f;
        static bool capStart = true;
        static bool capEnd = true;
        static int counterpartIndex = 0;
        static char outputName[128] = "";

        if (splineObject->profile_preview_operation == "profile.sweep") operation = 0;
        else if (splineObject->profile_preview_operation == "profile.revolve") operation = 1;
        else if (splineObject->profile_preview_operation == "profile.loft") operation = 2;

        const char* operations[] = {
            "Sweep (Closed Profile + Open Spine)",
            "Revolve / Screw",
            "Loft (Closed Sections)"};
        if (ImGui::Combo("Operation", &operation, operations, IM_ARRAYSIZE(operations))) {
            counterpartIndex = 0;
            clearLinkedPreview(ctx, splineObject);
        }
        ImGui::InputText("Output Name", outputName, IM_ARRAYSIZE(outputName));

        if (operation == 0) {
            const bool selectedIsProfile = splineObject->spline.isClosed;
            const auto counterparts = splineSources(ctx, !selectedIsProfile, splineObject.get());
            ImGui::Text("Selected role: %s", selectedIsProfile ? "Closed Profile" : "Open Path");
            if (counterparts.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.25f, 1.0f),
                    "Add an %s spline to use as the %s.",
                    selectedIsProfile ? "open" : "closed",
                    selectedIsProfile ? "path" : "profile");
            } else {
                for (int i = 0; i < static_cast<int>(counterparts.size()); ++i) {
                    if (counterparts[static_cast<size_t>(i)]->nodeName ==
                        splineObject->profile_preview_counterpart) counterpartIndex = i;
                }
                counterpartIndex = std::clamp(counterpartIndex, 0,
                    static_cast<int>(counterparts.size()) - 1);
                if (ImGui::BeginCombo(selectedIsProfile ? "Path" : "Profile",
                                      counterparts[static_cast<size_t>(counterpartIndex)]->nodeName.c_str())) {
                    for (int i = 0; i < static_cast<int>(counterparts.size()); ++i) {
                        const bool selected = i == counterpartIndex;
                        if (ImGui::Selectable(counterparts[static_cast<size_t>(i)]->nodeName.c_str(), selected))
                            counterpartIndex = i;
                        if (selected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            }
            if (!counterparts.empty()) {
                const auto counterpart = counterparts[static_cast<size_t>(counterpartIndex)];
                const auto profileObject = selectedIsProfile ? splineObject : counterpart;
                drawProfilePlaneScale(*profileObject, "Profile Object Scale");
            }
            ImGui::SliderInt("Path Samples", &pathSamples, 2, 256);
            ImGui::SliderInt("Profile Samples", &profileSamples, 3, 256);
            ImGui::DragFloat("Profile Scale", &profileScale, 0.05f, 0.001f, 1000.0f);
            ImGui::Checkbox("Cap Start", &capStart); ImGui::SameLine();
            ImGui::Checkbox("Cap End", &capEnd);
            const bool previewClicked = !counterparts.empty() &&
                ImGui::Button(splineObject->profile_preview_operation == "profile.sweep"
                    ? "Refresh Sweep Preview" : "Start Sweep Preview", ImVec2(-1.0f, 0.0f));
            if (!counterparts.empty()) {
                const auto counterpart = counterparts[static_cast<size_t>(counterpartIndex)];
                const auto profileObject = selectedIsProfile ? splineObject : counterpart;
                const auto pathObject = selectedIsProfile ? counterpart : splineObject;
                size_t signature = sourceSignature(*profileObject);
                hashCombine(signature, sourceSignature(*pathObject));
                hashCombine(signature, static_cast<size_t>(pathSamples));
                hashCombine(signature, static_cast<size_t>(profileSamples));
                hashFloat(signature, profileScale);
                hashCombine(signature, capStart ? 1u : 0u);
                hashCombine(signature, capEnd ? 1u : 0u);
                const bool refresh = previewClicked ||
                    (splineObject->profile_preview_operation == "profile.sweep" &&
                     splineObject->profile_preview_signature != signature);
                if (refresh) {
                if (profileObject->plane == SplinePlane::Free) {
                    ProfileSweepResult result;
                    result.report.operation_id = "profile.sweep";
                    result.report.addError("non_planar_profile",
                        "Sweep profile must be an XY/XZ/YZ planar spline; '" +
                        profileObject->nodeName + "' uses the unconstrained Free plane.");
                    setPreview(splineObject, counterpart, result, "profile.sweep",
                               Matrix4x4::identity(), signature);
                } else {
                const BezierSpline profile = selectedIsProfile
                    ? canonicalProfile(*splineObject, true) : canonicalProfile(*counterpart, true);
                const BezierSpline path = worldSpline(*pathObject);
                ProfileSweepSettings settings;
                settings.path_samples = pathSamples;
                settings.profile_samples = profileSamples;
                settings.profile_scale = profileScale;
                settings.cap_start = capStart;
                settings.cap_end = capEnd;
                const auto result = buildProfileSweep(profile, path, settings);
                    setPreview(splineObject, counterpart, result, "profile.sweep",
                               Matrix4x4::identity(), signature);
                }
                }
            }
        } else if (operation == 1) {
            ImGui::Text("Source: %s", splineObject->nodeName.c_str());
            ImGui::TextDisabled("Entire spline: %d controls -> %d evaluated profile samples.",
                static_cast<int>(splineObject->spline.points.size()), profileSamples);
            const char* profileMapping = splineObject->plane == SplinePlane::XY
                ? "radius=X, height=Y" : splineObject->plane == SplinePlane::XZ
                ? "radius=X, height=Z" : "radius=Z, height=Y";
            ImGui::TextDisabled("Profile mapping: %s.", profileMapping);
            ImGui::TextDisabled("Winding is normalized outward independently of point order.");
            drawProfilePlaneScale(*splineObject, "Profile Object Scale");
            const char* axes[] = {"X", "Y", "Z"};
            ImGui::Combo("Axis", &revolveAxis, axes, IM_ARRAYSIZE(axes));
            ImGui::DragFloat("Start Angle", &startAngleDegrees, 1.0f, -360.0f, 360.0f, "%.1f deg");
            ImGui::DragFloat("End Angle", &endAngleDegrees, 1.0f, -360.0f, 360.0f, "%.1f deg");
            ImGui::DragFloat("Axis Radius Offset", &revolveRadiusOffset, 0.05f, 0.0f, 10000.0f, "%.3f");
            ImGui::DragFloat3("Axis Pivot Offset", revolvePivotOffset, 0.05f, -10000.0f, 10000.0f, "%.3f");
            const Vec3 objectPivot = splineObject->transform
                ? splineObject->transform->getPivotMatrix().getTranslation() : Vec3(0.0f);
            const Vec3 axisPivot = objectPivot + Vec3(
                revolvePivotOffset[0], revolvePivotOffset[1], revolvePivotOffset[2]);
            ImGui::TextDisabled("Spline pivot is the default axis origin.");
            ImGui::TextDisabled("World %s axis through (%.2f, %.2f, %.2f)",
                axes[revolveAxis], axisPivot.x, axisPivot.y, axisPivot.z);
            ImGui::SliderInt("Angle Segments", &angleSegments, 3, 256);
            ImGui::SliderInt("Profile Samples", &profileSamples, 3, 256);
            const bool previewClicked = ImGui::Button(
                splineObject->profile_preview_operation == "profile.revolve"
                    ? "Refresh Revolve Preview" : "Start Revolve Preview", ImVec2(-1.0f, 0.0f));
            size_t signature = sourceSignature(*splineObject);
            hashCombine(signature, static_cast<size_t>(angleSegments));
            hashCombine(signature, static_cast<size_t>(profileSamples));
            hashCombine(signature, static_cast<size_t>(revolveAxis));
            hashFloat(signature, startAngleDegrees);
            hashFloat(signature, endAngleDegrees);
            hashFloat(signature, revolveRadiusOffset);
            hashFloat(signature, revolvePivotOffset[0]);
            hashFloat(signature, revolvePivotOffset[1]);
            hashFloat(signature, revolvePivotOffset[2]);
            const bool refresh = previewClicked ||
                (splineObject->profile_preview_operation == "profile.revolve" &&
                 splineObject->profile_preview_signature != signature);
            if (refresh) {
                ProfileRevolveResult result;
                if (splineObject->plane == SplinePlane::Free) {
                    result.report.operation_id = "profile.revolve";
                    result.report.addError("non_planar_profile",
                        "Revolve profile must be an XY/XZ/YZ planar spline; '" +
                        splineObject->nodeName + "' uses the unconstrained Free plane.");
                } else {
                ProfileRevolveSettings settings;
                settings.angle_segments = angleSegments;
                settings.profile_samples = profileSamples;
                settings.axis = static_cast<ProfileRevolveAxis>(revolveAxis);
                settings.start_angle = startAngleDegrees * M_PI / 180.0f;
                settings.end_angle = endAngleDegrees * M_PI / 180.0f;
                settings.radius_offset = revolveRadiusOffset;
                settings.axis_pivot = Vec3(
                    revolvePivotOffset[0], revolvePivotOffset[1], revolvePivotOffset[2]);
                const BezierSpline profile = canonicalProfile(*splineObject, true, true);
                result = buildProfileRevolve(profile, settings);
                }
                setPreview(splineObject, {}, result, "profile.revolve",
                           Matrix4x4::translation(objectPivot), signature);
            }
        } else {
            const auto counterparts = splineSources(ctx, true, splineObject.get());
            if (!splineObject->spline.isClosed) {
                ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.25f, 1.0f),
                    "This open spline is a spine. Use Sweep with one closed profile;\n"
                    "Loft connects two or more closed profile sections.");
                if (ImGui::Button("Use This as Sweep Spine", ImVec2(-1.0f, 0.0f))) {
                    operation = 0;
                    clearLinkedPreview(ctx, splineObject);
                }
            } else if (counterparts.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.25f, 1.0f),
                                   "Add another closed spline section for Loft.");
            } else {
                for (int i = 0; i < static_cast<int>(counterparts.size()); ++i) {
                    if (counterparts[static_cast<size_t>(i)]->nodeName ==
                        splineObject->profile_preview_counterpart) counterpartIndex = i;
                }
                counterpartIndex = std::clamp(counterpartIndex, 0,
                    static_cast<int>(counterparts.size()) - 1);
                if (ImGui::BeginCombo("Second Section",
                                      counterparts[static_cast<size_t>(counterpartIndex)]->nodeName.c_str())) {
                    for (int i = 0; i < static_cast<int>(counterparts.size()); ++i) {
                        const bool selected = i == counterpartIndex;
                        if (ImGui::Selectable(counterparts[static_cast<size_t>(i)]->nodeName.c_str(), selected))
                            counterpartIndex = i;
                        if (selected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            }
            ImGui::SliderInt("Section Samples", &loftSamples, 3, 256);
            ImGui::Checkbox("Cap Start", &capStart); ImGui::SameLine();
            ImGui::Checkbox("Cap End", &capEnd);
            const bool previewClicked = splineObject->spline.isClosed && !counterparts.empty() &&
                ImGui::Button(splineObject->profile_preview_operation == "profile.loft"
                    ? "Refresh Loft Preview" : "Start Loft Preview", ImVec2(-1.0f, 0.0f));
            if (splineObject->spline.isClosed && !counterparts.empty()) {
                const auto counterpart = counterparts[static_cast<size_t>(counterpartIndex)];
                size_t signature = sourceSignature(*splineObject);
                hashCombine(signature, sourceSignature(*counterpart));
                hashCombine(signature, static_cast<size_t>(loftSamples));
                hashCombine(signature, capStart ? 1u : 0u);
                hashCombine(signature, capEnd ? 1u : 0u);
                const bool refresh = previewClicked ||
                    (splineObject->profile_preview_operation == "profile.loft" &&
                     splineObject->profile_preview_signature != signature);
                if (refresh) {
                const BezierSpline first = worldSpline(*splineObject);
                const BezierSpline second = worldSpline(*counterpart);
                const std::vector<const BezierSpline*> sections{&first, &second};
                ProfileLoftSettings settings;
                settings.samples_per_section = loftSamples;
                settings.cap_start = capStart;
                settings.cap_end = capEnd;
                const auto result = buildProfileLoft(sections, settings);
                    setPreview(splineObject, counterpart, result, "profile.loft",
                               Matrix4x4::identity(), signature);
                }
            }
        }
        if (!splineObject->profile_preview_status.empty()) {
            const bool validPreview = static_cast<bool>(splineObject->profile_preview_geometry);
            ImGui::SeparatorText(validPreview ? "Preview Ready" : "Preview Error");
            ImGui::PushStyleColor(ImGuiCol_Text, validPreview
                ? ImVec4(0.35f, 1.0f, 0.65f, 1.0f)
                : ImVec4(1.0f, 0.35f, 0.25f, 1.0f));
            ImGui::TextWrapped("%s", splineObject->profile_preview_status.c_str());
            ImGui::PopStyleColor();
        }
        if (splineObject->profile_preview_geometry) {
            if (ImGui::Button("Apply as Mesh", ImVec2(-1.0f, 0.0f))) {
                if (!ctx.scene_ui_ptr) {
                    splineObject->profile_preview_status =
                        "scene_history_unavailable: Preview cannot be applied without SceneHistory.";
                } else {
                    const auto geometry = splineObject->profile_preview_geometry;
                    const std::string previewOperation = splineObject->profile_preview_operation;
                    const Matrix4x4 previewTransform = splineObject->profile_preview_transform;
                    const ProfilePublishResult published = publishGeneratedProfile(
                        ctx, ctx.scene_ui_ptr->history, geometry, outputName,
                        previewOperation, &previewTransform);
                    if (published.report.ok) {
                        clearLinkedPreview(ctx, splineObject);
                        return;
                    }
                    splineObject->profile_preview_status = !published.report.diagnostics.empty()
                        ? published.report.diagnostics.front().code + ": " +
                          published.report.diagnostics.front().message
                        : "publish_failed: Preview mesh could not be applied.";
                }
            }
        }
        if (!splineObject->profile_preview_operation.empty() &&
            ImGui::Button("Cancel Preview", ImVec2(-1.0f, 0.0f))) {
            clearLinkedPreview(ctx, splineObject);
        }
    }
}

} // namespace MeshEdit
