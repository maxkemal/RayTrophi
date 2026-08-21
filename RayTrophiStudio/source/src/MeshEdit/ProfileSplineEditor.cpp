#include "MeshEdit/ProfileSplineEditor.h"

#include "MeshEdit/SplineObject.h"
#include "MeshEdit/SplineEditService.h"
#include "SceneSelection.h"
#include "ProjectManager.h"
#include "scene_ui.h"
#include "imgui.h"

#include <string>
#include <algorithm>
#include <functional>

namespace MeshEdit {

void drawProfileSplineEditControls(UIContext& ctx) {
    auto splineObject = ctx.selection.selected.spline_object;
    if (!splineObject) return;

    ImGui::TextColored(ImVec4(0.35f, 1.0f, 0.85f, 1.0f), "2D Spline Source");
    ImGui::TextDisabled("Mesh-free authoring source; evaluated by modifiers/Geometry Nodes.");
    ImGui::Separator();
    ImGui::Text("Object: %s", splineObject->nodeName.c_str());
    ImGui::Text("Plane: XZ / Y Up");
    bool changed = false;
    const char* curveTypeNames[] = {"Linear", "Bezier", "B-Spline"};
    int curveType = static_cast<int>(splineObject->spline.curveType);
    if (ImGui::Combo("Spline Type", &curveType, curveTypeNames, 3)) {
        splineObject->spline.curveType = static_cast<SplineCurveType>(curveType);
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
            if (splineObject->spline.curveType == SplineCurveType::BSpline) {
                ImGui::TextDisabled("B-Spline insert uses knot insertion in the next phase.");
            } else {
                ImGui::TextDisabled("Hover a segment, then left-click to insert.");
            }
        } else if (splineObject->edit_tool == SplineEditTool::Subdivide) {
            ImGui::SliderInt("Cuts", &splineObject->subdivide_cuts, 1, 32);
            ImGui::TextDisabled("Selected point starts the segment.");
        } else if (splineObject->edit_tool == SplineEditTool::Extrude) {
            ImGui::TextDisabled("Select an open endpoint, then click the button.");
        }
    }

    if (ImGui::Button("Add Control Point", ImVec2(-1.0f, 0.0f))) {
        Vec3 position(0.0f, 0.0f, 0.0f);
        if (!splineObject->spline.points.empty()) {
            position = splineObject->spline.points.back().position + Vec3(0.5f, 0.0f, 0.5f);
        }
        splineObject->spline.points.emplace_back(position);
        splineObject->spline.calculateAutoTangents();
        splineObject->selected_point = static_cast<int>(splineObject->spline.points.size()) - 1;
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
                if (SplineEditService::subdivideBezierSegment(
                        splineObject->spline, segment, splineObject->subdivide_cuts, &lastInserted)) {
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
        changed |= ImGui::DragFloat3("Position (X/Y/Z)", &point.position.x, 0.05f);
        if (splineObject->spline.curveType == SplineCurveType::Bezier) {
            const bool inChanged = ImGui::DragFloat3("Incoming Handle", &point.tangentIn.x, 0.05f);
            const bool outChanged = ImGui::DragFloat3("Outgoing Handle", &point.tangentOut.x, 0.05f);
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
            if (SplineEditService::subdivideBezierSegment(
                    splineObject->spline, index, splineObject->subdivide_cuts, &inserted)) {
                splineObject->selected_point = inserted;
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
                int inserted = -1;
                if (SplineEditService::extrudeEndpoint(
                        splineObject->spline, index, point.position + direction, &inserted)) {
                    splineObject->selected_point = inserted;
                    preserveTangents = true;
                    changed = true;
                }
            }
        }
        if (ImGui::Button("Delete Selected Point", ImVec2(-1.0f, 0.0f))) {
            splineObject->spline.removePoint(index);
            splineObject->selected_point = -1;
            changed = true;
        }
    }
    if (changed && !preserveTangents) {
        splineObject->spline.calculateAutoTangents();
    }
    if (changed) {
        ProjectManager::getInstance().markModified();
    }
}

} // namespace MeshEdit
