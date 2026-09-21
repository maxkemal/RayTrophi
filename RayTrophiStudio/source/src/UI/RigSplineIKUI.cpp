#include "UI/RigIKUI.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "imgui.h"
#include <cmath>

namespace RigUI {
void drawRigSplineControls(UIContext& ctx, const std::string& character, const std::string& control,
                           uint64_t revision, const nlohmann::json& row) {
    const auto solver = row["solver"].get<std::string>();
    if ((solver != "fabrik" && solver != "spline_fabrik") || row["bones"].size() < 4 ||
        !ImGui::TreeNode("Spline shape")) {
        return;
    }
    static std::string error;
    auto report = [&](const rtapi::Result& result) { error = result.ok ? "" : result.error; };
    std::vector<Vec3> points;
    for (const auto& point : row["spline_world"]) {
        points.emplace_back(point[0].get<float>(), point[1].get<float>(), point[2].get<float>());
    }
    bool enabled = row["spline_enabled"].get<bool>();
    if (ImGui::Checkbox("Use spline shape", &enabled)) {
        const auto result = rtapi::setRigIKSpline(character, control, points, enabled, revision);
        report(result);
        if (result.ok) {
            report(rtapi::applyRigPosePreview(character));
        }
    }
    const auto& handle = ctx.scene.rigView.pose.controlHandle;
    ImGui::BeginDisabled(ctx.scene.rigView.pose.hasPreview);
    if (ImGui::RadioButton("Shape point 1", handle == "spline_0")) {
        report(rtapi::selectRigControl(character, control, "spline_0"));
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Shape point 2", handle == "spline_1")) {
        report(rtapi::selectRigControl(character, control, "spline_1"));
    }
    ImGui::EndDisabled();
    bool changed = false, finished = false;
    for (size_t i = 0; i < points.size(); ++i) {
        float value[3] = {points[i].x, points[i].y, points[i].z};
        const char* label = i == 0 ? "Shape 1 world" : "Shape 2 world";
        changed = ImGui::InputFloat3(label, value) || changed;
        finished = ImGui::IsItemDeactivatedAfterEdit() || finished;
        points[i] = Vec3(value[0], value[1], value[2]);
    }
    if (changed) {
        report(rtapi::setRigIKSpline(character, control, points, true, revision));
    }
    if (finished && ctx.scene.rigView.pose.hasPreview) {
        report(rtapi::applyRigPosePreview(character));
    }
    ImGui::TextWrapped("Two shape points guide the curve between the fixed root and target. "
                       "Bone lengths and joint limits can alter the achieved shape. "
                       "To bend a fully extended straight chain, move its target closer first.");
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    ImGui::TreePop();
}

void drawRigSplineGuide(const nlohmann::json& row, const float* view, const float* projection,
                        const std::string& handle) {
    if (!row["spline_enabled"].get<bool>() && handle != "spline_0" && handle != "spline_1") {
        return;
    }
    const auto& io = ImGui::GetIO();
    auto project = [&](const nlohmann::json& value, ImVec2& result) {
        float point[4] = {value[0].get<float>(), value[1].get<float>(), value[2].get<float>(), 1};
        float camera[4] = {}, clip[4] = {};
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                camera[row] += view[col * 4 + row] * point[col];
            }
        }
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                clip[row] += projection[col * 4 + row] * camera[col];
            }
        }
        if (!std::isfinite(clip[3]) || clip[3] <= 1e-6f || clip[2] < -clip[3]) {
            return false;
        }
        result = ImVec2((clip[0] / clip[3] * .5f + .5f) * io.DisplaySize.x,
                        (.5f - clip[1] / clip[3] * .5f) * io.DisplaySize.y);
        return std::isfinite(result.x) && std::isfinite(result.y);
    };
    auto* draw = ImGui::GetBackgroundDrawList();
    const auto& curve = row["spline_guide_world"];
    for (size_t i = 1; i < curve.size(); ++i) {
        ImVec2 a, b;
        if (project(curve[i - 1], a) && project(curve[i], b)) {
            draw->AddLine(a, b, IM_COL32(115, 220, 160, 210), 1.5f);
        }
    }
    const auto& points = row["spline_world"];
    for (size_t i = 0; i < points.size(); ++i) {
        ImVec2 point;
        if (project(points[i], point)) {
            draw->AddCircleFilled(point, 5.f, IM_COL32(255, 190, 65, 240));
            draw->AddText(ImVec2(point.x + 8, point.y - 6), IM_COL32(255, 190, 65, 240),
                          i == 0 ? "S1" : "S2");
        }
    }
}
}
