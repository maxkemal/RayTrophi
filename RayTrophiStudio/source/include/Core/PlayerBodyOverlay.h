#pragma once

#include "Api/RtApi.h"
#include "Input/BodyPoseRuntime.h"
#include "imgui.h"

#include <array>
#include <cmath>
#include <cstddef>

namespace AppRuntime {

struct PlayerOverlayProjection {
    PlayerOverlayProjection(int render_width, int render_height) {
        rtapi::CameraState camera;
        if (render_width <= 0 || render_height <= 0 || !rtapi::getCamera(camera).ok) {
            return;
        }
        viewport = ImGui::GetMainViewport();
        origin = camera.position;
        forward = (camera.target - origin).normalize();
        right = Vec3::cross(forward, camera.up).normalize();
        up = Vec3::cross(right, forward);
        half_height = std::tan(camera.fov * 0.00872664626f);
        half_width = half_height * render_width / render_height;
    }

    bool project(const Vec3& world, ImVec2& point) const {
        if (!(half_width > 0.0f) || !(half_height > 0.0f)) {
            return false;
        }
        const Vec3 relative = world - origin;
        const float z = Vec3::dot(relative, forward);
        if (!(z > 0.01f)) {
            return false;
        }
        const float x = Vec3::dot(relative, right) / (z * half_width);
        const float y = Vec3::dot(relative, up) / (z * half_height);
        point = ImVec2(viewport->Pos.x + (1.0f + x) * viewport->Size.x * 0.5f,
                       viewport->Pos.y + (1.0f - y) * viewport->Size.y * 0.5f);
        return std::isfinite(point.x) && std::isfinite(point.y);
    }

    const ImGuiViewport* viewport = nullptr;
    Vec3 origin, forward, right, up;
    float half_width = 0.0f;
    float half_height = 0.0f;
};

inline void drawPlayerBodyOverlay(const PlayerOverlayProjection& projection) {
    auto& input = Input::bodyPoses();
    input.expire();
    auto* draw = ImGui::GetBackgroundDrawList();
    // Indices into Input::kBodyJoints.
    constexpr std::array<std::array<int, 2>, 14> bones{{
        {0, 1}, {1, 2}, {1, 3}, {3, 4}, {4, 5}, {2, 6}, {6, 7}, {7, 8},
        {1, 9}, {9, 10}, {10, 11}, {2, 12}, {12, 13}, {13, 14},
    }};
    constexpr ImU32 color = IM_COL32(104, 232, 255, 255);
    for (const auto& entry : input.persons()) {
        const auto& person = entry.second;
        if (person.confidence < 0.5) {
            continue;
        }
        std::array<ImVec2, Input::kBodyJoints.size()> points{};
        std::array<bool, Input::kBodyJoints.size()> visible{};
        for (std::size_t i = 0; i < points.size(); ++i) {
            const auto& joint = person.joints.at(Input::kBodyJoints[i]);
            const Vec3 world(static_cast<float>(joint[0]), static_cast<float>(joint[1]),
                             static_cast<float>(joint[2]));
            visible[i] = joint[3] >= 0.5 && projection.project(world, points[i]);
        }
        for (const auto& bone : bones) {
            if (visible[bone[0]] && visible[bone[1]]) {
                draw->AddLine(points[bone[0]], points[bone[1]], color, 2.5f);
            }
        }
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (visible[i]) {
                draw->AddCircleFilled(points[i], 3.5f, color);
            }
        }
    }
}

} // namespace AppRuntime
