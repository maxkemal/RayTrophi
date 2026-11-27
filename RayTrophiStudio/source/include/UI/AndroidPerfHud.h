#pragma once

#ifdef __ANDROID__

#include "imgui.h"
#include "json.hpp"

#include <SDL.h>

#include <algorithm>
#include <cmath>

namespace AppRuntime {

inline float androidPerfHudFontScale() {
    static float cached_scale = 0.0f;
    if (cached_scale <= 0.0f) {
        float dpi = 0.0f;
        if (SDL_GetDisplayDPI(0, &dpi, nullptr, nullptr) != 0 || dpi < 80.0f) {
            dpi = 220.0f;
        }
        const float density = dpi / 160.0f;
        cached_scale = std::clamp(density * 0.85f, 1.1f, 1.65f);
    }
    return cached_scale;
}

inline void drawAndroidPerfHud(const nlohmann::json& frame_stats, int render_w, int render_h) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImVec2 origin = viewport ? viewport->WorkPos : ImVec2(0.0f, 0.0f);
    ImGui::SetNextWindowPos(ImVec2(origin.x + 18.0f, origin.y + 18.0f), ImGuiCond_Always);

    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 11.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.06f, 0.08f, 0.12f, 0.48f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 1.0f, 1.0f, 0.22f));

    if (ImGui::Begin("##android_perf_hud", nullptr, flags)) {
        ImGui::SetWindowFontScale(androidPerfHudFontScale());

        const bool has_stats = !frame_stats.empty();
        double fps = 0.0;
        double frame_ms = 0.0;
        if (has_stats) {
            const double rendered_fps = frame_stats.value("rendered_fps", 0.0);
            const double presented_fps = frame_stats.value("presented_fps", 0.0);
            fps = rendered_fps > 0.05 ? rendered_fps : presented_fps;
            frame_ms = frame_stats.value("backend_frame_ms_avg", 0.0);
            if (frame_ms <= 0.05) {
                frame_ms = frame_stats.value("frame_cost_ms", 0.0);
            }
        }

        const int fps_rounded = static_cast<int>(std::lround(std::max(0.0, fps)));
        ImVec4 fps_color(0.78f, 0.83f, 0.90f, 0.92f);
        if (has_stats) {
            if (fps_rounded >= 28) {
                fps_color = ImVec4(0.36f, 0.90f, 0.56f, 0.98f);
            } else if (fps_rounded >= 18) {
                fps_color = ImVec4(0.98f, 0.76f, 0.28f, 0.98f);
            } else {
                fps_color = ImVec4(0.96f, 0.42f, 0.42f, 0.98f);
            }
        }

        const float line_h = ImGui::GetTextLineHeight();
        const float dot_r = std::max(3.0f, line_h * 0.21f);
        const ImVec2 dot_pos = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(dot_r * 2.0f, line_h));
        ImGui::GetWindowDrawList()->AddCircleFilled(
            ImVec2(dot_pos.x + dot_r, dot_pos.y + line_h * 0.5f),
            dot_r,
            ImGui::ColorConvertFloat4ToU32(fps_color),
            16);

        ImGui::SameLine(0.0f, 7.0f);
        if (has_stats) {
            ImGui::TextColored(fps_color, "%d FPS", fps_rounded);
        } else {
            ImGui::TextColored(fps_color, "-- FPS");
        }

        const ImVec4 sep_color(1.0f, 1.0f, 1.0f, 0.28f);
        const ImVec4 res_color(0.90f, 0.93f, 0.97f, 0.92f);
        const ImVec4 ms_color(0.74f, 0.80f, 0.88f, 0.84f);

        if (render_w > 0 && render_h > 0) {
            ImGui::SameLine();
            ImGui::TextColored(sep_color, "|");
            ImGui::SameLine();
            ImGui::TextColored(res_color, "%d\xC3\x97%d", render_w, render_h);
        }

        if (has_stats && frame_ms > 0.05) {
            ImGui::SameLine();
            ImGui::TextColored(sep_color, "|");
            ImGui::SameLine();
            ImGui::TextColored(ms_color, "%.1f ms", frame_ms);
        }

        ImGui::SetWindowFontScale(1.0f);
    }
    ImGui::End();

    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(4);
}

} // namespace AppRuntime

#endif
