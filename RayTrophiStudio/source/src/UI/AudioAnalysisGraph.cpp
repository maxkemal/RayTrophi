#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "UI/AudioAnalysisGraph.h"
#include "UI/AudioWorkspaceUI.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace AudioUI {
namespace {
using json = nlohmann::json;

struct Axis {
    double zoom = 1;
    double pan = 0;

    double origin() const {
        return pan * (1 - 1 / zoom);
    }

    double project(double value) const {
        return (value - origin()) * zoom;
    }

    void move(double delta) {
        const double extent = 1 - 1 / zoom;
        pan = extent > 1e-9 ? std::clamp((origin() + delta) / extent, 0.0, 1.0) : 0;
    }

    void scale(double factor, double anchor) {
        const double value = origin() + anchor / zoom;
        zoom = std::clamp(zoom * factor, 1.0, 32.0);
        const double extent = 1 - 1 / zoom;
        pan = extent > 1e-9 ?
            std::clamp((value - anchor / zoom) / extent, 0.0, 1.0) : 0;
    }
};

ImU32 heat(double db) {
    // Fixed dBFS colors: silence stays dark, overload is at the bright end.
    const float t = static_cast<float>(std::clamp((db + 96) / 96, 0.0, 1.0));
    ImVec4 color;
    ImGui::ColorConvertHSVtoRGB(0.67f * (1 - t), 0.85f, 0.12f + 0.88f * t,
        color.x, color.y, color.z);
    color.w = 1;
    return ImGui::ColorConvertFloat4ToU32(color);
}
} // namespace

void drawLiveAnalysisGraph(UIContext& ctx, const json& report, const json& workspace) {
    const auto mode = workspace.value("analysis_view", std::string("line"));
    const bool history = mode == "spectrogram";
    const bool bands = mode == "bands";
    const auto& points = report.at("spectrum");
    if (points.size() < 2) {
        return;
    }
    Axis horizontal{workspace.value("analysis_zoom_x", 1.0),
        workspace.value("analysis_pan_x", 0.0)};
    Axis vertical{workspace.value("analysis_zoom_y", 1.0),
        workspace.value("analysis_pan_y", 0.0)};
    const float font = ImGui::GetFontSize();
    const float width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
    const float height = std::clamp(ImGui::GetContentRegionAvail().y * 0.52f,
        font * 12, font * 23);
    ImGui::InvisibleButton("LiveAnalysisGraph", ImVec2(width, height),
        ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    const auto begin = ImGui::GetItemRectMin();
    const auto end = ImGui::GetItemRectMax();
    const ImVec2 a(begin.x + std::min(font * 3.7f, width * 0.3f), begin.y + 10);
    const ImVec2 b(std::max(a.x + 1, end.x - 10), end.y - font * 1.8f);
    const float plotWidth = b.x - a.x;
    const float plotHeight = b.y - a.y;
    const auto& io = ImGui::GetIO();
    const bool overPlot = hovered && io.MousePos.x >= a.x && io.MousePos.x <= b.x &&
        io.MousePos.y >= a.y && io.MousePos.y <= b.y;
    bool changed = false;
    if (overPlot && io.KeyCtrl) {
        ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);
        if (io.MouseWheel != 0) {
            const double factor = std::pow(1.15, io.MouseWheel);
            horizontal.scale(factor, (io.MousePos.x - a.x) / plotWidth);
            vertical.scale(factor, (io.MousePos.y - a.y) / plotHeight);
            changed = true;
        }
    }
    if (active && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
        horizontal.move(-io.MouseDelta.x / (plotWidth * horizontal.zoom));
        vertical.move(-io.MouseDelta.y / (plotHeight * vertical.zoom));
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        changed = true;
    }
    if (changed) {
        workspaceCall(ctx, "workspace_update", {{"analysis_zoom_x", horizontal.zoom},
            {"analysis_zoom_y", vertical.zoom}, {"analysis_pan_x", horizontal.pan},
            {"analysis_pan_y", vertical.pan}});
    }
    const double maxHz = points.back()[0].get<double>();
    const auto frequency = [&](double hz) {
        return std::log(hz / 20) / std::log(maxHz / 20);
    };
    const auto x = [&](double normalized) {
        return a.x + static_cast<float>(horizontal.project(normalized)) * plotWidth;
    };
    const auto y = [&](double normalized) {
        return a.y + static_cast<float>(vertical.project(normalized)) * plotHeight;
    };
    const auto level = [](double db) {
        return (6 - std::clamp(db, -96.0, 6.0)) / 102;
    };
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(begin, end, ImGui::GetColorU32(ImGuiCol_FrameBg), 4);
    draw->PushClipRect(a, b, true);
    const ImU32 accent = ImGui::GetColorU32(ImGuiCol_PlotLines);
    if (history && report.contains("history")) {
        const double sampleRate = report.at("sample_rate").get<double>();
        const double latest = report.at("end_frame").get<double>() / sampleRate;
        const double window = report.at("window_frames").get<double>() / sampleRate;
        for (const auto& row : report.at("history")) {
            const double time = row.at("end_frame").get<double>() / sampleRate;
            const float left = x(1 + (time - window - latest) / 10);
            const float right = x(1 + (time - latest) / 10);
            if (right < a.x || left > b.x) {
                continue;
            }
            const auto& db = row.at("dbfs");
            // Merge adjacent cells when fully zoomed out, preserving narrow peaks.
            // With clipping this also bounds draw vertices for 16-bit index backends.
            const std::size_t stride = vertical.zoom < 2 ? 2 : 1;
            for (std::size_t i = 0; i < db.size(); i += stride) {
                const auto last = std::min(i + stride - 1, db.size() - 1);
                const double low = std::max(0.0, (double(i) - 0.5) / (db.size() - 1));
                const double high = std::min(1.0, (double(last) + 0.5) / (db.size() - 1));
                const float top = y(1 - high);
                const float bottom = y(1 - low);
                if (bottom < a.y || top > b.y) {
                    continue;
                }
                const double peak = std::max(db[i].get<double>(), db[last].get<double>());
                draw->AddRectFilled(ImVec2(left, top), ImVec2(right, bottom), heat(peak));
            }
        }
    } else if (bands) {
        for (const auto& band : report.at("bands")) {
            const float left = x(frequency(band.at("low_hz"))) + 2;
            const float right = x(frequency(band.at("high_hz"))) - 2;
            const float top = y(level(band.at("dbfs")));
            if (right > left) {
                draw->AddRectFilled(ImVec2(left, top), ImVec2(right, y(1)),
                    ImGui::GetColorU32(ImGuiCol_PlotHistogram), 2);
            }
        }
    } else {
        for (std::size_t i = 1; i < points.size(); ++i) {
            const ImVec2 previous(x(frequency(points[i - 1][0])), y(level(points[i - 1][1])));
            const ImVec2 current(x(frequency(points[i][0])), y(level(points[i][1])));
            if (mode == "filled") {
                draw->AddQuadFilled(previous, current, ImVec2(current.x, y(1)),
                    ImVec2(previous.x, y(1)), ImGui::GetColorU32(ImGuiCol_PlotLines, 0.25f));
            }
            draw->AddLine(previous, current, accent, 1.5f);
        }
    }
    draw->PopClipRect();
    const auto muted = ImGui::GetColorU32(ImGuiCol_TextDisabled);
    const auto grid = ImGui::GetColorU32(ImGuiCol_Border);
    const auto tickX = [&](double value, const char* text) {
        const float position = x(value);
        if (position >= a.x - 0.5f && position <= b.x + 0.5f) {
            draw->AddLine(ImVec2(position, a.y), ImVec2(position, b.y), grid);
            draw->AddText(ImVec2(std::clamp(position - 8, a.x,
                std::max(a.x, end.x - ImGui::CalcTextSize(text).x)), b.y + 4), muted, text);
        }
    };
    const auto tickY = [&](double value, const char* text) {
        const float position = y(value);
        if (position >= a.y - 0.5f && position <= b.y + 0.5f) {
            draw->AddLine(ImVec2(a.x, position), ImVec2(b.x, position), grid);
            draw->AddText(ImVec2(begin.x + 3, position - font * 0.5f), muted, text);
        }
    };
    if (history) {
        for (int seconds = -10; seconds <= 0; ++seconds) {
            const auto label = std::to_string(seconds) + "s";
            tickX(1 + seconds / 10.0, label.c_str());
        }
        for (int hz : {20, 125, 500, 1000, 4000, 10000, 20000}) {
            if (hz <= maxHz) {
                const auto label = std::to_string(hz);
                tickY(1 - frequency(hz), label.c_str());
            }
        }
    } else {
        for (int hz : {20, 125, 250, 500, 1000, 2000, 4000, 10000, 20000}) {
            if (hz <= maxHz) {
                char label[24];
                std::snprintf(label, sizeof(label), hz >= 1000 ? "%.0fk" : "%.0f", hz >= 1000 ?
                    hz / 1000.0 : double(hz));
                tickX(frequency(hz), label);
            }
        }
        for (int db = -96; db <= 0; db += 12) {
            const auto label = std::to_string(db);
            tickY(level(db), label.c_str());
        }
    }
    draw->AddRect(a, b, grid);
    if (overPlot && !active) {
        const double u = horizontal.origin() + (io.MousePos.x - a.x) /
            (plotWidth * horizontal.zoom);
        const double v = vertical.origin() + (io.MousePos.y - a.y) /
            (plotHeight * vertical.zoom);
        draw->AddLine(ImVec2(io.MousePos.x, a.y), ImVec2(io.MousePos.x, b.y), muted);
        if (history) {
            ImGui::SetTooltip("%.2f s relative to latest snapshot | %.0f Hz\n"
                "Ctrl+wheel: zoom | Middle drag: pan", (u - 1) * 10,
                20 * std::pow(maxHz / 20, 1 - v));
        } else {
            ImGui::SetTooltip("%.0f Hz | cursor %.1f dBFS\n"
                "Ctrl+wheel: zoom | Middle drag: pan", 20 * std::pow(maxHz / 20, u),
                6 - v * 102);
        }
    }
    if (history) {
        ImGui::TextDisabled("Time / Hz | color: -96 to 0 dBFS | observed windows; gaps = not sampled");
    } else {
        ImGui::TextDisabled(bands ? "Hz / band mean-square dBFS" : "Hz / sine-peak dBFS");
    }
}
} // namespace AudioUI
