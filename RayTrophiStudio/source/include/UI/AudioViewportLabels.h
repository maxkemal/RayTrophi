#pragma once

#include "imgui.h"
#include <algorithm>
#include <string>
#include <vector>

namespace AudioUI {
// Display-only layout. Authored names, IDs and marker picking remain unchanged.
class ViewportLabels {
public:
    void draw(ImDrawList* drawList, ImVec2 marker, ImVec2 viewport,
              const std::string& name, ImU32 color) {
        const float padding = 4.0f;
        const float limit = std::min(180.0f, viewport.x - padding * 2.0f);
        if (limit <= 0.0f || viewport.y < ImGui::GetFontSize() + padding * 2.0f) {
            return;
        }
        const auto label = shorten(name, limit);
        const auto size = ImGui::CalcTextSize(label.c_str(), nullptr, false);
        const float x = std::clamp(marker.x + 12.0f, padding,
                                  std::max(padding, viewport.x - size.x - padding));
        for (int attempt = 0; attempt < 5; ++attempt) {
            const int row = attempt == 0 ? 0 : (attempt + 1) / 2;
            const float offset = row * (size.y + padding) * (attempt % 2 ? 1 : -1);
            const ImVec2 origin(x, marker.y - size.y * 0.5f + offset);
            const Box box{ImVec2(origin.x - padding, origin.y - padding),
                          ImVec2(origin.x + size.x + padding,
                                 origin.y + size.y + padding)};
            if (box.low.y < 0 || box.high.y > viewport.y) {
                continue;
            }
            const bool occupied = std::any_of(boxes_.begin(), boxes_.end(),
                [&](const Box& other) {
                    return box.low.x < other.high.x && box.high.x > other.low.x &&
                        box.low.y < other.high.y && box.high.y > other.low.y;
                });
            if (occupied) {
                continue;
            }
            if (attempt != 0) {
                drawList->AddLine(marker, ImVec2(origin.x, origin.y + size.y * 0.5f),
                                  color);
            }
            drawList->AddText(origin, color, label.c_str());
            boxes_.push_back(box);
            return;
        }
        // The selected source is submitted first, so it takes label priority.
        // Dense remaining sources retain their markers and full hover details.
    }

private:
    struct Box {
        ImVec2 low;
        ImVec2 high;
    };
    std::vector<Box> boxes_;

    static std::string shorten(std::string text, float width) {
        // Newlines/tabs must not expand a marker into a multiline overlay.
        for (auto& character : text) {
            if (character == '\n' || character == '\r' || character == '\t') {
                character = ' ';
            }
        }
        if (ImGui::CalcTextSize(text.c_str(), nullptr, false).x <= width) {
            return text;
        }
        const std::string suffix = "...";
        if (ImGui::CalcTextSize(suffix.c_str()).x > width) {
            return {};
        }
        while (!text.empty()) {
            // Remove an entire UTF-8 codepoint, preserving multibyte source names.
            auto end = text.size() - 1;
            while (end > 0 && (static_cast<unsigned char>(text[end]) & 0xc0) == 0x80) {
                --end;
            }
            text.resize(end);
            if (ImGui::CalcTextSize((text + suffix).c_str(), nullptr, false).x <= width) {
                return text + suffix;
            }
        }
        return suffix;
    }
};
}
