#pragma once
#include "imgui.h"
namespace AudioUI {
inline void drawAudioIcon(ImDrawList* draw, ImVec2 position, float size, ImU32 color,
                          float thickness) {
    const float amplitudes[] = {0.22f, 0.55f, 0.9f, 0.45f, 0.7f};
    for (int index = 0; index < 5; ++index) {
        const float x = position.x + size * (0.1f + index * 0.2f);
        const float center = position.y + size * 0.5f;
        const float radius = size * amplitudes[index] * 0.5f;
        draw->AddLine(ImVec2(x, center - radius), ImVec2(x, center + radius), color, thickness);
    }
}
}
