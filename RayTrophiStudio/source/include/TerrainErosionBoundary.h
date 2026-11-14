#pragma once
#include <algorithm>
#include <vector>

inline void blendTerrainEdgesToLevel(std::vector<float>& data, int w, int h, int fadeWidth,
                                     float targetLevel) {
    if (w <= 0 || h <= 0 || data.size() != static_cast<size_t>(w) * h || fadeWidth <= 0)
        return;
    fadeWidth = std::clamp(fadeWidth, 1, std::max(1, std::min(w, h) / 2));
#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int distFromEdge = std::min({x, y, w - 1 - x, h - 1 - y});
            if (distFromEdge >= fadeWidth)
                continue;
            const float t = static_cast<float>(distFromEdge) / static_cast<float>(fadeWidth);
            const float blend = t * t * (3.0f - 2.0f * t);
            const int idx = y * w + x;
            data[idx] = targetLevel * (1.0f - blend) + data[idx] * blend;
        }
    }
}
