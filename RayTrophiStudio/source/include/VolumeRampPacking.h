#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// The current GPU ABI holds eight RGB stops. Keep ordinary ramps bit-for-bit;
// larger ramps retain both endpoints and the knots with the largest RGB error
// instead of silently discarding everything after stop eight. This is a bounded
// approximation, not a replacement for a future full-resolution ramp LUT.
namespace volume_material {

template <typename Ramp, typename Payload>
void packColorRamp(const Ramp& ramp, Payload& out) {
    constexpr std::size_t capacity = 8;
    const auto writeStop = [&](std::size_t dst, const auto& stop) {
        out.ramp_positions[dst] = stop.position;
        out.ramp_colors_r[dst] = static_cast<float>(stop.color.x);
        out.ramp_colors_g[dst] = static_cast<float>(stop.color.y);
        out.ramp_colors_b[dst] = static_cast<float>(stop.color.z);
    };
    out.color_ramp_enabled = ramp.enabled ? 1 : 0;
    if (ramp.stops.size() <= capacity) {
        out.ramp_stop_count = static_cast<int>(ramp.stops.size());
        for (std::size_t i = 0; i < ramp.stops.size(); ++i) {
            writeStop(i, ramp.stops[i]);
        }
        return;
    }

    auto stops = ramp.stops;
    std::stable_sort(stops.begin(), stops.end(), [](const auto& a, const auto& b) {
        return a.position < b.position;
    });
    std::vector<std::size_t> selected = {0, stops.size() - 1};
    selected.reserve(capacity);
    while (selected.size() < capacity) {
        float largestError = -1.0f;
        std::size_t candidate = 0;
        for (std::size_t segment = 1; segment < selected.size(); ++segment) {
            const auto left = selected[segment - 1];
            const auto right = selected[segment];
            const float width = stops[right].position - stops[left].position;
            for (std::size_t i = left + 1; i < right; ++i) {
                const float t = width > 0.0f
                    ? (stops[i].position - stops[left].position) / width : 0.0f;
                const auto error = [&](float a, float b, float value) {
                    return std::abs(value - (a + (b - a) * t));
                };
                const float rgbError = (std::max)({
                    error(stops[left].color.x, stops[right].color.x, stops[i].color.x),
                    error(stops[left].color.y, stops[right].color.y, stops[i].color.y),
                    error(stops[left].color.z, stops[right].color.z, stops[i].color.z)
                });
                if (rgbError > largestError) {
                    largestError = rgbError;
                    candidate = i;
                }
            }
        }
        // There are more input stops than capacity, so an unselected knot exists.
        selected.insert(std::lower_bound(selected.begin(), selected.end(), candidate),
                        candidate);
    }
    out.ramp_stop_count = static_cast<int>(selected.size());
    for (std::size_t i = 0; i < selected.size(); ++i) {
        writeStop(i, stops[selected[i]]);
    }
}

} // namespace volume_material
