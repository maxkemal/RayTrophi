#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <string>
#include "Paint/PaintLayer.h"
#include "Paint/PaintTextureSet.h"
#include "Texture.h"

namespace Paint {

// Axis-aligned bounding box in pixel coordinates for dirty-region compositing.
struct PaintDirtyRect {
    int min_x = 0, min_y = 0, max_x = -1, max_y = -1;

    bool empty() const { return max_x < min_x || max_y < min_y; }

    void expand(int x0, int y0, int x1, int y1) {
        if (empty()) { min_x = x0; min_y = y0; max_x = x1; max_y = y1; }
        else {
            if (x0 < min_x) min_x = x0;
            if (y0 < min_y) min_y = y0;
            if (x1 > max_x) max_x = x1;
            if (y1 > max_y) max_y = y1;
        }
    }

    void expand(const PaintDirtyRect& other) {
        if (!other.empty()) expand(other.min_x, other.min_y, other.max_x, other.max_y);
    }
};

// Per-layer pixel storage for each PBR channel.
// Each layer owns its own pixel buffers; compositing flattens
// the visible stack into the PaintTextureSet that the material references.
struct PaintLayerData {
    PaintLayer meta;                            // name, visibility, opacity, blend, lock
    uint32_t   id = 0;                          // unique id within the stack

    int width  = 0;
    int height = 0;

    // One pixel buffer per PaintChannel (BaseColor..Transmission = 7 channels).
    // Empty vector means "this channel has no paint on this layer yet" and
    // compositing should treat it as fully transparent / default.
    std::array<std::vector<CompactVec4>, 7> channel_pixels;

    // ---------- helpers ----------

    bool hasPixels(PaintChannel channel) const {
        const size_t idx = static_cast<size_t>(channel);
        return idx < channel_pixels.size() && !channel_pixels[idx].empty();
    }

    std::vector<CompactVec4>& ensurePixels(PaintChannel channel) {
        const size_t idx = static_cast<size_t>(channel);
        auto& buf = channel_pixels[idx];
        if (buf.empty() && width > 0 && height > 0) {
            // Allocate with fully-transparent black so untouched areas
            // contribute nothing during compositing.
            buf.assign(static_cast<size_t>(width) * static_cast<size_t>(height),
                       CompactVec4(0, 0, 0, 0));
        }
        return buf;
    }

    const std::vector<CompactVec4>& getPixels(PaintChannel channel) const {
        return channel_pixels[static_cast<size_t>(channel)];
    }

    void clearChannel(PaintChannel channel) {
        channel_pixels[static_cast<size_t>(channel)].clear();
    }

    void clearAll() {
        for (auto& buf : channel_pixels) buf.clear();
    }

    void resize(int new_width, int new_height);
};

} // namespace Paint
