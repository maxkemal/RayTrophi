#pragma once

#include <string>
#include <cstdint>

namespace Paint {

enum class LayerBlendMode : uint8_t {
    Normal = 0,
    Add,
    Multiply,
    Screen,
    Overlay
};

struct PaintLayer {
    std::string name = "Layer";
    bool visible = true;
    bool locked = false;
    float opacity = 1.0f;
    LayerBlendMode blend_mode = LayerBlendMode::Normal;
    uint32_t enabled_channels = 0xFFFFFFFFu;  // bitmask: which PaintChannels this layer affects
};

inline const char* blendModeName(LayerBlendMode mode) {
    switch (mode) {
        case LayerBlendMode::Normal:   return "Normal";
        case LayerBlendMode::Add:      return "Add";
        case LayerBlendMode::Multiply: return "Multiply";
        case LayerBlendMode::Screen:   return "Screen";
        case LayerBlendMode::Overlay:  return "Overlay";
    }
    return "Normal";
}

constexpr int kLayerBlendModeCount = 5;

} // namespace Paint
