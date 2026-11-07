#pragma once

#include <vulkan/vulkan.h>

namespace Backend {

// The world color is also the spectral sky's sun tint. Only a flat-color
// environment authors the solid viewport background; sky/HDRI use neutral gray.
inline VkClearColorValue rasterBackgroundColor(bool flatColor, float r, float g, float b) {
    if (flatColor) {
        return {{r, g, b, 1.0f}};
    }
    return {{0.13f, 0.14f, 0.16f, 1.0f}};
}

} // namespace Backend
