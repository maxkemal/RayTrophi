#pragma once

#include "Texture.h"

#include <algorithm>
#include <cstdint>

namespace Paint {

inline void seedUntexturedRoughness(Texture& texture, float materialRoughnessColor) {
    // The bound map replaces the color term. The material's intensity still
    // multiplies the sampled byte, so baking it here would apply it twice.
    const float scalar = std::clamp(materialRoughnessColor, 0.0f, 1.0f);
    const auto value = static_cast<uint8_t>(scalar * 255.0f + 0.5f);
    std::fill(texture.pixels.begin(), texture.pixels.end(),
              CompactVec4(value, value, value, 255));
}

} // namespace Paint
