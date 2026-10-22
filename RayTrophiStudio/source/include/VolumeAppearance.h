#pragma once

#include "Vec3.h"
#include <algorithm>
#include <cmath>

// Shared linear-light authoring math. Evaluate uniform controls before GPU upload,
// never once per march sample. Callers validate inputs before committing state.
namespace volume_appearance {

inline Vec3 saturation(const Vec3& color, float amount) {
    if (amount == 1.0f) {
        return color;
    }
    const auto luminance = color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f;
    return Vec3(luminance + (color.x - luminance) * amount,
                luminance + (color.y - luminance) * amount,
                luminance + (color.z - luminance) * amount);
}

inline Vec3 scatterColor(const Vec3& color, float amount) {
    const Vec3 graded = saturation(color, amount);
    // Scattering colour is an albedo, so grading cannot create gain.
    return Vec3((std::clamp)(graded.x, 0.0f, 1.0f),
                (std::clamp)(graded.y, 0.0f, 1.0f),
                (std::clamp)(graded.z, 0.0f, 1.0f));
}

inline Vec3 emissionColor(const Vec3& color, float amount, float exposureEv) {
    const Vec3 graded = saturation(color, amount);
    const float gain = std::exp2(exposureEv);
    // Preserve HDR emission; only negative channels are removed.
    return Vec3((std::max)(0.0f, graded.x) * gain,
                (std::max)(0.0f, graded.y) * gain,
                (std::max)(0.0f, graded.z) * gain);
}

} // namespace volume_appearance
