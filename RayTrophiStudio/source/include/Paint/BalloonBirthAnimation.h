#pragma once

#include <algorithm>
#include <cmath>

namespace Paint {
struct BalloonBirthScale {
    float radial = 1.0f;
    float vertical = 1.0f;
};

// Visual-only, absolute scale relative to the authored transform. A gentle
// back-ease overshoot plus volume-preserving squash/stretch settles at identity.
inline BalloonBirthScale balloonBirthScale(double age, double duration) {
    if (!std::isfinite(age) || !std::isfinite(duration) || duration <= 0.0 || age >= duration) {
        return {};
    }
    const float t = static_cast<float>(std::clamp(age / duration, 0.0, 1.0));
    const float remaining = 1.0f - t;
    const float back =
        1.0f - 2.8f * remaining * remaining * remaining + 1.8f * remaining * remaining;
    const float growth = 0.08f + 0.92f * back;
    const float stretch =
        1.0f + 0.18f * std::sin(3.0f * 3.14159265359f * t) * remaining * remaining;
    return {growth / std::sqrt(stretch), growth * stretch};
}
} // namespace Paint
