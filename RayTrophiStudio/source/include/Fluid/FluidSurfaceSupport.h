#pragma once

#include <algorithm>

namespace RayTrophiSim::Fluid {

// A few isolated carriers must retain their individual support rather than
// filling the gap around a weighted centroid. Dense liquid keeps the existing
// reconstruction. Shared CPU policy; sim_surface_support.glsl is its GPU twin.
inline float surfaceSupportBlend(float weight, float squaredWeight) {
    const float effective = weight * weight / std::max(squaredWeight, 1.0e-12f);
    const float t = std::clamp((effective - 4.0f) / 12.0f, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline float constrainSparseSurface(float reconstructed, float nearest,
                                    float supportBlend) {
    return reconstructed + (1.0f - supportBlend) *
        (std::max(reconstructed, nearest) - reconstructed);
}

} // namespace RayTrophiSim::Fluid
