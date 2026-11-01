#pragma once

#include "VolumeShader.h"

#include <string>

namespace volume_material {

// What smoke looks like, as a view of the four physical coefficients. Editing one
// leaves the others alone: color does not change density, opacity does not change
// color. Scatter and absorption coefficients stay the stored truth.
struct SmokeAppearance {
    Vec3 color{1.0f, 1.0f, 1.0f}; // Albedo per channel
    float opacity = 0.0f;         // Total extinction per unit density
    Vec3 tint{1.0f, 1.0f, 1.0f};  // Wavelengths absorption removes with depth
};
SmokeAppearance smokeAppearance(const VolumeShader& shader);
void applySmokeAppearance(VolumeShader& shader, const SmokeAppearance& appearance);

nlohmann::json inspect(const VolumeShader& shader);
// Atomic overlay: unknown keys, non-finite values and invalid ranges fail.
bool configure(VolumeShader& shader, const nlohmann::json& patch, std::string& error);

} // namespace volume_material
