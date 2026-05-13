#pragma once

#include "Vec3.h"

namespace PBRTextureChannelPolicy {

// Packed PBR texture policy used across CPU sampling and GPU shaders.
// Grayscale textures remain compatible because all channels typically carry
// the same value, but packed ORM-style textures are interpreted as:
//   G -> roughness
//   B -> metallic
static constexpr int RoughnessChannel = 1;
static constexpr int MetallicChannel = 2;
static constexpr int SpecularChannel = 0;

inline float sampleScalar(const Vec3& value, int channel) {
    switch (channel) {
        case RoughnessChannel: return static_cast<float>(value.y);
        case MetallicChannel: return static_cast<float>(value.z);
        case SpecularChannel: return static_cast<float>(value.x);
        default: return static_cast<float>(value.x);
    }
}

} // namespace PBRTextureChannelPolicy
