#include "MaterialDamagePattern.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim {
namespace {
uint32_t hash2(int x, int y, uint32_t seed) {
    uint32_t h = static_cast<uint32_t>(x) * 0x8da6b343u;
    h ^= static_cast<uint32_t>(y) * 0xd8163841u;
    h ^= seed * 0xcb1ab31fu;
    h ^= h >> 13u; h *= 0x85ebca6bu; h ^= h >> 16u;
    return h;
}

float lattice(int x, int y, uint32_t seed) {
    return static_cast<float>(hash2(x, y, seed) & 0x00ffffffu) / 16777215.0f;
}

float valueNoise(float x, float y, uint32_t seed) {
    const int ix = static_cast<int>(std::floor(x));
    const int iy = static_cast<int>(std::floor(y));
    float fx = x - static_cast<float>(ix), fy = y - static_cast<float>(iy);
    fx = fx * fx * (3.0f - 2.0f * fx);
    fy = fy * fy * (3.0f - 2.0f * fy);
    const float a = lattice(ix, iy, seed), b = lattice(ix + 1, iy, seed);
    const float c = lattice(ix, iy + 1, seed), d = lattice(ix + 1, iy + 1, seed);
    return (a + (b - a) * fx) +
           ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
}

float fbm(float x, float y, uint32_t seed) {
    float sum = 0.0f, weight = 0.58f, norm = 0.0f;
    for (int octave = 0; octave < 4; ++octave) {
        sum += valueNoise(x, y, seed + static_cast<uint32_t>(octave) * 101u) * weight;
        norm += weight;
        x = x * 2.03f + 7.1f; y = y * 2.11f - 3.7f;
        weight *= 0.48f;
    }
    return sum / std::max(norm, 1.0e-6f);
}
}

float applyMaterialDamagePattern(const std::string& substance,
                                 uint32_t texel, int resolution, float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    if (resolution <= 1 || value <= 0.0f || value >= 1.0f) return value;
    float amplitude = 0.0f;
    uint32_t seed = 17u;
    if (substance.find("Wood") != std::string::npos) {
        amplitude = 0.34f; seed = 0x574f4f44u;
    } else if (substance == "Paper") {
        amplitude = 0.08f; seed = 0x50415045u;
    } else if (substance == "Cloth") {
        amplitude = 0.14f; seed = 0x434c4f54u;
    }
    if (amplitude <= 0.0f) return value;

    const float u = (static_cast<float>(texel % static_cast<uint32_t>(resolution)) + 0.5f) /
                    static_cast<float>(resolution);
    const float v = (static_cast<float>(texel / static_cast<uint32_t>(resolution)) + 0.5f) /
                    static_cast<float>(resolution);
    float pattern;
    if (substance.find("Wood") != std::string::npos) {
        // Tall anisotropic fibres plus broad knot-like warping; no periodic sine,
        // so the front cannot settle into evenly spaced bands.
        const float warp = fbm(u * 4.0f, v * 5.0f, seed + 9u) - 0.5f;
        const float fibres = fbm(u * 7.0f + warp * 2.2f, v * 30.0f, seed);
        const float knots = fbm(u * 13.0f, v * 8.0f + warp, seed + 31u);
        pattern = fibres * 0.68f + knots * 0.32f;
    } else {
        pattern = fbm(u * 18.0f, v * 18.0f, seed);
    }
    const float active_front = 4.0f * value * (1.0f - value);
    return std::clamp(value + (pattern - 0.5f) * amplitude * active_front, 0.0f, 1.0f);
}

} // namespace RayTrophiSim
