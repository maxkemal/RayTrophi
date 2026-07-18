/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          MaterialProceduralMath.h
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

// =============================================================================
// SHARED PROCEDURAL MATERIAL MATH  (Faz 2 "ortak header" — StylizeCore deseni)
// =============================================================================
// The exact procedural primitives used by BOTH the editor-side node graph
// (MaterialNodesV2.h: NoiseTextureNode::sampleFac, previews) AND the per-pixel
// runtime interpreter (MaterialProgram.h). Keeping them here — plain floats,
// no std containers, no engine types — means the node preview, the CPU render,
// and (Faz 2b/2c) the GLSL/CUDA ports all sample the identical function, so a
// pattern authored in the editor matches what renders. Everything is a
// straight port target: replace std::floor/fabs/sqrt with the device intrinsics
// and the bodies are unchanged.
//
// PCG avalanche chain — NOT a Teschner-style XOR fold, which mirror-ghosts at
// negative coordinates (see feedback_spatial_hash_mirror_symmetry).

namespace MaterialNodesV2 {

    inline uint32_t pcgHash(uint32_t x) {
        x = x * 747796405u + 2891336453u;
        x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
        return (x >> 22u) ^ x;
    }
    /// Raw float bits, with -0.0 folded to +0.0 (they hash differently but ARE the same
    /// point; a coordinate that lands on a negative zero would otherwise get its own
    /// random value). GLSL's floatBitsToUint, minus that trap.
    inline uint32_t floatBitsToU32(float f) {
        if (f == 0.0f) f = 0.0f;
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    }
    inline uint32_t hash2i(int xi, int yi, uint32_t seed) {
        return pcgHash(pcgHash(static_cast<uint32_t>(xi) + seed * 0x9E3779B9u) + static_cast<uint32_t>(yi));
    }
    inline float hash2f(int xi, int yi, uint32_t seed) {
        return static_cast<float>(hash2i(xi, yi, seed)) * (1.0f / 4294967295.0f);
    }
    inline float smoothFade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

    /// Object Info -> Random: a stable [0,1) per object, hashed from its WORLD-SPACE
    /// ORIGIN rather than from an instance id. The id would be the obvious key and is
    /// the wrong one: the CPU numbers instances in Embree geomID order and the GPU in
    /// TLAS order, two orderings built independently, so the same rock would land on a
    /// different id — and a different color — in each backend. The origin is a physical
    /// quantity both sides already hold bit-identically (Matrix4x4 is float32 and the
    /// TLAS transform is a verbatim copy of it), so the hash agrees exactly.
    ///
    /// Feeds the float BITS through the PCG avalanche chain (never a flat XOR fold —
    /// see feedback_spatial_hash_mirror_symmetry: XOR mirror-ghosts across the axes,
    /// which for a grid of scattered objects would visibly repeat colors about x/z=0).
    /// -0.0f is folded to +0.0f first: the two compare equal but have different bit
    /// patterns, and an object sitting on an axis can pick up a negative zero from a
    /// rotation multiply on one path and not the other.
    inline float objectRandom01(float ox, float oy, float oz) {
        const auto& bits = floatBitsToU32;
        const uint32_t h = pcgHash(pcgHash(pcgHash(bits(ox)) ^ bits(oy)) ^ bits(oz));
        return static_cast<float>(h) * (1.0f / 4294967296.0f);   // [0,1)
    }

    // ---- HSV (Hue/Saturation node) ------------------------------------------
    // Ported verbatim to GLSL (mp_rgbToHsv / mp_hsvToRgb). Same branch structure on both
    // sides on purpose: a "nicer" branchless GLSL rewrite is what makes two backends
    // disagree on the grey axis, where s == 0 and the hue is undefined.
    inline void rgbToHsv(float r, float g, float b, float& h, float& s, float& v) {
        const float mx = std::max(r, std::max(g, b));
        const float mn = std::min(r, std::min(g, b));
        const float d  = mx - mn;
        v = mx;
        s = (mx > 1e-8f) ? (d / mx) : 0.0f;
        if (d < 1e-8f) { h = 0.0f; return; }   // grey: hue is arbitrary, pick 0 on BOTH sides
        if (mx == r)      h = (g - b) / d + (g < b ? 6.0f : 0.0f);
        else if (mx == g) h = (b - r) / d + 2.0f;
        else              h = (r - g) / d + 4.0f;
        h *= 1.0f / 6.0f;
    }
    inline void hsvToRgb(float h, float s, float v, float& r, float& g, float& b) {
        h = h - std::floor(h);                 // wrap into [0,1)
        s = s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s);
        const float i = std::floor(h * 6.0f);
        const float f = h * 6.0f - i;
        const float p = v * (1.0f - s);
        const float q = v * (1.0f - s * f);
        const float t = v * (1.0f - s * (1.0f - f));
        switch (static_cast<int>(i) % 6) {
            case 0:  r = v; g = t; b = p; break;
            case 1:  r = q; g = v; b = p; break;
            case 2:  r = p; g = v; b = t; break;
            case 3:  r = p; g = q; b = v; break;
            case 4:  r = t; g = p; b = v; break;
            default: r = v; g = p; b = q; break;
        }
    }

    /// 2D value noise, [0,1]
    inline float valueNoise2D(float x, float y, uint32_t seed) {
        const int xi = static_cast<int>(std::floor(x));
        const int yi = static_cast<int>(std::floor(y));
        const float fx = x - static_cast<float>(xi);
        const float fy = y - static_cast<float>(yi);
        const float v00 = hash2f(xi,     yi,     seed);
        const float v10 = hash2f(xi + 1, yi,     seed);
        const float v01 = hash2f(xi,     yi + 1, seed);
        const float v11 = hash2f(xi + 1, yi + 1, seed);
        const float tx = smoothFade(fx), ty = smoothFade(fy);
        const float a = v00 + (v10 - v00) * tx;
        const float b = v01 + (v11 - v01) * tx;
        return a + (b - a) * ty;
    }

    /// fbm over value noise, [0,1]
    inline float fbm2D(float x, float y, int octaves, float gain, uint32_t seed) {
        float sum = 0.0f, amp = 1.0f, norm = 0.0f, freq = 1.0f;
        octaves = std::clamp(octaves, 1, 8);
        for (int i = 0; i < octaves; ++i) {
            sum += valueNoise2D(x * freq, y * freq, seed + static_cast<uint32_t>(i) * 101u) * amp;
            norm += amp;
            amp *= gain;
            freq *= 2.0f;
        }
        return (norm > 0.0f) ? sum / norm : 0.0f;
    }

    /// Ridged fbm, [0,1] — sharp crease lines (terrain Noise Generator's "Ridge"
    /// look, rebuilt on the hash-based value noise so it stays point-samplable
    /// and ports 1:1 to GLSL/CUDA in Faz 2).
    inline float ridge2D(float x, float y, int octaves, float gain, uint32_t seed) {
        float sum = 0.0f, amp = 1.0f, norm = 0.0f, freq = 1.0f;
        octaves = std::clamp(octaves, 1, 8);
        for (int i = 0; i < octaves; ++i) {
            const float n = 1.0f - std::fabs(2.0f * valueNoise2D(x * freq, y * freq, seed + static_cast<uint32_t>(i) * 101u) - 1.0f);
            sum += n * n * amp;
            norm += amp;
            amp *= gain;
            freq *= 2.0f;
        }
        return (norm > 0.0f) ? sum / norm : 0.0f;
    }

    /// Billow fbm, [0,1] — soft puffy lobes (terrain "Billow").
    inline float billow2D(float x, float y, int octaves, float gain, uint32_t seed) {
        float sum = 0.0f, amp = 1.0f, norm = 0.0f, freq = 1.0f;
        octaves = std::clamp(octaves, 1, 8);
        for (int i = 0; i < octaves; ++i) {
            sum += std::fabs(2.0f * valueNoise2D(x * freq, y * freq, seed + static_cast<uint32_t>(i) * 101u) - 1.0f) * amp;
            norm += amp;
            amp *= gain;
            freq *= 2.0f;
        }
        return (norm > 0.0f) ? sum / norm : 0.0f;
    }

    // -------------------------------------------------------------------------
    // 3D variants — for position-driven procedural (seamless solid texturing, no
    // UV seams). Same PCG-hash lattice, trilinear interpolation.
    // -------------------------------------------------------------------------
    inline uint32_t hash3i(int xi, int yi, int zi, uint32_t seed) {
        return pcgHash(pcgHash(pcgHash(static_cast<uint32_t>(xi) + seed * 0x9E3779B9u)
                               + static_cast<uint32_t>(yi)) + static_cast<uint32_t>(zi));
    }
    inline float hash3f(int xi, int yi, int zi, uint32_t seed) {
        return static_cast<float>(hash3i(xi, yi, zi, seed)) * (1.0f / 4294967295.0f);
    }
    inline float valueNoise3D(float x, float y, float z, uint32_t seed) {
        const int xi = static_cast<int>(std::floor(x));
        const int yi = static_cast<int>(std::floor(y));
        const int zi = static_cast<int>(std::floor(z));
        const float fx = smoothFade(x - static_cast<float>(xi));
        const float fy = smoothFade(y - static_cast<float>(yi));
        const float fz = smoothFade(z - static_cast<float>(zi));
        auto L = [](float a, float b, float t) { return a + (b - a) * t; };
        const float c000 = hash3f(xi,     yi,     zi,     seed), c100 = hash3f(xi + 1, yi,     zi,     seed);
        const float c010 = hash3f(xi,     yi + 1, zi,     seed), c110 = hash3f(xi + 1, yi + 1, zi,     seed);
        const float c001 = hash3f(xi,     yi,     zi + 1, seed), c101 = hash3f(xi + 1, yi,     zi + 1, seed);
        const float c011 = hash3f(xi,     yi + 1, zi + 1, seed), c111 = hash3f(xi + 1, yi + 1, zi + 1, seed);
        const float x00 = L(c000, c100, fx), x10 = L(c010, c110, fx);
        const float x01 = L(c001, c101, fx), x11 = L(c011, c111, fx);
        return L(L(x00, x10, fy), L(x01, x11, fy), fz);
    }
    inline float fbm3D(float x, float y, float z, int octaves, float gain, uint32_t seed) {
        float sum = 0.0f, amp = 1.0f, norm = 0.0f, freq = 1.0f;
        octaves = std::clamp(octaves, 1, 8);
        for (int i = 0; i < octaves; ++i) {
            sum += valueNoise3D(x * freq, y * freq, z * freq, seed + static_cast<uint32_t>(i) * 101u) * amp;
            norm += amp; amp *= gain; freq *= 2.0f;
        }
        return (norm > 0.0f) ? sum / norm : 0.0f;
    }
    inline float ridge3D(float x, float y, float z, int octaves, float gain, uint32_t seed) {
        float sum = 0.0f, amp = 1.0f, norm = 0.0f, freq = 1.0f;
        octaves = std::clamp(octaves, 1, 8);
        for (int i = 0; i < octaves; ++i) {
            const float n = 1.0f - std::fabs(2.0f * valueNoise3D(x * freq, y * freq, z * freq, seed + static_cast<uint32_t>(i) * 101u) - 1.0f);
            sum += n * n * amp;
            norm += amp; amp *= gain; freq *= 2.0f;
        }
        return (norm > 0.0f) ? sum / norm : 0.0f;
    }
    inline float billow3D(float x, float y, float z, int octaves, float gain, uint32_t seed) {
        float sum = 0.0f, amp = 1.0f, norm = 0.0f, freq = 1.0f;
        octaves = std::clamp(octaves, 1, 8);
        for (int i = 0; i < octaves; ++i) {
            sum += std::fabs(2.0f * valueNoise3D(x * freq, y * freq, z * freq, seed + static_cast<uint32_t>(i) * 101u) - 1.0f) * amp;
            norm += amp; amp *= gain; freq *= 2.0f;
        }
        return (norm > 0.0f) ? sum / norm : 0.0f;
    }
    inline float voronoi3D_F1(float px, float py, float pz, float randomness, uint32_t s, uint32_t* outCellHash = nullptr) {
        const int xi = static_cast<int>(std::floor(px));
        const int yi = static_cast<int>(std::floor(py));
        const int zi = static_cast<int>(std::floor(pz));
        float best = 1e30f;
        uint32_t bestHash = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const int cx = xi + dx, cy = yi + dy, cz = zi + dz;
            const uint32_t h = hash3i(cx, cy, cz, s);
            const float jx = (static_cast<float>(pcgHash(h ^ 0x1u)) * (1.0f / 4294967295.0f) - 0.5f) * randomness;
            const float jy = (static_cast<float>(pcgHash(h ^ 0x2u)) * (1.0f / 4294967295.0f) - 0.5f) * randomness;
            const float jz = (static_cast<float>(pcgHash(h ^ 0x3u)) * (1.0f / 4294967295.0f) - 0.5f) * randomness;
            const float fx = static_cast<float>(cx) + 0.5f + jx - px;
            const float fy = static_cast<float>(cy) + 0.5f + jy - py;
            const float fz = static_cast<float>(cz) + 0.5f + jz - pz;
            const float d = fx * fx + fy * fy + fz * fz;
            if (d < best) { best = d; bestHash = h; }
        }
        if (outCellHash) *outCellHash = bestHash;
        return std::sqrt(best);
    }

    /// Voronoi F1 distance (shared by the node's compute() and its preview).
    inline float voronoiF1(float px, float py, float randomness, uint32_t s, uint32_t* outCellHash = nullptr) {
        const int xi = static_cast<int>(std::floor(px));
        const int yi = static_cast<int>(std::floor(py));
        float best = 1e30f;
        uint32_t bestHash = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int cx = xi + dx, cy = yi + dy;
                const uint32_t h = hash2i(cx, cy, s);
                const float jx = (static_cast<float>(pcgHash(h ^ 0x1u)) * (1.0f / 4294967295.0f) - 0.5f) * randomness;
                const float jy = (static_cast<float>(pcgHash(h ^ 0x2u)) * (1.0f / 4294967295.0f) - 0.5f) * randomness;
                const float fx = static_cast<float>(cx) + 0.5f + jx - px;
                const float fy = static_cast<float>(cy) + 0.5f + jy - py;
                const float d = fx * fx + fy * fy;
                if (d < best) { best = d; bestHash = h; }
            }
        }
        if (outCellHash) *outCellHash = bestHash;
        return std::sqrt(best);
    }

} // namespace MaterialNodesV2
