#pragma once

#include "VolumeShader.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace RayTrophiSim::Fluid {

struct FogOpticalBounds {
    // Exclusive voxel bounds, including trilinear support and block padding.
    std::array<int, 3> first{0, 0, 0};
    std::array<int, 3> end{0, 0, 0};
    bool empty = false;
    bool conservative_fallback = true;
};

inline std::string fogOpticalBoundsKey(const VolumeShader* shader, float scale) {
    if (!shader) {
        return "no-material";
    }
    return nlohmann::json{{"density", shader->density.toJson()},
        {"graph", shader->material_graph}, {"program", shader->material_program.toJson()},
        {"scale", scale}, {"scattering", shader->scattering.coefficient},
        {"absorption", shader->absorption.coefficient},
        {"emission_mode", static_cast<int>(shader->emission.mode)},
        {"independent_emission", shader->emission_response.independent_emission}}.dump();
}

inline FogOpticalBounds fogOpticalBounds(const std::vector<float>& density,
    int nx, int ny, int nz, const VolumeShader* shader, float scale = 1.0f) {
    FogOpticalBounds result;
    result.end = {nx, ny, nz};
    if (!shader || nx <= 0 || ny <= 0 || nz <= 0) {
        return result;
    }
    const auto& d = shader->density;
    // Graphs/noise may synthesize or reshape density in source-empty space.
    // Unknown policies retain the full grid, never a guessed raw-density box.
    if (!shader->material_graph.empty() || shader->material_program.active ||
        shader->material_program.density_noise_enabled || d.edge_falloff != 0.0f ||
        !std::isfinite(scale) || scale < 0.0f || !std::isfinite(d.multiplier) ||
        d.multiplier < 0.0f || !std::isfinite(d.cutoff_threshold) ||
        !std::isfinite(d.remap_low) || !std::isfinite(d.remap_high)) {
        return result;
    }
    if (d.curve.enabled) {
        for (int i = 0; i < 8; ++i) {
            if (!std::isfinite(d.curve.values[i]) || d.curve.values[i] < 0.0f ||
                d.curve.values[i] > 1.0f || (i == 0 && d.curve.values[i] != 0.0f) ||
                (i > 0 && d.curve.values[i] < d.curve.values[i - 1])) {
                return result;
            }
        }
    }
    std::size_t cells = 1;
    for (int size : {nx, ny, nz}) {
        if (cells > (std::numeric_limits<std::size_t>::max)() /
                static_cast<std::size_t>(size)) {
            return result;
        }
        cells *= static_cast<std::size_t>(size);
    }
    if (density.size() != cells) {
        return result;
    }
    const float denominator = (std::max)(d.remap_high - d.remap_low, 1e-6f);
    if (!std::isfinite(denominator)) {
        return result;
    }
    const float cutoff = (std::max)(d.cutoff_threshold, 0.0f);
    auto mapped = [&](float raw) {
        const float normalized = (std::max)((raw - d.remap_low) / denominator, 0.0f);
        const float optical = d.curve.evaluate(normalized);
        return shader->emission_response.independent_emission &&
                shader->emission.mode != VolumeEmissionMode::None
            ? (std::max)(optical, normalized) : optical;
    };
    // A positive mapped background cannot be bounded by occupied source cells.
    if (mapped(0.0f) > 0.0f) {
        return result;
    }
    result.conservative_fallback = false;
    result.empty = true;
    std::array<int, 3> low{nx, ny, nz};
    std::array<int, 3> high{-1, -1, -1};
    std::size_t index = 0;
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x, ++index) {
                const float raw = density[index];
                if (!std::isfinite(raw) || raw < 0.0f) {
                    return FogOpticalBounds{{0, 0, 0}, {nx, ny, nz}, false, true};
                }
                const float value = mapped(raw);
                if (!std::isfinite(value)) {
                    return FogOpticalBounds{{0, 0, 0}, {nx, ny, nz}, false, true};
                }
                // Cutoff is applied BEFORE multiplier in the canonical sampler.
                // Keep a small rounding margin at the CPU/GPU float boundary.
                const float margin = 16.0f * std::numeric_limits<float>::epsilon() *
                    (std::max)(1.0f, std::abs(value));
                if (((scale == 0.0f || d.multiplier == 0.0f) &&
                        !(shader->emission_response.independent_emission &&
                          shader->emission.mode != VolumeEmissionMode::None)) || value <= 0.0f ||
                    value + margin < cutoff) {
                    continue;
                }
                result.empty = false;
                const int point[3] = {x, y, z};
                for (int axis = 0; axis < 3; ++axis) {
                    low[axis] = (std::min)(low[axis], point[axis]);
                    high[axis] = (std::max)(high[axis], point[axis]);
                }
            }
        }
    }
    if (result.empty) {
        return result;
    }
    const int sizes[3] = {nx, ny, nz};
    for (int axis = 0; axis < 3; ++axis) {
        // Trilinear interpolation is a convex combination of eight raw values.
        // With a monotone remap/curve, an active interpolated sample requires
        // at least one active corner. Pad that corner's complete support first.
        result.first[axis] = (std::max)(0, ((low[axis] - 1) / 8) * 8);
        result.end[axis] = (std::min)(sizes[axis], ((high[axis] + 9) / 8) * 8);
    }
    return result;
}

} // namespace RayTrophiSim::Fluid
