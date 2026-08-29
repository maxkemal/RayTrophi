/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          TerrainFieldMath.h
* Author:        Kemal Demirtaş
* Date:          August 2026
* License:       MIT
* =========================================================================
*/
#pragma once

/**
 * @file TerrainFieldMath.h
 * @brief The single definition of terrain slope, curvature and mask range.
 *
 * Before this header the graph carried three incompatible answers to
 * "how steep is this pixel": raw degrees, atan(g)/(pi/2), and a
 * degrees/60 ramp that saturated every slope above 60 degrees to 1.
 * A 45 degree face therefore read as 45.0, 0.50 or 0.75 depending on
 * which node asked, so Auto Splat and Surface Composer could not agree
 * on the same terrain. Every slope in the terrain graph now routes
 * through the functions below.
 *
 * The canonical quantity is the *gradient*: rise over run in meters,
 * dimensionless. Degrees and 0-1 mask space are both derived from it,
 * never from each other.
 */

#include "NodeSystem/NodeCore.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace TerrainFieldMath {

    constexpr float kRad2Deg = 57.2957795f;
    constexpr float kHalfPi = 1.57079632679f;
    constexpr float kPi = 3.14159265359f;

    inline float clamp01f(float value) {
        return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
    }

    /**
     * @brief Sample spacing and vertical scale for one terrain field.
     *
     * cellSize is the distance between two adjacent samples. For a field
     * of @p width samples spanning @p worldScale meters that is
     * worldScale/(width-1), not worldScale/width: the samples sit on the
     * fence posts, not in the gaps. The graph previously mixed both
     * derivations, which put a resolution-dependent bias on every slope.
     */
    struct FieldMetric {
        float worldScale = 1.0f;   ///< Terrain extent in meters along one axis.
        float heightScale = 1.0f;  ///< Meters per unit of normalized height.
        float cellSize = 1.0f;     ///< Meters between adjacent samples.
    };

    inline FieldMetric makeFieldMetric(float worldScale, float heightScale, int width) {
        FieldMetric metric;
        metric.worldScale = (std::max)(worldScale, 1e-3f);
        metric.heightScale = (std::max)(std::abs(heightScale), 1e-3f);
        metric.cellSize = (std::max)(metric.worldScale /
            static_cast<float>((std::max)(width - 1, 1)), 1e-5f);
        return metric;
    }

    // ------------------------------------------------------------------
    // Canonical conversions. Gradient in, everything else out.
    // ------------------------------------------------------------------

    /// Slope angle in degrees. 0 = flat, 90 = vertical.
    inline float slopeDegrees(float gradient) {
        return std::atan(gradient) * kRad2Deg;
    }

    /**
     * @brief Slope in 0-1 mask space, linear in *angle*.
     *
     * 30 deg -> 0.333, 45 deg -> 0.500, 60 deg -> 0.667, 90 deg -> 1.0.
     * Nothing saturates before vertical, so steep rock keeps gradation.
     */
    inline float slope01(float gradient) {
        return clamp01f(std::atan(gradient) / kHalfPi);
    }

    /// Inverse of slope01, for turning an authored 0-1 dial back into a gradient.
    inline float gradientFromSlope01(float slope) {
        return std::tan(clamp01f(slope) * kHalfPi * 0.999f);
    }

    /// Convert an authored angle dial straight into mask space.
    inline float slope01FromDegrees(float degrees) {
        return clamp01f(degrees / 90.0f);
    }

    // ------------------------------------------------------------------
    // Gradient estimation
    // ------------------------------------------------------------------

    /**
     * @brief Central-difference gradient magnitude on a single-channel field.
     *
     * Neighbors are clamped to the field, and the run is the *actual*
     * distance walked, so border pixels get a correct one-sided gradient
     * instead of a halved one. Several call sites used to skip the border
     * row entirely, which left a one-pixel frame of "perfectly flat" around
     * every slope mask.
     *
     * @param heights Normalized height samples (multiplied by metric.heightScale here).
     */
    inline float gradientAt(const std::vector<float>& heights, int w, int h,
                            int x, int y, const FieldMetric& metric) {
        if (w < 2 || h < 2) return 0.0f;
        const int xl = (std::max)(x - 1, 0);
        const int xr = (std::min)(x + 1, w - 1);
        const int yu = (std::max)(y - 1, 0);
        const int yd = (std::min)(y + 1, h - 1);
        const float runX = (std::max)(static_cast<float>(xr - xl) * metric.cellSize, 1e-6f);
        const float runY = (std::max)(static_cast<float>(yd - yu) * metric.cellSize, 1e-6f);
        const float dzdx = (heights[static_cast<size_t>(y) * w + xr] -
                            heights[static_cast<size_t>(y) * w + xl]) * metric.heightScale / runX;
        const float dzdy = (heights[static_cast<size_t>(yd) * w + x] -
                            heights[static_cast<size_t>(yu) * w + x]) * metric.heightScale / runY;
        return std::sqrt(dzdx * dzdx + dzdy * dzdy);
    }

    inline float gradientAt(const NodeSystem::Image2DData& height, int x, int y,
                            const FieldMetric& metric) {
        if (!height.isValid() || !height.data) return 0.0f;
        return gradientAt(*height.data, height.width, height.height, x, y, metric);
    }

    /// Signed partial derivatives dz/dx and dz/dy in meters per meter.
    struct GradientComponents {
        float dzdx = 0.0f;
        float dzdy = 0.0f;
        float magnitude() const { return std::sqrt(dzdx * dzdx + dzdy * dzdy); }
    };

    /**
     * @brief Directional gradient, for consumers that need a surface normal.
     *
     * Shares the vertical scaling with gradientAt. Exposure used to build its
     * normal from *normalized* heights, so a 100 m terrain presented a
     * gradient a hundred times too shallow and read as almost flat to the sun.
     */
    inline GradientComponents gradientComponentsAt(const std::vector<float>& heights,
                                                   int w, int h, int x, int y,
                                                   const FieldMetric& metric) {
        GradientComponents components;
        if (w < 2 || h < 2) return components;
        const int xl = (std::max)(x - 1, 0);
        const int xr = (std::min)(x + 1, w - 1);
        const int yu = (std::max)(y - 1, 0);
        const int yd = (std::min)(y + 1, h - 1);
        const float runX = (std::max)(static_cast<float>(xr - xl) * metric.cellSize, 1e-6f);
        const float runY = (std::max)(static_cast<float>(yd - yu) * metric.cellSize, 1e-6f);
        components.dzdx = (heights[static_cast<size_t>(y) * w + xr] -
                           heights[static_cast<size_t>(y) * w + xl]) * metric.heightScale / runX;
        components.dzdy = (heights[static_cast<size_t>(yd) * w + x] -
                           heights[static_cast<size_t>(yu) * w + x]) * metric.heightScale / runY;
        return components;
    }

    /**
     * @brief Gradient built from an arbitrary bilinear sampler.
     *
     * Used where the evaluation domain and the height image differ in
     * resolution, so index arithmetic would sample the wrong texels.
     * @p sample takes normalized (u,v) and returns the normalized height.
     */
    template <typename SampleFn>
    inline float gradientAtUV(SampleFn&& sample, float u, float v,
                              int domainWidth, int domainHeight,
                              const FieldMetric& metric) {
        const float du = 1.0f / static_cast<float>((std::max)(domainWidth - 1, 1));
        const float dv = 1.0f / static_cast<float>((std::max)(domainHeight - 1, 1));
        const float uL = (std::max)(u - du, 0.0f), uR = (std::min)(u + du, 1.0f);
        const float vU = (std::max)(v - dv, 0.0f), vD = (std::min)(v + dv, 1.0f);
        const float runX = (std::max)((uR - uL) * metric.worldScale, 1e-6f);
        const float runY = (std::max)((vD - vU) * metric.worldScale, 1e-6f);
        const float dzdx = (sample(uR, v) - sample(uL, v)) * metric.heightScale / runX;
        const float dzdy = (sample(u, vD) - sample(u, vU)) * metric.heightScale / runY;
        return std::sqrt(dzdx * dzdx + dzdy * dzdy);
    }

    // ------------------------------------------------------------------
    // PhysicalScalar -> Mask range reconciliation
    // ------------------------------------------------------------------

    /**
     * @brief Largest magnitude in a single-channel field, or 0 for an empty one.
     *
     * A PhysicalScalar field (discharge, drainage area, water depth) has no
     * bounded range. Feeding one to a consumer that clamps to 0-1 collapses
     * it into a binary stencil: every channel pixel becomes exactly 1 and
     * everything else exactly 0. Normalizing against this maximum keeps the
     * gradation the field actually carries.
     */
    inline float maximumMagnitude(const NodeSystem::Image2DData& field) {
        if (!field.isValid() || !field.data || field.data->empty()) return 0.0f;
        float maximum = 0.0f;
        for (const float value : *field.data) {
            const float magnitude = std::abs(value);
            if (magnitude > maximum) maximum = magnitude;
        }
        return maximum;
    }

    /// True when the field carries SI values rather than an already-bounded 0-1 mask.
    inline bool isUnboundedField(const NodeSystem::Image2DData& field) {
        return field.semantic == NodeSystem::ImageSemantic::PhysicalScalar;
    }

    /**
     * @brief Units whose fields are accumulations down a drainage network.
     *
     * Drainage area and discharge are not merely large, they are heavy
     * tailed: every cell contributes one unit and the total gathers
     * downstream, so a hillslope sits near 1 while a trunk channel reaches
     * six figures. Stretching that linearly puts 98% of the terrain in the
     * bottom of the ramp and every channel in one saturated step - which
     * reads as a hard cut at the channel edge rather than a river.
     */
    inline bool isAccumulationUnit(NodeSystem::ImageUnit unit) {
        return unit == NodeSystem::ImageUnit::SquareMeters ||
               unit == NodeSystem::ImageUnit::CubicMetersPerSecond;
    }

    /**
     * @brief Ratio above which a field is treated as an accumulation.
     *
     * Measured, not assumed: a bounded physical field (water depth in
     * meters, say) keeps its maximum within an order of magnitude of its
     * mean, while a drainage accumulation runs hundreds to thousands of
     * times its mean. 32 sits well clear of both.
     */
    constexpr float kHeavyTailRatio = 32.0f;

    /// Mean magnitude, used to decide whether a field is heavy tailed.
    inline float meanMagnitude(const NodeSystem::Image2DData& field) {
        if (!field.isValid() || !field.data || field.data->empty()) return 0.0f;
        double total = 0.0;
        for (const float value : *field.data) total += std::abs(static_cast<double>(value));
        return static_cast<float>(total / static_cast<double>(field.data->size()));
    }

    /**
     * @brief Return @p field in 0-1 mask space, normalizing only if it needs it.
     *
     * A Mask input passes through untouched (no copy). A PhysicalScalar
     * input is brought into 0-1 and retagged as a Mask, so the consumer
     * downstream reads a real gradient instead of a clamped stencil. A
     * field whose maximum is zero comes back all-zero rather than dividing
     * by epsilon and exploding.
     *
     * The mapping depends on the shape of the field, because one mapping
     * cannot serve both:
     *  - A bounded field (water depth, channel width) divides by its max.
     *  - An accumulation (drainage area, discharge) is log-compressed, the
     *    same transform the flow overlay and the erosion solver already
     *    apply to this data. Stretched linearly, a drainage field puts the
     *    whole hillslope in the bottom of the range and every channel in
     *    one saturated step, so the river reads as a hard-edged cut.
     */
    inline NodeSystem::Image2DData asMaskField(const NodeSystem::Image2DData& field) {
        if (!field.isValid() || !field.data || field.channels != 1) return field;
        if (!isUnboundedField(field)) return field;

        const float maximum = maximumMagnitude(field);
        NodeSystem::Image2DData normalized;
        normalized.width = field.width;
        normalized.height = field.height;
        normalized.channels = 1;
        normalized.semantic = NodeSystem::ImageSemantic::Mask;
        normalized.data = std::make_shared<std::vector<float>>(field.data->size(), 0.0f);
        if (maximum <= 1e-9f) return normalized;

        const float mean = meanMagnitude(field);
        const bool accumulation = isAccumulationUnit(field.unit) ||
            (mean > 1e-9f && maximum / mean > kHeavyTailRatio);

        const std::vector<float>& source = *field.data;
        std::vector<float>& target = *normalized.data;
        if (accumulation) {
            const float denominator = std::log1p(maximum);
            if (denominator <= 1e-9f) return normalized;
            const float inverse = 1.0f / denominator;
            for (size_t i = 0; i < source.size(); ++i) {
                target[i] = clamp01f(std::log1p(std::abs(source[i])) * inverse);
            }
        } else {
            const float inverse = 1.0f / maximum;
            for (size_t i = 0; i < source.size(); ++i) {
                target[i] = clamp01f(std::abs(source[i]) * inverse);
            }
        }
        return normalized;
    }

} // namespace TerrainFieldMath
