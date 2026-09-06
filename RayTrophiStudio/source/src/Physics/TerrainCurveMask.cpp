#include "TerrainCurveMask.h"

#include "MeshEdit/SplineEvaluationService.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace TerrainNodesV2 {
namespace {

struct Sample2D {
    float x = 0.0f;
    float z = 0.0f;
    float radius = 0.0f;
};

float saturate(float value) {
    return (std::max)(0.0f, (std::min)(1.0f, value));
}

float smoothMask(float distance, float radius, float falloff) {
    if (distance <= radius) return 1.0f;
    if (falloff <= 1.0e-6f || distance >= radius + falloff) return 0.0f;
    const float t = saturate((distance - radius) / falloff);
    const float smooth = t * t * (3.0f - 2.0f * t);
    return 1.0f - smooth;
}

float pointWidthAt(const BezierSpline& spline,
                   const MeshEdit::SplineEvaluation& evaluation) {
    if (spline.points.empty()) return 1.0f;
    const size_t count = spline.points.size();
    if (count == 1 || evaluation.segment < 0) {
        return (std::max)(0.0f, spline.points.front().userData1);
    }

    size_t a = static_cast<size_t>(evaluation.segment);
    size_t b = a + 1;
    if (spline.curveType == SplineCurveType::BSpline && count >= 4) {
        const size_t span = spline.bsplineSpanForSegment(a);
        a = span > 0 ? span - 1 : 0;
        b = (std::min)(span, count - 1);
    } else {
        a = (std::min)(a, count - 1);
        b = spline.isClosed ? (b % count) : (std::min)(b, count - 1);
    }
    const float value = spline.points[a].userData1 +
        (spline.points[b].userData1 - spline.points[a].userData1) *
        saturate(evaluation.local_t);
    return (std::max)(0.0f, value);
}

struct DenseSample {
    float t = 0.0f;
    float distance = 0.0f;
    Vec3 terrainPosition = Vec3(0.0f);
};

bool buildArcLengthSamples(const MeshEdit::CurveNodeData& curve,
                           const Matrix4x4& terrainWorldToLocal,
                           float cellSize,
                           float baseRadius,
                           bool usePointWidth,
                           std::vector<Sample2D>& samples,
                           std::string* error) {
    std::string validationError;
    if (!MeshEdit::SplineEvaluationService::validate(curve.spline, &validationError)) {
        if (error) *error = validationError;
        return false;
    }

    const int segments = MeshEdit::SplineEvaluationService::segmentCount(curve.spline);
    const int denseCount = (std::max)(segments * 32, 64);
    std::vector<DenseSample> dense(static_cast<size_t>(denseCount + 1));
    float length = 0.0f;
    for (int i = 0; i <= denseCount; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(denseCount);
        const auto evaluated = MeshEdit::SplineEvaluationService::evaluate(curve.spline, t);
        const Vec3 world = curve.local_to_world.transform_point(evaluated.position);
        const Vec3 local = terrainWorldToLocal.transform_point(world);
        if (i > 0) {
            const float dx = local.x - dense[static_cast<size_t>(i - 1)].terrainPosition.x;
            const float dz = local.z - dense[static_cast<size_t>(i - 1)].terrainPosition.z;
            length += std::sqrt(dx * dx + dz * dz);
        }
        dense[static_cast<size_t>(i)] = {t, length, local};
    }
    if (!(length > 1.0e-6f) || !std::isfinite(length)) {
        if (error) *error = "curve has no measurable XZ length";
        return false;
    }

    const float spacing = (std::max)(0.02f, cellSize * 0.5f);
    const int sampleCount = (std::min)(65536, (std::max)(2,
        static_cast<int>(std::ceil(length / spacing)) + 1));
    samples.reserve(static_cast<size_t>(sampleCount));
    size_t denseIndex = 1;
    for (int i = 0; i < sampleCount; ++i) {
        const float target = length * static_cast<float>(i) /
            static_cast<float>(sampleCount - 1);
        while (denseIndex < dense.size() && dense[denseIndex].distance < target) {
            ++denseIndex;
        }
        denseIndex = (std::min)(denseIndex, dense.size() - 1);
        const DenseSample& hi = dense[denseIndex];
        const DenseSample& lo = dense[denseIndex > 0 ? denseIndex - 1 : 0];
        const float span = hi.distance - lo.distance;
        const float alpha = span > 1.0e-8f ? (target - lo.distance) / span : 0.0f;
        const float t = lo.t + (hi.t - lo.t) * saturate(alpha);
        const auto evaluated = MeshEdit::SplineEvaluationService::evaluate(curve.spline, t);
        const Vec3 world = curve.local_to_world.transform_point(evaluated.position);
        const Vec3 local = terrainWorldToLocal.transform_point(world);
        const float multiplier = usePointWidth ? pointWidthAt(curve.spline, evaluated) : 1.0f;
        samples.push_back({local.x, local.z, baseRadius * multiplier});
    }
    return true;
}

void rasterBoundary(const std::vector<Sample2D>& samples,
                    bool closed,
                    int width,
                    int height,
                    float scale,
                    float falloff,
                    std::vector<float>& pixels) {
    const float pixelX = scale / static_cast<float>(width - 1);
    const float pixelZ = scale / static_cast<float>(height - 1);
    const size_t segmentCount = closed ? samples.size() : samples.size() - 1;
    for (size_t segment = 0; segment < segmentCount; ++segment) {
        const Sample2D& a = samples[segment];
        const Sample2D& b = samples[(segment + 1) % samples.size()];
        const float reach = (std::max)(a.radius, b.radius) + falloff;
        const int minX = (std::max)(0, static_cast<int>(std::floor(((std::min)(a.x, b.x) - reach) / pixelX)));
        const int maxX = (std::min)(width - 1, static_cast<int>(std::ceil(((std::max)(a.x, b.x) + reach) / pixelX)));
        const int minY = (std::max)(0, static_cast<int>(std::floor(((std::min)(a.z, b.z) - reach) / pixelZ)));
        const int maxY = (std::min)(height - 1, static_cast<int>(std::ceil(((std::max)(a.z, b.z) + reach) / pixelZ)));
        if (minX > maxX || minY > maxY) continue;

        const float dx = b.x - a.x;
        const float dz = b.z - a.z;
        const float lengthSquared = dx * dx + dz * dz;
        for (int y = minY; y <= maxY; ++y) {
            const float pz = static_cast<float>(y) * pixelZ;
            for (int x = minX; x <= maxX; ++x) {
                const float px = static_cast<float>(x) * pixelX;
                const float projection = lengthSquared > 1.0e-12f
                    ? saturate(((px - a.x) * dx + (pz - a.z) * dz) / lengthSquared)
                    : 0.0f;
                const float qx = a.x + dx * projection;
                const float qz = a.z + dz * projection;
                const float ddx = px - qx;
                const float ddz = pz - qz;
                const float radius = a.radius + (b.radius - a.radius) * projection;
                const float value = smoothMask(std::sqrt(ddx * ddx + ddz * ddz), radius, falloff);
                float& current = pixels[static_cast<size_t>(y) * width + x];
                current = (std::max)(current, value);
            }
        }
    }
}

void fillClosedPolygon(const std::vector<Sample2D>& samples,
                       int width,
                       int height,
                       float scale,
                       std::vector<float>& pixels) {
    const float pixelX = scale / static_cast<float>(width - 1);
    const float pixelZ = scale / static_cast<float>(height - 1);
    std::vector<float> intersections;
    intersections.reserve(samples.size());
    for (int y = 0; y < height; ++y) {
        const float z = static_cast<float>(y) * pixelZ;
        intersections.clear();
        for (size_t i = 0; i < samples.size(); ++i) {
            const Sample2D& a = samples[i];
            const Sample2D& b = samples[(i + 1) % samples.size()];
            if ((a.z > z) == (b.z > z)) continue;
            const float alpha = (z - a.z) / (b.z - a.z);
            intersections.push_back(a.x + (b.x - a.x) * alpha);
        }
        std::sort(intersections.begin(), intersections.end());
        for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
            const int begin = (std::max)(0, static_cast<int>(std::ceil(intersections[i] / pixelX)));
            const int end = (std::min)(width - 1, static_cast<int>(std::floor(intersections[i + 1] / pixelX)));
            for (int x = begin; x <= end; ++x) pixels[static_cast<size_t>(y) * width + x] = 1.0f;
        }
    }
}

} // namespace

bool sampleTerrainCurveXZ(const MeshEdit::CurveNodeData& curve,
                          const Matrix4x4& terrainWorldToLocal,
                          float cellSizeMeters,
                          bool usePointWidth,
                          std::vector<TerrainCurveSample>& samples,
                          std::string* error) {
    std::vector<Sample2D> internal;
    if (!buildArcLengthSamples(curve, terrainWorldToLocal, cellSizeMeters,
                               1.0f, usePointWidth, internal, error)) return false;
    samples.clear();
    samples.reserve(internal.size());
    float distance = 0.0f;
    for (size_t i = 0; i < internal.size(); ++i) {
        if (i > 0) {
            const float dx = internal[i].x - internal[i - 1].x;
            const float dz = internal[i].z - internal[i - 1].z;
            distance += std::sqrt(dx * dx + dz * dz);
        }
        samples.push_back({internal[i].x, internal[i].z,
                           internal[i].radius, distance});
    }
    return true;
}

bool rasterizeCurveMask(const MeshEdit::CurveNodeData& curve,
                        const Matrix4x4& terrainWorldToLocal,
                        int width,
                        int height,
                        float terrainScaleXZ,
                        const CurveMaskSettings& settings,
                        NodeSystem::Image2DData& output,
                        std::string* error) {
    if (width < 2 || height < 2 || !std::isfinite(terrainScaleXZ) || terrainScaleXZ <= 0.0f) {
        if (error) *error = "invalid terrain dimensions or scale";
        return false;
    }
    if (!std::isfinite(settings.widthMeters) || settings.widthMeters < 0.0f ||
        !std::isfinite(settings.falloffMeters) || settings.falloffMeters < 0.0f) {
        if (error) *error = "width and falloff must be finite and non-negative";
        return false;
    }
    if (settings.mode == CurveMaskMode::ClosedFill && !curve.spline.isClosed) {
        if (error) *error = "Closed Fill requires a closed spline";
        return false;
    }

    std::vector<Sample2D> samples;
    const float cell = terrainScaleXZ /
        static_cast<float>((std::max)(width - 1, height - 1));
    const float radius = settings.mode == CurveMaskMode::Stroke
        ? settings.widthMeters * 0.5f : 0.0f;
    if (!buildArcLengthSamples(curve, terrainWorldToLocal, cell, radius,
                               settings.usePointWidth && settings.mode == CurveMaskMode::Stroke,
                               samples, error)) {
        return false;
    }

    output.width = width;
    output.height = height;
    output.channels = 1;
    output.semantic = NodeSystem::ImageSemantic::Mask;
    output.unit = NodeSystem::ImageUnit::Unitless;
    output.data = std::make_shared<std::vector<float>>(
        static_cast<size_t>(width) * height, 0.0f);
    if (settings.mode == CurveMaskMode::ClosedFill) {
        fillClosedPolygon(samples, width, height, terrainScaleXZ, *output.data);
    }
    rasterBoundary(samples, curve.spline.isClosed, width, height,
                   terrainScaleXZ, settings.falloffMeters, *output.data);
    if (settings.invert) {
        for (float& value : *output.data) value = 1.0f - saturate(value);
    }
    return true;
}

} // namespace TerrainNodesV2
