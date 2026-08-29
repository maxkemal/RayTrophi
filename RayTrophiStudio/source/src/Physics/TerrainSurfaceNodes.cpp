/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          TerrainSurfaceNodes.cpp
 * License:       MIT
 * =========================================================================
 */

#include "TerrainSurfaceNodes.h"

#include "NodeSystem/NodeRegistry.h"
#include "TerrainFieldMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace TerrainNodesV2 {
namespace {

float clamp01(float value) {
    return (std::clamp)(value, 0.0f, 1.0f);
}

float smoothstep(float low, float high, float value) {
    if (high <= low + 1.0e-8f) return value >= high ? 1.0f : 0.0f;
    const float t = clamp01((value - low) / (high - low));
    return t * t * (3.0f - 2.0f * t);
}

uint32_t hash2(int x, int y, int seed) {
    uint32_t h = static_cast<uint32_t>(x) * 0x8da6b343u;
    h ^= static_cast<uint32_t>(y) * 0xd8163841u;
    h ^= static_cast<uint32_t>(seed) * 0xcb1ab31fu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

float hashSigned(int x, int y, int seed) {
    return static_cast<float>(hash2(x, y, seed) & 0x00ffffffu) /
        static_cast<float>(0x007fffffu) - 1.0f;
}

float valueNoise(float x, float y, int seed) {
    const int ix = static_cast<int>(std::floor(x));
    const int iy = static_cast<int>(std::floor(y));
    float fx = x - static_cast<float>(ix);
    float fy = y - static_cast<float>(iy);
    fx = fx * fx * (3.0f - 2.0f * fx);
    fy = fy * fy * (3.0f - 2.0f * fy);
    const float a = hashSigned(ix, iy, seed);
    const float b = hashSigned(ix + 1, iy, seed);
    const float c = hashSigned(ix, iy + 1, seed);
    const float d = hashSigned(ix + 1, iy + 1, seed);
    const float ab = a + (b - a) * fx;
    const float cd = c + (d - c) * fx;
    return ab + (cd - ab) * fy;
}

float fbm(float x, float y, int seed, int bands = 4) {
    float sum = 0.0f;
    float amplitude = 1.0f;
    float weight = 0.0f;
    for (int band = 0; band < bands; ++band) {
        sum += valueNoise(x, y, seed + band * 1013) * amplitude;
        weight += amplitude;
        amplitude *= 0.5f;
        x = x * 2.03f + 7.17f;
        y = y * 2.03f - 3.41f;
    }
    return weight > 1.0e-8f ? sum / weight : 0.0f;
}

struct HardnessSettings {
    float base = 0.28f;
    float slopeStartDegrees = 22.0f;
    float slopeFullDegrees = 58.0f;
    float exposureHardening = 0.68f;
    float convexityStart = 0.012f;
    float convexityFull = 0.18f;
    float convexityInfluence = 0.42f;
    float sharpnessInfluence = 0.35f;
    float heterogeneity = 0.10f;
    float geologyScaleMeters = 42.0f;
    float fractureStrength = 0.36f;
    float fractureSoftening = 0.22f;
    float authoredInfluence = 1.0f;
    int seed = 4171;
};

struct HardnessFields {
    std::vector<float> hardness;
    std::vector<float> exposure;
    std::vector<float> fracture;
};

bool deriveHardness(const std::vector<float>& height, int w, int h,
                    const TerrainFieldMath::FieldMetric& metric,
                    const std::vector<float>* lithology,
                    const std::vector<float>* authored,
                    const HardnessSettings& settings,
                    HardnessFields& result) {
    const size_t count = static_cast<size_t>(w) * h;
    if (w < 2 || h < 2 || height.size() != count) return false;
    if (lithology && lithology->size() != count) return false;
    if (authored && authored->size() != count) return false;

    result.hardness.assign(count, 0.0f);
    result.exposure.assign(count, 0.0f);
    result.fracture.assign(count, 0.0f);
    const float geologyScale = (std::max)(settings.geologyScaleMeters,
                                          metric.cellSize * 4.0f);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t index = static_cast<size_t>(y) * w + x;
            const int xl = (std::max)(x - 1, 0);
            const int xr = (std::min)(x + 1, w - 1);
            const int yu = (std::max)(y - 1, 0);
            const int yd = (std::min)(y + 1, h - 1);
            const float center = height[index];
            const float neighborMean = (height[static_cast<size_t>(y) * w + xl] +
                height[static_cast<size_t>(y) * w + xr] +
                height[static_cast<size_t>(yu) * w + x] +
                height[static_cast<size_t>(yd) * w + x]) * 0.25f;
            const float curvatureGrade = (center - neighborMean) *
                metric.heightScale / (std::max)(metric.cellSize, 1.0e-5f);
            const float convexity = smoothstep(settings.convexityStart,
                settings.convexityFull, (std::max)(curvatureGrade, 0.0f));
            const float sharpness = smoothstep(settings.convexityStart * 0.5f,
                settings.convexityFull * 1.35f, std::abs(curvatureGrade));
            const float slopeDegrees = TerrainFieldMath::slopeDegrees(
                TerrainFieldMath::gradientAt(height, w, h, x, y, metric));
            const float slopeExposure = smoothstep(settings.slopeStartDegrees,
                settings.slopeFullDegrees, slopeDegrees);
            const float structure = clamp01(convexity * settings.convexityInfluence +
                sharpness * settings.sharpnessInfluence);
            const float exposure = clamp01(slopeExposure * (0.58f + structure * 0.42f));

            const float worldX = static_cast<float>(x) * metric.cellSize;
            const float worldY = static_cast<float>(y) * metric.cellSize;
            const float geology = fbm(worldX / geologyScale, worldY / geologyScale,
                                      settings.seed, 4);
            float substrate = lithology ? clamp01((*lithology)[index]) :
                clamp01(settings.base + geology * settings.heterogeneity);

            // Fractures follow abrupt curvature but receive a broad,
            // deterministic geological breakup. They reduce resistance rather
            // than being mistaken for exposed hard rock.
            const float fractureNoise = 0.5f + 0.5f * fbm(
                worldX / (geologyScale * 0.45f),
                worldY / (geologyScale * 0.45f), settings.seed + 1907, 3);
            const float fracture = clamp01(settings.fractureStrength * sharpness *
                (0.35f + fractureNoise * 0.65f));
            float hardness = substrate + (1.0f - substrate) * exposure *
                settings.exposureHardening;
            hardness *= 1.0f - fracture * settings.fractureSoftening;
            if (authored) {
                hardness = hardness + (clamp01((*authored)[index]) - hardness) *
                    clamp01(settings.authoredInfluence);
            }

            result.hardness[index] = clamp01(hardness);
            result.exposure[index] = exposure;
            result.fracture[index] = fracture;
        }
    }
    return true;
}

bool sameScalarShape(const NodeSystem::Image2DData& image, int w, int h) {
    return !image.isValid() ||
        (image.width == w && image.height == h && image.channels == 1);
}

float scalarAt(const NodeSystem::Image2DData& image, size_t index,
               float fallback) {
    return image.isValid() ? clamp01((*image.data)[index]) : fallback;
}

} // namespace

StructuralHardnessNode::StructuralHardnessNode() {
    name = "Structural Hardness";
    terrainNodeType = NodeType::StructuralHardness;
    inputs.push_back(NodeSystem::Pin::createInput(
        "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
    inputs.push_back(NodeSystem::Pin::createInput(
        "Lithology (optional)", NodeSystem::DataType::Image2D,
        NodeSystem::ImageSemantic::Mask, true));
    inputs.push_back(NodeSystem::Pin::createInput(
        "Authored Hardness (optional)", NodeSystem::DataType::Image2D,
        NodeSystem::ImageSemantic::Mask, true));
    for (const char* label : {"Hardness", "Rock Exposure", "Fracture"}) {
        outputs.push_back(NodeSystem::Pin::createOutput(
            label, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
    }
    metadata.displayName = "Structural Hardness";
    metadata.iconType = (int)UIWidgets::IconType::ClayTool;
    metadata.category = "Geology";
    metadata.description = "Derives exposed-rock erosion resistance from slope and structure";
    metadata.headerColor = IM_COL32(142, 92, 64, 255);
    headerColor = ImVec4(0.56f, 0.36f, 0.25f, 1.0f);
}

NodeSystem::PinValue StructuralHardnessNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
    const auto height = getHeightInput(0, ctx);
    const auto lithology = getMaskInput(1, ctx);
    const auto authored = getMaskInput(2, ctx);
    if (!height.isValid() || height.width < 2 || height.height < 2) {
        ctx.addError(id, "Structural Hardness requires at least a 2x2 Height input");
        return NodeSystem::PinValue{};
    }
    if (!sameScalarShape(lithology, height.width, height.height) ||
        !sameScalarShape(authored, height.width, height.height)) {
        ctx.addError(id, "Structural Hardness inputs must use the Height resolution");
        return NodeSystem::PinValue{};
    }

    HardnessSettings settings;
    settings.base = baseHardness;
    settings.slopeStartDegrees = slopeStartDegrees;
    settings.slopeFullDegrees = slopeFullDegrees;
    settings.exposureHardening = exposureHardening;
    settings.convexityStart = convexityStart;
    settings.convexityFull = convexityFull;
    settings.convexityInfluence = convexityInfluence;
    settings.sharpnessInfluence = sharpnessInfluence;
    settings.heterogeneity = heterogeneity;
    settings.geologyScaleMeters = geologyScaleMeters;
    settings.fractureStrength = fractureStrength;
    settings.fractureSoftening = fractureSoftening;
    settings.authoredInfluence = authoredInfluence;
    settings.seed = seed;

    HardnessFields fields;
    const std::vector<float>* lithologyData = lithology.isValid() ? lithology.data.get() : nullptr;
    const std::vector<float>* authoredData = authored.isValid() ? authored.data.get() : nullptr;
    if (!deriveHardness(*height.data, height.width, height.height,
                        getFieldMetric(ctx, height.width), lithologyData,
                        authoredData, settings, fields)) {
        ctx.addError(id, "Structural Hardness could not derive its fields");
        return NodeSystem::PinValue{};
    }

    std::array<NodeSystem::Image2DData, 3> result = {
        createMaskOutput(height.width, height.height),
        createMaskOutput(height.width, height.height),
        createMaskOutput(height.width, height.height)
    };
    *result[0].data = std::move(fields.hardness);
    *result[1].data = std::move(fields.exposure);
    *result[2].data = std::move(fields.fracture);
    for (int i = 0; i < 3; ++i) ctx.setCachedValue(id, i, result[i]);
    return outputIndex >= 0 && outputIndex < 3
        ? NodeSystem::PinValue{result[outputIndex]} : NodeSystem::PinValue{};
}

void StructuralHardnessNode::drawContent() {
    bool edited = false;
    edited |= ImGui::SliderFloat("Base Hardness", &baseHardness, 0.0f, 1.0f);
    edited |= ImGui::SliderFloat("Rock Starts", &slopeStartDegrees, 0.0f, 75.0f, "%.0f deg");
    edited |= ImGui::SliderFloat("Rock Full", &slopeFullDegrees, 5.0f, 89.0f, "%.0f deg");
    edited |= ImGui::SliderFloat("Exposure Hardening", &exposureHardening, 0.0f, 1.0f);
    edited |= ImGui::DragFloat("Convexity Start", &convexityStart, 0.002f, 0.0f, 2.0f, "%.3f");
    edited |= ImGui::DragFloat("Convexity Full", &convexityFull, 0.005f, 0.001f, 4.0f, "%.3f");
    edited |= ImGui::SliderFloat("Convexity", &convexityInfluence, 0.0f, 1.0f);
    edited |= ImGui::SliderFloat("Sharpness", &sharpnessInfluence, 0.0f, 1.0f);
    edited |= ImGui::SliderFloat("Heterogeneity", &heterogeneity, 0.0f, 0.5f);
    edited |= ImGui::DragFloat("Geology Scale", &geologyScaleMeters, 1.0f, 1.0f, 100000.0f, "%.0f m");
    edited |= ImGui::SliderFloat("Fractures", &fractureStrength, 0.0f, 1.0f);
    edited |= ImGui::SliderFloat("Fracture Softening", &fractureSoftening, 0.0f, 1.0f);
    edited |= ImGui::SliderFloat("Authored Influence", &authoredInfluence, 0.0f, 1.0f);
    edited |= ImGui::DragInt("Seed", &seed, 1.0f);
    if (slopeFullDegrees < slopeStartDegrees + 1.0f)
        slopeFullDegrees = slopeStartDegrees + 1.0f;
    if (convexityFull < convexityStart + 0.001f)
        convexityFull = convexityStart + 0.001f;
    if (edited) dirty = true;
}

void StructuralHardnessNode::serializeToJson(nlohmann::json& j) const {
    TerrainNodeBase::serializeToJson(j);
    j["baseHardness"] = baseHardness;
    j["slopeStartDegrees"] = slopeStartDegrees;
    j["slopeFullDegrees"] = slopeFullDegrees;
    j["exposureHardening"] = exposureHardening;
    j["convexityStart"] = convexityStart;
    j["convexityFull"] = convexityFull;
    j["convexityInfluence"] = convexityInfluence;
    j["sharpnessInfluence"] = sharpnessInfluence;
    j["heterogeneity"] = heterogeneity;
    j["geologyScaleMeters"] = geologyScaleMeters;
    j["fractureStrength"] = fractureStrength;
    j["fractureSoftening"] = fractureSoftening;
    j["authoredInfluence"] = authoredInfluence;
    j["seed"] = seed;
}

void StructuralHardnessNode::deserializeFromJson(const nlohmann::json& j) {
    TerrainNodeBase::deserializeFromJson(j);
    baseHardness = clampValue(j.value("baseHardness", baseHardness), 0.0f, 1.0f);
    slopeStartDegrees = clampValue(j.value("slopeStartDegrees", slopeStartDegrees), 0.0f, 75.0f);
    slopeFullDegrees = clampValue(j.value("slopeFullDegrees", slopeFullDegrees), slopeStartDegrees + 1.0f, 89.0f);
    exposureHardening = clampValue(j.value("exposureHardening", exposureHardening), 0.0f, 1.0f);
    convexityStart = clampValue(j.value("convexityStart", convexityStart), 0.0f, 2.0f);
    convexityFull = clampValue(j.value("convexityFull", convexityFull), convexityStart + 0.001f, 4.0f);
    convexityInfluence = clampValue(j.value("convexityInfluence", convexityInfluence), 0.0f, 1.0f);
    sharpnessInfluence = clampValue(j.value("sharpnessInfluence", sharpnessInfluence), 0.0f, 1.0f);
    heterogeneity = clampValue(j.value("heterogeneity", heterogeneity), 0.0f, 0.5f);
    geologyScaleMeters = (std::max)(j.value("geologyScaleMeters", geologyScaleMeters), 1.0f);
    fractureStrength = clampValue(j.value("fractureStrength", fractureStrength), 0.0f, 1.0f);
    fractureSoftening = clampValue(j.value("fractureSoftening", fractureSoftening), 0.0f, 1.0f);
    authoredInfluence = clampValue(j.value("authoredInfluence", authoredInfluence), 0.0f, 1.0f);
    seed = j.value("seed", seed);
}

SurfaceReliefNode::SurfaceReliefNode() {
    name = "Surface Relief";
    terrainNodeType = NodeType::SurfaceRelief;
    inputs.push_back(NodeSystem::Pin::createInput(
        "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
    for (const char* label : {"Slope (optional)", "Flow (optional)",
                              "Soil (optional)", "Hardness (optional)",
                              "Mask (optional)"}) {
        inputs.push_back(NodeSystem::Pin::createInput(
            label, NodeSystem::DataType::Image2D,
            NodeSystem::ImageSemantic::Mask, true));
        inputs.back().acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);
    }
    // There is deliberately no Flow Direction input any more. It demanded a
    // 2-channel VECTOR direction and Hydraulic Erosion's removed slot 5 was the
    // only pin in the terrain set that published one; measured across every
    // terrain node, the single remaining Direction OUTPUT is Watershed
    // Analysis's, and it is 1 channel. So nothing could connect here: the pin
    // was an authoring choice the graph could not honour, and Surface Relief
    // fell back to the downhill gradient every single time. The gradient is
    // computed from the height this node is already reading, so no measurement
    // was lost -- only a socket that could never be filled.
    outputs.push_back(NodeSystem::Pin::createOutput(
        "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
    for (const char* label : {"Rock Detail", "Rills", "Soil Detail"}) {
        outputs.push_back(NodeSystem::Pin::createOutput(
            label, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
    }
    metadata.displayName = "Surface Relief";
    metadata.iconType = (int)UIWidgets::IconType::DrawSharpTool;
    metadata.category = "Landform";
    metadata.description = "Flow-aligned rock, soil and rill micro geometry";
    metadata.headerColor = IM_COL32(122, 103, 76, 255);
    headerColor = ImVec4(0.48f, 0.40f, 0.30f, 1.0f);
}

NodeSystem::PinValue SurfaceReliefNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
    const auto height = getHeightInput(0, ctx);
    const auto slope = getMaskInput(1, ctx);
    const auto flow = getMaskInput(2, ctx);
    const auto soil = getMaskInput(3, ctx);
    const auto hardness = getMaskInput(4, ctx);
    const auto mask = getMaskInput(5, ctx);
    if (!height.isValid() || height.width < 2 || height.height < 2) {
        ctx.addError(id, "Surface Relief requires at least a 2x2 Height input");
        return NodeSystem::PinValue{};
    }
    const int w = height.width;
    const int h = height.height;
    for (const auto* field : {&slope, &flow, &soil, &hardness, &mask}) {
        if (!sameScalarShape(*field, w, h)) {
            ctx.addError(id, "Surface Relief scalar inputs must use the Height resolution");
            return NodeSystem::PinValue{};
        }
    }

    std::array<NodeSystem::Image2DData, 4> result = {
        createHeightOutput(w, h), createMaskOutput(w, h),
        createMaskOutput(w, h), createMaskOutput(w, h)
    };
    const auto metric = getFieldMetric(ctx, w);
    const float heightScale = (std::max)(metric.heightScale, 1.0e-4f);
    float representationCell = metric.cellSize;
    if (const auto* tctx = getTerrainContext(ctx); tctx && tctx->terrain) {
        representationCell = (std::max)(representationCell,
            metric.worldScale / static_cast<float>((std::max)(
                tctx->terrain->meshGridWidth() - 1, 1)));
    }
    const float feature = (std::max)(featureSizeMeters,
        representationCell * (std::max)(samplesPerFeature, 2.0f));
    const float stretch = (std::max)(directionStretch, 1.0f);
    const float maxAmplitude = std::tan(clampValue(maxAddedSlopeDegrees, 0.0f, 60.0f) /
        TerrainFieldMath::kRad2Deg) * feature * 0.5f;
    lastEffectiveFeatureMeters = feature;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t index = static_cast<size_t>(y) * w + x;
            const float worldX = static_cast<float>(x) * metric.cellSize;
            const float worldY = static_cast<float>(y) * metric.cellSize;
            const auto gradient = TerrainFieldMath::gradientComponentsAt(
                *height.data, w, h, x, y, metric);
            const float derivedSlope = TerrainFieldMath::slope01(gradient.magnitude());
            const float slopeValue = scalarAt(slope, index, derivedSlope);
            const float slopeClaim = smoothstep(slopeStartDegrees / 90.0f,
                slopeFullDegrees / 90.0f, slopeValue);
            const float hard = scalarAt(hardness, index, 0.38f);
            const float soilValue = scalarAt(soil, index, 1.0f - hard);
            const float flowValue = scalarAt(flow, index, 0.0f);
            const float authoredMask = scalarAt(mask, index, 1.0f);

            std::array<float, 2> flowDirection{-gradient.dzdx, -gradient.dzdy};
            float directionLength = std::hypot(flowDirection[0], flowDirection[1]);
            if (directionLength <= 1.0e-8f) {
                flowDirection = {1.0f, 0.0f};
            } else {
                flowDirection[0] /= directionLength;
                flowDirection[1] /= directionLength;
            }
            const float acrossX = -flowDirection[1];
            const float acrossY = flowDirection[0];
            const float along = worldX * flowDirection[0] + worldY * flowDirection[1];
            const float across = worldX * acrossX + worldY * acrossY;

            const float rockNoise = fbm(along / (feature * stretch), across / feature,
                                        seed + 101, 4);
            const float soilNoise = fbm(worldX / (feature * 1.8f),
                                        worldY / (feature * 1.8f), seed + 809, 3);
            const float rillCarrier = valueNoise(
                along / (feature * stretch * 0.75f),
                across / (feature * 0.34f), seed + 1877);
            const float rillLine = std::pow(clamp01(1.0f - std::abs(rillCarrier)), 5.0f);

            const float rockClaim = clamp01(hard * (0.22f + slopeClaim * 0.78f) *
                (1.0f - soilValue * 0.55f) * authoredMask);
            const float soilClaim = clamp01(soilValue * (1.0f - slopeClaim * 0.68f) *
                (1.0f - hard * 0.45f) * authoredMask);
            const float rillClaim = clamp01(flowValue * flowInfluence *
                (0.20f + slopeValue * 0.80f) * (0.32f + (1.0f - hard) * 0.68f) *
                authoredMask);

            float displacementMeters = rockNoise * rockAmplitudeMeters * rockClaim +
                soilNoise * soilAmplitudeMeters * soilClaim -
                rillLine * rillDepthMeters * rillClaim;
            displacementMeters = clampValue(displacementMeters, -maxAmplitude, maxAmplitude);
            (*result[0].data)[index] = (*height.data)[index] + displacementMeters / heightScale;
            (*result[1].data)[index] = clamp01(std::abs(rockNoise) * rockClaim);
            (*result[2].data)[index] = clamp01(rillLine * rillClaim);
            (*result[3].data)[index] = clamp01(std::abs(soilNoise) * soilClaim);
        }
    }

    for (int i = 0; i < 4; ++i) ctx.setCachedValue(id, i, result[i]);
    return outputIndex >= 0 && outputIndex < 4
        ? NodeSystem::PinValue{result[outputIndex]} : NodeSystem::PinValue{};
}

void SurfaceReliefNode::drawContent() {
    bool edited = false;
    edited |= ImGui::DragFloat("Feature Size", &featureSizeMeters, 0.5f, 0.25f, 10000.0f, "%.1f m");
    edited |= ImGui::DragFloat("Rock Relief", &rockAmplitudeMeters, 0.05f, 0.0f, 100.0f, "%.2f m");
    edited |= ImGui::DragFloat("Soil Relief", &soilAmplitudeMeters, 0.02f, 0.0f, 50.0f, "%.2f m");
    edited |= ImGui::DragFloat("Rill Depth", &rillDepthMeters, 0.02f, 0.0f, 50.0f, "%.2f m");
    edited |= ImGui::SliderFloat("Direction Stretch", &directionStretch, 1.0f, 16.0f);
    edited |= ImGui::SliderFloat("Flow Influence", &flowInfluence, 0.0f, 2.0f);
    edited |= ImGui::SliderFloat("Slope Starts", &slopeStartDegrees, 0.0f, 75.0f, "%.0f deg");
    edited |= ImGui::SliderFloat("Slope Full", &slopeFullDegrees, 5.0f, 89.0f, "%.0f deg");
    edited |= ImGui::DragFloat("Samples / Feature", &samplesPerFeature, 0.25f, 2.0f, 12.0f, "%.1f cells");
    edited |= ImGui::SliderFloat("Added Slope Limit", &maxAddedSlopeDegrees, 0.0f, 60.0f, "%.0f deg");
    edited |= ImGui::DragInt("Seed", &seed, 1.0f);
    if (slopeFullDegrees < slopeStartDegrees + 1.0f)
        slopeFullDegrees = slopeStartDegrees + 1.0f;
    if (lastEffectiveFeatureMeters > 0.0f) {
        ImGui::TextDisabled("effective %.1f m (downhill direction)",
            lastEffectiveFeatureMeters);
    }
    if (edited) dirty = true;
}

void SurfaceReliefNode::serializeToJson(nlohmann::json& j) const {
    TerrainNodeBase::serializeToJson(j);
    j["featureSizeMeters"] = featureSizeMeters;
    j["rockAmplitudeMeters"] = rockAmplitudeMeters;
    j["soilAmplitudeMeters"] = soilAmplitudeMeters;
    j["rillDepthMeters"] = rillDepthMeters;
    j["directionStretch"] = directionStretch;
    j["flowInfluence"] = flowInfluence;
    j["slopeStartDegrees"] = slopeStartDegrees;
    j["slopeFullDegrees"] = slopeFullDegrees;
    j["samplesPerFeature"] = samplesPerFeature;
    j["maxAddedSlopeDegrees"] = maxAddedSlopeDegrees;
    j["seed"] = seed;
}

void SurfaceReliefNode::deserializeFromJson(const nlohmann::json& j) {
    TerrainNodeBase::deserializeFromJson(j);
    featureSizeMeters = clampValue(j.value("featureSizeMeters", featureSizeMeters), 0.25f, 10000.0f);
    rockAmplitudeMeters = clampValue(j.value("rockAmplitudeMeters", rockAmplitudeMeters), 0.0f, 100.0f);
    soilAmplitudeMeters = clampValue(j.value("soilAmplitudeMeters", soilAmplitudeMeters), 0.0f, 50.0f);
    rillDepthMeters = clampValue(j.value("rillDepthMeters", rillDepthMeters), 0.0f, 50.0f);
    directionStretch = clampValue(j.value("directionStretch", directionStretch), 1.0f, 16.0f);
    flowInfluence = clampValue(j.value("flowInfluence", flowInfluence), 0.0f, 2.0f);
    slopeStartDegrees = clampValue(j.value("slopeStartDegrees", slopeStartDegrees), 0.0f, 75.0f);
    slopeFullDegrees = clampValue(j.value("slopeFullDegrees", slopeFullDegrees), slopeStartDegrees + 1.0f, 89.0f);
    samplesPerFeature = clampValue(j.value("samplesPerFeature", samplesPerFeature), 2.0f, 12.0f);
    maxAddedSlopeDegrees = clampValue(j.value("maxAddedSlopeDegrees", maxAddedSlopeDegrees), 0.0f, 60.0f);
    seed = j.value("seed", seed);
}

bool generateStructuralHardness(TerrainObject* terrain, float slopeWeight,
                                float heterogeneity, int seed) {
    if (!terrain || terrain->heightmap.width < 2 || terrain->heightmap.height < 2)
        return false;
    HardnessSettings settings;
    settings.exposureHardening = clampValue(slopeWeight, 0.0f, 1.0f);
    settings.heterogeneity = clampValue(heterogeneity, 0.0f, 0.5f);
    settings.seed = seed;
    HardnessFields fields;
    if (!deriveHardness(terrain->heightmap.data, terrain->heightmap.width,
                        terrain->heightmap.height,
                        TerrainFieldMath::makeFieldMetric(
                            terrain->heightmap.scale_xz,
                            terrain->heightmap.scale_y,
                            terrain->heightmap.width),
                        nullptr, nullptr, settings, fields)) {
        return false;
    }
    terrain->hardnessMap = std::move(fields.hardness);
    return true;
}

namespace {
NodeSystem::AutoRegisterNode<StructuralHardnessNode>
    regStructuralHardness("TerrainV2.StructuralHardness");
NodeSystem::AutoRegisterNode<SurfaceReliefNode>
    regSurfaceRelief("TerrainV2.SurfaceRelief");
} // namespace

} // namespace TerrainNodesV2
