#pragma once

#include "NodeSystem/NodeCore.h"

#include <array>

namespace TerrainPaintEvaluation {

    struct PaintDomain {
        int width = 0;
        int height = 0;
        float worldScale = 1.0f;
        float heightScale = 1.0f;
    };

    struct AutoSplatRule {
        float heightMin = 0.0f;
        float heightMax = 1000.0f;
        float slopeMin = 0.0f;
        float slopeMax = 90.0f;
        float heightWeight = 0.5f;
        float slopeWeight = 0.5f;
        float falloff = 10.0f;
        float noiseAmount = 0.05f;
        bool enabled = true;
    };

    struct AutoSplatSettings {
        std::array<AutoSplatRule, 4> rules;
        bool normalizeOutput = true;
        int noiseSeed = 42;
    };

    struct SurfaceComposerSettings {
        float textureScale = 12.0f;
        float patchiness = 0.35f;
        float slopeInfluence = 0.65f;
        float soilInfluence = 0.80f;
        float flowInfluence = 0.45f;
        float wetnessInfluence = 0.75f;
        float hardnessInfluence = 0.60f;
        float grassInfluence = 1.0f;
        float rockInfluence = 1.0f;
        float snowInfluence = 1.0f;
        float iceInfluence = 0.85f;
        float contrast = 1.25f;
        int seed = 73;
    };

    struct SurfaceComposerInputs {
        NodeSystem::Image2DData height;
        NodeSystem::Image2DData soil;
        NodeSystem::Image2DData flow;
        NodeSystem::Image2DData wetness;
        NodeSystem::Image2DData hardness;
        NodeSystem::Image2DData snow;
        NodeSystem::Image2DData ice;
        NodeSystem::Image2DData meltwater;
        NodeSystem::Image2DData grass;
        NodeSystem::Image2DData rock;
    };

    NodeSystem::Image2DData evaluateAutoSplat(
        const NodeSystem::Image2DData& height,
        const PaintDomain& domain,
        const AutoSplatSettings& settings);

    NodeSystem::Image2DData evaluateFlowSemantic(
        const NodeSystem::Image2DData& flow,
        const PaintDomain& domain);

    NodeSystem::Image2DData evaluateSurfaceComposer(
        const SurfaceComposerInputs& inputs,
        const PaintDomain& domain,
        const SurfaceComposerSettings& settings,
        int outputKind); // 0 Surface mask, 1 Material RGBA, 2 Semantic RGBA

} // namespace TerrainPaintEvaluation
