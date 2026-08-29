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
        /// Renamed from normalizeOutput, which named an operation Splat Output
        /// performed unconditionally anyway. This dial has a real effect:
        /// soil absorbs whatever budget no other layer claimed.
        bool soilFillsRemainder = true;
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
        /// Shared Terrain Analysis slope in 0-1. Empty falls back to a local
        /// gradient computed with the same canonical curve.
        NodeSystem::Image2DData slope;
        /// Thermal Erosion's talus mask: loose angular debris at cliff bases.
        /// It joins the rock claim rather than getting a layer of its own,
        /// because in a four-way Grass/Rock/Snow/Soil partition talus IS rock.
        NodeSystem::Image2DData talus;
    };

    /**
     * @param slope Optional shared slope field in 0-1 mask space. When empty
     *              the angle is derived from @p height using the same curve,
     *              so a wired and an unwired graph classify identically.
     */
    NodeSystem::Image2DData evaluateAutoSplat(
        const NodeSystem::Image2DData& height,
        const NodeSystem::Image2DData& slope,
        const PaintDomain& domain,
        const AutoSplatSettings& settings);

    /// The four independent semantic controls, packed as R/G/B/A.
    struct SemanticInputs {
        NodeSystem::Image2DData flow;
        NodeSystem::Image2DData wetness;
        NodeSystem::Image2DData ice;
        NodeSystem::Image2DData hardness;
    };

    /**
     * @brief Pack the semantic control texture. Renamed from
     *        evaluateFlowSemantic, which wrote R and hard-zeroed G/B/A.
     *
     * Nothing is synthesized here. An unconnected channel stays 0, which
     * terrain.list_layers reports as channel_coverage 0, rather than being
     * filled with a plausible guess: these channels drive erosion resistance
     * and soil capacity, so an invented value would be a guess steering a
     * solver.
     *
     * The four are deliberately NOT normalized against each other. They are
     * independent measurements of conditions, not shares of a surface.
     */
    NodeSystem::Image2DData evaluateSemanticControls(
        const SemanticInputs& inputs,
        const PaintDomain& domain);

    NodeSystem::Image2DData evaluateSurfaceComposer(
        const SurfaceComposerInputs& inputs,
        const PaintDomain& domain,
        const SurfaceComposerSettings& settings,
        int outputKind); // 0 Surface mask, 1 Material RGBA, 2 Semantic RGBA

} // namespace TerrainPaintEvaluation
