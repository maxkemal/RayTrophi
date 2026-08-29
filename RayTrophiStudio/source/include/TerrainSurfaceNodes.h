/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          TerrainSurfaceNodes.h
 * License:       MIT
 * =========================================================================
 */
#pragma once

#include "TerrainNodesV2.h"

namespace TerrainNodesV2 {

    /**
     * Derives the effective erosion resistance of exposed terrain from the
     * actual landform. Lithology remains the intrinsic substrate; slope,
     * convexity and local sharpness only reveal how much resistant bedrock is
     * exposed at the surface.
     */
    class StructuralHardnessNode final : public TerrainNodeBase {
    public:
        float baseHardness = 0.28f;
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

        StructuralHardnessNode();

        NodeSystem::PinValue compute(int outputIndex,
                                     NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override {
            return "TerrainV2.StructuralHardness";
        }
        void serializeToJson(nlohmann::json& j) const override;
        void deserializeFromJson(const nlohmann::json& j) override;
    };

    /**
     * Adds resolution-safe, terrain-aware micro relief after erosion. Rock
     * fabric follows steep resistant faces, soil stays broad and soft, and
     * rills use the authoritative two-channel hydraulic flow direction when
     * present (falling back to downhill gradient otherwise).
     */
    class SurfaceReliefNode final : public TerrainNodeBase {
    public:
        float featureSizeMeters = 14.0f;
        float rockAmplitudeMeters = 0.75f;
        float soilAmplitudeMeters = 0.24f;
        float rillDepthMeters = 0.34f;
        float directionStretch = 5.0f;
        float flowInfluence = 0.85f;
        float slopeStartDegrees = 16.0f;
        float slopeFullDegrees = 62.0f;
        float samplesPerFeature = 4.0f;
        float maxAddedSlopeDegrees = 24.0f;
        int seed = 7331;

        // Evaluation diagnostics, not authored or serialized.
        float lastEffectiveFeatureMeters = 0.0f;

        SurfaceReliefNode();

        NodeSystem::PinValue compute(int outputIndex,
                                     NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override {
            return "TerrainV2.SurfaceRelief";
        }
        void serializeToJson(nlohmann::json& j) const override;
        void deserializeFromJson(const nlohmann::json& j) override;
    };

    // Shared core used by the legacy Terrain UI action and the graph node.
    // Keeping this operation here removes the old OpenMP/shared-RNG data race
    // and makes UI, scripting and IPC observe the same deterministic result.
    bool generateStructuralHardness(TerrainObject* terrain,
                                    float slopeWeight,
                                    float heterogeneity,
                                    int seed = 4171);

} // namespace TerrainNodesV2
