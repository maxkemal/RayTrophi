#pragma once

#include "TerrainNodesV2.h"
#include <vector>

namespace TerrainNodesV2 {

    // ------------------------------------------------------------------------
    // TerrainSatMapColorRampNode
    // Computes an RGB macro color map (SatMap) based on a scalar input (Height/Slope)
    // ------------------------------------------------------------------------
    class TerrainSatMapColorRampNode : public TerrainNodeBase {
    public:
        struct Stop {
            float pos;
            float r, g, b, a;
        };
        std::vector<Stop> stops;
        std::vector<Stop> slopeStops;
        std::vector<Stop> flowStops;
        std::vector<Stop> soilStops;
        std::vector<Stop> grassStops;

        // Height/physical scalar inputs are usually authored in metres, not
        // in the 0..1 mask range expected by a ramp.  Auto-normalization uses
        // a robust histogram range so isolated peaks do not collapse the
        // whole terrain into the last stop.
        bool autoNormalize = true;
        float normalizeLowPercentile = 2.0f;
        float normalizeHighPercentile = 94.0f;
        std::string preset = "Temperate";
        float slopeBlend = 0.72f;
        float flowBlend = 0.55f;
        float soilBlend = 0.45f;
        float grassBlend = 0.68f;
        float slopeR = 0.34f, slopeG = 0.31f, slopeB = 0.27f;
        float flowR = 0.10f, flowG = 0.24f, flowB = 0.12f;
        float soilR = 0.42f, soilG = 0.25f, soilB = 0.12f;
        float grassR = 0.12f, grassG = 0.34f, grassB = 0.10f;
        float snowBlend = 1.0f;
        float meltBlend = 0.75f;
        float avalancheBlend = 0.35f;
        float freshSnowR = 0.92f, freshSnowG = 0.96f, freshSnowB = 0.97f;
        float wetSnowR = 0.60f, wetSnowG = 0.68f, wetSnowB = 0.69f;
        float dirtySnowR = 0.47f, dirtySnowG = 0.43f, dirtySnowB = 0.37f;
        float iceR = 0.66f, iceG = 0.85f, iceB = 0.91f;
        bool autoDeriveMasks = true;
        float detailStrength = 0.11f;
        float detailScale = 180.0f;
        int debugView = 0; // 0 Final, 1 Height, 2 Slope, 3 Flow, 4 Soil, 5 Snow, 6 Ice, 7 Melt, 8 Avalanche, 9 Grass

        TerrainSatMapColorRampNode() {
            name = "SatMap ColorRamp";
            terrainNodeType = NodeType::SatMapColorRamp;

            // Primary height field. The optional fields let the preset make a
            // terrain-aware color decision instead of treating the whole map
            // as a single height ramp.
            inputs.push_back(NodeSystem::Pin::createInput("Height / Primary", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.back().acceptImageSemantic(NodeSystem::ImageSemantic::Height);
            inputs.back().acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);

            // Keep the original output pin immediately after the original
            // input so serialized links from existing graphs remain stable.
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Albedo));

            for (const char* label : {"Slope (optional)", "Flow (optional)", "Soil (optional)"}) {
                inputs.push_back(NodeSystem::Pin::createInput(label, NodeSystem::DataType::Image2D,
                    NodeSystem::ImageSemantic::Mask, true));
                inputs.back().acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);
            }
            for (const char* label : {"Snow (protected)", "Ice (optional)",
                                      "Meltwater (optional)", "Avalanche (optional)"}) {
                inputs.push_back(NodeSystem::Pin::createInput(label, NodeSystem::DataType::Image2D,
                    NodeSystem::ImageSemantic::Mask, true));
            }
            // Appended for serialized pin stability. Explicit biome grass wins;
            // Auto Derive supplies a terrain-aware fallback when unconnected.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Grass (optional)", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::Mask, true));

            // Default gradient stops
            stops.push_back({0.0f, 0.05f, 0.2f, 0.6f, 1.0f}); // Deep Blue (Water)
            stops.push_back({0.2f, 0.1f, 0.5f, 0.2f, 1.0f}); // Green (Grass)
            stops.push_back({0.6f, 0.4f, 0.3f, 0.2f, 1.0f}); // Brown (Dirt)
            stops.push_back({0.9f, 0.9f, 0.9f, 0.9f, 1.0f}); // White (Snow)
            slopeStops = {{0.0f, 0.18f, 0.20f, 0.14f, 1.0f},
                          {0.45f, 0.28f, 0.27f, 0.22f, 1.0f},
                          {0.75f, 0.38f, 0.36f, 0.32f, 1.0f},
                          {1.0f, 0.54f, 0.53f, 0.50f, 1.0f}};
            flowStops = {{0.0f, 0.30f, 0.22f, 0.12f, 1.0f},
                         {0.40f, 0.18f, 0.27f, 0.12f, 1.0f},
                         {0.75f, 0.08f, 0.24f, 0.13f, 1.0f},
                         {1.0f, 0.06f, 0.18f, 0.22f, 1.0f}};
            soilStops = {{0.0f, 0.18f, 0.12f, 0.07f, 1.0f},
                         {0.40f, 0.34f, 0.20f, 0.10f, 1.0f},
                         {0.75f, 0.48f, 0.30f, 0.14f, 1.0f},
                         {1.0f, 0.58f, 0.43f, 0.25f, 1.0f}};
            applyPreset("Temperate");
        }

        std::string getTypeId() const override { return "Terrain.SatMapColorRamp"; }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void serializeToJson(nlohmann::json& j) const override;
        void deserializeFromJson(const nlohmann::json& j) override;
        void drawContent() override;
        void applyPreset(const std::string& name);

    private:
        void sortStops();
    };

    enum class GrassMaskPreset {
        Temperate = 0,
        Lush,
        Alpine,
        Arid,
        Boreal,
        Custom
    };

    class TerrainGrassMaskNode : public TerrainNodeBase {
    public:
        GrassMaskPreset preset = GrassMaskPreset::Temperate;
        float density = 0.82f;
        float maxSlope = 0.52f;
        float slopeSoftness = 0.18f;
        float soilInfluence = 0.78f;
        float flowAvoidance = 0.86f;
        float wetnessPreference = 0.58f;
        float wetnessRange = 0.52f;
        float hardnessAvoidance = 0.62f;
        float patchiness = 0.28f;
        float detailScale = 96.0f;
        int seed = 731;

        TerrainGrassMaskNode();
        std::string getTypeId() const override { return "Terrain.GrassMask"; }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        void serializeToJson(nlohmann::json& j) const override;
        void deserializeFromJson(const nlohmann::json& j) override;
        void applyPreset(GrassMaskPreset value);
        static const char* presetName(GrassMaskPreset value);
    };

    class TerrainSatMapBlendNode : public TerrainNodeBase {
    public:
        float opacity = 1.0f;
        float maskPower = 1.0f;
        bool invertMask = false;

        TerrainSatMapBlendNode();
        std::string getTypeId() const override { return "Terrain.SatMapBlend"; }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        void serializeToJson(nlohmann::json& j) const override;
        void deserializeFromJson(const nlohmann::json& j) override;
    };

    class TerrainSurfaceMasksNode : public TerrainNodeBase {
    public:
        std::string preset = "Temperate";
        float cavityPower = 0.80f;
        float mudStrength = 0.82f;
        float mossStrength = 0.72f;
        float slopeSuppression = 0.72f;
        float detailScale = 110.0f;
        int seed = 947;

        TerrainSurfaceMasksNode();
        std::string getTypeId() const override { return "Terrain.SurfaceMasks"; }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        void serializeToJson(nlohmann::json& j) const override;
        void deserializeFromJson(const nlohmann::json& j) override;
        void applyPreset(const std::string& value);
    };

    class TerrainPaintMaskCombineNode : public TerrainNodeBase {
    public:
        TerrainPaintMaskCombineNode();
        std::string getTypeId() const override { return "Terrain.PaintMaskCombine"; }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
    };

    NodeSystem::PinValue computeAdaptiveCurvatureMask(
        CurvatureMaskNode& node, NodeSystem::EvaluationContext& ctx);

    // ------------------------------------------------------------------------
    // TerrainSatMapOutputNode
    // Takes an RGB image and applies it to the TerrainObject's macroColorMap
    // ------------------------------------------------------------------------
    class TerrainSatMapOutputNode : public TerrainNodeBase {
    public:
        float strength = 1.0f;

        TerrainSatMapOutputNode() {
            name = "SatMap Output";
            terrainNodeType = NodeType::SatMapOutput;

            // Output node (sink), so publication controls whether it modifies the scene
            publicationEnabled = true;

            // Inputs
            inputs.push_back(NodeSystem::Pin::createInput("Macro Color", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Albedo, true));
            inputs.push_back(NodeSystem::Pin::createInput("Strength", NodeSystem::DataType::Float, NodeSystem::ImageSemantic::Generic, true));
        }

        std::string getTypeId() const override { return "Terrain.SatMapOutput"; }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void serializeToJson(nlohmann::json& j) const override;
        void deserializeFromJson(const nlohmann::json& j) override;
        void drawContent() override;
    };

} // namespace TerrainNodesV2
