#pragma once

#include "TerrainNodesV2.h"
#include "TerrainCurveMask.h"

#include <functional>
#include <string>
#include <vector>

namespace TerrainNodesV2 {

// UI-only provider. Evaluation never follows this callback; TerrainContext owns
// immutable snapshots captured before asynchronous graph work begins.
inline std::function<std::vector<std::string>()> g_terrainSplineListProvider;

class TerrainCurveInputNode final : public TerrainNodeBase {
public:
    char splineObject[256]{};

    TerrainCurveInputNode();
    std::string getTypeId() const override { return "TerrainV2.CurveInput"; }
    NodeSystem::PinValue compute(int outputIndex,
                                 NodeSystem::EvaluationContext& ctx) override;
    void drawContent() override;
    void serializeToJson(nlohmann::json& j) const override;
    void deserializeFromJson(const nlohmann::json& j) override;
    float getCustomWidth() const override { return 180.0f; }
};

class TerrainCurveToMaskNode final : public TerrainNodeBase {
public:
    CurveMaskSettings settings;

    TerrainCurveToMaskNode();
    std::string getTypeId() const override { return "TerrainV2.CurveToMask"; }
    NodeSystem::PinValue compute(int outputIndex,
                                 NodeSystem::EvaluationContext& ctx) override;
    void drawContent() override;
    void serializeToJson(nlohmann::json& j) const override;
    void deserializeFromJson(const nlohmann::json& j) override;
    float getCustomWidth() const override { return 175.0f; }
};

} // namespace TerrainNodesV2
