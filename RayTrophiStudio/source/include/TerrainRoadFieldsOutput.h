#pragma once

#include "TerrainNodesV2.h"

#include <cstdint>
#include <string>

namespace TerrainNodesV2 {

// Atomic Phase-B publication sink for the immutable field bundle produced by
// TerrainRoadCarveNode. Downstream systems consume these stable names instead
// of depending on graph-local pin wiring.
class TerrainRoadFieldsOutputNode final : public TerrainNodeBase {
public:
    TerrainRoadFieldsOutputNode();

    std::string getTypeId() const override {
        return "TerrainV2.RoadFieldsOutput";
    }
    NodeSystem::PinValue compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) override;
    void drawContent() override;
    float getCustomWidth() const override { return 220.0f; }

    uint64_t lastPublishedRevision = 0;
    bool lastPublishSucceeded = false;
    std::string lastPublishError;
};

} // namespace TerrainNodesV2
