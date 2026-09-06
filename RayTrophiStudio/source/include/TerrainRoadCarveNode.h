#pragma once

#include "TerrainNodesV2.h"
#include "TerrainRoadCarve.h"

#include <cstdint>
#include <memory>

namespace TerrainNodesV2 {

class TerrainRoadCarveNode final : public TerrainNodeBase {
public:
    RoadCarveSettings settings;

    TerrainRoadCarveNode();
    std::string getTypeId() const override { return "TerrainV2.RoadCarve"; }
    NodeSystem::PinValue compute(int outputIndex,
                                 NodeSystem::EvaluationContext& ctx) override;
    void drawContent() override;
    void serializeToJson(nlohmann::json& j) const override;
    void deserializeFromJson(const nlohmann::json& j) override;
    float getCustomWidth() const override { return 190.0f; }

private:
    uint64_t cachedKey_ = 0;
    uint64_t nextRevision_ = 1;
    std::shared_ptr<const RoadCarveResult> cachedResult_;
};

} // namespace TerrainNodesV2
