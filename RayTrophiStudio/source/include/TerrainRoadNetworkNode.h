#pragma once

#include "TerrainNodesV2.h"
#include "TerrainRoadCarve.h"
#include "TerrainRoadNetwork.h"

#include <cstdint>
#include <memory>
#include <string>

namespace TerrainNodesV2 {

// Carves every curve that carries a road assignment, in ONE solve.
//
// This is the node that removes the graph explosion: Road Carve needs a Curve
// pin, so ten road segments meant ten Curve Input nodes feeding ten Road Carve
// nodes. Nothing in the graph knew a curve was a road. The assignment registry
// answers that, so the network node needs no Curve pin at all - it reads the
// registry and pulls the matching immutable snapshots the terrain context
// already captured.
class TerrainRoadNetworkNode final : public TerrainNodeBase {
public:
    TerrainRoadNetworkNode();
    std::string getTypeId() const override { return "TerrainV2.RoadNetwork"; }
    NodeSystem::PinValue compute(int outputIndex,
                                 NodeSystem::EvaluationContext& ctx) override;
    void drawContent() override;
    float getCustomWidth() const override { return 210.0f; }

    // The last solve, or null when the graph has not produced one. The optional
    // road surface mesh is built from THIS - the very route that carved the
    // terrain. Re-sampling the curve for the mesh would drift from the fields on
    // exactly the bends where alignment matters most.
    std::shared_ptr<const RoadCarveResult> latestResult() const { return cachedResult_; }

private:
    uint64_t cachedKey_ = 0;
    uint64_t nextRevision_ = 1;
    std::shared_ptr<const RoadCarveResult> cachedResult_;
    // Reported in the panel. A road whose curve vanished must be visible: a
    // segment that silently stops grading is the failure nobody files.
    std::string lastStatus_;
    std::string lastMissingCurves_;
};

} // namespace TerrainNodesV2
