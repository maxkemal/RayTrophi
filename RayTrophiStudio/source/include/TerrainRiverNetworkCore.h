#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace TerrainHydrology {

    struct RiverNetworkParams {
        float legacyAccumulationThreshold = 0.0015f;
        float minimumCatchmentAreaSquareMeters = 5000.0f;
        int minimumBranchLength = 8;
        bool accumulationIsPhysicalArea = false;
    };

    struct RiverNetworkResult {
        std::vector<int> parent;
        std::vector<int> streamOrder;
        std::vector<uint8_t> active;
        std::vector<float> channelStrength;
        std::vector<float> normalizedStreamOrder;
        std::vector<float> sources;
        int maximumStreamOrder = 1;
    };

    // Accepts both the legacy scalar D8 encoding and an interleaved XY vector
    // field. Vector directions are snapped to the nearest D8 receiver so all
    // river consumers share the same downstream graph.
    int decodeFlowDirection(const std::vector<float>& values,
                            int channels,
                            std::size_t pixelIndex);

    bool buildRiverNetwork(int width,
                           int height,
                           const std::vector<float>& accumulation,
                           int accumulationChannels,
                           const std::vector<float>& direction,
                           int directionChannels,
                           const std::vector<float>* catchmentArea,
                           const std::vector<float>* lakeMask,
                           const std::vector<float>* lakeSpillPoints,
                           // Cells that must never be NAMED a river channel:
                           // authored infrastructure. A carved road is a linear
                           // depression and therefore a perfect channel from
                           // this function's point of view, which is how roads
                           // came to be classified as river beds.
                           //
                           // It excludes only the CLASSIFICATION. Flow direction,
                           // parentage and accumulation are untouched, so the
                           // water an excluded cell receives still reaches the
                           // same downstream cells. Cutting the routing instead
                           // would be the plausible-looking failure: no river on
                           // the road, and a downstream basin quietly underfed.
                           const std::vector<float>* channelExclusion,
                           const RiverNetworkParams& params,
                           RiverNetworkResult& result,
                           std::string* error = nullptr);

} // namespace TerrainHydrology
