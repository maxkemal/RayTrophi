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
                           const RiverNetworkParams& params,
                           RiverNetworkResult& result,
                           std::string* error = nullptr);

} // namespace TerrainHydrology
