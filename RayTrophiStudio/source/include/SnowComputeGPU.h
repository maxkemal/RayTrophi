#pragma once

#include <string>
#include <vector>

namespace TerrainNodesV2 {

struct SnowGpuParams {
    int width = 0;
    int height = 0;
    int settlePasses = 0;
    int runoffPasses = 0;
    int softSnowPasses = 0;
    int geometryRelaxPasses = 0;
    int maximumTransportStride = 1;
    int windDx = 0;
    int windDy = 0;
    float cellSize = 1.0f;
    float maxDepth = 1.0f;
    float transportRate = 0.3f;
    float slipAngleDegrees = 38.0f;
    float windStrength = 0.0f;
    float meltAmount = 0.0f;
    float solarMelt = 0.0f;
    float refreezeRate = 0.0f;
    float effectiveSlipTangent = 0.0f;
    float geometryMaxStep = 1.0f;
    float relaxationHeightLimit = 1.0f;
    float geometryDepthCap = 1.0f;
    float geometrySurfaceWindow = 1.0f;
};

// Runs the iterative, bandwidth-heavy part of Snow Layer on the shared Vulkan
// compute backend. Climate/exposure initialization remains on the CPU because
// it includes an integral-image valley query and is only one O(N) pass.
// Returns false without modifying the host fields when compute is unavailable.
bool solveSnowOnGpu(const SnowGpuParams& params,
                    const std::vector<float>& baseMeters,
                    const std::vector<float>& coldness,
                    const std::vector<float>& solar,
                    const std::vector<float>& windExposure,
                    std::vector<float>& snowDepth,
                    std::vector<float>& iceDepth,
                    std::vector<float>& water,
                    std::vector<float>& runoffTrace,
                    std::vector<float>& avalancheDeposit,
                    std::vector<float>& geometryDepth,
                    std::string* failureReason = nullptr);

} // namespace TerrainNodesV2
