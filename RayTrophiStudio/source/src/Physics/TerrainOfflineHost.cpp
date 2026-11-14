#include "TerrainManager.h"
#include "TerrainErosionGpuTypes.h"

void TerrainManager::initCuda() {
    cudaInitialized = false;
}

void TerrainManager::hydraulicErosionGPUCuda(TerrainObject* terrain,
                                             const HydraulicErosionParams& params,
                                             const std::vector<float>& mask) {
    hydraulicErosion(terrain, params, mask);
}

void TerrainManager::thermalErosionGPUCuda(TerrainObject* terrain, const ThermalErosionParams& p,
                                           const std::vector<float>& mask) {
    thermalErosion(terrain, p, mask);
}

void TerrainManager::windErosionGPUCuda(TerrainObject* terrain, float strength, float direction,
                                        int iterations, const std::vector<float>& mask) {
    windErosion(terrain, strength, direction, iterations, mask);
}

void TerrainManager::fluvialErosionGPUCuda(TerrainObject* terrain, const HydraulicErosionParams& p,
                                           const std::vector<float>& mask,
                                           const std::vector<float>& flowGuide,
                                           TerrainPhysics::StreamPowerParamsGPU sp, int numPasses,
                                           const std::function<void()>& updateFlowCPU) {
    fluvialErosion(terrain, p, mask, std::function<void(float)>{}, flowGuide);
}

void TerrainManager::updateFoliage(TerrainObject* terrain, OptixWrapper* optix) {
    // Legacy OptiX foliage is not linked; common InstanceGroups remain available.
}

void TerrainManager::clearFoliage(TerrainObject* terrain, OptixWrapper* optix) {
    // Legacy OptiX foliage is not linked; common InstanceGroups remain available.
}

void TerrainManager::reapplyAllFoliage(OptixWrapper* optix) {
    // Legacy OptiX foliage is not linked; common InstanceGroups remain available.
}
