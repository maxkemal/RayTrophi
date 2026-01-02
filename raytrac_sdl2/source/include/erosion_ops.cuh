#pragma once

#include <cuda_runtime.h>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace TerrainPhysics {

struct HydraulicErosionParamsGPU {
    int mapWidth;
    int mapHeight;
    int brushRadius;
    int dropletLifetime;
    float inertia;
    float sedimentCapacity;
    float minSlope;
    float erodeSpeed;
    float depositSpeed;
    float evaporateSpeed;
    float gravity;
    unsigned int seed;
    float* hardnessMap;  // Optional hardness map (nullptr if not used)
};

struct ThermalErosionParamsGPU {
    int mapWidth;
    int mapHeight;
    float talusAngle; // Tangent of angle (height diff / width)
    float erosionAmount;
    bool useHardness;
    float* hardnessMap;  // Optional hardness map (nullptr if not used)
};

// Post-processing parameters for pit filling, spike removal, edge preservation
struct PostProcessParamsGPU {
    int mapWidth;
    int mapHeight;
    float cellSize;
    float pitThreshold;     // Threshold for pit detection (0.05f * cellSize)
    float spikeThreshold;   // Threshold for spike detection (0.1f * cellSize)
    int edgeFadeWidth;      // Width of edge fade zone (w / 40)
};

struct FluvialErosionParamsGPU {
    int mapWidth;
    int mapHeight;
    float fixedDeltaTime;
    float pipeLength;
    float cellSize;
    float erosionRate;
    float depositionRate;
    float evaporationRate;
    float gravity;
    float sedimentCapacityConstant;
};

struct WindErosionParamsGPU {
    int mapWidth;
    int mapHeight;
    float windDirX;
    float windDirY;
    float strength;
    float suspensionRate;
    float depositionRate;
};

}
