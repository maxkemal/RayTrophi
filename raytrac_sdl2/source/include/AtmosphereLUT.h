#pragma once
#include <cuda_runtime.h>
#include "World.h"

// LUT Resolutions (Optimized for quality/performance)
// 32-bit Float format ensures no banding (Preserving Quality)
#define TRANSMITTANCE_LUT_W 256
#define TRANSMITTANCE_LUT_H 64

#define SKYVIEW_LUT_W 256
#define SKYVIEW_LUT_H 128

#define MULTI_SCATTER_LUT_RES 32
#define AERIAL_PERSPECTIVE_RES 32

#ifndef __CUDACC__
class AtmosphereLUT {
public:
    AtmosphereLUT();
    ~AtmosphereLUT();

    void initialize();
    
    // Triggered when Nishita parameters change
    void precompute(const NishitaSkyParams& params);

    AtmosphereLUTData getGPUData() const { return data; }

    bool is_initialized() const { return initialized; }

private:
    void cleanup();
    
    // CUDA resources
    cudaArray_t transmittance_array = nullptr;
    cudaArray_t skyview_array = nullptr;
    cudaArray_t multi_scatter_array = nullptr;
    cudaArray_t aerial_perspective_array = nullptr; // 3D Array
    
    AtmosphereLUTData data;
    bool initialized = false;
};
#endif
