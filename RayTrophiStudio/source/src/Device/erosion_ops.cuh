#pragma once

#include <cuda_runtime.h>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#include "TerrainErosionGpuTypes.h"
