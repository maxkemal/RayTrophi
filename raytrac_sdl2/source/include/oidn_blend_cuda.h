#pragma once

#include <cstddef>
#include <cuda_runtime.h>

bool launchOidnBlendKernel(float* denoisedPackedFloat3Dev,
                           const void* originalColorFloat4Dev,
                           size_t pixelCount,
                           float blend,
                           size_t originalPixelByteStride,
                           cudaStream_t stream);

// Prepares a Vulkan-produced float4 AOV buffer for OIDN consumption.
// - Reads src as tightly-packed float4 (16 B / pixel) in Vulkan storage order.
// - Writes dst as tightly-packed float4 (16 B / pixel), Y-flipped to match the
//   CPU denoiser path that feeds OIDN in display orientation.
// - If decodeNormal=true, .rgb is remapped from [0,1] to [-1,1] (Vulkan raygen
//   stores normals encoded); otherwise passed through.
// - Alpha channel (sample-count) is ignored by OIDN, so dst.a is written as 1.
bool launchVulkanDenoiserPrepKernel(void* dstFloat4Dev,
                                    const void* srcFloat4Dev,
                                    int width,
                                    int height,
                                    bool decodeNormal,
                                    cudaStream_t stream);
