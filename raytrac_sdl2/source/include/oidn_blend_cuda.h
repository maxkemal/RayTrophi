#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

bool launchOidnBlendKernel(float* denoisedPackedFloat3Dev,
                           const void* originalColorFloat4Dev,
                           size_t pixelCount,
                           float blend,
                           size_t originalPixelByteStride,
                           cudaStream_t stream);

// Fuses Reinhard tonemap + sRGB-approx (gamma 2.2) encode + pack-to-uint32 in
// the SDL_Surface channel layout. Replaces the per-pixel CPU std::pow loop and
// shrinks the D2H transfer from float3 (12 B/px) to packed RGBA8 (4 B/px).
//
//   hdrFloat3Dev      OIDN output, packed float3, row-major, top-down.
//   packedDstDev      uint32-per-pixel output, row-major. If flipY=true, row
//                     y of the source lands at row (height-1-y) of the dst.
//   exposure          linear pre-tonemap scale (auto-exposure / EV / physical).
//   aMaskOr           OR'd into every pixel (use the SDL Amask to leave alpha
//                     at "fully opaque" per SDL's convention).
//   rShift,gShift,bShift  SDL_PixelFormat channel shifts.
//   flipY             true to mirror Y while packing (matches the original CPU
//                     loop's screen_y = h-1-y indexing).
bool launchOidnTonemapKernel(const float* hdrFloat3Dev,
                             void* packedDstDev,
                             int width, int height,
                             float exposure,
                             uint32_t aMaskOr,
                             int rShift, int gShift, int bShift,
                             bool flipY,
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
