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

// Field names/order mirror RtPostParams (source/shaders/post_chain.glsl) and
// TonemapPush (VulkanBackend.cpp) 1:1 on purpose — this IS the same transform,
// third language. See post_chain.glsl's header comment: the project already
// paid once for three shaders disagreeing on tonemap/sRGB; this struct exists
// so the denoiser's re-tonemap can't become a fourth, silently-different copy.
// New operator or grade step -> add it in post_chain.glsl's rtApplyPost FIRST,
// then port the same change here.
struct OidnPostParams {
    float exposure          = 1.0f;   // post.* color-grading exposure (postExposure)
    float cameraExposure    = 1.0f;   // physical ISO/shutter/aperture triangle (postCameraExposure)
    float gamma              = 1.0f;
    float saturation         = 1.0f;
    float colorTemperature   = 6500.0f;
    float vignetteStrength   = 0.0f;
    uint32_t toneMapping     = 4u;    // ToneMappingType: 0 AGX,1 ACES,2 Uncharted,3 Filmic,4 None(=Reinhard)
    uint32_t vignetteEnabled = 0u;
    uint32_t aMaskOr         = 0u;
    int rShift = 0, gShift = 8, bShift = 16;
    bool flipY = true;
};

// Fuses the FULL post_chain.glsl transform (exposure -> tonemap operator ->
// color temperature -> saturation -> gamma -> vignette -> sRGB encode) +
// pack-to-uint32 in the SDL_Surface channel layout. Replaces the per-pixel
// CPU std::pow loop and shrinks the D2H transfer from float3 (12 B/px) to
// packed RGBA8 (4 B/px).
//
//   hdrFloat3Dev      OIDN output, packed float3, row-major, top-down.
//   packedDstDev      uint32-per-pixel output, row-major. If flipY=true, row
//                     y of the source lands at row (height-1-y) of the dst.
//   p                 see OidnPostParams above.
bool launchOidnTonemapKernel(const float* hdrFloat3Dev,
                             void* packedDstDev,
                             int width, int height,
                             const OidnPostParams& p,
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

void launchPostHistogram(const float*, int width, int height, int stride, cudaStream_t);
bool launchOptixDisplayPost(const void*, void*, int width, int height, cudaStream_t, float lensAmount, float lensFalloff);
