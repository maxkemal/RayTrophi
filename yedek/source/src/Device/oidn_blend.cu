#include "oidn_blend_cuda.h"

namespace {

__global__ void oidnBlendKernel(float* __restrict__ denoisedPackedFloat3,
                                const unsigned char* __restrict__ originalColorBase,
                                size_t pixelCount,
                                float blend,
                                size_t originalPixelByteStride) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= pixelCount) return;

    const float* original = reinterpret_cast<const float*>(originalColorBase + idx * originalPixelByteStride);
    float* denoised = denoisedPackedFloat3 + idx * 3;
    const float invBlend = 1.0f - blend;

    denoised[0] = denoised[0] * blend + original[0] * invBlend;
    denoised[1] = denoised[1] * blend + original[1] * invBlend;
    denoised[2] = denoised[2] * blend + original[2] * invBlend;
}

} // namespace

bool launchOidnBlendKernel(float* denoisedPackedFloat3Dev,
                           const void* originalColorFloat4Dev,
                           size_t pixelCount,
                           float blend,
                           size_t originalPixelByteStride,
                           cudaStream_t stream) {
    if (!denoisedPackedFloat3Dev || !originalColorFloat4Dev || pixelCount == 0) {
        return false;
    }

    constexpr int kBlockSize = 256;
    const int blocks = static_cast<int>((pixelCount + static_cast<size_t>(kBlockSize) - 1) / static_cast<size_t>(kBlockSize));
    oidnBlendKernel<<<blocks, kBlockSize, 0, stream>>>(
        denoisedPackedFloat3Dev,
        static_cast<const unsigned char*>(originalColorFloat4Dev),
        pixelCount,
        blend,
        originalPixelByteStride);

    return cudaGetLastError() == cudaSuccess;
}

namespace {

__global__ void vulkanDenoiserPrepKernel(float4* __restrict__ dst,
                                         const float4* __restrict__ src,
                                         int width,
                                         int height,
                                         bool decodeNormal) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int srcIdx = y * width + x;
    const int dstIdx = (height - 1 - y) * width + x;

    float4 s = src[srcIdx];
    if (decodeNormal) {
        s.x = s.x * 2.0f - 1.0f;
        s.y = s.y * 2.0f - 1.0f;
        s.z = s.z * 2.0f - 1.0f;
    }
    s.w = 1.0f;
    dst[dstIdx] = s;
}

} // namespace

bool launchVulkanDenoiserPrepKernel(void* dstFloat4Dev,
                                    const void* srcFloat4Dev,
                                    int width,
                                    int height,
                                    bool decodeNormal,
                                    cudaStream_t stream) {
    if (!dstFloat4Dev || !srcFloat4Dev || width <= 0 || height <= 0) return false;

    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);
    vulkanDenoiserPrepKernel<<<grid, block, 0, stream>>>(
        static_cast<float4*>(dstFloat4Dev),
        static_cast<const float4*>(srcFloat4Dev),
        width, height, decodeNormal);
    return cudaGetLastError() == cudaSuccess;
}

namespace {

__global__ void oidnTonemapKernel(const float* __restrict__ hdrFloat3,
                                  uint32_t* __restrict__ packedDst,
                                  int width, int height,
                                  float exposure,
                                  uint32_t aMaskOr,
                                  int rShift, int gShift, int bShift,
                                  bool flipY) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int srcIdx = (y * width + x) * 3;
    float r = fmaxf(hdrFloat3[srcIdx + 0] * exposure, 0.0f);
    float g = fmaxf(hdrFloat3[srcIdx + 1] * exposure, 0.0f);
    float b = fmaxf(hdrFloat3[srcIdx + 2] * exposure, 0.0f);

    // Reinhard
    r = r / (1.0f + r);
    g = g / (1.0f + g);
    b = b / (1.0f + b);

    // sRGB approximation matching the displaced CPU loop (Main.cpp).
    r = __powf(r, 1.0f / 2.2f);
    g = __powf(g, 1.0f / 2.2f);
    b = __powf(b, 1.0f / 2.2f);

    const uint32_t ri = static_cast<uint32_t>(fminf(r, 1.0f) * 255.0f + 0.5f);
    const uint32_t gi = static_cast<uint32_t>(fminf(g, 1.0f) * 255.0f + 0.5f);
    const uint32_t bi = static_cast<uint32_t>(fminf(b, 1.0f) * 255.0f + 0.5f);

    const int dstY = flipY ? (height - 1 - y) : y;
    const int dstIdx = dstY * width + x;
    packedDst[dstIdx] = aMaskOr
                      | (ri << rShift)
                      | (gi << gShift)
                      | (bi << bShift);
}

} // namespace

bool launchOidnTonemapKernel(const float* hdrFloat3Dev,
                             void* packedDstDev,
                             int width, int height,
                             float exposure,
                             uint32_t aMaskOr,
                             int rShift, int gShift, int bShift,
                             bool flipY,
                             cudaStream_t stream) {
    if (!hdrFloat3Dev || !packedDstDev || width <= 0 || height <= 0) return false;

    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);
    oidnTonemapKernel<<<grid, block, 0, stream>>>(
        hdrFloat3Dev,
        static_cast<uint32_t*>(packedDstDev),
        width, height,
        exposure,
        aMaskOr,
        rShift, gShift, bShift,
        flipY);
    return cudaGetLastError() == cudaSuccess;
}
