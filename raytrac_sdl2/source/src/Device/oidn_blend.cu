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
