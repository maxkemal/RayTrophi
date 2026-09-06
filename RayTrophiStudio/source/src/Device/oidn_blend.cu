#include "PostProcess/ColorMath.h"
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

// ── post_chain.glsl ported to CUDA, 1:1 ─────────────────────────────────────
// Keep this block in the same order as source/shaders/post_chain.glsl's
// operators/rtApplyPost. See OidnPostParams in oidn_blend_cuda.h for why this
// duplication (a third language, not a fourth divergent implementation) is
// deliberate and must stay math-identical to the GLSL.

// Plain device-side mirror of OidnPostParams (host struct, see
// oidn_blend_cuda.h) — kept separate so this .cu doesn't need the host
// header's non-CUDA default-member-initializer syntax inside a __device__
// context.
struct OidnPostParamsDevice {
    float exposure, cameraExposure, gamma, saturation, colorTemperature, vignetteStrength;
    uint32_t toneMapping, vignetteEnabled;
};

__device__ __forceinline__ float rtLinearToSRGB(float c) {
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    return (c <= 0.0031308f) ? (12.92f * c) : (1.055f * __powf(c, 1.0f / 2.4f) - 0.055f);
}

__device__ __forceinline__ float3 rtLinearToSRGB(float3 c) {
    return make_float3(rtLinearToSRGB(c.x), rtLinearToSRGB(c.y), rtLinearToSRGB(c.z));
}

__device__ __forceinline__ float rtSanitizeChannel(float v) {
    if (isnan(v)) return 0.0f;
    if (isinf(v)) return v > 0.0f ? 65504.0f : 0.0f;
    return fmaxf(v, 0.0f);
}

// uv01: pixel's 0..1 screen position. Vignette matches post_chain.glsl /
// the CPU applyVignette formula: u=(x/w-0.5)*2, falloff = 1-strength*(u^2+v^2).
__device__ __forceinline__ float3 rtApplyPost(float3 linearColor, const OidnPostParamsDevice& p,
                                              float u01, float v01) {
    float3 c = make_float3(rtSanitizeChannel(linearColor.x), rtSanitizeChannel(linearColor.y),
                            rtSanitizeChannel(linearColor.z));
    const float ex = p.exposure * p.cameraExposure;
    c = make_float3(c.x * ex, c.y * ex, c.z * ex);

    auto graded=rtPcGrade(RtColor(c.x,c.y,c.z),int(p.toneMapping),p.colorTemperature,p.saturation,p.gamma);
    c=make_float3(graded.x,graded.y,graded.z);

    if (p.vignetteEnabled != 0u && p.vignetteStrength > 0.0f) {
        const float uu = (u01 - 0.5f) * 2.0f;
        const float vv = (v01 - 0.5f) * 2.0f;
        const float falloff = fminf(fmaxf(1.0f - p.vignetteStrength * (uu * uu + vv * vv), 0.0f), 1.0f);
        c = make_float3(c.x * falloff, c.y * falloff, c.z * falloff);
    }

    return rtLinearToSRGB(c);
}

__global__ void oidnTonemapKernel(const float* __restrict__ hdrFloat3,
                                  uint32_t* __restrict__ packedDst,
                                  int width, int height,
                                  OidnPostParamsDevice p,
                                  uint32_t aMaskOr,
                                  int rShift, int gShift, int bShift,
                                  bool flipY) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int srcIdx = (y * width + x) * 3;
    const float3 hdr = make_float3(hdrFloat3[srcIdx + 0], hdrFloat3[srcIdx + 1], hdrFloat3[srcIdx + 2]);

    const float u01 = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
    const float v01 = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
    const float3 srgb = rtApplyPost(hdr, p, u01, v01);

    const uint32_t ri = static_cast<uint32_t>(fminf(srgb.x, 1.0f) * 255.0f + 0.5f);
    const uint32_t gi = static_cast<uint32_t>(fminf(srgb.y, 1.0f) * 255.0f + 0.5f);
    const uint32_t bi = static_cast<uint32_t>(fminf(srgb.z, 1.0f) * 255.0f + 0.5f);

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
                             const OidnPostParams& p,
                             cudaStream_t stream) {
    if (!hdrFloat3Dev || !packedDstDev || width <= 0 || height <= 0) return false;

    launchPostHistogram(hdrFloat3Dev, width, height, 3, stream);
    OidnPostParamsDevice dp{ p.exposure, p.cameraExposure, p.gamma, p.saturation,
                             p.colorTemperature, p.vignetteStrength,
                             p.toneMapping, p.vignetteEnabled };

    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);
    oidnTonemapKernel<<<grid, block, 0, stream>>>(
        hdrFloat3Dev,
        static_cast<uint32_t*>(packedDstDev),
        width, height,
        dp,
        p.aMaskOr,
        p.rShift, p.gShift, p.bShift,
        p.flipY);
    return cudaGetLastError() == cudaSuccess;
}

#include "post_exposure.cuh"
