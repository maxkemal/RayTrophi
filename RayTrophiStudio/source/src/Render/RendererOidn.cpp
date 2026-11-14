// Offline denoiser implementation extracted from Renderer.cpp.
// SDK and CUDA dependencies stay here; Renderer.h contains only an opaque state.
#include "Renderer.h"
#include "Backend/IBackend.h"
#include "oidn_blend_cuda.h"
#include <OpenImageDenoise/oidn.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <string>

static_assert(Renderer::kOidnQualityDefault == OIDN_QUALITY_DEFAULT);
static_assert(Renderer::kOidnQualityFast == OIDN_QUALITY_FAST);

struct Renderer::OidnState {
    // Keep the original SDK reference destruction order: buffers, filter, device.
    oidn::DeviceRef oidnDevice;
    oidn::FilterRef oidnFilter;
    oidn::BufferRef oidnColorBuffer;
    oidn::BufferRef oidnAlbedoBuffer;
    oidn::BufferRef oidnNormalBuffer;
    oidn::BufferRef oidnOutputBuffer;
};

void Renderer::OidnStateDeleter::operator()(OidnState* state) const {
    delete state;
}

Renderer::OidnStatePtr Renderer::createOidnState() {
    return OidnStatePtr(new OidnState());
}

bool Renderer::isCudaAvailable() {
    try {
        oidn::DeviceRef testDevice = oidn::newDevice(oidn::DeviceType::CUDA);
        testDevice.commit();
        return true; // CUDA destekleniyor
    }
    catch (const std::exception& e) {
        return false; // CUDA desteklenmiyor
    }
}
// Helper to initialize OIDN device once
void Renderer::initOIDN() {
    if (oidnInitialized) return;

    // Device-selection cascade for the host (CPU-visible buffer) denoise path.
    // The denoise itself runs on whichever device we bind here; for a GPU device
    // OIDN handles the host<->device buffer copies internally, so AMD/Intel cards
    // with Vulkan-RT but no CUDA still get a GPU-accelerated denoise instead of
    // the much slower CPU path. Priority:
    //   1. CUDA — NVIDIA (the only backend that also feeds the zero-copy interop)
    //   2. HIP  — AMD GPUs
    //   3. SYCL — Intel Arc / Xe GPUs
    //   4. CPU  — universal fallback
    // OIDN ships a separate device DLL per backend; a missing DLL (or absent
    // runtime, e.g. cudart on an AMD box) just makes newDevice()/commit() fail,
    // so each attempt is wrapped and we fall through to the next.
    auto tryDevice = [this](oidn::DeviceType type, const char* label) -> bool {
        try {
            oidn::DeviceRef dev = oidn::newDevice(type);
            dev.commit();
            const char* errMsg = nullptr;
            if (dev.getError(errMsg) != oidn::Error::None) {
                SCENE_LOG_WARN(std::string("[OIDN] ") + label + " commit failed: "
                               + (errMsg ? errMsg : "unknown error"));
                return false;
            }
            oidnState->oidnDevice = dev;
            oidnInitialized = true;
            SCENE_LOG_INFO(std::string("[OIDN] Initialized with ") + label + ".");
            return true;
        }
        catch (const std::exception& e) {
            SCENE_LOG_WARN(std::string("[OIDN] ") + label + " unavailable: " + e.what());
            return false;
        }
    };

    if (tryDevice(oidn::DeviceType::CUDA, "CUDA")) return;
    if (tryDevice(oidn::DeviceType::HIP,  "HIP (AMD)")) return;
    if (tryDevice(oidn::DeviceType::SYCL, "SYCL (Intel)")) return;
    if (tryDevice(oidn::DeviceType::CPU,  "CPU")) return;

    SCENE_LOG_ERROR("[OIDN] No usable denoiser device (CUDA/HIP/SYCL/CPU all failed).");
}

void Renderer::applyOIDNDenoising(SDL_Surface* surface, int numThreads, bool denoise, float blend) {
    if (!surface) return;
    const int w = surface->w;
    const int h = surface->h;
    const size_t pixelCount = (size_t)w * (size_t)h;
    std::vector<float> linearColor(pixelCount * 3);

    const SDL_PixelFormat* fmt = surface->format;
    Uint32* pixels = static_cast<Uint32*>(surface->pixels);
    const Uint32 rMask = fmt->Rmask, gMask = fmt->Gmask, bMask = fmt->Bmask, aMask = fmt->Amask;
    const Uint8 rShift = fmt->Rshift, gShift = fmt->Gshift, bShift = fmt->Bshift;
    const float inv255 = 1.0f / 255.0f;

    // Parallel sRGB→linear decode (gamma-2 approximation: x*x)
    {
        float* __restrict dst = linearColor.data();
        Uint32* __restrict pxBase = pixels;
        std::for_each_n(std::execution::par_unseq,
            pxBase, pixelCount,
            [=](Uint32& px) {
                const size_t i = static_cast<size_t>(&px - pxBase);
                float r = ((px & rMask) >> rShift) * inv255;
                float g = ((px & gMask) >> gShift) * inv255;
                float b = ((px & bMask) >> bShift) * inv255;
                const size_t idx = i * 3;
                dst[idx]     = r * r;
                dst[idx + 1] = g * g;
                dst[idx + 2] = b * b;
            });
    }

    OIDNFrameData frame;
    frame.width = w;
    frame.height = h;
    frame.color = linearColor.data();

    std::vector<float> denoised;
    if (!applyOIDNDenoising(frame, blend, denoised)) {
        return;
    }

    // Parallel linear→sRGB encode + pack back into SDL surface
    //
    // [GAMMA FIX] The decode block above used r = (x/255)^2 (gamma-2
    // approximation of sRGB→linear). The original encode here clamped+
    // packed the *linear* value straight into the 8-bit pixel WITHOUT
    // applying the inverse — so the saved PNG (and viewport display) was
    // treated as sRGB while actually carrying linear values, causing the
    // entire image to read significantly darker than the pre-denoise
    // surface. Apply the matching inverse sqrt so the roundtrip is
    // gamma-neutral.
    {
        const float* __restrict src = denoised.data();
        Uint32* __restrict pxBase = pixels;
        std::for_each_n(std::execution::par_unseq,
            pxBase, pixelCount,
            [=](Uint32& px) {
                const size_t i = static_cast<size_t>(&px - pxBase);
                const size_t idx = i * 3;
                float r = std::max(src[idx],     0.0f);
                float g = std::max(src[idx + 1], 0.0f);
                float b = std::max(src[idx + 2], 0.0f);

                // Inverse of the gamma-2 decode (r = x*x). Matches the
                // approximation used at line 607 so denoiser-on output
                // visually matches denoiser-off.
                r = std::sqrt(std::min(r, 1.0f));
                g = std::sqrt(std::min(g, 1.0f));
                b = std::sqrt(std::min(b, 1.0f));

                Uint8 ri = static_cast<Uint8>(r * 255.0f + 0.5f);
                Uint8 gi = static_cast<Uint8>(g * 255.0f + 0.5f);
                Uint8 bi = static_cast<Uint8>(b * 255.0f + 0.5f);

                Uint32 alpha = px & aMask;
                px = alpha
                    | ((Uint32)ri << rShift)
                    | ((Uint32)gi << gShift)
                    | ((Uint32)bi << bShift);
            });
    }
}

bool Renderer::applyOIDNDenoising(const OIDNFrameData& frame, float blend, std::vector<float>& output,
                                  int quality) {
    if (!frame.color || frame.width <= 0 || frame.height <= 0) return false;
    std::lock_guard<std::mutex> lock(oidnMutex);

    // If we were previously bound to an external CUDA stream (GPU zero-copy path on OptiX),
    // that stream may now be destroyed (e.g., after switching OptiX→Vulkan). Using the stale
    // device from the host path crashes inside nvcuda64.dll during execute/write. Drop the
    // bound device and let initOIDN() create a fresh, self-managed CUDA or CPU device.
    if (oidnCudaInitialized) {
        oidnState->oidnColorBuffer  = oidn::BufferRef();
        oidnState->oidnAlbedoBuffer = oidn::BufferRef();
        oidnState->oidnNormalBuffer = oidn::BufferRef();
        oidnState->oidnOutputBuffer = oidn::BufferRef();
        oidnState->oidnFilter       = oidn::FilterRef();
        oidnState->oidnDevice       = oidn::DeviceRef();
        oidnInitialized     = false;
        oidnCudaInitialized = false;
        oidnCudaOrdinal = -1;
        oidnCudaStream  = nullptr;
        if (oidnGpuOutputDevPtr) {
            cudaFree(oidnGpuOutputDevPtr);
            oidnGpuOutputDevPtr = nullptr;
        }
        oidnGpuOutputBytes = 0;
        if (oidnGpuPackedDevPtr) {
            cudaFree(oidnGpuPackedDevPtr);
            oidnGpuPackedDevPtr = nullptr;
        }
        oidnGpuPackedBytes = 0;
        oidnCachedWidth = 0;
        oidnCachedHeight = 0;
        oidnUsingAlbedo = false;
        oidnUsingNormal = false;
        oidnGpuCachedColor  = nullptr;
        oidnGpuCachedAlbedo = nullptr;
        oidnGpuCachedNormal = nullptr;
    }

    if (!oidnInitialized) {
        initOIDN();
        if (!oidnInitialized) return false;
    }

    const int width = frame.width;
    const int height = frame.height;
    size_t pixelCount = (size_t)width * height;
    size_t bufferSize = pixelCount * 3;  // Float3
    const bool useAlbedo = frame.albedo != nullptr;
    const bool useNormal = frame.normal != nullptr;
    const bool filterLayoutChanged =
        (useAlbedo != oidnUsingAlbedo) ||
        (useNormal != oidnUsingNormal);
    bool sizeChanged = (width != oidnCachedWidth || height != oidnCachedHeight);
    const bool qualityChanged = (quality != oidnGpuCachedQuality);
    if (sizeChanged || filterLayoutChanged || qualityChanged) {
        try {
            oidnState->oidnColorBuffer = oidnState->oidnDevice.newBuffer(bufferSize * sizeof(float));
            oidnState->oidnAlbedoBuffer = useAlbedo ? oidnState->oidnDevice.newBuffer(bufferSize * sizeof(float)) : oidn::BufferRef();
            oidnState->oidnNormalBuffer = useNormal ? oidnState->oidnDevice.newBuffer(bufferSize * sizeof(float)) : oidn::BufferRef();
            oidnState->oidnOutputBuffer = oidnState->oidnDevice.newBuffer(bufferSize * sizeof(float));

            oidnState->oidnFilter = oidnState->oidnDevice.newFilter("RT");
            oidnState->oidnFilter.setImage("color", oidnState->oidnColorBuffer,
                oidn::Format::Float3, width, height);
            if (useAlbedo) {
                oidnState->oidnFilter.setImage("albedo", oidnState->oidnAlbedoBuffer,
                    oidn::Format::Float3, width, height);
            }
            if (useNormal) {
                oidnState->oidnFilter.setImage("normal", oidnState->oidnNormalBuffer,
                    oidn::Format::Float3, width, height);
            }
            oidnState->oidnFilter.setImage("output", oidnState->oidnOutputBuffer,
                oidn::Format::Float3, width, height);
            oidnState->oidnFilter.set("hdr", true);
            oidnState->oidnFilter.set("srgb", false);
            oidnState->oidnFilter.set("quality", quality);
            oidnState->oidnFilter.commit();

            oidnGpuCachedQuality = quality;
            oidnCachedWidth = width;
            oidnCachedHeight = height;
            oidnUsingAlbedo = useAlbedo;
            oidnUsingNormal = useNormal;
        }
        catch (const std::exception& e) {
            SCENE_LOG_ERROR(std::string("[OIDN] Buffer creation failed: ") + e.what());
            return false;
        }
    }

    output.resize(bufferSize);

    try {
        // Direct upload from caller's contiguous float3 buffer — no staging copy
        oidnState->oidnColorBuffer.write(0, bufferSize * sizeof(float), frame.color);
        if (useAlbedo) {
            oidnState->oidnAlbedoBuffer.write(0, bufferSize * sizeof(float), frame.albedo);
        }
        if (useNormal) {
            oidnState->oidnNormalBuffer.write(0, bufferSize * sizeof(float), frame.normal);
        }
        oidnState->oidnFilter.execute();

        const char* errMsg;
        if (oidnState->oidnDevice.getError(errMsg) != oidn::Error::None) {
            SCENE_LOG_ERROR(std::string("[OIDN] ") + errMsg);
            return false;
        }

        // Download straight into output — no intermediate vector
        oidnState->oidnOutputBuffer.read(0, bufferSize * sizeof(float), output.data());
    }
    catch (const std::exception& e) {
        SCENE_LOG_ERROR(std::string("[OIDN] Execution failed: ") + e.what());
        return false;
    }

    // Blend in-place with caller's color as the "original" reference
    if (blend < 0.999f) {
        const float blend_inv = 1.0f - blend;
        const float* __restrict src = frame.color;
        float* __restrict dst = output.data();
        std::for_each_n(std::execution::par_unseq,
            dst, bufferSize,
            [=](float& d) {
                const size_t i = static_cast<size_t>(&d - dst);
                d = d * blend + src[i] * blend_inv;
            });
    }

    return true;
}

bool Renderer::applyOIDNDenoisingToCPUAccumulation(float blend, bool useAuxiliary) {
    const size_t pixelCount = (size_t)image_width * (size_t)image_height;
    if (!cpu_accumulation_valid || cpu_accumulation_buffer.size() != pixelCount || pixelCount == 0) {
        cpu_denoised_valid = false;
        return false;
    }

    std::vector<float> linearColor(pixelCount * 3);
    std::vector<float> linearAlbedo;
    std::vector<float> linearNormal;
    const bool hasAlbedo = useAuxiliary && cpu_albedo_accumulation_buffer.size() == pixelCount;
    const bool hasNormal = useAuxiliary && cpu_normal_accumulation_buffer.size() == pixelCount;
    if (hasAlbedo) linearAlbedo.resize(pixelCount * 3);
    if (hasNormal) linearNormal.resize(pixelCount * 3);

    {
        const int W = image_width;
        const int H = image_height;
        const Vec4* __restrict accSrc = cpu_accumulation_buffer.data();
        const Vec4* __restrict albSrc = hasAlbedo ? cpu_albedo_accumulation_buffer.data() : nullptr;
        const Vec4* __restrict nrmSrc = hasNormal ? cpu_normal_accumulation_buffer.data() : nullptr;
        float* __restrict colDst = linearColor.data();
        float* __restrict albDst = hasAlbedo ? linearAlbedo.data() : nullptr;
        float* __restrict nrmDst = hasNormal ? linearNormal.data() : nullptr;
        // Iterate over source pixels — pointer range is a valid parallel iterator.
        // Y-flip is computed per element from index arithmetic.
        std::for_each_n(std::execution::par_unseq,
            accSrc, pixelCount,
            [=](const Vec4& sRef) {
                const size_t srcIdx = static_cast<size_t>(&sRef - accSrc);
                const int y_from_bottom = static_cast<int>(srcIdx / (size_t)W);
                const int x = static_cast<int>(srcIdx % (size_t)W);
                const int dstY = H - 1 - y_from_bottom;
                const size_t dstIdx = ((size_t)dstY * (size_t)W + (size_t)x) * 3;
                colDst[dstIdx]     = sRef.x;
                colDst[dstIdx + 1] = sRef.y;
                colDst[dstIdx + 2] = sRef.z;
                if (albDst) {
                    const Vec4& a = albSrc[srcIdx];
                    albDst[dstIdx]     = a.x;
                    albDst[dstIdx + 1] = a.y;
                    albDst[dstIdx + 2] = a.z;
                }
                if (nrmDst) {
                    const Vec4& n = nrmSrc[srcIdx];
                    nrmDst[dstIdx]     = n.x;
                    nrmDst[dstIdx + 1] = n.y;
                    nrmDst[dstIdx + 2] = n.z;
                }
            });
    }

    OIDNFrameData frame;
    frame.width = image_width;
    frame.height = image_height;
    frame.color = linearColor.data();
    frame.albedo = hasAlbedo ? linearAlbedo.data() : nullptr;
    frame.normal = hasNormal ? linearNormal.data() : nullptr;

    cpu_denoised_valid = applyOIDNDenoising(frame, blend, cpu_denoised_buffer);
    return cpu_denoised_valid;
}


bool Renderer::applyOIDNDenoisingGPU(const Backend::DenoiserFrameDataGPU& frame,
                                     float blend,
                                     const OIDNTonemapParams& tm,
                                     std::vector<uint32_t>& packedOutput) {
    if (!frame.colorDevPtr || frame.width <= 0 || frame.height <= 0) return false;

    // Skip while the viewport backend is tearing down/rebuilding its CUDA-imported
    // textures or while OptiX TLAS rebuild is running. Touching frame.colorDevPtr
    // during either window hits cudaErrorIllegalAddress (700), poisons the CUDA
    // context, and trashes the subsequent OptiX BLAS build.
    extern std::atomic<bool> g_viewport_rebuild_in_progress;
    extern std::atomic<bool> g_optix_rebuild_in_progress;
    if (g_viewport_rebuild_in_progress.load(std::memory_order_acquire) ||
        g_optix_rebuild_in_progress.load(std::memory_order_acquire)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(oidnMutex);

    // Resolve CUDA device ordinal
    int desiredOrd = frame.cudaDeviceOrdinal;
    if (desiredOrd < 0) {
        if (cudaGetDevice(&desiredOrd) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
    }
    cudaStream_t desiredStream = static_cast<cudaStream_t>(frame.cudaStream);

    const bool deviceChanged =
        !oidnCudaInitialized ||
        oidnCudaOrdinal != desiredOrd ||
        oidnCudaStream != frame.cudaStream;

    if (deviceChanged) {
        try {
            // CRITICAL: drain in-flight work on the OLD device/stream BEFORE we
            // reassign oidnState->oidnDevice. FilterRef/BufferRef dtors release handles
            // owned by the old device; if the old stream still has pending
            // OIDN kernels in flight, dtor races against running work and we
            // get an access violation in oidn::FilterRef::~FilterRef
            // (oidn.hpp:384). Sync first, then drop refs while the old
            // device is still the live oidnState->oidnDevice.
            if (oidnInitialized) {
                if (oidnCudaStream) {
                    cudaStreamSynchronize(static_cast<cudaStream_t>(oidnCudaStream));
                    cudaGetLastError();
                }
                try { oidnState->oidnDevice.sync(); } catch (...) {}
                oidnState->oidnFilter       = oidn::FilterRef();
                oidnState->oidnColorBuffer  = oidn::BufferRef();
                oidnState->oidnAlbedoBuffer = oidn::BufferRef();
                oidnState->oidnNormalBuffer = oidn::BufferRef();
                oidnState->oidnOutputBuffer = oidn::BufferRef();
                if (oidnGpuOutputDevPtr) {
                    cudaFree(oidnGpuOutputDevPtr);
                    oidnGpuOutputDevPtr = nullptr;
                }
                oidnGpuOutputBytes = 0;
                if (oidnGpuPackedDevPtr) {
                    cudaFree(oidnGpuPackedDevPtr);
                    oidnGpuPackedDevPtr = nullptr;
                }
                oidnGpuPackedBytes = 0;
                oidnCachedWidth = 0;
                oidnCachedHeight = 0;
                oidnUsingAlbedo = false;
                oidnUsingNormal = false;
                oidnGpuCachedColor = nullptr;
                oidnGpuCachedAlbedo = nullptr;
                oidnGpuCachedNormal = nullptr;
            }

            oidnState->oidnDevice = oidn::newCUDADevice(desiredOrd, desiredStream);
            oidnState->oidnDevice.commit();
            const char* errMsg = nullptr;
            if (oidnState->oidnDevice.getError(errMsg) != oidn::Error::None) {
                SCENE_LOG_WARN(std::string("[OIDN GPU] CUDA device commit failed: ")
                               + (errMsg ? errMsg : "unknown"));
                oidnCudaInitialized = false;
                return false;
            }
            oidnCudaOrdinal = desiredOrd;
            oidnCudaStream = frame.cudaStream;
            oidnCudaInitialized = true;
            oidnInitialized = true;
        } catch (const std::exception& e) {
            SCENE_LOG_WARN(std::string("[OIDN GPU] CUDA device bind failed: ") + e.what());
            oidnCudaInitialized = false;
            return false;
        }
    }

    const int width = frame.width;
    const int height = frame.height;
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t outBufferSize = pixelCount * 3;

    const bool useAlbedo = frame.albedoDevPtr != nullptr;
    const bool useNormal = frame.normalDevPtr != nullptr;

    // Basic validation of device-resident layout to avoid illegal accesses
    if (frame.pixelByteStride < sizeof(float) * 3) {
        SCENE_LOG_ERROR("[OIDN GPU] invalid pixelByteStride (too small)");
        return false;
    }
    if (frame.rowByteStride < frame.pixelByteStride * static_cast<size_t>(width)) {
        SCENE_LOG_ERROR("[OIDN GPU] invalid rowByteStride (too small for width)");
        return false;
    }

    const bool sizeChanged   = (width != oidnCachedWidth) || (height != oidnCachedHeight);
    const bool layoutChanged = (useAlbedo != oidnUsingAlbedo) || (useNormal != oidnUsingNormal);
    const bool ptrsChanged   =
        (frame.colorDevPtr  != oidnGpuCachedColor)  ||
        (frame.albedoDevPtr != oidnGpuCachedAlbedo) ||
        (frame.normalDevPtr != oidnGpuCachedNormal);
    const bool outputSizeChanged = oidnGpuOutputBytes != outBufferSize * sizeof(float);
    const bool qualityChanged = (tm.quality != oidnGpuCachedQuality);
    const bool rebuildFilter = sizeChanged || layoutChanged || ptrsChanged || outputSizeChanged || qualityChanged || !oidnState->oidnFilter;

    if (rebuildFilter) {
        try {
            // Ensure we're on the device expected by the frame to avoid stale-context issues
            int currentDev = -1;
            if (cudaGetDevice(&currentDev) != cudaSuccess) {
                cudaGetLastError();
                SCENE_LOG_WARN("[OIDN GPU] cudaGetDevice failed before setup");
                return false;
            }
            if (desiredOrd >= 0 && desiredOrd != currentDev) {
                if (cudaSetDevice(desiredOrd) != cudaSuccess) {
                    cudaGetLastError();
                    SCENE_LOG_WARN("[OIDN GPU] cudaSetDevice failed when binding desired device");
                    return false;
                }
            }

            // Drain any in-flight OIDN/CUDA work on this stream before tearing
            // down the existing filter and (optionally) freeing the output
            // buffer. Reassigning oidnState->oidnFilter destroys the old FilterRef; if its
            // device still has pending async work, the dtor races and we crash
            // in oidn::FilterRef::~FilterRef (oidn.hpp:384).
            if (oidnState->oidnFilter) {
                if (desiredStream) {
                    cudaStreamSynchronize(desiredStream);
                    cudaGetLastError();
                }
                try { oidnState->oidnDevice.sync(); } catch (...) {}
                oidnState->oidnFilter = oidn::FilterRef();
            }

            if (outputSizeChanged) {
                if (oidnGpuOutputDevPtr) {
                    cudaFree(oidnGpuOutputDevPtr);
                    oidnGpuOutputDevPtr = nullptr;
                }
                if (cudaMalloc(&oidnGpuOutputDevPtr, outBufferSize * sizeof(float)) != cudaSuccess) {
                    cudaGetLastError();
                    oidnGpuOutputDevPtr = nullptr;
                    oidnGpuOutputBytes = 0;
                    return false;
                }
                oidnGpuOutputBytes = outBufferSize * sizeof(float);

                // Packed RGBA8 output buffer (one uint32 per pixel). Tracks the
                // same lifetime as the float3 OIDN output so a single size gate
                // governs both.
                if (oidnGpuPackedDevPtr) {
                    cudaFree(oidnGpuPackedDevPtr);
                    oidnGpuPackedDevPtr = nullptr;
                }
                const size_t packedBytes = pixelCount * sizeof(uint32_t);
                if (cudaMalloc(&oidnGpuPackedDevPtr, packedBytes) != cudaSuccess) {
                    cudaGetLastError();
                    oidnGpuPackedDevPtr = nullptr;
                    oidnGpuPackedBytes = 0;
                    // Free the float3 buffer we just allocated to keep state consistent.
                    cudaFree(oidnGpuOutputDevPtr);
                    oidnGpuOutputDevPtr = nullptr;
                    oidnGpuOutputBytes = 0;
                    return false;
                }
                oidnGpuPackedBytes = packedBytes;
            }

            oidnState->oidnFilter = oidnState->oidnDevice.newFilter("RT");
            // Device-resident inputs with explicit stride.
            oidnState->oidnFilter.setImage("color", frame.colorDevPtr, oidn::Format::Float3,
                                static_cast<size_t>(width), static_cast<size_t>(height),
                                /*byteOffset=*/0,
                                frame.pixelByteStride, frame.rowByteStride);
            if (useAlbedo) {
                oidnState->oidnFilter.setImage("albedo", frame.albedoDevPtr, oidn::Format::Float3,
                                    static_cast<size_t>(width), static_cast<size_t>(height),
                                    0, frame.pixelByteStride, frame.rowByteStride);
            }
            if (useNormal) {
                oidnState->oidnFilter.setImage("normal", frame.normalDevPtr, oidn::Format::Float3,
                                    static_cast<size_t>(width), static_cast<size_t>(height),
                                    0, frame.pixelByteStride, frame.rowByteStride);
            }
            // Output: packed float3 CUDA buffer owned by Renderer so we can optionally
            // blend in-place on device before the final readback.
            oidnState->oidnFilter.setImage("output", oidnGpuOutputDevPtr, oidn::Format::Float3,
                                static_cast<size_t>(width), static_cast<size_t>(height),
                                /*byteOffset=*/0,
                                sizeof(float) * 3,
                                static_cast<size_t>(width) * sizeof(float) * 3);
            oidnState->oidnFilter.set("hdr", true);
            oidnState->oidnFilter.set("srgb", false);
            // OIDN model tier, user-selectable (RenderSettings::denoiser_quality).
            // Default Fast is the cheapest model — on a mid-range GPU OIDN execute
            // dominates the denoise cost (~14-30ms profiled at 720p+aux), so the
            // tier is the single highest-leverage knob. Final renders pass High
            // here via OIDNTonemapParams::quality.
            oidnState->oidnFilter.set("quality", tm.quality);
            oidnState->oidnFilter.commit();

            oidnCachedWidth  = width;
            oidnCachedHeight = height;
            oidnUsingAlbedo  = useAlbedo;
            oidnUsingNormal  = useNormal;
            oidnGpuCachedColor  = frame.colorDevPtr;
            oidnGpuCachedAlbedo = frame.albedoDevPtr;
            oidnGpuCachedNormal = frame.normalDevPtr;
            oidnGpuCachedQuality = tm.quality;
        } catch (const std::exception& e) {
            SCENE_LOG_ERROR(std::string("[OIDN GPU] filter setup failed: ") + e.what());
            return false;
        }
    }

    packedOutput.resize(pixelCount);

#if RT_OIDN_PROFILING
    // ── [PROFILE] CUDA-event timestamps around each GPU stage. Events live on
    // the desiredStream; elapsed time is in float ms (cudaEventElapsedTime).
    // Lazily created and reused across calls; recreated if the stream changes.
    static thread_local cudaEvent_t profEvBeforeExec = nullptr;
    static thread_local cudaEvent_t profEvAfterExec  = nullptr;
    static thread_local cudaEvent_t profEvAfterTm    = nullptr;
    static thread_local cudaEvent_t profEvAfterCopy  = nullptr;
    auto ensureProfEvents = [&]() {
        if (!profEvBeforeExec) cudaEventCreate(&profEvBeforeExec);
        if (!profEvAfterExec)  cudaEventCreate(&profEvAfterExec);
        if (!profEvAfterTm)    cudaEventCreate(&profEvAfterTm);
        if (!profEvAfterCopy)  cudaEventCreate(&profEvAfterCopy);
    };
    ensureProfEvents();
#endif

    try {
        // Ensure OptiX writes on its stream are visible before OIDN reads shared memory.
        if (desiredStream) {
            if (cudaStreamSynchronize(desiredStream) != cudaSuccess) {
                cudaGetLastError();
                SCENE_LOG_WARN("[OIDN GPU] cudaStreamSynchronize failed before execute");
                // proceed — OIDN may still report an error
            }
        }

#if RT_OIDN_PROFILING
        cudaEventRecord(profEvBeforeExec, desiredStream);
#endif
        oidnState->oidnFilter.execute();
#if RT_OIDN_PROFILING
        cudaEventRecord(profEvAfterExec, desiredStream);
#endif

        const char* errMsg = nullptr;
        if (oidnState->oidnDevice.getError(errMsg) != oidn::Error::None) {
            SCENE_LOG_ERROR(std::string("[OIDN GPU] ") + (errMsg ? errMsg : "execute error"));
            return false;
        }

        if (blend < 0.999f) {
            if (!launchOidnBlendKernel(static_cast<float*>(oidnGpuOutputDevPtr),
                                       frame.colorDevPtr,
                                       pixelCount,
                                       blend,
                                       frame.pixelByteStride,
                                       desiredStream)) {
                cudaError_t le = cudaGetLastError();
                SCENE_LOG_ERROR(std::string("[OIDN GPU] blend kernel launch failed: ") + std::to_string(static_cast<int>(le)));
                // Drain sticky CUDA error and force OIDN device rebind on next frame
                // so we don't keep launching kernels into a poisoned context.
                if (le == cudaErrorIllegalAddress) {
                    cudaGetLastError();
                    oidnCudaInitialized = false;
                }
                return false;
            }
        }

        // Fused FULL post_chain (exposure/tonemap operator/temperature/
        // saturation/gamma/vignette) + sRGB + pack-to-uint32. Replaces the
        // per-pixel CPU std::pow loop and shrinks the D2H transfer by 3×.
        OidnPostParams postParams;
        postParams.exposure         = tm.exposure;
        postParams.cameraExposure   = tm.cameraExposure;
        postParams.gamma            = tm.gamma;
        postParams.saturation       = tm.saturation;
        postParams.colorTemperature = tm.colorTemperature;
        postParams.vignetteStrength = tm.vignetteStrength;
        postParams.toneMapping      = tm.toneMapping;
        postParams.vignetteEnabled  = tm.vignetteEnabled;
        postParams.aMaskOr = tm.aMaskOr;
        postParams.rShift  = tm.rShift;
        postParams.gShift  = tm.gShift;
        postParams.bShift  = tm.bShift;
        postParams.flipY   = tm.flipY;
        if (!launchOidnTonemapKernel(static_cast<const float*>(oidnGpuOutputDevPtr),
                                     oidnGpuPackedDevPtr,
                                     width, height,
                                     postParams,
                                     desiredStream)) {
            cudaError_t le = cudaGetLastError();
            SCENE_LOG_ERROR(std::string("[OIDN GPU] tonemap kernel launch failed: ") + std::to_string(static_cast<int>(le)));
            if (le == cudaErrorIllegalAddress) {
                cudaGetLastError();
                oidnCudaInitialized = false;
            }
            return false;
        }
#if RT_OIDN_PROFILING
        cudaEventRecord(profEvAfterTm, desiredStream);
#endif

        if (desiredStream) {
            if (cudaStreamSynchronize(desiredStream) != cudaSuccess) {
                cudaError_t se = cudaGetLastError();
                SCENE_LOG_ERROR(std::string("[OIDN GPU] cudaStreamSynchronize failed after tonemap: ") + std::to_string(static_cast<int>(se)));
                // Same recovery: drain sticky error and force device rebind.
                if (se == cudaErrorIllegalAddress) {
                    cudaGetLastError();
                    oidnCudaInitialized = false;
                }
                return false;
            }
        }

        // Final transfer: tonemapped RGBA8 → host. 4 B/px vs the original 12 B/px.
        if (cudaMemcpy(packedOutput.data(), oidnGpuPackedDevPtr,
                       pixelCount * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaError_t me = cudaGetLastError();
            SCENE_LOG_ERROR(std::string("[OIDN GPU] final cudaMemcpy failed: ") + std::to_string(static_cast<int>(me)));
            return false;
        }
#if RT_OIDN_PROFILING
        cudaEventRecord(profEvAfterCopy, desiredStream);

        // ── [PROFILE] Elapsed times between events. cudaEventSynchronize on the
        // last event ensures all the intermediate events have valid timing.
        cudaEventSynchronize(profEvAfterCopy);
        float msExec = 0.0f, msTm = 0.0f, msCopy = 0.0f;
        cudaEventElapsedTime(&msExec, profEvBeforeExec, profEvAfterExec);
        cudaEventElapsedTime(&msTm,   profEvAfterExec,  profEvAfterTm);
        cudaEventElapsedTime(&msCopy, profEvAfterTm,    profEvAfterCopy);
        static thread_local int   profCounter = 0;
        static thread_local float profExecAvg = 0.0f, profTmAvg = 0.0f, profCopyAvg = 0.0f;
        profExecAvg = profExecAvg * 0.95f + msExec * 0.05f;
        profTmAvg   = profTmAvg   * 0.95f + msTm   * 0.05f;
        profCopyAvg = profCopyAvg * 0.95f + msCopy * 0.05f;
        if (++profCounter % 300 == 0) {
            SCENE_LOG_INFO(std::string("[OIDN][CUDA] oidn=")
                           + std::to_string(profExecAvg)
                           + "ms tonemap+blend=" + std::to_string(profTmAvg)
                           + "ms d2h=" + std::to_string(profCopyAvg) + "ms");
        }
#endif
    } catch (const std::exception& e) {
        SCENE_LOG_ERROR(std::string("[OIDN GPU] execute failed: ") + e.what());
        return false;
    }

    return true;
}


void Renderer::releaseGpuDenoiserResources() {
    std::lock_guard<std::mutex> lock(oidnMutex);
    // OIDN filters may reference backend-owned CUDA frame pointers. Release
    // those bindings before the backend destroys its render buffers/context.
    oidnState->oidnFilter = oidn::FilterRef();
    oidnState->oidnColorBuffer = oidn::BufferRef();
    oidnState->oidnAlbedoBuffer = oidn::BufferRef();
    oidnState->oidnNormalBuffer = oidn::BufferRef();
    oidnState->oidnOutputBuffer = oidn::BufferRef();
    oidnState->oidnDevice = oidn::DeviceRef();
    oidnInitialized = false;
    oidnCudaInitialized = false;
    oidnCudaOrdinal = -1;
    oidnCudaStream = nullptr;
    oidnGpuCachedColor = nullptr;
    oidnGpuCachedAlbedo = nullptr;
    oidnGpuCachedNormal = nullptr;
    oidnGpuCachedQuality = -1;
    oidnCachedWidth = 0;
    oidnCachedHeight = 0;

    if (oidnGpuOutputDevPtr) {
        cudaFree(oidnGpuOutputDevPtr);
        oidnGpuOutputDevPtr = nullptr;
    }
    oidnGpuOutputBytes = 0;
    if (oidnGpuPackedDevPtr) {
        cudaFree(oidnGpuPackedDevPtr);
        oidnGpuPackedDevPtr = nullptr;
    }
    oidnGpuPackedBytes = 0;
}


