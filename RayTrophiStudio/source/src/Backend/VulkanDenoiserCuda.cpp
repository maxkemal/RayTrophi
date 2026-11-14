// Studio offline denoiser interop. Link instead of VulkanDenoiserHost.cpp.
#include "Backend/VulkanBackend.h"
#include "VulkanBackend_Internal.h"
#include "globals.h"
#include "oidn_blend_cuda.h"
#include <cuda_runtime.h>
#include <windows.h>
#include <cstring>
#include <mutex>

namespace Backend {
struct VulkanBackendAdapter::VulkanCudaDenoiserInterop {
    // Imported Vulkan memory (one per AOV). CUDA owns the Win32 handle once
    // cudaImportExternalMemory succeeds; the memory itself stays owned by
    // Vulkan, so we must destroy this import BEFORE freeing VkDeviceMemory.
    cudaExternalMemory_t colorExt = nullptr;
    cudaExternalMemory_t albedoExt = nullptr;
    cudaExternalMemory_t normalExt = nullptr;
    // Mapped device pointers into the imported memory (read-only source).
    void* colorSrcDev = nullptr;
    void* albedoSrcDev = nullptr;
    void* normalSrcDev = nullptr;
    // CUDA-owned destination buffers the prep kernel writes into. OIDN reads
    // from these (float4, 16 B stride, Y-flipped + normal-decoded).
    void* colorDstDev = nullptr;
    void* albedoDstDev = nullptr;
    void* normalDstDev = nullptr;
    // Cached shape — invalidates on mismatch.
    int width = 0;
    int height = 0;
    bool hasAux = false;
    cudaStream_t stream = nullptr;
    int cudaOrdinal = -1;
};

void VulkanBackendAdapter::destroyGpuDenoiserInterop() {
    if (!m_gpuDenoiserInterop)
        return;
    VulkanCudaDenoiserInterop* s = m_gpuDenoiserInterop;

    // Order matters: free CUDA-owned dst buffers, drop mapped ptrs via destroyExternalMemory
    // (which also closes the Win32 HANDLE CUDA took ownership of), then the VkDeviceMemory
    // gets freed later when the caller destroys the VkBuffer handles.
    auto freeDst = [](void*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };
    freeDst(s->colorDstDev);
    freeDst(s->albedoDstDev);
    freeDst(s->normalDstDev);

    auto dropExt = [](cudaExternalMemory_t& e, void*& mapped) {
        mapped = nullptr; // was a child of `e`; destroyed with it
        if (e) {
            cudaDestroyExternalMemory(e);
            e = nullptr;
        }
    };
    dropExt(s->colorExt, s->colorSrcDev);
    dropExt(s->albedoExt, s->albedoSrcDev);
    dropExt(s->normalExt, s->normalSrcDev);

    s->width = 0;
    s->height = 0;
    s->hasAux = false;

    delete s;
    m_gpuDenoiserInterop = nullptr;
}

bool VulkanBackendAdapter::ensureGpuDenoiserInterop(int width, int height, bool needAux) {
    if (m_gpuDenoiserDisabled)
        return false;
    if (!m_device || !m_device->isInitialized())
        return false;
    if (!m_device->getCapabilities().supportsExternalMemoryWin32)
        return false;
    if (width <= 0 || height <= 0)
        return false;

    // Fast path FIRST — after the initial import the Win32 handles have been
    // nulled (CUDA took ownership) so the handle-presence checks below would
    // permanently fail on every subsequent frame. The interop object itself
    // remains valid and reusable. NOTE: hasAux is a runtime usage flag, not a
    // structural property — we don't invalidate on aux toggle, because re-import
    // is impossible once handles are consumed (would permanently kill GPU OIDN).
    if (m_gpuDenoiserInterop && m_gpuDenoiserInterop->width == width &&
        m_gpuDenoiserInterop->height == height) {
        m_gpuDenoiserInterop->hasAux = needAux; // keep flag synced for callers
        return true;
    }

    // Initial-import path: handles must still be present. We always import ALL 3
    // AOVs regardless of needAux at this moment, so that subsequent calls with a
    // different needAux value can reuse the interop. Allocation already produced
    // all three exportable stagings up front.
    if (!m_denoiserColorSharedStaging.buffer || !m_denoiserColorSharedHandle)
        return false;
    if (!m_denoiserAlbedoSharedStaging.buffer || !m_denoiserAlbedoSharedHandle ||
        !m_denoiserNormalSharedStaging.buffer || !m_denoiserNormalSharedHandle) {
        return false;
    }

    // Shape changed (resize) — tear down and rebuild.
    destroyGpuDenoiserInterop();

    // Match Vulkan physical device to a CUDA device ordinal via UUID. Without
    // UUID match, the imported memory would be read from the wrong device and
    // OIDN results would be garbage.
    int cudaOrdinal = -1;
    if (m_device->getCapabilities().hasDeviceUUID) {
        int cudaDeviceCount = 0;
        if (cudaGetDeviceCount(&cudaDeviceCount) == cudaSuccess) {
            for (int i = 0; i < cudaDeviceCount; ++i) {
                cudaDeviceProp props{};
                if (cudaGetDeviceProperties(&props, i) != cudaSuccess)
                    continue;
                if (std::memcmp(props.uuid.bytes, m_device->getCapabilities().deviceUUID, 16) ==
                    0) {
                    cudaOrdinal = i;
                    break;
                }
            }
        }
        cudaGetLastError(); // clear any sticky error from enumeration
    }
    if (cudaOrdinal < 0) {
        SCENE_LOG_WARN("[Vulkan] No matching CUDA device for Vulkan physical device — GPU-direct "
                       "OIDN disabled.");
        m_gpuDenoiserDisabled = true;
        return false;
    }

    int prevDevice = 0;
    cudaGetDevice(&prevDevice);
    cudaSetDevice(cudaOrdinal);

    auto* s = new VulkanCudaDenoiserInterop();
    s->width = width;
    s->height = height;
    s->hasAux = needAux;
    s->cudaOrdinal = cudaOrdinal;
    s->stream = nullptr; // default stream; CPU already fence-synced the Vulkan copy

    auto importOne = [&](void*& win32Handle, uint64_t allocSize, uint64_t mappedBytes,
                         cudaExternalMemory_t& outExt, void*& outDev) -> bool {
        cudaExternalMemoryHandleDesc desc{};
        desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        desc.handle.win32.handle = win32Handle;
        desc.handle.win32.name = nullptr;
        desc.size = allocSize;
        desc.flags = cudaExternalMemoryDedicated;
        if (cudaImportExternalMemory(&outExt, &desc) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        // CUDA owns the Win32 handle after a successful import.
        win32Handle = nullptr;
        cudaExternalMemoryBufferDesc bufDesc{};
        bufDesc.offset = 0;
        bufDesc.size = mappedBytes;
        bufDesc.flags = 0;
        if (cudaExternalMemoryGetMappedBuffer(&outDev, outExt, &bufDesc) != cudaSuccess) {
            cudaGetLastError();
            cudaDestroyExternalMemory(outExt);
            outExt = nullptr;
            return false;
        }
        return true;
    };

    const uint64_t mappedBytes = static_cast<uint64_t>(width) * height * 4ull * sizeof(float);

    // Always import all 3 AOVs so runtime aux toggle is just a flag (handles
    // can be consumed exactly once per process; importing on-demand later is
    // impossible). Memory cost is ~30 MB extra at 720p — acceptable.
    bool ok = importOne(m_denoiserColorSharedHandle, m_denoiserColorSharedAllocSize, mappedBytes,
                        s->colorExt, s->colorSrcDev);

    if (ok) {
        ok = importOne(m_denoiserAlbedoSharedHandle, m_denoiserAlbedoSharedAllocSize, mappedBytes,
                       s->albedoExt, s->albedoSrcDev);
    }
    if (ok) {
        ok = importOne(m_denoiserNormalSharedHandle, m_denoiserNormalSharedAllocSize, mappedBytes,
                       s->normalExt, s->normalSrcDev);
    }

    if (ok) {
        // Allocate CUDA-owned destination buffers (Y-flipped / decoded AOVs) for
        // all AOVs unconditionally — same rationale as the imports above.
        auto allocDst = [&](void*& p) -> bool {
            return cudaMalloc(&p, mappedBytes) == cudaSuccess;
        };
        ok = allocDst(s->colorDstDev) && allocDst(s->albedoDstDev) && allocDst(s->normalDstDev);
        if (!ok) {
            cudaGetLastError();
        }
    }

    if (!ok) {
        SCENE_LOG_WARN("[Vulkan] CUDA external-memory import failed — disabling GPU-direct OIDN.");
        // Swap it in so destroyGpuDenoiserInterop can clean up what did succeed.
        m_gpuDenoiserInterop = s;
        destroyGpuDenoiserInterop();
        m_gpuDenoiserDisabled = true;
        cudaSetDevice(prevDevice);
        return false;
    }

    m_gpuDenoiserInterop = s;
    cudaSetDevice(prevDevice);
    return true;
}

bool VulkanBackendAdapter::getDenoiserFrameGPU(DenoiserFrameDataGPU& frame, bool useAuxiliary) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (m_gpuDenoiserDisabled) {
        static bool loggedDisable = false;
        if (!loggedDisable) {
            loggedDisable = true;
            SCENE_LOG_WARN("[Vulkan][OIDN-GPU] Disabled — falling back to CPU-visible OIDN path.");
        }
        return false;
    }
    if (!m_device || !m_device->isInitialized() || m_imageWidth <= 0 || m_imageHeight <= 0)
        return false;
    if (!m_denoiserColorImage.image)
        return false;
    if (useAuxiliary && (!m_denoiserAlbedoImage.image || !m_denoiserNormalImage.image))
        return false;
    if (!m_denoiserColorSharedStaging.buffer) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            SCENE_LOG_WARN(
                std::string("[Vulkan][OIDN-GPU] Skipped: exportable staging not allocated "
                            "(supportsExternalMemoryWin32=") +
                std::to_string((int)m_device->getCapabilities().supportsExternalMemoryWin32) +
                ").");
        }
        return false;
    }
    if (useAuxiliary &&
        (!m_denoiserAlbedoSharedStaging.buffer || !m_denoiserNormalSharedStaging.buffer))
        return false;

    if (!ensureGpuDenoiserInterop(m_imageWidth, m_imageHeight, useAuxiliary)) {
        return false;
    }
    VulkanCudaDenoiserInterop* s = m_gpuDenoiserInterop;
    static bool firstSuccessLogged = false;
    if (!firstSuccessLogged) {
        firstSuccessLogged = true;
        SCENE_LOG_INFO(std::string("[Vulkan][OIDN-GPU] Active: Vulkan-CUDA external-memory path "
                                   "engaged (cudaOrdinal=") +
                       std::to_string(s->cudaOrdinal) + ", " + std::to_string(m_imageWidth) + "x" +
                       std::to_string(m_imageHeight) +
                       ", aux=" + std::to_string((int)useAuxiliary) + ").");
    }

    // ── Fence-deferred ping-pong ──────────────────────────────────────────────
    // Steady-state ordering each call:
    //   1. Wait the previously-submitted copy fence (near-zero at steady state).
    //   2. CUDA prep kernel reads staging → writes dst.
    //   3. cudaStreamSynchronize so staging is fully consumed.
    //   4. Submit NEW Vulkan copy of m_denoiserColorImage → staging, async (no
    //      wait). The next call's wait covers this.
    //   5. Return dst pointers for OIDN. While Renderer runs OIDN+tonemap+D2H,
    //      the GPU is concurrently executing RT(N+1) + copy(N+1).
    //
    // First call: there is no prior submit, so we have nothing prep'd to return.
    // Submit the seed copy and bail with `false`; the caller falls back to last
    // frame's display. From the second call on, the pipeline is filled.

#if RT_OIDN_PROFILING
    const auto profStartTotal = std::chrono::high_resolution_clock::now();
#endif

    const bool firstCall = !m_device->hasDenoiserCopyEverSubmitted();

    // Submit list — same triple in aux mode, single color in performance mode.
    const VulkanRT::ImageHandle srcsAux[3] = {m_denoiserColorImage, m_denoiserAlbedoImage,
                                              m_denoiserNormalImage};
    const VulkanRT::BufferHandle dstsAux[3] = {
        m_denoiserColorSharedStaging, m_denoiserAlbedoSharedStaging, m_denoiserNormalSharedStaging};
    const VulkanRT::ImageHandle srcsPerf[1] = {m_denoiserColorImage};
    const VulkanRT::BufferHandle dstsPerf[1] = {m_denoiserColorSharedStaging};
    const VulkanRT::ImageHandle* srcsPtr = useAuxiliary ? srcsAux : srcsPerf;
    const VulkanRT::BufferHandle* dstsPtr = useAuxiliary ? dstsAux : dstsPerf;
    const size_t copyCount = useAuxiliary ? 3 : 1;

    // ★★★ The ping-pong returns the copy submitted by the PREVIOUS call. That
    // is one sample old when the caller denoises every sample, but the viewport
    // denoise schedule (progressive / converged_only) calls sparsely, and then
    // "the previous call" can be another accumulation: measured 2026-10-26, a
    // converged_only image showed the camera pose from before a 0.25-unit move,
    // 6 px off ("the camera turned slightly"). The copy is tagged; anything
    // other than this accumulation at most one sample back is re-copied now,
    // synchronously (queued after the in-flight trace, so it holds this sample).
    const bool stagedIsCurrent = !firstCall && m_denoiserCopyEpoch == m_accumulationEpoch &&
                                 m_denoiserCopySamples >= 0 &&
                                 m_currentSamples - m_denoiserCopySamples <= 1;
    if (firstCall || !stagedIsCurrent) {
        // Seed the pipeline (first call) or replace a stale staged copy. Blocks
        // on this one submit; the steady every-sample path never comes here.
        if (!firstCall)
            m_device->waitDenoiserCopy();
        m_device->submitDenoiserCopyAsync(srcsPtr, dstsPtr, copyCount);
        m_device->waitDenoiserCopy();
        m_denoiserCopyEpoch = m_accumulationEpoch;
        m_denoiserCopySamples = m_currentSamples;
        // Resubmit isn't needed — the data is already in staging. But we have
        // no prep'd dst yet, so we'd still have to do prep + return. To keep
        // first-call logic uniform with the steady state, fall through to the
        // wait+prep block below.
    }

    // Wait on the previously-submitted copy (signaled at steady state).
#if RT_OIDN_PROFILING
    const auto profWaitStart = std::chrono::high_resolution_clock::now();
#endif
    if (!m_device->waitDenoiserCopy()) {
        // First-call path with no prior submit must have happened above; if we
        // still see no submit, something is wrong. Bail.
        return false;
    }
#if RT_OIDN_PROFILING
    const auto profWaitEnd = std::chrono::high_resolution_clock::now();
    const float profWaitMs =
        std::chrono::duration<float, std::milli>(profWaitEnd - profWaitStart).count();
#endif

    int prevDevice = 0;
    cudaGetDevice(&prevDevice);
    cudaSetDevice(s->cudaOrdinal);

    // CUDA prep kernel: staging → dst buffers (Y-flip + normal-decode).
#if RT_OIDN_PROFILING
    const auto profPrepStart = std::chrono::high_resolution_clock::now();
#endif
    bool kernelOK =
        launchVulkanDenoiserPrepKernel(s->colorDstDev, s->colorSrcDev, m_imageWidth, m_imageHeight,
                                       /*decodeNormal=*/false, s->stream);
    if (kernelOK && useAuxiliary) {
        kernelOK = kernelOK && launchVulkanDenoiserPrepKernel(s->albedoDstDev, s->albedoSrcDev,
                                                              m_imageWidth, m_imageHeight,
                                                              /*decodeNormal=*/false, s->stream);
        kernelOK = kernelOK && launchVulkanDenoiserPrepKernel(s->normalDstDev, s->normalSrcDev,
                                                              m_imageWidth, m_imageHeight,
                                                              /*decodeNormal=*/true, s->stream);
    }
    // Sync so staging is fully consumed before we submit the next overwrite.
    cudaStreamSynchronize(s->stream);
#if RT_OIDN_PROFILING
    const auto profPrepEnd = std::chrono::high_resolution_clock::now();
    const float profPrepMs =
        std::chrono::duration<float, std::milli>(profPrepEnd - profPrepStart).count();
#endif

    if (!kernelOK) {
        cudaGetLastError();
        cudaSetDevice(prevDevice);
        return false;
    }

    // Submit the NEXT copy asynchronously. Vulkan queues this after RT(N) is
    // already in flight; the fence will be checked next call.
#if RT_OIDN_PROFILING
    const auto profSubmitStart = std::chrono::high_resolution_clock::now();
#endif
    m_device->submitDenoiserCopyAsync(srcsPtr, dstsPtr, copyCount);
    m_denoiserCopyEpoch = m_accumulationEpoch;
    m_denoiserCopySamples = m_currentSamples;
#if RT_OIDN_PROFILING
    const auto profSubmitEnd = std::chrono::high_resolution_clock::now();
    const float profSubmitMs =
        std::chrono::duration<float, std::milli>(profSubmitEnd - profSubmitStart).count();

    const auto profEndTotal = std::chrono::high_resolution_clock::now();
    const float profTotalMs =
        std::chrono::duration<float, std::milli>(profEndTotal - profStartTotal).count();

    // Light telemetry: every 300 frames so the log stays quiet but regressions
    // (sudden wait spike from a re-introduced sync wait, prep ballooning, etc.)
    // are still visible during long sessions.
    {
        static thread_local int profCounter = 0;
        static thread_local float profWaitAvg = 0.0f;
        static thread_local float profPrepAvg = 0.0f;
        static thread_local float profSubmitAvg = 0.0f;
        static thread_local float profTotalAvg = 0.0f;
        const float a = 0.95f, b = 0.05f;
        profWaitAvg = profWaitAvg * a + profWaitMs * b;
        profPrepAvg = profPrepAvg * a + profPrepMs * b;
        profSubmitAvg = profSubmitAvg * a + profSubmitMs * b;
        profTotalAvg = profTotalAvg * a + profTotalMs * b;
        if (++profCounter % 300 == 0) {
            SCENE_LOG_INFO(std::string("[OIDN][Vulkan] wait=") + std::to_string(profWaitAvg) +
                           "ms prep=" + std::to_string(profPrepAvg) +
                           "ms submit=" + std::to_string(profSubmitAvg) +
                           "ms total=" + std::to_string(profTotalAvg) +
                           "ms (aux=" + std::to_string((int)useAuxiliary) + ")");
        }
    }
#endif

    frame.width = m_imageWidth;
    frame.height = m_imageHeight;
    frame.colorDevPtr = s->colorDstDev;
    frame.albedoDevPtr = useAuxiliary ? s->albedoDstDev : nullptr;
    frame.normalDevPtr = useAuxiliary ? s->normalDstDev : nullptr;
    frame.pixelByteStride = sizeof(float) * 4;
    frame.rowByteStride = static_cast<size_t>(m_imageWidth) * sizeof(float) * 4;
    frame.cudaStream = static_cast<void*>(s->stream);
    frame.cudaDeviceOrdinal = s->cudaOrdinal;

    cudaSetDevice(prevDevice);
    return true;
}

} // namespace Backend
