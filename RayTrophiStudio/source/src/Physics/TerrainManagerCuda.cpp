#include "TerrainErosionBoundary.h"
#include "TerrainManager.h"
#include "TerrainSystem.h"
#include "globals.h"
#include "erosion_ops.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <random>
#include <algorithm>

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma warning(disable : 4996)

void TerrainManager::initCuda() {
    if (cudaInitialized)
        return;

    if (!g_hasCUDA) {
        SCENE_LOG_WARN("[GPU Erosion] CUDA not available, skipping GPU erosion init");
        return;
    }

    SCENE_LOG_INFO("[GPU Erosion] Initializing CUDA...");

    // Initialize CUDA Driver API
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] cuInit failed: " + std::to_string(res));
        return;
    }

    // Get Device
    CUdevice device;
    res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] cuDeviceGet failed");
        return;
    }

    // Use Primary Context (sharing with OptiX/CUDART)
    CUcontext ctx;
    res = cuDevicePrimaryCtxRetain(&ctx, device);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] cuDevicePrimaryCtxRetain failed");
        return;
    }
    res = cuCtxSetCurrent(ctx);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] cuCtxSetCurrent failed");
        return;
    }

    // Force CUDART initialization to ensure interop is ready
    cudaFree(0);

    // Load PTX Module (prefer ptx/ subfolder next to exe, fall back to legacy root / build layout)
    std::string ptxPath = "ptx/erosion_kernels.ptx";
    if (!std::filesystem::exists(ptxPath))
        ptxPath = "erosion_kernels.ptx";
    if (!std::filesystem::exists(ptxPath))
        ptxPath = "../ptx/erosion_kernels.ptx";
    if (!std::filesystem::exists(ptxPath))
        ptxPath = "../erosion_kernels.ptx";

    res = cuModuleLoad((CUmodule*)&cudaModule, ptxPath.c_str());
    if (res != CUDA_SUCCESS) {
        // Try reading file content manually to debug or just informative error
        SCENE_LOG_ERROR("[GPU Erosion] Failed to load module 'erosion_kernels.ptx' (Error: " +
                        std::to_string(res) + ")");
        SCENE_LOG_WARN("Make sure to run 'compile_ptx.bat' first!");
        return;
    }

    // Get Function
    res = cuModuleGetFunction((CUfunction*)&erosionKernelFunc, (CUmodule)cudaModule,
                              "hydraulicErosionKernel");
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] Failed to get kernel function 'hydraulicErosionKernel'");
        return;
    }

    res = cuModuleGetFunction((CUfunction*)&smoothKernelFunc, (CUmodule)cudaModule,
                              "smoothTerrainKernel");
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] Failed to get kernel function 'smoothTerrainKernel'");
        return;
    }

    res = cuModuleGetFunction((CUfunction*)&thermalKernelFunc, (CUmodule)cudaModule,
                              "thermalErosionKernel");
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] Failed to get kernel function 'thermalErosionKernel'");
        return;
    }

    cuModuleGetFunction((CUfunction*)&fluvRainKernelFunc, (CUmodule)cudaModule,
                        "fluvialRainKernel");
    cuModuleGetFunction((CUfunction*)&fluvFluxKernelFunc, (CUmodule)cudaModule,
                        "fluvialFluxKernel");
    cuModuleGetFunction((CUfunction*)&fluvWaterKernelFunc, (CUmodule)cudaModule,
                        "fluvialWaterKernel");
    cuModuleGetFunction((CUfunction*)&fluvErodeKernelFunc, (CUmodule)cudaModule,
                        "fluvialErosionKernel");
    cuModuleGetFunction((CUfunction*)&streamPowerKernelFunc, (CUmodule)cudaModule,
                        "fluvialStreamPowerKernel");
    cuModuleGetFunction((CUfunction*)&applyStreamPowerKernelFunc, (CUmodule)cudaModule,
                        "applyFluvialStreamPowerKernel");
    cuModuleGetFunction((CUfunction*)&windKernelFunc, (CUmodule)cudaModule, "windErosionKernel");

    // Post-processing kernels (for CPU-GPU parity)
    cuModuleGetFunction((CUfunction*)&pitFillKernelFunc, (CUmodule)cudaModule, "pitFillingKernel");
    cuModuleGetFunction((CUfunction*)&spikeRemovalKernelFunc, (CUmodule)cudaModule,
                        "spikeRemovalKernel");
    cuModuleGetFunction((CUfunction*)&edgePreservationKernelFunc, (CUmodule)cudaModule,
                        "edgePreservationKernel");
    cuModuleGetFunction((CUfunction*)&thermalWithHardnessKernelFunc, (CUmodule)cudaModule,
                        "thermalErosionWithHardnessKernel");

    cudaInitialized = true;
    SCENE_LOG_INFO("[GPU Erosion] CUDA Initialized Successfully (with post-processing kernels)");
}

void TerrainManager::hydraulicErosionGPUCuda(TerrainObject* terrain,
                                             const HydraulicErosionParams& params,
                                             const std::vector<float>& mask) {
    // Lazy Init
    if (!cudaInitialized) {
        // Deprecated on purpose (see the removal plan in TerrainManager.h).
        // /sdl promotes C4996 to an error, so the attribute has to be
        // suppressed at each surviving call site rather than dropped --
        // deleting the marker to make the build quiet would erase the only
        // in-code record that this path is on its way out.
#pragma warning(suppress : 4996)
        initCuda();
        if (!cudaInitialized) {
            SCENE_LOG_WARN(
                "[GPU Erosion] CUDA not initialized. Falling back to CPU hydraulic erosion.");
            hydraulicErosion(terrain, params, mask);
            return;
        }
    }

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    size_t mapSize = w * h * sizeof(float);
    bool hasHardness =
        !terrain->hardnessMap.empty() && terrain->hardnessMap.size() == (size_t)(w * h);
    bool hasMask = !mask.empty() && mask.size() == (size_t)(w * h);

    // Allocate Device Memory
    CUdeviceptr d_heightmap = 0;
    CUdeviceptr d_original = 0;
    CUdeviceptr d_hardness = 0;
    CUdeviceptr d_mask = 0;
    auto freeGpuBuffers = [&]() {
        if (d_mask)
            cuMemFree(d_mask);
        if (d_hardness)
            cuMemFree(d_hardness);
        if (d_original)
            cuMemFree(d_original);
        if (d_heightmap)
            cuMemFree(d_heightmap);
    };

    CUresult res = cuMemAlloc(&d_heightmap, mapSize);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_WARN("[GPU Erosion] Memory allocation failed. Falling back to CPU.");
        hydraulicErosion(terrain, params, mask);
        return;
    }

    // Copy Host to Device
    res = cuMemcpyHtoD(d_heightmap, terrain->heightmap.data.data(), mapSize);
    if (res != CUDA_SUCCESS) {
        freeGpuBuffers();
        SCENE_LOG_WARN("[GPU Erosion] HtoD copy failed. Falling back to CPU.");
        hydraulicErosion(terrain, params, mask);
        return;
    }

    res = cuMemAlloc(&d_original, mapSize);
    if (res == CUDA_SUCCESS) {
        res = cuMemcpyHtoD(d_original, terrain->heightmap.data.data(), mapSize);
    }
    if (res != CUDA_SUCCESS) {
        freeGpuBuffers();
        SCENE_LOG_WARN(
            "[GPU Erosion] Failed to preserve original heights on GPU. Falling back to CPU.");
        hydraulicErosion(terrain, params, mask);
        return;
    }

    if (hasHardness) {
        res = cuMemAlloc(&d_hardness, mapSize);
        if (res == CUDA_SUCCESS) {
            res = cuMemcpyHtoD(d_hardness, terrain->hardnessMap.data(), mapSize);
        }
        if (res != CUDA_SUCCESS) {
            freeGpuBuffers();
            SCENE_LOG_WARN("[GPU Erosion] Hardness upload failed. Falling back to CPU.");
            hydraulicErosion(terrain, params, mask);
            return;
        }
    }

    if (hasMask) {
        res = cuMemAlloc(&d_mask, mapSize);
        if (res == CUDA_SUCCESS) {
            res = cuMemcpyHtoD(d_mask, mask.data(), mapSize);
        }
        if (res != CUDA_SUCCESS) {
            freeGpuBuffers();
            SCENE_LOG_WARN("[GPU Erosion] Mask upload failed. Falling back to CPU.");
            hydraulicErosion(terrain, params, mask);
            return;
        }
    }

    // Prepare Params
    TerrainPhysics::HydraulicErosionParamsGPU gpuParams;
    memset(&gpuParams, 0, sizeof(gpuParams)); // Ensure no garbage pointers

    gpuParams.mapWidth = w;
    gpuParams.mapHeight = h;
    gpuParams.brushRadius = params.erosionRadius;
    gpuParams.dropletLifetime = params.dropletLifetime;
    gpuParams.inertia = params.inertia;
    gpuParams.sedimentCapacity = params.sedimentCapacity;
    gpuParams.minSlope = params.minSlope;
    gpuParams.erodeSpeed = params.erodeSpeed;
    gpuParams.depositSpeed = params.depositSpeed;
    gpuParams.evaporateSpeed = params.evaporateSpeed;
    gpuParams.gravity = params.gravity;
    gpuParams.cellSize = (float)terrain->heightmap.scale_xz / (float)w;
    gpuParams.heightScale = (float)terrain->heightmap.scale_y;
    gpuParams.seed = params.seed;
    gpuParams.hardnessMap = hasHardness ? reinterpret_cast<float*>(d_hardness) : nullptr;
    gpuParams.maskMap = hasMask ? reinterpret_cast<float*>(d_mask) : nullptr;

    unsigned long long seed_offset = 0;

    // Kernel Args
    void* args[] = {&d_heightmap, &gpuParams, &seed_offset};

    // Launch Config
    int blockSize = 128;
    int totalDroplets = params.iterations;
    int batchSize = 256000; // Process in smaller batches to avoid TDR timeout

    SCENE_LOG_INFO("[GPU Erosion] Launching kernel with " + std::to_string(totalDroplets) +
                   " total droplets in batches...");

    int dropletsProcessed = 0;
    while (dropletsProcessed < totalDroplets) {
        int currentBatch = std::min(batchSize, totalDroplets - dropletsProcessed);
        int numBlocks = (currentBatch + blockSize - 1) / blockSize;

        // Kernel Args
        void* args[] = {&d_heightmap, &gpuParams, &seed_offset};

        res = cuLaunchKernel((CUfunction)erosionKernelFunc, numBlocks, 1, 1, // Grid Dim
                             blockSize, 1, 1,                                // Block Dim
                             0, nullptr,                                     // Shared Mem, Stream
                             args, nullptr);

        if (res != CUDA_SUCCESS) {
            SCENE_LOG_ERROR("[GPU Erosion] Launch Failed at droplet " +
                            std::to_string(dropletsProcessed) + ": Error " + std::to_string(res));
            freeGpuBuffers();
            hydraulicErosion(terrain, params, mask);
            return;
        }

        // Synchronize batch and update offset
        cuCtxSynchronize();
        seed_offset += currentBatch;
        dropletsProcessed += currentBatch;
    }

    // Launch Smoothing Pass
    // Grid dim for 2D heightmap
    int tx = 16, ty = 16;
    int bx = (w + tx - 1) / tx;
    int by = (h + ty - 1) / ty;

    SCENE_LOG_INFO("[GPU Erosion] Running post-processing (CPU-GPU parity)...");

    // Calculate cellSize for threshold computation
    float cellSize = terrain->heightmap.scale_xz / w;

    // Prepare PostProcessParams
    TerrainPhysics::PostProcessParamsGPU postParams;
    postParams.mapWidth = w;
    postParams.mapHeight = h;
    postParams.cellSize = cellSize;
    postParams.pitThreshold = cellSize * 0.05f;  // Match CPU
    postParams.spikeThreshold = cellSize * 0.1f; // Match CPU
    postParams.edgeFadeWidth =
        params.boundaryWidth > 0 ? params.boundaryWidth : getEdgeFadeWidth(terrain);
    postParams.edgeFadeWidth =
        std::clamp(postParams.edgeFadeWidth, 1, std::max(1, std::min(w, h) / 2));

    // 1. SPIKE REMOVAL PASS (matches CPU post-erosion spike removal)
    if (spikeRemovalKernelFunc) {
        void* spikeArgs[] = {&d_heightmap, &postParams};
        res = cuLaunchKernel((CUfunction)spikeRemovalKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr,
                             spikeArgs, nullptr);
        if (res != CUDA_SUCCESS) {
            SCENE_LOG_WARN("[GPU Erosion] Spike Removal Failed: " + std::to_string(res));
        }
    }

    // 2. PIT FILLING PASS (matches CPU post-erosion pit filling)
    if (pitFillKernelFunc) {
        void* pitArgs[] = {&d_heightmap, &postParams};
        res = cuLaunchKernel((CUfunction)pitFillKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr,
                             pitArgs, nullptr);
        if (res != CUDA_SUCCESS) {
            SCENE_LOG_WARN("[GPU Erosion] Pit Filling Failed: " + std::to_string(res));
        }
    }

    // 3. FINAL SMOOTHING PASS
    void* smoothArgs[] = {&d_heightmap, &w, &h};
    res = cuLaunchKernel((CUfunction)smoothKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr, smoothArgs,
                         nullptr);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_WARN("[GPU Erosion] Smooth Kernel Failed: " + std::to_string(res));
    }

    // 4. Apply preservation last so smoothing cannot pull border samples away
    // from the authored pre-erosion heightfield. Open and SeaLevel are handled
    // without this kernel.
    if (params.boundaryMode == ErosionBoundaryMode::Preserve && edgePreservationKernelFunc) {
        void* edgeArgs[] = {&d_heightmap, &d_original, &postParams};
        res = cuLaunchKernel((CUfunction)edgePreservationKernelFunc, bx, by, 1, tx, ty, 1, 0,
                             nullptr, edgeArgs, nullptr);
        if (res != CUDA_SUCCESS) {
            SCENE_LOG_WARN("[GPU Erosion] Edge Preservation Failed: " + std::to_string(res));
        }
    }

    // Synchronize (Wait for completion)
    cuCtxSynchronize();

    // Copy Device to Host
    res = cuMemcpyDtoH(terrain->heightmap.data.data(), d_heightmap, mapSize);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Erosion] DtoH Copy Failed! CUDA Result: " + std::to_string(res));
        if (res == CUDA_ERROR_LAUNCH_FAILED) {
            SCENE_LOG_ERROR("[GPU Erosion] Kernel crash detected (likely OOB or TDR).");
        }
    }
    if (res == CUDA_SUCCESS && params.boundaryMode == ErosionBoundaryMode::SeaLevel) {
        const int boundaryWidth =
            params.boundaryWidth > 0 ? params.boundaryWidth : getEdgeFadeWidth(terrain);
        blendTerrainEdgesToLevel(terrain->heightmap.data, w, h, boundaryWidth,
                                 params.boundaryLevel);
    }

    // Cleanup
    freeGpuBuffers();

    // Update Mesh (Visuals)
    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("[GPU Erosion] Complete with post-processing!");
}

void TerrainManager::thermalErosionGPUCuda(TerrainObject* terrain, const ThermalErosionParams& p,
                                           const std::vector<float>& mask) {
    if (!cudaInitialized) {
        // Deprecated on purpose (see the removal plan in TerrainManager.h).
        // /sdl promotes C4996 to an error, so the attribute has to be
        // suppressed at each surviving call site rather than dropped --
        // deleting the marker to make the build quiet would erase the only
        // in-code record that this path is on its way out.
#pragma warning(suppress : 4996)
        initCuda();
        if (!cudaInitialized) {
            SCENE_LOG_WARN(
                "[GPU Thermal] CUDA not initialized. Falling back to CPU thermal erosion.");
            thermalErosion(terrain, p, mask);
            return;
        }
    }

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    size_t mapSize = w * h * sizeof(float);

    // Device Alloc
    CUdeviceptr d_heightmap;
    CUresult res = cuMemAlloc(&d_heightmap, mapSize);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_WARN("[GPU Thermal] Allocation failed. Falling back to CPU.");
        thermalErosion(terrain, p, mask);
        return;
    }

    // Allocate hardness map if available
    CUdeviceptr d_hardness = 0;
    bool hasHardness =
        !terrain->hardnessMap.empty() && terrain->hardnessMap.size() == (size_t)(w * h);
    if (hasHardness) {
        res = cuMemAlloc(&d_hardness, mapSize);
        if (res == CUDA_SUCCESS) {
            cuMemcpyHtoD(d_hardness, terrain->hardnessMap.data(), mapSize);
        } else {
            hasHardness = false;
        }
    }

    // HtoD
    cuMemcpyHtoD(d_heightmap, terrain->heightmap.data.data(), mapSize);

    // Params
    TerrainPhysics::ThermalErosionParamsGPU gpuOps;
    gpuOps.mapWidth = w;
    gpuOps.mapHeight = h;
    gpuOps.talusAngle = p.talusAngle;
    gpuOps.erosionAmount = p.erosionAmount;
    gpuOps.cellSize = (float)terrain->heightmap.scale_xz / (float)w;
    gpuOps.heightScale = (float)terrain->heightmap.scale_y;
    gpuOps.useHardness = hasHardness;
    gpuOps.hardnessMap = hasHardness ? (float*)d_hardness : nullptr;

    // Launch Loop
    int tx = 16, ty = 16;
    int bx = (w + tx - 1) / tx;
    int by = (h + ty - 1) / ty;

    // Use hardness-aware kernel if available and hardness map exists
    if (hasHardness && thermalWithHardnessKernelFunc) {
        void* args[] = {&d_heightmap, &d_hardness, &gpuOps};
        for (int i = 0; i < p.iterations; i++) {
            res = cuLaunchKernel((CUfunction)thermalWithHardnessKernelFunc, bx, by, 1, tx, ty, 1, 0,
                                 nullptr, args, nullptr);
        }
        SCENE_LOG_INFO("[GPU Thermal] Using hardness-aware kernel.");
    } else {
        void* args[] = {&d_heightmap, &gpuOps};
        for (int i = 0; i < p.iterations; i++) {
            res = cuLaunchKernel((CUfunction)thermalKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr,
                                 args, nullptr);
        }
    }

    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Thermal] Launch Failed: " + std::to_string(res));
    }

    // POST-PROCESSING: Pit filling (matches CPU behavior)
    float cellSize = terrain->heightmap.scale_xz / w;
    TerrainPhysics::PostProcessParamsGPU postParams;
    postParams.mapWidth = w;
    postParams.mapHeight = h;
    postParams.cellSize = cellSize;
    postParams.pitThreshold = cellSize * 0.05f;
    postParams.spikeThreshold = cellSize * 0.1f;
    postParams.edgeFadeWidth = std::max(3, w / 40);

    if (pitFillKernelFunc) {
        void* pitArgs[] = {&d_heightmap, &postParams};
        cuLaunchKernel((CUfunction)pitFillKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr, pitArgs,
                       nullptr);
    }

    cuCtxSynchronize();

    // DtoH
    cuMemcpyDtoH(terrain->heightmap.data.data(), d_heightmap, mapSize);

    // Cleanup
    cuMemFree(d_heightmap);
    if (d_hardness)
        cuMemFree(d_hardness);

    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("[GPU Thermal] Completed " + std::to_string(p.iterations) + " iterations" +
                   (hasHardness ? " with hardness." : "."));
}

void TerrainManager::windErosionGPUCuda(TerrainObject* terrain, float strength, float direction,
                                        int iterations, const std::vector<float>& mask) {
    if (!cudaInitialized) {
        // Deprecated on purpose (see the removal plan in TerrainManager.h).
        // /sdl promotes C4996 to an error, so the attribute has to be
        // suppressed at each surviving call site rather than dropped --
        // deleting the marker to make the build quiet would erase the only
        // in-code record that this path is on its way out.
#pragma warning(suppress : 4996)
        initCuda();
        if (!cudaInitialized) {
            SCENE_LOG_WARN("[GPU Wind] CUDA not initialized. Falling back to CPU wind erosion.");
            windErosion(terrain, strength, direction, iterations, mask);
            return;
        }
    }

    int w = terrain->heightmap.width;
    int h = terrain->heightmap.height;
    size_t mapSize = w * h * sizeof(float);

    // Device Alloc
    CUdeviceptr d_heightmap;
    CUresult res = cuMemAlloc(&d_heightmap, mapSize);
    if (res != CUDA_SUCCESS) {
        SCENE_LOG_WARN("[GPU Wind] Allocation failed. Falling back to CPU.");
        windErosion(terrain, strength, direction, iterations, mask);
        return;
    }

    // HtoD
    cuMemcpyHtoD(d_heightmap, terrain->heightmap.data.data(), mapSize);

    // Params
    TerrainPhysics::WindErosionParamsGPU gpuOps;
    gpuOps.mapWidth = w;
    gpuOps.mapHeight = h;

    // Convert direction (degrees) to unit vector
    float rad = direction * 3.14159265f / 180.0f;
    gpuOps.windDirX = cosf(rad);
    gpuOps.windDirY = sinf(rad);
    gpuOps.strength = strength * 0.1f; // Increased from 0.01f
    gpuOps.suspensionRate = 0.3f;
    gpuOps.depositionRate = 0.8f;
    gpuOps.cellSize = (float)terrain->heightmap.scale_xz / (float)w;
    gpuOps.heightScale = (float)terrain->heightmap.scale_y;

    int tx = 16, ty = 16;
    int bx = (w + tx - 1) / tx;
    int by = (h + ty - 1) / ty;

    void* args[] = {&d_heightmap, &gpuOps};

    for (int i = 0; i < iterations; i++) {
        res = cuLaunchKernel((CUfunction)windKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr, args,
                             nullptr);
    }

    if (res != CUDA_SUCCESS) {
        SCENE_LOG_ERROR("[GPU Wind] Launch Failed: " + std::to_string(res));
    }

    // POST-PROCESSING: Smoothing pass (wind erosion tends to create small artifacts)
    void* smoothArgs[] = {&d_heightmap, &w, &h};
    if (smoothKernelFunc) {
        cuLaunchKernel((CUfunction)smoothKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr, smoothArgs,
                       nullptr);
    }

    cuCtxSynchronize();

    // DtoH
    cuMemcpyDtoH(terrain->heightmap.data.data(), d_heightmap, mapSize);
    cuMemFree(d_heightmap);

    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("[GPU Wind] Completed " + std::to_string(iterations) +
                   " iterations with smoothing.");
}

void TerrainManager::fluvialErosionGPUCuda(TerrainObject* terrain, const HydraulicErosionParams& p,
                                           const std::vector<float>& mask,
                                           const std::vector<float>& flowGuide,
                                           TerrainPhysics::StreamPowerParamsGPU sp, int numPasses,
                                           const std::function<void()>& updateFlowCPU) {
    const int w = terrain->heightmap.width;
    const int h = terrain->heightmap.height;
    const int numPixels = w * h;
    auto& height = terrain->heightmap.data;
    if (!cudaInitialized) {
        // Deprecated on purpose (see the removal plan in TerrainManager.h).
        // /sdl promotes C4996 to an error, so the attribute has to be
        // suppressed at each surviving call site rather than dropped --
        // deleting the marker to make the build quiet would erase the only
        // in-code record that this path is on its way out.
#pragma warning(suppress : 4996)
        initCuda();
        if (!cudaInitialized) {
            SCENE_LOG_WARN(
                "[GPU Fluvial] CUDA not initialized. Falling back to CPU fluvial erosion.");
            fluvialErosion(terrain, p, mask, std::function<void(float)>{}, flowGuide);
            return;
        }
    }

    // Alloc temporary buffers
    CUdeviceptr d_height, d_flow, d_mask, d_hardness, d_erosionAccum;
    cuMemAlloc(&d_height, numPixels * sizeof(float));
    cuMemAlloc(&d_flow, numPixels * sizeof(float));
    cuMemAlloc(&d_mask, numPixels * sizeof(float));
    cuMemAlloc(&d_hardness, numPixels * sizeof(float));
    cuMemAlloc(&d_erosionAccum, numPixels * sizeof(float));

    if (!terrain->hardnessMap.empty()) {
        cuMemcpyHtoD(d_hardness, terrain->hardnessMap.data(), numPixels * sizeof(float));
    } else {
        cuMemsetD8(d_hardness, 0, numPixels * sizeof(float));
    }

    if (!mask.empty()) {
        cuMemcpyHtoD(d_mask, mask.data(), numPixels * sizeof(float));
    } else {
        std::vector<float> fullMask(numPixels, 1.0f);
        cuMemcpyHtoD(d_mask, fullMask.data(), numPixels * sizeof(float));
    }

    int tx = 16, ty = 16;
    int bx = (w + tx - 1) / tx;
    int by = (h + ty - 1) / ty;

    for (int pass = 0; pass < numPasses; pass++) {
        updateFlowCPU();
        cuMemcpyHtoD(d_height, height.data(), numPixels * sizeof(float));
        cuMemcpyHtoD(d_flow, terrain->flowMap.data(), numPixels * sizeof(float));
        cuMemsetD8(d_erosionAccum, 0, numPixels * sizeof(float));

        void* args[] = {&d_height, &d_erosionAccum, &d_flow, &d_hardness, &d_mask, &sp};
        cuLaunchKernel((CUfunction)streamPowerKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr, args,
                       nullptr);
        if (applyStreamPowerKernelFunc) {
            void* applyArgs[] = {&d_height, &d_erosionAccum, &sp};
            cuLaunchKernel((CUfunction)applyStreamPowerKernelFunc, bx, by, 1, tx, ty, 1, 0, nullptr,
                           applyArgs, nullptr);
        }
        cuCtxSynchronize();

        cuMemcpyDtoH(height.data(), d_height, numPixels * sizeof(float));
    }

    // Cleanup
    cuMemFree(d_height);
    cuMemFree(d_flow);
    cuMemFree(d_mask);
    cuMemFree(d_hardness);
    cuMemFree(d_erosionAccum);

    updateTerrainMesh(terrain);
    terrain->dirty_mesh = true;
    SCENE_LOG_INFO("[GPU Fluvial] River Carving Complete.");
}
