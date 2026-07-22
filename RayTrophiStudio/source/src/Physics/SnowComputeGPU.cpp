#include "SnowComputeGPU.h"

#include "SimulationCompute.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <mutex>
#include <utility>

namespace TerrainNodesV2 {
namespace {

struct alignas(4) SnowGpuConstants {
    int width, height, mode, stride;
    int windDx, windDy, pad0, pad1;
    float cellSize, maxDepth, transportRate, slipAngle;
    float windStrength, meltAmount, solarMelt, refreezeRate;
    float slipTangent, geometryMaxStep, relaxationLimit, geometryDepthCap;
    float geometrySurfaceWindow, padf0;
};
static_assert(sizeof(SnowGpuConstants) == 88, "Snow shader push constants must stay ABI-compatible");

} // namespace

bool solveSnowOnGpu(const SnowGpuParams& p,
                    const std::vector<float>& baseMeters,
                    const std::vector<float>& coldness,
                    const std::vector<float>& solar,
                    const std::vector<float>& windExposure,
                    std::vector<float>& snowDepth,
                    std::vector<float>& iceDepth,
                    std::vector<float>& water,
                    std::vector<float>& runoffTrace,
                    std::vector<float>& avalancheDeposit,
                    std::vector<float>& geometryDepth,
                    std::string* failureReason) {
    const size_t count = static_cast<size_t>(p.width) * static_cast<size_t>(p.height);
    const auto validInput = [count](const std::vector<float>& v) { return v.size() == count; };
    if (p.width < 2 || p.height < 2 || !validInput(baseMeters) || !validInput(coldness) ||
        !validInput(solar) || !validInput(windExposure) || !validInput(snowDepth)) {
        if (failureReason) *failureReason = "invalid snow field dimensions";
        return false;
    }

    auto* backend = RayTrophiSim::acquireSharedMeshComputeBackend();
    if (!backend || !backend->supportsDispatch()) {
        if (failureReason) *failureReason = "Vulkan compute backend is unavailable";
        return false;
    }
    std::lock_guard<std::recursive_mutex> computeLock(RayTrophiSim::sharedMeshComputeMutex());

    using RayTrophiSim::ComputeBufferHandle;
    std::vector<ComputeBufferHandle> owned;
    const size_t bytes = count * sizeof(float);
    const auto makeBuffer = [&](const char* name) {
        RayTrophiSim::ComputeBufferDesc desc;
        desc.debug_name = name;
        desc.size_bytes = bytes;
        desc.usage = RayTrophiSim::ComputeBufferUsage::Storage |
                     RayTrophiSim::ComputeBufferUsage::Upload |
                     RayTrophiSim::ComputeBufferUsage::Download |
                     RayTrophiSim::ComputeBufferUsage::ReadWrite;
        ComputeBufferHandle handle = backend->createBuffer(desc);
        if (handle.valid()) owned.push_back(handle);
        return handle;
    };
    const auto cleanup = [&]() {
        for (auto it = owned.rbegin(); it != owned.rend(); ++it) backend->destroyBuffer(*it);
    };
    const auto fail = [&](const char* reason) {
        backend->synchronize();
        cleanup();
        if (failureReason) *failureReason = reason;
        return false;
    };

    ComputeBufferHandle base = makeBuffer("snow.base");
    ComputeBufferHandle snowA = makeBuffer("snow.depth.a");
    ComputeBufferHandle snowB = makeBuffer("snow.depth.b");
    ComputeBufferHandle cold = makeBuffer("snow.coldness");
    ComputeBufferHandle sun = makeBuffer("snow.solar");
    ComputeBufferHandle wind = makeBuffer("snow.wind_exposure");
    ComputeBufferHandle iceA = makeBuffer("snow.ice.a");
    ComputeBufferHandle iceB = makeBuffer("snow.ice.b");
    ComputeBufferHandle waterA = makeBuffer("snow.water.a");
    ComputeBufferHandle waterB = makeBuffer("snow.water.b");
    ComputeBufferHandle trace = makeBuffer("snow.runoff_trace");
    ComputeBufferHandle avalanche = makeBuffer("snow.avalanche");
    ComputeBufferHandle geometryA = makeBuffer("snow.geometry.a");
    ComputeBufferHandle geometryB = makeBuffer("snow.geometry.b");
    if (owned.size() != 14) return fail("GPU snow buffer allocation failed");

    std::vector<float> zero(count, 0.0f);
    const auto upload = [&](ComputeBufferHandle h, const std::vector<float>& v) {
        return backend->uploadBuffer(h, v.data(), bytes, 0);
    };
    backend->beginTransferBatch();
    const bool uploaded = upload(base, baseMeters) && upload(snowA, snowDepth) && upload(snowB, zero) &&
        upload(cold, coldness) && upload(sun, solar) && upload(wind, windExposure) &&
        upload(iceA, zero) && upload(iceB, zero) && upload(waterA, zero) && upload(waterB, zero) &&
        upload(trace, zero) && upload(avalanche, zero) && upload(geometryA, zero) && upload(geometryB, zero);
    if (!backend->endTransferBatch() || !uploaded) return fail("GPU snow upload failed");

    SnowGpuConstants c{};
    c.width = p.width; c.height = p.height; c.windDx = p.windDx; c.windDy = p.windDy;
    c.cellSize = p.cellSize; c.maxDepth = p.maxDepth; c.transportRate = p.transportRate;
    c.slipAngle = p.slipAngleDegrees; c.windStrength = p.windStrength; c.meltAmount = p.meltAmount;
    c.solarMelt = p.solarMelt; c.refreezeRate = p.refreezeRate;
    c.slipTangent = p.effectiveSlipTangent; c.geometryMaxStep = p.geometryMaxStep;
    c.relaxationLimit = p.relaxationHeightLimit;
    c.geometryDepthCap = p.geometryDepthCap;
    c.geometrySurfaceWindow = p.geometrySurfaceWindow;
    const uint32_t groups = static_cast<uint32_t>((count + 255u) / 256u);
    const auto dispatch = [&](int mode, int stride) {
        c.mode = mode; c.stride = stride;
        const ComputeBufferHandle buffers[] = { base, snowA, snowB, cold, sun, wind,
            iceA, iceB, waterA, waterB, trace, avalanche, geometryA, geometryB };
        RayTrophiSim::ComputeDispatch cmd;
        cmd.kernel = "terrain_snow_solver";
        cmd.groups = { groups, 1, 1 };
        cmd.buffers = buffers;
        cmd.buffer_count = std::size(buffers);
        cmd.constants = &c;
        cmd.constants_size = sizeof(c);
        return backend->dispatch(cmd);
    };

    for (int iteration = 0; iteration < p.settlePasses; ++iteration) {
        const float coarseT = p.settlePasses > 1
            ? 1.0f - static_cast<float>(iteration) / static_cast<float>(p.settlePasses - 1) : 0.0f;
        const int stride = (std::max)(1, static_cast<int>(std::round(
            1.0f + static_cast<float>(p.maximumTransportStride - 1) * coarseT * coarseT)));
        if (!dispatch(0, stride)) return fail("GPU snow settling dispatch failed");
        std::swap(snowA, snowB);
    }
    for (int pass = 0; pass < 2; ++pass) {
        if (!dispatch(1, 1)) return fail("GPU snow relaxation dispatch failed");
        std::swap(snowA, snowB);
    }
    if (!dispatch(2, 1)) return fail("GPU snow melt dispatch failed");
    for (int iteration = 0; iteration < p.softSnowPasses; ++iteration) {
        if (!dispatch(3, 1)) return fail("GPU snow creep dispatch failed");
        std::swap(snowA, snowB);
    }
    if (!dispatch(4, 1)) return fail("GPU snow exposure clearing dispatch failed");
    for (int iteration = 0; iteration < p.runoffPasses; ++iteration) {
        if (!dispatch(5, 1)) return fail("GPU snow runoff dispatch failed");
        std::swap(iceA, iceB);
        std::swap(waterA, waterB);
    }
    if (!dispatch(7, 1)) return fail("GPU snow geometry initialization failed");
    for (int pass = 0; pass < p.geometryRelaxPasses; ++pass) {
        if (!dispatch(6, 1)) return fail("GPU snow geometry relaxation failed");
        std::swap(geometryA, geometryB);
    }
    backend->synchronize();

    std::vector<float> outSnow(count), outIce(count), outWater(count), outTrace(count), outAvalanche(count), outGeometry(count);
    const auto download = [&](ComputeBufferHandle h, std::vector<float>& v) {
        return backend->downloadBuffer(h, v.data(), bytes, 0);
    };
    backend->beginTransferBatch();
    const bool downloaded = download(snowA, outSnow) && download(iceA, outIce) &&
        download(waterA, outWater) && download(trace, outTrace) &&
        download(avalanche, outAvalanche) && download(geometryA, outGeometry);
    const bool transfersComplete = backend->endTransferBatch();
    cleanup();
    if (!downloaded || !transfersComplete) {
        if (failureReason) *failureReason = "GPU snow download failed";
        return false;
    }

    snowDepth.swap(outSnow);
    iceDepth.swap(outIce);
    water.swap(outWater);
    runoffTrace.swap(outTrace);
    avalancheDeposit.swap(outAvalanche);
    geometryDepth.swap(outGeometry);
    return true;
}

} // namespace TerrainNodesV2
