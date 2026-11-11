// User test target: link SimulationCompute.cpp and its normal platform dependencies.
#include "Fluid/MixedGpuStageDiagnostics.h"
#include <stdexcept>
#include <string>

namespace F = RayTrophiSim::Fluid;
using namespace RayTrophiSim;

static void require(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main() {
    SimulationComputeContext compute; // CPU buffer backend; no Studio or GPU needed.
    const auto buffer = compute.createBuffer({"profile_test", sizeof(float),
        ComputeBufferUsage::Storage | ComputeBufferUsage::Upload | ComputeBufferUsage::Download});
    require(buffer.valid(), "CPU test buffer allocation failed");
    SimulationComputeContext::TransferStats parent;
    parent.measured = true;
    parent.upload_calls = 5;
    parent.upload_bytes = 20;
    compute.setTransferProbe(&parent);
    F::MixedGpuStageStats stats;
    {
        F::MixedGpuStageProbe probe(&compute, stats);
        const float input = 3.0f;
        require(compute.uploadBuffer(buffer, &input, sizeof(input)), "upload failed");
        probe.select(F::MixedGpuStage::Pressure);
        float output = 0.0f;
        require(compute.downloadBuffer(buffer, &output, sizeof(output)), "download failed");
        require(output == input, "profile changed transfer semantics");
        probe.select(F::MixedGpuStage::Publication);
        require(compute.uploadBuffer(buffer, &input, sizeof(input)), "second upload failed");
        probe.finish();
        probe.finish(); // Destructor and repeated finish must not double-count.
    }
    require(parent.upload_calls == 7 && parent.upload_bytes == 28,
        "outer frame totals lost or double-counted stage transfers");
    require(parent.download_calls == 1 && parent.download_bytes == sizeof(float),
        "download not propagated to frame probe");
    require(parent.synchronize_calls == 0 && parent.batch_end_calls == 0,
        "profiling inserted a GPU wait");
    float after = 1.0f;
    require(compute.uploadBuffer(buffer, &after, sizeof(after)), "restored probe upload failed");
    require(parent.upload_calls == 8, "frame probe not restored");
    const auto json = F::mixedGpuStagesJson(stats);
    require(json.at("timing_kind") == "host_wall_including_waits", "ambiguous timer units");
    require(json.at("stages").at("pressure").at("download_calls") == 1,
        "wrong phase attribution");
    require(json.at("stages").at("setup").at("upload_calls") == 1,
        "wrong setup attribution");
    require(!F::mixedGpuStagesJson(F::MixedGpuStageStats{}).at("measured").get<bool>(),
        "unmeasured must not appear as measured zero");
    F::MixedGpuStageStats early_stats;
    try {
        F::MixedGpuStageProbe early(&compute, early_stats);
        early.select(F::MixedGpuStage::Contact);
        require(compute.uploadBuffer(buffer, &after, sizeof(after)), "early path upload failed");
        throw std::runtime_error("simulated solver failure");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()) == "simulated solver failure", error.what());
    }
    require(parent.upload_calls == 9, "failure unwinding lost or duplicated outer counters");
    require(compute.uploadBuffer(buffer, &after, sizeof(after)), "post-failure upload failed");
    require(parent.upload_calls == 10, "failure unwinding did not restore the parent probe");
    compute.setTransferProbe(nullptr);
    compute.destroyBuffer(buffer);
}
