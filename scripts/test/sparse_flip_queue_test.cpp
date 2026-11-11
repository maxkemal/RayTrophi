// User test target links the simulation core. No Vulkan device/app required.
#include "ParticleSimulation.h"
#include "Fluid/SparseMacTransferGpu.h"
#include <stdexcept>
#include <vector>

using namespace RayTrophiSim;

static void require(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

class QueueRecorder final : public ISimulationComputeBackend {
public:
    int waits = 0;
    int calls = 0;
    int fail_at = -1;
    std::vector<std::string> kernels;
    ComputeBackendType type() const override { return ComputeBackendType::VulkanCompute; }
    const char* name() const override { return "QueueRecorder"; }
    ComputeBackendCaps caps() const override { return {}; }
    ComputeBufferHandle createBuffer(const ComputeBufferDesc&) override { return {}; }
    bool destroyBuffer(ComputeBufferHandle) override { return true; }
    bool resizeBuffer(ComputeBufferHandle, std::size_t) override { return false; }
    std::size_t getBufferSize(ComputeBufferHandle) const override { return 0; }
    bool uploadBuffer(ComputeBufferHandle, const void*, std::size_t, std::size_t) override {
        throw std::runtime_error("FLIP snapshot unexpectedly uploaded host data");
    }
    bool downloadBuffer(ComputeBufferHandle, void*, std::size_t, std::size_t) const override {
        throw std::runtime_error("FLIP snapshot unexpectedly read back to host");
    }
    void synchronize() override { ++waits; }
    bool dispatch(const ComputeDispatch& command) override {
        kernels.emplace_back(command.kernel);
        return calls++ != fail_at;
    }
};

int main() {
    SimulationComputeContext compute;
    auto recorder = std::make_unique<QueueRecorder>();
    auto* queue = recorder.get();
    compute.setBackend(std::move(recorder));
    SimulationGridDomainComputeBuffers buffers;
    auto& storage = buffers.sparse_mac_transfer;
    storage.used = true;
    storage.canonical = true;
    storage.active_tiles = 1;
    require(Fluid::captureSparseMacFlip(compute, buffers), "canonical capture failed");
    require(queue->calls == 3 && queue->waits == 0 && storage.snapshot_valid,
        "canonical capture inserted a CPU fence or lost snapshot publication");
    require(queue->kernels[0] == "sim_sparse_mac_capture_compact",
        "canonical path read a dense source");
    storage.canonical = false;
    require(Fluid::captureSparseMacFlip(compute, buffers), "legacy capture failed");
    require(queue->waits == 1 && queue->kernels[3] == "sim_sparse_mac_capture",
        "legacy host-overwrite safety fence was removed");
    storage.canonical = true;
    queue->fail_at = queue->calls + 1;
    require(!Fluid::captureSparseMacFlip(compute, buffers), "failed dispatch was accepted");
    require(queue->waits == 2 && !storage.snapshot_valid,
        "failure recovery did not fence pending work or invalidated snapshot incorrectly");
}
