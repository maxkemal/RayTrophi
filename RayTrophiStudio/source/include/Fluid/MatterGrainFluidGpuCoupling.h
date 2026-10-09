#pragma once

#include "MatterGrainCoupling.h"
#include <array>

namespace RayTrophiSim::Fluid {

struct MatterGrainFluidGpuStorage {
    std::array<ComputeBufferHandle, 21> handles{};
};
void releaseMatterGrainFluidGpuStorage(SimulationComputeContext& compute,
                                     MatterGrainFluidGpuStorage& storage);

// Rebins canonical positions on the device each tick. The existing Di Felice
// lump-mass partition and trilinear support are evaluated at current geometry.
class MatterGrainFluidGpuCoupling {
public:
    explicit MatterGrainFluidGpuCoupling(SimulationComputeContext& compute,
        MatterGrainFluidGpuStorage* storage = nullptr);
    ~MatterGrainFluidGpuCoupling();
    MatterGrainFluidGpuCoupling(const MatterGrainFluidGpuCoupling&) = delete;
    MatterGrainFluidGpuCoupling& operator=(const MatterGrainFluidGpuCoupling&) = delete;
    bool prepare(const FluidParticles& continuum, const MatterGrainCouplingFrame& frame,
        SimulationGridDomainComputeBuffers& buffers, MatterGrainGpuRuntime& grains,
        std::size_t budget_bytes, std::string& error,
        const APICSolverParams& params, const MatterGrainMotion& motion);
    bool refresh(uint32_t index, float dt, std::string& error);
    bool react(uint32_t index, float dt, std::string& error);
    bool publish(const std::vector<MatterGrainCouplingOutput>& drag,
                 MatterGrainStepReport& report, std::string& error);
    std::size_t workingSetBytes() const { return bytes_; }

private:
    bool dispatch(const char* kernel, uint32_t threads, uint32_t index,
                  float dt, std::string& error);
    SimulationComputeContext& compute_;
    MatterGrainFluidGpuStorage* storage_ = nullptr;
    std::array<ComputeBufferHandle, 21> buffers_{};
    std::array<uint32_t, 4> counts_{};
    std::array<float, 4> origin_h_{};
    std::array<float, 4> material_{};
    std::array<uint32_t, 4> dimensions_{};
    std::array<float, 4> gravity_{};
    std::size_t bytes_ = 0;
    std::size_t upload_bytes_ = 0;
    uint64_t dispatches_ = 0;
};

} // namespace RayTrophiSim::Fluid
