#pragma once

#include "MatterGrain.h"
#include <array>

namespace RayTrophiSim::Fluid {

// Transient frame view: borrows the continuum device state already uploaded by
// the MPM lane. Only indices/mass/radius enter and compact impulses leave the GPU.
class MatterGrainMpmContact {
public:
    explicit MatterGrainMpmContact(SimulationComputeContext& compute);
    ~MatterGrainMpmContact();
    MatterGrainMpmContact(const MatterGrainMpmContact&) = delete;
    MatterGrainMpmContact& operator=(const MatterGrainMpmContact&) = delete;

    bool prepare(FluidParticles& continuum,
                 SimulationGridDomainComputeBuffers& buffers,
                 std::size_t grain_count, float grain_radius, float friction,
                 float frame_dt, std::size_t budget_bytes, std::string& error);
    bool step(MatterGrainGpuRuntime& grains, uint32_t substep, std::string& error);
    bool publish(MatterGrainStepReport& report, std::string& error,
                 bool apply_host_reaction = true);
    std::size_t workingSetBytes() const { return working_set_bytes_; }
    float maximumSpeed() const { return maximum_speed_; }
    bool active() const { return !indices_.empty(); }

private:
    SimulationComputeContext& compute_;
    FluidParticles* continuum_ = nullptr;
    std::array<ComputeBufferHandle, 13> buffers_{};
    std::vector<uint32_t> indices_;
    std::vector<float> masses_;
    // grain count, MPM count, hash buckets per owner, DEM read bank
    std::array<uint32_t, 4> meta_{};
    // grain radius, hash cell size, friction, frame dt
    std::array<float, 4> params_{};
    std::size_t working_set_bytes_ = 0;
    float maximum_speed_ = 0.0f;
    uint32_t dispatches_ = 0;
};

} // namespace RayTrophiSim::Fluid
