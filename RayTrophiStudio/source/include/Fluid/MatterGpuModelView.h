#pragma once
#include "../SimulationCompute.h"
#include "FluidGpuDispatch.h"
#include <array>

namespace RayTrophiSim::Fluid {

struct MatterGpuModelView {
    ComputeBufferHandle indices;
    ComputeBufferHandle counters;
    ComputeBufferHandle rest_mass;
    ComputeBufferHandle mass_fraction;
    ComputeBufferHandle wet_response;
    ComputeBufferHandle dry_volume;
    std::array<ComputeBufferHandle, 3> mass_gradient;
    uint32_t lane = 0;
    uint32_t boundary = 1;
    bool enabled = false;
    // The face mass gradient has one reader, the liquid/granular contact
    // (sim_matter_contact), which runs only with liquid present. Without it
    // the P2G skips the gradient's three atomics per node: 3 of its 5, ~60 %
    // of a dry MPM lane's scatter (resting snow, 2026-11-04 measured).
    bool mass_gradient_wanted = true;
    bool pressure_statics_uploaded = false;
    bool viscosity_statics_uploaded = false;
};

inline bool copyMatterGpuFloat(SimulationComputeContext& compute,
    ComputeBufferHandle source, ComputeBufferHandle target, uint32_t count) {
    const ComputeBufferHandle buffers[] = {source, target};
    ComputeDispatch command;
    command.kernel = "sim_matter_copy";
    command.groups = FluidGpuDispatch::groups256(count);
    command.buffers = buffers;
    command.buffer_count = 2;
    command.constants = &count;
    command.constants_size = sizeof(count);
    return compute.dispatch(command);
}

bool dispatchMatterGpuModel(SimulationComputeContext& compute,
    const ComputeDispatch& command, const MatterGpuModelView& model);

} // namespace RayTrophiSim::Fluid
