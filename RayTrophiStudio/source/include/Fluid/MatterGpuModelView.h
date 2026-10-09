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
