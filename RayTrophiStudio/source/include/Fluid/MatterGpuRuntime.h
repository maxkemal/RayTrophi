#pragma once
#include "../ParticleSimulation.h"
#include "MatterGpuPartition.h"
#include "MatterGrainFluidGpuCoupling.h"

namespace RayTrophiSim::Fluid {

struct MatterGpuRuntime {
    MatterGrainGpuRuntime grain;
    MatterGrainFluidGpuStorage liquid_coupling;
    SimulationGridDomainComputeBuffers granular;
    MatterGpuPartition partition;
    ComputeBufferHandle rest_mass;
    ComputeBufferHandle transport_fraction;
    ComputeBufferHandle contact_pairs;
    ComputeBufferHandle wet_response;
    ComputeBufferHandle dry_volume;
    std::array<std::array<ComputeBufferHandle, 3>, 2> gradient;
    std::array<std::size_t, 3> face_count{};
    std::size_t particle_capacity = 0;
};

} // namespace RayTrophiSim::Fluid
