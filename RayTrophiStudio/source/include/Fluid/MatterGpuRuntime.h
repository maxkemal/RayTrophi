#pragma once
#include "../ParticleSimulation.h"
#include "MatterGpuPartition.h"

namespace RayTrophiSim::Fluid {

struct MatterGpuRuntime {
    MatterGrainGpuRuntime grain;
    SimulationGridDomainComputeBuffers granular;
    MatterGpuPartition partition;
    ComputeBufferHandle rest_mass;
    ComputeBufferHandle transport_fraction;
    ComputeBufferHandle contact_pairs;
    ComputeBufferHandle wet_response;
    ComputeBufferHandle dry_volume;
    std::array<std::array<ComputeBufferHandle, 3>, 2> gradient;
    // Grain coupling (B5): the liquid lane's kinematic pressure averaged over
    // its substeps with weight dt/frame_dt. Gravity is integrated once per
    // frame before the substeps, so the first substep's projection carries the
    // hydrostatic load and the last one almost nothing; only the average is the
    // pressure the frame's impulse came from. Written only when requested.
    ComputeBufferHandle frame_pressure;
    bool frame_pressure_requested = false;
    bool frame_pressure_valid = false;
    std::array<std::size_t, 3> face_count{};
    std::size_t particle_capacity = 0;
};

} // namespace RayTrophiSim::Fluid
