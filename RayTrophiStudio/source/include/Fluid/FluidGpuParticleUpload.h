#pragma once

#include "ParticleSimulation.h"

namespace RayTrophiSim::FluidGpuParticleUpload {

// The stamp belongs to one fluid step. Its producer uploads every stream before
// force integration; the force pass downloads its updated velocity before P2G.
// A CPU writer or another step must clear it before the next reuse decision.
inline void invalidate(SimulationGridDomainComputeBuffers& buffers) {
    buffers.fluid_uploaded_particle_count = 0;
}

inline void record(SimulationGridDomainComputeBuffers& buffers,
                   std::size_t particle_count,
                   bool positions_only,
                   bool upload_succeeded) {
    buffers.fluid_uploaded_particle_count =
        (upload_succeeded && !positions_only) ? particle_count : 0;
}

inline bool canReuse(const SimulationGridDomainComputeBuffers& buffers,
                     const SimulationComputeContext& compute,
                     std::size_t particle_count) {
    if (particle_count == 0 ||
        buffers.fluid_uploaded_particle_count != particle_count ||
        buffers.fluid_particle_capacity < particle_count) {
        return false;
    }
    const auto backend = compute.backendType();
    struct RequiredBuffer {
        ComputeBufferHandle handle;
        std::size_t bytes;
    };
    const RequiredBuffer required[] = {
        {buffers.fluid_positions, particle_count * sizeof(Vec3)},
        {buffers.fluid_velocities, particle_count * sizeof(Vec3)},
        {buffers.fluid_affine, particle_count * sizeof(Fluid::AffineC)},
        {buffers.fluid_mass_fraction, particle_count * sizeof(float)}
    };
    for (const auto& item : required) {
        if (!item.handle.valid() || item.handle.backend != backend ||
            compute.getBufferSize(item.handle) < item.bytes) {
            return false;
        }
    }
    return true;
}

} // namespace RayTrophiSim::FluidGpuParticleUpload
