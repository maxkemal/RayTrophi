#pragma once

#include "../ParticleSimulation.h"

#include <array>
#include <string>

namespace RayTrophiSim::Fluid {

// Model grids/stress remain separately owned. Only the canonical transport
// streams are borrowed for one common step. No allocation, copy or ownership
// transfer occurs; restore also runs on every failure return.
class MatterGpuParticleLease {
public:
    MatterGpuParticleLease() = default;
    MatterGpuParticleLease(const MatterGpuParticleLease&) = delete;
    MatterGpuParticleLease& operator=(const MatterGpuParticleLease&) = delete;

    ~MatterGpuParticleLease() {
        restore();
    }

    bool bind(SimulationGridDomainComputeBuffers& target,
              const SimulationGridDomainComputeBuffers& source,
              std::size_t count, std::string& error) {
        if (target_ || &target == &source || count == 0 ||
            source.fluid_particle_capacity < count ||
            source.fluid_uploaded_particle_count != count) {
            error = "Matter shared particle lease requires fresh canonical transport";
            return false;
        }
        const auto source_handles = streams(source);
        const auto target_handles = streams(target);
        for (const auto handle : source_handles) {
            if (!handle.valid() || handle.backend != source_handles[0].backend) {
                error = "Matter shared particle lease has invalid canonical handles";
                return false;
            }
            for (const auto owned : target_handles) {
                if (owned.valid() && owned.id == handle.id && owned.backend == handle.backend) {
                    error = "Matter shared particle lease refuses existing aliased ownership";
                    return false;
                }
            }
        }
        saved_ = target_handles;
        capacity_ = target.fluid_particle_capacity;
        uploaded_ = target.fluid_uploaded_particle_count;
        target_ = &target;
        assign(target, source_handles);
        target.fluid_particle_capacity = source.fluid_particle_capacity;
        target.fluid_uploaded_particle_count = count;
        error.clear();
        return true;
    }

    void restore() {
        if (target_) {
            assign(*target_, saved_);
            target_->fluid_particle_capacity = capacity_;
            target_->fluid_uploaded_particle_count = uploaded_;
            target_ = nullptr;
        }
    }

private:
    using Streams = std::array<ComputeBufferHandle, 4>;

    static Streams streams(const SimulationGridDomainComputeBuffers& buffers) {
        return {buffers.fluid_positions, buffers.fluid_velocities,
            buffers.fluid_affine, buffers.fluid_mass_fraction};
    }

    static void assign(SimulationGridDomainComputeBuffers& buffers, const Streams& handles) {
        buffers.fluid_positions = handles[0];
        buffers.fluid_velocities = handles[1];
        buffers.fluid_affine = handles[2];
        buffers.fluid_mass_fraction = handles[3];
    }

    SimulationGridDomainComputeBuffers* target_ = nullptr;
    Streams saved_{};
    std::size_t capacity_ = 0;
    std::size_t uploaded_ = 0;
};

} // namespace RayTrophiSim::Fluid
