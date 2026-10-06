#include "Fluid/MatterGpuParticleLease.h"

#include <cassert>
#include <string>

using namespace RayTrophiSim;

static SimulationGridDomainComputeBuffers buffers(uint64_t base) {
    SimulationGridDomainComputeBuffers result;
    result.fluid_positions = {base, ComputeBackendType::VulkanCompute};
    result.fluid_velocities = {base + 1, ComputeBackendType::VulkanCompute};
    result.fluid_affine = {base + 2, ComputeBackendType::VulkanCompute};
    result.fluid_mass_fraction = {base + 3, ComputeBackendType::VulkanCompute};
    result.fluid_particle_capacity = 32;
    result.fluid_uploaded_particle_count = 16;
    return result;
}

int main() {
    auto source = buffers(100);
    auto target = buffers(200);
    target.fluid_particle_capacity = 64;
    target.fluid_uploaded_particle_count = 8;
    target.vel_x = {300, ComputeBackendType::VulkanCompute};
    std::string error;
    {
        Fluid::MatterGpuParticleLease lease;
        assert(lease.bind(target, source, 16, error));
        assert(target.fluid_positions.id == 100);
        assert(target.fluid_velocities.id == 101);
        assert(target.fluid_affine.id == 102);
        assert(target.fluid_mass_fraction.id == 103);
        assert(target.vel_x.id == 300);
        assert(target.fluid_particle_capacity == 32);
        assert(target.fluid_uploaded_particle_count == 16);
        assert(!lease.bind(target, source, 16, error));
    }
    assert(target.fluid_positions.id == 200);
    assert(target.fluid_velocities.id == 201);
    assert(target.fluid_affine.id == 202);
    assert(target.fluid_mass_fraction.id == 203);
    assert(target.fluid_particle_capacity == 64);
    assert(target.fluid_uploaded_particle_count == 8);
    assert(source.fluid_positions.id == 100);
    Fluid::MatterGpuParticleLease lease;
    assert(!lease.bind(target, source, 0, error));
    assert(!lease.bind(target, source, 17, error));
    source.fluid_particle_capacity = 15;
    assert(!lease.bind(target, source, 16, error));
    source.fluid_particle_capacity = 32;
    source.fluid_affine.backend = ComputeBackendType::CPU;
    assert(!lease.bind(target, source, 16, error));
    source.fluid_affine = {};
    assert(!lease.bind(target, source, 16, error));
    source.fluid_affine = {102, ComputeBackendType::VulkanCompute};
    target.fluid_velocities = source.fluid_affine;
    assert(!lease.bind(target, source, 16, error));
    target.fluid_velocities = {201, ComputeBackendType::VulkanCompute};
    assert(lease.bind(target, source, 16, error));
    lease.restore();
    lease.restore();
    assert(target.fluid_velocities.id == 201);
    assert(target.fluid_uploaded_particle_count == 8);
}
