#pragma once

#include "../SimulationCompute.h"
#include "../Vec3.h"
#include <cstdint>
#include <cstddef>

namespace RayTrophiSim::Fluid::Granular {

// One flat GPU buffer per field. Keeping this bundle separate from the liquid
// buffers prevents an ABI change in the existing P2G/G2P kernels.
struct GpuBuffers {
    ComputeBufferHandle deformation_col0{};
    ComputeBufferHandle deformation_col1{};
    ComputeBufferHandle deformation_col2{};
    ComputeBufferHandle plastic_volume{};
    ComputeBufferHandle hardening{};
    ComputeBufferHandle material_flags{};
    ComputeBufferHandle stress_diag{};
    ComputeBufferHandle stress_shear{};
    ComputeBufferHandle yield_value{};
    ComputeBufferHandle plastic_increment{};
    ComputeBufferHandle state_flags{};
    ComputeBufferHandle damage{};
    ComputeBufferHandle fracture_history{};
    std::size_t capacity = 0;
    bool valid() const {
        return deformation_col0.valid() && deformation_col1.valid() &&
               deformation_col2.valid() && plastic_volume.valid() &&
               hardening.valid() && material_flags.valid() &&
               stress_diag.valid() && stress_shear.valid() &&
               yield_value.valid() && plastic_increment.valid() &&
               state_flags.valid() && damage.valid() && fracture_history.valid();
    }
};

struct alignas(16) ConstitutiveConstants {
    uint32_t count = 0;
    float friction_tangent = 0.7002075f;
    float cohesion = 0.0f;
    float hardening = 0.0f;
    float tensile_cutoff = 0.0f;
    float detach_pressure = 1.0e-4f;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
};
static_assert(sizeof(ConstitutiveConstants) == 32,
              "granular constitutive push constant ABI changed");

struct Counters {
    uint32_t yielded = 0;
    uint32_t detached = 0;
    uint32_t invalid = 0;
};

inline void destroy(SimulationComputeContext* compute, GpuBuffers& b) {
    if (!compute) return;
    compute->destroyBuffer(b.deformation_col0);
    compute->destroyBuffer(b.deformation_col1);
    compute->destroyBuffer(b.deformation_col2);
    compute->destroyBuffer(b.plastic_volume);
    compute->destroyBuffer(b.hardening);
    compute->destroyBuffer(b.material_flags);
    compute->destroyBuffer(b.stress_diag);
    compute->destroyBuffer(b.stress_shear);
    compute->destroyBuffer(b.yield_value);
    compute->destroyBuffer(b.plastic_increment);
    compute->destroyBuffer(b.state_flags);
    compute->destroyBuffer(b.damage);
    compute->destroyBuffer(b.fracture_history);
    b = {};
}

} // namespace RayTrophiSim::Fluid::Granular
