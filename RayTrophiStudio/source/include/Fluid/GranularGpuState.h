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
    ComputeBufferHandle softening{};
    // Cohesion/tensile multiplier. Separate from `softening` because bond
    // strength does not track stiffness through the melt (SofteningParams).
    ComputeBufferHandle bond_scale{};
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
               state_flags.valid() && damage.valid() && fracture_history.valid() &&
               softening.valid();
    }
};


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
    compute->destroyBuffer(b.softening);
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
