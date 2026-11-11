#pragma once

#include "Fluid/FluidParticles.h"
#include "Fluid/GranularGpuState.h"
#include <type_traits>

namespace RayTrophiSim::Fluid {

// The same column list drives standalone readback and mixed-frame publication.
// Transport, identities, mass, substance, pore water and model ownership are
// deliberately absent: the canonical shared particle stream owns those fields.
#define RT_GRANULAR_READBACK_COLUMNS(X) \
    X(granular_stress_diag, stress_diag) \
    X(granular_stress_shear, stress_shear) \
    X(granular_plastic_volume, plastic_volume) \
    X(granular_material_flags, state_flags) \
    X(granular_yield_value, yield_value) \
    X(granular_plastic_increment, plastic_increment) \
    X(granular_damage, damage) \
    X(granular_hardening, hardening) \
    X(granular_fracture_history, fracture_history) \
    X(granular_support_prev_vg, support_prev_vg) \
    X(granular_support_accel, support_accel) \
    X(granular_deformation_col0, deformation_col0) \
    X(granular_deformation_col1, deformation_col1) \
    X(granular_deformation_col2, deformation_col2)

template<class Columns, class Gpu, class Visitor>
void forEachGranularReadbackColumn(Columns& columns, Gpu& gpu,
                                  Visitor&& visit) {
#define RT_VISIT_COLUMN(field, buffer) visit(gpu.buffer, columns.field);
    RT_GRANULAR_READBACK_COLUMNS(RT_VISIT_COLUMN)
#undef RT_VISIT_COLUMN
}

// Enqueue only. Caller must keep columns alive and unmoved until its matching
// endTransferBatch(), including a failed enqueue after earlier successful copies.
template<class Columns>
bool enqueueGranularSidecars(SimulationComputeContext& compute,
                            const Granular::GpuBuffers& gpu, Columns& columns,
                            std::size_t count) {
    bool ok = true;
    forEachGranularReadbackColumn(columns, gpu, [&](ComputeBufferHandle handle, auto& values) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        if (values.size() < count) {
            ok = false;
            return;
        }
        if (count != 0) {
            ok = compute.downloadBuffer(handle, values.data(), count * sizeof(Value)) && ok;
        }
    });
    return ok;
}

struct GranularFrameReadback {
#define RT_DECLARE_COLUMN(field, buffer) decltype(FluidParticles::field) field;
    RT_GRANULAR_READBACK_COLUMNS(RT_DECLARE_COLUMN)
#undef RT_DECLARE_COLUMN

    explicit GranularFrameReadback(std::size_t count) {
#define RT_SIZE_COLUMN(field, buffer) field.resize(count);
        RT_GRANULAR_READBACK_COLUMNS(RT_SIZE_COLUMN)
#undef RT_SIZE_COLUMN
    }
    bool canCommit(const FluidParticles& destination, std::size_t count) const {
        bool valid = true;
#define RT_CHECK_COLUMN(field, buffer) \
        valid = valid && destination.field.size() >= count && field.size() >= count;
        RT_GRANULAR_READBACK_COLUMNS(RT_CHECK_COLUMN)
#undef RT_CHECK_COLUMN
        return valid;
    }
    void commitParticle(FluidParticles& destination, std::size_t index) const {
#define RT_COMMIT_COLUMN(field, buffer) destination.field[index] = field[index];
        RT_GRANULAR_READBACK_COLUMNS(RT_COMMIT_COLUMN)
#undef RT_COMMIT_COLUMN
    }
};

#undef RT_GRANULAR_READBACK_COLUMNS
} // namespace RayTrophiSim::Fluid
