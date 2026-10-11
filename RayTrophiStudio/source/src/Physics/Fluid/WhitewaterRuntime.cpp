#include "Fluid/WhitewaterRuntime.h"
#include "ParticleSimulation.h"
#include "Fluid/MatterSubstanceState.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace RayTrophiSim::Fluid {
bool isWhitewaterFluidParcel(const FluidParticles& particles, std::size_t p,
                            const APICSolverParams* source_params) {
    if (p >= particles.size() || p >= particles.velocity.size() ||
        (p < particles.flags.size() && (particles.flags[p] & kParticleFlagFrozen)) ||
        (p < particles.mass_fraction.size() && !(particles.mass_fraction[p] > 0.02f))) {
        return false;
    }
    auto model = p < particles.constitutive_model.size()
        ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[p])
        : MatterConstitutiveModel::Auto;
    if (model == MatterConstitutiveModel::Auto && source_params) {
        const uint32_t tag = p < particles.substance_tag.size() ? particles.substance_tag[p] : 0u;
        const float kelvin = p < particles.temperature.size() ? particles.temperature[p] : 0.0f;
        model = substanceBirthModel(tag, *source_params, kelvin);
    }
    if (model != MatterConstitutiveModel::Fluid && model != MatterConstitutiveModel::Auto) {
        return false;
    }
    const auto& x = particles.position[p];
    const auto& v = particles.velocity[p];
    return std::isfinite(x.x) && std::isfinite(x.y) && std::isfinite(x.z) &&
        std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

WhitewaterBinWindow whitewaterBinWindow(const SimulationGridDomainState& state,
                                        float radius_voxels, bool device_only,
                                        const APICSolverParams* source_params) {
    const auto& grid = state.grid;
    WhitewaterBinWindow full{grid.nx, grid.ny, grid.nz, 0, 0, 0};
    if (device_only || !(grid.voxel_size > 0) || !std::isfinite(radius_voxels)) {
        return full; // Host bounds must never constrain unseen device positions.
    }
    const int size[3] = {grid.nx, grid.ny, grid.nz};
    int lo[3] = {grid.nx, grid.ny, grid.nz}, hi[3] = {-1, -1, -1};
    const auto visit = [&](const Vec3& position) {
        const Vec3 local = (position - grid.origin) / grid.voxel_size;
        const float xyz[3] = {local.x, local.y, local.z};
        if (!std::isfinite(local.x) || !std::isfinite(local.y) || !std::isfinite(local.z)) {
            return;
        }
        // Outside particles are ignored by the full-domain bin too.
        if (local.x < 0 || local.x >= grid.nx || local.y < 0 || local.y >= grid.ny ||
            local.z < 0 || local.z >= grid.nz) {
            return;
        }
        for (int axis = 0; axis < 3; ++axis) {
            const int cell = static_cast<int>(xyz[axis]);
            lo[axis] = std::min(lo[axis], cell);
            hi[axis] = std::max(hi[axis], cell);
        }
    };
    for (std::size_t p = 0; p < state.particles.size(); ++p) {
        if (isWhitewaterFluidParcel(state.particles, p, source_params)) {
            visit(state.particles.position[p]);
        }
    }
    for (const Vec3& position : state.foam.position) {
        visit(position);
    }
    if (hi[0] < 0) {
        return {}; // No in-domain source or consumer.
    }
    const int halo = static_cast<int>(std::ceil(std::max(0.0f, radius_voxels))) + 1;
    for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = std::max(0, lo[axis] - halo);
        hi[axis] = std::min(size[axis] - 1, hi[axis] + halo);
    }
    return {hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1,
            lo[0], lo[1], lo[2]};
}

void disableWhitewaterRuntime(SimulationGridDomainState& state,
                             SimulationComputeContext* compute,
                             SimulationGridDomainComputeBuffers* buffers) {
    auto& foam = state.foam;
    const bool had_storage = foam.position.capacity() || foam.velocity.capacity() ||
        foam.lifetime.capacity() || foam.type.capacity();
    if (had_storage) {
        foam.clear();
        decltype(foam.position)().swap(foam.position);
        decltype(foam.velocity)().swap(foam.velocity);
        decltype(foam.lifetime)().swap(foam.lifetime);
        decltype(foam.type)().swap(foam.type);
        ++state.version;
    }
    state.foam_stats = {};
    if (!buffers) {
        return;
    }
    if (compute && buffers->backend == compute->backendType()) {
        for (auto* handle : {&buffers->foam_positions, &buffers->foam_bin_counts,
                            &buffers->foam_bin_items, &buffers->foam_expected,
                            &buffers->foam_neigh, &buffers->foam_eligible}) {
            if (handle->valid()) {
                compute->destroyBuffer(*handle);
                *handle = {};
            }
        }
    }
    if (buffers->foam_render.count > 0) {
        ++buffers->foam_render.version;
        ++state.version;
    }
    // Rendering may still reference this shared sphere allocation. Publish an
    // empty view; its existing render-bridge/domain teardown owns retirement.
    buffers->foam_render.count = 0;
    buffers->foam_bin_cell_capacity = 0;
    buffers->foam_bin_cell_count = 0;
    buffers->foam_bin_allocated_cells = 0;
    buffers->foam_bin_window = {};
    buffers->foam_eligible_capacity = 0;
    buffers->foam_expected_capacity = 0;
    buffers->foam_expected_pending = false;
    buffers->foam_expected_pending_count = 0;
    buffers->foam_positions_capacity = 0;
    buffers->foam_neigh_capacity = 0;
    buffers->foam_neigh_pending = false;
    buffers->foam_neigh_pending_count = 0;
    buffers->foam_neigh_pending_generation = 0;
    buffers->foam_bin_ready_this_step = false;
}
}
