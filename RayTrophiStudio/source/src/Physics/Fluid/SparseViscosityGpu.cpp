#include "Fluid/SparseViscosityGpu.h"
#include "Fluid/FluidGpuDispatch.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace RayTrophiSim::Fluid {
namespace {
struct Constants {
    int nx, ny, nz, boundary;
    float voxel, dt, alpha, solid_weight;
    int parity, has_nu, pad1, pad2, pad3;
    uint32_t tiles_x, tiles_y, tiles_z, compact_faces;
};
static_assert(sizeof(Constants) == 68, "Sparse viscosity push ABI");
static_assert(offsetof(Constants, tiles_x) == sizeof(ViscosityGpuConstants),
              "Sparse viscosity base constant prefix");

bool allocate(SimulationComputeContext& compute, ComputeBufferHandle& handle,
              std::size_t bytes, bool exact = false,
              const char* name = "SparseViscosityStorage") {
    const auto limit = compute.caps().max_storage_buffer_bytes;
    if (bytes == 0 || (limit != 0 && bytes > limit)) {
        return false;
    }
    const auto capacity = compute.getBufferSize(handle);
    if (capacity >= bytes && (!exact || capacity == bytes)) {
        return true;
    }
    ComputeBufferDesc description;
    description.debug_name = name;
    description.size_bytes = bytes;
    description.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::ReadWrite |
        ComputeBufferUsage::Upload | ComputeBufferUsage::Download;
    const auto candidate = compute.createBuffer(description);
    if (!candidate.valid()) {
        return false;
    }
    if (handle.valid()) {
        compute.destroyBuffer(handle);
    }
    handle = candidate;
    return true;
}
} // namespace

void releaseSparseViscosity(SimulationComputeContext& compute,
                            SparseViscosityGpuStorage& storage) {
    for (const auto handle : storage.owned) {
        if (handle.valid()) {
            compute.destroyBuffer(handle);
        }
    }
    storage = {};
}

bool ensureMacRhsScratch(SimulationComputeContext& compute,
                         SimulationGridDomainComputeBuffers& buffers,
                         const std::array<std::size_t, 3>& faces, bool compact) {
    ComputeBufferHandle* handles[] = {
        &buffers.scratch2_vel_x, &buffers.scratch2_vel_y, &buffers.scratch2_vel_z
    };
    if (compact) {
        for (auto* handle : handles) {
            if (handle->valid()) {
                compute.destroyBuffer(*handle);
                *handle = {};
            }
        }
        return true;
    }
    releaseSparseViscosity(compute, buffers.sparse_viscosity);
    for (std::size_t component = 0; component < 3; ++component) {
        if (faces[component] > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
            !allocate(compute, *handles[component], faces[component] * sizeof(float),
                      false, "DenseMacRhs")) {
            return false;
        }
    }
    return true;
}

bool runSparseViscosity(SimulationComputeContext& compute,
                        SimulationGridDomainComputeBuffers& buffers,
                        const APICSolverParams& params,
                        const ViscosityGpuConstants& viscosity_constants, int sweeps,
                        std::string& error) {
    auto& storage = buffers.sparse_viscosity;
    storage.used = false;
    Constants constants{};
    std::memcpy(&constants, &viscosity_constants, sizeof(viscosity_constants));
    const auto fail = [&](const char* reason) {
        error = reason;
        return false;
    };
    if (sweeps <= 0 || constants.nx <= 0 || constants.ny <= 0 || constants.nz <= 0 ||
        !std::isfinite(constants.dt) || constants.dt <= 0.0f ||
        !std::isfinite(constants.voxel) || constants.voxel <= 0.0f ||
        !std::isfinite(constants.alpha) || constants.alpha < 0.0f ||
        !std::isfinite(constants.solid_weight) || constants.solid_weight < 0.0f ||
        constants.solid_weight > 1.0f) {
        return fail("invalid sparse viscosity layout/timestep");
    }
    const uint64_t xy = uint64_t(constants.nx) * constants.ny;
    if (xy > uint64_t(std::numeric_limits<int32_t>::max()) / uint64_t(constants.nz)) {
        return fail("viscosity mask exceeds signed shader index width");
    }
    const uint64_t cells = xy * constants.nz;
    constants.tiles_x = (uint32_t(constants.nx) + 7u) / 8u;
    constants.tiles_y = (uint32_t(constants.ny) + 7u) / 8u;
    constants.tiles_z = (uint32_t(constants.nz) + 7u) / 8u;
    const uint64_t tiles = uint64_t(constants.tiles_x) * constants.tiles_y * constants.tiles_z;
    const uint64_t lookup_bytes = (tiles * 2u + 1u) * sizeof(uint32_t);
    const auto budget = params.mixed_working_set_budget_bytes;
    if (budget != 0 && lookup_bytes + 12u > budget) {
        releaseSparseViscosity(compute, storage);
        return fail("MAC tile lookup exceeds authored working budget");
    }
    if (!allocate(compute, storage.owned[0], tiles * sizeof(uint32_t), budget != 0) ||
        !allocate(compute, storage.owned[1], (tiles + 1u) * sizeof(uint32_t), budget != 0)) {
        return fail("MAC tile lookup allocation failed");
    }
    for (std::size_t index = 2; index < 5; ++index) {
        if (!allocate(compute, storage.owned[index], sizeof(float))) {
            return fail("MAC RHS bootstrap allocation failed");
        }
    }
    // Velocity is the dense bank, or the canonical compact MAC pages; then the
    // capture/sweep twins read it through the MAC tile map and list
    // (docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md). The RHS pages keep this pool's
    // own cell-tile layout either way.
    const auto mac = macVelocityBinding(buffers);
    std::array<ComputeBufferHandle, 15> bindings{};
    const auto bind = [&]() {
        bindings = {mac.velocity[0], mac.velocity[1], mac.velocity[2], buffers.fluid_mask,
            storage.owned[2], storage.owned[3], storage.owned[4],
            buffers.var_svx, buffers.var_svy, buffers.var_svz,
            buffers.substance_viscosity.valid() ? buffers.substance_viscosity : buffers.fluid_mask,
            storage.owned[0], storage.owned[1], mac.map, mac.list};
    };
    bind();
    const auto dispatch = [&](const char* kernel, uint64_t lanes) {
        const bool velocity_kernel = std::strcmp(kernel, "sim_sparse_viscosity_capture") == 0 ||
            std::strcmp(kernel, "sim_sparse_viscosity_sweep") == 0;
        ComputeDispatch command;
        command.kernel = kernel;
        if (mac.compact && velocity_kernel) {
            command.kernel = std::strcmp(kernel, "sim_sparse_viscosity_capture") == 0
                ? "sim_sparse_mac_viscosity_capture" : "sim_sparse_mac_viscosity_sweep";
        }
        command.groups = FluidGpuDispatch::groups256(static_cast<uint32_t>(lanes));
        command.buffers = bindings.data();
        command.buffer_count = mac.compact && velocity_kernel ? 15 : 13;
        command.constants = &constants;
        command.constants_size = sizeof(constants);
        return compute.dispatch(command);
    };
    if (!dispatch("sim_sparse_viscosity_clear", tiles) ||
        !dispatch("sim_sparse_viscosity_mark", cells)) {
        return fail("MAC tile classification dispatch failed");
    }
    uint32_t active = 0;
    compute.beginTransferBatch();
    bool ok = compute.downloadBuffer(storage.owned[1], &active, sizeof(active));
    ok = compute.endTransferBatch() && ok;
    if (!ok || active > tiles) {
        return fail("MAC tile count readback failed");
    }
    const uint64_t compact_faces = uint64_t(active) * 576u;
    if (compact_faces > (uint64_t(std::numeric_limits<uint32_t>::max()) & ~255ull)) {
        return fail("MAC page exceeds 32-bit dispatch lane width");
    }
    constants.compact_faces = static_cast<uint32_t>(compact_faces);
    const auto bytes = std::max<uint64_t>(compact_faces, 1u) * sizeof(float);
    if (budget != 0 && lookup_bytes + 3u * bytes > budget) {
        releaseSparseViscosity(compute, storage);
        return fail("MAC RHS pages exceed authored working budget");
    }
    uint64_t retained = lookup_bytes;
    for (std::size_t index = 2; index < 5; ++index) {
        retained += std::max<uint64_t>(bytes, compute.getBufferSize(storage.owned[index]));
    }
    const bool trim = budget != 0 && retained > budget;
    for (std::size_t index = 2; index < 5; ++index) {
        if (!allocate(compute, storage.owned[index], bytes, trim)) {
            return fail("MAC RHS page allocation failed");
        }
    }
    bind();
    if (compact_faces != 0) {
        if (!dispatch("sim_sparse_viscosity_capture", compact_faces)) {
            return fail("MAC RHS capture failed");
        }
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            for (int parity = 0; parity < 2; ++parity) {
                constants.parity = parity;
                if (!dispatch("sim_sparse_viscosity_sweep", compact_faces)) {
                    return fail("sparse viscosity sweep failed");
                }
            }
        }
    }
    storage.used = true;
    storage.active_tiles = active;
    storage.allocated_tiles = compute.getBufferSize(storage.owned[2]) / (576u * sizeof(float));
    storage.resident_bytes = 0;
    for (const auto handle : storage.owned) {
        storage.resident_bytes += compute.getBufferSize(handle);
    }
    return true;
}
} // namespace RayTrophiSim::Fluid
