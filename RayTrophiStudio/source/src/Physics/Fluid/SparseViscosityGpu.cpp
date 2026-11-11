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

struct PcgConstants {
    int nx, ny, nz, boundary;
    float voxel, dt, alpha, solid_weight;
    int mode, has_nu, stage;
    uint32_t partial_count, lanes_per_component;
    uint32_t tiles_x, tiles_y, tiles_z, compact_faces;
    float particles_per_cell;  // liquid volume fraction = parcel count / this
};
static_assert(sizeof(PcgConstants) == 72, "Viscosity PCG push ABI");

void releaseViscosityPcg(SimulationComputeContext& compute, ViscosityPcgStorage& storage) {
    for (const auto handle : storage.owned) {
        if (handle.valid()) {
            compute.destroyBuffer(handle);
        }
    }
    storage = {};
}

bool solveViscosityPcg(SimulationComputeContext& compute, ViscosityPcgStorage& storage,
                       const char* kernel, const ComputeBufferHandle* base,
                       const ComputeBufferHandle* tiles, const ComputeBufferHandle* mac,
                       const ViscosityGpuConstants& viscosity,
                       const std::array<uint32_t, 3>& tile_dims, uint32_t compact_faces,
                       uint32_t lanes_per_component, float particles_per_cell,
                       ViscosityPcgResult& result, std::string& error) {
    result = {};
    if (lanes_per_component == 0) {
        result.converged = true;
        return true;
    }
    const uint64_t lanes = uint64_t(lanes_per_component) * 3u;
    if (lanes > (uint64_t(std::numeric_limits<uint32_t>::max()) & ~255ull)) {
        error = "viscosity PCG lanes exceed 32-bit dispatch width";
        return false;
    }
    const auto groups = FluidGpuDispatch::groups256(static_cast<uint32_t>(lanes));
    const uint32_t partial_count = groups.groups_x * groups.groups_y;
    const std::size_t vector_bytes = static_cast<std::size_t>(lanes) * sizeof(float);
    for (std::size_t index = 0; index < 4; ++index) {
        if (!allocate(compute, storage.owned[index], vector_bytes, false, "ViscosityPcgVector")) {
            error = "viscosity PCG vector allocation failed";
            return false;
        }
    }
    if (!allocate(compute, storage.owned[4], 8 * sizeof(double), false, "ViscosityPcgScalars") ||
        !allocate(compute, storage.owned[5], std::size_t(partial_count) * sizeof(double), false,
                  "ViscosityPcgPartials")) {
        error = "viscosity PCG scalar allocation failed";
        return false;
    }
    std::array<ComputeBufferHandle, 21> bindings{};
    for (int i = 0; i < 11; ++i) bindings[i] = base[i];
    for (int i = 0; i < 6; ++i) bindings[11 + i] = storage.owned[i];
    uint32_t count = 17;
    if (tiles) {
        bindings[17] = tiles[0];
        bindings[18] = tiles[1];
        count = 19;
    }
    if (mac) {
        bindings[19] = mac[0];
        bindings[20] = mac[1];
        count = 21;
    }
    PcgConstants c{};
    c.nx = viscosity.nx; c.ny = viscosity.ny; c.nz = viscosity.nz;
    c.boundary = viscosity.boundary;
    c.voxel = viscosity.voxel_size; c.dt = viscosity.dt; c.alpha = viscosity.alpha;
    c.solid_weight = viscosity.solid_weight; c.has_nu = viscosity.has_nu;
    c.partial_count = partial_count;
    c.lanes_per_component = lanes_per_component;
    c.tiles_x = tile_dims[0]; c.tiles_y = tile_dims[1]; c.tiles_z = tile_dims[2];
    c.compact_faces = compact_faces;
    c.particles_per_cell = std::max(particles_per_cell, 1.0f);
    const auto run = [&](int mode, int stage) {
        c.mode = mode;
        c.stage = stage;
        ComputeDispatch command;
        command.kernel = kernel;
        command.buffers = bindings.data();
        command.buffer_count = count;
        command.constants = &c;
        command.constants_size = sizeof(c);
        if (mode == 4) {
            command.groups.groups_x = 1;
            command.groups.groups_y = 1;
            command.groups.groups_z = 1;
        } else {
            command.groups = groups;
        }
        return compute.dispatch(command);
    };
    double scalars[8] = {};
    const auto read = [&]() {
        compute.beginTransferBatch();
        bool ok = compute.downloadBuffer(storage.owned[4], scalars, sizeof(scalars));
        return compute.endTransferBatch() && ok;
    };
    if (!run(0, 0) || !run(4, 0)) {
        error = "viscosity PCG init dispatch failed";
        return false;
    }
    // Check every few iterations: each readback is a GPU round trip.
    constexpr int kCheckEvery = 8;
    for (int iteration = 1; iteration <= kViscosityPcgMaxIterations; ++iteration) {
        if (!run(1, 0) || !run(4, 1) || !run(2, 0) || !run(4, 2)) {
            error = "viscosity PCG iteration dispatch failed";
            return false;
        }
        if (iteration == 1 || iteration % kCheckEvery == 0 ||
            iteration == kViscosityPcgMaxIterations) {
            if (!read()) {
                error = "viscosity PCG scalar readback failed";
                return false;
            }
            const double rz0 = scalars[4];
            const double rel = rz0 > 0.0 ? std::sqrt(std::max(scalars[0], 0.0) / rz0) : 0.0;
            result.iterations = static_cast<int>(scalars[6]);
            result.relative_residual = static_cast<float>(rel);
            if (scalars[5] != 0.0 || rel < kViscosityPcgTolerance) {
                result.converged = true;
                return true;
            }
        }
        if (!run(3, 0)) {
            error = "viscosity PCG search update failed";
            return false;
        }
    }
    return true;  // ran out of iterations: result.converged stays false
}

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
                        const ViscosityGpuConstants& viscosity_constants,
                        ViscosityPcgStorage& pcg, ViscosityPcgResult& result,
                        std::string& error) {
    auto& storage = buffers.sparse_viscosity;
    storage.used = false;
    Constants constants{};
    std::memcpy(&constants, &viscosity_constants, sizeof(viscosity_constants));
    const auto fail = [&](const char* reason) {
        error = reason;
        return false;
    };
    if (constants.nx <= 0 || constants.ny <= 0 || constants.nz <= 0 ||
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
        // The RHS pages hold u_old; the solve runs to a tolerance on them.
        const ComputeBufferHandle tile_pair[] = {storage.owned[0], storage.owned[1]};
        const ComputeBufferHandle mac_pair[] = {mac.map, mac.list};
        std::string pcg_error;
        if (!solveViscosityPcg(compute, pcg,
                mac.compact ? "sim_sparse_mac_viscosity_pcg" : "sim_sparse_viscosity_pcg",
                bindings.data(), tile_pair, mac.compact ? mac_pair : nullptr,
                viscosity_constants, {constants.tiles_x, constants.tiles_y, constants.tiles_z},
                constants.compact_faces, constants.compact_faces,
                static_cast<float>(params.particles_per_cell), result, pcg_error)) {
            return fail(pcg_error.empty() ? "sparse viscosity PCG failed" : pcg_error.c_str());
        }
    } else {
        result = {};
        result.converged = true;
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
