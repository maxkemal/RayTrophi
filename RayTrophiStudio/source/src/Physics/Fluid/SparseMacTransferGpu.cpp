#include "Fluid/SparseMacTransferGpu.h"
#include "Fluid/FluidGpuDispatch.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace RayTrophiSim::Fluid {
namespace {
bool allocate(SimulationComputeContext& compute, ComputeBufferHandle& handle,
              uint64_t bytes, bool exact) {
    const auto limit = compute.caps().max_storage_buffer_bytes;
    if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max() ||
        (limit != 0 && bytes > limit)) {
        return false;
    }
    const auto capacity = compute.getBufferSize(handle);
    if (handle.valid() && capacity >= bytes && (!exact || capacity == bytes)) {
        return true;
    }
    ComputeBufferDesc desc;
    desc.debug_name = "SparseMacTransfer";
    desc.size_bytes = static_cast<std::size_t>(bytes);
    desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::ReadWrite |
        ComputeBufferUsage::Upload | ComputeBufferUsage::Download;
    auto candidate = compute.createBuffer(desc);
    if (!candidate.valid()) {
        return false;
    }
    if (handle.valid()) {
        compute.destroyBuffer(handle);
    }
    handle = candidate;
    return true;
}

bool operation(SimulationComputeContext& compute, SimulationGridDomainComputeBuffers& buffers,
               SparseMacTransferConstants& constants, const char* kernel, uint32_t lanes) {
    const auto& pool = buffers.sparse_mac_transfer.owned;
    const std::array<ComputeBufferHandle, 3> velocity = {
        buffers.vel_x, buffers.vel_y, buffers.vel_z};
    const std::array<ComputeBufferHandle, 3> weights = {
        buffers.temperature, buffers.fuel, buffers.scratch_scalar};
    const auto axis = static_cast<std::size_t>(constants.component);
    // A compact owner has no dense bank: those two bindings are never read
    // by the page-only kernels it dispatches, but every descriptor must be bound.
    const auto dense = [&](ComputeBufferHandle handle, ComputeBufferHandle page) {
        return handle.valid() ? handle : page;
    };
    const ComputeBufferHandle bindings[] = {
        buffers.fluid_positions, dense(velocity[axis], pool[2 + axis]),
        dense(weights[axis], pool[5 + axis]), pool[2 + axis],
        pool[5 + axis], pool[0], pool[1], pool[8 + axis]};
    ComputeDispatch command;
    command.kernel = kernel;
    command.buffers = bindings;
    command.buffer_count = 8;
    command.constants = &constants;
    command.constants_size = sizeof(constants);
    command.groups = FluidGpuDispatch::groups256(lanes);
    return lanes == 0 || compute.dispatch(command);
}
} // namespace

void releaseSparseMacTransfer(SimulationComputeContext& compute,
                              SparseMacTransferGpuStorage& storage) {
    for (const auto handle : storage.owned) {
        if (handle.valid()) {
            compute.destroyBuffer(handle);
        }
    }
    const bool blocked = storage.compact_blocked;
    const auto status = storage.status;
    storage = {};
    storage.compact_blocked = blocked;
    storage.status = status;
}

bool runSparseMacP2G(SimulationComputeContext& compute,
                     SimulationGridDomainComputeBuffers& buffers,
                     const APICSolverParams& params,
                     SparseMacTransferConstants constants, bool canonical,
                     std::string& error) {
    auto& storage = buffers.sparse_mac_transfer;
    storage.used = false;
    storage.canonical = false;
    storage.snapshot_valid = false;
    storage.flip_gather_used = false;
    storage.solid_weights_ready = false;
    const auto fail = [&](const char* reason) {
        // Queued compact work must complete before dense/host recovery writes.
        compute.synchronize();
        storage.status = reason;
        storage.compact_blocked = storage.compact_blocked || canonical;
        releaseSparseMacTransfer(compute, storage);
        error = reason;
        return false;
    };
    if (compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch() || params.granular_enabled ||
        params.boundary == APICSolverParams::BoundaryMode::Periodic ||
        constants.nx <= 0 || constants.ny <= 0 || constants.nz <= 0 ||
        constants.particle_count <= 0 ||
        constants.particle_count > std::numeric_limits<int32_t>::max() / 9 ||
        !std::isfinite(constants.voxel_size) || constants.voxel_size <= 1e-6f ||
        !std::isfinite(constants.origin_x) || !std::isfinite(constants.origin_y) ||
        !std::isfinite(constants.origin_z)) {
        return fail("unsupported sparse MAC transfer layout/backend/boundary");
    }
    const uint64_t nx = constants.nx, ny = constants.ny, nz = constants.nz;
    const uint64_t signed_limit = std::numeric_limits<int32_t>::max();
    if (nx * ny > signed_limit / nz) {
        return fail("sparse MAC domain exceeds signed shader index width");
    }
    const std::array<uint64_t, 3> faces = {
        (nx + 1u) * ny * nz, nx * (ny + 1u) * nz, nx * ny * (nz + 1u)};
    if (*std::max_element(faces.begin(), faces.end()) >
        uint64_t(std::numeric_limits<int32_t>::max()) / (buffers.matter_model.enabled ? 3u : 1u)) {
        return fail("sparse MAC publication exceeds shader index width");
    }
    const uint64_t tiles = ((nx + 7u) / 8u) * ((ny + 7u) / 8u) * ((nz + 7u) / 8u);
    const uint64_t lookup_bytes = (tiles * 2u + 1u) * sizeof(uint32_t);
    const uint64_t budget = params.mixed_working_set_budget_bytes;
    const uint64_t particles = static_cast<uint64_t>(constants.particle_count);
    const std::size_t pool_end = canonical && params.variational_solids ? 14u : 11u;
    const uint64_t page_fields = pool_end - 2u;
    for (std::size_t index = pool_end; index < storage.owned.size(); ++index) {
        if (storage.owned[index].valid()) {
            compute.destroyBuffer(storage.owned[index]);
            storage.owned[index] = {};
        }
    }
    const std::array<ComputeBufferHandle, 3> particle_fields = {
        buffers.fluid_positions, buffers.fluid_velocities, buffers.fluid_affine};
    const std::array<ComputeBufferHandle, 3> dense_velocity = {
        buffers.vel_x, buffers.vel_y, buffers.vel_z};
    const std::array<ComputeBufferHandle, 3> dense_weight = {
        buffers.temperature, buffers.fuel, buffers.scratch_scalar};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const uint64_t particle_bytes = particles * (axis == 2 ? 9u : 3u) * sizeof(float);
        if (!particle_fields[axis].valid() ||
            compute.getBufferSize(particle_fields[axis]) < particle_bytes) {
            return fail("sparse MAC particle input buffer capacity mismatch");
        }
        // Only a publication writes the dense bank.
        if (!canonical && (!dense_velocity[axis].valid() || !dense_weight[axis].valid() ||
            compute.getBufferSize(dense_velocity[axis]) < faces[axis] * sizeof(float) ||
            compute.getBufferSize(dense_weight[axis]) < faces[axis] * sizeof(float))) {
            return fail("sparse MAC publication buffer capacity mismatch");
        }
    }
    if (budget != 0 && lookup_bytes + page_fields * sizeof(float) > budget) {
        return fail("sparse MAC lookup exceeds authored working budget");
    }
    if (!allocate(compute, storage.owned[0], tiles * sizeof(uint32_t), budget != 0) ||
        !allocate(compute, storage.owned[1], (tiles + 1u) * sizeof(uint32_t), budget != 0)) {
        return fail("sparse MAC lookup allocation failed");
    }
    for (std::size_t index = 2; index < pool_end; ++index) {
        if (!allocate(compute, storage.owned[index], sizeof(float), false)) {
            return fail("sparse MAC bootstrap allocation failed");
        }
    }
    constants.component = 0;
    if (!operation(compute, buffers, constants, "sim_sparse_mac_clear", uint32_t(tiles)) ||
        !operation(compute, buffers, constants, "sim_sparse_mac_mark",
                   uint32_t(constants.particle_count))) {
        return fail("sparse MAC topology dispatch failed");
    }
    // Count only: no particle/grid readback or host domain occupancy bitmap.
    uint32_t active = 0;
    compute.beginTransferBatch();
    bool ok = compute.downloadBuffer(storage.owned[1], &active, sizeof(active));
    ok = compute.endTransferBatch() && ok;
    if (!ok || active > tiles || uint64_t(active) * 576u >
        uint64_t(std::numeric_limits<int32_t>::max())) {
        return fail("sparse MAC tile count/compact index overflow");
    }
    const auto lanes = static_cast<uint32_t>(uint64_t(active) * 576u);
    const uint64_t bytes = std::max<uint64_t>(lanes, 1u) * sizeof(float);
    if (budget != 0 && lookup_bytes + page_fields * bytes > budget) {
        return fail("sparse MAC pages exceed authored working budget");
    }
    uint64_t retained = lookup_bytes;
    for (std::size_t index = 2; index < pool_end; ++index) {
        retained += std::max<uint64_t>(bytes, compute.getBufferSize(storage.owned[index]));
    }
    const bool trim = budget != 0 && retained > budget;
    for (std::size_t index = 2; index < pool_end; ++index) {
        if (!allocate(compute, storage.owned[index], bytes, trim)) {
            return fail("sparse MAC page allocation failed");
        }
    }
    for (int component = 0; component < 3; ++component) {
        constants.component = component;
        if (!operation(compute, buffers, constants, "sim_sparse_mac_reset", lanes)) {
            return fail("sparse MAC page clear failed");
        }
        const auto& pool = storage.owned;
        const ComputeBufferHandle bindings[] = {
            buffers.fluid_positions, buffers.fluid_velocities, buffers.fluid_affine,
            pool[2 + component], pool[5 + component], pool[0], pool[1]};
        ComputeDispatch scatter;
        scatter.kernel = "sim_sparse_mac_p2g";
        scatter.buffers = bindings;
        scatter.buffer_count = 7;
        scatter.constants = &constants;
        scatter.constants_size = sizeof(constants);
        scatter.groups = FluidGpuDispatch::groups256(uint32_t(constants.particle_count));
        if (!dispatchMatterGpuModel(compute, scatter, buffers.matter_model) ||
            !operation(compute, buffers, constants, "sim_sparse_mac_normalize", lanes) ||
            (!canonical && !operation(compute, buffers, constants, "sim_sparse_mac_publish",
                                      static_cast<uint32_t>(faces[component])))) {
            return fail("sparse MAC P2G/normalize/publication failed");
        }
    }
    storage.nx = constants.nx;
    storage.ny = constants.ny;
    storage.nz = constants.nz;
    storage.active_tiles = active;
    storage.allocated_tiles = compute.getBufferSize(storage.owned[2]) / (576u * sizeof(float));
    storage.resident_bytes = 0;
    for (const auto handle : storage.owned) {
        storage.resident_bytes += compute.getBufferSize(handle);
    }
    storage.status = canonical
        ? "compact canonical MAC velocity (no dense bank)"
        : "compact P2G/FLIP with dense projection/contact publication";
    storage.canonical = canonical;
    storage.used = true;
    return true;
}

bool captureSparseMacFlip(SimulationComputeContext& compute,
                          SimulationGridDomainComputeBuffers& buffers) {
    auto& storage = buffers.sparse_mac_transfer;
    storage.snapshot_valid = false;
    if (!storage.used) {
        return false;
    }
    SparseMacTransferConstants constants;
    constants.nx = storage.nx;
    constants.ny = storage.ny;
    constants.nz = storage.nz;
    // A canonical field is already on the pages: the baseline is a page copy.
    const char* kernel = storage.canonical ? "sim_sparse_mac_capture_compact"
                                           : "sim_sparse_mac_capture";
    for (int component = 0; component < 3; ++component) {
        constants.component = component;
        if (!operation(compute, buffers, constants, kernel,
                       static_cast<uint32_t>(storage.active_tiles * 576u))) {
            compute.synchronize();
            return false;
        }
    }
    // A CPU boundary upload may directly overwrite host-visible source memory.
    compute.synchronize();
    storage.snapshot_valid = true;
    return true;
}

bool captureSparseMacPost(SimulationComputeContext& compute,
                          SimulationGridDomainComputeBuffers& buffers) {
    const auto& storage = buffers.sparse_mac_transfer;
    if (!storage.used) {
        return false;
    }
    if (storage.canonical) {
        return true;  // projection and contact wrote the pages themselves
    }
    SparseMacTransferConstants constants;
    constants.nx = storage.nx;
    constants.ny = storage.ny;
    constants.nz = storage.nz;
    for (int component = 0; component < 3; ++component) {
        constants.component = component;
        if (!operation(compute, buffers, constants, "sim_sparse_mac_gather",
                       static_cast<uint32_t>(storage.active_tiles * 576u))) {
            compute.synchronize();
            return false;
        }
    }
    return true;
}

bool releaseDenseMacForCompactOwner(SimulationComputeContext& compute,
                                    SimulationGridDomainComputeBuffers& buffers) {
    if (!buffers.sparse_mac_transfer.compact_owner) {
        return false;
    }
    for (auto* handle : {&buffers.vel_x, &buffers.vel_y, &buffers.vel_z,
                         &buffers.scratch_vel_x, &buffers.scratch_vel_y,
                         &buffers.scratch_vel_z, &buffers.temperature,
                         &buffers.fuel, &buffers.scratch_scalar,
                         &buffers.var_u_weight, &buffers.var_v_weight,
                         &buffers.var_w_weight}) {
        if (handle->valid()) {
            compute.destroyBuffer(*handle);
        }
        *handle = {};
    }
    return true;
}

MacVelocityBinding macVelocityBinding(const SimulationGridDomainComputeBuffers& buffers) {
    MacVelocityBinding binding;
    const auto& storage = buffers.sparse_mac_transfer;
    if (storage.used && storage.canonical) {
        const auto& pool = storage.owned;
        binding.velocity = {pool[2], pool[3], pool[4]};
        binding.weight = {pool[5], pool[6], pool[7]};
        binding.flip = {pool[8], pool[9], pool[10]};
        binding.solid_weight = {pool[11], pool[12], pool[13]};
        binding.map = pool[0];
        binding.list = pool[1];
        binding.compact = true;
        binding.face_lanes = static_cast<uint32_t>(storage.active_tiles * 576u);
        return binding;
    }
    binding.velocity = {buffers.vel_x, buffers.vel_y, buffers.vel_z};
    binding.weight = {buffers.temperature, buffers.fuel, buffers.scratch_scalar};
    binding.flip = {buffers.scratch_vel_x, buffers.scratch_vel_y, buffers.scratch_vel_z};
    binding.solid_weight = {buffers.var_u_weight, buffers.var_v_weight,
                            buffers.var_w_weight};
    return binding;
}

bool publishCompactMacToHost(SimulationComputeContext& compute,
                             const SimulationGridDomainComputeBuffers& buffers,
                             const std::array<std::vector<float>*, 3>& host,
                             std::string& error) {
    const auto& storage = buffers.sparse_mac_transfer;
    if (!storage.used || !storage.canonical) {
        error = "compact MAC host publication without a canonical compact field";
        return false;
    }
    const std::size_t nx = std::size_t(storage.nx);
    const std::size_t ny = std::size_t(storage.ny);
    const std::size_t nz = std::size_t(storage.nz);
    const std::array<std::size_t, 3> cells = {nx, ny, nz};
    for (int axis = 0; axis < 3; ++axis) {
        std::array<std::size_t, 3> dims = cells;
        ++dims[axis];
        if (!host[axis] || host[axis]->size() != dims[0] * dims[1] * dims[2]) {
            error = "compact MAC host publication: host face array size mismatch";
            return false;
        }
    }
    const std::size_t tx = (nx + 7u) / 8u;
    const std::size_t ty = (ny + 7u) / 8u;
    const std::size_t tz = (nz + 7u) / 8u;
    const std::size_t active = std::size_t(storage.active_tiles);
    const std::size_t values = active * 576u;
    std::vector<uint32_t> list(active + 1u, 0u);
    std::array<std::vector<float>, 3> pages;
    compute.beginTransferBatch();
    bool ok = compute.downloadBuffer(storage.owned[1], list.data(),
                                     list.size() * sizeof(uint32_t));
    for (int axis = 0; axis < 3 && values != 0; ++axis) {
        pages[axis].resize(values);
        ok = compute.downloadBuffer(storage.owned[2 + axis], pages[axis].data(),
                                    values * sizeof(float)) && ok;
    }
    ok = compute.endTransferBatch() && ok;
    if (!ok || list[0] != active) {
        error = "compact MAC host publication readback failed";
        return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        std::array<std::size_t, 3> dims = cells;
        ++dims[axis];
        std::vector<float> dense(dims[0] * dims[1] * dims[2], 0.0f);
        std::array<std::size_t, 3> extent = {8u, 8u, 8u};
        extent[axis] = 9u;
        for (std::size_t slot = 0; slot < active; ++slot) {
            const std::size_t key = list[slot + 1u];
            if (key >= tx * ty * tz) {
                error = "compact MAC host publication: tile key out of range";
                return false;
            }
            const std::array<std::size_t, 3> tile = {key % tx, (key / tx) % ty, key / (tx * ty)};
            for (std::size_t local = 0; local < 576u; ++local) {
                const std::array<std::size_t, 3> face = {
                    tile[0] * 8u + local % extent[0],
                    tile[1] * 8u + (local / extent[0]) % extent[1],
                    tile[2] * 8u + local / (extent[0] * extent[1])};
                if (face[0] >= dims[0] || face[1] >= dims[1] || face[2] >= dims[2]) {
                    continue;
                }
                // sparseMacOwned: the positive-side tile owns an interior face,
                // the last cell's tile the final domain face.
                std::array<std::size_t, 3> owner = face;
                owner[axis] = std::min(owner[axis], cells[axis] - 1u);
                if (owner[0] / 8u != tile[0] || owner[1] / 8u != tile[1] ||
                    owner[2] / 8u != tile[2]) {
                    continue;
                }
                const float value = pages[axis][slot * 576u + local];
                if (!std::isfinite(value)) {
                    error = "compact MAC host publication: nonfinite velocity";
                    return false;
                }
                dense[face[0] + dims[0] * (face[1] + dims[1] * face[2])] = value;
            }
        }
        *host[axis] = std::move(dense);
    }
    return true;
}

void publishSparseMacTransferStats(const SparseMacTransferGpuStorage& storage,
                                   APICSolverStats& stats) {
    stats.transfer_sparse_used = storage.used && stats.p2g_on_gpu;
    stats.flip_sparse_used = stats.transfer_sparse_used && storage.flip_gather_used &&
        stats.g2p_on_gpu;
    stats.transfer_sparse_active_tiles = stats.transfer_sparse_used ? storage.active_tiles : 0;
    stats.transfer_sparse_allocated_tiles = stats.transfer_sparse_used
        ? storage.allocated_tiles : 0;
    stats.transfer_sparse_resident_bytes = stats.transfer_sparse_used ? storage.resident_bytes : 0;
    stats.transfer_sparse_status = storage.status;
    stats.transfer_sparse_canonical = stats.transfer_sparse_used && storage.canonical;
    stats.transfer_sparse_blocked = storage.compact_blocked;
}
} // namespace RayTrophiSim::Fluid
