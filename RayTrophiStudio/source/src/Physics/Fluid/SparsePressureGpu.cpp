#include "Fluid/SparsePressureGpu.h"
#include "Fluid/FluidGpuDispatch.h"
#include "Fluid/SparseMacTransferGpu.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace RayTrophiSim::Fluid {
namespace {

struct Constants {
    int nx, ny, nz, boundary;
    float voxel_size, dt, sor_omega;
    int iterations, parity;
    float density_correction;
    int particles_per_cell, variational, gfm_active;
    uint32_t tiles_x, tiles_y, tiles_z, tile_count;
    uint32_t compact_count, partial_count, reserved;
};
static_assert(sizeof(Constants) == 80, "Sparse pressure push constant ABI");

constexpr auto usage = ComputeBufferUsage::Storage;

bool allocate(SimulationComputeContext& compute, ComputeBufferHandle& buffer,
              std::size_t bytes, const char* name, bool exact = false) {
    const auto limit = compute.caps().max_storage_buffer_bytes;
    if (bytes == 0 || (limit != 0 && bytes > limit)) {
        return false;
    }
    if (buffer.valid() && compute.getBufferSize(buffer) >= bytes &&
        (!exact || compute.getBufferSize(buffer) == bytes)) {
        return true;
    }
    const ComputeBufferDesc desc = {
        name, bytes, usage | ComputeBufferUsage::Upload | ComputeBufferUsage::Download |
            ComputeBufferUsage::ReadWrite
    };
    const auto candidate = compute.createBuffer(desc);
    if (!candidate.valid()) {
        return false;
    }
    if (buffer.valid()) {
        compute.destroyBuffer(buffer);
    }
    buffer = candidate;
    return true;
}

ComputeDispatchSize groups(uint64_t count) {
    return FluidGpuDispatch::groups256(static_cast<uint32_t>(count));
}

} // namespace

bool usesSparsePressure(const SimulationComputeContext& compute, bool sparse,
                        const APICSolverParams& params) {
    return sparse && compute.backendType() == ComputeBackendType::VulkanCompute &&
        !params.ghost_fluid_surface && params.boundary != APICSolverParams::BoundaryMode::Periodic;
}

void releaseSparsePressure(SimulationComputeContext& compute,
                           SparsePressureGpuStorage& storage) {
    for (auto& buffer : storage.owned) {
        if (buffer.valid()) {
            compute.destroyBuffer(buffer);
        }
    }
    storage = {};
}

bool ensurePressureScratch(SimulationComputeContext& compute,
                           SimulationGridDomainComputeBuffers& buffers,
                           std::size_t cells, bool compact) {
    ComputeBufferHandle* dense[] = {
        &buffers.cg_residual, &buffers.cg_z, &buffers.cg_search, &buffers.cg_As,
        &buffers.cg_diag, &buffers.cg_partials, &buffers.cg_scalars
    };
    if (compact) {
        for (auto* buffer : dense) {
            if (buffer->valid()) {
                compute.destroyBuffer(*buffer);
                *buffer = {};
            }
        }
        return true;
    }
    releaseSparsePressure(compute, buffers.sparse_pressure);
    if (cells > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return false;
    }
    const char* names[] = {"GridDomainCGResidual", "GridDomainCGZ", "GridDomainCGSearch",
        "GridDomainCGAs", "GridDomainCGDiag", "GridDomainCGPartials", "GridDomainCGScalars"};
    for (std::size_t index = 0; index < 7; ++index) {
        const auto bytes = index < 5 ? cells * sizeof(float)
            : (index == 5 ? ((cells + 255u) / 256u) * sizeof(double) : 7u * sizeof(double));
        if (!allocate(compute, *dense[index], bytes, names[index])) {
            return false;
        }
    }
    return true;
}

bool solveSparsePressure(SimulationComputeContext& compute,
                         SimulationGridDomainComputeBuffers& buffers,
                         const APICSolverParams& params, int nx, int ny, int nz,
                         float voxel, float dt, bool variational,
                         APICSolverStats* stats, std::string& error) {
    auto& storage = buffers.sparse_pressure;
    storage.used = false;
    storage.dispatches = 0;
    storage.download_bytes = 0;
    const auto fail = [&](const char* message) {
        error = message;
        return false;
    };
    if (nx <= 0 || ny <= 0 || nz <= 0 || !std::isfinite(voxel) || voxel <= 0.0f ||
        !std::isfinite(dt) || dt <= 0.0f) {
        return fail("invalid sparse pressure physical layout");
    }
    const uint64_t xy = uint64_t(nx) * ny;
    // Existing dense mask/divergence/gradient ABIs use signed 32-bit cell indices.
    if (xy > uint64_t(std::numeric_limits<int32_t>::max()) / uint64_t(nz)) {
        return fail("pressure mask/gradient exceeds signed shader index width");
    }
    const uint64_t cells = xy * nz;
    Constants constants{};
    constants.nx = nx;
    constants.ny = ny;
    constants.nz = nz;
    constants.boundary = params.boundary == APICSolverParams::BoundaryMode::Open ? 0
        : (params.boundary == APICSolverParams::BoundaryMode::Periodic ? 2 : 1);
    constants.voxel_size = voxel;
    constants.dt = dt;
    constants.density_correction = params.density_correction;
    constants.particles_per_cell = params.particles_per_cell;
    constants.variational = variational ? 1 : 0;
    constants.tiles_x = (uint32_t(nx) + 7u) / 8u;
    constants.tiles_y = (uint32_t(ny) + 7u) / 8u;
    constants.tiles_z = (uint32_t(nz) + 7u) / 8u;
    const uint64_t tile_count = uint64_t(constants.tiles_x) * constants.tiles_y *
        constants.tiles_z;
    if (tile_count >= uint64_t(std::numeric_limits<uint32_t>::max()) - 1u) {
        return fail("sparse tile lookup exceeds shader index width");
    }
    constants.tile_count = static_cast<uint32_t>(tile_count);
    const auto budget = params.mixed_working_set_budget_bytes;
    const uint64_t lookup_bytes = (2u * tile_count + 1u) * sizeof(uint32_t);
    if (budget != 0 && lookup_bytes + 8u * sizeof(double) > budget) {
        releaseSparsePressure(compute, storage);
        return fail("sparse tile lookup exceeds authored working budget");
    }
    // No parcel download or host occupancy bitmap. Only a uint count is read.
    if (!allocate(compute, storage.owned[0], tile_count * sizeof(uint32_t), "SparseTileMap",
                  budget != 0) ||
        !allocate(compute, storage.owned[1], (tile_count + 1u) * sizeof(uint32_t),
                  "SparseTileList", budget != 0)) {
        return fail("sparse tile lookup allocation failed");
    }
    for (std::size_t index = 2; index < storage.owned.size(); ++index) {
        if (!allocate(compute, storage.owned[index], sizeof(double), "SparsePressureBootstrap")) {
            return fail("sparse pressure bootstrap allocation failed");
        }
    }
    const auto mac = macVelocityBinding(buffers);
    const bool compact_weights = variational && mac.compact;
    if (compact_weights && !buffers.sparse_mac_transfer.solid_weights_ready) {
        return fail("sparse pressure: compact solid weight pages are not ready");
    }
    std::array<ComputeBufferHandle, 18> bindings{};
    const auto bind = [&]() {
        std::copy(storage.owned.begin(), storage.owned.end(), bindings.begin());
        bindings[10] = buffers.fluid_mask;
        bindings[11] = buffers.divergence;
        bindings[12] = variational ? mac.solid_weight[0] : buffers.fluid_mask;
        bindings[13] = variational ? mac.solid_weight[1] : buffers.fluid_mask;
        bindings[14] = variational ? mac.solid_weight[2] : buffers.fluid_mask;
        bindings[15] = buffers.pressure;
        bindings[16] = mac.map;
        bindings[17] = mac.list;
    };
    bind();
    const auto dispatch = [&](const char* name, uint64_t count) {
        ComputeDispatch command;
        command.kernel = name;
        const bool compact_init = compact_weights &&
            std::strcmp(name, "sim_sparse_pressure_init") == 0;
        const bool compact_spmv = compact_weights &&
            std::strcmp(name, "sim_sparse_pressure_spmv") == 0;
        if (compact_init) {
            command.kernel = "sim_sparse_pressure_init_mac_weights";
        } else if (compact_spmv) {
            command.kernel = "sim_sparse_pressure_spmv_mac_weights";
        }
        command.groups = groups(count);
        command.buffers = bindings.data();
        command.buffer_count = compact_init || compact_spmv ? 18 : 16;
        command.constants = &constants;
        command.constants_size = sizeof(constants);
        if (!compute.dispatch(command)) {
            return false;
        }
        ++storage.dispatches;
        return true;
    };
    if (!dispatch("sim_sparse_pressure_clear", tile_count) ||
        !dispatch("sim_sparse_pressure_mark", cells)) {
        return fail("sparse tile GPU classification dispatch failed");
    }
    uint32_t active_tiles = 0;
    compute.beginTransferBatch();
    bool ok = compute.downloadBuffer(storage.owned[1], &active_tiles, sizeof(active_tiles));
    ok = compute.endTransferBatch() && ok;
    storage.download_bytes += sizeof(active_tiles);
    if (!ok || active_tiles > tile_count) {
        return fail("sparse tile count readback failed");
    }
    const uint64_t compact_cells = uint64_t(active_tiles) * 512u;
    const uint64_t max_dispatch_lanes = uint64_t(std::numeric_limits<uint32_t>::max()) & ~255ull;
    if (compact_cells > max_dispatch_lanes) {
        return fail("sparse field exceeds 32-bit dispatch lane width");
    }
    constants.compact_count = static_cast<uint32_t>(compact_cells);
    constants.partial_count = static_cast<uint32_t>((compact_cells + 255u) / 256u);
    const uint64_t field_bytes = std::max<uint64_t>(compact_cells, 1u) * sizeof(float);
    const uint64_t partial_bytes = std::max(constants.partial_count, 1u) * sizeof(double);
    const uint64_t required = lookup_bytes + 6u * field_bytes + partial_bytes + 7u * sizeof(double);
    if (budget != 0 && required > budget) {
        releaseSparsePressure(compute, storage);
        return fail("sparse pressure pages exceed authored working budget");
    }
    uint64_t retained = lookup_bytes;
    for (std::size_t index = 2; index < 10; ++index) {
        const uint64_t wanted = index < 8 ? field_bytes
            : (index == 8 ? partial_bytes : 7u * sizeof(double));
        retained += std::max<uint64_t>(wanted, compute.getBufferSize(storage.owned[index]));
    }
    // An explicitly lowered budget must not retain an earlier large pool.
    const bool trim = budget != 0 && retained > budget;
    for (std::size_t index = 2; index < 8; ++index) {
        if (!allocate(compute, storage.owned[index],
                      field_bytes, "SparsePressureField", trim)) {
            return fail("sparse pressure tile field allocation failed");
        }
    }
    if (!allocate(compute, storage.owned[8],
                  partial_bytes, "SparseCgPartials", trim) ||
        !allocate(compute, storage.owned[9], 7u * sizeof(double), "SparseCgScalars", trim)) {
        return fail("sparse pressure reduction allocation failed");
    }
    storage.active_tiles = active_tiles;
    storage.allocated_tiles = compute.getBufferSize(storage.owned[2]) / (512u * sizeof(float));
    storage.resident_bytes = 0;
    for (const auto buffer : storage.owned) {
        storage.resident_bytes += compute.getBufferSize(buffer);
    }
    bind();
    // Clear the dense publication field, including tiles retired since the last
    // projection. CG arithmetic never reads it. Existing G2P/gradient users keep
    // their canonical dense address contract during this migration stage.
    if (!dispatch("sim_sparse_pressure_dense_clear", cells)) {
        return fail("dense pressure publication clear failed");
    }
    double host_scalars[7]{};
    int iterations = 0;
    int checks = 0;
    float check_ms = 0.0f;
    const auto scalarStep = [&](int operation) {
        constants.iterations = static_cast<int>(constants.partial_count);
        constants.parity = operation;
        const ComputeBufferHandle scalar_buffers[] = {storage.owned[8], storage.owned[9]};
        ComputeDispatch command;
        command.kernel = "sim_fluid_cg_scalar_step";
        command.buffers = scalar_buffers;
        command.buffer_count = 2;
        command.constants = &constants;
        command.constants_size = 52;
        if (!compute.dispatch(command)) {
            return false;
        }
        ++storage.dispatches;
        return true;
    };
    const int maximum = std::max(params.pressure_iterations, 1);
    const double tolerance = std::clamp(double(params.pressure_relative_residual), 1e-8, 1e-2);
    if (compact_cells != 0) {
        if (!dispatch("sim_sparse_pressure_init", compact_cells) ||
            !dispatch("sim_sparse_pressure_jacobi", compact_cells) || !scalarStep(0) ||
            !dispatch("sim_sparse_pressure_copy", compact_cells)) {
            return fail("sparse pressure initialization failed");
        }
        while (iterations < maximum) {
            const int batch = std::min(8, maximum - iterations);
            for (int tick = 0; tick < batch; ++tick) {
                if (!dispatch("sim_sparse_pressure_spmv", compact_cells) || !scalarStep(1) ||
                    !dispatch("sim_sparse_pressure_axpy", compact_cells) ||
                    !dispatch("sim_sparse_pressure_jacobi", compact_cells) || !scalarStep(2) ||
                    !dispatch("sim_sparse_pressure_zpby", compact_cells)) {
                    return fail("sparse pressure CG dispatch failed");
                }
            }
            iterations += batch;
            const auto before = std::chrono::steady_clock::now();
            compute.beginTransferBatch();
            ok = compute.downloadBuffer(storage.owned[9], host_scalars, sizeof(host_scalars));
            ok = compute.endTransferBatch() && ok;
            storage.download_bytes += sizeof(host_scalars);
            check_ms += std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - before).count();
            ++checks;
            if (!ok || !std::isfinite(host_scalars[0]) || !std::isfinite(host_scalars[1]) ||
                !std::isfinite(host_scalars[5])) {
                return fail("sparse pressure convergence readback failed or nonfinite");
            }
            if (host_scalars[6] != 0.0 || host_scalars[1] <= 0.0 ||
                host_scalars[5] <= tolerance * tolerance * host_scalars[1]) {
                break;
            }
        }
        if (!dispatch("sim_sparse_pressure_scatter", compact_cells)) {
            return fail("sparse pressure publication failed");
        }
    }
    storage.used = true;
    if (stats) {
        stats->pressure_sparse_used = true;
        stats->pressure_sparse_active_tiles = storage.active_tiles;
        stats->pressure_sparse_allocated_tiles = storage.allocated_tiles;
        stats->pressure_sparse_resident_bytes = storage.resident_bytes;
        stats->pressure_cg_iterations = iterations;
        stats->pressure_cg_max_iterations = maximum;
        stats->pressure_cg_dot_count = checks;
        stats->pressure_cg_dot_ms = check_ms;
        stats->pressure_cg_multigrid = false;
        stats->pressure_cg_final_relative_residual = host_scalars[1] > 0.0
            ? std::sqrt(std::max(0.0, host_scalars[5]) / host_scalars[1]) : 0.0;
    }
    return true;
}

} // namespace RayTrophiSim::Fluid
