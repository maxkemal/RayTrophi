#include "Fluid/FluidParticleLabelsGpu.h"

#include "ParticleSimulation.h"
#include "SimulationCompute.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace RayTrophiSim::Fluid {
namespace {

struct LabelGpuConstants {
    int bin_nx = 0, bin_ny = 0, bin_nz = 0, particle_count = 0;
    int bin_min_x = 0, bin_min_y = 0, bin_min_z = 0, max_per_bin = 0;
    int bin_count = 0;
    float radius = 0.0f;
    uint32_t label_mask = 0, label_shift = 0;
    uint32_t frozen_mask = 0, body_label = 0, spray_label = 0, frozen_label = 0;
    float mist_mass_fraction = 0.0f;
    uint32_t mist_label = 0;
    uint32_t padding[2] = {};
};
static_assert(sizeof(LabelGpuConstants) == 80);

constexpr uint32_t kLinkedLabelAbi = 1u;
constexpr uint32_t kThreads = 256;

void setLabelGroups(ComputeDispatch& command, std::size_t threads) {
    const uint32_t groups = static_cast<uint32_t>((threads + kThreads - 1u) / kThreads);
    command.groups.groups_x = std::min(groups, 65535u);
    command.groups.groups_y = (groups + command.groups.groups_x - 1u) /
        command.groups.groups_x;
    command.groups.groups_z = 1u;
}

bool ensureBuffer(SimulationComputeContext& compute, ComputeBufferHandle& handle,
                  std::size_t& capacity, std::size_t required,
                  const char* debug_name) {
    const std::size_t keep_limit = 2u * std::max<std::size_t>(required, 4096u);
    if (handle.valid() && handle.backend == compute.backendType() &&
        capacity >= required && capacity <= keep_limit) {
        return true;
    }
    if (handle.valid()) {
        compute.destroyBuffer(handle);
        handle = {};
    }
    std::size_t grown = std::max<std::size_t>(required, 4096u);
    grown += grown / 2u;
    ComputeBufferDesc desc;
    desc.debug_name = debug_name;
    desc.size_bytes = grown;
    desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                 ComputeBufferUsage::Download | ComputeBufferUsage::ReadWrite;
    handle = compute.createBuffer(desc);
    capacity = handle.valid() ? grown : 0u;
    return handle.valid();
}

} // namespace

bool updateParticleLabelsGpu(SimulationGridDomainState& state,
                             SimulationComputeContext* compute,
                             SimulationGridDomainComputeBuffers& buffers,
                             ParticleLabelStepStats& stats) {
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    auto& particles = state.particles;
    const std::size_t count = particles.size();
    if (!compute || compute->backendType() != ComputeBackendType::VulkanCompute ||
        !compute->supportsDispatch() || count == 0 || state.grid.nx <= 0 ||
        state.grid.ny <= 0 || state.grid.nz <= 0 ||
        !(state.voxel_size > 0.0f) || !std::isfinite(state.voxel_size) ||
        !buffers.fluid_positions.valid()) {
        return false;
    }
    particles.flags.resize(count, 0u);
    const float radius = state.voxel_size * kParticleLabelRadiusVoxels;
    const auto lower = [&](float value) { return static_cast<int>(std::floor(value / radius)) - 1; };
    const auto upper = [&](float value) { return static_cast<int>(std::floor(value / radius)) + 1; };
    const Vec3 extent(state.grid.nx * state.voxel_size,
                      state.grid.ny * state.voxel_size,
                      state.grid.nz * state.voxel_size);
    LabelGpuConstants c;
    c.bin_min_x = lower(state.grid.origin.x);
    c.bin_min_y = lower(state.grid.origin.y);
    c.bin_min_z = lower(state.grid.origin.z);
    c.bin_nx = upper(state.grid.origin.x + extent.x) - c.bin_min_x + 1;
    c.bin_ny = upper(state.grid.origin.y + extent.y) - c.bin_min_y + 1;
    c.bin_nz = upper(state.grid.origin.z + extent.z) - c.bin_min_z + 1;
    const int64_t bin_count64 = int64_t(c.bin_nx) * c.bin_ny * c.bin_nz;
    if (bin_count64 <= 0 || bin_count64 > std::numeric_limits<int>::max() ||
        count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    c.bin_count = static_cast<int>(bin_count64);
    c.particle_count = static_cast<int>(count);
    c.max_per_bin = 0; // Reserved in the 80-byte ABI; linked bins have no fixed slot cap.
    c.padding[0] = kLinkedLabelAbi;
    c.radius = radius;
    c.label_mask = kParticleLabelMask;
    c.label_shift = kParticleLabelShift;
    c.frozen_mask = kParticleFlagFrozen;
    c.body_label = static_cast<uint32_t>(ParticleLabel::Body);
    c.spray_label = static_cast<uint32_t>(ParticleLabel::Spray);
    c.frozen_label = static_cast<uint32_t>(ParticleLabel::Frozen);
    c.mist_mass_fraction = kParticleMistMassFraction;
    c.mist_label = static_cast<uint32_t>(ParticleLabel::Mist);

    const std::size_t count_bytes = static_cast<std::size_t>(c.bin_count) * sizeof(uint32_t);
    const std::size_t item_bytes = count * sizeof(uint32_t);
    const std::size_t flag_bytes = count * sizeof(uint32_t);
    constexpr std::size_t stat_bytes = 5u * sizeof(uint32_t);
    const auto max_buffer = compute->caps().max_storage_buffer_bytes;
    if ((max_buffer && (item_bytes > max_buffer || count_bytes > max_buffer ||
                        flag_bytes > max_buffer))) {
        return false;
    }
    if (!ensureBuffer(*compute, buffers.label_bin_counts, buffers.label_bin_count_capacity,
                      count_bytes, "FluidLabelBinHeads") ||
        !ensureBuffer(*compute, buffers.label_bin_items, buffers.label_bin_item_capacity,
                      item_bytes, "FluidLabelNext") ||
        !ensureBuffer(*compute, buffers.label_flags, buffers.label_flag_capacity,
                      flag_bytes, "FluidLabelFlags") ||
        !ensureBuffer(*compute, buffers.label_stats, buffers.label_stat_capacity,
                      stat_bytes, "FluidLabelStats")) {
        return false;
    }
    uint32_t gpu_stats[5] = {};
    compute->beginTransferBatch();
    bool uploaded = compute->uploadBuffer(buffers.label_flags, particles.flags.data(), flag_bytes) &&
        compute->uploadBuffer(buffers.label_stats, gpu_stats, stat_bytes);
    uploaded = compute->endTransferBatch() && uploaded;
    if (!uploaded) {
        return false;
    }

    ComputeBufferHandle clear_buffers[] = {buffers.label_bin_counts, buffers.label_stats};
    ComputeDispatch cmd;
    cmd.kernel = "sim_fluid_label_link_clear";
    cmd.buffers = clear_buffers;
    cmd.buffer_count = 2;
    cmd.constants = &c;
    cmd.constants_size = sizeof(c);
    setLabelGroups(cmd, std::max<std::size_t>(static_cast<std::size_t>(c.bin_count), 4u));
    if (!compute->dispatch(cmd)) return false;

    ComputeBufferHandle scatter_buffers[] = {
        buffers.fluid_positions, buffers.label_bin_counts,
        buffers.label_bin_items, buffers.label_stats,
        buffers.fluid_mass_fraction
    };
    cmd.kernel = "sim_fluid_label_link_scatter";
    cmd.buffers = scatter_buffers;
    cmd.buffer_count = 5;
    setLabelGroups(cmd, count);
    if (!compute->dispatch(cmd)) return false;

    ComputeBufferHandle classify_buffers[] = {
        buffers.fluid_positions, buffers.label_flags, buffers.label_bin_counts,
        buffers.label_bin_items, buffers.label_stats,
        buffers.fluid_mass_fraction
    };
    cmd.kernel = "sim_fluid_label_link_classify";
    cmd.buffers = classify_buffers;
    cmd.buffer_count = 6;
    if (!compute->dispatch(cmd)) return false;

    static thread_local std::vector<uint32_t> result;
    result.resize(count);
    compute->beginTransferBatch();
    bool ok = compute->downloadBuffer(buffers.label_flags, result.data(), flag_bytes) &&
              compute->downloadBuffer(buffers.label_stats, gpu_stats, stat_bytes);
    ok = compute->endTransferBatch() && ok;
    // All three new entry points must have run. Missing/stale modules fall back
    // before publishing flags; legacy fixed-slot shaders never bind this layout.
    if (!ok || gpu_stats[0] != 0u || gpu_stats[4] != 7u) {
        return false;
    }
    particles.flags.assign(result.begin(), result.end());
    stats = {};
    stats.on_gpu = true;
    stats.particles = count;
    stats.occupied_bins = gpu_stats[1];
    stats.changed = gpu_stats[2];
    stats.center_resolved = gpu_stats[3];
    stats.classify_milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    stats.milliseconds = stats.classify_milliseconds;
    return true;
}

} // namespace RayTrophiSim::Fluid
