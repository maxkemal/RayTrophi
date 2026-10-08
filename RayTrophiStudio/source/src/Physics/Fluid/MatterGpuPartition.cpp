#include "Fluid/MatterGpuPartition.h"
#include "PerfProfile.h"

#include <limits>

namespace RayTrophiSim::Fluid {

bool MatterGpuPartition::valid() const {
    return models.valid() && indices[0].valid() && indices[1].valid() && counters.valid();
}

void destroyMatterGpuPartition(SimulationComputeContext& compute, MatterGpuPartition& buffers) {
    compute.destroyBuffer(buffers.models);
    compute.destroyBuffer(buffers.indices[0]);
    compute.destroyBuffer(buffers.indices[1]);
    compute.destroyBuffer(buffers.counters);
    buffers = {};
}

bool ensureMatterGpuPartition(SimulationComputeContext& compute, MatterGpuPartition& buffers,
    std::size_t count, std::size_t budget_bytes, std::string& error) {
    error.clear();
    if (!compute.supportsDispatch() || compute.backendType() != ComputeBackendType::VulkanCompute) {
        error = "Matter GPU partition requires Vulkan compute";
        return false;
    }
    if (count == 0 || count > std::numeric_limits<uint32_t>::max() - 255u) {
        error = "Matter GPU partition count is outside uint32 dispatch range";
        return false;
    }
    const auto bytes = count * sizeof(uint32_t);
    const auto total = 3 * bytes + 4 * sizeof(uint32_t);
    if ((budget_bytes && total > budget_bytes) ||
        (compute.caps().max_storage_buffer_bytes &&
         bytes > compute.caps().max_storage_buffer_bytes)) {
        error = "Matter GPU partition exceeds resource budget or device buffer limit";
        return false;
    }
    if (buffers.valid() && buffers.capacity >= count) {
        return true;
    }
    MatterGpuPartition candidate;
    const auto usage = ComputeBufferUsage::Storage | ComputeBufferUsage::ReadWrite |
        ComputeBufferUsage::Upload | ComputeBufferUsage::Download;
    auto make = [&](const char* name, std::size_t size) {
        ComputeBufferDesc desc;
        desc.debug_name = name;
        desc.size_bytes = size;
        desc.usage = usage;
        return compute.createBuffer(desc);
    };
    candidate.models = make("matter_models", bytes);
    candidate.indices[0] = make("matter_fluid_indices", bytes);
    candidate.indices[1] = make("matter_granular_indices", bytes);
    candidate.counters = make("matter_model_counts", 4 * sizeof(uint32_t));
    candidate.capacity = count;
    if (!candidate.valid()) {
        destroyMatterGpuPartition(compute, candidate);
        error = "Matter GPU partition allocation failed";
        return false;
    }
    destroyMatterGpuPartition(compute, buffers);
    buffers = candidate;
    return true;
}

bool dispatchMatterGpuPartition(SimulationComputeContext& compute,
    MatterGpuPartition& buffers, const FluidParticles& particles,
    MatterConstitutiveModel legacy_model, std::string& error) {
    error.clear();
    if (!buffers.valid() || buffers.capacity < particles.size() || particles.empty() ||
        compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch()) {
        error = "Matter GPU partition buffers/backend are not ready";
        return false;
    }
    if (legacy_model != MatterConstitutiveModel::Fluid &&
        legacy_model != MatterConstitutiveModel::Granular) {
        error = "Matter GPU partition requires resolved legacy model";
        return false;
    }
    std::vector<uint32_t> models(particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i) {
        models[i] = i < particles.constitutive_model.size()
            ? particles.constitutive_model[i] : 0;
        if (models[i] > static_cast<uint32_t>(MatterConstitutiveModel::Elastic)) {
            error = "Matter GPU partition received unknown constitutive model";
            return false;
        }
    }
    const uint32_t zero[4] = {};
    // Only compact model metadata crosses the boundary. No particle download,
    // host partition, per-model SoA upload, or synchronous counter readback.
    // Timed per dispatch: the model upload is 4 B per parcel every frame, and
    // this is the number that tells us when it becomes a bottleneck.
    bool uploaded;
    {
        RTPERF_FRAME_SCOPE("sim.matter.gpu_partition_upload");
        uploaded = compute.uploadBuffer(buffers.models, models.data(), models.size() * sizeof(uint32_t)) &&
                   compute.uploadBuffer(buffers.counters, zero, sizeof(zero));
    }
    if (!uploaded) {
        error = "Matter GPU partition metadata upload failed";
        return false;
    }
    struct Constants {
        uint32_t count;
        uint32_t legacy;
    } constants{static_cast<uint32_t>(particles.size()), static_cast<uint32_t>(legacy_model)};
    static_assert(sizeof(Constants) == 8);
    const ComputeBufferHandle handles[] = {buffers.models, buffers.indices[0],
        buffers.indices[1], buffers.counters};
    ComputeDispatch command;
    command.kernel = "sim_matter_partition";
    command.groups.groups_x = (constants.count + 255u) / 256u;
    command.buffers = handles;
    command.buffer_count = 4;
    command.constants = &constants;
    command.constants_size = sizeof(constants);
    if (!compute.dispatch(command)) {
        error = "Matter GPU partition dispatch failed";
        return false;
    }
    return true;
}

} // namespace RayTrophiSim::Fluid
