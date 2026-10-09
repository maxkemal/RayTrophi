#include "Fluid/SparseGridGpu.h"
#include "Fluid/FluidGpuDispatch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace RayTrophiSim::Fluid::Sparse {
namespace {
bool allocate(SimulationComputeContext& compute, ComputeBufferHandle& handle, uint64_t bytes) {
    const auto limit = compute.caps().max_storage_buffer_bytes;
    if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max() ||
        (limit != 0 && bytes > limit)) {
        return false;
    }
    ComputeBufferDesc description;
    description.debug_name = "SparseGridTransaction";
    description.size_bytes = static_cast<std::size_t>(bytes);
    description.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::ReadWrite |
        ComputeBufferUsage::Upload | ComputeBufferUsage::Download;
    handle = compute.createBuffer(description);
    return handle.valid();
}

bool constantsFor(const Topology& topology, DeviceGridConstants& constants) {
    const auto& dimensions = topology.dimensions();
    constants.nx = dimensions[0];
    constants.ny = dimensions[1];
    constants.nz = dimensions[2];
    constants.tiles_x = (uint32_t(dimensions[0]) + 7u) / 8u;
    constants.tiles_y = (uint32_t(dimensions[1]) + 7u) / 8u;
    constants.tiles_z = (uint32_t(dimensions[2]) + 7u) / 8u;
    const uint64_t xy = uint64_t(constants.tiles_x) * constants.tiles_y;
    if (xy > uint64_t(std::numeric_limits<uint32_t>::max()) / constants.tiles_z ||
        topology.tiles().size() > std::numeric_limits<uint32_t>::max() / 576u) {
        return false;
    }
    constants.new_active = static_cast<uint32_t>(topology.tiles().size());
    return true;
}

bool validSource(SimulationComputeContext& compute, const DeviceGrid& source) {
    DeviceGridConstants constants;
    if (compute.backendType() != ComputeBackendType::VulkanCompute || !source.topology ||
        !constantsFor(*source.topology, constants) ||
        source.map.backend != compute.backendType() || source.keys.backend != compute.backendType()) {
        return false;
    }
    const uint64_t tiles = uint64_t(constants.tiles_x) * constants.tiles_y * constants.tiles_z;
    if (compute.getBufferSize(source.map) < tiles * sizeof(uint32_t) ||
        compute.getBufferSize(source.keys) <
            std::max<uint64_t>(source.topology->tiles().size(), 1u) * sizeof(uint32_t)) {
        return false;
    }
    for (std::size_t field = 0; field < channelCount; ++field) {
        const auto& description = source.descriptions[field];
        if (!std::isfinite(description.background) ||
            (description.location != Location::Cell && faceAxis(description.location) < 0)) {
            return false;
        }
        if (description.enabled) {
            const uint64_t page = faceAxis(description.location) >= 0 ? 576u : 512u;
            if (source.fields[field].backend != compute.backendType() ||
                compute.getBufferSize(source.fields[field]) <
                    std::max<uint64_t>(page * source.topology->tiles().size(), 1u) * sizeof(float)) {
                return false;
            }
        }
    }
    return true;
}

uint64_t requiredBytes(const Topology& topology, const ChannelDescriptions& descriptions,
                       const DeviceGridConstants& constants) {
    const uint64_t tiles = uint64_t(constants.tiles_x) * constants.tiles_y * constants.tiles_z;
    uint64_t bytes = tiles * sizeof(uint32_t) +
        std::max<uint64_t>(topology.tiles().size(), 1u) * sizeof(uint32_t);
    for (const auto& description : descriptions) {
        if (description.enabled) {
            const uint64_t page = faceAxis(description.location) >= 0 ? 576u : 512u;
            bytes += std::max<uint64_t>(page * topology.tiles().size(), 1u) * sizeof(float);
        }
    }
    return bytes;
}

bool makeCandidate(SimulationComputeContext& compute, std::shared_ptr<const Topology> topology,
                    const ChannelDescriptions& descriptions, DeviceGrid& candidate,
                    DeviceGridConstants& constants, uint64_t retained, uint64_t budget,
                    std::string& error) {
    if (compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch() || !topology || !constantsFor(*topology, constants)) {
        error = "unsupported sparse grid backend or shader address width";
        return false;
    }
    const uint64_t required = requiredBytes(*topology, descriptions, constants);
    if (retained > std::numeric_limits<uint64_t>::max() - sizeof(uint32_t) ||
        required > std::numeric_limits<uint64_t>::max() - retained - sizeof(uint32_t) ||
        (budget != 0 && (retained > budget || required + sizeof(uint32_t) > budget - retained))) {
        error = "sparse topology transaction exceeds authored GPU working budget";
        return false;
    }
    candidate.topology = std::move(topology);
    candidate.descriptions = descriptions;
    const auto tiles = uint64_t(constants.tiles_x) * constants.tiles_y * constants.tiles_z;
    if (!allocate(compute, candidate.map, tiles * sizeof(uint32_t)) ||
        !allocate(compute, candidate.keys,
                  std::max<uint64_t>(constants.new_active, 1u) * sizeof(uint32_t))) {
        error = "sparse topology map/key allocation failed";
        return false;
    }
    for (std::size_t field = 0; field < channelCount; ++field) {
        const auto& description = descriptions[field];
        if (!description.enabled) {
            continue;
        }
        const uint64_t page = faceAxis(description.location) >= 0 ? 576u : 512u;
        if (!allocate(compute, candidate.fields[field],
                      std::max<uint64_t>(page * constants.new_active, 1u) * sizeof(float))) {
            error = "sparse channel page allocation failed";
            return false;
        }
    }
    candidate.resident_bytes = compute.getBufferSize(candidate.map) +
        compute.getBufferSize(candidate.keys);
    for (const auto handle : candidate.fields) {
        if (handle.valid()) {
            candidate.resident_bytes += compute.getBufferSize(handle);
        }
    }
    if (budget != 0 && candidate.resident_bytes + retained + sizeof(uint32_t) > budget) {
        error = "actual sparse GPU capacity exceeds authored working budget";
        return false;
    }
    return true;
}

bool dispatch(SimulationComputeContext& compute, const DeviceGrid& old,
               DeviceGrid& next, ComputeBufferHandle validation,
               DeviceGridConstants& constants, const char* kernel, uint32_t lanes,
               std::size_t field) {
    if (lanes == 0) {
        return true;
    }
    // Topology stages never access fields; keys serve as valid dummy bindings
    // until the first real page pass. Every declared descriptor is bound.
    const ComputeBufferHandle old_field = old.fields[field].valid()
        ? old.fields[field] : next.keys;
    const ComputeBufferHandle new_field = next.fields[field].valid()
        ? next.fields[field] : next.keys;
    const ComputeBufferHandle bindings[] = {
        old.map.valid() ? old.map : next.map, next.map, old_field, new_field,
        old.keys.valid() ? old.keys : next.keys, next.keys, validation};
    ComputeDispatch command;
    command.kernel = kernel;
    command.buffers = bindings;
    command.buffer_count = 7;
    command.constants = &constants;
    command.constants_size = sizeof(constants);
    command.groups = FluidGpuDispatch::groups256(lanes);
    return compute.dispatch(command);
}

bool seedMap(SimulationComputeContext& compute, DeviceGrid& candidate,
              ComputeBufferHandle validation, DeviceGridConstants& constants) {
    std::vector<uint32_t> keys;
    keys.reserve(std::max<uint32_t>(constants.new_active, 1u));
    for (const auto& tile : candidate.topology->tiles()) {
        keys.push_back(uint32_t(tile.x) + constants.tiles_x *
            (uint32_t(tile.y) + constants.tiles_y * uint32_t(tile.z)));
    }
    if (keys.empty()) {
        keys.push_back(0);
    }
    if (!compute.uploadBuffer(candidate.keys, keys.data(), keys.size() * sizeof(uint32_t))) {
        return false;
    }
    const uint32_t zero = 0;
    if (!compute.uploadBuffer(validation, &zero, sizeof(zero))) {
        return false;
    }
    const DeviceGrid empty;
    const uint32_t tiles = constants.tiles_x * constants.tiles_y * constants.tiles_z;
    return dispatch(compute, empty, candidate, validation, constants,
                    "sim_sparse_grid_map_clear", tiles, 0) &&
        dispatch(compute, empty, candidate, validation, constants,
                 "sim_sparse_grid_map_seed", constants.new_active, 0);
}
} // namespace

void releaseGrid(SimulationComputeContext& compute, DeviceGrid& grid) {
    if (grid.map.valid()) {
        compute.destroyBuffer(grid.map);
    }
    if (grid.keys.valid()) {
        compute.destroyBuffer(grid.keys);
    }
    for (const auto handle : grid.fields) {
        if (handle.valid()) {
            compute.destroyBuffer(handle);
        }
    }
    grid = {};
}

bool uploadGrid(SimulationComputeContext& compute, const GridStorage& host,
                std::shared_ptr<const Topology> topology, DeviceGrid& destination,
                uint64_t working_budget_bytes, std::string& error) {
    error.clear();
    DeviceGrid candidate;
    DeviceGridConstants constants;
    ComputeBufferHandle validation;
    const auto fail = [&](const char* reason) {
        compute.synchronize();
        releaseGrid(compute, candidate);
        if (validation.valid()) {
            compute.destroyBuffer(validation);
        }
        if (error.empty()) {
            error = reason;
        }
        return false;
    };
    ChannelDescriptions descriptions{};
    for (std::size_t field = 0; field < channelCount; ++field) {
        descriptions[field] = host.description(static_cast<Channel>(field));
    }
    if (!topology || host.snapshot().topology().dimensions() != topology->dimensions() ||
        host.snapshot().topology().tiles() != topology->tiles()) {
        return fail("sparse host/device topology mismatch");
    }
    if (!makeCandidate(compute, std::move(topology), descriptions, candidate, constants,
                        destination.resident_bytes, working_budget_bytes, error) ||
        !allocate(compute, validation, sizeof(uint32_t)) ||
        !seedMap(compute, candidate, validation, constants)) {
        return fail("sparse GPU bootstrap failed");
    }
    try {
        for (std::size_t field = 0; field < channelCount; ++field) {
            if (!descriptions[field].enabled) {
                continue;
            }
            const auto page = faceAxis(descriptions[field].location) >= 0 ? 576u : 512u;
            std::vector<float> packed(
                std::max<std::size_t>(candidate.topology->tiles().size() * page, 1u),
                descriptions[field].background);
            std::size_t slot = 0;
            for (const auto& tile : candidate.topology->tiles()) {
                host.exportPage(static_cast<Channel>(field), tile,
                                packed.data() + slot * page, page);
                ++slot;
            }
            if (!compute.uploadBuffer(candidate.fields[field], packed.data(),
                                       packed.size() * sizeof(float))) {
                return fail("sparse channel bootstrap upload failed");
            }
        }
    } catch (const std::exception& exception) {
        error = exception.what();
        return fail("sparse host page export failed");
    }
    compute.synchronize();
    compute.destroyBuffer(validation);
    candidate.generation = host.statistics().generation;
    releaseGrid(compute, destination);
    destination = std::move(candidate);
    return true;
}

bool rebindGrid(SimulationComputeContext& compute, DeviceGrid& grid,
                std::shared_ptr<const Topology> next, uint64_t working_budget_bytes,
                std::string& error) {
    error.clear();
    DeviceGrid candidate;
    DeviceGridConstants constants;
    ComputeBufferHandle validation;
    const auto fail = [&](const char* reason) {
        compute.synchronize();
        releaseGrid(compute, candidate);
        if (validation.valid()) {
            compute.destroyBuffer(validation);
        }
        if (error.empty()) {
            error = reason;
        }
        return false;
    };
    if (!validSource(compute, grid) || !next || grid.topology->dimensions() != next->dimensions() ||
        grid.generation == std::numeric_limits<uint64_t>::max()) {
        return fail("invalid sparse device topology transaction");
    }
    if (!makeCandidate(compute, std::move(next), grid.descriptions, candidate, constants,
                        grid.resident_bytes, working_budget_bytes, error) ||
        !allocate(compute, validation, sizeof(uint32_t)) ||
        !seedMap(compute, candidate, validation, constants)) {
        return fail("sparse device transaction allocation/map failed");
    }
    constants.old_active = static_cast<uint32_t>(grid.topology->tiles().size());
    for (std::size_t field = 0; field < channelCount; ++field) {
        if (!grid.descriptions[field].enabled) {
            continue;
        }
        constants.face_axis = faceAxis(grid.descriptions[field].location);
        constants.page_values = constants.face_axis >= 0 ? 576u : 512u;
        constants.background = grid.descriptions[field].background;
        if (!dispatch(compute, grid, candidate, validation, constants,
                      "sim_sparse_grid_retire", constants.old_active * constants.page_values, field) ||
            !dispatch(compute, grid, candidate, validation, constants,
                      "sim_sparse_grid_remap", constants.new_active * constants.page_values, field)) {
            return fail("sparse channel remap/retirement dispatch failed");
        }
    }
    uint32_t invalid = 0;
    compute.beginTransferBatch();
    bool ok = compute.downloadBuffer(validation, &invalid, sizeof(invalid));
    ok = compute.endTransferBatch() && ok;
    if (!ok) {
        return fail("sparse topology validation readback failed");
    }
    if ((invalid & 2u) != 0) {
        return fail("sparse channel contains nonfinite device values");
    }
    if (invalid != 0) {
        return fail("sparse retirement would discard a live channel");
    }
    compute.destroyBuffer(validation);
    candidate.generation = grid.generation + 1u;
    releaseGrid(compute, grid);
    grid = std::move(candidate);
    return true;
}

bool downloadGrid(SimulationComputeContext& compute, const DeviceGrid& device,
                  GridStorage& host, std::string& error) {
    error.clear();
    if (!validSource(compute, device) ||
        device.topology->dimensions() != host.snapshot().topology().dimensions()) {
        error = "sparse publication layout mismatch";
        return false;
    }
    try {
        // Every channel is staged before the host canonical state is published.
        // Explicit snapshot/export publication is the only field readback here.
        GridStorage candidate = host;
        for (std::size_t field = 0; field < channelCount; ++field) {
            const auto channel = static_cast<Channel>(field);
            const auto host_description = candidate.description(channel);
            const auto& device_description = device.descriptions[field];
            if (host_description.enabled != device_description.enabled ||
                host_description.location != device_description.location ||
                host_description.background != device_description.background) {
                error = "sparse publication channel contract mismatch";
                return false;
            }
        }
        // Clear a candidate only: a failed transfer/import leaves host intact.
        for (std::size_t field = 0; field < channelCount; ++field) {
            if (candidate.description(static_cast<Channel>(field)).enabled) {
                candidate.clear(static_cast<Channel>(field));
            }
        }
        candidate.rebind(device.topology);
        for (std::size_t field = 0; field < channelCount; ++field) {
            const auto& description = device.descriptions[field];
            if (!description.enabled) {
                continue;
            }
            const auto host_description = candidate.description(static_cast<Channel>(field));
            if (!host_description.enabled || host_description.location != description.location ||
                host_description.background != description.background) {
                error = "sparse publication channel contract mismatch";
                return false;
            }
            const std::size_t page = faceAxis(description.location) >= 0 ? 576u : 512u;
            std::vector<float> packed(std::max<std::size_t>(
                device.topology->tiles().size() * page, 1u));
            compute.beginTransferBatch();
            bool ok = compute.downloadBuffer(device.fields[field], packed.data(),
                                              packed.size() * sizeof(float));
            ok = compute.endTransferBatch() && ok;
            if (!ok) {
                error = "sparse publication channel readback failed";
                return false;
            }
            std::size_t slot = 0;
            for (const auto& tile : device.topology->tiles()) {
                candidate.importPage(static_cast<Channel>(field), tile,
                                     packed.data() + slot * page, page);
                ++slot;
            }
        }
        host = std::move(candidate);
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
    return true;
}
} // namespace RayTrophiSim::Fluid::Sparse
