#include "Fluid/GasInventoryGpu.h"
#include "Fluid/MatterPhaseGrid.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace RayTrophiSim::Fluid {
namespace {
constexpr uint32_t kThreads = 256;
constexpr auto kUsage = ComputeBufferUsage::Storage;

struct Deposit {
    uint32_t cell;
    float mass;
    float energy;
    uint32_t padding = 0;
};
static_assert(sizeof(Deposit) == 16);

struct SummaryConstants {
    uint32_t cells;
    uint32_t blocks;
    uint32_t phase;
    int nx;
    int ny;
    float voxel;
    float origin_x;
    float origin_y;
    float origin_z;
};
static_assert(sizeof(SummaryConstants) == 36);

// Device slots that hold state (the rest is advection scratch), with their
// host image. Every snapshot, upload and backend switch walks this one list.
constexpr std::size_t kPersistentSlots[] = {0, 3, 6, 9, 10, 11};
std::array<std::vector<float>*, 6> persistentImages(SimulationGridDomainState& state) {
    return {&state.gas_phase_mass_kg, &state.gas_phase_energy_j,
            &state.gas_phase_condensate_kg, &state.gas_phase_rain_kg,
            &state.gas_phase_rain_energy_j, &state.gas_phase_foreign_kg};
}

bool create(SimulationComputeContext& compute, ComputeBufferHandle& handle,
            std::size_t bytes, const char* name) {
    const auto usage = kUsage | ComputeBufferUsage::Upload | ComputeBufferUsage::Download |
                       ComputeBufferUsage::ReadWrite;
    handle = compute.createBuffer({name, bytes, usage});
    return handle.valid();
}

bool materialize(SimulationGridDomainState& state) {
    auto owner = gasInventoryDevice(state);
    if (!owner) {
        return state.gas_inventory_revision == 0;
    }
    if (!owner->pending.empty() || !owner->pending_condensate.empty() ||
        !owner->pending_foreign.empty() || !owner->pending_momentum.empty()) {
        return false; // A snapshot cannot silently discard uncommitted phase deposits.
    }
    const auto images = persistentImages(state);
    if (!owner->populated) {
        for (auto* image : images) {
            image->clear();
        }
        state.gas_inventory_device.reset();
        state.gas_inventory_revision = 0;
        return true;
    }
    std::array<std::vector<float>, 6> values;
    auto& compute = *owner->compute;
    const std::size_t bytes = owner->cells * sizeof(float);
    compute.beginTransferBatch();
    bool ok = true;
    for (std::size_t k = 0; k < values.size(); ++k) {
        values[k].resize(owner->cells);
        ok = compute.downloadBuffer(owner->fields[kPersistentSlots[k]], values[k].data(),
                                    bytes) && ok;
    }
    ok = compute.endTransferBatch() && ok;
    if (!ok) {
        return false;
    }
    owner->snapshot_download_bytes += values.size() * bytes;
    for (std::size_t k = 0; k < values.size(); ++k) {
        images[k]->swap(values[k]);
    }
    state.gas_inventory_device.reset();
    state.gas_inventory_revision = 0;
    return true;
}
} // namespace

std::shared_ptr<GasInventoryDevice> gasInventoryDevice(const SimulationGridDomainState& state) {
    auto owner = state.gas_inventory_device.lock();
    if (owner && owner->revision == state.gas_inventory_revision) {
        return owner;
    }
    return {};
}

void markGasInventoryDeviceChanged(SimulationGridDomainState& state) {
    auto owner = gasInventoryDevice(state);
    if (!owner) {
        throw std::runtime_error("gas inventory device ownership was lost");
    }
    ++owner->revision;
    owner->summary_valid = false;
    state.gas_inventory_revision = owner->revision;
}

void releaseGasInventoryGpu(SimulationComputeContext& compute, GasInventoryGpuBuffers& buffers) {
    if (!buffers.owner) {
        return;
    }
    auto& owner = *buffers.owner;
    auto& owner_compute = owner.compute ? *owner.compute : compute;
    auto destroy = [&](ComputeBufferHandle handle) {
        if (handle.valid()) {
            owner_compute.destroyBuffer(handle);
        }
    };
    for (const auto handle : owner.fields) {
        destroy(handle);
    }
    for (const auto handle : owner.surface) {
        destroy(handle);
    }
    destroy(owner.partials);
    destroy(owner.totals);
    destroy(owner.zero);
    destroy(owner.deposits);
    destroy(owner.condensate_deposits);
    destroy(owner.foreign_deposits);
    destroy(owner.phase_momentum);
    destroy(owner.summary_partials);
    destroy(owner.summary_totals);
    destroy(owner.saturation_records);
    destroy(owner.saturation_counter);
    destroy(owner.rain_landing);
    buffers.owner.reset();
}

bool beginGasInventoryDevice(SimulationGridDomainState& state,
                             SimulationComputeContext& compute,
                             SimulationGridDomainComputeBuffers& buffers) {
    const auto& grid = gasGrid(state);
    const std::size_t cells = grid.getCellCount();
    if (compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch() || !state.gas_inventory_error.empty() || cells == 0 ||
        cells > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    if (auto current = gasInventoryDevice(state)) {
        if (current == buffers.gas_inventory.owner && current->compute == &compute &&
            current->cells == cells) {
            if (current->nx != grid.nx || current->ny != grid.ny || current->nz != grid.nz ||
                current->voxel != grid.voxel_size) {
                return false; // A physical grid reshape requires an authored reset.
            }
            current->grid_upload_bytes = 0;
            current->grid_download_bytes = 0;
            return true;
        }
        // Moving to a different device/layout is an explicit ownership boundary.
        if (!materialize(state)) {
            return false;
        }
    } else if (state.gas_inventory_revision != 0) {
        return false;
    }
    releaseGasInventoryGpu(compute, buffers.gas_inventory);
    auto owner = std::make_shared<GasInventoryDevice>();
    owner->compute = &compute;
    owner->cells = cells;
    owner->nx = grid.nx;
    owner->ny = grid.ny;
    owner->nz = grid.nz;
    owner->voxel = grid.voxel_size;
    owner->revision = 1;
    buffers.gas_inventory.owner = owner;
    state.gas_inventory_device = owner;
    state.gas_inventory_revision = owner->revision;
    // A restored CPU/cache image uploads once at this boundary. Steady physics
    // never reads or uploads these host vectors again.
    const auto images = persistentImages(state);
    bool populated = false;
    for (const auto* field : images) {
        if (!field->empty() && field->size() != cells) {
            return false;
        }
        populated = populated || std::any_of(field->begin(), field->end(),
            [](float value) { return std::isfinite(value) && value > 0.0f; });
    }
    if (populated && !ensureGasInventoryStorage(state)) {
        return false;
    }
    bool ok = true;
    compute.beginTransferBatch();
    for (std::size_t field = 0; field < images.size(); ++field) {
        const auto& values = *images[field];
        if (values.empty() || !populated) {
            continue;
        }
        if (values.size() != cells) {
            ok = false;
            break;
        }
        std::vector<float> clean = values;
        for (float& value : clean) {
            if (!std::isfinite(value) || value < 0.0f) {
                value = 0.0f;
            }
        }
        ok = compute.uploadBuffer(owner->fields[kPersistentSlots[field]], clean.data(),
                                  cells * sizeof(float)) && ok;
        owner->grid_upload_bytes += cells * sizeof(float);
    }
    ok = compute.endTransferBatch() && ok;
    if (!ok) {
        return false;
    }
    owner->populated = populated;
    const auto positive = [](float value) { return std::isfinite(value) && value > 0.0f; };
    state.gas_liquid_water_present =
        std::any_of(state.gas_phase_condensate_kg.begin(), state.gas_phase_condensate_kg.end(),
                    positive) ||
        std::any_of(state.gas_phase_rain_kg.begin(), state.gas_phase_rain_kg.end(), positive);
    // Prevent accidental old CPU readers from observing a plausible stale image.
    for (auto* image : images) {
        std::vector<float>().swap(*image);
    }
    return true;
}

bool ensureGasInventoryStorage(SimulationGridDomainState& state) {
    auto owner = gasInventoryDevice(state);
    if (!owner) {
        return false;
    }
    if (owner->storage_initialized) {
        return true;
    }
    auto& compute = *owner->compute;
    bool ok = true;
    for (auto& handle : owner->fields) {
        if (!handle.valid()) {
            ok = create(compute, handle, owner->cells * sizeof(float), "GasInventoryField") && ok;
        }
    }
    const uint32_t blocks = static_cast<uint32_t>((owner->cells + kThreads - 1) / kThreads);
    if (!owner->partials.valid()) {
        ok = create(compute, owner->partials, blocks * 4 * sizeof(double),
                    "GasInventoryPartials") && ok;
    }
    if (!owner->totals.valid()) {
        ok = create(compute, owner->totals, 4 * sizeof(double), "GasInventoryTotals") && ok;
    }
    if (!ok) {
        return false;
    }
    const uint32_t count = static_cast<uint32_t>(owner->cells);
    ComputeBufferHandle fields[2] = {owner->fields[0], owner->fields[3]};
    ComputeDispatch command;
    command.kernel = "sim_inventory_clear";
    command.buffers = fields;
    command.buffer_count = 2;
    command.constants = &count;
    command.constants_size = sizeof(count);
    command.groups = gasInventoryDispatch(owner->cells);
    bool cleared = compute.dispatch(command);
    // The clear kernel takes two fields: condensate + rain, then rain energy +
    // non-condensable gas.
    ComputeBufferHandle water[2] = {owner->fields[6], owner->fields[9]};
    command.buffers = water;
    cleared = cleared && compute.dispatch(command);
    ComputeBufferHandle rain_energy[2] = {owner->fields[10], owner->fields[11]};
    command.buffers = rain_energy;
    cleared = cleared && compute.dispatch(command);
    owner->storage_initialized = cleared;
    return owner->storage_initialized;
}

ComputeBufferHandle gasInventoryCondensateBuffer(SimulationGridDomainState& state) {
    auto owner = gasInventoryDevice(state);
    if (!owner) {
        return {};
    }
    if (owner->storage_initialized) {
        return owner->fields[6];
    }
    return gasInventoryReadBuffers(state)[0];   // the zero buffer
}

std::array<ComputeBufferHandle, 2> gasInventoryRainBuffers(SimulationGridDomainState& state) {
    auto owner = gasInventoryDevice(state);
    if (!owner) {
        return {};
    }
    if (owner->storage_initialized) {
        return {owner->fields[9], owner->fields[10]};
    }
    return gasInventoryReadBuffers(state);   // the zero buffer
}

ComputeBufferHandle gasInventoryRainLanding(SimulationGridDomainState& state) {
    auto owner = gasInventoryDevice(state);
    if (!owner) {
        return {};
    }
    if (owner->rain_landing.valid()) {
        return owner->rain_landing;
    }
    auto& compute = *owner->compute;
    if (!create(compute, owner->rain_landing, owner->cells * sizeof(uint32_t),
                "GasRainLanding")) {
        return {};
    }
    // Zero bits: no cell carries a live stamp (stamps start at 1).
    ComputeBufferHandle fields[2] = {owner->rain_landing, owner->rain_landing};
    const uint32_t count = static_cast<uint32_t>(owner->cells);
    ComputeDispatch command;
    command.kernel = "sim_inventory_clear";
    command.buffers = fields;
    command.buffer_count = 2;
    command.constants = &count;
    command.constants_size = sizeof(count);
    command.groups = gasInventoryDispatch(owner->cells);
    if (!compute.dispatch(command)) {
        compute.destroyBuffer(owner->rain_landing);
        owner->rain_landing = {};
    }
    return owner->rain_landing;
}

std::array<ComputeBufferHandle, 3> gasInventoryScratchBuffers(SimulationGridDomainState& state) {
    auto owner = gasInventoryDevice(state);
    if (!owner || !owner->storage_initialized) {
        return {};
    }
    return {owner->fields[1], owner->fields[2], owner->fields[4]};
}

std::array<ComputeBufferHandle, 2> gasInventoryReadBuffers(SimulationGridDomainState& state) {
    auto owner = gasInventoryDevice(state);
    if (!owner) {
        return {};
    }
    if (owner->storage_initialized) {
        return {owner->fields[0], owner->fields[3]};
    }
    auto& compute = *owner->compute;
    if (!owner->zero.valid()) {
        if (!create(compute, owner->zero, sizeof(float), "GasInventoryZero")) {
            return {};
        }
        ComputeBufferHandle fields[2] = {owner->zero, owner->zero};
        const uint32_t count = 1;
        ComputeDispatch command;
        command.kernel = "sim_inventory_clear";
        command.buffers = fields;
        command.buffer_count = 2;
        command.constants = &count;
        command.constants_size = sizeof(count);
        if (!compute.dispatch(command)) {
            return {};
        }
    }
    return {owner->zero, owner->zero};
}

void prepareGasInventoryHostProducer(SimulationGridDomainState& state) {
    if (!state.gas_inventory_error.empty()) {
        throw std::runtime_error(state.gas_inventory_error);
    }
    if (gasInventoryDevice(state)) {
        return;
    }
    if (state.gas_inventory_revision != 0) {
        throw std::runtime_error("gas inventory producer has a stale device view");
    }
    const std::size_t cells = gasGrid(state).getCellCount();
    state.gas_phase_mass_kg.resize(cells, 0.0f);
    state.gas_phase_energy_j.resize(cells, 0.0f);
    state.gas_phase_condensate_kg.resize(cells, 0.0f);
    state.gas_phase_rain_kg.resize(cells, 0.0f);
    state.gas_phase_rain_energy_j.resize(cells, 0.0f);
    state.gas_phase_foreign_kg.resize(cells, 0.0f);
}

void depositGasInventory(SimulationGridDomainState& state, std::size_t cell,
                         float mass_kg, float energy_j) {
    if (cell >= gasGrid(state).getCellCount() || !std::isfinite(mass_kg) ||
        !std::isfinite(energy_j) || mass_kg < 0.0f || energy_j < 0.0f) {
        throw std::runtime_error("invalid gas phase inventory deposit");
    }
    if (mass_kg == 0.0f && energy_j == 0.0f) {
        return;
    }
    if (auto owner = gasInventoryDevice(state)) {
        auto& value = owner->pending[static_cast<uint32_t>(cell)];
        const float mass = value.first + mass_kg;
        const float energy = value.second + energy_j;
        if (!std::isfinite(mass) || !std::isfinite(energy)) {
            throw std::runtime_error("gas inventory deposit overflow");
        }
        value = {mass, energy};
    } else {
        prepareGasInventoryHostProducer(state);
        state.gas_phase_mass_kg[cell] += mass_kg;
        state.gas_phase_energy_j[cell] += energy_j;
    }
}

void depositGasCondensate(SimulationGridDomainState& state, std::size_t cell,
                          float mass_kg, float energy_j) {
    if (cell >= gasGrid(state).getCellCount() || !std::isfinite(mass_kg) ||
        !std::isfinite(energy_j) || mass_kg < 0.0f || energy_j < 0.0f) {
        throw std::runtime_error("invalid gas condensate deposit");
    }
    if (mass_kg == 0.0f && energy_j == 0.0f) {
        return;
    }
    state.gas_liquid_water_present = true;
    if (auto owner = gasInventoryDevice(state)) {
        auto& value = owner->pending_condensate[static_cast<uint32_t>(cell)];
        const float mass = value.first + mass_kg;
        const float energy = value.second + energy_j;
        if (!std::isfinite(mass) || !std::isfinite(energy)) {
            throw std::runtime_error("gas condensate deposit overflow");
        }
        value = {mass, energy};
    } else {
        prepareGasInventoryHostProducer(state);
        state.gas_phase_condensate_kg[cell] += mass_kg;
        state.gas_phase_energy_j[cell] += energy_j;
    }
}

void depositGasForeign(SimulationGridDomainState& state, std::size_t cell,
                       float mass_kg, float energy_j) {
    if (cell >= gasGrid(state).getCellCount() || !std::isfinite(mass_kg) ||
        !std::isfinite(energy_j) || mass_kg < 0.0f || energy_j < 0.0f) {
        throw std::runtime_error("invalid non-condensable gas deposit");
    }
    if (mass_kg == 0.0f && energy_j == 0.0f) {
        return;
    }
    if (auto owner = gasInventoryDevice(state)) {
        auto& value = owner->pending_foreign[static_cast<uint32_t>(cell)];
        const float mass = value.first + mass_kg;
        const float energy = value.second + energy_j;
        if (!std::isfinite(mass) || !std::isfinite(energy)) {
            throw std::runtime_error("non-condensable gas deposit overflow");
        }
        value = {mass, energy};
    } else {
        prepareGasInventoryHostProducer(state);
        state.gas_phase_foreign_kg[cell] += mass_kg;
        state.gas_phase_energy_j[cell] += energy_j;
    }
}

namespace {
// One sorted unique-cell deposit batch into (mass_field, energy_field).
bool dispatchDeposits(GasInventoryDevice& owner, ComputeBufferHandle& buffer,
                      const std::map<uint32_t, std::pair<float, float>>& pending,
                      ComputeBufferHandle mass_field, ComputeBufferHandle energy_field) {
    std::vector<Deposit> deposits;
    deposits.reserve(pending.size());
    for (const auto& [cell, values] : pending) {
        deposits.push_back({cell, values.first, values.second});
    }
    auto& compute = *owner.compute;
    const std::size_t bytes = deposits.size() * sizeof(Deposit);
    if (!buffer.valid()) {
        if (!create(compute, buffer, bytes, "GasInventoryDeposits")) {
            return false;
        }
    } else if (compute.getBufferSize(buffer) < bytes &&
               !compute.resizeBuffer(buffer, bytes)) {
        return false;
    }
    if (!compute.uploadBuffer(buffer, deposits.data(), bytes)) {
        return false;
    }
    const uint32_t count = static_cast<uint32_t>(deposits.size());
    ComputeBufferHandle bindings[3] = {mass_field, energy_field, buffer};
    ComputeDispatch command;
    command.kernel = "sim_inventory_deposit";
    command.buffers = bindings;
    command.buffer_count = 3;
    command.constants = &count;
    command.constants_size = sizeof(count);
    command.groups = gasInventoryDispatch(count);
    return compute.dispatch(command);
}
} // namespace

bool flushGasInventoryDeposits(SimulationGridDomainState& state) {
    auto owner = gasInventoryDevice(state);
    if (!owner) {
        return false;
    }
    if (owner->pending.empty() && owner->pending_condensate.empty() &&
        owner->pending_foreign.empty()) {
        return true;
    }
    if (!ensureGasInventoryStorage(state)) {
        return false;
    }
    // One upload buffer per batch: an upload is not ordered after a recorded
    // dispatch, so reusing one buffer could overwrite the earlier batch.
    if (!owner->pending.empty() &&
        !dispatchDeposits(*owner, owner->deposits, owner->pending,
                          owner->fields[0], owner->fields[3])) {
        return false;
    }
    if (!owner->pending_condensate.empty() &&
        !dispatchDeposits(*owner, owner->condensate_deposits, owner->pending_condensate,
                          owner->fields[6], owner->fields[3])) {
        return false;
    }
    if (!owner->pending_foreign.empty() &&
        !dispatchDeposits(*owner, owner->foreign_deposits, owner->pending_foreign,
                          owner->fields[11], owner->fields[3])) {
        return false;
    }
    owner->pending.clear();
    owner->pending_condensate.clear();
    owner->pending_foreign.clear();
    owner->populated = true;
    markGasInventoryDeviceChanged(state);
    return true;
}

bool snapshotGasInventory(SimulationGridDomainState& snapshot) {
    if (!snapshot.gas_inventory_error.empty() || !snapshot.gas_phase_pending_momentum.empty()) {
        return false;
    }
    return materialize(snapshot);
}

void holdGasInventory(SimulationGridDomainState& state, const char* reason) {
    if (state.gas_inventory_error.empty()) {
        state.gas_inventory_error = reason;
    }
    state.gas_stats.inventory_transport = "gpu_failed";
    state.gas_stats.inventory_step_held = true;
    state.gas_stats.inventory_error = state.gas_inventory_error;
}

bool materializeGasInventoryForBackendChange(SimulationGridDomainState& state) {
    return materialize(state);
}

void prepareGasInventoryRestore(std::vector<SimulationGridDomainState>& states) {
    for (auto& state : states) {
        if (!snapshotGasInventory(state)) {
            throw std::runtime_error("gas inventory cache restore lacks a current snapshot");
        }
    }
}

void releaseGasInventoryDomainsForBackendChange(
    std::vector<SimulationGridDomainState>& states,
    std::vector<SimulationGridDomainComputeBuffers>& buffers,
    SimulationComputeContext& compute) {
    // Validate every mirror before destroying any device owner.
    std::vector<GasInventoryHostImage> mirrors;
    if (!captureGasInventoryHostImages(states, mirrors)) {
        throw std::runtime_error("gas inventory backend switch readback failed");
    }
    for (std::size_t index = 0; index < states.size(); ++index) {
        states[index].gas_phase_mass_kg.swap(mirrors[index].mass);
        states[index].gas_phase_energy_j.swap(mirrors[index].energy);
        states[index].gas_phase_condensate_kg.swap(mirrors[index].condensate);
        states[index].gas_phase_rain_kg.swap(mirrors[index].rain);
        states[index].gas_phase_rain_energy_j.swap(mirrors[index].rain_energy);
        states[index].gas_phase_foreign_kg.swap(mirrors[index].foreign);
        states[index].gas_inventory_device.reset();
        states[index].gas_inventory_revision = 0;
    }
    for (auto& bank : buffers) {
        releaseGasInventoryGpu(compute, bank.gas_inventory);
    }
}

GasInventorySummary readGasInventorySummary(const SimulationGridDomainState& state) {
    if (!state.gas_inventory_error.empty()) {
        throw std::runtime_error(state.gas_inventory_error);
    }
    auto owner = gasInventoryDevice(state);
    if (!owner) {
        if (state.gas_inventory_revision != 0) {
            throw std::runtime_error("gas inventory summary has a stale device view");
        }
        GasInventorySummary summary;
        const auto& grid = gasGrid(state);
        double weighted[3]{};
        const bool layout_valid = grid.nx > 0 && grid.ny > 0 && grid.nz > 0 &&
            state.gas_phase_mass_kg.size() == grid.getCellCount();
        for (std::size_t cell = 0; layout_valid && cell < state.gas_phase_mass_kg.size(); ++cell) {
            const float mass = state.gas_phase_mass_kg[cell];
            if (!std::isfinite(mass) || mass <= 0.0f) {
                continue;
            }
            const int x = static_cast<int>(cell % grid.nx);
            const int y = static_cast<int>((cell / grid.nx) % grid.ny);
            const int z = static_cast<int>(cell / (std::size_t(grid.nx) * grid.ny));
            const Vec3 position = grid.gridToWorld(x, y, z);
            weighted[0] += static_cast<double>(position.x) * mass;
            weighted[1] += static_cast<double>(position.y) * mass;
            weighted[2] += static_cast<double>(position.z) * mass;
            summary.mass_kg += mass;
            ++summary.active_cells;
        }
        for (const float energy : state.gas_phase_energy_j) {
            if (std::isfinite(energy) && energy > 0.0f) {
                summary.energy_j += energy;
            }
        }
        for (const float water : state.gas_phase_condensate_kg) {
            if (std::isfinite(water) && water > 0.0f) {
                summary.condensate_kg += water;
                ++summary.condensate_cells;
            }
        }
        double rain_weighted[3]{};
        for (std::size_t cell = 0; cell < state.gas_phase_rain_kg.size(); ++cell) {
            const float rain = state.gas_phase_rain_kg[cell];
            if (std::isfinite(rain) && rain > 0.0f) {
                summary.rain_kg += rain;
                ++summary.rain_cells;
                if (layout_valid) {
                    const int x = static_cast<int>(cell % grid.nx);
                    const int y = static_cast<int>((cell / grid.nx) % grid.ny);
                    const int z = static_cast<int>(cell / (std::size_t(grid.nx) * grid.ny));
                    const Vec3 position = grid.gridToWorld(x, y, z);
                    rain_weighted[0] += static_cast<double>(position.x) * rain;
                    rain_weighted[1] += static_cast<double>(position.y) * rain;
                    rain_weighted[2] += static_cast<double>(position.z) * rain;
                }
            }
        }
        if (summary.rain_kg > 0.0) {
            summary.rain_centroid = Vec3(static_cast<float>(rain_weighted[0] / summary.rain_kg),
                                         static_cast<float>(rain_weighted[1] / summary.rain_kg),
                                         static_cast<float>(rain_weighted[2] / summary.rain_kg));
        }
        for (const float foreign : state.gas_phase_foreign_kg) {
            if (std::isfinite(foreign) && foreign > 0.0f) {
                summary.foreign_kg += foreign;
                ++summary.foreign_cells;
            }
        }
        if (summary.mass_kg > 0.0) {
            summary.centroid = Vec3(static_cast<float>(weighted[0] / summary.mass_kg),
                                    static_cast<float>(weighted[1] / summary.mass_kg),
                                    static_cast<float>(weighted[2] / summary.mass_kg));
        }
        return summary;
    }
    if (!owner->pending.empty() || !owner->pending_condensate.empty() ||
        !owner->pending_foreign.empty()) {
        throw std::runtime_error("gas inventory summary requested before deposit commit");
    }
    const auto& grid = gasGrid(state);
    if (owner->summary_valid && owner->summary_voxel == grid.voxel_size &&
        owner->summary_origin.x == grid.origin.x && owner->summary_origin.y == grid.origin.y &&
        owner->summary_origin.z == grid.origin.z) {
        return owner->summary;
    }
    if (!owner->populated) {
        return {};
    }
    auto& compute = *owner->compute;
    const uint32_t blocks = static_cast<uint32_t>((owner->cells + kThreads - 1) / kThreads);
    if (!owner->summary_partials.valid() &&
        !create(compute, owner->summary_partials, blocks * 8 * sizeof(double),
                "GasInventorySummaryPartials")) {
        throw std::runtime_error("gas inventory summary allocation failed");
    }
    if (!owner->summary_totals.valid() &&
        !create(compute, owner->summary_totals, 8 * sizeof(double), "GasInventorySummary")) {
        throw std::runtime_error("gas inventory summary allocation failed");
    }
    SummaryConstants pc{static_cast<uint32_t>(owner->cells), blocks, 0,
                        grid.nx, grid.ny, grid.voxel_size,
                        grid.origin.x, grid.origin.y, grid.origin.z};
    ComputeBufferHandle bindings[4] = {owner->fields[0], owner->fields[3],
                                      owner->summary_partials, owner->summary_totals};
    ComputeDispatch command;
    command.kernel = "sim_inventory_summary";
    command.buffers = bindings;
    command.buffer_count = 4;
    command.constants = &pc;
    command.constants_size = sizeof(pc);
    command.groups = gasInventoryDispatch(owner->cells);
    if (!compute.dispatch(command)) {
        throw std::runtime_error("gas inventory summary reduction failed");
    }
    pc.phase = 1;
    command.groups = {1, 1, 1};
    double totals[8]{};
    if (!compute.dispatch(command)) {
        throw std::runtime_error("gas inventory summary dispatch failed");
    }
    compute.beginTransferBatch();
    const bool downloaded = compute.downloadBuffer(
        owner->summary_totals, totals, sizeof(totals));
    const bool completed = compute.endTransferBatch();
    if (!downloaded || !completed) {
        throw std::runtime_error("gas inventory summary readback failed");
    }
    owner->summary = {totals[0], totals[1], static_cast<uint64_t>(totals[2]), Vec3()};
    if (totals[0] > 0.0) {
        owner->summary.centroid = Vec3(static_cast<float>(totals[3] / totals[0]),
                                       static_cast<float>(totals[4] / totals[0]),
                                       static_cast<float>(totals[5] / totals[0]));
    }
    // Fog water, rain water and non-condensable gas: the same reduction over
    // their mass slots (the "energy" binding reads the same field, ignored).
    if (owner->storage_initialized) {
        for (const std::size_t slot : {std::size_t{6}, std::size_t{9}, std::size_t{11}}) {
            ComputeBufferHandle water[4] = {owner->fields[slot], owner->fields[slot],
                                            owner->summary_partials, owner->summary_totals};
            pc.phase = 0;
            command.buffers = water;
            command.groups = gasInventoryDispatch(owner->cells);
            double water_totals[8]{};
            if (!compute.dispatch(command)) {
                throw std::runtime_error("gas water summary reduction failed");
            }
            pc.phase = 1;
            command.groups = {1, 1, 1};
            if (!compute.dispatch(command)) {
                throw std::runtime_error("gas water summary dispatch failed");
            }
            compute.beginTransferBatch();
            const bool water_downloaded = compute.downloadBuffer(
                owner->summary_totals, water_totals, sizeof(water_totals));
            if (!compute.endTransferBatch() || !water_downloaded) {
                throw std::runtime_error("gas water summary readback failed");
            }
            if (slot == 6) {
                owner->summary.condensate_kg = water_totals[0];
                owner->summary.condensate_cells = static_cast<uint64_t>(water_totals[2]);
            } else if (slot == 9) {
                owner->summary.rain_kg = water_totals[0];
                owner->summary.rain_cells = static_cast<uint64_t>(water_totals[2]);
                // The reduction weights positions by the bound mass slot.
                if (water_totals[0] > 0.0) {
                    owner->summary.rain_centroid = Vec3(
                        static_cast<float>(water_totals[3] / water_totals[0]),
                        static_cast<float>(water_totals[4] / water_totals[0]),
                        static_cast<float>(water_totals[5] / water_totals[0]));
                }
            } else {
                owner->summary.foreign_kg = water_totals[0];
                owner->summary.foreign_cells = static_cast<uint64_t>(water_totals[2]);
            }
            owner->query_download_bytes += sizeof(water_totals);
        }
    }
    owner->summary_valid = true;
    owner->populated = totals[0] > 0.0 || totals[1] > 0.0 ||
        owner->summary.condensate_kg > 0.0 || owner->summary.rain_kg > 0.0 ||
        owner->summary.foreign_kg > 0.0;
    owner->summary_origin = grid.origin;
    owner->summary_voxel = grid.voxel_size;
    owner->query_download_bytes += sizeof(totals);
    return owner->summary;
}
bool gasInventoryHasContent(const SimulationGridDomainState& state) {
    if (!state.gas_inventory_error.empty()) {
        return true;
    }
    if (state.gas_inventory_revision != 0) {
        const auto owner = gasInventoryDevice(state);
        return !owner || owner->populated || !owner->pending.empty() ||
               !owner->pending_condensate.empty() || !owner->pending_foreign.empty();
    }
    const auto positive = [](float value) { return std::isfinite(value) && value > 0.0f; };
    return std::any_of(state.gas_phase_mass_kg.begin(), state.gas_phase_mass_kg.end(), positive) ||
           std::any_of(state.gas_phase_energy_j.begin(), state.gas_phase_energy_j.end(), positive) ||
           std::any_of(state.gas_phase_condensate_kg.begin(),
                       state.gas_phase_condensate_kg.end(), positive) ||
           std::any_of(state.gas_phase_rain_kg.begin(), state.gas_phase_rain_kg.end(), positive) ||
           std::any_of(state.gas_phase_rain_energy_j.begin(),
                       state.gas_phase_rain_energy_j.end(), positive) ||
           std::any_of(state.gas_phase_foreign_kg.begin(),
                       state.gas_phase_foreign_kg.end(), positive);
}
} // namespace RayTrophiSim::Fluid
