#include "Fluid/GasSourceProvenance.h"
#include "Fluid/GasInventoryGpu.h"
#include "ParticleSimulation.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace RayTrophiSim::Fluid {
namespace {
using Field = SimulationGridDomainComputeBuffers::GridField;
struct SourceBuffers {
    std::array<ComputeBufferHandle, 2> fields{};
};
struct Constants {
    uint32_t cells;
    uint32_t phase;
    uint32_t residual;
    uint32_t policy;
};
struct AdvectionConstants {
    int nx;
    int ny;
    int nz;
    float voxel;
    float dt;
    int boundary;
    float background = 0.0f;
};
static_assert(sizeof(Constants) == 16);
static_assert(sizeof(AdvectionConstants) == 28);
void require(bool ok) {
    if (!ok) {
        throw std::runtime_error("Gas source provenance failed; reset simulation required");
    }
}
bool pending(const GasSourceImage& image) {
    return !image.pending_density.empty() || !image.pending_fuel.empty();
}
}

struct GasSourceDevice {
    SimulationComputeContext* compute = nullptr;
    std::size_t cells = 0;
    std::map<uint64_t, SourceBuffers> sources;
    // Shared advection scratch, reduction and source staging (not per emitter).
    std::array<ComputeBufferHandle, 5> scratch{};
    void release() {
        if (!compute) {
            return;
        }
        compute->synchronize();
        for (auto& [uid, source] : sources) {
            for (auto handle : source.fields) {
                if (handle.valid()) {
                    compute->destroyBuffer(handle);
                }
            }
        }
        for (auto handle : scratch) {
            if (handle.valid()) {
                compute->destroyBuffer(handle);
            }
        }
        compute = nullptr;
    }
    ~GasSourceDevice() {
        release();
    }
};

void validateGasSourceBudget(std::size_t cells, std::size_t streams) {
    constexpr std::size_t budget = 256u * 1024u * 1024u;
    // Device currents + shared scratch, host snapshots and source staging.
    const std::size_t floats = streams * 6u + 6u;
    if (streams > 9 || cells == 0 || cells > budget / sizeof(float) / floats) {
        throw std::runtime_error("Gas source provenance exceeds 8 emitters / 256 MiB budget");
    }
}

GasSourceImage* prepareGasSourceDeposit(SimulationGridDomainState& state,
    uint64_t uid, const std::string& name) {
    if (state.type != SimulationDomainType::Matter) {
        return nullptr;
    }
    if (uid == 0) {
        throw std::runtime_error("Gas source requires a stable nonzero timeline UID");
    }
    auto& images = state.gas_source_images;
    const std::size_t cells = state.grid.getCellCount();
    validateGasSourceBudget(cells, images.size() +
        (images.empty() ? 1u : 0u) + (images.count(uid) ? 0u : 1u));
    if (images.empty()) {
        auto& residual = images[0];
        residual.name = "Vapor / Unattributed Gas";
        residual.density = state.grid.density;
        residual.fuel = state.grid.fuel;
    }
    auto& image = images[uid];
    image.name = name;
    if (image.density.empty()) {
        image.density.assign(cells, 0.0f);
        image.fuel.assign(cells, 0.0f);
    }
    if (!pending(image)) {
        image.pending_density.assign(cells, 0.0f);
        image.pending_fuel.assign(cells, 0.0f);
    }
    return &image;
}

bool snapshotGasSources(SimulationGridDomainState& snapshot) {
    auto owner = snapshot.gas_source_device.lock();
    if (!owner) {
        return !snapshot.gas_source_device_current;
    }
    for (auto& [uid, image] : snapshot.gas_source_images) {
        if (!owner->sources.count(uid)) {
            if (image.density.size() == owner->cells) {
                continue;
            }
            return false;
        }
        image.density.resize(owner->cells);
        image.fuel.resize(owner->cells);
        const auto& source = owner->sources.at(uid);
        if (!owner->compute->downloadBuffer(source.fields[0], image.density.data(),
                owner->cells * sizeof(float)) ||
            !owner->compute->downloadBuffer(source.fields[1], image.fuel.data(),
                owner->cells * sizeof(float))) {
            return false;
        }
    }
    snapshot.gas_source_device.reset();
    snapshot.gas_source_device_current = false;
    return true;
}

void materializeGasSources(SimulationGridDomainState& state) {
    require(snapshotGasSources(state));
}

void releaseGasSources(SimulationGridDomainComputeBuffers& buffers) {
    if (buffers.gas_source_owner) {
        buffers.gas_source_owner->release();
        buffers.gas_source_owner.reset();
    }
}

void releaseGasSourceDomains(std::vector<SimulationGridDomainState>& states,
    std::vector<SimulationGridDomainComputeBuffers>& buffers) {
    std::vector<SimulationGridDomainState> snapshots(states.size());
    for (std::size_t i = 0; i < states.size(); ++i) {
        snapshots[i].gas_source_images = states[i].gas_source_images;
        snapshots[i].gas_source_device = states[i].gas_source_device;
        snapshots[i].gas_source_device_current = states[i].gas_source_device_current;
        require(snapshotGasSources(snapshots[i]));
    }
    for (std::size_t i = 0; i < states.size(); ++i) {
        states[i].gas_source_images = std::move(snapshots[i].gas_source_images);
        states[i].gas_source_device.reset();
        states[i].gas_source_device_current = false;
    }
    for (auto& bank : buffers) {
        releaseGasSources(bank);
    }
}

namespace {
void cpuReconcile(SimulationGridDomainState& state, unsigned policy) {
    auto& images = state.gas_source_images;
    const auto& grid = state.grid;
    for (std::size_t cell = 0; cell < grid.getCellCount(); ++cell) {
        double density = 0.0;
        double fuel = 0.0;
        for (const auto& [uid, image] : images) {
            density += std::max(0.0f, image.density[cell]);
            fuel += std::max(0.0f, image.fuel[cell]);
        }
        const float actual_d = std::max(0.0f, grid.density[cell]);
        const float actual_f = std::max(0.0f, grid.fuel[cell]);
        for (auto& [uid, image] : images) {
            const float d = std::max(0.0f, image.density[cell]);
            const float f = std::max(0.0f, image.fuel[cell]);
            const float excess = std::max(0.0, actual_d - density);
            const float share = policy == 1 && fuel > 0.0 ? float(f / fuel) :
                (uid == 0 ? 1.0f : 0.0f);
            image.density[cell] = density > actual_d || policy == 2
                ? (density > 0.0 ? float(d * actual_d / density) :
                    (uid == 0 ? actual_d : 0.0f))
                : d + excess * share;
            image.fuel[cell] = fuel > actual_f || policy == 2
                ? (fuel > 0.0 ? float(f * actual_f / fuel) :
                    (uid == 0 ? actual_f : 0.0f))
                : f + (uid == 0 ? float(actual_f - fuel) : 0.0f);
        }
    }
}

void ensureReference(SimulationComputeContext& compute,
    SimulationGridDomainComputeBuffers& buffers, Field field,
    ComputeBufferHandle handle, const std::vector<float>& values) {
    if (!buffers.deviceHasCurrent(field)) {
        require(buffers.hostHasCurrent(field));
        require(compute.uploadBuffer(handle, values.data(), values.size() * sizeof(float)));
        buffers.markSynced(field);
    }
}

void gpuStage(SimulationGridDomainState& state, const GridFluid::SolverParams& params,
    float dt, GasSourceStage stage, SimulationComputeContext& compute,
    SimulationGridDomainComputeBuffers& buffers) {
    const auto& grid = state.grid;
    const std::size_t cells = grid.getCellCount();
    validateGasSourceBudget(cells, state.gas_source_images.size());
    auto owner = state.gas_source_device.lock();
    if (!owner) {
        require(!state.gas_source_device_current);
        releaseGasSources(buffers);
        owner = std::make_shared<GasSourceDevice>();
        owner->compute = &compute;
        owner->cells = cells;
        buffers.gas_source_owner = owner;
        state.gas_source_device = owner;
        state.gas_source_device_current = true;
        const auto usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
            ComputeBufferUsage::Download | ComputeBufferUsage::ReadWrite;
        for (std::size_t i = 0; i < owner->scratch.size(); ++i) {
            owner->scratch[i] = compute.createBuffer({"GasSourceScratch",
                cells * sizeof(float) * (i == 2 ? 2u : 1u), usage});
            require(owner->scratch[i].valid());
        }
    }
    require(owner->compute == &compute && owner->cells == cells);
    const auto dispatch = [&](Constants pc, SourceBuffers& source) {
        ComputeBufferHandle bindings[] = {source.fields[0], source.fields[1],
            buffers.density, buffers.fuel, owner->scratch[2],
            owner->scratch[3], owner->scratch[4]};
        ComputeDispatch command;
        command.kernel = "sim_gas_source_partition";
        command.buffers = bindings;
        command.buffer_count = 7;
        command.constants = &pc;
        command.constants_size = sizeof(pc);
        command.groups = gasInventoryDispatch(cells);
        require(compute.dispatch(command));
    };
    for (auto& [uid, image] : state.gas_source_images) {
        if (!owner->sources.count(uid)) {
            const auto usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                ComputeBufferUsage::Download | ComputeBufferUsage::ReadWrite;
            auto& source = owner->sources[uid];
            for (auto& handle : source.fields) {
                handle = compute.createBuffer({"GasSourceCurrent", cells * sizeof(float), usage});
                require(handle.valid());
            }
            require(image.density.size() == cells && image.fuel.size() == cells);
            require(compute.uploadBuffer(source.fields[0], image.density.data(), cells * sizeof(float)));
            require(compute.uploadBuffer(source.fields[1], image.fuel.data(), cells * sizeof(float)));
        }
        if (pending(image)) {
            // Staging is reused only after completion: uploads are not stream ordered.
            compute.synchronize();
            require(compute.uploadBuffer(owner->scratch[3], image.pending_density.data(),
                cells * sizeof(float)));
            require(compute.uploadBuffer(owner->scratch[4], image.pending_fuel.data(),
                cells * sizeof(float)));
            dispatch({uint32_t(cells), 3, uid == 0, 0}, owner->sources.at(uid));
            compute.synchronize();
            image.pending_density.clear();
            image.pending_fuel.clear();
        }
    }
    if (stage == GasSourceStage::Advect) {
        ensureReference(compute, buffers, Field::VelX, buffers.vel_x, grid.vel_x);
        ensureReference(compute, buffers, Field::VelY, buffers.vel_y, grid.vel_y);
        ensureReference(compute, buffers, Field::VelZ, buffers.vel_z, grid.vel_z);
        const AdvectionConstants pc{grid.nx, grid.ny, grid.nz, grid.voxel_size,
            dt, int(params.boundary)};
        for (auto& [uid, source] : owner->sources) {
            for (std::size_t channel = 0; channel < source.fields.size(); ++channel) {
                if ((channel == 0 && !params.channel_density) ||
                    (channel == 1 && !params.channel_fuel)) {
                    continue;
                }
                auto& field = source.fields[channel];
                ComputeBufferHandle bindings[] = {buffers.vel_x, buffers.vel_y, buffers.vel_z,
                    field, owner->scratch[0], owner->scratch[1], buffers.gas_solid_mask};
                ComputeDispatch command;
                command.kernel = "sim_grid_advect_scalar";
                command.buffers = bindings;
                command.buffer_count = 7;
                command.constants = &pc;
                command.constants_size = sizeof(pc);
                command.groups.groups_x = (uint32_t(cells) + 255u) / 256u;
                require(compute.dispatch(command));
                std::size_t result = 0;
                if (params.advection == GridFluid::Advection::MacCormack) {
                    command.kernel = "sim_grid_maccormack_scalar";
                    require(compute.dispatch(command));
                    result = 1;
                }
                std::swap(field, owner->scratch[result]);
            }
        }
        return;
    }
    ensureReference(compute, buffers, Field::Density, buffers.density, grid.density);
    ensureReference(compute, buffers, Field::Fuel, buffers.fuel, grid.fuel);
    const unsigned policy = stage == GasSourceStage::Reacted ? 1u :
        stage == GasSourceStage::Transported ? 2u : 0u;
    dispatch({uint32_t(cells), 0, 0, policy}, owner->sources.begin()->second);
    for (auto& [uid, source] : owner->sources) {
        dispatch({uint32_t(cells), 1, uid == 0, policy}, source);
    }
    for (auto& [uid, source] : owner->sources) {
        dispatch({uint32_t(cells), 2, uid == 0, policy}, source);
    }
}
}

static void gasSourceStageImpl(SimulationGridDomainState& state,
    const GridFluid::SolverParams& params, float dt, GasSourceStage stage,
    SimulationComputeContext* compute, SimulationGridDomainComputeBuffers* buffers) {
    if (state.gas_source_images.empty()) {
        return;
    }
    if (compute && buffers && compute->backendType() == ComputeBackendType::VulkanCompute) {
        gpuStage(state, params, dt, stage, *compute, *buffers);
        return;
    }
    materializeGasSources(state);
    for (auto& [uid, image] : state.gas_source_images) {
        if (pending(image)) {
            for (std::size_t cell = 0; cell < image.density.size(); ++cell) {
                image.density[cell] += image.pending_density[cell];
                image.fuel[cell] += image.pending_fuel[cell];
            }
            image.pending_density.clear();
            image.pending_fuel.clear();
        }
        if (stage == GasSourceStage::Advect) {
            if (params.channel_density) {
                GridFluid::advectPassiveScalarField(state.grid, params, image.density, 0.0f, dt);
            }
            if (params.channel_fuel) {
                GridFluid::advectPassiveScalarField(state.grid, params, image.fuel, 0.0f, dt);
            }
        }
    }
    if (stage != GasSourceStage::Advect) {
        cpuReconcile(state, stage == GasSourceStage::Reacted ? 1u :
            stage == GasSourceStage::Transported ? 2u : 0u);
    }
}

void gasSourceStage(SimulationGridDomainState& state,
    const GridFluid::SolverParams& params, float dt, GasSourceStage stage,
    SimulationComputeContext* compute, SimulationGridDomainComputeBuffers* buffers) {
    try {
        gasSourceStageImpl(state, params, dt, stage, compute, buffers);
    } catch (const std::exception& exception) {
        holdGasInventory(state, exception.what());
        throw;
    }
}
} // namespace RayTrophiSim::Fluid
