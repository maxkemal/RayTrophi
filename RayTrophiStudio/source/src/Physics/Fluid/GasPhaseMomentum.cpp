#include "Fluid/GasPhaseMomentum.h"
#include "Fluid/GasInventoryGpu.h"
#include "Fluid/MatterPhaseGrid.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace RayTrophiSim::Fluid {
namespace {
constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();
struct FaceDeposit {
    uint32_t axis;
    uint32_t face;
    float mass;
    float momentum;
    uint32_t left;
    uint32_t right;
    float padding[2]{};
};
static_assert(sizeof(FaceDeposit) == 32);
struct Constants {
    uint32_t count;
    float cell_volume;
    float ambient;
    float gas_ambient;
    float kelvin_per_unit;
};
static_assert(sizeof(Constants) == 20);

bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
}

void queueGasPhaseMomentum(SimulationGridDomainState& gas, std::size_t cell,
                           float mass, const Vec3& momentum) {
    if (cell >= gasGrid(gas).getCellCount() || !(mass >= 0) ||
        !std::isfinite(mass) || !finite(momentum)) {
        throw std::runtime_error("invalid phase momentum deposit");
    }
    if (!(mass > 0)) {
        return;
    }
    auto owner = gasInventoryDevice(gas);
    auto& deposits = owner ? owner->pending_momentum : gas.gas_phase_pending_momentum;
    auto& entry = deposits[static_cast<uint32_t>(cell)];
    entry.mass += mass;
    entry.momentum += momentum;
    if (!std::isfinite(entry.mass) || !finite(entry.momentum)) {
        throw std::runtime_error("phase momentum deposit overflow");
    }
}

bool flushGasPhaseMomentum(SimulationGridDomainState& gas,
                           const SimulationGridDomainDesc& domain,
                           const MaterialTemperatureScale& scale,
                           float ambient_kelvin, float gas_ambient_normalized,
                           SimulationComputeContext* compute,
                           SimulationGridDomainComputeBuffers* buffers) {
    auto owner = gasInventoryDevice(gas);
    auto& deposits = owner ? owner->pending_momentum : gas.gas_phase_pending_momentum;
    if (deposits.empty()) {
        return true;
    }
    const auto& grid = gasGrid(gas);
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        !(grid.voxel_size > 0) || !(scale.kelvin_per_unit > 0)) {
        return false;
    }
    std::map<uint64_t, FaceDeposit> accumulated;
    const bool periodic = domain.boundary_mode == SimulationGridDomainBoundaryMode::Periodic;
    const bool closed = domain.boundary_mode == SimulationGridDomainBoundaryMode::Closed;
    const int dimensions[3] = {grid.nx, grid.ny, grid.nz};
    for (const auto& entry : deposits) {
        const uint32_t cell = entry.first;
        const int xyz[3] = {static_cast<int>(cell % grid.nx),
                           static_cast<int>((cell / grid.nx) % grid.ny),
                           static_cast<int>(cell / (grid.nx * grid.ny))};
        const float momentum[3] = {entry.second.momentum.x,
                                    entry.second.momentum.y, entry.second.momentum.z};
        for (uint32_t axis = 0; axis < 3; ++axis) {
            for (int side = 0; side < 2; ++side) {
                int face_xyz[3] = {xyz[0], xyz[1], xyz[2]};
                face_xyz[axis] += side;
                const bool edge = face_xyz[axis] == 0 ||
                                  face_xyz[axis] == dimensions[axis];
                if (closed && edge) {
                    continue; // A container wall receives the constrained reaction.
                }
                int adjacent[2][3] = {
                    {face_xyz[0], face_xyz[1], face_xyz[2]},
                    {face_xyz[0], face_xyz[1], face_xyz[2]}
                };
                --adjacent[0][axis];
                uint32_t cells[2] = {invalid, invalid};
                for (int n = 0; n < 2; ++n) {
                    if (periodic) {
                        adjacent[n][axis] = (adjacent[n][axis] + dimensions[axis]) %
                                             dimensions[axis];
                    }
                    if (adjacent[n][axis] >= 0 && adjacent[n][axis] < dimensions[axis]) {
                        cells[n] = static_cast<uint32_t>(grid.cellIndex(
                            adjacent[n][0], adjacent[n][1], adjacent[n][2]));
                    }
                }
                const auto insert_face = [&](int coordinate) {
                    int face[3] = {face_xyz[0], face_xyz[1], face_xyz[2]};
                    face[axis] = coordinate;
                    const int nx = grid.nx + (axis == 0);
                    const int ny = grid.ny + (axis == 1);
                    const uint32_t index = static_cast<uint32_t>(
                        face[0] + nx * (face[1] + ny * face[2]));
                    const uint64_t key = (uint64_t{axis} << 32u) | index;
                    auto [found, inserted] = accumulated.try_emplace(key,
                        FaceDeposit{axis, index, 0, 0, cells[0], cells[1]});
                    found->second.mass += 0.5f * entry.second.mass;
                    found->second.momentum += 0.5f * momentum[axis];
                };
                insert_face(face_xyz[axis]);
                if (periodic && edge) {
                    // Duplicated end faces represent the same periodic MAC face.
                    insert_face(face_xyz[axis] == 0 ? dimensions[axis] : 0);
                }
            }
        }
    }
    std::vector<FaceDeposit> records;
    records.reserve(accumulated.size());
    for (const auto& entry : accumulated) {
        records.push_back(entry.second);
    }
    if (records.empty()) {
        deposits.clear();
        return true;
    }
    const float volume = grid.voxel_size * grid.voxel_size * grid.voxel_size;
    if (owner) {
        if (!compute || !buffers || compute != owner->compute ||
            !owner->storage_initialized ||
            !buffers->temperature.valid() || !buffers->gas_solid_mask.valid()) {
            return false;
        }
        const std::size_t bytes = records.size() * sizeof(FaceDeposit);
        if (!owner->phase_momentum.valid()) {
            const auto usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload;
            owner->phase_momentum = compute->createBuffer({"GasPhaseMomentum", bytes, usage});
        } else if (compute->getBufferSize(owner->phase_momentum) < bytes &&
                   !compute->resizeBuffer(owner->phase_momentum, bytes)) {
            return false;
        }
        if (!owner->phase_momentum.valid() ||
            !compute->uploadBuffer(owner->phase_momentum, records.data(), bytes)) {
            return false;
        }
        const Constants pc{static_cast<uint32_t>(records.size()), volume, ambient_kelvin,
                           gas_ambient_normalized, scale.kelvin_per_unit};
        ComputeBufferHandle bindings[7] = {buffers->vel_x, buffers->vel_y, buffers->vel_z,
            owner->fields[0], buffers->temperature, buffers->gas_solid_mask,
            owner->phase_momentum};
        ComputeDispatch command;
        command.kernel = "sim_inventory_momentum";
        command.buffers = bindings;
        command.buffer_count = 7;
        command.constants = &pc;
        command.constants_size = sizeof(pc);
        command.groups = gasInventoryDispatch(records.size());
        if (!compute->dispatch(command)) {
            return false;
        }
        using Field = SimulationGridDomainComputeBuffers::GridField;
        buffers->markDeviceWrote(Field::VelX);
        buffers->markDeviceWrote(Field::VelY);
        buffers->markDeviceWrote(Field::VelZ);
    } else {
        auto& mutable_grid = gasGrid(gas);
        std::vector<float>* velocity[3] = {&mutable_grid.vel_x, &mutable_grid.vel_y,
                                         &mutable_grid.vel_z};
        for (const auto& record : records) {
            double gas_mass = 0, air_mass = 0;
            bool wall = false;
            for (const uint32_t cell : {record.left, record.right}) {
                if (cell == invalid) {
                    continue;
                }
                wall = wall || (grid.solid.size() == grid.getCellCount() && grid.solid[cell]);
                const double t = std::max(1.0, static_cast<double>(ambient_kelvin) +
                    (grid.temperature[cell] - gas_ambient_normalized) * scale.kelvin_per_unit);
                air_mass += 0.5 * volume * 101325.0 * 0.028965 / (8.314462 * t);
                if (gas.gas_phase_mass_kg.size() == grid.getCellCount()) {
                    gas_mass += 0.5 * gas.gas_phase_mass_kg[cell];
                }
            }
            if (wall) {
                continue;
            }
            const double before = air_mass + std::max(0.0, gas_mass - record.mass);
            float& v = velocity[record.axis]->at(record.face);
            v = static_cast<float>((before * v + record.momentum) / (before + record.mass));
        }
    }
    deposits.clear();
    return true;
}
}
