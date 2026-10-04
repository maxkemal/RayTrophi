// Source test for the user's build environment; Codex does not compile projects.
#include "Fluid/MatterPhaseGrid.h"

#include <cassert>
#include <limits>
#include <stdexcept>

using namespace RayTrophiSim;
namespace Phase = RayTrophiSim::Fluid;

static void prepare(SimulationGridDomainState& state) {
    state.type = SimulationDomainType::Matter;
    state.grid.nx = 8;
    state.grid.ny = 12;
    state.grid.nz = 16;
    state.grid.origin = Vec3(-4.0f, 0.0f, -8.0f);
    state.grid.voxel_size = 1.0f;
    state.grid.density = {7.0f};
    state.matter_liquid_grid.nx = 20;
    state.matter_liquid_grid.ny = 10;
    state.matter_liquid_grid.nz = 6;
    state.matter_liquid_grid.origin = Vec3(-1.0f, 1.0f, -1.0f);
    state.matter_liquid_grid.voxel_size = 0.1f;
    state.matter_liquid_grid.pressure = {3.0f};
    state.bounds_min = state.grid.origin;
    state.bounds_max = Phase::gridBoundsMax(state.grid);
    state.resolution_x = 8;
    state.resolution_y = 12;
    state.resolution_z = 16;
    state.voxel_size = 1.0f;
}

int main() {
    SimulationGridDomainState state;
    prepare(state);
    SimulationGridDomainComputeBuffers gas_buffers;
    SimulationGridDomainComputeBuffers liquid_buffers;
    gas_buffers.fluid_mask_device_valid = false;
    liquid_buffers.fluid_mask_device_valid = true;
    const auto& read_only = state;
    assert(Phase::liquidGrid(read_only).nx == 20);
    assert(Phase::gasGrid(read_only).nx == 8);
    assert(Phase::gridsOverlap(Phase::liquidGrid(state), Phase::gasGrid(state)));
    assert(Phase::gridContains(Phase::gasGrid(state), Vec3(0.0f, 0.0f, 0.0f)));
    assert(!Phase::gridContains(Phase::gasGrid(state), Vec3(4.0f, 1.0f, 1.0f)));
    assert(!Phase::gridContains(Phase::gasGrid(state),
                               Vec3(std::numeric_limits<float>::quiet_NaN())));

    try {
        Phase::MatterLiquidScope scope(state, &gas_buffers, &liquid_buffers);
        assert(state.matter_liquid_active);
        assert(state.resolution_x == 20 && state.voxel_size == 0.1f);
        assert(state.bounds_min.x == -1.0f);
        assert(Phase::gasGrid(state).density[0] == 7.0f);
        assert(Phase::liquidGrid(state).pressure[0] == 3.0f);
        assert(gas_buffers.fluid_mask_device_valid);
        {
            Phase::MatterLiquidScope nested(state, &gas_buffers, &liquid_buffers);
        }
        assert(state.matter_liquid_active);
        Phase::liquidGrid(state).pressure[0] = 9.0f;
        throw std::runtime_error("exercise unwind");
    } catch (const std::runtime_error&) {
    }
    assert(!state.matter_liquid_active);
    assert(state.resolution_x == 8 && state.voxel_size == 1.0f);
    assert(state.bounds_min.x == -4.0f);
    assert(Phase::liquidGrid(state).pressure[0] == 9.0f);
    assert(!gas_buffers.fluid_mask_device_valid);
    assert(liquid_buffers.fluid_mask_device_valid);

    {
        Phase::MatterLiquidScope scope(state, nullptr, nullptr);
        scope.restore();
        scope.restore();
        assert(!state.matter_liquid_active && state.grid.nx == 8);
    }
    Phase::translatePhaseGrids(state, Vec3(10.0f, 2.0f, 3.0f));
    assert(Phase::gasGrid(state).origin.x == 6.0f);
    assert(Phase::liquidGrid(state).origin.x == 9.0f);
    state.particles.position = {Vec3(1.0f, 2.0f, 3.0f)};
    state.particles.uvw = state.particles.position;
    state.particles.uvw_b = state.particles.position;
    state.foam.position = state.particles.position;
    const uint64_t old_version = state.version;
    Phase::translateLiquidParticles(state, Vec3(10.0f, 0.0f, 0.0f));
    assert(state.particles.position[0].x == 11.0f);
    assert(state.particles.uvw[0].x == 1.0f);
    assert(state.particles.uvw_b[0].x == 1.0f);
    assert(state.foam.position[0].x == 1.0f);
    assert(state.version == old_version);
    assert(!Phase::gridContains(Phase::liquidGrid(state), Vec3(0.0f)));
    state.matter_liquid_grid.origin = Vec3(100.0f);
    assert(!Phase::gridsOverlap(Phase::liquidGrid(state), Phase::gasGrid(state)));

    for (auto type : {SimulationDomainType::Fluid, SimulationDomainType::Gas}) {
        state.type = type;
        Phase::MatterLiquidScope scope(state, nullptr, nullptr);
        assert(!state.matter_liquid_active);
        assert(&Phase::liquidGrid(state) == &state.grid);
        assert(&Phase::gasGrid(state) == &state.grid);
    }
}
