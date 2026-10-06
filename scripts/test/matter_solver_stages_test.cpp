#include "Fluid/MatterSolverStages.h"

#include <cassert>
#include <cmath>

namespace F = RayTrophiSim::Fluid;

void sameState(const F::FluidParticles& a, const F::FluidParticles& b) {
    assert(a.size() == b.size());
    assert(a.particle_id == b.particle_id);
    assert(a.uvw_step == b.uvw_step);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const Vec3 delta = a.position[i] - b.position[i];
        const Vec3 velocity_delta = a.velocity[i] - b.velocity[i];
        assert(std::abs(delta.x) < 2e-6f && std::abs(delta.y) < 2e-6f &&
               std::abs(delta.z) < 2e-6f);
        assert(std::abs(velocity_delta.x) < 2e-5f &&
               std::abs(velocity_delta.y) < 2e-5f &&
               std::abs(velocity_delta.z) < 2e-5f);
    }
}

int main() {
    FluidSim::FluidGrid initial_grid(12, 12, 12, 0.1f, Vec3(0.0f));
    F::FluidParticles initial;
    F::seedBox(initial, initial_grid, Vec3(0.3f), Vec3(0.7f), 8, 17, 5000, 293.0f);
    F::APICSolverParams params;
    params.inherit_atmosphere = false;
    params.air_drag = 0.0f;
    params.reseed_enabled = false;
    params.boundary = F::APICSolverParams::BoundaryMode::Closed;
    constexpr float dt = 1.0f / 240.0f;
    auto direct = initial;
    auto direct_grid = initial_grid;
    F::step(direct, direct_grid, params, dt, nullptr, 0.0f, nullptr);

    auto split = initial;
    auto split_grid = initial_grid;
    F::MatterModelGridStep workspace;
    std::string error;
    assert(F::prepareMatterModelGrid(split, split_grid, params, dt, nullptr,
                                    0.0f, workspace, error));
    assert(split.uvw_step == initial.uvw_step);
    assert(split.particle_id == initial.particle_id);
    for (std::size_t i = 0; i < split.size(); ++i) {
        assert(split.position[i].x == initial.position[i].x);
        assert(split.position[i].y == initial.position[i].y);
        assert(split.position[i].z == initial.position[i].z);
    }
    // Preparing a second liquid overwrites legacy global FLIP scratch.
    // The first model must still finish from its OWN pre-projection snapshot.
    auto other = initial;
    for (auto& velocity : other.velocity) {
        velocity = Vec3(0.2f, 0.0f, 0.0f);
    }
    auto other_grid = initial_grid;
    F::MatterModelGridStep other_workspace;
    assert(F::prepareMatterModelGrid(other, other_grid, params, dt, nullptr,
                                    0.0f, other_workspace, error));
    F::APICSolverStats stats;
    assert(F::finishMatterModelGrid(workspace, stats, error));
    sameState(direct, split);
    assert(!stats.p2g_on_gpu);
    const auto consumed = split;
    assert(!F::finishMatterModelGrid(workspace, stats, error));
    sameState(consumed, split);
    assert(F::finishMatterModelGrid(other_workspace, stats, error));

    // A topology change between stages must be rejected before gather/tail.
    F::MatterModelGridStep changed;
    assert(F::prepareMatterModelGrid(other, other_grid, params, dt, nullptr,
                                    dt, changed, error));
    other.removeSwap(0);
    const uint32_t age = other.uvw_step;
    assert(!F::finishMatterModelGrid(changed, stats, error));
    assert(other.uvw_step == age);
}
