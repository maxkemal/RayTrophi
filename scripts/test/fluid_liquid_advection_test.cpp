#include "Fluid/FluidLiquidAdvection.h"
#include "Fluid/FluidDensityProjection.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

using RayTrophiSim::Fluid::FluidParticles;
using RayTrophiSim::Fluid::LiquidAdvectionSupport;

int main() {
    FluidSim::FluidGrid grid(4, 4, 4, 1.0f, Vec3(0.0f));
    FluidParticles particles;
    particles.emit(Vec3(0.5f, 0.5f, 0.5f), Vec3(4.0f, 0.0f, 0.0f));
    LiquidAdvectionSupport support(particles, grid, true);
    const auto sample = [](const Vec3&) { return Vec3(0.0f, 2.0f, 0.0f); };
    const Vec3 own(4.0f, 0.0f, 0.0f);

    const Vec3 inside = support.midpoint(Vec3(0.5f, 0.5f, 0.5f), own, 0.1f, sample);
    assert(std::abs(inside.y - 0.7f) < 1.0e-6f);

    // Empty in-domain cells and off-grid positions retain the particle speed.
    const Vec3 empty = support.midpoint(Vec3(2.5f, 0.5f, 0.5f), own, 0.1f, sample);
    assert(std::abs(empty.x - 2.9f) < 1.0e-6f);
    const Vec3 outside = support.midpoint(Vec3(-0.5f, 0.5f, 0.5f), own, 0.1f, sample);
    assert(std::abs(outside.x + 0.1f) < 1.0e-6f);

    // The first sample is supported, but its midpoint has left the liquid.
    const Vec3 crossed = support.midpoint(Vec3(0.5f, 0.9f, 0.5f), own, 0.2f, sample);
    assert(std::abs(crossed.x - 1.3f) < 1.0e-6f);
    assert(std::abs(crossed.y - 0.9f) < 1.0e-6f);

    particles.flags[0] |= RayTrophiSim::Fluid::kParticleFlagFrozen;
    LiquidAdvectionSupport frozen(particles, grid, true);
    const Vec3 held = frozen.midpoint(particles.position[0], own, 0.1f, sample);
    assert(std::abs(held.x - 0.9f) < 1.0e-6f);

    // Density correction changes positions, never physical velocity/affine.
    particles.flags[0] &= ~RayTrophiSim::Fluid::kParticleFlagFrozen;
    RayTrophiSim::Fluid::APICSolverParams params;
    params.density_relaxation_s = 0.05f;
    params.boundary = RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Closed;
    std::fill(grid.vel_x.begin(), grid.vel_x.end(), 7.0f);
    const auto before_grid = grid.vel_x;
    const auto before_velocity = particles.velocity;
    particles.affine[0].col0 = Vec3(1.0f, 2.0f, 3.0f);
    particles.affine[0].col1 = Vec3(4.0f, 5.0f, 6.0f);
    particles.affine[0].col2 = Vec3(7.0f, 8.0f, 9.0f);
    const auto before_affine = particles.affine;
    const auto before_position = particles.position;
    const bool shifted = RayTrophiSim::Fluid::projectLiquidDensityPositions(
        particles, grid, params, 0.1f,
        [&](const auto&, const auto&, auto&) {
            std::fill(grid.vel_x.begin(), grid.vel_x.end(), 1.0f);
            return true;
        });
    assert(shifted);
    assert(std::abs(particles.position[0].x - 0.6f) < 1.0e-6f);
    assert(particles.velocity[0].x == before_velocity[0].x);
    assert((particles.affine[0].col0 - before_affine[0].col0).length() == 0.0f);
    assert((particles.affine[0].col1 - before_affine[0].col1).length() == 0.0f);
    assert((particles.affine[0].col2 - before_affine[0].col2).length() == 0.0f);
    assert(grid.vel_x == before_grid);

    // A throwing backend must restore the physical grid and leave positions.
    bool threw = false;
    const auto before_throw = particles.position[0];
    try {
        RayTrophiSim::Fluid::projectLiquidDensityPositions(
            particles, grid, params, 0.1f,
            [](const auto&, const auto&, auto&) -> bool {
                throw std::runtime_error("backend failure");
            });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    assert((particles.position[0] - before_throw).length() == 0.0f);
    assert(grid.vel_x == before_grid);

    // Frozen-only domains require neither a scratch solve nor a GPU restore.
    particles.flags[0] |= RayTrophiSim::Fluid::kParticleFlagFrozen;
    bool solved = false;
    const auto frozen_position = particles.position[0];
    assert(RayTrophiSim::Fluid::projectLiquidDensityPositions(
        particles, grid, params, 0.1f,
        [&](const auto&, const auto&, auto&) {
            solved = true;
            return false;
        }));
    assert(!solved);
    assert((particles.position[0] - frozen_position).length() == 0.0f);
    assert(grid.vel_x == before_grid);
    particles.flags[0] &= ~RayTrophiSim::Fluid::kParticleFlagFrozen;

    particles.position = before_position;
    const bool failed = RayTrophiSim::Fluid::projectLiquidDensityPositions(
        particles, grid, params, 0.1f,
        [](const auto&, const auto&, auto&) { return false; });
    assert(!failed);
    assert(particles.position[0].x == before_position[0].x);
    assert(grid.vel_x == before_grid);
}
