// Standalone source test. User compiles with MatterLiquidInflow.cpp; no Studio launch.
#include "Fluid/MatterLiquidInflow.h"
#include "Fluid/LiquidEmissionAuthoring.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace RayTrophiSim;
using namespace RayTrophiSim::Fluid;

static void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    LiquidLattice lattice;
    lattice.h = 0.02f;
    lattice.nx = lattice.ny = lattice.nz = 64;
    LiquidFlowRegion nozzle;
    nozzle.center = Vec3(0.4f, 0.6f, 0.6f);
    nozzle.axis = Vec3(1.0f, 0.0f, 0.0f);
    nozzle.radius = 0.1f;
    nozzle.sweep = lattice.h;
    constexpr int ppc = 8;
    constexpr double dt = 0.02;
    const double parcel_volume = std::pow(double(lattice.h), 3) / ppc;
    const double target = 3.141592653589793 * 0.1 * 0.1;
    FluidParticles particles;
    double emitted_volume = 0.0;
    for (int step = 0; step < 20; ++step) {
        LiquidInflowPlan plan;
        planLiquidInflow(nozzle, lattice, ppc, particles, step, plan);
        require(!plan.births.empty(), "interrupted free inflow");
        for (const Vec3& position : plan.births) {
            require(insideLiquidFlowRegion(nozzle, position), "birth outside nozzle");
            particles.emit(position, Vec3(1.0f, 0.0f, 0.0f), 293.0f, 0.0f,
                           0u, nullptr, nullptr, float(parcel_volume * 1000));
        }
        emitted_volume += plan.births.size() * parcel_volume;
        for (Vec3& position : particles.position) {
            position.x += float(dt);
        }
    }
    require(std::abs(emitted_volume / (20 * dt) / target - 1) <= 0.05,
            "Q differs from disk area times speed by more than 5 percent");
    // Every axial slice of the emitted thread must contain material.
    for (int slice = 0; slice < 20; ++slice) {
        bool occupied = false;
        const float lo = nozzle.center.x + slice * lattice.h;
        for (const Vec3& position : particles.position) {
            occupied = occupied || (position.x >= lo && position.x < lo + lattice.h);
        }
        require(occupied, "empty axial slice in continuous thread");
    }
    LiquidInflowPlan blocked;
    planLiquidInflow(nozzle, lattice, ppc, particles, 30, blocked);
    for (const Vec3& position : blocked.births) {
        particles.emit(position, Vec3(0.0f), 293.0f, 0.0f,
                       0u, nullptr, nullptr, float(parcel_volume * 1000));
    }
    LiquidInflowPlan same_step;
    planLiquidInflow(nozzle, lattice, ppc, particles, 30, same_step);
    require(same_step.births.empty(), "full nozzle emits twice without transport");

    // Analytic source-volume oracle with deliberately excessive fill demand.
    // This covers low-CFL/noisy occupancy without relying on solver dynamics.
    bool initialized = false;
    double remainder = 0.0;
    LiquidInflowPlan limited;
    limited.births.assign(100, Vec3(0.0f));
    limitLiquidInflowBirths(limited, 2.25, initialized, remainder, 1);
    require(limited.births.size() == 100, "initial reservoir fill was throttled");
    std::size_t admitted = 0;
    for (uint64_t step = 0; step < 80; ++step) {
        limited.births.assign(100, Vec3(0.0f));
        limitLiquidInflowBirths(limited, 2.25, initialized, remainder, step);
        admitted += limited.births.size();
    }
    require(admitted == 180, "fractional source volume is not conserved");
    for (uint64_t step = 0; step < 80; ++step) {
        limited.births.clear();
        limitLiquidInflowBirths(limited, 2.25, initialized, remainder, step);
    }
    limited.births.assign(100, Vec3(0.0f));
    limitLiquidInflowBirths(limited, 2.25, initialized, remainder, 100);
    require(limited.births.size() <= 3, "blocked nozzle released a backlog");
    limited.births.assign(100, Vec3(0.0f));
    limitLiquidInflowBirths(limited, 0.0, initialized, remainder, 101);
    require(limited.births.empty(), "zero timestep emitted material");

    LiquidFlowRegion drain;
    drain.shape = LiquidFlowRegion::Shape::Sphere;
    drain.center = Vec3(0.6f);
    drain.radius = 2.0f;
    double before_mass = 0.0;
    for (float mass : particles.rest_mass_kg) {
        before_mass += mass;
    }
    std::vector<uint8_t> mask;
    const auto count = markLiquidOutflow(drain, particles, mask);
    require(count == particles.size(), "drain missed material");
    double removed_mass = 0.0;
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            removed_mass += particles.rest_mass_kg[i] * particles.mass_fraction[i] +
                            particles.pore_water_mass_kg[i];
        }
    }
    particles.compact(mask);
    require(particles.size() == 0 && std::abs(removed_mass - before_mass) < 1e-8,
            "outflow mass balance failed");
    SimulationFlowSourceDesc source;
    source.phase = SimulationFlowSourceDesc::Phase::Liquid;
    std::string error;
    require(setLiquidEmission(source, "inflow", error), "valid mode refused");
    require(!setLiquidEmission(source, "typo", error), "invalid mode accepted");
    require(source.liquid_emission == SimulationFlowSourceDesc::LiquidEmission::Inflow,
            "invalid mode changed state");
    source.source_mode = SimulationFlowSourceMode::MeshSurface;
    require(!setLiquidEmission(source, "outflow", error), "mesh drain accepted");
    std::cout << "PASS nozzle flux, thread continuity, full-region hold, drainage mass, validation\n";
}
