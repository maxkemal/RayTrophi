#include "Fluid/MatterGrainBirth.h"

#include <cassert>

int main() {
    using namespace RayTrophiSim::Fluid;
    FluidSim::FluidGrid grid(20, 20, 20, .1f, Vec3(-1, -1, -1));
    FluidParticles particles;
    particles.resizeAll(1);
    particles.position[0] = Vec3(0);
    MatterGrainBirthFilter filter(particles, grid, .05f);
    assert(!filter.accept(Vec3(0)));
    assert(!filter.accept(Vec3(.099f, 0, 0)));
    assert(filter.accept(Vec3(.101f, 0, 0)));
    assert(!filter.accept(Vec3(.151f, 0, 0)));
    assert(filter.accept(Vec3(-.101f, 0, 0)));
    assert(filter.accept(Vec3(-.94f, 0, 0)));
    assert(!filter.accept(Vec3(-.99f, 0, 0)));
    MatterGrainBirthFilter disabled(particles, grid, 0.0f);
    assert(disabled.accept(Vec3(0)));
}
