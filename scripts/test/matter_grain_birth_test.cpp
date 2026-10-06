#include "Fluid/MatterGrainBirth.h"

#include <cassert>

int main() {
    using namespace RayTrophiSim::Fluid;
    FluidSim::FluidGrid grid(20, 20, 20, .1f, Vec3(-1, -1, -1));
    FluidParticles particles;
    particles.emit(Vec3(0), Vec3(0), 0.0f, 0.0f, 0u, nullptr, nullptr, 0.1f,
        MatterConstitutiveModel::Granular);
    // A liquid parcel is not a rigid neighbour: it never blocks a grain birth.
    particles.emit(Vec3(.5f, .5f, .5f), Vec3(0), 0.0f, 0.0f, 0u, nullptr, nullptr, 0.1f,
        MatterConstitutiveModel::Fluid);
    MatterGrainBirthFilter filter(particles, grid, .05f);
    assert(!filter.accept(Vec3(0)));
    assert(!filter.accept(Vec3(.099f, 0, 0)));
    assert(filter.accept(Vec3(.101f, 0, 0)));
    assert(!filter.accept(Vec3(.151f, 0, 0)));
    assert(filter.accept(Vec3(-.101f, 0, 0)));
    assert(filter.accept(Vec3(-.94f, 0, 0)));
    assert(filter.accept(Vec3(.5f, .5f, .5f)));
    assert(!filter.accept(Vec3(-.99f, 0, 0)));
    MatterGrainBirthFilter disabled(particles, grid, 0.0f);
    assert(disabled.accept(Vec3(0)));
}
