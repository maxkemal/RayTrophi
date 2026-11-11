// User-built regression; no application launch required.
#include "Fluid/FluidSeedBox.h"
#include <cassert>

using namespace RayTrophiSim;

bool equal(const Vec3& a, const Vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

int main() {
    SimulationGridDomainDesc fresh;
    const Vec3 lo(10.0f, 20.0f, 30.0f), hi(20.0f, 30.0f, 40.0f);
    Fluid::initializeFluidSeedBox(fresh, lo, hi);
    assert(fresh.fluid_seed_box_initialized);
    assert(fresh.fluid_seed_min.x >= lo.x && fresh.fluid_seed_max.x <= hi.x);
    const Vec3 initial_min = fresh.fluid_seed_min;
    const Vec3 initial_max = fresh.fluid_seed_max;
    Fluid::initializeFluidSeedBox(fresh, lo + Vec3(100.0f), hi + Vec3(100.0f));
    assert(equal(fresh.fluid_seed_min, initial_min));
    assert(equal(fresh.fluid_seed_max, initial_max));

    SimulationGridDomainDesc restored;
    restored.fluid_seed_min = Vec3(12.0f, 21.0f, 32.0f);
    restored.fluid_seed_max = Vec3(13.0f, 22.0f, 33.0f);
    restored.fluid_seed_box_initialized = true;
    const Vec3 saved_min = restored.fluid_seed_min;
    const Vec3 saved_max = restored.fluid_seed_max;
    Fluid::initializeFluidSeedBox(restored, lo, hi);
    assert(equal(restored.fluid_seed_min, saved_min));
    assert(equal(restored.fluid_seed_max, saved_max));
    // Out-of-domain authored coordinates must not be silently replaced either.
    Fluid::initializeFluidSeedBox(restored, Vec3(-5.0f), Vec3(5.0f));
    assert(equal(restored.fluid_seed_min, saved_min));
}
