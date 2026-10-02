// Standalone core regression. User builds/runs this with
// source/src/Physics/Fluid/FluidParticleRedistribution.cpp,
// source/src/Math/Vec3.cpp and the source/include include directory.
// Assertions must remain enabled (no NDEBUG).
#include "Fluid/FluidParticleRedistribution.h"
#include "Fluid/APICFluidSolver.h"

#include <cassert>
#include <cstring>
#include <limits>

using namespace RayTrophiSim::Fluid;

template<class T>
void same(const std::vector<T>& a, const std::vector<T>& b) {
    assert(a.size() == b.size());
    assert(a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

static FluidParticles fixture(bool interior = true, uint32_t donor_tag = 7u) {
    FluidParticles p;
    p.emit(Vec3(1.5f, 1.5f, 1.5f), Vec3(1, 2, 3), 290, 0.2f, 7u);
    for (int i = 0; i < 8; ++i) {
        p.emit(Vec3(0.5f, 1.5f, 1.5f), Vec3(i, i + 1, i + 2),
               300.0f + i, 0.1f * i, donor_tag);
    }
    if (interior) {
        for (const auto& pos : {Vec3(2.5f, 1.5f, 1.5f), Vec3(1.5f, 0.5f, 1.5f),
                                Vec3(1.5f, 2.5f, 1.5f), Vec3(1.5f, 1.5f, 0.5f),
                                Vec3(1.5f, 1.5f, 2.5f)}) {
            p.emit(pos, Vec3(0, 0, 0), 310, 0, 7u);
        }
    }
    p.ensureGranularStateSize();
    for (std::size_t i = 0; i < p.size(); ++i) {
        p.mass_fraction[i] = 0.2f + 0.01f * i;
        p.rest_mass_kg[i] = 0.5f + 0.02f * static_cast<float>(i);
        p.affine[i].col0 = Vec3(0.1f * i, 0.2f, 0.3f);
        p.flags[i] = 0x100u;
        p.granular_damage[i] = 0.01f * i;
    }
    return p;
}

int main() {
    FluidSim::FluidGrid grid(3, 3, 3, 1.0f);
    APICSolverParams params;
    params.particles_per_cell = 4;
    params.reseed_min_per_cell = 3;
    params.reseed_max_per_cell = 5;
    auto p = fixture();
    const auto before = p;
    assert(redistributeFluidParticles(p, grid, params, 123) == 3);
    assert(p.size() == before.size());
    std::size_t changed = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (p.position[i].x != before.position[i].x ||
            p.position[i].y != before.position[i].y ||
            p.position[i].z != before.position[i].z) {
            ++changed;
            assert(p.position[i].x >= 1 && p.position[i].x < 2);
            assert(p.position[i].y >= 1 && p.position[i].y < 2);
            assert(p.position[i].z >= 1 && p.position[i].z < 2);
        }
    }
    assert(changed == 3);
    same(p.velocity, before.velocity);
    same(p.affine, before.affine);
    same(p.flags, before.flags);
    same(p.mass_fraction, before.mass_fraction);
    same(p.rest_mass_kg, before.rest_mass_kg);
    same(p.temperature, before.temperature);
    same(p.combustible_fraction, before.combustible_fraction);
    same(p.substance_tag, before.substance_tag);
    same(p.uvw, before.uvw);
    same(p.uvw_b, before.uvw_b);
    same(p.granular_damage, before.granular_damage);
    auto repeat = before;
    assert(redistributeFluidParticles(repeat, grid, params, 123) == 3);
    same(p.position, repeat.position);
    // Lowering the emitter budget must not destroy already existing liquid.
    repeat = before;
    params.max_particles = 1;
    assert(redistributeFluidParticles(repeat, grid, params, 123) == 3);
    same(p.position, repeat.position);

    // No destination: never delete the crowded pool. Different material:
    // never turn source parcels into the destination substance.
    for (auto untouched : {fixture(false), fixture(true, 8u)}) {
        const auto old = untouched;
        assert(redistributeFluidParticles(untouched, grid, params, 4) == 0);
        same(untouched.position, old.position);
        assert(untouched.size() == old.size());
    }
    p = fixture();
    for (std::size_t i = 1; i <= 8; ++i) {
        p.flags[i] |= kParticleFlagFrozen;
    }
    assert(redistributeFluidParticles(p, grid, params, 4) == 0);
    p = fixture();
    params.granular_enabled = true;
    assert(redistributeFluidParticles(p, grid, params, 4) == 0);
    params.granular_enabled = false;
    params.reseed_enabled = false;
    assert(redistributeFluidParticles(p, grid, params, 4) == 0);
    params.reseed_enabled = true;
    std::vector<uint32_t> solid_tags{7u};
    params.solid_substance_tags = &solid_tags;
    assert(redistributeFluidParticles(p, grid, params, 4) == 0);
    params.solid_substance_tags = nullptr;
    grid.solid[grid.cellIndex(0, 1, 1)] = 1;
    assert(redistributeFluidParticles(p, grid, params, 4) == 0);
    grid.solid[grid.cellIndex(0, 1, 1)] = 0;
    p.clear();
    p.emit(Vec3(std::numeric_limits<float>::quiet_NaN(), 0, 0), Vec3(0, 0, 0));
    assert(redistributeFluidParticles(p, grid, params, 4) == 0);
    assert(p.size() == 1);
}
