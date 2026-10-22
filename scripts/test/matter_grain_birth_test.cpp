#include "Fluid/MatterGrainBirth.h"

#include "Fluid/APICFluidSolver.h"

#include <cassert>
#include <cmath>

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
    // DEM seed spacing follows physical radii, not voxel resolution or PPC.
    APICSolverParams params;
    params.grain.enabled = true;
    params.grain.radius_m = .05f;
    params.grain.radius_max_m = 0.0f;
    FluidParticles seeded;
    const auto seed = seedMatterGrainBox(seeded, grid, Vec3(-.3f), Vec3(.3f),
        params, 293.15f, 10000);
    assert(seed.emitted > 0 && !seed.truncated);
    for (std::size_t i = 0; i < seeded.size(); ++i) {
        assert(seeded.rest_mass_kg[i] > 0.0f);
        assert(seeded.constitutive_model[i] ==
            static_cast<uint8_t>(MatterConstitutiveModel::Granular));
        for (std::size_t j = 0; j < i; ++j) {
            const Vec3 d = seeded.position[i] - seeded.position[j];
            assert(d.x * d.x + d.y * d.y + d.z * d.z >= .1f * .1f);
        }
    }
    FluidSim::FluidGrid finer(40, 40, 40, .05f, Vec3(-1));
    FluidParticles fine_seed;
    seedMatterGrainBox(fine_seed, finer, Vec3(-.3f), Vec3(.3f), params, 293.15f, 10000);
    assert(fine_seed.size() == seeded.size());
    // A column must not preserve identical x/z centers in every layer.
    // Repeated seeds (including a different voxel size) remain reproducible.
    bool x_varies = false;
    bool z_varies = false;
    for (std::size_t i = 0; i < seeded.size(); ++i) {
        const auto& a = seeded.position[i];
        const auto& b = fine_seed.position[i];
        assert(a.x == b.x && a.y == b.y && a.z == b.z);
        assert(a.x >= -.25f && a.x <= .25f);
        assert(a.y >= -.25f && a.y <= .25f);
        assert(a.z >= -.25f && a.z <= .25f);
        if (i > 0 && i < 5) {
            x_varies |= std::abs(a.x - seeded.position[0].x) > 1e-6f;
            z_varies |= std::abs(a.z - seeded.position[0].z) > 1e-6f;
        }
    }
    assert(x_varies && z_varies);
    // Five centers per axis for this box: index 25 is directly above index 0
    // in the candidate lattice, but jitter must break that vertical alignment.
    assert(seeded.size() == 125);
    const Vec3 column = seeded.position[25] - seeded.position[0];
    assert(std::abs(column.x) > 1e-6f || std::abs(column.z) > 1e-6f);
    FluidParticles changed_seed;
    ++params.grain.size_seed;
    seedMatterGrainBox(changed_seed, grid, Vec3(-.3f), Vec3(.3f),
        params, 293.15f, 10000);
    assert(changed_seed.size() == seeded.size());
    assert(changed_seed.position[0].x != seeded.position[0].x);
    --params.grain.size_seed;
    const auto appended = seedMatterGrainBox(seeded, grid, Vec3(-.3f), Vec3(.3f),
        params, 293.15f, 10000);
    assert(appended.emitted == 0 && !appended.truncated);
    FluidParticles limited;
    const auto capped = seedMatterGrainBox(limited, grid, Vec3(-.3f), Vec3(.3f),
        params, 293.15f, 3);
    assert(capped.emitted == 3 && capped.truncated && limited.size() == 3);
    for (std::size_t i = 0; i < limited.size(); ++i) {
        assert(limited.position[i].x == seeded.position[i].x);
        assert(limited.position[i].y == seeded.position[i].y);
        assert(limited.position[i].z == seeded.position[i].z);
    }
    FluidParticles tiny;
    const auto empty = seedMatterGrainBox(tiny, grid, Vec3(0), Vec3(.01f),
        params, 293.15f, 100);
    assert(empty.emitted == 0 && !empty.truncated);
}
