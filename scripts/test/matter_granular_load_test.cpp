// User-built regression target. Codex does not compile this source.
#include "Fluid/MatterGranularLoad.h"
#include "Fluid/SubstanceTag.h"

#include <cassert>
#include <cmath>

using namespace RayTrophiSim::Fluid;

int main() {
    FluidParticles particles;
    for (float y : {0.0f, 2.0f}) {
        particles.emit(Vec3(0.0f, y, 0.0f), Vec3(0.0f), 293.15f, 0.0f,
            substanceTag("Sand"), nullptr, nullptr, 0.2f, MatterConstitutiveModel::Granular);
    }
    particles.emit(Vec3(0.0f, 100.0f, 0.0f), Vec3(0.0f), 293.15f, 0.0f,
        substanceTag("Water"), nullptr, nullptr, 0.125f, MatterConstitutiveModel::Fluid);
    particles.affine[0].col0.x = 3.0f;
    particles.affine[0].col1.y = 4.0f;
    particles.affine[2].col0.x = 10000.0f;
    particles.granular_softening[1] = 0.5f;
    particles.granular_softening[2] = 0.0f;
    const auto identities = particles.particle_id;
    const auto load = measureMatterGranularLoad(particles, false);
    assert(load.column_height == 2.0f);
    assert(load.strain_rate == 5.0f);
    assert(load.softening_min == 0.5f);
    assert(load.softened_particles == 1);
    assert(std::abs(load.overburden_pressure - 1600.0f * 9.81f * 2.0f) < 0.01f);
    const auto elastic = Granular::elasticStepInfo(2000000.0f, 0.1f, 1.0f / 60.0f,
        load.strain_rate, load.overburden_pressure, load.softening_min);
    assert(elastic.overburden_pressure > 0.0f && elastic.young_modulus_for_load > 0.0f);
    assert(particles.particle_id == identities);
    particles.constitutive_model[1] = static_cast<uint8_t>(MatterConstitutiveModel::Auto);
    assert(measureMatterGranularLoad(particles, false).column_height == 0.0f);
    assert(measureMatterGranularLoad(particles, true).column_height == 2.0f);
    particles.constitutive_model[0] = static_cast<uint8_t>(MatterConstitutiveModel::Fluid);
    assert(measureMatterGranularLoad(particles, false).overburden_pressure == 0.0f);
}
