// Regression source for the user's test target; no Codex build.
#include "Fluid/MatterWetAppearance.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/SubstanceTag.h"

#include <cassert>

using namespace RayTrophiSim::Fluid;

int main() {
    FluidParticles particles;
    for (int i = 0; i < 3; ++i) {
        particles.emit(Vec3(static_cast<float>(i), 0.0f, 0.0f), Vec3(0.0f),
            293.15f, 0.0f, substanceTag("Sand"), nullptr, nullptr,
            0.2f, MatterConstitutiveModel::Granular);
        particles.pore_capacity_kg[i] = 1.0f;
    }
    particles.pore_water_mass_kg[0] = 0.5f;
    particles.pore_water_mass_kg[1] = 1.0f;
    particles.pore_water_mass_kg[2] = 0.0f; // Spatial dry control.
    MatterWetPalette palette;
    MatterWetPaletteEntry entry;
    entry.substance_tag = substanceTag("Sand");
    entry.dry_material = 10;
    entry.materials = {10, 11, 12, 13, 14, 15, 16, 17};
    palette.entries.push_back(entry);
    std::vector<int> keys{10};
    palette.appendMaterialKeys(keys);
    assert(keys.size() == 8);
    palette.appendMaterialKeys(keys);
    assert(keys.size() == 8);
    assert(palette.sourceIndex(particles, 0, keys, 0) == 4);
    assert(palette.sourceIndex(particles, 1, keys, 0) == 7);
    assert(palette.sourceIndex(particles, 2, keys, 0) == 0);
    particles.pore_water_mass_kg[2] = 0.047456805f;
    assert(palette.sourceIndex(particles, 2, keys, 0) == 1);
    palette.appearance_full_saturation = 0.05f;
    assert(palette.sourceIndex(particles, 2, keys, 0) == 7);
    palette.appearance_full_saturation = 1.0f;
    particles.pore_water_mass_kg[2] = 0.0f;
    particles.constitutive_model[0] = static_cast<uint8_t>(MatterConstitutiveModel::Fluid);
    assert(palette.sourceIndex(particles, 0, keys, 0) == 0);
    particles.constitutive_model[0] = static_cast<uint8_t>(MatterConstitutiveModel::Auto);
    assert(palette.sourceIndex(particles, 0, keys, 0) == 0);
    palette.legacy_granular = true;
    assert(palette.sourceIndex(particles, 0, keys, 0) == 4);
    assert(palette.sourceIndex(particles, 99, keys, 0) == 0);
    assert(palette.sourceIndex(particles, 1, {10}, 0) == 0);
}
