// User-built regression target. Codex does not compile this source.
#include "Fluid/MatterAcceptanceMetrics.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/SubstanceTag.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace RayTrophiSim::Fluid;

int main() {
    FluidParticles particles;
    const auto empty = inspectMatterAcceptanceMetrics(particles, false);
    assert(empty["granular"]["bounds_min"].is_null());
    assert(empty["granular"]["particles"] == 0);
    for (float x : {-1.0f, 1.0f}) {
        particles.emit(Vec3(x, 2.0f, 0.0f), Vec3(0.0f), 293.15f, 0.0f,
            substanceTag("Sand"), nullptr, nullptr, 1.0f, MatterConstitutiveModel::Granular);
    }
    particles.emit(Vec3(100.0f), Vec3(0.0f), 293.15f, 0.0f, substanceTag("Water"),
        nullptr, nullptr, 1.0f, MatterConstitutiveModel::Fluid);
    particles.pore_capacity_kg[0] = 1.0f;
    particles.pore_capacity_kg[1] = 1.0f;
    particles.pore_water_mass_kg[1] = 0.5f;
    const auto identities = particles.particle_id;
    const auto positions = particles.position;
    const auto metrics = inspectMatterAcceptanceMetrics(particles, false);
    assert(metrics["granular"]["particles"] == 2);
    assert(metrics["granular"]["dry_mass_kg"] == 2.0);
    assert(metrics["granular"]["pore_water_kg"] == 0.5);
    assert(metrics["granular"]["mean_saturation"] == 0.25);
    assert(metrics["granular"]["horizontal_rms_radius_m"] == 1.0);
    assert(metrics["granular"]["dry_center_of_mass"][1] == 2.0);
    assert(metrics["exactly_dry_particles"] == 1);
    assert(metrics["saturation_bands"][0]["particles"] == 1);
    assert(metrics["saturation_bands"][4]["particles"] == 1);
    assert(particles.particle_id == identities);
    assert(particles.position[0].x == positions[0].x);
    particles.pore_water_mass_kg[0] = 0.047456805f;
    const auto low_wet = inspectMatterAcceptanceMetrics(particles, false);
    assert(low_wet["exactly_dry_particles"] == 0);
    assert(low_wet["saturation_bands"][0]["particles"] == 0);
    assert(low_wet["saturation_bands"][1]["particles"] == 1);
    const auto sensitive_wet = inspectMatterAcceptanceMetrics(particles, false, 0.05f);
    assert(sensitive_wet["saturation_bands"][7]["particles"] == 2);
    assert(sensitive_wet["granular"] == low_wet["granular"]);
    particles.constitutive_model[0] = static_cast<uint8_t>(MatterConstitutiveModel::Auto);
    assert(inspectMatterAcceptanceMetrics(particles, false)["granular"]["particles"] == 1);
    assert(inspectMatterAcceptanceMetrics(particles, true)["granular"]["particles"] == 2);
    particles.position[1].x = std::numeric_limits<float>::quiet_NaN();
    bool rejected = false;
    try {
        inspectMatterAcceptanceMetrics(particles, true);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}
