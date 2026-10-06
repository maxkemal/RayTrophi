#include "Fluid/MatterMacContact.h"
#include "Fluid/MatterMixedStep.h"

#include <cassert>
#include <cmath>

namespace F = RayTrophiSim::Fluid;

struct Measurement {
    double momentum = 0.0;
    double energy = 0.0;
};

Measurement measure(const F::FluidParticles& particles,
                    const FluidSim::FluidGrid& liquid, const FluidSim::FluidGrid& granular) {
    F::MatterTransferFrame frame;
    std::string error;
    assert(F::buildMatterTransfer(particles, liquid.origin, liquid.voxel_size,
        {liquid.nx, liquid.ny, liquid.nz}, F::MatterConstitutiveModel::Fluid,
        1.0, frame, error));
    Measurement result;
    for (const auto& [cell, entry] : frame.cells) {
        const auto a = liquid.velXIndex(cell[0], cell[1], cell[2]);
        const auto b = liquid.velXIndex(cell[0] + 1, cell[1], cell[2]);
        const FluidSim::FluidGrid* grids[] = {&liquid, &granular};
        for (int lane = 0; lane < 2; ++lane) {
            for (const auto face : {a, b}) {
                const double mass = 0.5 * entry.model[lane].mass_kg;
                const double velocity = grids[lane]->vel_x[face];
                result.momentum += mass * velocity;
                result.energy += 0.5 * mass * velocity * velocity;
            }
        }
    }
    return result;
}

int main() {
    FluidSim::FluidGrid liquid(12, 12, 12, 0.1f, Vec3(0.0f));
    auto granular = liquid;
    F::FluidParticles particles;
    particles.emit(Vec3(0.45f, 0.55f, 0.55f), Vec3(1.0f, 0.0f, 0.0f));
    particles.emit(Vec3(0.55f), Vec3(-1.0f, 0.0f, 0.0f));
    particles.constitutive_model[0] = static_cast<uint8_t>(F::MatterConstitutiveModel::Fluid);
    particles.constitutive_model[1] = static_cast<uint8_t>(F::MatterConstitutiveModel::Granular);
    particles.rest_mass_kg[0] = 1.0f;
    particles.rest_mass_kg[1] = 2.0f;
    for (auto& velocity : liquid.vel_x) {
        velocity = 1.0f;
    }
    for (auto& velocity : granular.vel_x) {
        velocity = -1.0f;
    }
    const auto before = measure(particles, liquid, granular);
    F::MatterContactResult result;
    std::string error;
    assert(F::applyMatterMacContact(particles, liquid, granular,
        F::MatterConstitutiveModel::Fluid, 0.0, result, error));
    const auto after = measure(particles, liquid, granular);
    assert(result.pairs > 0);
    assert(std::abs(after.momentum - before.momentum) < 1e-6);
    assert(after.energy <= before.energy + 1e-6);
    assert(result.kinetic_energy_loss >= 0.0);

    const auto saved_velocity = liquid.vel_x;
    granular.vel_x.pop_back();
    assert(!F::applyMatterMacContact(particles, liquid, granular,
        F::MatterConstitutiveModel::Fluid, 0.0, result, error));
    assert(liquid.vel_x == saved_velocity);

    F::APICSolverParams params;
    params.mixed_working_set_budget_bytes = 1;
    F::APICSolverStats stats;
    const auto identities = particles.particle_id;
    assert(!F::stepMixedMatter(particles, liquid, params, 1.0f / 24.0f,
        nullptr, 0.0f, stats, error));
    assert(particles.particle_id == identities);
    assert(liquid.vel_x == saved_velocity);
    assert(error == "mixed CPU working set exceeds domain resource budget");
}
