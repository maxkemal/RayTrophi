#include "Fluid/MatterModelBatch.h"

#include <cassert>

namespace F = RayTrophiSim::Fluid;

int main() {
    FluidSim::FluidGrid grid(12, 12, 12, 0.1f, Vec3(0.0f));
    F::FluidParticles particles;
    F::seedBox(particles, grid, Vec3(0.3f), Vec3(0.7f), 8, 17, 5000, 293.0f);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles.constitutive_model[i] = static_cast<uint8_t>(i % 2
            ? F::MatterConstitutiveModel::Granular : F::MatterConstitutiveModel::Fluid);
        particles.rest_mass_kg[i] = 1.0f;
        particles.substance_tag[i] = static_cast<uint32_t>(i + 1);
    }
    const auto original = particles;
    const auto original_velocity = grid.vel_x;
    std::array<F::APICSolverParams, 2> params;
    for (auto& model : params) {
        model.inherit_atmosphere = false;
        model.air_drag = 0.0f;
        model.reseed_enabled = false;
        model.boundary = F::APICSolverParams::BoundaryMode::Closed;
    }
    F::MatterModelBatchResult result;
    result.removed_particles = 123;
    std::string error;
    bool contact_called = false;
    const F::MatterBatchContact fail_contact = [&](const F::FluidParticles& liquid,
        FluidSim::FluidGrid& liquid_grid, const F::FluidParticles& granular,
        FluidSim::FluidGrid&, std::string& message) {
        contact_called = true;
        assert(liquid.uvw_step == original.uvw_step);
        assert(granular.uvw_step == original.uvw_step);
        for (const auto* lane : {&liquid, &granular}) {
            for (std::size_t i = 0; i < lane->size(); ++i) {
                const auto source = lane->substance_tag[i] - 1;
                assert(lane->particle_id[i] == original.particle_id[source]);
                assert(lane->position[i].x == original.position[source].x);
                assert(lane->rest_mass_kg[i] == original.rest_mass_kg[source]);
            }
        }
        liquid_grid.vel_x[0] = 999.0f;
        message = "injected contact failure";
        return false;
    };
    assert(!F::stepMatterModelBatch(particles, grid, params,
        F::MatterConstitutiveModel::Fluid, 1.0f / 240.0f, nullptr, 0.0f,
        fail_contact, result, error));
    assert(contact_called);
    assert(error == "injected contact failure");
    assert(particles.particle_id == original.particle_id);
    assert(particles.substance_tag == original.substance_tag);
    assert(particles.next_particle_id == original.next_particle_id);
    assert(particles.uvw_step == original.uvw_step);
    assert(grid.vel_x == original_velocity);
    assert(result.removed_particles == 123);
    assert(!F::stepMatterModelBatch(particles, grid, params,
        F::MatterConstitutiveModel::Fluid, 1.0f / 240.0f, nullptr, 0.0f,
        {}, result, error));
}
