#include "Fluid/MatterTransfer.h"

#include <cassert>
#include <cmath>
#include <limits>

namespace F = RayTrophiSim::Fluid;

bool near(double a, double b, double tolerance = 1e-10) {
    return std::abs(a - b) <= tolerance;
}

void emit(F::FluidParticles& particles, const Vec3& position, const Vec3& velocity,
          float mass, F::MatterConstitutiveModel model) {
    particles.emit(position, velocity, 293.0f, 0.0f, 0u,
                   nullptr, nullptr, mass, model);
}

void identities() {
    F::FluidParticles particles;
    for (int i = 0; i < 4; ++i) {
        emit(particles, Vec3(static_cast<float>(i)), Vec3(0.0f), 1.0f,
             F::MatterConstitutiveModel::Fluid);
    }
    const auto born = particles.particle_id;
    particles.removeSwap(1);
    assert(particles.particle_id[0] == born[0]);
    assert(particles.particle_id[1] == born[3]);
    particles.compact({1, 0, 0});
    assert(particles.particle_id[0] == born[3]);
    assert(particles.particle_id[1] == born[2]);
    const auto snapshot = particles;
    emit(particles, Vec3(0.0f), Vec3(0.0f), 1.0f, F::MatterConstitutiveModel::Fluid);
    assert(particles.particle_id.back() > born.back());
    particles = snapshot;
    assert(particles.particle_id == snapshot.particle_id);
    assert(particles.next_particle_id == snapshot.next_particle_id);
    const uint64_t allocator = particles.next_particle_id;
    particles.clear();
    emit(particles, Vec3(0.0f), Vec3(0.0f), 1.0f, F::MatterConstitutiveModel::Fluid);
    assert(particles.particle_id[0] == allocator);
    uint64_t exhausted = std::numeric_limits<uint64_t>::max();
    bool threw = false;
    try {
        F::allocateMatterParticleId(exhausted);
    } catch (const std::overflow_error&) {
        threw = true;
    }
    assert(threw);
}

void transfer() {
    F::FluidParticles particles;
    emit(particles, Vec3(0.0f), Vec3(2.0f, -1.0f, 0.0f), 2.0f,
         F::MatterConstitutiveModel::Fluid);
    emit(particles, Vec3(0.1f), Vec3(-1.0f, 3.0f, 0.0f), 3.0f,
         F::MatterConstitutiveModel::Granular);
    emit(particles, Vec3(1.0f), Vec3(1.0f), 7.0f,
         F::MatterConstitutiveModel::Elastic);
    emit(particles, Vec3(4.0f, 1.0f, 1.0f), Vec3(0.0f), 11.0f,
         F::MatterConstitutiveModel::Fluid);
    F::MatterTransferFrame frame;
    std::string error;
    assert(F::buildMatterTransfer(particles, Vec3(0.0f), 1.0f, {4, 4, 4},
        F::MatterConstitutiveModel::Fluid, 1.0, frame, error));
    assert(frame.totals[0].particles == 2 && frame.totals[1].particles == 1);
    assert(frame.totals[2].particles == 1);
    assert(frame.outside_particles == 1 && near(frame.outside_mass_kg, 11.0));
    assert(near(frame.deposited_mass_kg, 5.0));
    assert(near(frame.deposited_momentum.x, 1.0));
    assert(near(frame.deposited_momentum.y, 7.0));
    assert(frame.overlapping_cells > 0);
    double mass[2] = {};
    double momentum_x = 0.0;
    double gradient_x = 0.0;
    for (const auto& entry : frame.cells) {
        for (int lane = 0; lane < 2; ++lane) {
            const auto& field = entry.second.model[lane];
            mass[lane] += field.mass_kg;
            momentum_x += field.momentum.x;
            gradient_x += field.mass_gradient.x;
        }
    }
    assert(near(mass[0], 2.0) && near(mass[1], 3.0));
    assert(near(momentum_x, 1.0) && near(gradient_x, 0.0));
    const auto old_cells = frame.cells.size();
    particles.particle_id[1] = particles.particle_id[0];
    assert(!F::buildMatterTransfer(particles, Vec3(0.0f), 1.0f, {4, 4, 4},
        F::MatterConstitutiveModel::Fluid, 1.0, frame, error));
    assert(frame.cells.size() == old_cells && near(frame.deposited_mass_kg, 5.0));
}

void contact() {
    F::MatterTransferFrame frame;
    auto& cell = frame.cells[{0, 0, 0}];
    cell.model[0].mass_kg = 2.0;
    cell.model[0].momentum = {2.0, 2.0, 0.0};
    cell.model[0].mass_gradient = {-2.0, 0.0, 0.0};
    cell.model[1].mass_kg = 3.0;
    cell.model[1].mass_gradient = {3.0, 0.0, 0.0};
    F::MatterContactResult result;
    std::string error;
    assert(F::applyMatterGridContact(frame, 0.5, result, error));
    assert(result.pairs == 1);
    assert(near(cell.model[0].momentum.x / 2.0, 0.4));
    assert(near(cell.model[1].momentum.x / 3.0, 0.4));
    assert(near(cell.model[1].momentum.y, 0.6));
    assert(near(result.momentum_error.x, 0.0));
    assert(near(result.momentum_error.y, 0.0));
    assert(result.kinetic_energy_loss > 0.0);
    const auto before = cell.model[0].momentum;
    assert(!F::applyMatterGridContact(frame, -1.0, result, error));
    assert(near(cell.model[0].momentum.x, before.x));
    cell.model[0].momentum = {-2.0, 0.0, 0.0};
    cell.model[1].momentum = {};
    assert(F::applyMatterGridContact(frame, 0.5, result, error));
    assert(result.pairs == 0); // Separating motion must not receive attraction.
}

int main() {
    identities();
    transfer();
    contact();
}
