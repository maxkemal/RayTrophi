// Core regression: link the application physics/substance core and Vec3; no app launch.
#include "Fluid/DomainWallSettings.h"
#include <cassert>
#include <cmath>
#include <limits>

namespace F = RayTrophiSim::Fluid;

static void close(float actual, float expected) {
    assert(std::abs(actual - expected) < 1.0e-5f);
}

int main() {
    F::DomainWallContact wall{0.5f, 0.0f, true};
    const Vec3 gravity(0.0f, -10.0f, 0.0f);
    // Sustained support brakes BOTH tangential axes proportionally.
    auto velocity = F::resolveDomainWallVelocity(Vec3(3.0f, 0.0f, 4.0f),
        1, 1.0f, wall, gravity, 0.1f);
    close(velocity.x, 2.7f);
    close(velocity.y, 0.0f);
    close(velocity.z, 3.6f);
    // Static traction stops small slip without reversing it.
    velocity = F::resolveDomainWallVelocity(Vec3(0.1f, 0.0f, 0.0f),
        1, 1.0f, wall, gravity, 0.1f);
    close(velocity.length(), 0.0f);
    // A departing parcel is not glued or dragged back down.
    velocity = F::resolveDomainWallVelocity(Vec3(2.0f, 1.0f, 0.0f),
        1, 1.0f, wall, gravity, 0.1f);
    close(velocity.x, 2.0f);
    close(velocity.y, 1.0f);
    // Free slip changes only the normal component; restitution is independent.
    wall.friction = 0.0f;
    wall.restitution = 0.25f;
    velocity = F::resolveDomainWallVelocity(Vec3(3.0f, -4.0f, 5.0f),
        1, 1.0f, wall, Vec3(0.0f), 0.1f);
    close(velocity.x, 3.0f);
    close(velocity.y, 1.0f);
    close(velocity.z, 5.0f);
    // A supported body with restitution must not bounce on each gravity tick.
    wall.restitution = 0.4f;
    for (float dt : {1.0f / 24.0f, 1.0f / 60.0f, 1.0f / 480.0f}) {
        Vec3 resting(0.0f);
        for (int step = 0; step < 1000; ++step) {
            resting = F::resolveDomainWallVelocity(resting + gravity * dt,
                1, 1.0f, wall, gravity, dt);
            close(resting.y, 0.0f);
        }
        // An actual incoming impact still rebounds after support is removed.
        const auto impact = F::resolveDomainWallVelocity(
            Vec3(0.0f, -4.0f - 10.0f * dt, 0.0f),
            1, 1.0f, wall, gravity, dt);
        close(impact.y, 1.6f);
    }
    wall.restitution = 0.25f;
    // All six walls use an inward normal and identical rebound semantics.
    for (int axis = 0; axis < 3; ++axis) {
        for (float sign : {-1.0f, 1.0f}) {
            Vec3 input(0.0f);
            F::setWallComponent(input, axis, -4.0f * sign);
            velocity = F::resolveDomainWallVelocity(input, axis, sign,
                wall, Vec3(0.0f), 0.1f);
            close(F::wallComponent(velocity, axis), sign);
        }
    }
    // Constant support: traction integrates physical time, not frame count.
    wall.friction = 0.5f;
    wall.restitution = 0.0f;
    const auto whole = F::resolveDomainWallVelocity(Vec3(2.0f, 0.0f, 0.0f),
        1, 1.0f, wall, gravity, 0.1f);
    auto halves = F::resolveDomainWallVelocity(Vec3(2.0f, 0.0f, 0.0f),
        1, 1.0f, wall, gravity, 0.05f);
    halves = F::resolveDomainWallVelocity(halves, 1, 1.0f, wall, gravity, 0.05f);
    close(whole.x, halves.x);
    // Interior MAC faces stay free; floor-adjacent faces get support traction.
    const int size[] = {8, 8, 8};
    const int floor_face[] = {4, 0, 4};
    const int interior[] = {4, 4, 4};
    const auto snapshot = [](int axis, const int*) { return axis == 0 ? 2.0f : 0.0f; };
    close(F::resolveDomainWallFace(2.0f, 0, floor_face, size,
        wall, gravity, 0.1f, snapshot), 1.5f);
    close(F::resolveDomainWallFace(2.0f, 0, interior, size,
        wall, gravity, 0.1f, snapshot), 2.0f);
    wall.closed = false;
    close(F::resolveDomainWallFace(2.0f, 0, floor_face, size,
        wall, gravity, 0.1f, snapshot), 2.0f);
    // MPM grid normal input may be a large constitutive stress impulse,
    // unrelated to g*dt. Wall restitution must not rebound this support load.
    wall.closed = true;
    const int normal_face[] = {4, 0, 4};
    for (float restitution : {0.0f, 0.4f, 1.0f}) {
        wall.restitution = restitution;
        for (float speed : {0.01f, 1.0f, 100.0f}) {
            const auto stressed = [speed](int axis, const int*) {
                return axis == 1 ? -speed : 0.0f;
            };
            close(F::resolveDomainWallFace(-speed, 1, normal_face, size,
                wall, gravity, 0.01f, stressed), 0.0f);
            close(F::resolveDomainWallFace(speed, 1, normal_face, size,
                wall, gravity, 0.01f, stressed), speed);
        }
    }
    // The MPM particle escape guard must not add a second rebound either.
    F::APICSolverParams mpm;
    mpm.granular_enabled = true;
    mpm.wall_damping = 1.0f;
    Vec3 escaped(0.5f, -0.1f, 0.5f);
    Vec3 entering(0.0f, -10.0f, 0.0f);
    F::resolveDomainWallTail(escaped, entering, Vec3(0.0f), Vec3(1.0f), mpm, 0.01f);
    close(escaped.y, 0.0f);
    close(entering.y, 0.0f);
    // Analytic traction: 10000 Pa / (1000 kg/m2) for 0.1 s = 1 m/s.
    wall.friction = 0.0f;
    wall.restitution = 0.0f;
    wall.adhesion_pa = 10000.0f;
    wall.mass_per_area = 1000.0f;
    velocity = F::resolveDomainWallVelocity(Vec3(2.0f, 0.5f, 0.0f),
        1, 1.0f, wall, Vec3(0.0f), 0.1f);
    close(velocity.y, 0.0f);
    close(velocity.x, 1.5f); // Tension used half the cohesive budget.
    velocity = F::resolveDomainWallVelocity(Vec3(2.0f, 2.0f, 0.0f),
        1, 1.0f, wall, Vec3(0.0f), 0.1f);
    close(velocity.y, 1.0f); // Strong separation breaks free; no reverse impulse.
    close(velocity.x, 2.0f);
    const auto glued_whole = F::resolveDomainWallVelocity(Vec3(0.0f, 2.0f, 0.0f),
        1, 1.0f, wall, Vec3(0.0f), 0.1f);
    auto glued_halves = F::resolveDomainWallVelocity(Vec3(0.0f, 2.0f, 0.0f),
        1, 1.0f, wall, Vec3(0.0f), 0.05f);
    glued_halves = F::resolveDomainWallVelocity(glued_halves,
        1, 1.0f, wall, Vec3(0.0f), 0.05f);
    close(glued_whole.y, glued_halves.y);
    wall.closed = false;
    velocity = F::resolveDomainWallVelocity(Vec3(2.0f, 0.5f, 0.0f),
        1, 1.0f, wall, Vec3(0.0f), 0.1f);
    close(velocity.y, 0.5f);
    wall.closed = true;
    // Domain adhesion does not affect an interior parcel.
    Vec3 inside(0.5f), inside_velocity(0.0f, 2.0f, 0.0f);
    F::resolveDomainWallParticle(inside, inside_velocity, Vec3(0.0f), Vec3(1.0f),
        wall, Vec3(0.0f), 0.1f, 0.01f);
    close(inside_velocity.y, 2.0f);

    // Invalid partial patches fail atomically, including non-finite inputs.
    F::APICSolverParams params;
    std::string error;
    assert(F::patchDomainWallSettings(params,
        {{"friction", 0.8f}, {"restitution", 0.2f}}, error));
    assert(!F::patchDomainWallSettings(params,
        {{"friction", 1.0f}, {"restitution", -1.0f}}, error));
    close(params.wall_friction, 0.8f);
    close(params.wall_damping, 0.2f);
    assert(!F::patchDomainWallSettings(params, {{"unknown", 0.0}}, error));
    assert(!F::patchDomainWallSettings(params, {{"friction", true}}, error));
    assert(!F::patchDomainWallSettings(params,
        {{"friction", std::numeric_limits<double>::infinity()}}, error));
    assert(!F::patchDomainWallSettings(params, nullptr, error));
    assert(F::patchDomainWallSettings(params, {{"friction", 0.0f}}, error));
    close(params.wall_damping, 0.2f);
    assert(F::patchDomainWallSettings(params, {{"adhesion_pa", 1234.0f}}, error));
    assert(!F::patchDomainWallSettings(params,
        {{"friction", 0.2f}, {"adhesion_pa", -1.0f}}, error));
    close(params.wall_friction, 0.0f);
    close(params.wall_adhesion_pa, 1234.0f);
    assert(!F::patchDomainWallSettings(params, {{"adhesion_pa", true}}, error));
}
