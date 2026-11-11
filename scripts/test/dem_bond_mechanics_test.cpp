// Link only DemBondMechanics.cpp. No renderer, device or application needed.
// User compiles/runs this probe; Codex must not invoke C++ builds.
#include "Fluid/DemBondMechanics.h"

#include <cassert>
#include <cmath>
#include <limits>

namespace B = RayTrophiSim::Fluid::DemBond;

static void close(double actual, double expected, double tolerance = 1.0e-9) {
    assert(std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)));
}

static B::Vector cross(const B::Vector& a, const B::Vector& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

int main() {
    B::Endpoint a{11, {0.0, 0.0, 0.0}};
    B::Endpoint b{29, {1.0, 0.0, 0.0}};
    B::Material material{1.0e6, 0.25, 0.1, 1.0e9, 1.0e9};
    B::State state;
    B::Response response;
    std::string error;
    assert(B::create(a, b, state, error));
    const B::State reference = state;
    assert(B::evaluate(a, b, material, state, response, error));
    close(response.elastic_energy_j, 0.0);

    // Translation + finite common rigid rotation cannot create elastic strain.
    constexpr double root_half = 0.7071067811865475244;
    a.position = {5.0, -2.0, 3.0};
    b.position = {5.0, -1.0, 3.0};
    a.orientation = b.orientation = {root_half, 0.0, 0.0, root_half};
    assert(B::evaluate(a, b, material, state, response, error));
    close(response.elastic_energy_j, 0.0);
    // q and -q encode the same material pose.
    for (double& component : b.orientation) {
        component = -component;
    }
    assert(B::evaluate(a, b, material, state, response, error));
    close(response.elastic_energy_j, 0.0);

    a = {11, {0.0, 0.0, 0.0}};
    b = {29, {1.01, 0.0, 0.0}};
    state = reference;
    assert(B::evaluate(a, b, material, state, response, error));
    const double area = 3.14159265358979323846 * 0.01;
    close(response.first_force_n[0], material.young_pa * area * 0.01);
    close(response.tensile_stress_pa, 10000.0);
    close(response.elastic_energy_j, 0.5 * material.young_pa * area * 0.0001);

    // Pure transverse displacement: pair forces + endpoint moments conserve
    // angular momentum about an arbitrary origin, not only linear momentum.
    b.position = {1.0, 0.01, 0.0};
    assert(B::evaluate(a, b, material, state, response, error));
    const auto orbital = cross(b.position, response.second_force_n);
    for (int k = 0; k < 3; ++k) {
        close(response.first_force_n[k] + response.second_force_n[k], 0.0);
        close(orbital[k] + response.first_torque_nm[k] + response.second_torque_nm[k], 0.0);
    }

    // Equal-and-opposite torsional couples, with the expected GJ/L stiffness.
    b.position = {1.0, 0.0, 0.0};
    constexpr double angle = 0.01;
    b.orientation = {std::cos(angle / 2.0), std::sin(angle / 2.0), 0.0, 0.0};
    assert(B::evaluate(a, b, material, state, response, error));
    const double shear_modulus = material.young_pa / (2.0 * (1.0 + material.poisson));
    const double polar = area * 0.01 / 2.0;
    close(response.first_torque_nm[0], shear_modulus * polar * angle);
    close(response.first_torque_nm[0] + response.second_torque_nm[0], 0.0);

    // Tension fracture is irreversible. Released energy is recorded once;
    // a closed/recompressed pair must not silently heal.
    b.orientation = {1.0, 0.0, 0.0, 0.0};
    b.position = {1.01, 0.0, 0.0};
    material.tensile_strength_pa = 5000.0;
    assert(B::evaluate(a, b, material, state, response, error));
    assert(response.broken_now && state.failure == B::Failure::Tension);
    const double released = response.released_energy_j;
    assert(released > 0.0);
    close(state.fracture_dissipation_j, released);
    close(response.elastic_energy_j, 0.0);
    close(response.first_force_n[0], 0.0);
    b.position = {0.99, 0.0, 0.0};
    assert(B::evaluate(a, b, material, state, response, error));
    assert(!response.broken_now);
    close(response.released_energy_j, 0.0);
    close(state.fracture_dissipation_j, released);

    // Compression alone does not trigger a tensile fracture.
    state = reference;
    assert(B::evaluate(a, b, material, state, response, error));
    assert(state.failure == B::Failure::None);
    assert(response.first_force_n[0] < 0.0);

    // Shear failure has a distinct classification.
    state = reference;
    material.tensile_strength_pa = 1.0e9;
    material.shear_strength_pa = 100.0;
    b.position = {1.0, 0.01, 0.0};
    assert(B::evaluate(a, b, material, state, response, error));
    assert(state.failure == B::Failure::Shear);

    // Reorder/identity mismatches and invalid inputs are transactional.
    const auto saved_state = state;
    const auto saved_response = response;
    assert(!B::evaluate(b, a, material, state, response, error));
    assert(state.failure == saved_state.failure);
    close(state.fracture_dissipation_j, saved_state.fracture_dissipation_j);
    close(response.released_energy_j, saved_response.released_energy_j);
    material.young_pa = std::numeric_limits<double>::quiet_NaN();
    assert(!B::evaluate(a, b, material, state, response, error));
    assert(!B::create(a, a, state, error));
    assert(state.first_id == saved_state.first_id);

    // Stability follows stiffness and graph degree; it never softens E.
    material = {1.0e6, 0.25, 0.1, 1.0e6, 1.0e6};
    const double dt = B::stableStep(material, 1.0, 1.0, 1.0, 0.01, 0.01, 4, error);
    assert(dt > 0.0 && error.empty());
    material.young_pa *= 4.0;
    close(B::stableStep(material, 1.0, 1.0, 1.0, 0.01, 0.01, 4, error), dt / 2.0);
    material.young_pa /= 4.0;
    close(B::stableStep(material, 1.0, 1.0, 1.0, 0.01, 0.01, 16, error), dt / 2.0);
    assert(B::stableStep(material, 1.0, 0.0, 1.0, 0.01, 0.01, 4, error) == 0.0);
}
