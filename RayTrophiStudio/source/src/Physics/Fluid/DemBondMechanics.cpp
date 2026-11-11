#include "Fluid/DemBondMechanics.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace RayTrophiSim::Fluid::DemBond {
namespace {
constexpr double pi = 3.14159265358979323846;

Vector add(const Vector& a, const Vector& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vector scale(const Vector& a, double s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}

Vector subtract(const Vector& a, const Vector& b) {
    return add(a, scale(b, -1.0));
}

double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vector cross(const Vector& a, const Vector& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

double length(const Vector& v) {
    return std::hypot(v[0], v[1], v[2]);
}

template <std::size_t N>
bool finite(const std::array<double, N>& values) {
    for (double v : values) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

Quaternion conjugate(const Quaternion& q) {
    return {q[0], -q[1], -q[2], -q[3]};
}

Quaternion multiply(const Quaternion& a, const Quaternion& b) {
    return {
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}

bool normalized(const Quaternion& input, Quaternion& output) {
    if (!finite(input)) {
        return false;
    }
    const double n = std::hypot(std::hypot(input[0], input[1]),
                                std::hypot(input[2], input[3]));
    if (!(n > 1.0e-12) || !std::isfinite(n)) {
        return false;
    }
    for (std::size_t i = 0; i < 4; ++i) {
        output[i] = input[i] / n;
    }
    return true;
}

Vector rotate(const Quaternion& q, const Vector& v) {
    const Vector axis{q[1], q[2], q[3]};
    const Vector twice_cross = scale(cross(axis, v), 2.0);
    return add(v, add(scale(twice_cross, q[0]), cross(axis, twice_cross)));
}

bool endpoint(const Endpoint& point, Quaternion& q) {
    return point.id != 0 && finite(point.position) && normalized(point.orientation, q);
}

struct Section {
    double area;
    double moment;
    double polar;
    double normal_k;
    double shear_k;
    double bend_k;
    double twist_k;
};

bool section(const Material& m, double l, Section& s, std::string& error) {
    if (!std::isfinite(m.young_pa) || !(m.young_pa > 0.0) ||
        !std::isfinite(m.poisson) || m.poisson <= -1.0 || m.poisson >= 0.5 ||
        !std::isfinite(m.radius_m) || !(m.radius_m > 0.0) ||
        !std::isfinite(m.tensile_strength_pa) || m.tensile_strength_pa < 0.0 ||
        !std::isfinite(m.shear_strength_pa) || m.shear_strength_pa < 0.0 ||
        !std::isfinite(l) || !(l > 1.0e-12)) {
        error = "DEM bond: invalid material or reference length";
        return false;
    }
    s.area = pi * m.radius_m * m.radius_m;
    s.moment = s.area * m.radius_m * m.radius_m / 4.0;
    s.polar = 2.0 * s.moment;
    const double shear_modulus = m.young_pa / (2.0 * (1.0 + m.poisson));
    s.normal_k = m.young_pa * s.area / l;
    s.shear_k = shear_modulus * s.area / l;
    s.bend_k = m.young_pa * s.moment / l;
    s.twist_k = shear_modulus * s.polar / l;
    const std::array<double, 7> values{s.area, s.moment, s.polar, s.normal_k,
                                      s.shear_k, s.bend_k, s.twist_k};
    for (double v : values) {
        if (!std::isfinite(v) || !(v > 0.0)) {
            error = "DEM bond: section underflow or overflow";
            return false;
        }
    }
    return true;
}
} // namespace

bool create(const Endpoint& first, const Endpoint& second, State& state,
            std::string& error) {
    error.clear();
    State next;
    if (!endpoint(first, next.first_rest_orientation) ||
        !endpoint(second, next.second_rest_orientation) || first.id == second.id) {
        error = "DEM bond: distinct nonzero identities and finite poses required";
        return false;
    }
    next.first_id = first.id;
    next.second_id = second.id;
    next.rest_separation = subtract(second.position, first.position);
    next.rest_length_m = length(next.rest_separation);
    if (!finite(next.rest_separation) || !std::isfinite(next.rest_length_m) ||
        !(next.rest_length_m > 1.0e-12)) {
        error = "DEM bond: coincident or invalid reference endpoints";
        return false;
    }
    state = next;
    return true;
}

bool evaluate(const Endpoint& first, const Endpoint& second, const Material& material,
              State& state, Response& response, std::string& error) {
    error.clear();
    Quaternion qa, qb, rest_a, rest_b;
    Section s{};
    const double rest_length = length(state.rest_separation);
    if (first.id != state.first_id || second.id != state.second_id ||
        first.id == second.id || !endpoint(first, qa) || !endpoint(second, qb) ||
        !finite(state.rest_separation) ||
        !normalized(state.first_rest_orientation, rest_a) ||
        !normalized(state.second_rest_orientation, rest_b) ||
        !std::isfinite(state.fracture_dissipation_j) || state.fracture_dissipation_j < 0.0 ||
        !std::isfinite(rest_length) ||
        std::abs(rest_length - state.rest_length_m) > 1.0e-10 * rest_length ||
        !section(material, state.rest_length_m, s, error)) {
        if (error.empty()) {
            error = "DEM bond: invalid state or endpoint identity mismatch";
        }
        return false;
    }
    if (state.failure != Failure::None && state.failure != Failure::Tension &&
        state.failure != Failure::Shear) {
        error = "DEM bond: invalid failure state";
        return false;
    }
    Response next;
    if (state.failure != Failure::None) {
        response = next;
        return true;
    }
    const Quaternion da = multiply(qa, conjugate(rest_a));
    Quaternion db = multiply(qb, conjugate(rest_b));
    double alignment = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        alignment += da[i] * db[i];
    }
    if (alignment < 0.0) {
        for (double& v : db) {
            v = -v;
        }
    }
    Quaternion average;
    Quaternion sum{};
    for (std::size_t i = 0; i < 4; ++i) {
        sum[i] = da[i] + db[i];
    }
    if (!normalized(sum, average)) {
        error = "DEM bond: undefined corotational frame";
        return false;
    }
    const Vector separation = subtract(second.position, first.position);
    const Vector displacement = subtract(rotate(conjugate(average), separation),
                                          state.rest_separation);
    const Vector axis = scale(state.rest_separation, 1.0 / state.rest_length_m);
    const double extension = dot(displacement, axis);
    const Vector shear = subtract(displacement, scale(axis, extension));
    const Quaternion relative = multiply(conjugate(da), db);
    const Vector imaginary{relative[1], relative[2], relative[3]};
    const double sine = length(imaginary);
    const double angle = 2.0 * std::atan2(sine, std::max(0.0, relative[0]));
    const Vector rotation = scale(imaginary, sine > 1.0e-12 ? angle / sine : 2.0);
    const double twist = dot(rotation, axis);
    const Vector bend = subtract(rotation, scale(axis, twist));
    const Vector local_force = add(scale(axis, s.normal_k * extension),
                                   scale(shear, s.shear_k));
    const Vector local_couple = add(scale(axis, s.twist_k * twist),
                                    scale(bend, s.bend_k));
    next.tensile_stress_pa = std::max(0.0, s.normal_k * extension / s.area +
        s.bend_k * length(bend) * material.radius_m / s.moment);
    next.shear_stress_pa = s.shear_k * length(shear) / s.area +
        s.twist_k * std::abs(twist) * material.radius_m / s.polar;
    next.elastic_energy_j = 0.5 * (s.normal_k * extension * extension +
        s.shear_k * dot(shear, shear) + s.bend_k * dot(bend, bend) +
        s.twist_k * twist * twist);
    next.first_force_n = rotate(average, local_force);
    next.second_force_n = scale(next.first_force_n, -1.0);
    const Vector couple = rotate(average, local_couple);
    const Vector arm_moment = scale(cross(separation, next.first_force_n), 0.5);
    next.first_torque_nm = add(arm_moment, couple);
    next.second_torque_nm = subtract(arm_moment, couple);
    if (!finite(next.first_force_n) || !finite(next.first_torque_nm) ||
        !finite(next.second_torque_nm) || !std::isfinite(next.elastic_energy_j) ||
        !std::isfinite(next.tensile_stress_pa) || !std::isfinite(next.shear_stress_pa)) {
        error = "DEM bond: nonfinite response";
        return false;
    }
    State updated = state;
    if (next.tensile_stress_pa > material.tensile_strength_pa ||
        next.shear_stress_pa > material.shear_strength_pa) {
        updated.failure = next.tensile_stress_pa > material.tensile_strength_pa
            ? Failure::Tension : Failure::Shear;
        next.released_energy_j = next.elastic_energy_j;
        updated.fracture_dissipation_j += next.released_energy_j;
        if (!std::isfinite(updated.fracture_dissipation_j)) {
            error = "DEM bond: fracture energy overflow";
            return false;
        }
        next.elastic_energy_j = 0.0;
        next.first_force_n = next.second_force_n = {};
        next.first_torque_nm = next.second_torque_nm = {};
        next.broken_now = true;
    }
    state = updated;
    response = next;
    return true;
}

double stableStep(const Material& material, double length_m, double first_mass_kg,
                  double second_mass_kg, double first_inertia_kg_m2,
                  double second_inertia_kg_m2, uint32_t maximum_degree,
                  std::string& error) {
    error.clear();
    Section s{};
    const std::array<double, 4> weights{first_mass_kg, second_mass_kg,
                                       first_inertia_kg_m2, second_inertia_kg_m2};
    for (double weight : weights) {
        if (!std::isfinite(weight) || !(weight > 0.0)) {
            error = "DEM bond: positive finite masses and inertias required";
            return 0.0;
        }
    }
    if (maximum_degree == 0 || !section(material, length_m, s, error)) {
        if (error.empty()) {
            error = "DEM bond: positive graph degree required";
        }
        return 0.0;
    }
    const double inverse_mass = 1.0 / first_mass_kg + 1.0 / second_mass_kg;
    const double inverse_inertia = 1.0 / first_inertia_kg_m2 + 1.0 / second_inertia_kg_m2;
    // Trace bounds the positive semidefinite linearized beam stiffness. Shear
    // couples translation and endpoint spin at half the reference arm.
    const double omega_squared = maximum_degree * (
        (s.normal_k + 2.0 * s.shear_k) * inverse_mass +
        (2.0 * s.bend_k + s.twist_k +
         0.5 * s.shear_k * length_m * length_m) * inverse_inertia);
    if (!std::isfinite(omega_squared) || !(omega_squared > 0.0)) {
        error = "DEM bond: invalid stability bound";
        return 0.0;
    }
    return 0.25 / std::sqrt(omega_squared);
}

} // namespace RayTrophiSim::Fluid::DemBond
