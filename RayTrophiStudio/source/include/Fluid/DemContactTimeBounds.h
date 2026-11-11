#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace RayTrophiSim::Fluid {

// Material-independent reference for the existing explicit DEM integrator.
// Contacts, tangential/rolling response and capillary bridges keep separate
// stiffness contributions. An implicit solver must still respect collision
// accuracy and travel limits even when its elastic stability restriction changes.
struct DemContactTimeInput {
    double mass_kg = 0.0;
    double normal_stiffness_n_m = 0.0;
    double tangential_ratio = 0.0;
    double rolling_friction = 0.0;
    double rolling_factor = 2.8125;
    double bridge_stiffness_n_m = 0.0;
    double damping_ratio = 0.0;
    double contacts = 0.0;
    double contact_resolution = 0.0;
};

struct DemContactTimeBounds {
    double accuracy = 0.0;
    double stability = 0.0;
    double k_effective = 0.0;
    double aggregate_zeta = 0.0;
    const char* source = "contact";
    const char* stability_name = "stability";
};

inline DemContactTimeBounds demExplicitContactTimeBounds(
    const DemContactTimeInput& input, double stiffness_scale) {
    DemContactTimeBounds result;
    const double k = input.normal_stiffness_n_m * stiffness_scale;
    const std::array<std::pair<double, const char*>, 4> springs{{
        {k, "contact"},
        {3.5 * input.tangential_ratio * k, "tangential"},
        {input.rolling_factor * input.rolling_friction * input.rolling_friction * k,
            "rolling"},
        {input.bridge_stiffness_n_m, "bridge"}
    }};
    for (const auto& spring : springs) {
        if (spring.first > result.k_effective) {
            result.k_effective = spring.first;
            result.source = spring.second;
        }
    }
    result.accuracy = 3.14159265358979 * std::sqrt(0.5 * input.mass_kg / k) /
        input.contact_resolution;
    const double normal_damping = 2.0 * input.damping_ratio *
        std::sqrt(k * input.mass_kg);
    const double damping_rate = input.contacts * 2.0 * normal_damping / input.mass_kg;
    const double omega = std::sqrt(2.0 * input.contacts * result.k_effective /
        input.mass_kg);
    result.aggregate_zeta = damping_rate / (2.0 * omega);
    // Algebraically equal to (sqrt(1+z*z)-z)/omega. The reciprocal form
    // avoids catastrophic cancellation for highly damped contact networks.
    result.stability = 1.0 / (omega *
        (std::hypot(1.0, result.aggregate_zeta) + result.aggregate_zeta));
    result.stability_name = result.aggregate_zeta > 0.1 ? "damping" : "stability";
    return result;
}

} // namespace RayTrophiSim::Fluid
