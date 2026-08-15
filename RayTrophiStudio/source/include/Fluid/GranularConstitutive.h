#pragma once

#include <algorithm>
#include <cmath>

namespace RayTrophiSim::Fluid::Granular {

struct Parameters {
    float friction_angle_radians = 0.61086524f; // 35 degrees
    float cohesion = 0.0f;
    float dilatancy = 0.0f;
    float hardening = 0.0f;
    float tensile_cutoff = 0.0f;
    float detach_pressure = 1.0e-4f;
};

struct InvariantState {
    float mean_pressure = 0.0f;
    float shear_measure = 0.0f;
    float yield_value = 0.0f;
    float plastic_increment = 0.0f;
    bool yielded = false;
    bool detached = false;
    bool finite = true;
};

// Drucker-Prager invariant update. The stress tensor is supplied through its
// mean pressure and deviatoric magnitude so the same scalar kernel can be
// mirrored exactly in GLSL and in the small CPU golden tests.
inline InvariantState evaluate(float mean_pressure, float shear_measure,
                               float plastic_volume, const Parameters& p) {
    InvariantState out;
    const float pressure = std::max(mean_pressure, 0.0f);
    const float friction = std::tan(std::clamp(p.friction_angle_radians,
                                               0.0f, 1.3962634f));
    const float strength = p.cohesion + pressure * friction;
    out.mean_pressure = mean_pressure;
    out.shear_measure = std::max(shear_measure, 0.0f);
    out.yield_value = out.shear_measure - std::max(strength, 0.0f);
    out.yielded = out.yield_value > 0.0f;
    out.plastic_increment = out.yielded
        ? out.yield_value / std::max(1.0f + p.hardening, 1.0e-6f)
        : 0.0f;
    out.detached = mean_pressure < -std::max(p.tensile_cutoff, 0.0f) ||
                   (mean_pressure <= p.detach_pressure && plastic_volume < 0.5f);
    out.finite = std::isfinite(out.yield_value) &&
                 std::isfinite(out.plastic_increment);
    if (!out.finite) {
        out.yielded = false;
        out.detached = true;
        out.plastic_increment = 0.0f;
    }
    return out;
}

} // namespace RayTrophiSim::Fluid::Granular
