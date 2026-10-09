#ifndef RT_MATTER_GRAIN_SLEEP_TRANSFER
#define RT_MATTER_GRAIN_SLEEP_TRANSFER

// Compare the recipient's kinetic response per degree of freedom with the
// existing linear/surface-angular sleep limits. J and L are impulses, not
// source velocities or static contact preloads. No authored threshold changes.
bool grainSleepKickSignificant(vec3 impulse, vec3 angular_impulse,
                              float inverse_mass, float inverse_inertia,
                              float radius) {
    vec3 velocity_change = impulse * inverse_mass;
    vec3 surface_change = angular_impulse * (inverse_inertia * radius);
    float limit_squared = pc.sleep.y * pc.sleep.y;
    return dot(velocity_change, velocity_change) >= limit_squared ||
        dot(surface_change, surface_change) >= limit_squared;
}

float grainSleepAuditHorizon() {
    return max(pc.step_contact.x, min(0.02, pc.sleep.w));
}

// Conservative finite-mass collision forecast prevents an incipient impact
// from penetrating a sleeper until the periodic audit. Tangential, rolling and
// twist impulses are also bounded by their actual Coulomb/contact-patch limits.
// The returned impulses act on the OTHER grain; its own audit decides whether
// its complete support/contact system remains balanced.
void grainSleepContactForecast(vec3 normal, vec3 arm, vec3 relative, vec3 spin,
                               float normal_force, float inverse_normal_mass,
                               float inverse_tangent_mass, float inverse_pair_inertia,
                               float effective_radius, float patch_radius,
                               out vec3 impulse, out vec3 angular_impulse) {
    float horizon = grainSleepAuditHorizon();
    float normal_speed = dot(relative, normal);
    float normal_impulse = 2.0 * max(-normal_speed, 0.0) / inverse_normal_mass;
    impulse = -normal * normal_impulse;
    angular_impulse = vec3(0.0);

    vec3 slip = relative - normal_speed * normal;
    float slip_speed = length(slip);
    if (slip_speed > 1e-9) {
        float tangent_impulse = min(pc.step_contact.w * normal_force * horizon,
                                    2.0 * slip_speed / inverse_tangent_mass);
        vec3 source_tangent_impulse = -slip * (tangent_impulse / slip_speed);
        impulse -= source_tangent_impulse;
        angular_impulse += cross(arm, source_tangent_impulse);
    }

    vec3 roll = spin - dot(spin, normal) * normal;
    float roll_speed = length(roll);
    if (roll_speed > 1e-9) {
        float rolling_impulse = min(pc.rolling.x * normal_force * effective_radius * horizon,
                                    2.0 * roll_speed / inverse_pair_inertia);
        angular_impulse += roll * (rolling_impulse / roll_speed);
    }
    float twist_speed = dot(spin, normal);
    float twisting_impulse = min(uintBitsToFloat(pc.meta.z) * normal_force * patch_radius * horizon,
                                 2.0 * abs(twist_speed) / inverse_pair_inertia);
    angular_impulse += sign(twist_speed) * normal * twisting_impulse;
}

#endif
