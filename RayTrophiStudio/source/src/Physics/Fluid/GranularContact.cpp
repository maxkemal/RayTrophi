#include "Fluid/GranularContact.h"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace RayTrophiSim::Fluid::Granular {
namespace {

bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

bool validBody(const ContactBody& body) {
    return body.id != 0 && finite(body.position) && finite(body.velocity) &&
        finite(body.angular_velocity) && std::isfinite(body.radius_m) &&
        body.radius_m > 0.0f && std::isfinite(body.mass_kg) && body.mass_kg > 0.0f &&
        std::isfinite(body.saturation) && body.saturation >= 0.0f && body.saturation <= 1.0f;
}

bool contactLaw(const ContactBody& a, uint64_t other_id, float other_saturation,
                const ContactParams& params, float dt_s, bool fresh, float overlap,
                const Vec3& normal, const Vec3& arm_a, const Vec3& arm_b,
                const Vec3& relative, const Vec3& relative_omega,
                float inverse_inertia_sum, float effective_radius,
                ContactHistory& history, ContactResult& result, std::string& error) {
    ContactHistory next;
    next.a_id = a.id;
    next.b_id = other_id;
    ContactResult candidate;
    candidate.overlap_m = overlap;
    if (candidate.overlap_m > 0.0f) {
        candidate.touching = true;
        const float approach = dot(relative, normal);
        const float normal_force = std::max(0.0f,
            params.normal_stiffness_n_m * candidate.overlap_m +
            params.normal_damping_n_s_m * approach);
        const Vec3 slip = relative - normal * approach;
        const Vec3 previous = fresh ? Vec3(0.0f) : history.tangential_displacement;
        next.tangential_displacement = previous - normal * dot(previous, normal) + slip * dt_s;
        Vec3 tangent = next.tangential_displacement * (-params.tangential_stiffness_n_m) -
            slip * params.tangential_damping_n_s_m;
        const auto friction = [&params](float saturation) {
            return params.dry_friction +
                (params.saturated_friction - params.dry_friction) * saturation;
        };
        // Weakest contacting surface, evaluated locally. No domain saturation
        // average and no change to stored water mass/capacity.
        const float limit = std::min(friction(a.saturation), friction(other_saturation)) *
            normal_force;
        const float tangent_length = tangent.length();
        if (tangent_length > limit && tangent_length > 0.0f) {
            tangent *= limit / tangent_length;
            next.tangential_displacement =
                (tangent + slip * params.tangential_damping_n_s_m) /
                (-params.tangential_stiffness_n_m);
        }
        const Vec3 rolling = relative_omega - normal * dot(relative_omega, normal);
        const float rolling_speed = rolling.length();
        Vec3 rolling_torque(0.0f);
        if (rolling_speed > 0.0f) {
            const float torque_limit = params.rolling_friction * normal_force * effective_radius;
            const float stop_torque = rolling_speed /
                (inverse_inertia_sum * dt_s);
            rolling_torque = rolling * (-std::min(torque_limit, stop_torque) / rolling_speed);
        }
        candidate.force_on_a = tangent - normal * normal_force;
        candidate.force_on_b = -candidate.force_on_a;
        candidate.torque_on_a = arm_a.cross(tangent) + rolling_torque;
        candidate.torque_on_b = arm_b.cross(-tangent) - rolling_torque;
    }
    if (!finite(candidate.force_on_a) || !finite(candidate.torque_on_a) ||
        !finite(candidate.torque_on_b) || !finite(next.tangential_displacement)) {
        error = "DEM contact arithmetic overflow";
        return false;
    }
    history = next;
    result = candidate;
    error.clear();
    return true;
}

} // namespace

bool evaluateSphereContact(const ContactBody& a, const ContactBody& b,
                           const ContactParams& params, float dt_s,
                           ContactHistory& history, ContactResult& result,
                           std::string& error) {
    const float coefficients[] = {params.normal_stiffness_n_m, params.normal_damping_n_s_m,
        params.tangential_stiffness_n_m, params.tangential_damping_n_s_m,
        params.dry_friction, params.saturated_friction, params.rolling_friction};
    if (!validBody(a) || !validBody(b) || a.id >= b.id || !std::isfinite(dt_s) ||
        dt_s <= 0.0f || !finite(history.tangential_displacement) ||
        !std::all_of(std::begin(coefficients), std::end(coefficients), [](float value) {
            return std::isfinite(value) && value >= 0.0f;
        }) || params.normal_stiffness_n_m <= 0.0f ||
        params.tangential_stiffness_n_m <= 0.0f) {
        error = "DEM contact requires ordered identities, finite spheres and positive dt/stiffness";
        return false;
    }
    const bool fresh = history.a_id == 0 && history.b_id == 0;
    if (!fresh && (history.a_id != a.id || history.b_id != b.id)) {
        error = "DEM contact history belongs to a different particle pair";
        return false;
    }
    const Vec3 separation = b.position - a.position;
    const float distance = separation.length();
    if (!std::isfinite(distance) || distance <= 1e-8f) {
        error = "DEM contact cannot resolve coincident sphere centers";
        return false;
    }
    const Vec3 normal = separation / distance;
    const Vec3 arm_a = normal * (distance * a.radius_m / (a.radius_m + b.radius_m));
    const Vec3 arm_b = arm_a - separation;
    const Vec3 relative = a.velocity + a.angular_velocity.cross(arm_a) -
        b.velocity - b.angular_velocity.cross(arm_b);
    const float inverse_inertia = 2.5f / (a.mass_kg * a.radius_m * a.radius_m) +
        2.5f / (b.mass_kg * b.radius_m * b.radius_m);
    return contactLaw(a, b.id, b.saturation, params, dt_s, fresh,
        std::max(0.0f, a.radius_m + b.radius_m - distance), normal, arm_a, arm_b,
        relative, a.angular_velocity - b.angular_velocity, inverse_inertia,
        a.radius_m * b.radius_m / (a.radius_m + b.radius_m), history, result, error);
}

bool evaluatePlaneContact(const ContactBody& body, const Vec3& outward_normal,
                          float offset_m, const ContactParams& params, float dt_s,
                          ContactHistory& history, ContactResult& result,
                          std::string& error) {
    const float coefficients[] = {params.normal_stiffness_n_m, params.normal_damping_n_s_m,
        params.tangential_stiffness_n_m, params.tangential_damping_n_s_m,
        params.dry_friction, params.saturated_friction, params.rolling_friction};
    const bool fresh = history.a_id == 0 && history.b_id == 0;
    if (!validBody(body) || !finite(outward_normal) || !std::isfinite(offset_m) ||
        std::abs(outward_normal.length_squared() - 1.0f) > 1e-4f ||
        !std::isfinite(dt_s) || dt_s <= 0.0f || !finite(history.tangential_displacement) ||
        (!fresh && (history.a_id != body.id || history.b_id != 0)) ||
        !std::all_of(std::begin(coefficients), std::end(coefficients), [](float value) {
            return std::isfinite(value) && value >= 0.0f;
        }) || params.normal_stiffness_n_m <= 0.0f ||
        params.tangential_stiffness_n_m <= 0.0f) {
        error = "DEM plane contact requires a unit normal and valid local contact state";
        return false;
    }
    const float distance = dot(body.position, outward_normal) - offset_m;
    // Contact point is the projection on the analytic plane, including overlap.
    const Vec3 arm = outward_normal * (-distance);
    const float inverse_inertia = 2.5f / (body.mass_kg * body.radius_m * body.radius_m);
    return contactLaw(body, 0, body.saturation, params, dt_s, fresh,
        std::max(0.0f, body.radius_m - distance), -outward_normal, arm, Vec3(0.0f),
        body.velocity + body.angular_velocity.cross(arm), body.angular_velocity,
        inverse_inertia, body.radius_m, history, result, error);
}

} // namespace RayTrophiSim::Fluid::Granular
