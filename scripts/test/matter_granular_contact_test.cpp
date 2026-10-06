#include "Fluid/GranularContact.h"

#include <cassert>
#include <cmath>
#include <limits>

int main() {
    using namespace RayTrophiSim::Fluid::Granular;
    ContactBody a;
    a.id = 1;
    a.radius_m = 0.1f;
    a.mass_kg = 1.0f;
    ContactBody b = a;
    b.id = 2;
    b.position = Vec3(0.19f, 0.0f, 0.0f);
    ContactParams params;
    params.rolling_friction = 0.0f;
    ContactHistory history;
    ContactResult result;
    std::string error;
    const float dt = 0.001f;
    assert(evaluateSphereContact(a, b, params, dt, history, result, error));
    assert(result.touching);
    assert(std::abs(result.force_on_a.x + 100.0f) < 0.001f);
    assert((result.force_on_a + result.force_on_b).length() < 1e-6f);

    a.velocity = Vec3(0.0f, 100.0f, 0.0f);
    history = ContactHistory{};
    assert(evaluateSphereContact(a, b, params, dt, history, result, error));
    const float dry_force = std::abs(result.force_on_a.y);
    assert(std::abs(dry_force - 50.0f) < 0.001f);
    const Vec3 net_torque = a.position.cross(result.force_on_a) + result.torque_on_a +
        b.position.cross(result.force_on_b) + result.torque_on_b;
    assert(net_torque.length() < 1e-5f);
    assert(result.force_on_a.y * a.velocity.y <= 0.0f);

    // Only b becomes wet: a's local state is unchanged, pair uses local surfaces.
    b.saturation = 1.0f;
    history = ContactHistory{};
    assert(evaluateSphereContact(a, b, params, dt, history, result, error));
    assert(std::abs(std::abs(result.force_on_a.y) - 20.0f) < 0.001f);
    assert(a.saturation == 0.0f);

    // Separation removes stored tangential spring energy/history.
    b.position.x = 0.3f;
    assert(evaluateSphereContact(a, b, params, dt, history, result, error));
    assert(!result.touching && history.tangential_displacement.length() == 0.0f);

    // Reusing a pair's history for another identity is rejected transactionally.
    b.id = 3;
    const ContactHistory old_history = history;
    assert(!evaluateSphereContact(a, b, params, dt, history, result, error));
    assert(history.b_id == old_history.b_id && !result.touching);
    b.id = 2;
    b.position.x = 0.19f;
    a.velocity = Vec3(0.0f);
    a.angular_velocity = Vec3(0.0f, 0.01f, 0.0f);
    params.rolling_friction = 1.0f;
    params.tangential_damping_n_s_m = 0.0f;
    history = ContactHistory{};
    assert(evaluateSphereContact(a, b, params, dt, history, result, error));
    // Isolate rolling component from sliding contact torque.
    const Vec3 arm_a(0.095f, 0.0f, 0.0f);
    const Vec3 rolling_torque = result.torque_on_a - arm_a.cross(result.force_on_a);
    assert(rolling_torque.y <= 0.0f);
    const float inverse_inertia_sum = 500.0f;
    assert(std::abs(rolling_torque.y) * inverse_inertia_sum * dt <= 0.010001f);

    // NaN dt, invalid saturation and coincident centers never publish state.
    assert(!evaluateSphereContact(a, b, params,
        std::numeric_limits<float>::quiet_NaN(), history, result, error));
    b.saturation = 2.0f;
    assert(!evaluateSphereContact(a, b, params, dt, history, result, error));
    b.saturation = 0.0f;
    b.position = a.position;
    assert(!evaluateSphereContact(a, b, params, dt, history, result, error));

    a.position = Vec3(0.0f, 0.095f, 0.0f);
    a.velocity = Vec3(1.0f, 0.0f, 0.0f);
    a.angular_velocity = Vec3(0.0f);
    params.rolling_friction = 0.0f;
    history = ContactHistory{};
    assert(evaluatePlaneContact(a, Vec3(0.0f, 1.0f, 0.0f), 0.0f,
        params, dt, history, result, error));
    assert(result.touching && std::abs(result.force_on_a.y - 50.0f) < 0.001f);
    assert(result.force_on_a.x < 0.0f && result.torque_on_a.z < 0.0f);
    assert((a.position.cross(result.force_on_a) + result.torque_on_a).length() < 1e-5f);
    assert(!evaluatePlaneContact(a, Vec3(0.0f, 2.0f, 0.0f), 0.0f,
        params, dt, history, result, error));
}
