#pragma once

#include "../Vec3.h"

#include <cstdint>
#include <string>

namespace RayTrophiSim::Fluid::Granular {

// Explicit physical spheres, never inferred from virtual render children or
// the radius of an MPM parcel. This staging core does not change solver routing.
struct ContactBody {
    uint64_t id = 0;
    Vec3 position = Vec3(0.0f);
    Vec3 velocity = Vec3(0.0f);
    Vec3 angular_velocity = Vec3(0.0f);
    float radius_m = 0.0f;
    float mass_kg = 0.0f;
    float saturation = 0.0f;
};

struct ContactParams {
    float normal_stiffness_n_m = 10000.0f;
    float normal_damping_n_s_m = 1.0f;
    float tangential_stiffness_n_m = 2500.0f;
    float tangential_damping_n_s_m = 0.5f;
    float dry_friction = 0.5f;
    float saturated_friction = 0.2f;
    float rolling_friction = 0.01f;
};

// Pair order is canonical (a.id < b.id). The caller owns lifetime, removes
// separated pairs, and transports history by identity rather than array index.
struct ContactHistory {
    uint64_t a_id = 0;
    uint64_t b_id = 0;
    Vec3 tangential_displacement = Vec3(0.0f);
};

struct ContactResult {
    bool touching = false;
    float overlap_m = 0.0f;
    Vec3 force_on_a = Vec3(0.0f);
    Vec3 torque_on_a = Vec3(0.0f);
    Vec3 force_on_b = Vec3(0.0f);
    Vec3 torque_on_b = Vec3(0.0f);
};

// Transactional: invalid input leaves history and result unchanged. Linear
// spring/dashpot + history Coulomb sliding; bounded viscous rolling resistance.
// No cohesion, pore-pressure PDE, fracture, collider lookup or MPM conversion.
bool evaluateSphereContact(const ContactBody& a, const ContactBody& b,
                           const ContactParams& params, float dt_s,
                           ContactHistory& history, ContactResult& result,
                           std::string& error);

// Analytic reference support only; this is not a scene mesh collider adapter.
bool evaluatePlaneContact(const ContactBody& body, const Vec3& outward_normal,
                          float offset_m, const ContactParams& params, float dt_s,
                          ContactHistory& history, ContactResult& result,
                          std::string& error);

} // namespace RayTrophiSim::Fluid::Granular
