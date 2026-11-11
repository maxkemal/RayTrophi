#pragma once

#include "Vec3.h"
#include <algorithm>
#include <cmath>
#include <string>

namespace RayTrophiSim::Fluid {

struct DomainWallContact {
    float friction = 0.5f;
    float restitution = 0.15f;
    bool closed = true;
    float adhesion_pa = 0.0f;
    float mass_per_area = 1000.0f; // rho*h; kg/m2 for continuum contact.
};

inline bool validateDomainWallContact(const DomainWallContact& wall, std::string& error) {
    if (!std::isfinite(wall.friction) || wall.friction < 0.0f || wall.friction > 2.0f) {
        error = "wall friction must be finite and in [0, 2]";
        return false;
    }
    if (!std::isfinite(wall.restitution) || wall.restitution < 0.0f ||
        wall.restitution > 1.0f) {
        error = "wall restitution must be finite and in [0, 1]";
        return false;
    }
    if (!std::isfinite(wall.adhesion_pa) || wall.adhesion_pa < 0.0f ||
        wall.adhesion_pa > 1.0e9f) {
        error = "wall adhesion_pa must be finite and in [0, 1e9] Pa";
        return false;
    }
    error.clear();
    return true;
}

inline float wallComponent(const Vec3& v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

inline void setWallComponent(Vec3& v, int axis, float value) {
    if (axis == 0) {
        v.x = value;
    } else if (axis == 1) {
        v.y = value;
    } else {
        v.z = value;
    }
}

// Stationary plane, inward normal. Coulomb impulse cannot reverse slip.
// Support acceleration provides sustained traction after the normal grid
// velocity has already been projected to zero. Separating bodies are free.
inline Vec3 resolveDomainWallVelocity(Vec3 velocity, int axis, float sign,
    const DomainWallContact& wall, const Vec3& acceleration, float dt) {
    const float vn = sign * wallComponent(velocity, axis);
    if (!wall.closed) {
        return velocity;
    }
    const float glue = std::max(wall.adhesion_pa, 0.0f) * std::max(dt, 0.0f) /
        std::max(wall.mass_per_area, 1.0e-6f);
    if (vn > 1.0e-6f && glue <= 0.0f) {
        return velocity;
    }
    const float incoming = std::max(-vn, 0.0f);
    const float support = std::max(-sign * wallComponent(acceleration, axis), 0.0f) * dt;
    const float rebound = wall.restitution * std::max(incoming - support, 0.0f);
    const float normal_change = incoming + rebound;
    // A finite traction budget resists separation without reversing it.
    // Tension spends the cohesive budget before it can resist tangential slip.
    const float cohesion = std::max(glue - std::max(vn, 0.0f), 0.0f);
    const float load = vn > 1.0e-6f ? 0.0f : std::max(normal_change, support);
    Vec3 tangent = velocity;
    setWallComponent(tangent, axis, 0.0f);
    const float speed = tangent.length();
    if (speed > 1.0e-9f) {
        tangent = tangent * (std::max(speed - wall.friction * load - cohesion, 0.0f) / speed);
    }
    const float separating = vn > 0.0f ? vn : rebound;
    setWallComponent(tangent, axis, sign * std::max(separating - glue, 0.0f));
    return tangent;
}

inline void resolveDomainWallParticle(Vec3& position, Vec3& velocity,
    const Vec3& low, const Vec3& high, const DomainWallContact& wall,
    const Vec3& acceleration, float dt, float band) {
    if (!wall.closed) {
        return;
    }
    for (int axis = 0; axis < 3; ++axis) {
        const float p = wallComponent(position, axis);
        const float lo = wallComponent(low, axis), hi = wallComponent(high, axis);
        if (p <= lo + band) {
            velocity = resolveDomainWallVelocity(velocity, axis, 1.0f, wall, acceleration, dt);
        } else if (p >= hi - band) {
            velocity = resolveDomainWallVelocity(velocity, axis, -1.0f, wall, acceleration, dt);
        }
        setWallComponent(position, axis, std::clamp(p, lo, hi));
    }
}

// MAC-face adapter. A snapshot is mandatory: neighbouring invocations must
// never observe partially constrained velocities. Tangential contact occupies
// the first half-cell on each side of the closed domain.
template <class Sample>
float resolveDomainWallFace(float value, int component, const int face[3],
    const int size[3], const DomainWallContact& wall, const Vec3& acceleration,
    float dt, const Sample& sample) {
    if (!wall.closed) {
        return value;
    }
    // MPM grid velocities contain constitutive stress impulses as well as
    // body forces. Rebounding them re-injects the support response on every
    // tick. Material elasticity owns rebound; grid contact only removes
    // inward normal motion and applies Coulomb traction.
    auto grid_wall = wall;
    grid_wall.restitution = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        const int last = size[axis] - (axis != component);
        if (face[axis] != 0 && face[axis] != last) {
            continue;
        }
        const float sign = face[axis] == 0 ? 1.0f : -1.0f;
        Vec3 velocity(0.0f);
        for (int a = 0; a < 3; ++a) {
            float speed = value;
            if (a != component) {
                int q[3] = {face[0], face[1], face[2]};
                q[component] -= 1;
                speed = sample(a, q);
                q[a] += 1;
                speed += sample(a, q);
                q[component] += 1;
                speed += sample(a, q);
                q[a] -= 1;
                speed = (speed + sample(a, q)) * 0.25f;
            }
            setWallComponent(velocity, a, speed);
        }
        if (axis != component) {
            int q[3] = {face[0], face[1], face[2]};
            q[axis] = sign > 0.0f ? 0 : size[axis];
            float normal = sample(axis, q);
            q[component] -= 1;
            normal = 0.5f * (normal + sample(axis, q));
            setWallComponent(velocity, axis, normal);
        }
        value = wallComponent(resolveDomainWallVelocity(velocity, axis, sign,
            grid_wall, acceleration, dt), component);
    }
    return value;
}

} // namespace RayTrophiSim::Fluid
