#pragma once

#include "Fluid/APICFluidSolver.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/FluidDomainSubstance.h"
#include "Fluid/SubstanceTag.h"
#include "Fluid/DomainWallContact.h"
#include "json.hpp"

namespace RayTrophiSim::Fluid {

inline DomainWallContact domainWallContact(const APICSolverParams& params, float h = 1.0f) {
    DomainWallContact wall{params.wall_friction, params.wall_damping,
        params.boundary == APICSolverParams::BoundaryMode::Closed};
    wall.adhesion_pa = params.wall_adhesion_pa;
    if (wall.adhesion_pa > 0.0f) {
        const bool solid = params.granular_enabled && !params.grain.enabled;
        const auto tag = solid ? 0u : substanceTag(params.liquid_physics_substance);
        const float density = solid ? std::max(params.granular_density, 1.0f)
            : fluidParticleRestMassKg(tag, domainSubstance(params), 1.0f, 1,
                MatterConstitutiveModel::Fluid, false);
        wall.mass_per_area = density * std::max(h, 1.0e-6f);
    }
    return wall;
}

inline nlohmann::json domainWallSettings(const APICSolverParams& params, bool closed) {
    return {{"friction", params.wall_friction}, {"restitution", params.wall_damping},
        {"adhesion_pa", params.wall_adhesion_pa}, {"active", closed}};
}

// Atomic patch, shared by UI, Python and IPC. No silent coercion/clamping.
inline bool patchDomainWallSettings(APICSolverParams& params,
    const nlohmann::json& patch, std::string& error) {
    if (!patch.is_object()) {
        error = "wall settings must be an object";
        return false;
    }
    auto wall = domainWallContact(params);
    for (const auto& item : patch.items()) {
        if (item.key() != "friction" && item.key() != "restitution" &&
            item.key() != "adhesion_pa") {
            error = "unknown wall setting: " + item.key();
            return false;
        }
        if (!item.value().is_number()) {
            error = "wall " + item.key() + " must be numeric";
            return false;
        }
        const double value = item.value().get<double>();
        const double maximum = item.key() == "adhesion_pa" ? 1.0e9
            : item.key() == "friction" ? 2.0 : 1.0;
        if (!std::isfinite(value) || value < 0.0 || value > maximum) {
            error = "wall " + item.key() + " outside allowed range";
            return false;
        }
        if (item.key() == "friction") {
            wall.friction = static_cast<float>(value);
        } else if (item.key() == "adhesion_pa") {
            wall.adhesion_pa = static_cast<float>(value);
        } else {
            wall.restitution = static_cast<float>(value);
        }
    }
    if (!validateDomainWallContact(wall, error)) {
        return false;
    }
    params.wall_adhesion_pa = wall.adhesion_pa;
    params.wall_friction = wall.friction;
    params.wall_damping = wall.restitution;
    return true;
}

inline void resolveDomainWallTail(Vec3& position, Vec3& velocity,
    const Vec3& low, const Vec3& high, const APICSolverParams& params, float dt, float h = 1.0f) {
    auto wall = domainWallContact(params, h);
    // MPM already resolves traction on the grid. The tail only guards escape.
    if (params.granular_enabled) {
        wall.adhesion_pa = 0.0f;
        wall.friction = 0.0f;
        wall.restitution = 0.0f;
    }
    const Vec3 before = velocity;
    resolveDomainWallParticle(position, velocity, low, high, wall,
        params.gravity, dt, wall.adhesion_pa > 0.0f ? h : 1.0e-6f);
    if (wall.adhesion_pa > 0.0f) {
        // The contact impulse also constrains this substep's displacement.
        // Otherwise grid advection can move a zero-velocity glued parcel away.
        position = position + (velocity - before) * dt;
        position = Vec3::max(low, Vec3::min(high, position));
    }
}

} // namespace RayTrophiSim::Fluid
