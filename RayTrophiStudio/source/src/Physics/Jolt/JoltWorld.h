#pragma once

// Faz 1: a thin, RayTrophi-typed wrapper around a single Jolt PhysicsSystem.
//
// JoltWorld owns the per-world Jolt objects (PhysicsSystem, TempAllocator,
// JobSystemThreadPool, broad-phase / object-layer filters) and handles the
// process-wide Jolt registration (allocator/factory/types) with refcounting
// so several worlds or a re-init are safe.
//
// This header deliberately exposes ONLY RayTrophi types (::Vec3, ::Matrix4x4)
// via a pimpl, so consumers (RigidBodySystem, UI, serializer) never need to
// include Jolt headers or match Jolt's compile-time defines.

#include <cstddef>
#include <cstdint>
#include <memory>

#include "Vec3.h"
#include "Matrix4x4.h"

namespace RayTrophiSim {
namespace JoltIntegration {

using JoltBodyHandle = uint32_t;
static constexpr JoltBodyHandle kInvalidBody = 0xffffffffu;

enum class JoltShapeType {
    Box,        // half_extents
    Sphere,     // radius
    Capsule     // radius + half_height (cylinder half-length, excludes caps)
};

enum class JoltMotionType {
    Static,
    Dynamic,
    Kinematic
};

struct JoltBodyDesc {
    JoltShapeType shape = JoltShapeType::Box;

    // Shape dimensions (already in world scale; JoltWorld bakes scale into the
    // shape, not the body pose).
    Vec3 half_extents = Vec3(0.5f, 0.5f, 0.5f); // Box
    float radius = 0.5f;                         // Sphere / Capsule
    float half_height = 0.5f;                    // Capsule

    Matrix4x4 transform = Matrix4x4::identity(); // initial world pose (rot+trans)

    bool dynamic = true;        // legacy alias; motion_type is authoritative
    JoltMotionType motion_type = JoltMotionType::Dynamic;
    float mass = 1.0f;          // dynamic only; <= 0 => auto from shape volume
    float friction = 0.5f;
    float restitution = 0.2f;
    Vec3 initial_linear_velocity = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 initial_angular_velocity = Vec3(0.0f, 0.0f, 0.0f);
    float linear_damping = 0.05f;
    float angular_damping = 0.05f;
    float gravity_scale = 1.0f;
    bool sleep_enabled = true;

    bool start_active = true;    // dynamic bodies: begin awake
};

class JoltWorld {
public:
    struct Config {
        Vec3 gravity = Vec3(0.0f, -9.81f, 0.0f);
        uint32_t max_bodies = 10240;
        uint32_t max_body_pairs = 16384;
        uint32_t max_contact_constraints = 8192;
        int num_threads = -1;   // -1 => hardware_concurrency() - 1
    };

    JoltWorld();
    ~JoltWorld();

    JoltWorld(const JoltWorld&) = delete;
    JoltWorld& operator=(const JoltWorld&) = delete;

    bool init(const Config& cfg = Config{});
    void shutdown();
    bool isInitialized() const;

    // Body lifecycle
    JoltBodyHandle createBody(const JoltBodyDesc& desc);
    void removeBody(JoltBodyHandle handle);
    void clearBodies();
    std::size_t bodyCount() const;

    // Simulation
    void update(float dt, int collision_steps = 1);
    void optimizeBroadPhase();   // call once after the initial body batch
    void setGravity(const Vec3& g);

    // Body state (RT types; transforms are rotation+translation, no scale)
    Matrix4x4 getBodyTransform(JoltBodyHandle handle) const;
    void setBodyTransform(JoltBodyHandle handle, const Matrix4x4& transform);
    Vec3 getBodyPosition(JoltBodyHandle handle) const;
    Vec3 getBodyLinearVelocity(JoltBodyHandle handle) const;
    void setBodyLinearVelocity(JoltBodyHandle handle, const Vec3& v);
    Vec3 getBodyAngularVelocity(JoltBodyHandle handle) const;
    Vec3 getBodyCenterOfMass(JoltBodyHandle handle) const;
    // World-space velocity of the body AT a world point (linear + angular
    // contribution). Used by fluid-coupling drag so off-centre sample points
    // see the rotational part of the body's motion.
    Vec3 getBodyPointVelocity(JoltBodyHandle handle, const Vec3& world_point) const;
    bool isBodyActive(JoltBodyHandle handle) const;
    void activateBody(JoltBodyHandle handle);

    // ---- External forces (accumulated, consumed by the next update()) --------
    // Call BEFORE update(). `wake` activates a sleeping body so a settled
    // floater still reacts when the water around it starts moving; pass false
    // to let a resting body stay asleep.
    void addForce(JoltBodyHandle handle, const Vec3& force, bool wake = true);
    void addForceAtPoint(JoltBodyHandle handle, const Vec3& force,
                         const Vec3& world_point, bool wake = true);
    void addTorque(JoltBodyHandle handle, const Vec3& torque, bool wake = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace JoltIntegration
} // namespace RayTrophiSim
