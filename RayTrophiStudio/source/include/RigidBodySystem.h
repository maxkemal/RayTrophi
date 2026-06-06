#pragma once

// Faz 2: rigid-body simulation system.
//
// Mirrors the gas/fluid pattern: SceneData owns a std::vector<RigidBodyObject>
// and a shared_ptr<RigidBodySystem>; the system is registered with the shared
// SimulationWorld and ticks on every stepOnce(). Each RigidBodyObject drives a
// scene object (by nodeName): the system creates a Jolt body sized/posed from
// the object's oriented bounds, steps the Jolt world, then writes the resulting
// rigid motion back onto the object's transform.
//
// Kept Jolt-free (JoltWorld is forward-declared, held by unique_ptr) so SceneData
// and the UI never include Jolt headers.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Vec3.h"
#include "Matrix4x4.h"
#include "SimulationWorld.h"   // ISimulationSystem, SimulationSystemKind, SimulationContext

namespace RayTrophiSim {

namespace JoltIntegration { class JoltWorld; }

enum class RigidBodyShape : uint8_t {
    Box = 0,      // oriented box from object bounds (default)
    Sphere = 1,   // sphere from object bounds radius
    Capsule = 2
};

enum class RigidBodyMotionType : uint8_t {
    Static = 0,
    Dynamic = 1,
    Kinematic = 2
};

struct RigidBodyObject {
    std::string name = "Rigid Body";
    std::string source_name;          // scene object nodeName this body drives
    std::string collider_name;        // authored collider descriptor used for Jolt shape/material
    bool enabled = true;
    bool dynamic = true;              // false => static collision geometry
    RigidBodyMotionType motion_type = RigidBodyMotionType::Dynamic;
    // Fallback values for legacy bodies or when no authored collider exists.
    RigidBodyShape shape = RigidBodyShape::Box;
    float mass = 1.0f;               // dynamic only; <= 0 => auto from shape
    bool auto_mass_from_density = false;
    float density = 500.0f;          // kg/m^3; water-ish floaters are < 1000
    float linear_damping = 0.05f;
    float angular_damping = 0.05f;
    float gravity_scale = 1.0f;
    float friction = 0.5f;
    float restitution = 0.2f;
    Vec3 initial_linear_velocity = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 initial_angular_velocity = Vec3(0.0f, 0.0f, 0.0f);
    bool sleep_enabled = true;
    bool lock_translation_x = false;
    bool lock_translation_y = false;
    bool lock_translation_z = false;
    bool lock_rotation_x = false;
    bool lock_rotation_y = false;
    bool lock_rotation_z = false;
    bool fluid_coupling_enabled = false;
    float buoyancy_scale = 1.0f;
    float fluid_density = 1000.0f;    // kg/m^3, water default
    float fluid_drag = 1.0f;
    float fluid_angular_drag = 1.0f;

    // ---- runtime state (not serialized) ----
    uint32_t handle = 0xffffffffu;    // JoltBodyHandle once created
    bool created = false;
    bool rest_captured = false;       // rest pose (initial_pivot/body_xf/half) is valid
    Matrix4x4 initial_body_xf = Matrix4x4::identity();  // B0: body world pose at creation
    Matrix4x4 initial_pivot = Matrix4x4::identity();    // P0: object REST pose (spawn point)
    Vec3 rest_half_extents = Vec3(0.5f, 0.5f, 0.5f);    // shape dims captured at rest (reused on shape/param change)
    Matrix4x4 last_written_pivot = Matrix4x4::identity(); // last pose the sim pushed onto the object
    bool has_written = false;                            // sim has pushed at least one pose

    // ---- fluid-coupling diagnostics (runtime, not serialized) ----
    // Filled by applyFluidCoupling each step a coupled body is processed; read
    // by the UI so the buoyancy behaviour can be inspected live.
    bool  dbg_coupled = false;          // coupling ran for this body this step
    int   dbg_sample_count = 0;         // total sample points generated
    int   dbg_submerged_pts = 0;        // sample points with w > 0
    float dbg_min_sd = 0.0f;            // most-submerged sample signed distance (m)
    float dbg_buoy_accel_y = 0.0f;      // net buoyancy accel (m/s^2, +up)
    float dbg_drag_accel_y = 0.0f;      // net drag accel y (m/s^2)
    float dbg_vel_y = 0.0f;             // body linear velocity y (m/s)
    float dbg_body_density = 0.0f;      // effective mass/volume (kg/m^3) used by the solver
};

class RigidBodySystem final : public ISimulationSystem {
public:
    RigidBodySystem();
    ~RigidBodySystem() override;

    RigidBodySystem(const RigidBodySystem&) = delete;
    RigidBodySystem& operator=(const RigidBodySystem&) = delete;

    const char* name() const override { return "RigidBodies"; }
    SimulationSystemKind kind() const override { return SimulationSystemKind::RigidBody; }
    int order() const override { return 50; }  // runs BEFORE gas/fluid (100)
    bool enabled() const override { return enabled_; }
    void setEnabled(bool e) { enabled_ = e; }

    void step(const SimulationContext& ctx) override;

    void setBodies(std::vector<RigidBodyObject>* bodies) { bodies_ = bodies; }

    // ---- scene wiring (set by SceneData) ----
    // Fills an oriented box for the object: world pose at the box CENTER (rot+trans,
    // no scale) and its half-extents. Returns false if the object can't be sized yet.
    using ShapeResolver = std::function<bool(const RigidBodyObject& body,
                                             Matrix4x4& out_box_pose,
                                             Vec3& out_half_extents,
                                             RigidBodyShape& out_shape)>;
    using PivotGetter   = std::function<bool(const std::string& node, Matrix4x4& out_pivot)>;
    using PivotSetter   = std::function<void(const std::string& node, const Matrix4x4& pivot)>;

    void setShapeResolver(ShapeResolver r) { shape_resolver_ = std::move(r); }
    void setPivotGetter(PivotGetter g) { pivot_getter_ = std::move(g); }
    void setPivotSetter(PivotSetter s) { pivot_setter_ = std::move(s); }

    // ---- Rigid-fluid coupling (buoyancy + drag) -----------------------------
    // Result of querying the fluid state at a world point. signed_distance < 0
    // means the point is inside the fluid (|signed_distance| ≈ depth below the
    // surface); `velocity` is the fluid velocity there (for drag).
    struct FluidSample {
        float signed_distance = 1.0e30f;  // <0 inside fluid (world units)
        Vec3  velocity = Vec3(0.0f, 0.0f, 0.0f);
        bool  valid = false;              // point lies within some fluid domain
    };
    // Sample the fluid at a world point. Returns false if no fluid domain
    // contains the point. Called once per sample point per coupled body.
    using FluidSampler = std::function<bool(const Vec3& world_point, FluidSample& out)>;
    // Invoked once at the top of each step (only when coupling is active) so the
    // owner can (re)build its per-step coupling fields before they are sampled.
    using FluidCouplingPrepare = std::function<void()>;

    void setFluidSampler(FluidSampler s) { fluid_sampler_ = std::move(s); }
    void setFluidCouplingPrepare(FluidCouplingPrepare p) { fluid_prepare_ = std::move(p); }

    void setGravity(const Vec3& g);

    // Drop all Jolt bodies and clear runtime caches. Simulation resets restore
    // dynamic objects to their captured rest pose first; authoring edits can skip
    // that restore so a user transform becomes the next rest pose.
    void resetRuntime(bool restore_rest_pose = true);

private:
    bool ensureWorld();
    void ensureBodyCreated(RigidBodyObject& rb);
    // Apply buoyancy + drag from the fluid onto one dynamic body (before update).
    void applyFluidCoupling(RigidBodyObject& rb, float dt);

    bool enabled_ = true;
    std::vector<RigidBodyObject>* bodies_ = nullptr;
    std::unique_ptr<JoltIntegration::JoltWorld> world_;
    Vec3 gravity_ = Vec3(0.0f, -9.81f, 0.0f);

    ShapeResolver shape_resolver_;
    PivotGetter pivot_getter_;
    PivotSetter pivot_setter_;
    FluidSampler fluid_sampler_;
    FluidCouplingPrepare fluid_prepare_;
};

} // namespace RayTrophiSim
