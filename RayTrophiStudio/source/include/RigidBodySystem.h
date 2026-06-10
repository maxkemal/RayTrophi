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
    Capsule = 2,
    Mesh = 3      // exact triangle mesh (static) / convex hull (dynamic) from the source mesh
};

enum class RigidBodyMotionType : uint8_t {
    Static = 0,
    Dynamic = 1,
    Kinematic = 2
};

// The kind of physics body a descriptor drives. Rigid uses the Jolt rigid path
// (motion_type + shape below). SoftBody / Cloth are deformable bodies that build
// a Jolt soft body from the source mesh; their authored parameters live in the
// `soft_*` block. Foundation note: the soft solver is not wired yet — non-Rigid
// kinds are skipped by RigidBodySystem (no Jolt body created) so they are inert
// but fully authored + serialized until the soft path lands.
enum class BodyKind : uint8_t {
    Rigid = 0,
    SoftBody = 1,
    Cloth = 2
};

// A world-space pin region for soft bodies / cloth. Every REST vertex inside the
// sphere is pinned (held at its rest world position) when the body is created.
// Authored with a gizmo; multiple regions can pin several edges/corners at once.
struct SoftPinRegion {
    Vec3 center = Vec3(0.0f, 0.0f, 0.0f);
    float radius = 0.25f;
    bool enabled = true;
};

struct RigidBodyObject {
    std::string name = "Rigid Body";
    std::string source_name;          // scene object nodeName this body drives
    std::string collider_name;        // authored collider descriptor used for Jolt shape/material
    bool enabled = true;
    BodyKind kind = BodyKind::Rigid;  // Rigid (Jolt rigid) vs SoftBody / Cloth (deformable)
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
    float fluid_drag = 1.0f;          // linear (viscous) drag ~ -k*v
    float fluid_quadratic_drag = 0.5f;// form/slam drag ~ -k*v*|v| (the water resistance that stops a body skipping off the surface)
    float fluid_angular_drag = 1.0f;
    // Speed cap (m/s) on the fluid velocity that drives drag. The grid velocity
    // near a plunging body is dominated by the splash the body itself stamps in;
    // feeding it back undamped flung light bodies sideways. Clamping (plus the
    // per-body temporal smoothing below) breaks that splash->drag->fling loop.
    // 0 disables the clamp.
    float fluid_max_coupling_speed = 4.0f;

    // ---- soft-body / cloth authoring (used when kind != Rigid) ---------------
    // Maps onto Jolt's SoftBodySharedSettings / SoftBodyCreationSettings when the
    // soft solver is wired. A soft body builds its particle/edge graph from the
    // source mesh; cloth is the same path restricted to a surface (no volume
    // constraints, two-sided collision). Ignored entirely by the rigid path.
    float soft_stiffness = 0.8f;        // edge constraint stiffness (0 = floppy, 1 = stiff)
    float soft_compliance = 0.0f;       // XPBD inverse stiffness; 0 = fully stiff (overrides nothing when 0)
    float soft_pressure = 0.0f;         // closed-volume inflation (balloons/soft solids); 0 = off (cloth)
    float soft_damping = 0.05f;         // velocity damping per step
    float soft_vertex_radius = 0.01f;   // per-vertex collision thickness (m)
    int   soft_iterations = 5;          // constraint solver iterations per step
    float soft_friction = 0.5f;
    float soft_restitution = 0.0f;
    float soft_gravity_factor = 1.0f;
    float soft_mass = 1.0f;             // total mass distributed over the mesh vertices (kg)
    bool  soft_two_sided = true;        // cloth: collide both faces
    // Cloth/soft pins: rest vertices inside any enabled region are held fixed.
    // Applied at create time (changing them forces a body rebuild). World-space.
    std::vector<SoftPinRegion> soft_pins;

    // ---- force-field coupling (all body kinds) -------------------------------
    // When enabled, the scene's force fields drive this body too (rigid: a force at
    // the COM; soft/cloth: a per-vertex velocity push, pinned vertices excluded).
    // Each field's affects_rigidbody / affects_cloth mask still gates it; this is a
    // per-body opt-out + an overall strength scale.
    bool  force_field_enabled = true;
    float force_field_scale = 1.0f;

    // ---- runtime state (not serialized) ----
    uint32_t handle = 0xffffffffu;    // JoltBodyHandle once created
    bool created = false;
    bool rest_captured = false;       // rest pose (initial_pivot/body_xf/half) is valid
    Matrix4x4 initial_body_xf = Matrix4x4::identity();  // B0: body world pose at creation
    Matrix4x4 initial_pivot = Matrix4x4::identity();    // P0: object REST pose (spawn point)
    Vec3 rest_half_extents = Vec3(0.5f, 0.5f, 0.5f);    // shape dims captured at rest (reused on shape/param change)
    Matrix4x4 last_written_pivot = Matrix4x4::identity(); // last pose the sim pushed onto the object
    bool has_written = false;                            // sim has pushed at least one pose
    // Temporally-smoothed fluid velocity used for drag (per-body average of the
    // submerged sample points, EMA-filtered + speed-clamped). Decouples drag from
    // the noisy per-frame splash field so the body is not flung. Runtime only.
    Vec3 smoothed_fluid_vel = Vec3(0.0f, 0.0f, 0.0f);
    bool fluid_vel_primed = false;                       // EMA has a seeded value

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
    int   dbg_pinned_count = 0;         // soft/cloth vertices pinned at last create
};

// One dynamic body's full kinematic state at a baked frame. Captured during the
// live (fluid-coupled) bake and replayed verbatim so a cached frame shows the
// rigid in EXACTLY the pose it held when that fluid frame was computed — instead
// of re-simulating it against a frozen fluid frame, which diverges. Matched back
// to its body by source_name (survives vector reordering).
struct RigidBodyFrameState {
    std::string source_name;
    Matrix4x4 pivot   = Matrix4x4::identity();  // pose written onto the source object
    Matrix4x4 body_xf = Matrix4x4::identity();  // Jolt body world pose (for seamless resume)
    Vec3 lin_vel = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 ang_vel = Vec3(0.0f, 0.0f, 0.0f);
    bool valid = false;                          // a pose was actually written this frame
};

// One impact between two bodies this step, surfaced from Jolt's contact listener
// and translated to scene-object names. Either source name is empty when that
// body is not a tracked RigidBodyObject (e.g. an untracked static collider). The
// impulse is an estimate (reduced mass * closing speed) suitable for thresholding
// a "hard hit" — the trigger the fracture stage (Faz 3) will consume.
struct RigidContactEvent {
    std::string source_a;   // scene node of body A (empty if untracked)
    std::string source_b;   // scene node of body B (empty if untracked)
    Vec3  point  = Vec3(0.0f, 0.0f, 0.0f);
    Vec3  normal = Vec3(0.0f, 0.0f, 0.0f);
    float closing_speed = 0.0f;
    float impulse = 0.0f;
    bool  is_new = false;   // first touch this contact (vs persisted)
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

    // Render write-back for rigid bodies: instead of moving the source object's
    // TRANSFORM handle (which corrupts imported/non-TRS meshes in the renderer — the
    // object grew / Y-flipped / lost geometry from frame 0), bake the body's
    // world-space rigid motion delta = B(t)*inv(B0) directly into the mesh vertices,
    // exactly like soft bodies do (which render imported meshes correctly). The mesh
    // is moved rigidly so the owner can preserve the authored per-corner normals.
    // delta == identity restores the rest mesh (used on reset).
    using RigidMeshBaker = std::function<void(const std::string& node, const Matrix4x4& world_delta)>;
    void setRigidMeshBaker(RigidMeshBaker b) { rigid_baker_ = std::move(b); }

    // ---- soft-body geometry I/O (kind != Rigid) -----------------------------
    // resolver(): build the welded REST mesh for a soft body — world-space rest
    // vertices + a triangle index list (3 per face). Return false if the source
    // mesh isn't available yet. Called once when the body is created.
    using SoftMeshResolver = std::function<bool(const RigidBodyObject& body,
                                                std::vector<Vec3>& out_vertices,
                                                std::vector<uint32_t>& out_indices)>;
    // writer(): push the simulated, deformed world-space vertices back onto the
    // scene mesh (and flag a geometry rebuild). Same vertex order/count as the
    // resolver produced. Called every step for each live soft body.
    using SoftMeshWriter = std::function<void(const std::string& node,
                                              const std::vector<Vec3>& world_vertices)>;
    // resetToRest(): restore the source mesh to its undeformed rest pose (called on
    // resetRuntime so a replay/respawn starts from the clean shape).
    using SoftMeshResetToRest = std::function<void(const std::string& node)>;

    void setSoftMeshResolver(SoftMeshResolver r) { soft_resolver_ = std::move(r); }
    void setSoftMeshWriter(SoftMeshWriter w) { soft_writer_ = std::move(w); }
    void setSoftMeshResetToRest(SoftMeshResetToRest r) { soft_reset_ = std::move(r); }

    // ---- Rigid-fluid coupling (buoyancy + drag) -----------------------------
    // Result of querying the fluid state at a world point. signed_distance is the
    // point's height relative to the fluid FREE SURFACE in its column (< 0 =>
    // below the surface, |value| ≈ depth) — measured against where the fluid
    // surface is, NOT whether particles happen to occupy this point, so a sunk
    // body that has displaced the fluid out of its own cells still reads as
    // submerged. `velocity` is the fluid velocity for drag, already gated to 0
    // where no genuine fluid is present (body cavity / air).
    struct FluidSample {
        float signed_distance = 1.0e30f;  // <0 below free surface (world units)
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

    // ---- Contact / impact events (foundation for fracture) ------------------
    // Off by default (zero cost in Jolt). When enabled, each step() drains the
    // contact listener and translates body handles to scene-node names; read the
    // result with contactEvents() AFTER the sim has stepped. Cleared and refilled
    // every step.
    void setContactEventsEnabled(bool e);
    bool contactEventsEnabled() const { return contact_events_enabled_; }
    const std::vector<RigidContactEvent>& contactEvents() const { return contact_events_; }

    // Drop all Jolt bodies and clear runtime caches. Simulation resets restore
    // dynamic objects to their captured rest pose first; authoring edits can skip
    // that restore so a user transform becomes the next rest pose.
    void resetRuntime(bool restore_rest_pose = true);

    // Destroy ONLY the Jolt body driving `node` (leaving every other body live and
    // mid-simulation untouched) and clear its runtime flags. Used by "Apply at
    // current frame": the caller freezes the object's current deformed mesh and
    // drops the descriptor, so this body must leave the Jolt world without the
    // global rest-restore resetRuntime() would do. Returns true if a body matched.
    bool destroyBodyForNode(const std::string& node);

    // ---- frame cache (replay alongside the fluid cache) ---------------------
    // Snapshot every dynamic body's kinematic state (pose + velocities). Cheap;
    // call it in lockstep with the fluid frame capture.
    void captureFrameState(std::vector<RigidBodyFrameState>& out) const;
    // Restore a captured snapshot: writes each pose onto its source object and
    // (when the Jolt bodies are live) sets their pose + velocities so a forward
    // resume from this frame continues smoothly. Creates bodies if needed.
    // Returns false only when the system can't run at all.
    bool restoreFrameState(const std::vector<RigidBodyFrameState>& in);

private:
    bool ensureWorld();
    void ensureBodyCreated(RigidBodyObject& rb);
    void ensureSoftBodyCreated(RigidBodyObject& rb);  // kind != Rigid path
    // Apply buoyancy + drag from the fluid onto one dynamic body (before update).
    void applyFluidCoupling(RigidBodyObject& rb, float dt);
    // Apply the scene's force-field snapshot onto every body (before update): a COM
    // force for rigids, a per-vertex velocity push for soft/cloth. No-op when the
    // snapshot is empty or null.
    void applyForceFields(const SimulationContext& ctx, float dt);

    bool enabled_ = true;
    std::vector<RigidBodyObject>* bodies_ = nullptr;
    std::unique_ptr<JoltIntegration::JoltWorld> world_;
    Vec3 gravity_ = Vec3(0.0f, -9.81f, 0.0f);

    ShapeResolver shape_resolver_;
    PivotGetter pivot_getter_;
    PivotSetter pivot_setter_;
    RigidMeshBaker rigid_baker_;
    SoftMeshResolver soft_resolver_;
    SoftMeshWriter soft_writer_;
    SoftMeshResetToRest soft_reset_;
    FluidSampler fluid_sampler_;
    FluidCouplingPrepare fluid_prepare_;

    bool contact_events_enabled_ = false;
    std::vector<RigidContactEvent> contact_events_;          // refilled each step()
};

} // namespace RayTrophiSim
