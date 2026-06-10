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
#include <vector>

#include "Vec3.h"
#include "Matrix4x4.h"

namespace RayTrophiSim {
namespace JoltIntegration {

using JoltBodyHandle = uint32_t;
static constexpr JoltBodyHandle kInvalidBody = 0xffffffffu;

enum class JoltShapeType {
    Box,        // half_extents
    Sphere,     // radius
    Capsule,    // radius + half_height (cylinder half-length, excludes caps)
    Mesh        // exact triangle mesh (mesh_vertices + mesh_indices, world space)
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

    // Mesh shape (JoltShapeType::Mesh): WORLD-space triangle soup. Vertices are the
    // object's rest world positions; indices are a triangle list (3 per face) into
    // them. A STATIC body builds an exact triangle MeshShape (the true mesh boundary,
    // not an OBB); a Dynamic/Kinematic body builds a ConvexHull from the same points
    // (Jolt cannot simulate a concave mesh as a moving body). The body is created at
    // identity, so the rest world pose IS the shape (mirrors the soft-body path).
    std::vector<Vec3>     mesh_vertices;
    std::vector<uint32_t> mesh_indices;

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

// Description for a deformable soft body built from a triangle mesh. Vertices and
// indices are WORLD-space at rest (the body is created at the origin with identity
// rotation, so the rest pose IS the world pose). Edges/bend constraints are derived
// automatically from the faces. A closed mesh + pressure > 0 inflates (balloons /
// soft solids); pressure 0 + two_sided is cloth.
struct JoltSoftBodyDesc {
    std::vector<Vec3>     vertices;     // welded rest positions (world space)
    std::vector<uint32_t> indices;      // triangle list, 3 indices per face, into `vertices`
    float total_mass = 1.0f;           // distributed evenly over the vertices (kg)
    float compliance = 0.0f;           // edge inverse-stiffness (0 = stiff)
    float pressure = 0.0f;             // closed-volume inflation; 0 = off (cloth)
    float damping = 0.1f;              // linear velocity damping
    int   num_iterations = 5;          // solver iterations per step
    float friction = 0.2f;
    float restitution = 0.0f;
    float gravity_factor = 1.0f;
    float vertex_radius = 0.0f;        // collision thickness per vertex (m)
    bool  two_sided = false;           // treat faces as double-sided for collision
    // Per-vertex pin flags (parallel to `vertices`; empty / shorter => those
    // vertices are free). A pinned vertex gets infinite mass (invMass = 0) so it
    // stays fixed at its rest world position — used to hang cloth from corners /
    // an edge.
    std::vector<uint8_t> vertex_pinned;
};

// One captured collision contact (drained after each update). The impulse is an
// ESTIMATE computed at contact time from the pre-solve relative velocity and the
// pair's reduced mass — enough to threshold "hard hit" events (e.g. trigger
// fracture). Bodies are sorted body_a.id < body_b.id (Jolt convention), so
// body_a may be the static one.
struct ContactEvent {
    JoltBodyHandle body_a = kInvalidBody;
    JoltBodyHandle body_b = kInvalidBody;
    Vec3 point = Vec3(0.0f, 0.0f, 0.0f);    // world-space contact point
    Vec3 normal = Vec3(0.0f, 0.0f, 0.0f);   // world normal, points from body_a toward body_b
    float closing_speed = 0.0f;             // approach speed along the normal (m/s, >0 = approaching)
    float impulse = 0.0f;                   // est. normal impulse = reduced_mass * closing_speed (kg·m/s)
    bool is_new = false;                    // true: first touch (OnContactAdded); false: persisted
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
    // Create a deformable soft body from a triangle mesh (see JoltSoftBodyDesc).
    // Returns kInvalidBody on empty/degenerate input. Soft bodies live in the same
    // world as rigid bodies and collide with them automatically.
    JoltBodyHandle createSoftBody(const JoltSoftBodyDesc& desc);
    // Read back the deformed vertex positions (world space) of a soft body, in the
    // same order as the JoltSoftBodyDesc::vertices used to create it. Returns false
    // if the handle is not a live soft body. `out` is resized to the vertex count.
    bool getSoftBodyVertices(JoltBodyHandle handle, std::vector<Vec3>& out) const;
    // Read back the per-vertex WORLD-space velocities of a soft body, same order as
    // createSoftBody's vertices. Returns false if the handle is not a live soft body.
    // Used to feed the relative velocity into force fields (e.g. drag) per vertex.
    bool getSoftBodyVertexVelocities(JoltBodyHandle handle, std::vector<Vec3>& out) const;
    // Add a per-vertex WORLD-space velocity delta to a soft body (dv[i] added to
    // vertex i). Pinned vertices (invMass == 0) are skipped so a force field can't
    // tear them off their pin. `dv` shorter than the vertex count leaves the rest
    // untouched. Used to drive soft bodies / cloth with external force fields.
    void addSoftBodyVertexVelocities(JoltBodyHandle handle, const std::vector<Vec3>& dv);
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
    void setBodyAngularVelocity(JoltBodyHandle handle, const Vec3& w);
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

    // ---- Contact events (impact detection; foundation for fracture) ----------
    // Off by default (zero cost). Enable when a consumer needs impact data; the
    // contact listener then accumulates events across update() from the physics
    // job threads. Drain them on the main thread AFTER update() returns. Only
    // approaching contacts above a small speed threshold are captured.
    void setContactCaptureEnabled(bool enabled);
    bool contactCaptureEnabled() const;
    void drainContactEvents(std::vector<ContactEvent>& out);  // swaps out + clears the buffer

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace JoltIntegration
} // namespace RayTrophiSim
