#include "JoltWorld.h"

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>
#include <Jolt/Physics/SoftBody/SoftBodyCreationSettings.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>

#include "JoltTypes.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

JPH_SUPPRESS_WARNINGS

namespace RayTrophiSim {
namespace JoltIntegration {

namespace {

// ---- Layers --------------------------------------------------------------
namespace Layers {
static constexpr JPH::ObjectLayer NON_MOVING = 0;
static constexpr JPH::ObjectLayer MOVING = 1;
static constexpr JPH::ObjectLayer NUM_LAYERS = 2;
}

namespace BroadPhaseLayers {
static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
static constexpr JPH::BroadPhaseLayer MOVING(1);
static constexpr JPH::uint NUM_LAYERS(2);
}

class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
public:
    JPH::uint GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::NUM_LAYERS; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override {
        return (layer == Layers::NON_MOVING) ? BroadPhaseLayers::NON_MOVING
                                             : BroadPhaseLayers::MOVING;
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer) const override { return "JoltWorld"; }
#endif
};

class ObjectVsBroadPhaseLayerFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer layer1, JPH::BroadPhaseLayer layer2) const override {
        if (layer1 == Layers::NON_MOVING) return layer2 == BroadPhaseLayers::MOVING;
        return true;
    }
};

class ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        if (a == Layers::NON_MOVING) return b == Layers::MOVING;
        return true;
    }
};

// ---- Contact capture listener --------------------------------------------
// Accumulates impact events from the physics job threads during Update. Reading
// bodies is allowed inside the callbacks (they are locked for read); we must NOT
// touch the BodyInterface here. Events are drained on the main thread after the
// step. The impulse is estimated from the PRE-SOLVE relative velocity (Jolt has
// not run the constraint solver yet at OnContactAdded) and the reduced mass.
class ContactCaptureListener final : public JPH::ContactListener {
public:
    void setEnabled(bool e) { enabled_.store(e, std::memory_order_relaxed); }
    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

    void drain(std::vector<ContactEvent>& out) {
        std::lock_guard<std::mutex> lk(mutex_);
        out.swap(events_);
        events_.clear();
    }
    void clear() {
        std::lock_guard<std::mutex> lk(mutex_);
        events_.clear();
    }

    void OnContactAdded(const JPH::Body& b1, const JPH::Body& b2,
                        const JPH::ContactManifold& m, JPH::ContactSettings&) override {
        capture(b1, b2, m, /*is_new=*/true);
    }
    void OnContactPersisted(const JPH::Body& b1, const JPH::Body& b2,
                            const JPH::ContactManifold& m, JPH::ContactSettings&) override {
        capture(b1, b2, m, /*is_new=*/false);
    }

private:
    void capture(const JPH::Body& b1, const JPH::Body& b2,
                 const JPH::ContactManifold& m, bool is_new) {
        if (!enabled_.load(std::memory_order_relaxed)) return;
        if (m.mRelativeContactPointsOn1.empty()) return;

        const JPH::RVec3 wp = m.GetWorldSpaceContactPointOn1(0);
        const JPH::Vec3 n = m.mWorldSpaceNormal;  // points from shape 1 toward shape 2
        // Pre-solve relative velocity at the contact point. >0 along n => the two
        // bodies are approaching (a real impact), <=0 => resting / separating.
        const JPH::Vec3 v_rel = b1.GetPointVelocity(wp) - b2.GetPointVelocity(wp);
        const float closing = v_rel.Dot(n);
        if (closing < kMinClosingSpeed) return;

        // Only DYNAMIC bodies have finite mass; static AND kinematic act as
        // infinite mass (inverse 0). GetInverseMass() asserts Dynamic, so guard on
        // IsDynamic() and use the unchecked accessor.
        const float inv_m1 = b1.IsDynamic() ? b1.GetMotionPropertiesUnchecked()->GetInverseMassUnchecked() : 0.0f;
        const float inv_m2 = b2.IsDynamic() ? b2.GetMotionPropertiesUnchecked()->GetInverseMassUnchecked() : 0.0f;
        const float inv_sum = inv_m1 + inv_m2;
        const float reduced_mass = (inv_sum > 0.0f) ? (1.0f / inv_sum) : 0.0f;

        ContactEvent e;
        e.body_a = b1.GetID().GetIndexAndSequenceNumber();
        e.body_b = b2.GetID().GetIndexAndSequenceNumber();
        e.point = toRT(wp);
        e.normal = toRT(n);
        e.closing_speed = closing;
        e.impulse = reduced_mass * closing;
        e.is_new = is_new;

        std::lock_guard<std::mutex> lk(mutex_);
        if (events_.size() < kMaxEvents) events_.push_back(e);
    }

    static constexpr float kMinClosingSpeed = 0.25f;  // m/s; ignore resting jitter
    static constexpr std::size_t kMaxEvents = 8192;   // hard cap (busy scenes)

    std::atomic<bool> enabled_{ false };
    std::mutex mutex_;
    std::vector<ContactEvent> events_;
};

// ---- Process-wide Jolt registration (refcounted) -------------------------
std::mutex g_jolt_reg_mutex;
int g_jolt_refcount = 0;

void TraceImpl(const char* fmt, ...) {
    va_list list;
    va_start(list, fmt);
    char buffer[1024];
    std::vsnprintf(buffer, sizeof(buffer), fmt, list);
    va_end(list);
    std::printf("[Jolt] %s\n", buffer);
}

#ifdef JPH_ENABLE_ASSERTS
bool AssertFailedImpl(const char* expr, const char* msg, const char* file, JPH::uint line) {
    std::printf("[Jolt][ASSERT] %s:%u: (%s) %s\n", file, line, expr, msg ? msg : "");
    return true; // trigger a breakpoint in the debugger
}
#endif

void ensureJoltRegistered() {
    std::lock_guard<std::mutex> lock(g_jolt_reg_mutex);
    if (g_jolt_refcount++ == 0) {
        JPH::RegisterDefaultAllocator();
        JPH::Trace = TraceImpl;
        JPH_IF_ENABLE_ASSERTS(JPH::AssertFailed = AssertFailedImpl;)
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    }
}

void releaseJoltRegistered() {
    std::lock_guard<std::mutex> lock(g_jolt_reg_mutex);
    if (g_jolt_refcount > 0 && --g_jolt_refcount == 0) {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
    }
}

inline JPH::BodyID toBodyID(JoltBodyHandle h) { return JPH::BodyID(h); }

} // namespace

// ---- Impl ----------------------------------------------------------------
struct JoltWorld::Impl {
    bool initialized = false;

    std::unique_ptr<JPH::TempAllocatorImpl> temp_allocator;
    std::unique_ptr<JPH::JobSystemThreadPool> job_system;

    // Filters MUST outlive physics_system (PhysicsSystem::Init stores refs to
    // them), so declare them before physics_system => destructed AFTER it.
    BPLayerInterfaceImpl bp_layer_interface;
    ObjectVsBroadPhaseLayerFilterImpl obj_vs_bp_filter;
    ObjectLayerPairFilterImpl obj_layer_pair_filter;
    // Must outlive physics_system (SetContactListener stores a raw pointer), so
    // declare it BEFORE physics_system => destructed AFTER it.
    ContactCaptureListener contact_listener;

    std::unique_ptr<JPH::PhysicsSystem> physics_system;
    std::vector<JPH::BodyID> bodies;
    // Soft bodies hold a reference to their shared settings; keep the refs alive for
    // the world's lifetime so the simulating bodies never dangle.
    std::vector<JPH::Ref<JPH::SoftBodySharedSettings>> soft_settings;
};

JoltWorld::JoltWorld() : impl_(std::make_unique<Impl>()) {}

JoltWorld::~JoltWorld() { shutdown(); }

bool JoltWorld::init(const Config& cfg) {
    if (impl_->initialized) return true;

    ensureJoltRegistered();

    impl_->temp_allocator = std::make_unique<JPH::TempAllocatorImpl>(16 * 1024 * 1024);

    const int threads = (cfg.num_threads >= 0)
                            ? cfg.num_threads
                            : std::max(1, (int)std::thread::hardware_concurrency() - 1);
    impl_->job_system = std::make_unique<JPH::JobSystemThreadPool>(
        JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, threads);

    impl_->physics_system = std::make_unique<JPH::PhysicsSystem>();
    impl_->physics_system->Init(cfg.max_bodies, 0, cfg.max_body_pairs, cfg.max_contact_constraints,
                                impl_->bp_layer_interface, impl_->obj_vs_bp_filter,
                                impl_->obj_layer_pair_filter);
    impl_->physics_system->SetGravity(toJolt(cfg.gravity));
    impl_->physics_system->SetContactListener(&impl_->contact_listener);

    impl_->initialized = true;
    return true;
}

void JoltWorld::shutdown() {
    if (!impl_->initialized) return;

    clearBodies();
    impl_->physics_system.reset();   // destroy before releasing Factory
    impl_->job_system.reset();
    impl_->temp_allocator.reset();
    impl_->initialized = false;

    releaseJoltRegistered();
}

bool JoltWorld::isInitialized() const { return impl_->initialized; }

JoltBodyHandle JoltWorld::createBody(const JoltBodyDesc& desc) {
    if (!impl_->initialized) return kInvalidBody;

    // Resolve the effective motion type up front: a Mesh shape needs it to decide
    // between an exact triangle MeshShape (static only) and a ConvexHull (moving).
    const JoltMotionType effective_motion =
        (!desc.dynamic && desc.motion_type == JoltMotionType::Dynamic)
            ? JoltMotionType::Static
            : desc.motion_type;
    const bool is_dynamic = effective_motion == JoltMotionType::Dynamic;
    const bool is_kinematic = effective_motion == JoltMotionType::Kinematic;

    JPH::ShapeRefC shape;
    switch (desc.shape) {
        case JoltShapeType::Box: {
            JPH::Vec3 he = toJolt(desc.half_extents);
            float min_extent = std::min({ he.GetX(), he.GetY(), he.GetZ() });
            float convex_radius = std::min(JPH::cDefaultConvexRadius, 0.5f * std::max(0.0f, min_extent));
            JPH::BoxShapeSettings s(he, convex_radius);
            s.SetEmbedded();
            JPH::ShapeSettings::ShapeResult r = s.Create();
            if (r.HasError()) return kInvalidBody;
            shape = r.Get();
            break;
        }
        case JoltShapeType::Sphere: {
            JPH::SphereShapeSettings s(std::max(1e-4f, desc.radius));
            s.SetEmbedded();
            JPH::ShapeSettings::ShapeResult r = s.Create();
            if (r.HasError()) return kInvalidBody;
            shape = r.Get();
            break;
        }
        case JoltShapeType::Capsule: {
            JPH::CapsuleShapeSettings s(std::max(1e-4f, desc.half_height), std::max(1e-4f, desc.radius));
            s.SetEmbedded();
            JPH::ShapeSettings::ShapeResult r = s.Create();
            if (r.HasError()) return kInvalidBody;
            shape = r.Get();
            break;
        }
        case JoltShapeType::Mesh: {
            if (desc.mesh_vertices.size() < 3 || desc.mesh_indices.size() < 3)
                return kInvalidBody;
            const JPH::uint vcount = (JPH::uint)desc.mesh_vertices.size();
            if (effective_motion == JoltMotionType::Static) {
                // Static: exact triangle MeshShape — the true mesh boundary (rocks,
                // terrain, hulls), not an OBB. Jolt forbids MeshShape on a moving
                // body, so this branch is static-only.
                JPH::VertexList verts;
                verts.reserve(desc.mesh_vertices.size());
                for (const Vec3& v : desc.mesh_vertices)
                    verts.push_back(JPH::Float3(v.x, v.y, v.z));
                JPH::IndexedTriangleList tris;
                tris.reserve(desc.mesh_indices.size() / 3);
                for (std::size_t t = 0; t + 2 < desc.mesh_indices.size(); t += 3) {
                    const JPH::uint a = desc.mesh_indices[t + 0];
                    const JPH::uint b = desc.mesh_indices[t + 1];
                    const JPH::uint c = desc.mesh_indices[t + 2];
                    if (a >= vcount || b >= vcount || c >= vcount) continue;
                    if (a == b || b == c || a == c) continue;
                    tris.push_back(JPH::IndexedTriangle(a, b, c, 0));
                }
                if (tris.empty()) return kInvalidBody;
                JPH::MeshShapeSettings s(std::move(verts), std::move(tris));
                s.SetEmbedded();
                JPH::ShapeSettings::ShapeResult r = s.Create();
                if (r.HasError()) return kInvalidBody;
                shape = r.Get();
            } else {
                // Dynamic / Kinematic: a concave mesh can't move in Jolt, so build a
                // ConvexHull from the same points (interior points are discarded by
                // the hull builder). Tighter than an OBB; concave cavities are filled.
                JPH::Array<JPH::Vec3> points;
                points.reserve(desc.mesh_vertices.size());
                for (const Vec3& v : desc.mesh_vertices)
                    points.push_back(JPH::Vec3(v.x, v.y, v.z));
                JPH::ConvexHullShapeSettings s(points, JPH::cDefaultConvexRadius);
                s.SetEmbedded();
                JPH::ShapeSettings::ShapeResult r = s.Create();
                if (r.HasError()) return kInvalidBody;
                shape = r.Get();
            }
            break;
        }
        default:
            return kInvalidBody;
    }

    JPH::RVec3 pos;
    JPH::Quat rot;
    decomposeToJolt(desc.transform, pos, rot);

    const JPH::EMotionType motion =
        is_dynamic ? JPH::EMotionType::Dynamic :
        (is_kinematic ? JPH::EMotionType::Kinematic : JPH::EMotionType::Static);
    const JPH::ObjectLayer layer = (is_dynamic || is_kinematic) ? Layers::MOVING : Layers::NON_MOVING;

    JPH::BodyCreationSettings settings(shape, pos, rot, motion, layer);
    settings.mFriction = desc.friction;
    settings.mRestitution = desc.restitution;
    settings.mAllowSleeping = desc.sleep_enabled;
    settings.mLinearDamping = desc.linear_damping;
    settings.mAngularDamping = desc.angular_damping;
    settings.mGravityFactor = desc.gravity_scale;
    if (is_dynamic || is_kinematic) {
        settings.mLinearVelocity = toJolt(desc.initial_linear_velocity);
        settings.mAngularVelocity = toJolt(desc.initial_angular_velocity);
    }
    if (is_dynamic) {
        if (desc.mass > 0.0f) {
            settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            settings.mMassPropertiesOverride.mMass = desc.mass;
        }
    }

    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    const JPH::EActivation act = ((is_dynamic || is_kinematic) && desc.start_active)
                                     ? JPH::EActivation::Activate
                                     : JPH::EActivation::DontActivate;
    JPH::BodyID id = bi.CreateAndAddBody(settings, act);
    if (id.IsInvalid()) return kInvalidBody;

    impl_->bodies.push_back(id);
    return id.GetIndexAndSequenceNumber();
}

JoltBodyHandle JoltWorld::createSoftBody(const JoltSoftBodyDesc& desc) {
    if (!impl_->initialized) return kInvalidBody;
    if (desc.vertices.empty() || desc.indices.size() < 3) return kInvalidBody;

    JPH::Ref<JPH::SoftBodySharedSettings> settings = new JPH::SoftBodySharedSettings();

    // Vertices: distribute the total mass evenly (inverse mass per particle).
    const float per_vertex_inv_mass =
        (desc.total_mass > 0.0f)
            ? (float)desc.vertices.size() / desc.total_mass
            : 1.0f;
    settings->mVertices.reserve(desc.vertices.size());
    for (std::size_t vi = 0; vi < desc.vertices.size(); ++vi) {
        const Vec3& v = desc.vertices[vi];
        JPH::SoftBodySharedSettings::Vertex sv;
        sv.mPosition = JPH::Float3(v.x, v.y, v.z);
        sv.mVelocity = JPH::Float3(0.0f, 0.0f, 0.0f);
        // Pinned vertices get infinite mass (invMass = 0) so they stay fixed at
        // their rest world position — the cloth/soft body hangs from them.
        const bool pinned = (vi < desc.vertex_pinned.size()) && desc.vertex_pinned[vi] != 0;
        sv.mInvMass = pinned ? 0.0f : per_vertex_inv_mass;
        settings->mVertices.push_back(sv);
    }

    // Faces from the triangle list (skip degenerate / out-of-range).
    const uint32_t vcount = (uint32_t)desc.vertices.size();
    for (std::size_t t = 0; t + 2 < desc.indices.size(); t += 3) {
        const uint32_t a = desc.indices[t + 0];
        const uint32_t b = desc.indices[t + 1];
        const uint32_t c = desc.indices[t + 2];
        if (a >= vcount || b >= vcount || c >= vcount) continue;
        if (a == b || b == c || a == c) continue;
        settings->AddFace(JPH::SoftBodySharedSettings::Face(a, b, c));
    }
    if (settings->mFaces.empty()) return kInvalidBody;

    // Auto-build edge (and distance-bend) constraints from the faces. A single
    // VertexAttributes is repeated for every vertex (compliance = inverse stiffness).
    JPH::SoftBodySharedSettings::VertexAttributes va(
        /*compliance=*/desc.compliance,
        /*shearCompliance=*/desc.compliance,
        /*bendCompliance=*/desc.compliance);
    settings->CreateConstraints(&va, 1, JPH::SoftBodySharedSettings::EBendType::Distance);
    settings->Optimize();

    // Vertices are already world-space; create at the origin with identity rotation.
    JPH::SoftBodyCreationSettings cs(settings, JPH::RVec3::sZero(), JPH::Quat::sIdentity(),
                                     Layers::MOVING);
    cs.mNumIterations = (uint32_t)std::max(1, desc.num_iterations);
    cs.mLinearDamping = desc.damping;
    cs.mPressure = desc.pressure;
    cs.mFriction = desc.friction;
    cs.mRestitution = desc.restitution;
    cs.mGravityFactor = desc.gravity_factor;
    cs.mVertexRadius = desc.vertex_radius;
    cs.mFacesDoubleSided = desc.two_sided;
    cs.mUpdatePosition = true;
    cs.mMakeRotationIdentity = false;  // keep our world-space rest pose untouched

    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    JPH::BodyID id = bi.CreateAndAddSoftBody(cs, JPH::EActivation::Activate);
    if (id.IsInvalid()) return kInvalidBody;

    impl_->bodies.push_back(id);
    impl_->soft_settings.push_back(settings);  // keep the shared settings alive
    return id.GetIndexAndSequenceNumber();
}

bool JoltWorld::getSoftBodyVertices(JoltBodyHandle handle, std::vector<Vec3>& out) const {
    out.clear();
    if (!impl_->initialized || handle == kInvalidBody) return false;
    JPH::BodyID id = toBodyID(handle);
    JPH::BodyLockRead lock(impl_->physics_system->GetBodyLockInterface(), id);
    if (!lock.Succeeded()) return false;
    const JPH::Body& body = lock.GetBody();
    if (!body.IsSoftBody()) return false;

    const JPH::SoftBodyMotionProperties* mp =
        static_cast<const JPH::SoftBodyMotionProperties*>(body.GetMotionProperties());
    const JPH::RMat44 com = body.GetCenterOfMassTransform();
    const auto& verts = mp->GetVertices();
    out.reserve(verts.size());
    for (const auto& v : verts) {
        // mPosition is relative to the body's center of mass.
        out.push_back(toRT(com * v.mPosition));
    }
    return true;
}

bool JoltWorld::getSoftBodyVertexVelocities(JoltBodyHandle handle, std::vector<Vec3>& out) const {
    out.clear();
    if (!impl_->initialized || handle == kInvalidBody) return false;
    JPH::BodyID id = toBodyID(handle);
    JPH::BodyLockRead lock(impl_->physics_system->GetBodyLockInterface(), id);
    if (!lock.Succeeded()) return false;
    const JPH::Body& body = lock.GetBody();
    if (!body.IsSoftBody()) return false;

    const JPH::SoftBodyMotionProperties* mp =
        static_cast<const JPH::SoftBodyMotionProperties*>(body.GetMotionProperties());
    const JPH::RMat44 com = body.GetCenterOfMassTransform();
    const auto& verts = mp->GetVertices();
    out.reserve(verts.size());
    for (const auto& v : verts) {
        // mVelocity is stored in the body's COM-local frame; rotate (no translate)
        // into world to match getSoftBodyVertices().
        out.push_back(toRT(com.Multiply3x3(v.mVelocity)));
    }
    return true;
}

void JoltWorld::addSoftBodyVertexVelocities(JoltBodyHandle handle, const std::vector<Vec3>& dv) {
    if (!impl_->initialized || handle == kInvalidBody || dv.empty()) return;
    JPH::BodyID id = toBodyID(handle);
    bool any = false;
    {
        JPH::BodyLockWrite lock(impl_->physics_system->GetBodyLockInterface(), id);
        if (!lock.Succeeded()) return;
        JPH::Body& body = lock.GetBody();
        if (!body.IsSoftBody()) return;
        JPH::SoftBodyMotionProperties* mp =
            static_cast<JPH::SoftBodyMotionProperties*>(body.GetMotionProperties());
        const JPH::RMat44 com = body.GetCenterOfMassTransform();
        auto& verts = mp->GetVertices();
        const std::size_t n = std::min(verts.size(), dv.size());
        for (std::size_t i = 0; i < n; ++i) {
            auto& vert = verts[i];
            if (vert.mInvMass <= 0.0f) continue;  // pinned: never push
            // World delta -> COM-local frame (transpose of the rotation).
            vert.mVelocity += com.Multiply3x3Transposed(toJolt(dv[i]));
            any = true;
        }
    }
    // Wake a settled cloth so a force field that starts mid-rest still moves it.
    if (any) impl_->physics_system->GetBodyInterface().ActivateBody(id);
}

void JoltWorld::setSoftBodyVertices(JoltBodyHandle handle, const std::vector<Vec3>& positions,
                                    const std::vector<Vec3>* velocities) {
    if (!impl_->initialized || handle == kInvalidBody || positions.empty()) return;
    JPH::BodyID id = toBodyID(handle);
    {
        JPH::BodyLockWrite lock(impl_->physics_system->GetBodyLockInterface(), id);
        if (!lock.Succeeded()) return;
        JPH::Body& body = lock.GetBody();
        if (!body.IsSoftBody()) return;
        JPH::SoftBodyMotionProperties* mp =
            static_cast<JPH::SoftBodyMotionProperties*>(body.GetMotionProperties());
        // Particles are stored relative to the body's center of mass; map the target
        // WORLD positions back into that frame (inverse of getSoftBodyVertices' com*).
        const JPH::RMat44 com = body.GetCenterOfMassTransform();
        const JPH::RMat44 inv_com = com.InversedRotationTranslation();
        auto& verts = mp->GetVertices();
        const std::size_t n = std::min(verts.size(), positions.size());
        const bool have_vel = velocities && velocities->size() >= n;
        for (std::size_t i = 0; i < n; ++i) {
            if (verts[i].mInvMass <= 0.0f) continue;  // pinned: leave on its pin
            const JPH::Vec3 local_pos = JPH::Vec3(inv_com * toJoltR(positions[i]));
            verts[i].mPosition = local_pos;
            // Match previous position so the solver doesn't infer a spurious one-step
            // velocity from the teleport; the explicit velocity below is authoritative.
            verts[i].mPreviousPosition = local_pos;
            verts[i].mVelocity = have_vel
                ? com.Multiply3x3Transposed(toJolt((*velocities)[i]))
                : JPH::Vec3::sZero();
        }
    }
    impl_->physics_system->GetBodyInterface().ActivateBody(id);
}

void JoltWorld::removeBody(JoltBodyHandle handle) {
    if (!impl_->initialized || handle == kInvalidBody) return;
    JPH::BodyID id = toBodyID(handle);
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    bi.RemoveBody(id);
    bi.DestroyBody(id);
    impl_->bodies.erase(std::remove(impl_->bodies.begin(), impl_->bodies.end(), id),
                        impl_->bodies.end());
}

void JoltWorld::clearBodies() {
    if (!impl_->initialized) return;
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    for (JPH::BodyID id : impl_->bodies) {
        bi.RemoveBody(id);
        bi.DestroyBody(id);
    }
    impl_->bodies.clear();
    impl_->soft_settings.clear();  // release shared settings once no body references them
}

std::size_t JoltWorld::bodyCount() const { return impl_->bodies.size(); }

void JoltWorld::update(float dt, int collision_steps) {
    if (!impl_->initialized || dt <= 0.0f) return;
    impl_->physics_system->Update(dt, std::max(1, collision_steps),
                                  impl_->temp_allocator.get(), impl_->job_system.get());
}

void JoltWorld::optimizeBroadPhase() {
    if (!impl_->initialized) return;
    impl_->physics_system->OptimizeBroadPhase();
}

void JoltWorld::setGravity(const Vec3& g) {
    if (!impl_->initialized) return;
    impl_->physics_system->SetGravity(toJolt(g));
}

Matrix4x4 JoltWorld::getBodyTransform(JoltBodyHandle handle) const {
    if (!impl_->initialized || handle == kInvalidBody) return Matrix4x4::identity();
    JPH::BodyID id = toBodyID(handle);
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    JPH::RVec3 pos = bi.GetPosition(id);
    JPH::Quat rot = bi.GetRotation(id);
    return composeRT(pos, rot);
}

void JoltWorld::setBodyTransform(JoltBodyHandle handle, const Matrix4x4& transform) {
    if (!impl_->initialized || handle == kInvalidBody) return;
    JPH::RVec3 pos;
    JPH::Quat rot;
    decomposeToJolt(transform, pos, rot);
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    bi.SetPositionAndRotation(toBodyID(handle), pos, rot, JPH::EActivation::Activate);
}

Vec3 JoltWorld::getBodyPosition(JoltBodyHandle handle) const {
    if (!impl_->initialized || handle == kInvalidBody) return Vec3(0.0f);
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    return toRT(bi.GetPosition(toBodyID(handle)));
}

Vec3 JoltWorld::getBodyLinearVelocity(JoltBodyHandle handle) const {
    if (!impl_->initialized || handle == kInvalidBody) return Vec3(0.0f);
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    return toRT(bi.GetLinearVelocity(toBodyID(handle)));
}

void JoltWorld::setBodyLinearVelocity(JoltBodyHandle handle, const Vec3& v) {
    if (!impl_->initialized || handle == kInvalidBody) return;
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    bi.SetLinearVelocity(toBodyID(handle), toJolt(v));
}

Vec3 JoltWorld::getBodyAngularVelocity(JoltBodyHandle handle) const {
    if (!impl_->initialized || handle == kInvalidBody) return Vec3(0.0f);
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    return toRT(bi.GetAngularVelocity(toBodyID(handle)));
}

void JoltWorld::setBodyAngularVelocity(JoltBodyHandle handle, const Vec3& w) {
    if (!impl_->initialized || handle == kInvalidBody) return;
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    bi.SetAngularVelocity(toBodyID(handle), toJolt(w));
}

Vec3 JoltWorld::getBodyCenterOfMass(JoltBodyHandle handle) const {
    if (!impl_->initialized || handle == kInvalidBody) return Vec3(0.0f);
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    return toRT(bi.GetCenterOfMassPosition(toBodyID(handle)));
}

Vec3 JoltWorld::getBodyPointVelocity(JoltBodyHandle handle, const Vec3& world_point) const {
    if (!impl_->initialized || handle == kInvalidBody) return Vec3(0.0f);
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    return toRT(bi.GetPointVelocity(toBodyID(handle), toJoltR(world_point)));
}

bool JoltWorld::isBodyActive(JoltBodyHandle handle) const {
    if (!impl_->initialized || handle == kInvalidBody) return false;
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    return bi.IsActive(toBodyID(handle));
}

void JoltWorld::activateBody(JoltBodyHandle handle) {
    if (!impl_->initialized || handle == kInvalidBody) return;
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    bi.ActivateBody(toBodyID(handle));
}

void JoltWorld::addForce(JoltBodyHandle handle, const Vec3& force, bool wake) {
    if (!impl_->initialized || handle == kInvalidBody) return;
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    bi.AddForce(toBodyID(handle), toJolt(force),
                wake ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
}

void JoltWorld::addForceAtPoint(JoltBodyHandle handle, const Vec3& force,
                                const Vec3& world_point, bool wake) {
    if (!impl_->initialized || handle == kInvalidBody) return;
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    bi.AddForce(toBodyID(handle), toJolt(force), toJoltR(world_point),
                wake ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
}

void JoltWorld::addTorque(JoltBodyHandle handle, const Vec3& torque, bool wake) {
    if (!impl_->initialized || handle == kInvalidBody) return;
    JPH::BodyInterface& bi = impl_->physics_system->GetBodyInterface();
    bi.AddTorque(toBodyID(handle), toJolt(torque),
                 wake ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
}

void JoltWorld::setContactCaptureEnabled(bool enabled) {
    impl_->contact_listener.setEnabled(enabled);
    if (!enabled) impl_->contact_listener.clear();
}

bool JoltWorld::contactCaptureEnabled() const {
    return impl_->contact_listener.enabled();
}

void JoltWorld::drainContactEvents(std::vector<ContactEvent>& out) {
    impl_->contact_listener.drain(out);
}

} // namespace JoltIntegration
} // namespace RayTrophiSim
