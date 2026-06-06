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

#include "JoltTypes.h"

#include <algorithm>
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

    std::unique_ptr<JPH::PhysicsSystem> physics_system;
    std::vector<JPH::BodyID> bodies;
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
        default:
            return kInvalidBody;
    }

    JPH::RVec3 pos;
    JPH::Quat rot;
    decomposeToJolt(desc.transform, pos, rot);

    const JoltMotionType effective_motion =
        (!desc.dynamic && desc.motion_type == JoltMotionType::Dynamic)
            ? JoltMotionType::Static
            : desc.motion_type;
    const bool is_dynamic = effective_motion == JoltMotionType::Dynamic;
    const bool is_kinematic = effective_motion == JoltMotionType::Kinematic;
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

} // namespace JoltIntegration
} // namespace RayTrophiSim
