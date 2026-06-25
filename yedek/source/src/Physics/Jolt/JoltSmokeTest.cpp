#include "JoltSmokeTest.h"
#include "JoltWorld.h"  // Faz 1 wrapper (pure RT types, no Jolt headers leaked)

// <Jolt/Jolt.h> MUST be the first Jolt include in every translation unit.
#include <Jolt/Jolt.h>

#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <thread>

JPH_SUPPRESS_WARNINGS

namespace {

// ---- Minimal layer setup (throwaway preview of Faz 1 JoltWorld) ----------
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
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer) const override { return "JoltSmokeTest"; }
#endif
};

class ObjectVsBroadPhaseLayerFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer layer1, JPH::BroadPhaseLayer layer2) const override {
        if (layer1 == Layers::NON_MOVING) return layer2 == BroadPhaseLayers::MOVING;
        return true; // MOVING collides with everything
    }
};

class ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        if (a == Layers::NON_MOVING) return b == Layers::MOVING;
        return true;
    }
};

static void TraceImpl(const char* fmt, ...) {
    va_list list;
    va_start(list, fmt);
    char buffer[1024];
    std::vsnprintf(buffer, sizeof(buffer), fmt, list);
    va_end(list);
    std::printf("[Jolt] %s\n", buffer);
}

} // namespace

namespace RayTrophiSim {
namespace JoltIntegration {

SmokeTestResult runSmokeTest() {
    SmokeTestResult result;

    JPH::RegisterDefaultAllocator();
    JPH::Trace = TraceImpl;

    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();
    result.initialized = true;

    {
        JPH::TempAllocatorImpl temp_allocator(16 * 1024 * 1024);
        JPH::JobSystemThreadPool job_system(
            JPH::cMaxPhysicsJobs,
            JPH::cMaxPhysicsBarriers,
            std::max(1, (int)std::thread::hardware_concurrency() - 1));

        const JPH::uint cMaxBodies = 1024;
        const JPH::uint cNumBodyMutexes = 0;
        const JPH::uint cMaxBodyPairs = 1024;
        const JPH::uint cMaxContactConstraints = 1024;

        BPLayerInterfaceImpl broad_phase_layer_interface;
        ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
        ObjectLayerPairFilterImpl object_vs_object_layer_filter;

        JPH::PhysicsSystem physics_system;
        physics_system.Init(cMaxBodies, cNumBodyMutexes, cMaxBodyPairs, cMaxContactConstraints,
                            broad_phase_layer_interface,
                            object_vs_broadphase_layer_filter,
                            object_vs_object_layer_filter);

        JPH::BodyInterface& body_interface = physics_system.GetBodyInterface();

        // Static floor at y = 0 (top surface), 100 x 1 x 100 box centred at y = -0.5.
        JPH::BoxShapeSettings floor_shape_settings(JPH::Vec3(50.0f, 0.5f, 50.0f));
        floor_shape_settings.SetEmbedded();
        JPH::ShapeSettings::ShapeResult floor_shape_result = floor_shape_settings.Create();
        JPH::ShapeRefC floor_shape = floor_shape_result.Get();
        JPH::BodyCreationSettings floor_settings(
            floor_shape, JPH::RVec3(0.0, -0.5, 0.0), JPH::Quat::sIdentity(),
            JPH::EMotionType::Static, Layers::NON_MOVING);
        JPH::Body* floor = body_interface.CreateBody(floor_settings);
        body_interface.AddBody(floor->GetID(), JPH::EActivation::DontActivate);

        // Dynamic sphere dropped from y = 5, radius 0.5 -> should settle at y ~= 0.5.
        const float sphere_radius = 0.5f;
        JPH::BodyCreationSettings sphere_settings(
            new JPH::SphereShape(sphere_radius), JPH::RVec3(0.0, 5.0, 0.0),
            JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);
        JPH::BodyID sphere_id = body_interface.CreateAndAddBody(sphere_settings, JPH::EActivation::Activate);

        result.start_y = (float)body_interface.GetCenterOfMassPosition(sphere_id).GetY();

        physics_system.OptimizeBroadPhase();

        const float fixed_dt = 1.0f / 60.0f;
        const int total_steps = 180; // ~3 seconds, plenty to settle
        for (int i = 0; i < total_steps; ++i) {
            physics_system.Update(fixed_dt, 1, &temp_allocator, &job_system);
        }
        result.stepped = true;
        result.steps = total_steps;
        result.final_y = (float)body_interface.GetCenterOfMassPosition(sphere_id).GetY();

        body_interface.RemoveBody(sphere_id);
        body_interface.DestroyBody(sphere_id);
        body_interface.RemoveBody(floor->GetID());
        body_interface.DestroyBody(floor->GetID());
    }

    JPH::UnregisterTypes();
    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;

    std::printf("[Jolt][SmokeTest] init=%d stepped=%d steps=%d start_y=%.3f final_y=%.3f (expected ~0.5)\n",
                result.initialized ? 1 : 0, result.stepped ? 1 : 0,
                result.steps, result.start_y, result.final_y);

    return result;
}

SmokeTestResult runWorldTest() {
    SmokeTestResult result;

    JoltWorld world;
    result.initialized = world.init();
    if (!result.initialized) {
        std::printf("[Jolt][WorldTest] init FAILED\n");
        return result;
    }

    // Static floor: top surface at y = 0 (box centred at y = -0.5, half 0.5).
    JoltBodyDesc floor;
    floor.shape = JoltShapeType::Box;
    floor.half_extents = Vec3(50.0f, 0.5f, 50.0f);
    floor.dynamic = false;
    floor.transform = Matrix4x4::identity();
    floor.transform.m[1][3] = -0.5f;
    world.createBody(floor);

    // Dynamic box dropped from y = 5 -> should settle so its centre is ~0.5.
    JoltBodyDesc box;
    box.shape = JoltShapeType::Box;
    box.half_extents = Vec3(0.5f, 0.5f, 0.5f);
    box.dynamic = true;
    box.mass = 1.0f;
    box.transform = Matrix4x4::identity();
    box.transform.m[1][3] = 5.0f;
    JoltBodyHandle box_handle = world.createBody(box);

    result.start_y = world.getBodyPosition(box_handle).y;

    world.optimizeBroadPhase();

    const float fixed_dt = 1.0f / 60.0f;
    const int total_steps = 180;
    for (int i = 0; i < total_steps; ++i) {
        world.update(fixed_dt, 1);
    }
    result.stepped = true;
    result.steps = total_steps;

    // Round-trip through the adapter: read the body pose back as a Matrix4x4.
    Matrix4x4 final_xf = world.getBodyTransform(box_handle);
    result.final_y = final_xf.m[1][3];

    world.shutdown();

    std::printf("[Jolt][WorldTest] init=%d stepped=%d steps=%d start_y=%.3f final_y=%.3f (expected ~0.5)\n",
                result.initialized ? 1 : 0, result.stepped ? 1 : 0,
                result.steps, result.start_y, result.final_y);

    return result;
}

} // namespace JoltIntegration
} // namespace RayTrophiSim
